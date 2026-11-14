"""Instruction -> sorting plan. The local LLM (llm.py) reads the text; Python checks every rule against the world.

    plan = understand("Sort the objects: red into right, blue into left, yellow into center.", seen={"red", ...})
    plan.bin_for("red", "cube")   # most specific rule wins: colour+shape > shape > colour > everything; None = leave it
    plan.mapping                  # {'red': 'right', 'blue': 'left', 'yellow': 'center'}  (the task's three colours)
    plan.issues, plan.question    # what is unclear and why, and the question to ask

Colours are open (match.PALETTE); `seen` = the colours the camera found (colors.py), checked by match.check_scene.
Chat mode (chat.py): show plan.question, then understand(answer, plan.messages) refines the plan with the answer.
Robot mode (robot_plan): ask through ask.py, wait 15 s; no answer -> sort only what is clear.
"""
import json
import re
from dataclasses import dataclass, field

from aaltonomous.brain import llm
from aaltonomous.brain.ask import ask
from aaltonomous.brain.match import PALETTE, check_scene
from aaltonomous.organisers.tasks import BINS, COLORS, SHAPES

SYSTEM = """You turn a sorting instruction for a robot into JSON rules.
The world:
- Objects have a colour (usually red, yellow or blue; possible: red, orange, yellow, green, cyan, blue, purple, pink,
  black, white) and a shape (cube, cylinder, triangle = triangular prism).
- Three identical bins in a row, named only by position from the robot's point of view: left, center, right.
Write one rule per thing the instruction asks: {"color": ..., "shape": ..., "bin": ...}. shape is null unless the
instruction names a shape; color is null unless it names a colour. Use the words above ("middle"/"centre" -> center,
"cubes" -> cube). A shade is the nearest colour above (scarlet -> red, lemon -> yellow, cobalt -> blue, violet ->
purple, lime -> green).
"The rest" / "everything else" / "all objects" is one rule with color null and shape null. Work out "the remaining
bin" yourself.
Never leave anything out: an unknown shape (ball, pyramid, star...) also gets its rule and an issue.
Only judge what the instruction says: a colour it never mentions is not a problem.
When the user answers a question, apply the answer to the whole plan and don't ask about that point again. If the
user agrees to treat one colour as another ("yes, treat orange as red"), write the rules with the colour the camera
really sees (orange).
If the instruction mentions something that does not exist (a colour, shape or bin not listed above) or contradicts
itself: still write the rules you are sure of, write unknown words into the rule exactly as given (never swap an
unknown word for a known one, e.g. "top bin" stays "top"), and add each problem to "issues", saying why you are
confused (e.g. "There is no top bin: the bins are left, center and right."). Then set "question" to one short
question that would resolve them. If nothing is wrong: issues = [] and question = null.
Examples:
"Scarlet ones left, cobalt in the middle, the rest go right." -> {"rules": [{"color": "red", "shape": null, "bin":
"left"}, {"color": "blue", "shape": null, "bin": "center"}, {"color": null, "shape": null, "bin": "right"}],
"issues": [], "question": null}
"All objects go left except yellow, which goes right." -> {"rules": [{"color": null, "shape": null, "bin": "left"},
{"color": "yellow", "shape": null, "bin": "right"}], "issues": [], "question": null}
"Striped ones in the top bin, green on the right." -> {"rules": [{"color": "striped", "shape": null, "bin": "top"},
{"color": "green", "shape": null, "bin": "right"}], "issues": ["I don't know which objects are striped: objects have
one plain colour.", "There is no top bin: the bins are left, center and right."],
"question": "Which objects did you mean by striped, and which bin by top?"}
"Tidy up please." -> {"rules": [], "issues": ["The instruction doesn't say which objects go to which bin."],
"question": "Which bin should each colour go to?"}"""

_FIELD = {"type": ["string", "null"]}
SCHEMA = {"type": "object", "required": ["rules", "issues", "question"], "properties": {
    "rules": {"type": "array", "items": {"type": "object", "required": ["color", "shape", "bin"],
                                         "properties": {"color": _FIELD, "shape": _FIELD, "bin": _FIELD}}},
    "issues": {"type": "array", "items": {"type": "string"}}, "question": _FIELD}}
SHAPE_WORDS = {"cube": ("cube", "cubic", "square"), "cylinder": ("cylind", "round", "circ", "disc", "disk"),
               "triangle": ("triang", "prism", "wedge")}   # a shape counts only if the user said one of these
ALIASES = {"centre": "center", "middle": "center", "central": "center", "prism": "triangle",
           "triangular prism": "triangle", "triangular": "triangle", "square": "cube", "round": "cylinder"}
RETRY = ("You gave no bin for the {} objects. Re-read every sentence of the instruction. If it says where they go, "
         "add those rules and keep the others. If it really doesn't say, answer the same as before.")


@dataclass
class Plan:
    rules: list        # checked rules: (colour or None, shape or None, bin)
    issues: list       # human-readable problems, with the reason
    question: str      # clarification question, or None
    messages: list     # conversation so far (pass back to understand() with the user's answer)
    hold: set = field(default_factory=set)   # colours waiting for a "treat X as Y?" answer: left on the table

    @property
    def mapping(self):
        return {c: b for c in COLORS if (b := self.bin_for(c))}

    def bin_for(self, color, shape=None):
        for key in ((color, shape), (None, shape), (color, None), (None, None)):
            hits = [b for c, s, b in self.rules if (c, s) == key]
            if hits and color not in self.hold:
                return hits[-1]
        return None


def _norm(v, known):
    if v is None or str(v).strip().lower() in ("", "null", "none", "any", "all"):
        return None
    v = ALIASES.get(str(v).strip().lower(), str(v).strip().lower())
    return v[:-1] if v not in known and v[:-1] in known else v   # plural: cubes -> cube


def understand(text, messages=None, timeout=60.0, retry=True, seen=None):
    messages = (messages or [{"role": "system", "content": SYSTEM}]) + [{"role": "user", "content": text}]
    try:
        reply = llm.chat(messages, SCHEMA, timeout)
    except Exception as e:  # LLM down / timeout / bad JSON: nothing is clear
        return Plan([], [f"I could not reach the language model ({e})."], None, messages)
    messages = messages + [{"role": "assistant", "content": json.dumps(reply)}]
    issues, rules = list(reply.get("issues") or []), {}
    said = " ".join(m["content"] for m in messages if m["role"] == "user").lower()
    for r in reply.get("rules") or []:
        c, s, b = _norm(r.get("color"), PALETTE), _norm(r.get("shape"), SHAPES), _norm(r.get("bin"), BINS)
        if s in SHAPES and not any(w in said for w in SHAPE_WORDS[s]):
            s = None   # the model added a shape nobody mentioned
        bad = [f"I don't know the {k} '{v}' (only {', '.join(ok)})." for k, v, ok in
               (("colour", c, PALETTE), ("shape", s, SHAPES), ("bin", b, BINS)) if v is not None and v not in ok]
        name = " ".join(x for x in (c, s or "objects") if x)
        if bad or not b:
            issues += bad or [f"No bin is given for the {name}."]
        elif rules.setdefault((c, s), b) != b:
            issues.append(f"The {name} are sent to two bins ({rules[(c, s)]} and {b}).")
            rules[(c, s)] = None
    issues[:] = dict.fromkeys(issues)   # drop repeated messages, keep order
    plan = Plan([(c, s, b) for (c, s), b in rules.items() if b], issues, reply.get("question"), messages)
    skipped = [c for c in PALETTE if c in re.findall(r"[a-z]+", said) and plan.bin_for(c) is None]
    if skipped and retry:   # a colour the user named got no bin: the model sometimes skips a sentence, ask once more
        again = understand(RETRY.format(" and ".join(skipped)), plan.messages, timeout, False, seen)
        if len(again.rules) > len(plan.rules):
            again.issues[:] = dict.fromkeys([i for i in issues if "don't know" in i] + again.issues)
            again.question = again.question or (again.issues and "Could you tell me what you meant? " +
                                                " ".join(again.issues)) or None
            return again
    check_scene(plan, seen)
    if plan.issues and not plan.question:
        plan.question = "Could you tell me what you meant? " + " ".join(plan.issues)
    # the model sees the final issues and question as its own, so the user's answer has its context
    messages[-1]["content"] = json.dumps({**reply, "issues": plan.issues, "question": plan.question})
    return plan


def robot_plan(instruction, seen=None, log=print, wait=15.0):
    """Robot mode: ask a person through ask.py and wait `wait` seconds. No answer: sort only what is clear."""
    plan = understand(instruction, seen=seen)
    for i in plan.issues:
        log(f"[brain] unclear: {i}")
    if plan.question:
        log(f"[brain] asking: {plan.question}  (waiting {wait:.0f} s for an answer)")
        answer = ask(plan.question, wait)
        log(f"[brain] answer: {answer}" if answer else "[brain] no answer: sorting only the clear part")
        if answer:
            plan = understand(answer, plan.messages, seen=seen)
    return plan
