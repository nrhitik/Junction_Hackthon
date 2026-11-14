"""Instruction -> sorting plan. The local LLM (llm.py) reads the text; Python checks every rule against the world.

    plan = understand("Sort the objects: red into right, blue into left, yellow into center.")
    plan.mapping                  # {'red': 'right', 'blue': 'left', 'yellow': 'center'}  (colour -> bin)
    plan.bin_for("red", "cube")   # most specific rule wins: colour+shape > shape > colour > everything; None = leave it
    plan.issues, plan.question    # what is unclear and why, and the question to ask

Chat mode (chat.py): show plan.question, then understand(answer, plan.messages) refines the plan with the answer.
Robot mode (robot_plan): nobody can answer, so the robot sorts only what is clear and the question goes to the log.
"""
import json
from dataclasses import dataclass

from aaltonomous.brain import llm
from aaltonomous.organisers.tasks import BINS, COLORS, SHAPES

SYSTEM = """You turn a sorting instruction for a robot into JSON rules.
The world (nothing else exists):
- Objects have a colour (red, yellow, blue) and a shape (cube, cylinder, triangle = triangular prism).
- Three identical bins in a row, named only by position from the robot's point of view: left, center, right.
Write one rule per thing the instruction asks: {"color": ..., "shape": ..., "bin": ...}. shape is null unless the
instruction names a shape; color is null unless it names a colour. Use the words above ("middle"/"centre" -> center,
"cubes" -> cube). Shades of red, yellow and blue (scarlet, lemon, cobalt...) are those colours; other colours are not.
Never leave anything out: an unknown shape (ball, pyramid, star...) also gets its rule and an issue.
When the user answers a question, apply the answer to the whole plan and don't ask about that point again.
Expand "the rest" / "everything else" / "the remaining bin" into explicit rules.
If the instruction mentions something that does not exist (another colour, shape or bin), contradicts itself, or does
not say where some colour goes: still write the rules you are sure of, write unknown words into the rule exactly as
given (never swap an unknown word for a known one, e.g. "top bin" stays "top"), and add each problem to "issues",
saying why you are confused (e.g. "There is no top bin: the bins are left, center and right."). Then set "question" to one short question that
would resolve them. If nothing is wrong: issues = [] and question = null.
Examples:
"Scarlet ones left, cobalt in the middle, the rest go right." -> {"rules": [{"color": "red", "shape": null, "bin":
"left"}, {"color": "blue", "shape": null, "bin": "center"}, {"color": "yellow", "shape": null, "bin": "right"}],
"issues": [], "question": null}
"All objects go left except yellow, which goes right." -> {"rules": [{"color": "red", "shape": null, "bin": "left"},
{"color": "blue", "shape": null, "bin": "left"}, {"color": "yellow", "shape": null, "bin": "right"}], "issues": [],
"question": null}
"Green ones in the top bin, red on the right." -> {"rules": [{"color": "green", "shape": null, "bin": "top"},
{"color": "red", "shape": null, "bin": "right"}], "issues": ["There is no green object: the colours are red, yellow
and blue.", "There is no top bin: the bins are left, center and right.", "Yellow and blue have no bin."],
"question": "Which colour did you mean by green, which bin by top, and where do yellow and blue go?"}
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


@dataclass
class Plan:
    rules: list        # checked rules: (colour or None, shape or None, bin)
    issues: list       # human-readable problems, with the reason
    question: str      # clarification question, or None
    messages: list     # conversation so far (pass back to understand() with the user's answer)

    @property
    def mapping(self):
        return {c: b for c in COLORS if (b := self.bin_for(c))}

    def bin_for(self, color, shape=None):
        for key in ((color, shape), (None, shape), (color, None), (None, None)):
            hits = [b for c, s, b in self.rules if (c, s) == key]
            if hits:
                return hits[-1]
        return None


def _norm(v, known):
    if v is None or str(v).strip().lower() in ("", "null", "none", "any", "all"):
        return None
    v = ALIASES.get(str(v).strip().lower(), str(v).strip().lower())
    return v[:-1] if v not in known and v[:-1] in known else v   # plural: cubes -> cube


def understand(text, messages=None, timeout=60.0):
    messages = (messages or [{"role": "system", "content": SYSTEM}]) + [{"role": "user", "content": text}]
    try:
        reply = llm.chat(messages, SCHEMA, timeout)
    except Exception as e:  # LLM down / timeout / bad JSON: nothing is clear
        return Plan([], [f"I could not reach the language model ({e})."], None, messages)
    messages = messages + [{"role": "assistant", "content": json.dumps(reply)}]
    issues, rules = list(reply.get("issues") or []), {}
    said = " ".join(m["content"] for m in messages if m["role"] == "user").lower()
    for r in reply.get("rules") or []:
        c, s, b = _norm(r.get("color"), COLORS), _norm(r.get("shape"), SHAPES), _norm(r.get("bin"), BINS)
        if s in SHAPES and not any(w in said for w in SHAPE_WORDS[s]):
            s = None   # the model added a shape nobody mentioned
        bad = [f"I don't know the {k} '{v}' (only {', '.join(ok)})." for k, v, ok in
               (("colour", c, COLORS), ("shape", s, SHAPES), ("bin", b, BINS)) if v is not None and v not in ok]
        name = " ".join(x for x in (c, s or "objects") if x)
        if bad or not b:
            issues += bad or [f"No bin is given for the {name}."]
        elif rules.setdefault((c, s), b) != b:
            issues.append(f"The {name} are sent to two bins ({rules[(c, s)]} and {b}).")
            rules[(c, s)] = None
    issues[:] = dict.fromkeys(issues)   # drop repeated messages, keep order
    plan = Plan([(c, s, b) for (c, s), b in rules.items() if b], issues, reply.get("question"), messages)
    if not any(s for _, s, _ in plan.rules):   # colour-based plan: every colour needs a bin
        miss = [c for c in COLORS if plan.bin_for(c) is None]
        if miss and not any("where" in i.lower() for i in issues):
            issues.append(f"The instruction doesn't say where the {' and '.join(miss)} objects go.")
    if plan.issues and not plan.question:
        plan.question = "Could you tell me what you meant? " + " ".join(plan.issues)
    return plan


def robot_plan(instruction, log=print):
    """Robot mode: no one can answer a question, so sort only what is clear and log the rest."""
    plan = understand(instruction)
    for i in plan.issues:
        log(f"[brain] unclear: {i}")
    if plan.question:
        log(f"[brain] would ask: {plan.question}  -> no one to ask, sorting only the clear part")
    return plan
