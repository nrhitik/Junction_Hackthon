"""Terminal chat for testing instruction understanding with new, unseen phrasings.

    python -m aaltonomous.brain.chat

Type an instruction -> see the plan. If the model is unsure it explains why and asks; your next line is the answer.
image <file> = look at a camera image first: the plan is then checked against the colours on that table.
Empty line = start over with a new instruction, q = quit.
Team robot link (sim task 2, when task2_live.py runs): a clear plan is sent to the waiting robot and its progress is
shown here; status / stop / reset control the robot. Otherwise testing only.
"""
import collections
import json
import os
import time

from aaltonomous.brain.instruction import understand
from aaltonomous.organisers.tasks import COLORS


def look(path):
    from aaltonomous.brain.colors import find_objects, load   # needs OpenCV: only for the image command
    return collections.Counter(o.color for o in find_objects(load(path)))


def show(plan, dt, seen):
    print(f"  plan ({dt:.1f} s):")
    for c, s, b in plan.rules:
        print(f"    {c or 'any colour'} {s or 'object'} -> {b} bin")
    if not any(s for _, s, _ in plan.rules):  # colour-only rules: the per-colour summary is meaningful
        print("  " + ", ".join(f"{c} -> {plan.bin_for(c) or 'stays on the table'}" for c in (seen or COLORS)))
    else:
        print("  everything not matched by a rule stays on the table")
    for i in plan.issues:
        if i not in (plan.question or ""):
            print(f"  ! {i}")
    if plan.question:
        print(f"  ? {plan.question}")


# --- team: exact parser for SHORT commands ("cube left", "yellow left", "cubes left, cylinders right, prisms center",
# "purple cylinder right and orange cube center"). Each part = colour and/or shape + exactly one bin word. Read word by
# word, so bins cannot be swapped (the LLM swapped two in "cubes left, cylinders right, prisms center"). Anything
# else (except / the rest / respectively / unknown words) -> None -> the LLM.
import re as _re
_BINW = {"left": "left", "right": "right", "center": "center", "centre": "center", "middle": "center"}
_SHAPEW = {"cube": "cube", "cubes": "cube", "square": "cube", "squares": "cube", "block": "cube", "blocks": "cube",
           "cylinder": "cylinder", "cylinders": "cylinder", "round": "cylinder",
           "prism": "prism", "prisms": "prism", "triangle": "prism", "triangles": "prism", "triangular": "prism"}
_COLW = ("red", "orange", "yellow", "green", "cyan", "blue", "purple", "violet", "pink", "black", "white")
_FILLER = {"the", "a", "an", "all", "every", "each", "put", "place", "move", "drop", "sort", "go", "goes", "to", "in",
           "into", "on", "at", "bin", "bins", "box", "boxes", "one", "ones", "object", "objects", "piece", "pieces",
           "please", "side", "should", "be", "is", "are", "then"}


def simple_rules(text):
    t = text.lower().strip().rstrip(".!")
    if _re.search(r"\b(except|but|not|rest|other|others|everything|respectively|unless|instead|if)\b", t):
        return None
    rules = []
    for part in _re.split(r",|;|\band\b|\bthen\b", t):
        words = _re.findall(r"[a-z]+", part)
        if not words:
            continue
        bins = [_BINW[w] for w in words if w in _BINW]
        cols = [("purple" if w == "violet" else w) for w in words if w in _COLW]
        shps = [_SHAPEW[w] for w in words if w in _SHAPEW]
        rest = [w for w in words if w not in _BINW and w not in _COLW and w not in _SHAPEW and w not in _FILLER]
        if len(bins) != 1 or len(cols) > 1 or len(set(shps)) > 1 or not (cols or shps) or rest:
            return None
        rules.append((cols[0] if cols else None, shps[0] if shps else None, bins[0]))
    keys = [(c, s) for c, s, _ in rules]
    if not rules or len(keys) != len(set(keys)):
        return None
    return rules


# --- team: robot link (sim, task 2). The task-2 runner (task2_live.py) waits IDLE for plans written here.
ROBOT_DIR = "/data/task2"
ROBOT = os.path.exists(f"{ROBOT_DIR}/live.log")


def robot_send(kind, payload):
    """kind: plan (rules) or command (!status / !stop / !reset); then stream the robot log until it is idle."""
    log = f"{ROBOT_DIR}/live.log"
    start = os.path.getsize(log)
    tmp = f"{ROBOT_DIR}/{kind}.tmp"
    with open(tmp, "w") as f:
        f.write(json.dumps(payload) if kind == "plan" else payload)
    os.replace(tmp, f"{ROBOT_DIR}/{'plan.json' if kind == 'plan' else 'prompt.txt'}")
    print("  [robot] sent, following the robot (Ctrl+C = stop following; the robot keeps going)")
    t0 = time.time()
    try:
        while time.time() - t0 < 600:
            with open(log) as f:
                f.seek(start)
                new = f.read()
            start += len(new.encode())
            for line in new.splitlines():
                print("  [robot] " + line)
                msg = line.split("] ", 1)[-1]
                if "Robot is IDLE" in line or msg.startswith(("table:", "new table:", "nothing clear")):
                    return
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("  [robot] (stopped following)")


def main():
    print(__doc__.split("\n\n", 2)[2])
    messages = seen = None   # conversation while the model waits for a clarification; colours on the table
    while True:
        try:
            text = input("\nyou> " if messages is None else "your answer> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("q", "quit", "exit"):
            break
        if ROBOT and text.lower() in ("status", "stop", "reset"):
            robot_send("command", "!" + text.lower())
            continue
        if text.lower().startswith("image"):
            try:
                seen = look(text[5:].strip().strip('"'))
                print("  I see: " + (", ".join(f"{n} {c}" for c, n in seen.items()) or "nothing"))
            except Exception as e:
                print(f"  could not read that image ({e})")
            continue
        if not text:
            messages = None
            continue
        t = time.time()
        direct = simple_rules(text) if messages is None else None
        if direct:
            print("  plan (read directly, no LLM):")
            for c, s, b in direct:
                print(f"    {c or 'any colour'} {s or 'object'} -> {b} bin")
            print("  everything else stays on the table")
            if ROBOT:
                robot_send("plan", {"text": text, "rules": [list(r) for r in direct]})
            continue
        plan = understand(text, messages, seen=seen)
        if ROBOT and plan.rules:
            # prompt mode: objects the user did not mention simply stay on the table -> no question about them
            plan.issues = [i for i in plan.issues if "doesn't say where" not in i and "leave them on the table" not in i]
            if plan.question and not plan.issues:
                plan.question = None
        show(plan, time.time() - t, seen)
        messages = plan.messages if plan.question else None
        if ROBOT and plan.rules and not plan.question:
            robot_send("plan", {"text": text, "rules": [list(r) for r in plan.rules]})
        elif ROBOT and plan.question:
            print("  [robot] waiting for your answer before doing anything")
        elif ROBOT:
            print("  [robot] nothing clear to do: the robot stays idle")


if __name__ == "__main__":
    main()
