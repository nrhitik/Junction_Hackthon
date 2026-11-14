"""Terminal chat for testing instruction understanding with new, unseen phrasings.

    python -m aaltonomous.brain.chat

Type an instruction -> see the plan. If the model is unsure it explains why and asks; your next line is the answer.
image <file> = look at a camera image first: the plan is then checked against the colours on that table.
Empty line = start over with a new instruction, q = quit. Testing only: it does not control the robot.
"""
import collections
import time

from aaltonomous.brain.instruction import understand
from aaltonomous.organisers.tasks import COLORS


def look(path):
    import cv2   # only needed for the image command
    import numpy as np
    from aaltonomous.brain.colors import find_objects
    img = cv2.imdecode(np.fromfile(path, np.uint8), 1)
    return collections.Counter(o.color for o in find_objects(img))


def show(plan, dt, seen):
    print(f"  plan ({dt:.1f} s):")
    for c, s, b in plan.rules:
        print(f"    {c or 'any colour'} {s or 'object'} -> {b} bin")
    print("  " + ", ".join(f"{c} -> {plan.bin_for(c) or 'stays on the table'}" for c in (seen or COLORS)))
    for i in plan.issues:
        if i not in (plan.question or ""):
            print(f"  ! {i}")
    if plan.question:
        print(f"  ? {plan.question}")


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
        plan = understand(text, messages, seen=seen)
        show(plan, time.time() - t, seen)
        messages = plan.messages if plan.question else None


if __name__ == "__main__":
    main()
