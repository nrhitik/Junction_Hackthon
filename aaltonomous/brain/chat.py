"""Terminal chat for testing instruction understanding with new, unseen phrasings.

    python -m aaltonomous.brain.chat

Type an instruction -> see the plan. If the model is unsure it explains why and asks; your next line is the answer.
Empty line = start over with a new instruction, q = quit. Testing only: the robot itself never waits for answers.
"""
import time

from aaltonomous.brain.instruction import COLORS, understand


def show(plan, dt):
    print(f"  plan ({dt:.1f} s):")
    for c, s, b in plan.rules:
        print(f"    {c or 'any colour'} {s or 'object'} -> {b} bin")
    print("  colour -> bin: " + ", ".join(f"{c} -> {plan.mapping.get(c, '?')}" for c in COLORS))
    for i in plan.issues:
        print(f"  ! {i}")
    if plan.question:
        print(f"  ? {plan.question}")


def main():
    print(__doc__.split("\n\n", 2)[2])
    messages = None   # conversation so far, kept while the model is waiting for a clarification
    while True:
        try:
            text = input("\nyou> " if messages is None else "your answer> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("q", "quit", "exit"):
            break
        if not text:
            messages = None
            continue
        t = time.time()
        plan = understand(text, messages)
        show(plan, time.time() - t)
        messages = plan.messages if plan.question else None


if __name__ == "__main__":
    main()
