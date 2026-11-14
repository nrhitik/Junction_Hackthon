"""Hand a question from the running robot brain to a person at a terminal on the same machine (or container).

Robot side:   answer = ask("Should I treat orange as red?", wait=15)   # None if nobody answers in time
Person side:  python -m aaltonomous.brain.ask                          # shows each question, sends what you type

Two small files in the temp folder carry the question and the answer.
"""
import os
import tempfile
import time

Q, A = (os.path.join(tempfile.gettempdir(), f"brain_{n}.txt") for n in ("question", "answer"))


def _drop(*paths):
    for p in paths:
        if os.path.exists(p):
            os.remove(p)


def ask(question, wait=15.0):
    _drop(A)
    with open(Q, "w", encoding="utf-8") as f:
        f.write(question)
    end = time.time() + wait
    try:
        while time.time() < end:
            if os.path.exists(A):
                with open(A, encoding="utf-8") as f:
                    return f.read().strip() or None
            time.sleep(0.2)
        return None
    finally:
        _drop(Q, A)


if __name__ == "__main__":
    print("Waiting for questions from the robot (Ctrl+C to quit).")
    shown = None
    while True:
        text = open(Q, encoding="utf-8").read() if os.path.exists(Q) else None
        if text and text != shown:
            shown = text
            answer = input(f"\nrobot asks: {text}\nyour answer> ").strip()
            with open(A + ".tmp", "w", encoding="utf-8") as f:
                f.write(answer)
            os.replace(A + ".tmp", A)   # appears all at once, never half-written
        time.sleep(0.2)
