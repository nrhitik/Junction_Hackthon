# brain: instruction understanding and colour detection

A local LLM (Gemma 3 4B via Ollama) reads the sorting instruction and turns it into rules (colour/shape -> bin).
Python then checks every rule against the world (10 colour names, 3 shapes, 3 bins) and against the colours the
camera sees, so an unknown or made-up word never reaches the robot. Anything unclear becomes an *issue* with the
reason, plus one clarification *question*.

| File | What |
|---|---|
| `llm.py` | Talks to Ollama (standard library only) |
| `instruction.py` | `understand(text)` -> `Plan` (`mapping`, `bin_for(colour, shape)`, `issues`, `question`); `robot_plan()` for the robot |
| `match.py` | Plan vs. the colours on the table: "treat orange as red?", colours that are missing, objects with no instruction |
| `ask.py` | Carries a question from the running robot code to a person at a terminal, and the answer back |
| `chat.py` | Terminal chat for trying new instructions, including answering the model's questions; `image <file>` looks at a camera image first |
| `test_instruction.py` | Offline checks (fake model replies) + live tests with Gemma (accuracy and time per call) |
| `colors.py` | `find_objects(image)`: every object on the table with its colour name and pixel position (OpenCV) |
| `test_colors.py` | 35 labelled head-camera frames from all 20 episodes and a phone photo (`testdata/`) + synthetic scenes for every palette colour |

Two modes:
- **Chat (testing):** the model explains what confuses it and asks; your answer refines the plan.
- **Robot (sim / finals):** `robot_plan(instruction, seen=colours)` logs the question and waits 15 s for an answer
  typed in a terminal on the same machine (`python -m aaltonomous.brain.ask`). No answer: it sorts only what is clear.
  Objects waiting for a "treat orange as red?" answer stay on the table.

Wiring it into the robot code (call once per episode, in a background thread: it can take 3 to 25 s):

```python
from aaltonomous.brain.colors import find_objects
from aaltonomous.brain.instruction import robot_plan

objs = find_objects(head_bgr)                          # head camera image, BGR (cv2.cvtColor from the RGB in obs)
plan = robot_plan(instruction, seen={o.color for o in objs})
for o in objs:
    target = plan.bin_for(o.color)                     # "left" / "center" / "right", or None = leave it on the table
```

## Setup (Windows)

1. Install Ollama (`winget install Ollama.Ollama`), then `ollama pull gemma3:4b` (~3.3 GB).
2. From the repo root: `python -m venv .venv` and `.venv\Scripts\python -m pip install -r requirements.txt`.

## Run

```
.venv\Scripts\python -m aaltonomous.brain.chat                       # chat
.venv\Scripts\python -m unittest aaltonomous.brain.test_instruction   # tests (live part needs Ollama running)
.venv\Scripts\python -m unittest aaltonomous.brain.test_colors        # colour detection tests
.venv\Scripts\python -m aaltonomous.brain.colors frame.jpg out.jpg     # detect on one image, save it annotated
set BRAIN_N=288 & .venv\Scripts\python -m unittest aaltonomous.brain.test_instruction   # all 288 template sentences
```

Other model or server: set `BRAIN_MODEL` (default `gemma3:4b`) or `OLLAMA_URL` (default `http://localhost:11434`).

## Colour detection: what it does and doesn't

- An object is whatever differs clearly from the table's own colour in that image, so it adapts to the camera and
  the lighting. Names come from the hue table `HUES` in `colors.py`. On the robot's head camera at the venue the
  "red" objects read as **orange**, the "blue" ones as **blue** and the bins as **red**.
- Object size is judged as a fraction of the image and assumes the head camera's view of the whole table. In a
  close-up, or from far away, a bin can look object-sized and be reported as an object. On the robot the bins are
  at known positions, so the code that has depth can drop anything inside them.
- The robot's silver arm reads as cyan under the venue lights, so cyan is ignored by default (`ignore=()` turns
  that off). Black and white are only looked for with `achromatic=True`; the robot's black hands can look like
  black objects. Depth (available on the robot, not in the dataset) is the proper fix for both.
- Tests: 35 labelled head-camera frames from all 20 episodes and one phone photo with 9 objects (incl. prisms).
  On the dataset video (every 30th frame, 391 frames) a wrong extra colour appears in 1 frame.
- Dataset (not in git): download https://huggingface.co/datasets/zupermiez/g1_pick_cube into `data/g1_pick_cube/`.
