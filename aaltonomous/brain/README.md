# brain: instruction understanding and colour detection

A local LLM (Gemma 3 4B via Ollama) reads the sorting instruction and turns it into rules (colour/shape -> bin).
Python then checks every rule against the world (3 colours, 3 shapes, 3 bins), so an unknown or made-up word never
reaches the robot. Anything unclear becomes an *issue* with the reason, plus one clarification *question*.

| File | What |
|---|---|
| `llm.py` | Talks to Ollama (standard library only) |
| `instruction.py` | `understand(text)` -> `Plan` (`mapping`, `bin_for(colour, shape)`, `issues`, `question`); `robot_plan()` for the robot |
| `chat.py` | Terminal chat for trying new instructions, including answering the model's questions |
| `test_instruction.py` | Offline checks (fake model replies) + live tests with Gemma (accuracy and time per call) |
| `colors.py` | `find_objects(image)`: every object on the table with its colour name and pixel position (OpenCV) |
| `test_colors.py` | 35 labelled head-camera frames from all 20 episodes (`testdata/`) + synthetic scenes for every palette colour |

Two modes:
- **Chat (testing):** the model explains what confuses it and asks; your answer refines the plan.
- **Robot (sim / finals):** nobody can answer, so the robot sorts only what is clear and the question goes to the log.

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

- Tuned on the robot's head camera at the venue (640x480). There the "red" objects read as **orange**, the "blue"
  ones as **blue**, and the bins as **red**. Names come from the hue table `HUES` in `colors.py`.
- The robot's silver arm reads as cyan under the venue lights, so cyan is ignored by default (`ignore=()` turns
  that off). Black and white are only looked for with `achromatic=True`; the robot's black hands can look like
  black objects. Depth (available on the robot, not in the dataset) is the proper fix for both.
- Measured on the dataset video (every 30th frame, 391 frames): a wrong extra colour in 8 frames, mostly while an
  arm crosses in front of the bins.
- Dataset (not in git): download https://huggingface.co/datasets/zupermiez/g1_pick_cube into `data/g1_pick_cube/`.
