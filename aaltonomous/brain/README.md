# brain: instruction understanding

A local LLM (Gemma 3 4B via Ollama) reads the sorting instruction and turns it into rules (colour/shape -> bin).
Python then checks every rule against the world (3 colours, 3 shapes, 3 bins), so an unknown or made-up word never
reaches the robot. Anything unclear becomes an *issue* with the reason, plus one clarification *question*.

| File | What |
|---|---|
| `llm.py` | Talks to Ollama (standard library only) |
| `instruction.py` | `understand(text)` -> `Plan` (`mapping`, `bin_for(colour, shape)`, `issues`, `question`); `robot_plan()` for the robot |
| `chat.py` | Terminal chat for trying new instructions, including answering the model's questions |
| `test_instruction.py` | Offline checks (fake model replies) + live tests with Gemma (accuracy and time per call) |

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
set BRAIN_N=288 & .venv\Scripts\python -m unittest aaltonomous.brain.test_instruction   # all 288 template sentences
```

Other model or server: set `BRAIN_MODEL` (default `gemma3:4b`) or `OLLAMA_URL` (default `http://localhost:11434`).
