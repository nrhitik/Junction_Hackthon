"""Task 2, prompt-driven (sim): the robot stays IDLE until it gets a prompt. Each prompt goes through the brain
(local LLM, aaltonomous/brain): the robot sorts only what the prompt asks for, everything else stays on the table,
then it is idle again until the next prompt.

    aalto py /data/team/tools/task2_live.py --seed 5        (in the container; normally started in the background)

Prompts: write a line to /data/task2/prompt.txt (host: /data/work/task2/prompt.txt), e.g. with `robot-prompt "..."`.
Special prompts: "!reset" = new random table, "!status" = where every object is, "!quit" = stop.
If the brain asks a clarification question, the NEXT prompt is taken as the answer.
Log: /data/task2/live.log (host: /data/work/task2/live.log).
"""
import argparse
import json
import os
import threading
import time

import numpy as np

DIR = "/data/task2"
PROMPT, LOG, PLAN = f"{DIR}/prompt.txt", f"{DIR}/live.log", f"{DIR}/plan.json"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--set", default=None)
    a = ap.parse_args()
    os.makedirs(DIR, exist_ok=True)
    import ssl  # noqa: F401
    from aaltonomous.interface.env import AaltoEnv
    from aaltonomous.sim.scene_client import SceneClient
    from aaltonomous.baselines.classical.policy import ClassicalPolicy
    from aaltonomous.brain.instruction import understand

    ov = json.loads(a.set) if a.set else {}
    ov["brain.enabled"] = False  # the plan comes from this runner (prompt by prompt), not from the env's instruction
    env = AaltoEnv(backend="sim", task="sort_language", seed=a.seed, time_limit=24 * 3600, img_server_ip="127.0.0.1",
                   head_rgbd=True, head_rgbd_port=55560)
    sc = SceneClient()
    pol = ClassicalPolicy(overrides=ov, safety=env.filter.cfg, threaded=True, time_limit=24 * 3600,
                          log=lambda m: log("   policy: " + m) if ("GUARD" in m or "error" in m) else None)
    zc = pol.scene.counter_z

    def table():
        out = {}
        for o in sc.state()["objects"]:
            if not o["active"]:
                continue
            pm = pol.scene.pelvis_to_mat(np.asarray(o["pos"], float))
            b = pol.scene.bin_of(pm)
            out[o["name"][4:]] = b or ("counter" if o["pos"][2] - zc > -0.03 else "off")
        return out

    obs = env.reset()
    last = {"a": np.asarray(obs["arm_q"], float).copy(), "h": np.asarray(obs["hand_q"], float).copy()}

    def hold(n=3):
        return {"arm_q_target": np.repeat(last["a"][None], n, 0), "hand_q_target": np.repeat(last["h"][None], n, 0)}

    def step(action):
        nonlocal obs
        action = dict(action)
        action["done"] = False  # the policy's "done" ends a sorting job here, never the session
        obs, info = env.step(action)
        if len(action["arm_q_target"]):
            last["a"] = np.asarray(action["arm_q_target"], float)[-1].copy()
            last["h"] = np.asarray(action["hand_q_target"], float)[-1].copy()
        return info

    queue = []

    def new_prompt():
        if queue:
            return queue.pop(0)
        if os.path.exists(PLAN):  # a plan from the brain chat terminal (aaltonomous.brain.chat robot link)
            with open(PLAN) as f:
                data = json.load(f)
            os.remove(PLAN)
            return data
        if os.path.exists(PROMPT):
            with open(PROMPT) as f:
                text = f.read().strip()
            os.remove(PROMPT)
            return text or None
        return None

    def run_job(text, plan):
        """Sort only what `plan` covers, then return to idle. !stop interrupts; other input is queued."""
        pol.reset({"instruction": text, "task": "sort_language"})
        pol.brain = {"plan": plan, "done": True, "t0": time.monotonic(), "error": None}
        before = table()
        seen = 0
        while True:
            act = pol.act(obs)
            done = bool(act.get("done"))
            step(act)
            for _, m in pol.events[seen:]:
                if m.startswith(("execute", "nothing left", "verify_pick", "not enough time", "GUARD")):
                    log("   " + m[:160])
            seen = len(pol.events)
            if done:
                break
            new = new_prompt()
            if new == "!stop":
                log("stop requested: holding")
                break
            if new:  # anything arriving mid-job waits until the job is done
                queue.append(new)
                log(f"queued (robot busy): {new if isinstance(new, str) else new.get('text')!r}")
        after = table()
        moved = {k: f"{before[k]} -> {after.get(k)}" for k in before if before[k] != after.get(k)}
        log(f"done: moved {moved or 'nothing'}. Robot is IDLE. Table: {after}")

    log(f"READY (seed {a.seed}). Robot is IDLE. Table: {table()}")
    log("Prompt from the brain chat: docker exec -it aalto bash -lc \"aalto py -m aaltonomous.brain.chat\"")
    messages = None  # set when the brain asked a question: the next prompt is the answer
    while True:
        text = new_prompt()
        if text is None:
            step(hold())  # idle: hold the last command (keeps the watchdog happy, no motion)
            time.sleep(0.01)
            continue
        if isinstance(text, dict):  # plan from the chat terminal: already understood (and clarified) there
            from aaltonomous.brain.instruction import Plan
            log(f">>> plan from chat for {text.get('text')!r}: rules {text['rules']}")
            plan = Plan([tuple(r) for r in text["rules"]], [], None, [])
            run_job(text.get("text", ""), plan)
            continue
        log(f">>> prompt: {text!r}")
        if text == "!quit":
            break
        if text == "!status":
            log(f"table: {table()}")
            continue
        if text == "!reset":
            obs = env.reset()
            last["a"], last["h"] = np.asarray(obs["arm_q"], float).copy(), np.asarray(obs["hand_q"], float).copy()
            messages = None
            log(f"new table: {table()}")
            continue
        box = {}
        th = threading.Thread(target=lambda: box.update(plan=understand(text, messages)), daemon=True)
        t0 = time.time()
        th.start()
        while th.is_alive():  # keep holding while the LLM thinks
            step(hold())
            th.join(0.01)
        plan = box["plan"]
        log(f"brain ({time.time() - t0:.1f} s): rules {plan.rules}")
        for i in plan.issues:
            log(f"brain issue: {i}")
        messages = plan.messages if plan.question else None
        if plan.question:
            log(f"brain asks: {plan.question}   (answer with your next prompt)")
        if not plan.rules:
            log("nothing clear to do: staying idle")
            continue
        run_job(text, plan)
    env.close()


if __name__ == "__main__":
    main()
