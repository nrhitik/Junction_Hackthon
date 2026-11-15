p = "/data/work/team/il_insert/run.py"
s = open(p).read()
old = '''            while True:
                obs, info = env.step(pol.act(obs))
                if info["episode_over"]:
                    break'''
new = '''            ne = len(pol.events)
            while True:
                obs, info = env.step(pol.act(obs))
                if os.environ.get("IL_EVENTS"):  # team: live event timeline
                    for tt, m in pol.events[ne:]:
                        print(f"   EV [t={tt if tt is None else round(tt, 1)} wall={time.time() - t0:.0f}s] {m[:220]}", flush=True)
                    ne = len(pol.events)
                if info["episode_over"]:
                    break'''
if "IL_EVENTS" not in s:
    assert old in s
    s = s.replace(old, new)
    open(p, "w").write(s)
print("IL_EVENTS" in s)
