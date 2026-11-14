"""Compare the sorting plan with the colours the camera sees. Adds issues and a question to the plan.

    check_scene(plan, seen={"orange", "yellow", "blue"})

- Instruction says red, no red on the table, but orange (a neighbour colour) is there without a rule of its own:
  ask "should I treat orange as red?". Until someone says yes, the orange objects stay on the table (plan.hold).
- Instruction names a colour that is not on the table and has no neighbour there: "I can't see any ... object".
- A colour on the table that no rule covers: reported, left on the table.
- seen=None (no camera): only check that the task's usual colours (red, yellow, blue) all have a bin.
"""
from aaltonomous.organisers.tasks import COLORS

# colour -> its neighbours on the colour wheel: what it can be mistaken for under odd lighting
NEAR = {"red": ("orange", "pink"), "orange": ("red", "yellow"), "yellow": ("orange", "green"),
        "green": ("yellow", "cyan"), "cyan": ("green", "blue"), "blue": ("cyan", "purple"),
        "purple": ("blue", "pink"), "pink": ("purple", "red")}
PALETTE = (*NEAR, "black", "white")


def check_scene(plan, seen=None):
    ruled = {c for c, _, _ in plan.rules if c}
    covers_all = any(c is None for c, _, _ in plan.rules)   # a rule for "everything", or rules by shape
    if seen is None:
        miss = [c for c in COLORS if c not in ruled and not covers_all]
        if miss and not any("where" in i.lower() for i in plan.issues):
            plan.issues.append(f"The instruction doesn't say where the {' and '.join(miss)} objects go.")
        return plan
    seen, asks = set(seen), []
    earlier = " ".join(m["content"] for m in plan.messages[:-1])
    for c in sorted(ruled - seen):
        near = [n for n in NEAR.get(c, ()) if n in seen and n not in ruled]
        ask = (f"The instruction says {c}, but I can't see a {c} object. I can see {' and '.join(near)} ones, "
               f"which look close. Should I treat {' or '.join(near)} as {c}?")
        if near:
            plan.hold |= set(near)
            if ask in earlier:   # already asked and not confirmed: don't ask again
                plan.issues.append(f"I'll leave the {' and '.join(near)} objects on the table.")
            else:
                asks.append(ask)
        elif not any(n in seen and plan.bin_for(n) == plan.bin_for(c) for n in NEAR.get(c, ())):   # no stand-in
            plan.issues.append(f"I can't see any {c} object on the table.")
    if not covers_all:
        plan.issues += [f"I can see {c} objects, but the instruction doesn't say where they go, so I'll leave them "
                        "on the table." for c in sorted(seen - ruled - plan.hold)]
    plan.issues += asks
    if asks:
        plan.question = " ".join(asks)
    return plan
