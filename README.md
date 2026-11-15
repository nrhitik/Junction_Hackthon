# Aaltonomous Humanoid — team whynot: main source code

Source of our Junction 2026 submission for the Unitree G1 + Dex3 sorting challenge.

**Docker image (what runs):** `docker.io/priyamb/aaltonomous-policy:v3`
digest `sha256:f4e9f112325342e5a865a28194d0a02290bb0a18de7a8c0406591dc80c4f60bb` (CPU only, ~1.5 GB).

One image serves all tasks: the runner's `reset` message carries `task` and `instruction`, and the policy
switches on `task` (`sort_fixed`, `sort_language`, `insert_shapes`).

## How it works

- **Perception** (`aaltonomous/baselines/classical/perception.py`): head RGB-D camera, colour (HSV) + depth;
  shape from the top-face outline, cross-checked with a small ONNX classifier (`aaltonomous/brain/shapes.onnx`).
  One physical object = one detection (shaded faces no longer become a second object). The bin row's front edge
  is measured from depth at the start of every episode, so the policy uses the real box positions.
- **Instructions** (`baselines/classical/rules_parser.py`): rule-based reader, no LLM — colour and/or shape → bin,
  "respectively", "the rest / everything else", "except", "leave / don't move". Anything unclear: nothing is
  sorted (0 points, never a wrong bin).
- **Planning** (`planner.py`, `kinematics.py`): Pinocchio IK, shape-aligned pinch grips (wedge grip for prisms),
  hand-over between arms for objects out of one arm's reach, deepest reachable release point for far boxes.
- **Safety** (`team_safety.py`): URDF convex-hull self-collision check and a no-cross-body rule on every plan and
  every command chunk, before the organisers' safety filter.
- **Feedback** (`policy.py`): head-camera pick/place checks (plus finger contact), a final look at the table and
  one extra attempt for what is left.
- **Real robot** (`config.yaml`, profile `real`): colour thresholds measured on the real head camera, arrival holds
  before closing/opening and aim compensation for the measured arm sag.

## Layout

| Path | What |
|---|---|
| `aaltonomous/baselines/classical/` | our policy: perception, planner, kinematics, team safety, parser, config, policy server |
| `aaltonomous/brain/` | teammate's perception helpers + ONNX shape model |
| `aaltonomous/{deploy,interface,scene,organisers}/` | organisers' interface, scene and protocol code (as used by the image) |
| `docker/team.Dockerfile` | the submission image |
| `tools/` | team test / diagnosis tools (sim episodes, plan checks, real-run analysis) and the IL experiments |

Not included: `third_party/xr_teleoperate/assets/g1/` (G1 URDF + meshes, from unitreerobotics/xr_teleoperate),
which the Dockerfile expects at `third_party/xr_teleoperate/assets/g1/` in the build context.

## Build and test

```bash
docker build -f docker/team.Dockerfile -t aaltonomous-policy:v3 .     # build context = repo root with app/ layout
docker run --rm -p 8000:8000 aaltonomous-policy:v3
python -m aaltonomous.deploy.replay --synthetic 100 --policy ws://localhost:8000
python -m aaltonomous.sim.evaluate --policy ws://localhost:8000 --task sort_fixed --trials 2
```
