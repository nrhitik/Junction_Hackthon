# brain: instruction understanding, colour and shape detection, 3D positions

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
| `render.py` | Synthetic training images of the three shapes from the organisers' 3D models (any pose, colour, light, camera angle) |
| `shapes.py` | `classify(image, objects)`: cube / cylinder / prism for each detected object; `train` makes `shapes.onnx` |
| `shapes.onnx` | The trained network (1 MB). Runs with OpenCV alone, no PyTorch, ~3 ms for 9 objects on the CPU |
| `test_shapes.py` | 79 real objects labelled by eye (`testdata/shapes.json`), never seen in training |
| `calibrate.py` | `fit_camera(image)`: the head camera's real pose, fitted so the three boxes (known positions) line up |
| `locate.py` | `locate(obj, depth, T)`: an object's 3D position in the robot frame, with depth or from the table plane |
| `test_locate.py` | Both position methods on made-up data with known answers, and the camera fit on real frames |
| `track.py` | `Watcher`: detection + shapes + tracking in a background thread, stable ids, events ("new", "gone") |
| `live.py` | Live view on the laptop webcam or a video: boxes, colours, shapes and ids drawn as it runs |
| `test_track.py` | Tracking on moving made-up objects, flicker filtering, shape voting, the background thread |
| `lid.py` | `find_lid(image, T)`: the lid with the shaped holes, which box it is on, each hole's shape, 3D centre and angle |
| `test_lid.py` | Real lid photos (white, black, three boxes), 41 robot frames with open boxes (no lid may be found), drawn scenes with known answers |
| `drop.py` | `drop_point(box, shape, lid, k)`: where to let go of an object, above its hole or at a free spot in an open box |
| `test_drop.py` | Drop spots stay inside the box and apart; lid drops land on the right hole |

Two modes:
- **Chat (testing):** the model explains what confuses it and asks; your answer refines the plan.
- **Robot (sim / finals):** `robot_plan(instruction, seen=colours)` logs the question and waits 8 s for an answer
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

Objects are named `(colour, shape)` (`Track.name`, e.g. `('red', 'cube')`; two copies share a name, the id tells them
apart) and holes `(box, shape)` (`Lid.pairs`, e.g. `('left', 'cube')`). For the lid task:

```python
from aaltonomous.brain.drop import drop_point
from aaltonomous.brain.lid import find_lid

lid = find_lid(head_bgr, T)                            # once, before the hand is over the boxes; None = no lid seen
d = drop_point(target, shape, lid=lid, k=n_in_box)     # Drop(x, y, z, yaw) in the robot frame; None = hole not in view
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

## Shape detection: what it does and doesn't

- Trained only on rendered images: the organisers' exact 3D models of the three shapes, resting on any face (cylinders
  and prisms upright or lying down, 10 % tumbled as if held), random colour, light and camera angle (from almost
  side-on, like a webcam on the table, to straight down), with a shadow, on
  patches of the real table, sometimes with a neighbouring object in the picture.
- Real photos it never saw: 97.5 % (77 of 79). The two misses are crops with two objects or the robot hand in them.
- Prisms: only 3 real examples so far (one phone photo), so that number rests on very little. More real photos of
  prisms, upright and lying, would tell us how good it really is.
- Retrain (needs PyTorch: `pip install -r requirements-train.txt`): `python -m aaltonomous.brain.shapes train 60000 15`,
  about 3 minutes on an RTX 5070. The accuracy printed per pass jumps around; judge the final one.

## 3D positions and the camera pose

- `locate(obj, depth_mm, T)` gives the top centre of the object in the robot frame (pelvis: x forward, y left, z up).
  With the depth image it is measured; without depth it uses the table height and the objects' 4.5 cm size.
- The organisers' head-camera pose comes from the robot's design file and is not yet checked on the real robot. On
  the dataset videos it does not fit: the boxes should appear lower and smaller than they do, and objects come out
  36 cm ahead instead of the 26 to 33 cm the scene file gives.
- `fit_camera(image)` adjusts the pose (and the boxes' wall height) until the drawn boxes cover the orange boxes in
  the image. On the dataset this puts the objects 27 cm ahead and makes them look 6.4 cm wide, both as expected.
- Not yet reliable: the fitted pose varies between frames (52 to 61 deg looking down), and it needs the boxes to
  stand where the scene file says. Measured box sizes (wall height!) and positions at the venue would fix that.
  The walls look about 10 to 13 cm high on camera; the scene file's 6 cm is an unmeasured default.

## Live detection and tracking

- Laptop webcam: `.venv\Scripts\python -m aaltonomous.brain.live` (q = quit, space = pause). A video file works too:
  `... live data\g1_pick_cube\cam_high.mp4`, and `--save out.mp4` writes the annotated video.
- The window needs OpenCV with window support. The robot uses `opencv-python-headless` (requirements.txt); on the
  laptop swap it once: `pip uninstall opencv-python-headless` then `pip install opencv-python`.
- On the robot: create one `Watcher()`, call `feed(bgr)` from the 30 Hz loop (returns at once), read `objects`.
  Detection runs about 20 times a second on the laptop CPU.
- Measured on whole dataset episodes: episodes 10 and 15 give exactly the six objects with stable ids and correct
  shapes. Objects held in the hand or partly hidden (by the hand, by a box wall) can get a wrong shape; objects cut
  by the image border get no shape at all (None) rather than a guess.
- The size of one object is estimated from each image, so a closer camera (the webcam) doesn't split one object into
  two boxes. Objects cut by the image border can still be split: keep them fully in view. From a low, side-on view
  (a webcam on the table) one object partly hidden behind another can also be split: on the robot's crowded table
  that looks exactly like two touching objects of one colour, and the robot's case wins.
- Dark patches (wood grain, stains, shadows) are not objects: an object needs a lit part (brightness 95+ of 255).
- Shape names: cube, cylinder, prism (the organisers call the prism "triangle"; instructions may use either word).

## Lid, holes and drop positions

- One box has a lid (20 x 20 x 1 cm, white or black) with a square, a round and a triangular hole (about 6 cm). The
  box and the lid can be turned, and the lid can lie anywhere from the box floor to the rim, so nothing is hard-coded:
  each hole is found and named by its shape.
- The lid is found inside a strongly orange box as one solid plate with holes, whatever its colour. On 41 robot
  frames with open boxes (both dataset drops) no lid is found; all three real lid photos give all three holes.
- Height without depth: a whole lid in view is 20 cm wide, which gives its height (positions within about 0.5 cm in
  the drawn tests). If part of it is hidden (box wall, hand, image border) the 6 cm holes give it instead (within about
  1.2 cm). **Please measure one square hole and the circle with a ruler**: the 6 cm is read off a photo.
- A low lid seen at the robot's slant has its front part hidden behind the box's front wall; holes there are not
  reported (`lid.hole(shape)` is None) rather than guessed. Bending the head or torso to look more steeply down helps.
- A hole is `filled` when most of it is a colour other than the box floor (an object in it).
- `drop_point`: lid box = above the matching hole, object bottom 2 cm above the lid, turned to the hole's `yaw`
  (square: any multiple of 90 deg; triangle: of 120 deg). Open box = spots 5 cm apart around the middle
  (k = objects already in that box), clearing the 8 cm walls. Box positions come from the organisers' scene file.
- Not yet tested on the robot's camera with a real lid: the drops have no lid. The lid's real colours under the
  venue lights may need a look once we have a head-camera frame of it.
- Without a camera pose (`T=False`, e.g. phone photos) only shapes and image positions are given; boxes are named by
  their order in the picture, which is the robot's left to right only if the photo looks the same way as the robot.
