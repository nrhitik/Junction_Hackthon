"""Tests for instruction understanding.

    python -m unittest aaltonomous.brain.test_instruction -v          # offline checks + live Gemma tests
    set BRAIN_N=288 & python -m unittest aaltonomous.brain.test_instruction   # all 288 template sentences (~15 min)

Offline tests feed fake model replies (instant, no Ollama). Live tests need Ollama running with gemma3:4b; they are
skipped otherwise. Each live test prints its accuracy, average time per call, and every failure.
"""
import itertools
import os
import random
import time
import unittest
import urllib.request
from unittest import mock

from aaltonomous.brain import instruction as ins
from aaltonomous.brain.instruction import understand
from aaltonomous.organisers.tasks import BINS, COLORS, FIXED_INSTRUCTION, FIXED_MAPPING, TEMPLATES

R, Y, B = COLORS


def fake(*rules, issues=(), question=None):
    """Patch the LLM to return these rules (colour, shape, bin)."""
    reply = {"rules": [dict(zip(("color", "shape", "bin"), r)) for r in rules], "issues": list(issues),
             "question": question}
    return mock.patch.object(ins.llm, "chat", return_value=reply)


class Checks(unittest.TestCase):   # offline
    def test_valid(self):
        with fake((R, None, "right"), ("Blue", None, "Left"), ("yellow", None, "middle")):
            p = understand("x")
        self.assertEqual(p.mapping, {R: "right", Y: "center", B: "left"})
        self.assertEqual((p.issues, p.question), ([], None))

    def test_unknown_words_are_flagged_and_dropped(self):
        with fake(("green", None, "left"), (R, "sphere", "center"), (B, None, "top"), (Y, None, "right")):
            p = understand("x")
        self.assertEqual(p.rules, [(Y, None, "right")])
        self.assertEqual(len([i for i in p.issues if "don't know" in i]), 3)
        self.assertTrue(p.question)

    def test_shape_nobody_mentioned_is_ignored(self):
        with fake((R, "cube", "left"), (Y, "cube", "center"), (B, "cube", "right")):
            p = understand("Red left, yellow center, blue right.")
        self.assertEqual(p.mapping, {R: "left", Y: "center", B: "right"})

    def test_contradiction(self):
        with fake((R, None, "left"), (R, None, "right"), (Y, None, "center"), (B, None, "center")):
            p = understand("x")
        self.assertNotIn(R, p.mapping)
        self.assertTrue(any("two bins" in i for i in p.issues))

    def test_missing_colour(self):
        with fake((R, None, "left"), (B, None, "right")):
            p = understand("x")
        self.assertTrue(any("yellow" in i for i in p.issues))

    def test_specific_rule_wins(self):
        with fake((None, None, "left"), (R, None, "center"), (R, "cube", "right"), (None, "triangle", "right")):
            p = understand("Everything left, red in the center, but red cubes and all triangles right.")
        self.assertEqual([p.bin_for(R, "cube"), p.bin_for(R, "cylinder"), p.bin_for(B, "triangle"),
                          p.bin_for(Y, "cube")], ["right", "center", "right", "left"])

    def test_llm_down(self):
        with mock.patch.object(ins.llm, "chat", side_effect=OSError("connection refused")):
            p = understand("x")
        self.assertEqual(p.rules, [])
        self.assertTrue(p.issues)


def _ollama_up():
    try:
        urllib.request.urlopen(ins.llm.URL, timeout=2)
        return True
    except OSError:
        return False


def fill(template, cols, bins):
    w = {f"c{i}": c for i, c in enumerate(cols)} | {f"C{i}": c.capitalize() for i, c in enumerate(cols)}
    return template.format(**w, **{f"b{i}": b for i, b in enumerate(bins)})


# Unseen phrasings with one clear answer: (instruction, expected colour -> bin)
PARAPHRASES = [
    ("Red goes in the middle, blue on the left, yellow on the right.", {R: "center", B: "left", Y: "right"}),
    ("left: yellow; middle: red; right: blue", {Y: "left", R: "center", B: "right"}),
    ("Yellow -> right, red -> center, blue -> left", {Y: "right", R: "center", B: "left"}),
    ("The bin on the far left is for blue things, the one on the far right for red, yellow in between.",
     {B: "left", R: "right", Y: "center"}),
    ("Red and blue go to the left and right bins respectively, yellow in the middle.",
     {R: "left", B: "right", Y: "center"}),
    ("Put red in the right bin, yellow in the left bin and the rest in the middle.",
     {R: "right", Y: "left", B: "center"}),
    ("Everything except the blue objects goes to the left bin; blue goes right.", {R: "left", Y: "left", B: "right"}),
    ("Could you sort these? Crimson pieces to the right, golden ones to the left, navy ones in the centre.",
     {R: "right", Y: "left", B: "center"}),
    ("Into the left bin go the yellow objects; into the right bin the red ones; blue into the remaining bin.",
     {Y: "left", R: "right", B: "center"}),
    ("I want the yellow items in the bin on the right, while the red ones should be placed in the bin on the left, "
     "whereas blue belongs in the middle bin.", {Y: "right", R: "left", B: "center"}),
    ("Place reds centrally, blues leftward and yellows rightward.", {R: "center", B: "left", Y: "right"}),
    ("PUT THE RED OBJECTS IN THE LEFT BIN, THE YELLOW OBJECTS IN THE CENTER BIN AND THE BLUE OBJECTS IN THE RIGHT "
     "BIN.", FIXED_MAPPING),
]
# Anomalies: (instruction, colour -> bin that must still be understood). Every one must report an issue.
ANOMALIES = [
    ("Put the green objects in the left bin, red in the center and blue on the right.", {R: "center", B: "right"}),
    ("Red goes in the top bin, yellow left, blue right.", {Y: "left", B: "right"}),
    ("Put the spheres in the left bin, red in the center, yellow on the right and blue on the left.",
     {R: "center", Y: "right", B: "left"}),
    ("Red goes left. Red goes right. Yellow in the center, blue on the right.", {Y: "center", B: "right"}),
    ("Only red into the left bin.", {R: "left"}),
    ("Do something nice with the objects.", {}),
]


@unittest.skipUnless(_ollama_up(), "Ollama is not running")
class Live(unittest.TestCase):
    def run_cases(self, name, cases, expect_issues):
        fails, times = [], []
        for text, want in cases:
            t = time.time()
            p = understand(text)
            times.append(time.time() - t)
            ok = (bool(p.issues) == expect_issues and
                  (p.mapping == want if not expect_issues else all(p.mapping.get(c) == b for c, b in want.items())))
            if not ok:
                fails.append(f"  {text}\n    got {p.mapping}, issues {p.issues}")
        print(f"\n[{name}] {len(cases) - len(fails)}/{len(cases)} correct, {sum(times) / len(times):.1f} s per call")
        print("\n".join(fails))
        self.assertFalse(fails, f"{len(fails)} {name} failures")

    def test_templates(self):
        cases = [(FIXED_INSTRUCTION, FIXED_MAPPING)] + [
            (fill(t, c, b), dict(zip(c, b))) for t in TEMPLATES
            for c in itertools.permutations(COLORS) for b in itertools.permutations(BINS)]
        n = int(os.environ.get("BRAIN_N", 24))   # default: a fixed random sample (all 8 templates appear)
        self.run_cases("templates", cases[:1] + random.Random(0).sample(cases[1:], min(n, len(cases) - 1)), False)

    def test_paraphrases(self):
        self.run_cases("paraphrases", PARAPHRASES, False)

    def test_anomalies(self):
        self.run_cases("anomalies", ANOMALIES, True)


if __name__ == "__main__":
    unittest.main()
