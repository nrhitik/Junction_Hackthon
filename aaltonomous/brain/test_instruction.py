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
        with fake(("turquoise", None, "left"), (R, "sphere", "center"), (B, None, "top"), (Y, None, "right")):
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
        with fake((None, None, "left"), (R, None, "center"), (R, "cube", "right"), (None, "prism", "right")):
            p = understand("Everything left, red in the center, but red cubes and all triangles right.")
        self.assertEqual([p.bin_for(R, "cube"), p.bin_for(R, "cylinder"), p.bin_for(B, "prism"),
                          p.bin_for(Y, "cube")], ["right", "center", "right", "left"])

    # ---- plan checked against the colours the camera sees (match.py)
    def test_alias_question_and_hold(self):
        with fake((R, None, "left"), (Y, None, "center"), (B, None, "right")):
            p = understand("x", seen={"orange", Y, B})
        self.assertIn("treat orange as red", p.question)
        self.assertEqual([p.bin_for("orange"), p.bin_for(Y), p.bin_for(B)], [None, "center", "right"])

    def test_alias_confirmed_and_declined(self):
        with fake((R, None, "left"), (Y, None, "center"), (B, None, "right")):
            p = understand("x", seen={"orange", Y, B})
        with fake(("orange", None, "left"), (Y, None, "center"), (B, None, "right")):   # the model applied "yes"
            yes = understand("yes", p.messages, seen={"orange", Y, B})
        self.assertEqual((yes.bin_for("orange"), yes.issues, yes.question), ("left", [], None))
        with fake((R, None, "left"), (Y, None, "center"), (B, None, "right")):          # model ignored the "yes":
            yes = understand("Yes please", p.messages, seen={"orange", Y, B})           # Python applies it anyway
        self.assertEqual((yes.bin_for("orange"), yes.issues, yes.question), ("left", [], None))
        with fake((R, None, "left"), (Y, None, "center"), (B, None, "right")):          # "no": nothing changes
            no = understand("no", p.messages, seen={"orange", Y, B})
        self.assertEqual((no.bin_for("orange"), no.question), (None, None))   # stays, and no second question
        self.assertTrue(any("orange" in i for i in no.issues))

    def test_declined_alias_follows_the_other_rules(self):   # "put the green in left and the rest in center"
        seen = {B, Y, "orange"}
        with fake(("green", None, "left"), (None, None, "center")):
            p = understand("x", seen=seen)
            self.assertIn("treat yellow as green", p.question)
            self.assertIsNone(p.bin_for(Y))
            for answer in ("no", "yes", "leave it"):   # whatever is typed, an unchanged plan never asks again
                p = understand(answer, p.messages, seen=seen)
                self.assertEqual((p.bin_for(Y), p.bin_for(B), p.question), ("center", "center", None))

    def test_any_colour_can_be_sorted(self):
        with fake(("green", None, "left"), ("black", None, "right")):
            p = understand("x", seen={"green", "black"})
        self.assertEqual((p.bin_for("green"), p.bin_for("black"), p.issues), ("left", "right", []))

    def test_colour_not_on_the_table(self):
        with fake(("green", None, "left"), (Y, None, "right")):
            p = understand("x", seen={Y})
        self.assertTrue(any("can't see any green" in i for i in p.issues))

    def test_object_without_instruction_stays(self):
        with fake((Y, None, "left"), (B, None, "right")):
            p = understand("x", seen={Y, B, "pink"})
        self.assertIsNone(p.bin_for("pink"))
        self.assertTrue(any("pink" in i and "leave" in i for i in p.issues))
        with fake((Y, None, "left"), (None, None, "right")):     # "everything else" covers it
            p = understand("x", seen={Y, B, "pink"})
        self.assertEqual((p.bin_for("pink"), p.issues), ("right", []))

    def test_robot_waits_for_an_answer(self):
        with fake((R, None, "left"), (Y, None, "center"), (B, None, "right")),                 mock.patch.object(ins, "ask", return_value=None) as asked:
            p = ins.robot_plan("x", seen={"orange", Y, B}, log=lambda *_: None)
        self.assertEqual((asked.call_args[0][1], p.bin_for("orange")), (8.0, None))   # no answer: stays

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

    def test_scene(self):   # the camera sees orange, yellow and blue objects
        seen = {"orange", Y, B}
        p = understand(FIXED_INSTRUCTION, seen=seen)
        self.assertIn("treat orange as red", p.question or "")
        self.assertEqual([p.bin_for("orange"), p.bin_for(Y), p.bin_for(B)], [None, "center", "right"])
        yes = understand("yes", p.messages, seen=seen)
        self.assertEqual([yes.bin_for("orange"), yes.bin_for(Y), yes.bin_for(B)], ["left", "center", "right"])
        p = understand("Orange ones go left, yellow in the middle, everything else on the right.", seen=seen)
        self.assertEqual(([p.bin_for("orange"), p.bin_for(Y), p.bin_for(B)], p.issues), (["left", "center", "right"], []))
        p = understand("put the green in left. and put the rest in center.", seen=seen)
        self.assertIn("treat yellow as green", p.question or "")
        no = understand("no", p.messages, seen=seen)
        self.assertEqual(([no.bin_for(c) for c in ("orange", Y, B)], no.question), (["center"] * 3, None))
        p = understand("Green objects go left, yellow center, blue right, orange left.", seen=seen)
        self.assertTrue(any("can't see any green" in i for i in p.issues), p.issues)


if __name__ == "__main__":
    unittest.main()
