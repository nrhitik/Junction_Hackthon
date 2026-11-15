"""Team: rule-based instruction parser (no LLM). Sentence -> [(colour|None, shape|None, bin|None), ...]
(bin None = leave it on the table). Returns None when it cannot read the sentence safely: the policy then sorts
nothing (0 points, never a wrong bin).

Handles: the organisers' sort_fixed / sort_language templates (one colour + one bin per clause, bin before or after
the colour), shape words ("cubes left", "purple cylinders to the right bin"), "respectively" lists, "everything
else / the rest / all other objects -> bin", "everything except X -> bin" (X stays), "leave / don't move X".
"""
import re

COLOR_WORDS = {"orange": "orange", "yellow": "yellow", "purple": "purple", "violet": "purple", "red": "red",
               "blue": "blue", "green": "green"}
SHAPE_WORDS = {"cube": "cube", "cubes": "cube", "square": "cube", "squares": "cube", "block": "cube", "blocks": "cube",
               "box": None, "cylinder": "cylinder", "cylinders": "cylinder", "round": "cylinder", "circle": "cylinder",
               "circles": "cylinder", "circular": "cylinder", "prism": "prism", "prisms": "prism",
               "triangle": "prism", "triangles": "prism", "triangular": "prism", "wedge": "prism", "wedges": "prism"}
BIN_WORDS = {"left": "left", "center": "center", "centre": "center", "middle": "center", "right": "right"}
REST = r"\b(everything else|anything else|the rest|all (?:the )?other(?: objects| ones| pieces)?|other objects|" \
       r"remaining(?: objects| ones)?|all remaining)\b"
LEAVE = r"\b(leave|don'?t (?:move|touch|sort|pick)|do not (?:move|touch|sort|pick)|ignore|keep .* on the table|" \
        r"stays? on the table|should stay)\b"


def _items(clause):
    """Colour/shape/bin words of a clause in order."""
    toks = re.findall(r"[a-z']+", clause)
    cols = [COLOR_WORDS[w] for w in toks if w in COLOR_WORDS]
    shps = [SHAPE_WORDS[w] for w in toks if SHAPE_WORDS.get(w)]
    bins = [BIN_WORDS[w] for w in toks if w in BIN_WORDS]
    return cols, shps, bins


def parse_rules(text):
    t = " " + text.lower().replace("’", "'") + " "
    t = re.sub(r"\bbox(es)?\b", "bin", t)
    if not t.strip():
        return None
    rules = []
    # "X, Y and Z ... in the A, B and C bins respectively"
    if "respectively" in t:
        cols, shps, bins = _items(t)
        keys = [(c, None) for c in cols] if cols else [(None, s) for s in shps]
        if keys and len(keys) == len(bins):
            return [(c, s, b) for (c, s), b in zip(keys, bins)]
        return None
    # "everything except the prisms in the left bin" -> (None, None, left) + (None, prism, None)
    m = re.search(r"\b(?:everything|all(?: the)?(?: objects)?|all of them)\s+(?:but|except)(?: for)?\s+(.*?)\s+"
                  r"(?:in|into|to|goes? (?:in|into|to))\s+the\s+(left|center|centre|middle|right)\b", t)
    if m:
        cols, shps, _ = _items(m.group(1))
        rules.append((None, None, BIN_WORDS[m.group(2)]))
        rules += [(c, None, None) for c in cols] + [(None, s, None) for s in shps]
        t = t.replace(m.group(0), " ")
    # clauses: sentence / comma / semicolon / "and" / "then" / "while" boundaries
    for clause in re.split(r"[.;,:!?]|\band\b|\bthen\b|\bwhile\b|\bwhereas\b", t):
        if not clause.strip():
            continue
        cols, shps, bins = _items(clause)
        if re.search(REST, clause):
            if len(bins) == 1:
                rules.append((None, None, bins[0]))
                continue
            if re.search(LEAVE, clause) and not bins:
                rules.append((None, None, None))
                continue
            return None
        if re.search(LEAVE, clause) and not bins:
            if cols or shps:
                rules += [(c, shps[0] if len(shps) == 1 else None, None) for c in cols] or \
                         [(None, s, None) for s in shps]
            continue
        if not cols and not shps:
            if bins:
                return None   # a bin with no object words ("left bin is for ..."? split badly): unsafe
            continue
        if len(bins) != 1 or len(cols) > 1 or len(set(shps)) > 1:
            return None       # more than one target in one clause: not safe to guess
        rules.append((cols[0] if cols else None, shps[0] if shps else None, bins[0]))
    if not rules:
        return None
    keys = [(c, s) for c, s, _ in rules]
    if len(keys) != len(set(keys)):
        # same key twice with different bins -> contradictory
        seen = {}
        for c, s, b in rules:
            if seen.setdefault((c, s), b) != b:
                return None
    return rules
