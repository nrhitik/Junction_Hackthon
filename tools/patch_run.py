p = "/data/work/team/il_insert/run.py"
s = open(p).read()
if "--shape" not in s:
    s = s.replace('''    ap.add_argument("--one-each", action="store_true", default=True)''',
                  '''    ap.add_argument("--one-each", action="store_true", default=True)
    ap.add_argument("--shape", default=None, help="team: ONE object of this shape per episode (cube/cylinder/triangle)")''')
    s = s.replace('''    import aaltonomous.interface.env as E
    if a.one_each:''', '''    import aaltonomous.interface.env as E
    if a.shape:  # team: one object of one shape (train / test the final stage shape by shape)
        _orig_s = E.sample_layout

        def _one(rng, **kw):
            lay = _orig_s(rng, **{**kw, "n_min": 1, "n_max": 1})
            lay[0]["shape"] = a.shape
            return lay
        E.sample_layout = _one
    elif a.one_each:''')
    open(p, "w").write(s)
print("--shape" in s)
