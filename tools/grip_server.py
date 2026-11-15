"""Classical baseline policy server + JSON config overrides from $AALTO_OVERRIDES (team tool, for A/B gate runs).

    AALTO_OVERRIDES='{"hand.closed": [...]}' PORT=8000 aalto py /data/team/tools/grip_server.py
"""
import json
import logging
import os
import time

from aaltonomous.baselines.classical.policy import ClassicalPolicy
from aaltonomous.baselines.classical import policy_server as ps

log = logging.getLogger("aaltonomous.baseline")
_orig = ClassicalPolicy.__init__


def _patched(self, *a, overrides=None, **kw):
    ov = dict(overrides or {})
    ov.update(json.loads(os.environ.get("AALTO_OVERRIDES", "{}")))
    _orig(self, *a, overrides=ov, **kw)


ClassicalPolicy.__init__ = _patched

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.monotonic()
    pol = ps.ClassicalServerPolicy()
    log.info("overrides %s; hand.closed=%s; loaded in %.1f s", os.environ.get("AALTO_OVERRIDES"),
             pol.pol.cfg["hand"]["closed"], time.monotonic() - t0)
    ps.serve(pol, port=int(os.environ.get("PORT", 8000)))
