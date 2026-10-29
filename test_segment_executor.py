"""Verifies overlay_core.SegmentExecutor's ordering: finals first, and only
the newest live update per caption section - no GPU/model needed."""

import threading
import time

import overlay_core


def test_segment_executor_latest_wins():
    ran, gate = [], threading.Event()
    ex = overlay_core.SegmentExecutor()
    ex.submit(lambda: gate.wait(5))  # hold the worker so the rest queue up
    ex.submit_live(("g1", 0), lambda: ran.append("g1 v1"))
    ex.submit_live(("g1", 0), lambda: ran.append("g1 v2"))
    ex.submit_live(("g2", 0), lambda: ran.append("g2 live"))
    ex.submit(lambda **kw: ran.append("g2 final"), caption_group="g2")
    ex.submit(lambda: ran.append("other"))
    gate.set()
    deadline = time.time() + 5
    while len(ran) < 3 and time.time() < deadline:
        time.sleep(0.01)
    time.sleep(0.1)
    assert ran == ["g2 final", "other", "g1 v2"], ran
    print("OK: executor runs finals first, only the newest live update, none superseded by a final")


if __name__ == "__main__":
    test_segment_executor_latest_wins()
    print("ALL OK")
