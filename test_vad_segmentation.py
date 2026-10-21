"""Verifies VAD-cut speech (the mic, and loopback when the Discord bridge is
idle) gets the same live -> final -> join treatment as Discord intervals,
by driving overlay_core._capture_loop_inner with synthetic audio and a
stand-in VAD that calls any loud frame speech - no GPU/model needed."""

import queue
import threading
import time

import numpy as np
import torch

import overlay_core

SR = overlay_core.VAD_SAMPLE_RATE
CHUNK = 1024


class _LoudnessVAD:
    def __call__(self, frame, sr):
        return torch.tensor(1.0 if float(frame.abs().max()) > 0.1 else 0.0)


class _RecordingExecutor:
    def __init__(self):
        self.calls = []

    def submit(self, fn, *args, **kwargs):
        self.calls.append((args, kwargs))


class _FakeModelHolder:
    def get(self):
        return "FAKE_MODEL"


def _run(pattern, captioning_on=True):
    """pattern: [(seconds, speaking?)] -> the submitted calls, after
    feeding it as 16kHz mic audio timestamped from t0."""
    overlay_core._runs.clear()
    rng = np.random.default_rng(0)
    audio = np.concatenate([
        (rng.uniform(-0.5, 0.5, int(sec * SR)) if loud else np.zeros(int(sec * SR))) for sec, loud in pattern
    ])
    pcm = (audio * 32767).astype(np.int16)
    t0 = time.time() - len(pcm) / SR
    q = queue.Queue()
    for i in range(0, len(pcm) - CHUNK + 1, CHUNK):
        q.put((t0 + i / SR, pcm[i:i + CHUNK].tobytes()))

    state = overlay_core.SelfCaptionState()
    state.enabled = captioning_on
    executor, stop = _RecordingExecutor(), threading.Event()
    worker = threading.Thread(target=overlay_core._capture_loop_inner,
                              args=(stop, q, SR, 1, _LoudnessVAD(), _FakeModelHolder(), executor,
                                    lambda *a, **k: None, "mic", state))
    worker.start()
    while not q.empty():
        time.sleep(0.05)
    time.sleep(0.3)
    stop.set()
    worker.join()
    return executor.calls, t0


def test_short_utterance_one_final():
    calls, t0 = _run([(0.5, False), (3.0, True), (1.5, False)])
    assert len(calls) == 1, calls
    (model, segment, source), kw = calls[0]
    assert source == "mic" and kw["final"] is True and kw["absorb"] == []
    assert 3.0 <= len(segment) / SR <= 3.6, len(segment) / SR
    assert abs(kw["speech_end"] - (t0 + 3.5)) < 0.15, kw["speech_end"] - t0
    print(f"OK: 3s utterance -> one {len(segment) / SR:.2f}s final caption, speech end at {kw['speech_end'] - t0:.2f}s")


def test_long_utterance_live_then_whole_final():
    calls, _ = _run([(0.5, False), (9.0, True), (1.5, False)])
    assert [k["final"] for _, k in calls] == [False, True], calls
    assert calls[0][1]["caption_group"] == calls[1][1]["caption_group"]
    assert len(calls[1][0][1]) / SR >= 8.9, len(calls[1][0][1]) / SR
    print(f"OK: 9s utterance -> live piece at 6s, then the whole {len(calls[1][0][1]) / SR:.2f}s as its final")


def test_back_to_back_utterances_joined():
    calls, _ = _run([(0.5, False), (2.0, True), (1.2, False), (2.0, True), (1.5, False)])
    assert len(calls) == 2 and calls[0][1]["caption_group"] == calls[1][1]["caption_group"], calls
    assert len(calls[1][0][1]) / SR >= 5.1, len(calls[1][0][1]) / SR
    print(f"OK: two utterances 1.2s apart joined into one {len(calls[1][0][1]) / SR:.2f}s caption")


def test_long_pause_not_joined():
    calls, _ = _run([(0.5, False), (2.0, True), (3.0, False), (2.0, True), (1.5, False)])
    assert len(calls) == 2 and calls[0][1]["caption_group"] != calls[1][1]["caption_group"], calls
    print("OK: utterances 3s apart stay separate captions")


def test_self_captioning_off_submits_nothing():
    calls, _ = _run([(0.5, False), (9.0, True), (1.5, False)], captioning_on=False)
    assert calls == [], calls
    print("OK: nothing submitted with self-captioning off")


if __name__ == "__main__":
    test_short_utterance_one_final()
    test_long_utterance_live_then_whole_final()
    test_back_to_back_utterances_joined()
    test_long_pause_not_joined()
    test_self_captioning_off_submits_nothing()
    print("ALL OK")
