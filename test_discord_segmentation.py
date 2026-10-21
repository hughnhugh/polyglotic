"""Verifies overlay_core's Discord-bridge-driven segmentation
(_slice_timestamped_buffer / _drain_discord_segments) in isolation, driven
through discord_bridge's real message-handling path, with synthetic
timestamped audio - no GPU/Whisper/real Discord call needed."""

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import discord_bridge
import overlay_core

SR = overlay_core.VAD_SAMPLE_RATE


def _make_buffer(seconds, chunk_sec=0.1, start_ts=None):
    start_ts = start_ts if start_ts is not None else time.time()
    chunk_len = int(chunk_sec * SR)
    n = int(seconds / chunk_sec)
    buf = deque()
    for i in range(n):
        buf.append((start_ts + i * chunk_sec, np.zeros(chunk_len, dtype=np.float32)))
    return buf, start_ts


class _FakeModelHolder:
    """Stands in for overlay_core.ModelHolder - _drain_discord_segments only
    ever needs .get()."""
    def __init__(self, model):
        self._model = model

    def get(self):
        return self._model


def _run_drain(buffer):
    calls = []
    executor = ThreadPoolExecutor(max_workers=1)
    overlay_core._drain_discord_segments(buffer, _FakeModelHolder("FAKE_MODEL"), executor, lambda *a, **kw: calls.append(a + (kw,)))
    executor.shutdown(wait=True)
    return calls


def _reset():
    overlay_core.live_options.update(interval=1.5, translate=True)
    discord_bridge._intervals.clear()
    overlay_core._runs.clear()
    overlay_core._live_intervals.clear()


def test_speaking_interval_cuts_expected_segment():
    _reset()
    buffer, t0 = _make_buffer(seconds=2.0)
    start_ts, end_ts = t0 + 0.3, t0 + 1.3  # 1s interval inside the buffered window

    discord_bridge._handle_message({"event": "channel_sync", "members": {"42": "Zaphod"}})
    discord_bridge._handle_message({"event": "speaking_start", "userId": "42", "ts": start_ts * 1000})
    discord_bridge._handle_message({"event": "speaking_stop", "userId": "42", "ts": end_ts * 1000})

    calls = _run_drain(buffer)
    assert len(calls) == 1, f"expected exactly one segment cut, got {calls}"
    model, segment, source_tag, speaker_hint, _ = calls[0]
    assert model == "FAKE_MODEL"
    assert source_tag == "", f"expected loopback source_tag '', got {source_tag!r}"
    assert speaker_hint == ("42", "Zaphod"), f"unexpected speaker_hint: {speaker_hint}"

    expected_len = int((end_ts - start_ts) * SR)
    pad_len = int(2 * overlay_core.BRIDGE_SLICE_PAD_SEC * SR)
    assert expected_len <= len(segment) <= expected_len + pad_len + int(0.2 * SR), (
        f"segment length {len(segment)} outside expected range around {expected_len}"
    )
    print(f"OK: {end_ts - start_ts:.1f}s interval -> {len(segment)} samples (~{len(segment)/SR:.2f}s)")


def test_short_interval_is_dropped():
    _reset()
    buffer, t0 = _make_buffer(seconds=1.0)
    discord_bridge._handle_message({"event": "speaking_start", "userId": "1", "username": "Blip", "ts": t0 * 1000})
    discord_bridge._handle_message({"event": "speaking_stop", "userId": "1", "ts": (t0 + 0.1) * 1000})  # < MIN_SPEECH_SEC

    calls = _run_drain(buffer)
    assert calls == [], f"expected sub-MIN_SPEECH_SEC interval to be dropped, got {calls}"
    print("OK: short interval correctly dropped")


def test_long_interval_transcribed_whole():
    # Up to BRIDGE_FINAL_MAX_SEC, a closed interval is transcribed in one
    # piece.
    _reset()
    buffer, t0 = _make_buffer(seconds=12.0)
    discord_bridge._handle_message({"event": "speaking_start", "userId": "7", "username": "Marvin", "ts": t0 * 1000})
    discord_bridge._handle_message({"event": "speaking_stop", "userId": "7", "ts": (t0 + 12.0) * 1000})

    calls = _run_drain(buffer)
    assert len(calls) == 1
    assert len(calls[0][1]) >= int(11.5 * SR), f"12s interval was cut to {len(calls[0][1]) / SR:.2f}s"
    print(f"OK: 12s interval transcribed whole ({len(calls[0][1]) / SR:.2f}s)")


def test_live_captions_then_final_replaces_them():
    # Still speaking -> a live caption of everything so far; once they stop,
    # the whole interval again, in the same caption group, marked final.
    _reset()
    now = time.time()
    buffer, _ = _make_buffer(seconds=12.0, start_ts=now - 12.0)
    _speak("8", "Slartibartfast", now - 7.0)

    live = _run_drain(buffer)
    assert len(live) == 1 and live[0][4]["final"] is False and live[0][4]["caption_group"], live
    group = live[0][4]["caption_group"]
    assert _run_drain(buffer) == [], "a second live update came before the interval passed"

    _stop("8", now)
    final = _run_drain(buffer)
    assert len(final) == 1 and final[0][4]["caption_group"] == group and final[0][4]["final"] is True and final[0][4]["absorb"] == [], final
    assert len(final[0][1]) >= int(6.5 * SR), f"final segment only {len(final[0][1]) / SR:.2f}s"
    print(f"OK: live caption at 7s, then a {len(final[0][1]) / SR:.2f}s final replacing it ({group})")


def test_skipped_interval_takes_back_its_live_captions():
    _reset()
    now = time.time()
    buffer, _ = _make_buffer(seconds=12.0, start_ts=now - 12.0)
    _speak("9", "Eddie", now - 9.0)
    _speak("10", "Deep Thought", now - 8.0)  # open-mic echo of Eddie's speech
    live = _run_drain(buffer)
    assert [c[3][1] for c in live] == ["Deep Thought"], live  # the echo pair: only the later starter gets live pieces
    group = next(c[4]["caption_group"] for c in live if c[3][1] == "Deep Thought")

    _stop("10", now - 0.5)
    _stop("9", now)
    calls = _run_drain(buffer)
    removal = [c for c in calls if c[1] is None]
    assert len(removal) == 1 and removal[0][4] == {"caption_group": group, "final": True}, calls
    assert [c[3][1] for c in calls if c[1] is not None] == ["Eddie"], calls
    print("OK: covered interval's live caption taken back, speaker's final kept")


def _speak(user_id, username, start_ts, end_ts=None):
    discord_bridge._handle_message({"event": "speaking_start", "userId": user_id, "username": username, "ts": start_ts * 1000})
    if end_ts is not None:
        discord_bridge._handle_message({"event": "speaking_stop", "userId": user_id, "ts": end_ts * 1000})


def _stop(user_id, end_ts):
    discord_bridge._handle_message({"event": "speaking_stop", "userId": user_id, "ts": end_ts * 1000})


def test_noise_during_speech_neither_cuts_nor_duplicates():
    # Someone's short noise mid-sentence: the speaker's interval stays open
    # (intervals are per user), so their sentence is captioned whole once,
    # and the noise-maker's slice - the same mixed audio - is skipped.
    _reset()
    buffer, t0 = _make_buffer(seconds=5.0)
    _speak("1", "Arthur", t0 + 0.2)
    _speak("2", "Ford", t0 + 1.5, t0 + 2.1)
    _stop("1", t0 + 4.2)

    calls = _run_drain(buffer)
    assert [c[3][1] for c in calls] == ["Arthur"], f"expected only Arthur's segment, got {[c[3] for c in calls]}"
    assert len(calls[0][1]) >= int(4.0 * SR), f"Arthur's segment was cut short: {len(calls[0][1]) / SR:.2f}s"
    print(f"OK: noise mid-sentence skipped, speaker's {len(calls[0][1]) / SR:.2f}s segment intact")


def test_near_identical_intervals_captioned_once():
    # Two users flagged speaking over the same span (e.g. one's mic picking
    # up the other) - the same audio would otherwise be captioned twice.
    _reset()
    buffer, t0 = _make_buffer(seconds=3.0)
    _speak("3", "Caleb", t0 + 0.30)
    _speak("4", "Resul", t0 + 0.35)
    _stop("3", t0 + 1.80)
    _stop("4", t0 + 1.85)

    calls = _run_drain(buffer)
    assert len(calls) == 1, f"expected one segment, got {[c[3] for c in calls]}"
    print(f"OK: near-identical intervals captioned once (as {calls[0][3][1]})")


def test_back_and_forth_keeps_both():
    # A reply that starts just before the previous speaker stops is mostly
    # its own audio - both get captioned.
    _reset()
    buffer, t0 = _make_buffer(seconds=4.0)
    _speak("5", "Trillian", t0 + 0.2, t0 + 1.5)
    _speak("6", "Zaphod", t0 + 1.3, t0 + 3.0)

    calls = _run_drain(buffer)
    assert [c[3][1] for c in calls] == ["Trillian", "Zaphod"], f"got {[c[3] for c in calls]}"
    print("OK: overlapping turn-taking keeps both speakers")


def test_consecutive_intervals_joined():
    # Discord split one person's sentence at a pause: the second interval's
    # final re-transcribes from the first's start and replaces its caption.
    _reset()
    buffer, t0 = _make_buffer(seconds=8.0)
    _speak("11", "Arthur", t0 + 0.5, t0 + 2.5)
    first = _run_drain(buffer)
    _speak("11", "Arthur", t0 + 3.5, t0 + 5.5)  # 1s pause
    second = _run_drain(buffer)
    group = first[0][4]["caption_group"]
    assert second[0][4]["caption_group"] == group, (first, second)
    assert len(second[0][1]) >= int(4.9 * SR), f"joined segment only {len(second[0][1]) / SR:.2f}s"
    _speak("11", "Arthur", t0 + 6.0, t0 + 7.5)
    third = _run_drain(buffer)
    assert third[0][4]["caption_group"] == group and len(third[0][1]) >= int(6.9 * SR), third
    print(f"OK: three back-to-back intervals joined into one {len(third[0][1]) / SR:.2f}s caption ({group})")


def test_join_breaks_on_other_speaker_or_long_gap():
    _reset()
    buffer, t0 = _make_buffer(seconds=12.0)
    _speak("12", "Ford", t0 + 0.5, t0 + 2.0)
    _speak("13", "Zaphod", t0 + 2.2, t0 + 3.0)
    _speak("12", "Ford", t0 + 3.2, t0 + 4.5)
    calls = _run_drain(buffer)
    groups = [c[4]["caption_group"] for c in calls]
    assert len(set(groups)) == 3, f"someone else spoke in between - expected no join, got {groups}"
    assert len(calls[2][1]) < int(3.0 * SR), "Ford's second interval was joined across Zaphod's"

    _speak("12", "Ford", t0 + 7.0, t0 + 8.5)  # 2.5s after Ford's last
    calls = _run_drain(buffer)
    assert calls[0][4]["caption_group"] not in groups and len(calls[0][1]) < int(3.0 * SR), calls
    print("OK: no join across another speaker's caption or a pause over BRIDGE_JOIN_GAP_SEC")


def test_join_absorbs_live_captions():
    # A long interval right after the same person's last caption: its live
    # captions get their own group, and the joined final absorbs them.
    _reset()
    now = time.time()
    buffer, _ = _make_buffer(seconds=12.0, start_ts=now - 12.0)
    _speak("14", "Trillian", now - 10.0, now - 8.5)
    first = _run_drain(buffer)
    _speak("14", "Trillian", now - 7.5)
    live = _run_drain(buffer)
    _stop("14", now)
    final = _run_drain(buffer)
    kw = final[0][4]
    assert kw["caption_group"] == first[0][4]["caption_group"] and kw["absorb"] == [live[0][4]["caption_group"]], (first, live, final)
    assert len(final[0][1]) >= int(9.9 * SR), f"joined final only {len(final[0][1]) / SR:.2f}s"
    print("OK: joined final replaces the earlier caption and absorbs the live ones")


def test_open_mic_cannot_cover_everyone():
    # Someone's mic held "speaking" by background noise for minutes: other
    # people's speech inside it must still be captioned, not skipped as
    # covered by it - and the open mic's 20s stretch, largely their speech,
    # isn't finalized under its name.
    _reset()
    now = time.time()
    buffer, _ = _make_buffer(seconds=24.0, start_ts=now - 24.0)
    _speak("20", "OpenMic", now - 24.0)            # still open, 24s and counting
    _speak("21", "Ford", now - 15.0, now - 9.0)
    _speak("23", "Zaphod", now - 8.0, now - 2.0)

    calls = _run_drain(buffer)
    speakers = [c[3][1] for c in calls if c[1] is not None]
    assert "Ford" in speakers and "Zaphod" in speakers, f"speech skipped as covered by the open mic: {speakers}"
    assert "OpenMic" not in speakers, f"open mic's stretch captioned under its name: {speakers}"
    print(f"OK: open mic for 24s doesn't swallow anyone's caption, nor get one of theirs ({speakers})")

    # A live update for the open mic made mostly of someone else's speech is skipped.
    _reset()
    now = time.time()
    buffer, _ = _make_buffer(seconds=12.0, start_ts=now - 12.0)
    _speak("20", "OpenMic", now - 7.0)
    _speak("21", "Ford", now - 6.5)
    calls = _run_drain(buffer)
    assert [c[3][1] for c in calls] == ["Ford"], [c[3] for c in calls]
    print("OK: live update that's mostly someone else's speech skipped")


def test_join_blocked_by_reply_in_gap():
    # Zaphod gets a word in during Arthur's pause (too short to caption on
    # its own) - joining Arthur's pieces would sweep it in under his name.
    _reset()
    buffer, t0 = _make_buffer(seconds=8.0)
    _speak("22", "Arthur", t0 + 0.5, t0 + 2.0)
    first = _run_drain(buffer)
    _speak("23", "Zaphod", t0 + 2.4, t0 + 2.7)
    _speak("22", "Arthur", t0 + 3.5, t0 + 5.0)
    second = _run_drain(buffer)
    assert second[0][4]["caption_group"] != first[0][4]["caption_group"], (first, second)
    print("OK: no join across a gap someone else spoke in")


def test_reply_at_the_end_is_trimmed_off():
    # Ford starts replying 0.8s before Arthur's interval closes - Arthur's
    # final stops 0.3s into Ford's reply instead of taking all of it.
    _reset()
    buffer, t0 = _make_buffer(seconds=6.0)
    _speak("24", "Arthur", t0 + 0.5)
    _speak("25", "Ford", t0 + 3.2)
    _stop("24", t0 + 4.0)
    calls = _run_drain(buffer)
    arthur = [c for c in calls if c[3][1] == "Arthur"]
    expected = (3.2 + 0.3 - 0.5) + 2 * overlay_core.BRIDGE_SLICE_PAD_SEC
    assert arthur and abs(len(arthur[0][1]) / SR - expected) <= 0.21, (len(arthur[0][1]) / SR, expected)
    print(f"OK: Arthur's final trimmed to {len(arthur[0][1]) / SR:.2f}s, leaving out Ford's reply")


def test_live_caption_grows_and_rolls_over():
    # Each live update covers everything since the caption started; at
    # BRIDGE_FINAL_MAX_SEC what's there is finalized and a new caption
    # starts. With live updates off there are none, only the final.
    _reset()
    now = time.time()
    buffer, _ = _make_buffer(seconds=26.0, start_ts=now - 26.0)
    _speak("30", "Marvin", now - 21.0)
    calls = _run_drain(buffer)
    assert len(calls) == 1 and calls[0][4]["final"] is True and len(calls[0][1]) / SR >= 20.5, calls
    first_group = calls[0][4]["caption_group"]
    live = overlay_core._live_intervals[("30", now - 21.0)]
    live["last_update"] -= 2.0  # as if 2s had passed since the rollover
    live["seg_start"] -= 2.0
    live["section_start"] -= 2.0
    calls = _run_drain(buffer)
    assert len(calls) == 1 and calls[0][4]["final"] is False and calls[0][4]["caption_group"] != first_group, calls
    assert 1.9 <= len(calls[0][1]) / SR <= 2.7, len(calls[0][1]) / SR
    print("OK: 21s of speech -> first 20s finalized, a new live caption carries on from there")

    _reset()
    overlay_core.live_options["interval"] = None
    now = time.time()
    buffer, _ = _make_buffer(seconds=8.0, start_ts=now - 8.0)
    _speak("31", "Trillian", now - 6.0)
    assert _run_drain(buffer) == [], "live update with live captions off"
    _stop("31", now)
    calls = _run_drain(buffer)
    assert len(calls) == 1 and calls[0][4]["final"] is True
    print("OK: live captions off -> only the final caption")


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
    test_speaking_interval_cuts_expected_segment()
    test_short_interval_is_dropped()
    test_long_interval_transcribed_whole()
    test_live_captions_then_final_replaces_them()
    test_skipped_interval_takes_back_its_live_captions()
    test_noise_during_speech_neither_cuts_nor_duplicates()
    test_near_identical_intervals_captioned_once()
    test_back_and_forth_keeps_both()
    test_consecutive_intervals_joined()
    test_join_breaks_on_other_speaker_or_long_gap()
    test_join_absorbs_live_captions()
    test_open_mic_cannot_cover_everyone()
    test_join_blocked_by_reply_in_gap()
    test_reply_at_the_end_is_trimmed_off()
    test_live_caption_grows_and_rolls_over()
    test_segment_executor_latest_wins()
    print("ALL OK")
