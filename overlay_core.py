"""
The capture pipeline behind translate_vc.py: system audio (WASAPI loopback)
and mic capture, speech segmentation (Silero VAD, or Discord's own speaking
events through the bridge plugin), live and final caption timing, and the
speech model and worker transcription runs on. A single process_segment_fn
serves every target language. Pipeline is the handle the UI (ui/) starts,
reconfigures and stops it through.
"""

import itertools
import os
import queue
import threading
import time
from collections import deque

import numpy as np
import pyaudiowpatch as pyaudio
import torch
from faster_whisper import WhisperModel

import discord_bridge
import settings

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

VAD_SAMPLE_RATE = 16000
FRAME_SAMPLES = 512              # 32ms @ 16kHz, required chunk size for Silero VAD
SPEECH_THRESHOLD = 0.5
SILENCE_HANGOVER_SEC = 0.8       # how much trailing silence ends a segment
MIN_SPEECH_SEC = 0.4             # ignore blips shorter than this

# How far back each capture thread keeps timestamped audio: the Discord
# bridge slices its intervals out of loopback's, and every caption - live
# or final - is re-sliced from it (see _slice_timestamped_buffer).
BRIDGE_BUFFER_WINDOW_SEC = 30.0
BRIDGE_SLICE_PAD_SEC = 0.2        # clock-skew cushion on each side of a slice
# While someone's talking, one live caption (marked "…") grows: every
# live_options["interval"] seconds, what they've said is transcribed again
# and replaces it in place, the text extending and correcting itself as
# context arrives. When they pause, a final pass over the whole stretch
# replaces it once more. Live updates never queue up behind each other -
# only the newest per caption section waits (see SegmentExecutor) - and
# wait for at least LIVE_MIN_AUDIO_SEC of speech, since a shorter clip is
# too little for the model to place the language.
LIVE_MIN_AUDIO_SEC = 1.0
# Transcription time grows with the audio (~0.35s for 1.5s, 1.1s for 6s,
# 3.5s for 20s on Qwen3-ASR-1.7B), so re-transcribing everything said so
# far would soon take longer than the update interval. Instead a live
# caption grows in sections: once the current one passes this long its
# text is kept as it stands and updates carry on with a new section from
# there - shown joined up as one caption, each update at most ~1s of work.
# (The final pass still redoes the whole stretch in one go.)
LIVE_SECTION_SEC = 6.0
# Speech running longer than this without a pause has what's there so far
# finalized as its own caption, and carries on in a new one - so no single
# transcription (or growing live caption) exceeds it.
BRIDGE_FINAL_MAX_SEC = 20.0
# Discord flips someone's speaking indicator off at every short pause, so
# one train of thought often arrives as several back-to-back intervals.
# When the same person speaks again within this long, with nobody else
# captioned in between, the joined audio (up to BRIDGE_FINAL_MAX_SEC from
# the first interval's start) is transcribed again and that one caption
# replaces their previous one - one sentence, not three fragments.
BRIDGE_JOIN_GAP_SEC = 2.0
# A live update at least this much covered by the speaking of others who
# started after this speaker did is skipped - in the mixed loopback audio
# it's mostly their speech, e.g. while someone's open mic keeps them
# "speaking" through everyone else's conversation. Only later starters
# count, so of two people flagged over the same speech (an echo) one still
# gets live captions.
LIVE_OTHERS_SKIP_FRAC = 0.5
# The same for a stretch finalized at BRIDGE_FINAL_MAX_SEC, but stricter: a
# final caption is permanent, and 20s+ of nonstop "speaking" is more often
# an open mic than a monologue - a real one has at most the odd
# interjection from anyone else.
LONG_OTHERS_SKIP_FRAC = 0.25
# Someone replying just before the speaker's interval closes: their opening
# words are in the mixed audio at its end (Discord also holds "speaking" a
# moment past the last word). A final caption's audio stops this long after
# the reply started, if the reply started within REPLY_TRIM_WINDOW_SEC of
# the end - so it isn't captioned under the wrong name, twice.
REPLY_TRIM_WINDOW_SEC = 1.5
REPLY_TRIM_KEEP_SEC = 0.3

# The speech model's conversation language (None = auto-detect), context
# words, and whether to add the Discord channel members' names to them -
# set live from Settings (see settings.CONVERSATION_LANGUAGES) and read by
# process_segment for every segment, through speech_context().
speech_options = {"language": None, "context": "", "auto_names": True}
# Live captions: seconds between updates of a speaker's growing live
# caption (None = off - only the caption on each pause), and whether those
# updates are translated too or show the original text until the pause.
# Set live from Settings.
live_options = {"interval": 1.5, "translate": True}


# At most this many channel names go into the speech model's context. A
# long list of names - several of them display names nobody says out loud
# - makes the model more likely to echo the list back as the
# "transcription" of a short or unclear clip (see translate_vc's
# _echoes_context, which catches that).
MAX_CONTEXT_NAMES = 12


def _speakable_name(name):
    """A Discord display name cut down to something a speech model could
    actually hear said - emoji and decoration stripped - or None if
    nothing name-like is left ("?", "🍁🍁") or it reads more like a status
    than what anyone calls them out loud: over two words or 16 characters
    ("silly student try hard"), or a Chinese name over 4 characters
    ("蟹黄宝神秘配方")."""
    cleaned = " ".join("".join(c if c.isalnum() or c in " '-" else " " for c in name).split())
    if not any(c.isalpha() for c in cleaned) or len(cleaned) > 16 or len(cleaned.split()) > 2:
        return None
    if all("⺀" <= c <= "鿿" for c in cleaned) and len(cleaned) > 4:
        return None
    return cleaned


def speech_context():
    """The context words passed to the speech model: Settings' "Names &
    slang", plus the Discord channel members' names when auto_names is on -
    recomputed per segment, so people joining or leaving the call are
    picked up as it happens."""
    words = [w.strip() for w in speech_options["context"].split(",") if w.strip()]
    if speech_options["auto_names"]:
        seen = {w.lower() for w in words}
        added = 0
        for name in discord_bridge.channel_names():
            name = _speakable_name(name)
            if name and name.lower() not in seen and added < MAX_CONTEXT_NAMES:
                seen.add(name.lower())
                words.append(name)
                added += 1
    return ", ".join(words)

MAX_HISTORY = 300                # cap on stored caption entries so the log doesn't grow unbounded

# Anti-hallucination thresholds: Whisper doesn't say "nothing here" for silence
# or noise it's unsure about - it fabricates plausible-sounding text instead.
# These mirror the heuristics OpenAI's own reference decoder uses to flag that.
NO_SPEECH_PROB_MAX = 0.6
AVG_LOGPROB_MIN = -1.0
LANGUAGE_PROB_MIN = 0.5

# Set live from the UI. paused: nothing is transcribed (capture keeps
# running, so resuming is instant). caption_mic: whether your own mic is
# captioned - off, its speech isn't even transcribed, saving the GPU work.
capture_options = {"paused": False, "caption_mic": True}


class SegmentExecutor:
    """The single worker every transcription (and model reload) runs on,
    one at a time - like ThreadPoolExecutor(max_workers=1), plus
    latest-wins live updates. submit(): queued in order, always run, and
    ahead of any live update. submit_live((group, section), ...): a live
    caption update - it replaces one for the same caption section still
    waiting (only the newest matters), and is dropped if a submit() for
    that caption group arrives first (its caption_group/absorb kwargs),
    since that final caption supersedes it. So live updates can't build a backlog however short the
    interval: when the worker's busy, the stale ones just never run."""

    def __init__(self):
        self._cv = threading.Condition()
        self._jobs: "deque" = deque()
        self._live: dict = {}  # key -> job; insertion order = oldest first
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, fn, *args, **kwargs):
        with self._cv:
            groups = {kwargs.get("caption_group"), *kwargs.get("absorb", ())}
            for key in [k for k in self._live if k[0] in groups]:
                del self._live[key]
            self._jobs.append((fn, args, kwargs))
            self._cv.notify()

    def submit_live(self, key, fn, *args, **kwargs):
        with self._cv:
            self._live.pop(key, None)  # re-insert at the back - its turn is by when it was last updated
            self._live[key] = (fn, args, kwargs)
            self._cv.notify()

    def _run(self):
        while True:
            with self._cv:
                while not self._jobs and not self._live:
                    self._cv.wait()
                fn, args, kwargs = self._jobs.popleft() if self._jobs else self._live.pop(next(iter(self._live)))
            try:
                fn(*args, **kwargs)
            except Exception as e:  # one bad segment mustn't take down the worker
                print(f"[worker] {type(e).__name__}: {e}")


def _submit_live(executor, fn, *args, **kwargs):
    submit_live = getattr(executor, "submit_live", None)
    if submit_live is None:  # a plain executor (tests) - just queue it
        executor.submit(fn, *args, **kwargs)
    else:
        submit_live((kwargs["caption_group"], kwargs["live_section"]), fn, *args, **kwargs)


def _load_speech_model(size, device, compute_type):
    """A WhisperModel, or - for a "qwen3-asr-*" size - Qwen3-ASR behind the
    same transcribe() interface (see qwen_asr_client.py)."""
    from qwen_asr_client import QwenASRModel, is_qwen_model  # local import - it imports this module
    if is_qwen_model(size):
        return QwenASRModel(size)
    return WhisperModel(size, device=device, compute_type=compute_type)


def _release_speech_model(model):
    close = getattr(model, "close", None)  # Qwen3-ASR's server process; Whisper just needs dropping
    if close is not None:
        close()


class ModelHolder:
    """Thread-safe holder for the current speech model (a WhisperModel, or
    Qwen3-ASR - see _load_speech_model), swappable at runtime.
    The capture loop fetches .get() at *submission* time (not once at
    worker-start), so a reload never yanks the model out from under an
    already-submitted transcription job. request_reload() queues the swap on
    the shared single-worker executor, so it's naturally serialized with
    in-flight/queued segments - no extra locking needed for that part."""

    def __init__(self, size, device="cuda", compute_type="float16"):
        self._lock = threading.Lock()
        self._model = None
        self.size = size
        self.device = device
        self.compute_type = compute_type
        self._executor = None

    def get(self):
        with self._lock:
            return self._model

    def _load_initial(self):
        status.set("whisper_model", f"Loading ({self.size})...")
        broadcast(f"Loading speech model ({self.size}) on {self.device}...")
        model = _load_speech_model(self.size, self.device, self.compute_type)
        with self._lock:
            self._model = model
        status.set("whisper_model", f"Ready ({self.size})")

    def _bind_executor(self, executor):
        self._executor = executor

    def request_reload(self, new_size):
        """Called from the Settings window (Tk thread) when the user picks a
        different model size. No-op if the pipeline isn't ready yet or the
        size didn't actually change."""
        if self._executor is None or new_size == self.size or not new_size:
            return

        def _do_reload():
            old_size = self.size
            status.set("whisper_model", f"Reloading ({old_size} -> {new_size})...")
            broadcast(f"Reloading speech model ({old_size} -> {new_size})...")
            # Unload before loading, rather than swapping once the new one is
            # up: next to the translator there isn't VRAM for two speech
            # models at once. Captioning is paused either way - this runs
            # on the same single-worker executor as transcription.
            old_model = self.get()
            with self._lock:
                self._model = None
            _release_speech_model(old_model)
            del old_model
            if self.device == "cuda":
                torch.cuda.empty_cache()
            loaded_size = new_size
            try:
                new_model = _load_speech_model(new_size, self.device, self.compute_type)
            except Exception as e:
                broadcast(f"[ERROR] model reload failed: {type(e).__name__}: {e} - reverting to {old_size}")
                loaded_size = old_size
                new_model = _load_speech_model(old_size, self.device, self.compute_type)
            with self._lock:
                self._model = new_model
                self.size = loaded_size
            if loaded_size == new_size:
                settings.update(model=new_size)
            status.set("whisper_model", f"Ready ({loaded_size})")
            broadcast(f"Model reloaded ({loaded_size}).")

        self._executor.submit(_do_reload)


class MicCapture:
    """Owns the mic PyAudio stream + capture thread, startable/stoppable
    independently of loopback capture and the rest of the pipeline. Usable
    (checkbox-driven) before the audio pipeline finishes loading - toggling
    early just changes what happens once bind() runs after load."""

    def __init__(self, initially_on):
        self._desired_on = initially_on
        self._p = None
        self._model_holder = None
        self._executor = None
        self._process_segment_fn = None
        self._stream = None
        self._thread = None
        self._stop_event = None

    @property
    def enabled(self):
        return self._desired_on

    @property
    def active(self):
        return self._thread is not None

    def bind(self, p, model_holder, executor, process_segment_fn):
        self._p, self._model_holder, self._executor = p, model_holder, executor
        self._process_segment_fn = process_segment_fn
        if self._desired_on:
            self._start_now()
        else:
            status.set("mic", "Off")

    def set_enabled(self, on):
        """Safe to call from the Tk thread (the Settings mic checkbox does) -
        the actual start work happens on a background thread since it can
        block for a while (torch.hub.load, opening the audio device), and
        blocking the Tk thread would freeze the whole UI."""
        self._desired_on = on
        if self._p is None:
            return  # not bound yet - bind() will apply _desired_on once loaded
        if on:
            threading.Thread(target=self._start_now, daemon=True).start()
        else:
            self._stop_now()

    def _start_now(self):
        if self.active:
            return
        status.set("mic", "Loading...")
        # Silero VAD is a stateful streaming model (it carries hidden state
        # between calls) - it needs its own instance, separate from
        # loopback's, same as at initial startup.
        vad_model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        if not self._desired_on:
            status.set("mic", "Off")
            return  # toggled off again while VAD was loading - abandon

        mic_q: "queue.Queue" = queue.Queue()
        stream, info, channels = open_mic_stream(self._p, make_pyaudio_callback(mic_q))
        if not self._desired_on:
            stream.close()
            status.set("mic", "Off")
            return  # toggled off again while the device was opening - abandon

        stream.start_stream()
        stop_event = threading.Event()
        thread = threading.Thread(
            target=_capture_loop,
            args=(stop_event, mic_q, int(info["defaultSampleRate"]), channels, vad_model,
                  self._model_holder, self._executor, self._process_segment_fn, "mic"),
            daemon=True,
        )
        thread.start()
        self._stream, self._thread, self._stop_event = stream, thread, stop_event

    def _stop_now(self):
        if not self.active:
            return
        self._stop_event.set()
        self._stream.stop_stream()
        self._stream.close()
        self._stream = self._thread = self._stop_event = None
        status.set("mic", "Off")

    def shutdown(self):
        self._stop_now()


_broadcast_queues: "list[queue.Queue]" = []


def register_broadcast_queues(queues):
    _broadcast_queues[:] = queues


def broadcast(msg):
    for q in _broadcast_queues:
        q.put(msg)


STATUS_FIELDS = [
    ("vad", "VAD"),
    ("whisper_model", "Speech model"),
    ("translator", "Translation (Hy-MT2)"),
    ("loopback", "System audio"),
    ("mic", "Microphone"),
    ("discord_bridge", "Discord bridge"),
]


class SystemStatus:
    """Structured per-subsystem status (VAD/speech model/translator/audio
    devices/Discord bridge), mirrored into the main window's status panel. set() is called from whichever
    thread owns that subsystem (the audio worker thread, the model-reload
    executor thread, etc.) - safe from any thread, since listeners are
    plain queue.Queue.put() calls."""

    def __init__(self):
        self.values = {key: "..." for key, _ in STATUS_FIELDS}
        self._listeners: "list[queue.Queue]" = []

    def subscribe(self, q: "queue.Queue"):
        self._listeners.append(q)
        q.put(dict(self.values))  # immediate snapshot

    def set(self, key, text):
        self.values[key] = text
        snapshot = dict(self.values)
        for q in self._listeners:
            q.put(snapshot)


status = SystemStatus()


def is_reliable_transcription(segments, info):
    """False if this looks like a hallucination on silence/noise rather than real speech."""
    if not segments or info.language_probability < LANGUAGE_PROB_MIN:
        return False
    avg_no_speech = sum(s.no_speech_prob for s in segments) / len(segments)
    avg_logprob = sum(s.avg_logprob for s in segments) / len(segments)
    return avg_no_speech <= NO_SPEECH_PROB_MAX and avg_logprob >= AVG_LOGPROB_MIN


def make_pyaudio_callback(audio_q: "queue.Queue[tuple[float, bytes]]"):
    def callback(in_data, frame_count, time_info, status):
        audio_q.put((time.time(), in_data))
        return (None, pyaudio.paContinue)
    return callback


def open_loopback_stream(p, callback):
    wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

    if not default_speakers.get("isLoopbackDevice", False):
        for loopback in p.get_loopback_device_info_generator():
            if default_speakers["name"] in loopback["name"]:
                default_speakers = loopback
                break
        else:
            raise RuntimeError(
                "Could not find a loopback device matching your default speakers. "
                "Try setting your Discord/Windows output device explicitly."
            )

    broadcast(f"[Capturing system audio: {default_speakers['name']}]")
    status.set("loopback", f"Capturing: {default_speakers['name']}")

    stream = p.open(
        format=pyaudio.paInt16,
        channels=default_speakers["maxInputChannels"],
        rate=int(default_speakers["defaultSampleRate"]),
        frames_per_buffer=1024,
        input=True,
        input_device_index=default_speakers["index"],
        stream_callback=callback,
    )
    return stream, default_speakers


def open_mic_stream(p, callback):
    default_mic = p.get_default_input_device_info()
    channels = min(int(default_mic["maxInputChannels"]), 2) or 1

    broadcast(f"[Capturing mic: {default_mic['name']}]")
    status.set("mic", f"Capturing: {default_mic['name']}")

    stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=int(default_mic["defaultSampleRate"]),
        frames_per_buffer=1024,
        input=True,
        input_device_index=default_mic["index"],
        stream_callback=callback,
    )
    return stream, default_mic, channels


def resample_linear(audio, orig_sr, target_sr):
    if orig_sr == target_sr or len(audio) == 0:
        return audio
    duration = len(audio) / orig_sr
    target_len = max(1, int(duration * target_sr))
    x_old = np.linspace(0, duration, num=len(audio), endpoint=False)
    x_new = np.linspace(0, duration, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def _capture_loop(stop_event, audio_q, native_rate, channels, vad_model, model_holder, executor,
                   process_segment_fn, source_tag):
    try:
        _capture_loop_inner(stop_event, audio_q, native_rate, channels, vad_model, model_holder, executor,
                             process_segment_fn, source_tag)
    except Exception as e:
        label = "mic" if source_tag == "mic" else "system audio"
        broadcast(f"[ERROR] {label} capture stopped: {type(e).__name__}: {e}")


def _slice_timestamped_buffer(buffer, start_ts, end_ts):
    """buffer: deque of (chunk_start_ts, pcm_16k) - concatenate whichever
    chunks overlap [start_ts, end_ts] (padded for clock skew). Only
    chunk-level, not sample-exact, precision - Discord's timestamps and our
    local capture-time timestamps are close but not perfectly aligned."""
    lo = start_ts - BRIDGE_SLICE_PAD_SEC
    hi = end_ts + BRIDGE_SLICE_PAD_SEC
    parts = [arr for ts, arr in buffer if ts + len(arr) / VAD_SAMPLE_RATE >= lo and ts <= hi]
    return np.concatenate(parts) if parts else None


# Open speaking intervals, keyed by (user_id, start): {"group": their live
# caption's group (None until the first update), "seg_start": where the
# current caption's audio starts (later than the interval's start once
# BRIDGE_FINAL_MAX_SEC has been finalized), "section"/"section_start": the
# live caption's current section (see LIVE_SECTION_SEC), "last_update":
# ts}. Only touched from the loopback capture thread.
_live_intervals: dict = {}
# The latest captioned run of back-to-back speech per source - a Discord
# user id, or "mic"/"loopback" for VAD-cut audio - as {"group", "start",
# "end"}; see BRIDGE_JOIN_GAP_SEC. Only the most recently captioned
# source's run can still be joined onto. Shared by the loopback and mic
# capture threads.
_runs: dict = {}
_runs_lock = threading.Lock()
_caption_group_ids = itertools.count(1)


def _new_caption_group():
    return f"g{next(_caption_group_ids)}"


def _plan_final(key, start, end, live_group, may_join=lambda run_end: True):
    """Where a finished stretch of speech by `key` (from `start` to `end`,
    with a live caption in `live_group`, if any) should be transcribed
    from, and which caption group its final caption takes: -> (slice_start,
    group, absorb). Joins it onto the same source's previous caption if
    that ended within BRIDGE_JOIN_GAP_SEC, the result stays within
    BRIDGE_FINAL_MAX_SEC, and may_join(previous_end) agrees; otherwise just
    the stretch itself."""
    with _runs_lock:
        run = _runs.get(key)
    if (run and start - run["end"] <= BRIDGE_JOIN_GAP_SEC and end - run["start"] <= BRIDGE_FINAL_MAX_SEC
            and may_join(run["end"])):
        # Same source again right after its last caption - redo the whole
        # run as one caption replacing it (and this stretch's live
        # caption, if it had one).
        return run["start"], run["group"], [live_group] if live_group else []
    return start, live_group or _new_caption_group(), []


def _commit_run(key, group, start, end):
    """Records a final caption about to be submitted - and that someone
    else's caption now sits after every other source's last one, so their
    runs can't be joined onto any more."""
    with _runs_lock:
        _runs.clear()
        _runs[key] = {"group": group, "start": start, "end": end}


def _drain_discord_segments(timestamped_buffer, model_holder, executor, process_segment_fn):
    """Cuts a segment for every speaking interval the Discord bridge has
    just closed, using its (user_id, username) as a speaker_hint instead of
    the VAD + voice-embedding guess - plus growing live captions for speech
    still in progress (see LIVE_MIN_AUDIO_SEC). Loopback-only - Discord
    audio only ever arrives via loopback, never the mic. model_holder:
    anything with a .get() returning the model to transcribe with (fetched
    fresh per segment, so a live model reload is picked up immediately)."""
    if capture_options["paused"]:
        discord_bridge.drain_closed_intervals()  # dropped, not left to all arrive on resume
        _live_intervals.clear()
        return
    now = time.time()
    interval = live_options["interval"]
    open_now = discord_bridge.open_intervals()
    open_keys = {(u, st) for u, _, st in open_now}
    for key in [k for k in _live_intervals if k[1] < now - 60 and k not in open_keys]:
        del _live_intervals[key]  # closed without a stop event reaching the queue
    for user_id, username, start in open_now:
        live = _live_intervals.setdefault((user_id, start), {"group": None, "seg_start": start, "last_update": start,
                                                             "section": 0, "section_start": start})
        others = discord_bridge.others_speaking_fraction(user_id, live["seg_start"], now, started_after=start)

        if now - live["seg_start"] >= BRIDGE_FINAL_MAX_SEC:
            # Talking past what one caption takes - finalize what's there
            # and carry on in a new caption.
            segment = _slice_timestamped_buffer(timestamped_buffer, live["seg_start"], now)
            if segment is not None and others < LONG_OTHERS_SKIP_FRAC:
                group = live["group"] or _new_caption_group()
                _commit_run(user_id, group, live["seg_start"], now)
                executor.submit(process_segment_fn, model_holder.get(), segment, "", (user_id, username),
                                caption_group=group, final=True, speech_end=now)
            elif live["group"]:  # mostly others' speech - take its live caption back
                executor.submit(process_segment_fn, model_holder.get(), None, "", (user_id, username),
                                caption_group=live["group"], final=True)
            live.update(group=None, seg_start=now, last_update=now, section=0, section_start=now)
            continue

        if (interval is None or now - live["last_update"] < interval
                or now - live["seg_start"] < LIVE_MIN_AUDIO_SEC):
            continue
        if now - live["section_start"] > LIVE_SECTION_SEC:
            # Keep the current section's text as last shown; carry on
            # with a new one from where that ended.
            live["section"] += 1
            live["section_start"] = live["last_update"]
        live["last_update"] = now
        if others >= LIVE_OTHERS_SKIP_FRAC:
            continue  # mostly someone else's speech in the mix
        segment = _slice_timestamped_buffer(timestamped_buffer, live["section_start"], now)
        if segment is not None:
            live["group"] = live["group"] or _new_caption_group()
            _submit_live(executor, process_segment_fn, model_holder.get(), segment, "", (user_id, username),
                         caption_group=live["group"], final=False, live_section=live["section"], speech_end=now)

    for closed in discord_bridge.drain_closed_intervals():
        live = _live_intervals.pop((closed["user_id"], closed["start"]), None)
        live_group = live["group"] if live else None
        seg_start = live["seg_start"] if live else closed["start"]
        user_id = closed["user_id"]
        speaker_hint = (user_id, closed["username"])
        duration = closed["end"] - closed["start"]
        if closed.get("covered_by"):
            # Same mixed audio as another user's segment - see
            # discord_bridge.OVERLAP_SKIP_FRAC.
            print(f"[debug] skipped {closed['username']}'s {duration:.1f}s interval - "
                  f"overlaps {closed['covered_by']}'s")
            if live_group:  # take back the live caption it already got
                executor.submit(process_segment_fn, model_holder.get(), None, "", speaker_hint,
                                caption_group=live_group, final=True)
            continue
        if duration < MIN_SPEECH_SEC or closed["end"] - seg_start < MIN_SPEECH_SEC:
            continue
        if (seg_start != closed["start"] and discord_bridge.others_speaking_fraction(
                user_id, seg_start, closed["end"], started_after=closed["start"]) >= LONG_OTHERS_SKIP_FRAC):
            # The rest of an overlong interval, largely others' speech
            if live_group:
                executor.submit(process_segment_fn, model_holder.get(), None, "", speaker_hint,
                                caption_group=live_group, final=True)
            continue

        start, group, absorb = _plan_final(
            user_id, seg_start, closed["end"], live_group,
            # Not across a gap someone else spoke in - the joined audio
            # would take in the start of their reply.
            may_join=lambda run_end: discord_bridge.others_speaking_fraction(user_id, run_end, closed["start"]) == 0,
        )
        end = closed["end"]
        reply = discord_bridge.first_reply_start(user_id, max(seg_start, end - REPLY_TRIM_WINDOW_SEC), end)
        if reply is not None:
            end = reply + REPLY_TRIM_KEEP_SEC
        segment = _slice_timestamped_buffer(timestamped_buffer, start, end)
        if segment is None or len(segment) < int(MIN_SPEECH_SEC * VAD_SAMPLE_RATE):
            continue
        _commit_run(user_id, group, start, closed["end"])
        executor.submit(process_segment_fn, model_holder.get(), segment, "", speaker_hint,
                        caption_group=group, final=True, absorb=absorb, speech_end=closed["end"])


def _capture_loop_inner(stop_event, audio_q, native_rate, channels, vad_model, model_holder, executor,
                         process_segment_fn, source_tag):
    leftover = np.zeros(0, dtype=np.float32)
    in_speech = False
    silence_frames = 0
    silence_frames_needed = int(SILENCE_HANGOVER_SEC * VAD_SAMPLE_RATE / FRAME_SAMPLES)
    min_speech_frames = int(MIN_SPEECH_SEC * VAD_SAMPLE_RATE / FRAME_SAMPLES)
    frame_sec = FRAME_SAMPLES / VAD_SAMPLE_RATE

    is_loopback = source_tag == ""
    run_key = "loopback" if is_loopback else source_tag
    # Recent audio by capture time, for both sources: the Discord bridge
    # slices its intervals out of loopback's, and VAD-cut speech is sliced
    # from it for every live update and final caption.
    timestamped_buffer: "deque" = deque()
    bridge_was_connected = False  # only meaningful when is_loopback
    last_bridge_status_text = None
    # The current VAD utterance: where its caption's audio starts, how many
    # frames of it were speech, its live caption's group (None until the
    # first update), current section (see LIVE_SECTION_SEC) and when it was
    # last updated.
    utt_start = utt_group = utt_last_update = utt_section_start = None
    utt_speech_frames = utt_section = 0

    while not stop_event.is_set():
        try:
            chunk_ts, raw = audio_q.get(timeout=1)
        except queue.Empty:
            continue

        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            pcm = pcm.reshape(-1, channels).mean(axis=1)

        pcm_16k = resample_linear(pcm, native_rate, VAD_SAMPLE_RATE)

        timestamped_buffer.append((chunk_ts, pcm_16k))
        cutoff = chunk_ts - BRIDGE_BUFFER_WINDOW_SEC
        while timestamped_buffer and timestamped_buffer[0][0] < cutoff:
            timestamped_buffer.popleft()

        if is_loopback:
            connected = discord_bridge.is_connected()
            bridge_active = discord_bridge.is_active()

            if connected != bridge_was_connected:
                # Surface genuine connect/disconnect only - is_active() is an
                # *activity* signal (gates segmentation below), not a
                # connection one, and flips on every lull in conversation
                # even though the WebSocket itself never dropped (see its
                # docstring). Logging every flip of that would spam the
                # overlay constantly during completely normal conversation.
                broadcast(
                    "[Discord bridge: connected]" if connected else
                    "[Discord bridge: disconnected - falling back to VAD segmentation]"
                )
                bridge_was_connected = connected

            status_text = (
                ("Connected (active)" if bridge_active else "Connected (idle - VAD fallback)") if connected
                else "Listening (no client - VAD fallback)"
            )
            if status_text != last_bridge_status_text:
                status.set("discord_bridge", status_text)
                last_bridge_status_text = status_text

            if bridge_active:
                _drain_discord_segments(timestamped_buffer, model_holder, executor, process_segment_fn)
        else:
            bridge_active = False

        buf = np.concatenate([leftover, pcm_16k])
        # Capture time at the end of buf - frame i ends (len(buf) - (i+1)
        # frames) earlier. Chunk-level precision, like the bridge slicing.
        buf_end_ts = chunk_ts + len(pcm_16k) / VAD_SAMPLE_RATE

        n_frames = len(buf) // FRAME_SAMPLES
        usable = n_frames * FRAME_SAMPLES
        leftover = buf[usable:]

        for i in range(n_frames):
            frame = buf[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
            frame_end = buf_end_ts - (len(buf) - (i + 1) * FRAME_SAMPLES) / VAD_SAMPLE_RATE
            prob = vad_model(torch.from_numpy(frame), VAD_SAMPLE_RATE).item()

            if prob >= SPEECH_THRESHOLD:
                if utt_start is None:
                    utt_start = utt_last_update = utt_section_start = frame_end - frame_sec
                    utt_section = 0
                utt_speech_frames += 1
                silence_frames = 0
                in_speech = True
            elif in_speech:
                silence_frames += 1
            if not in_speech:
                continue

            # the Discord bridge, when active, drives loopback segmentation
            # itself (see _drain_discord_segments) - still run VAD to keep
            # its state warm for an instant, glitch-free fallback the moment
            # the bridge drops, but don't double-emit the same audio
            # through both paths
            self_captioning_off = source_tag == "mic" and not capture_options["caption_mic"]
            emit = not bridge_active and not self_captioning_off and not capture_options["paused"]
            interval = live_options["interval"]

            if silence_frames >= silence_frames_needed:
                # They paused - the whole utterance (or the run it joins)
                # once more, replacing its live caption. Same treatment as
                # a closed Discord interval.
                speech_end = frame_end - silence_frames * frame_sec
                if emit and (utt_group is not None or utt_speech_frames >= min_speech_frames):
                    start, group, absorb = _plan_final(run_key, utt_start, speech_end, utt_group)
                    segment = _slice_timestamped_buffer(timestamped_buffer, start, speech_end)
                    if segment is not None:
                        _commit_run(run_key, group, start, speech_end)
                        executor.submit(process_segment_fn, model_holder.get(), segment, source_tag,
                                        caption_group=group, final=True, absorb=absorb, speech_end=speech_end)
                utt_start = utt_group = utt_last_update = utt_section_start = None
                utt_speech_frames = 0
                in_speech = False
                silence_frames = 0
            elif frame_end - utt_start >= BRIDGE_FINAL_MAX_SEC:
                # Talking past what one caption takes - finalize what's
                # there and carry on in a new caption.
                if emit:
                    group = utt_group or _new_caption_group()
                    segment = _slice_timestamped_buffer(timestamped_buffer, utt_start, frame_end)
                    if segment is not None:
                        _commit_run(run_key, group, utt_start, frame_end)
                        executor.submit(process_segment_fn, model_holder.get(), segment, source_tag,
                                        caption_group=group, final=True, speech_end=frame_end)
                utt_start = utt_last_update = utt_section_start = frame_end
                utt_group = None
                utt_speech_frames = utt_section = 0
            elif (emit and interval is not None and silence_frames == 0
                    and frame_end - utt_last_update >= interval and frame_end - utt_start >= LIVE_MIN_AUDIO_SEC):
                # Still talking - update the growing live caption.
                if frame_end - utt_section_start > LIVE_SECTION_SEC:
                    utt_section += 1
                    utt_section_start = utt_last_update
                utt_last_update = frame_end
                utt_group = utt_group or _new_caption_group()
                segment = _slice_timestamped_buffer(timestamped_buffer, utt_section_start, frame_end)
                if segment is not None:
                    _submit_live(executor, process_segment_fn, model_holder.get(), segment, source_tag,
                                 caption_group=utt_group, final=False, live_section=utt_section,
                                 speech_end=frame_end)


def audio_worker(stop_event: threading.Event, model_holder: ModelHolder, process_segment_fn,
                  mic_capture: MicCapture):
    try:
        status.set("vad", "Loading...")
        broadcast("Loading VAD model...")
        vad_model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        status.set("vad", "Ready")

        p = pyaudio.PyAudio()
        executor = SegmentExecutor()
        # Bind the executor before the initial load, not after - otherwise
        # there's a narrow window where status already says the model is
        # "Ready" (Settings re-enables the model combo off that signal) but
        # request_reload() would still silently no-op since _executor is None.
        model_holder._bind_executor(executor)
        model_holder._load_initial()

        loopback_q: "queue.Queue[bytes]" = queue.Queue()
        loopback_stream, loopback_info = open_loopback_stream(p, make_pyaudio_callback(loopback_q))
        loopback_stream.start_stream()
        t = threading.Thread(
            target=_capture_loop,
            args=(stop_event, loopback_q, int(loopback_info["defaultSampleRate"]),
                  loopback_info["maxInputChannels"], vad_model, model_holder, executor, process_segment_fn, ""),
            daemon=True,
        )
        t.start()

        mic_capture.bind(p, model_holder, executor, process_segment_fn)

        broadcast("Listening...")

        while not stop_event.is_set():
            time.sleep(0.2)

        mic_capture.shutdown()
        loopback_stream.stop_stream()
        loopback_stream.close()
        p.terminate()
    except Exception as e:
        broadcast(f"[ERROR] {type(e).__name__}: {e}")


class Pipeline:
    """The running capture pipeline, as the UI sees it: start() it once,
    reconfigure it live through model_holder / mic_capture /
    set_bridge_port(), and shutdown() on exit."""

    def __init__(self, process_segment_fn, model_size, capture_mic, bridge_port, device="cuda",
                 compute_type="float16"):
        self.process_segment_fn = process_segment_fn
        self.model_holder = ModelHolder(model_size, device=device, compute_type=compute_type)
        self.mic_capture = MicCapture(initially_on=capture_mic)
        self.bridge_port = bridge_port
        self._stop_event = threading.Event()

    def start(self):
        threading.Thread(target=audio_worker, daemon=True,
                         args=(self._stop_event, self.model_holder, self.process_segment_fn, self.mic_capture)).start()
        discord_bridge.start(port=self.bridge_port)
        status.set("discord_bridge", f"Listening on port {self.bridge_port} (no client - VAD fallback)")

    def set_bridge_port(self, port):
        """Restarts the bridge server on another port - off the calling
        (UI) thread, since stopping joins the server thread."""
        self.bridge_port = port

        def _apply():
            discord_bridge.stop()
            discord_bridge.start(port=port)
            status.set("discord_bridge", f"Listening on port {port} (no client - VAD fallback)")
        threading.Thread(target=_apply, daemon=True).start()

    def shutdown(self):
        self._stop_event.set()
        discord_bridge.stop()
