"""
Live translator for Discord (or any system audio).
Captures whatever your PC is currently playing (WASAPI loopback) plus your
mic, detects speech with Silero VAD, transcribes it with Whisper, and
translates it into one or more target languages of your choosing
(--target) with Hy-MT2, a local translation LLM (see translator.py).

Shows captions in a small always-on-top overlay window, one per target.

Usage:
    python translate_vc.py --target en
    python translate_vc.py --target zh
    python translate_vc.py --target en,vi

A single process handles capture/VAD/transcription once per speech segment
regardless of how many targets are requested - only the translation step
is repeated per target.
"""

import argparse
import re
import time

from opencc import OpenCC

import discord_bridge
import translator
from lang_codes import TARGET_LANGUAGES
from overlay_core import is_reliable_transcription, run_translator

_t2s = OpenCC("t2s")
_s2t = OpenCC("s2t")

# Whisper only ever reports "zh" (it doesn't distinguish script) - these are
# the two script variants selectable as --target, both reachable from a "zh"
# source without any model translation, just a script conversion. Also
# applied to the translator's output for these targets, so the script is
# guaranteed rather than left to the model.
CHINESE_TARGETS = {"zh": _t2s, "zh-hant": _s2t}

_CLAUSE_SPLIT_RE = re.compile(r"[,,。.!!??;;\n]+")


def _is_degenerate(text):
    """Guards against a translation that decoded into a degenerate loop
    instead of a real translation - either a long run of one repeated
    character (",,,,,,") or the same short phrase repeated over and over,
    comma-separated (e.g. "我只是很开心,我只是很开心,我只是很开心,...").
    NLLB did this on short, context-less fillers ("Uh-huh.", "Yeah, yeah,
    yeah."); Hy-MT2 didn't in evaluation, but an LLM can loop too, so keep
    catching both rather than show one as if it were a real translation."""
    stripped = text.replace(" ", "")
    if len(stripped) < 6:
        return False

    most_common_char_count = max(stripped.count(c) for c in set(stripped))
    if most_common_char_count / len(stripped) > 0.5:
        return True

    clauses = [c for c in _CLAUSE_SPLIT_RE.split(text) if c.strip()]
    if len(clauses) >= 4:
        most_common_clause_count = max(clauses.count(c) for c in set(clauses))
        if most_common_clause_count >= 4 and most_common_clause_count / len(clauses) > 0.4:
            return True

    return False


def _emit_caption(q, lead, lang, elapsed, native_text, translated_text=None):
    q.put({
        "speaker": lead.strip(),
        "meta": f"[{lang}, {elapsed:.1f}s]",
        "native": native_text,
        "translated": translated_text,
    })


def make_process_segment(targets, caption_queues, translation_enabled):
    """One shared Whisper transcription per speech segment, fanned out to
    every requested target language/overlay - instead of redoing that
    decode work once per target. translation_enabled: {target: bool},
    flipped live by each overlay's "T" button - when off, that target's
    window shows the native transcription only and skips the translation
    request entirely."""

    def process_segment(model, segment, source_tag="", speaker_hint=None):
        if model is None:
            # Submitted while a speech-model swap had the old one unloaded
            # (see ModelHolder.request_reload) - nothing to transcribe with.
            return
        t0 = time.time()
        segments_gen, info = model.transcribe(
            segment,
            task="transcribe",
            vad_filter=False,
            beam_size=5,
            condition_on_previous_text=False,
        )
        native_segments = list(segments_gen)

        if not is_reliable_transcription(native_segments, info):
            if source_tag != "mic":
                # Console-only (not the overlay) - loopback speech getting cut by
                # VAD/the Discord bridge but then discarded here would otherwise be
                # invisible, since this is a silent return.
                avg_no_speech = sum(s.no_speech_prob for s in native_segments) / len(native_segments) if native_segments else None
                avg_logprob = sum(s.avg_logprob for s in native_segments) / len(native_segments) if native_segments else None
                print(f"[debug] dropped unreliable segment (source={source_tag!r} speaker_hint={speaker_hint}): "
                      f"lang={info.language} lang_prob={info.language_probability:.2f} "
                      f"avg_no_speech={avg_no_speech} avg_logprob={avg_logprob}")
            return

        native_text = "".join(s.text for s in native_segments).strip()
        if not native_text:
            if source_tag != "mic":
                print(f"[debug] transcription came back empty (source={source_tag!r} speaker_hint={speaker_hint})")
            return

        if source_tag == "mic":
            lead = "[You] "
        elif speaker_hint is not None:
            # Ground truth from the Discord speaking bridge - the only
            # source of speaker labels this app has; without it, captions
            # are unlabeled rather than guessed.
            _user_id, username = speaker_hint
            lead = f"{username} "
        else:
            lead = ""

        if info.language == "zh":
            native_text = _t2s.convert(native_text)  # normalize display text to Simplified

        for target_lang in targets:
            q = caption_queues[target_lang]

            if info.language == "zh" and target_lang in CHINESE_TARGETS:
                elapsed = time.time() - t0
                shown = CHINESE_TARGETS[target_lang].convert(native_text)
                _emit_caption(q, lead, info.language, elapsed, shown)
                continue

            if info.language == target_lang:
                # already the target language - nothing to translate
                elapsed = time.time() - t0
                _emit_caption(q, lead, info.language, elapsed, native_text)
                continue

            if not translation_enabled.get(target_lang, True):
                elapsed = time.time() - t0
                _emit_caption(q, lead, info.language, elapsed, native_text)
                continue

            # Every target, English included, goes through the translator -
            # it replaced Whisper's own X->English translate task too, which
            # occasionally invented content (see translator.py).
            translated = translator.translate(native_text, info.language, target_lang)
            if target_lang in CHINESE_TARGETS:
                translated = CHINESE_TARGETS[target_lang].convert(translated)

            elapsed = time.time() - t0
            # The translation isn't reliability-checked like the transcription
            # above - it can come back empty (translator still loading, or a
            # failed request) or, rarely, as a degenerate loop, even though
            # the native transcription was solid. Show the native text either
            # way rather than silently dropping the caption or showing that as
            # if it were a translation.
            if _is_degenerate(translated):
                translated = ""
            _emit_caption(q, lead, info.language, elapsed, native_text, translated or None)

    return process_segment


def main():
    parser = argparse.ArgumentParser(description="Live translator overlay for Discord/system audio.")
    parser.add_argument(
        "--target", "-t", default=None,
        help="Comma-separated target language code(s), e.g. 'en,vi'. One shared audio/transcription "
             f"pipeline is used, with one overlay window per target. Supported: {', '.join(sorted(TARGET_LANGUAGES))} "
             "(default: shows a GUI picker for this and the other options below)",
    )
    parser.add_argument(
        "--model", "-m", default=None,
        help="Speech model: a Whisper size, or qwen3-asr-1.7b / qwen3-asr-0.6b (default: large-v3)",
    )
    parser.add_argument(
        "--position", choices=["top", "bottom"], default=None,
        help="Overlay screen anchor for every window (default: bottom for en, top for everything else, per target)",
    )
    parser.add_argument("--no-mic", action="store_true", help="Don't also capture your microphone")
    parser.add_argument(
        "--discord-bridge-port", type=int, default=discord_bridge.DEFAULT_PORT,
        help=f"Port the optional Discord speaking-bridge WebSocket server listens on (default: {discord_bridge.DEFAULT_PORT})",
    )
    args = parser.parse_args()

    # The Settings window (always available - see overlay_core.run_translator)
    # replaces what used to be a one-shot picker dialog. It's meant to start
    # hidden/tray-only for scripted launches that shouldn't pop a window
    # unasked (--target given) - but the tray icon is currently disabled
    # (see run_translator's comment - pystray segfaults against Tkinter in
    # this environment), and hidden-with-no-tray would strand it unreachable.
    # Always show it for now; revert to `args.target is None` once tray's
    # safe to re-enable.
    show_settings = True
    initial_targets = list(dict.fromkeys(t.strip().lower() for t in (args.target or "en").split(",") if t.strip()))
    for target in initial_targets:
        if target not in TARGET_LANGUAGES:
            raise SystemExit(f"Unsupported target language '{target}'. Supported: {', '.join(sorted(TARGET_LANGUAGES))}")

    # Targets are addable live from Settings at any time, not just at
    # startup - so the translator loads regardless of what's initially
    # requested. In the background: the window comes up immediately, with
    # the load tracked in its status panel.
    translator.start()

    model_size = args.model or "large-v3"

    run_translator(
        make_process_segment,
        initial_targets,
        model_size=model_size,
        capture_mic=not args.no_mic,
        discord_bridge_port=args.discord_bridge_port,
        show_settings=show_settings,
        initial_position=args.position,
    )


if __name__ == "__main__":
    main()
