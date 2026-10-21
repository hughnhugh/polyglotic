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
import settings
from overlay_core import is_reliable_transcription, live_options, run_translator, speech_context, speech_options

_t2s = OpenCC("t2s")
_s2t = OpenCC("s2t")

# Whisper only ever reports "zh" (it doesn't distinguish script) - these are
# the two script variants selectable as --target, both reachable from a "zh"
# source without any model translation, just a script conversion. Also
# applied to the translator's output for these targets, so the script is
# guaranteed rather than left to the model.
CHINESE_TARGETS = {"zh": _t2s, "zh-hant": _s2t}

_s2hk = OpenCC("s2hk")
# Simplified 系 is three Traditional characters: in spoken Cantonese almost
# always 係 ("is" - 係咪, 唔係, 真係), which s2hk outside its phrase list
# leaves as 系 ("但系") or turns into 繫 ("真繫"). So both become 係, except
# in words that really are 繫 (tie) or 系 (system, series, department).
_XI_WORD_RE = re.compile(r"系(?=統|列|數|譜|綜)|(?<=[體派星河語水學科陽文日韓港台美歐英直旁])系")
_TIE_WORD_RE = re.compile(r"(?<=[聯維牽心])繫|繫(?=上|住|鞋|緊)")


def _cantonese_script(text):
    """Cantonese in Hong Kong Traditional characters - for Cantonese speech
    (the speech model writes it in Simplified) and as a safety net on the
    translator's Cantonese output. Colloquial Cantonese characters (嘅、咗、
    唔、佢...) are the same in both scripts, so they pass straight through."""
    converted = _s2hk.convert(text)
    keep = {m.start() for m in _XI_WORD_RE.finditer(converted)} | {m.start() for m in _TIE_WORD_RE.finditer(converted)}
    return "".join("係" if c in "系繫" and i not in keep else c for i, c in enumerate(converted))

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


def _echoes_context(text, context):
    """True if a transcription is just the context words read back - the
    speech model echoing its prompt for a short or unclear clip ("Caleb,
    CrèmeBrûlée, JuGGerNauT, kaektus, marina, ..." as someone's caption)
    rather than what was said: three or more of the context's entries in
    it, or two making up most of it. Someone saying a single name still
    gets through."""
    lowered = text.lower()
    hits = [w for w in (w.strip().lower() for w in context.split(",")) if len(w) >= 2 and w in lowered]
    if len(hits) >= 3:
        return True
    covered = sum(len(w) for w in hits)
    return len(hits) == 2 and covered >= 0.6 * len(lowered.replace(" ", "").replace(",", ""))


def _emit_caption(q, lead, lang, elapsed, native_text, translated_text=None, group=None, final=False, absorb=(),
                  speech_end=None, live_section=0):
    q.put({
        "speaker": lead.strip(),
        "lang": lang,
        "proc": elapsed,
        "speech_end": speech_end,  # the overlay turns this into the shown delay
        "meta": f"[{lang}, {elapsed:.1f}s]",
        "native": native_text,
        "translated": translated_text,
        "group": group,
        "final": final,
        "absorb": list(absorb),
        "live_section": live_section,
    })


def make_process_segment(targets, caption_queues, translation_enabled):
    """One shared Whisper transcription per speech segment, fanned out to
    every requested target language/overlay - instead of redoing that
    decode work once per target. translation_enabled: {target: bool},
    flipped live by each overlay's "T" button - when off, that target's
    window shows the native transcription only and skips the translation
    request entirely."""

    def process_segment(model, segment, source_tag="", speaker_hint=None, caption_group=None, final=False,
                        absorb=(), speech_end=None, live_section=0):
        """caption_group/final: a long Discord interval's live captions
        (final=False) share a group, and its final caption (final=True)
        replaces them in each overlay - see overlay_core.BRIDGE_FINAL_MAX_SEC.
        A final also replaces the captions of the groups in absorb - a
        speaker's back-to-back intervals joined into one caption (see
        overlay_core.BRIDGE_JOIN_GAP_SEC). segment None with a group just
        removes that group's live captions. If the final pass yields
        nothing usable the earlier captions stay. speech_end: when the
        speech ended, for the overlay's end-to-end delay readout.
        live_section: which section of its growing live caption a live
        update is (see overlay_core.LIVE_SECTION_SEC)."""
        if caption_group and final and segment is None:
            for q in caption_queues.values():
                q.put({"group": caption_group, "final": True, "native": None})
            return
        if model is None:
            # Submitted while a speech-model swap had the old one unloaded
            # (see ModelHolder.request_reload) - nothing to transcribe with.
            return
        t0 = time.time()
        options = dict(task="transcribe", vad_filter=False, beam_size=5, condition_on_previous_text=False,
                       language=speech_options["language"], initial_prompt=speech_context() or None)
        try:
            segments_gen, info = model.transcribe(segment, **options)
        except ValueError:
            # A forced language this model doesn't know (Cantonese on a
            # pre-large-v3 Whisper) - fall back to detecting it.
            segments_gen, info = model.transcribe(segment, **dict(options, language=None))
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
        if options["initial_prompt"] and _echoes_context(native_text, options["initial_prompt"]):
            print(f"[debug] dropped a transcription echoing the context words (speaker_hint={speaker_hint}): "
                  f"{native_text!r}")
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
        elif info.language == "yue":
            native_text = _cantonese_script(native_text)  # Cantonese reads in Traditional

        for target_lang in targets:
            q = caption_queues[target_lang]

            if info.language == "zh" and target_lang in CHINESE_TARGETS:
                elapsed = time.time() - t0
                shown = CHINESE_TARGETS[target_lang].convert(native_text)
                _emit_caption(q, lead, info.language, elapsed, shown, None, caption_group, final, absorb, speech_end, live_section)
                continue

            if info.language == target_lang:
                # already the target language - nothing to translate
                elapsed = time.time() - t0
                _emit_caption(q, lead, info.language, elapsed, native_text, None, caption_group, final, absorb, speech_end, live_section)
                continue

            if not translation_enabled.get(target_lang, True) or (caption_group and not final
                                                                  and not live_options["translate"]):
                elapsed = time.time() - t0
                _emit_caption(q, lead, info.language, elapsed, native_text, None, caption_group, final, absorb, speech_end, live_section)
                continue

            # Every target, English included, goes through the translator -
            # it replaced Whisper's own X->English translate task too, which
            # occasionally invented content (see translator.py).
            translated = translator.translate(native_text, info.language, target_lang)
            if target_lang in CHINESE_TARGETS:
                translated = CHINESE_TARGETS[target_lang].convert(translated)
            elif target_lang == "yue":
                translated = _cantonese_script(translated)

            elapsed = time.time() - t0
            # The translation isn't reliability-checked like the transcription
            # above - it can come back empty (translator still loading, or a
            # failed request) or, rarely, as a degenerate loop, even though
            # the native transcription was solid. Show the native text either
            # way rather than silently dropping the caption or showing that as
            # if it were a translation.
            if _is_degenerate(translated):
                translated = ""
            _emit_caption(q, lead, info.language, elapsed, native_text, translated or None, caption_group, final, absorb, speech_end, live_section)

    return process_segment


def main():
    saved = settings.load()
    parser = argparse.ArgumentParser(
        description="Live translator overlay for Discord/system audio. Options left out come from the last run "
                    "(settings.json).")
    parser.add_argument(
        "--target", "-t", default=None,
        help="Comma-separated target language code(s), e.g. 'en,vi'. One shared audio/transcription "
             f"pipeline is used, with one overlay window per target. Supported: {', '.join(sorted(TARGET_LANGUAGES))}",
    )
    parser.add_argument(
        "--model", "-m", default=None,
        help="Speech model: a Whisper size, or qwen3-asr-1.7b / qwen3-asr-0.6b "
             "(default: qwen3-asr-1.7b if installed, else large-v3)",
    )
    parser.add_argument(
        "--position", choices=["top", "bottom"], default=None,
        help="Overlay screen anchor for every window (default: bottom for en, top for everything else, per target)",
    )
    parser.add_argument("--no-mic", action="store_true", help="Don't also capture your microphone")
    parser.add_argument(
        "--discord-bridge-port", type=int, default=None,
        help=f"Port the optional Discord speaking-bridge WebSocket server listens on (default: {discord_bridge.DEFAULT_PORT})",
    )
    args = parser.parse_args()

    # The Settings window is meant to start hidden/tray-only for scripted
    # launches that shouldn't pop a window unasked - but the tray icon is
    # currently disabled (see run_translator's comment - pystray segfaults
    # against Tkinter in this environment), and hidden-with-no-tray would
    # strand it unreachable. Always show it for now.
    show_settings = True
    target_arg = args.target or ",".join(saved["targets"]) or "en"
    initial_targets = list(dict.fromkeys(t.strip().lower() for t in target_arg.split(",") if t.strip()))
    for target in initial_targets:
        if target not in TARGET_LANGUAGES:
            raise SystemExit(f"Unsupported target language '{target}'. Supported: {', '.join(sorted(TARGET_LANGUAGES))}")

    speech_options["language"] = saved["language"]
    speech_options["context"] = saved["context"]
    speech_options["auto_names"] = saved["auto_names"]
    live_options["interval"] = saved["live_interval"]
    live_options["translate"] = saved["translate_live"]

    # Targets are addable live from Settings at any time, not just at
    # startup - so the translator loads regardless of what's initially
    # requested. In the background: the window comes up immediately, with
    # the load tracked in its status panel.
    translator.start()

    run_translator(
        make_process_segment,
        initial_targets,
        model_size=args.model or saved["model"],
        capture_mic=saved["mic"] and not args.no_mic,
        discord_bridge_port=args.discord_bridge_port or saved["bridge_port"],
        show_settings=show_settings,
        initial_position=args.position or saved["position"],
    )


if __name__ == "__main__":
    main()
