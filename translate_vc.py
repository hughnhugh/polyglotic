"""
Live translator for Discord (or any system audio).
Captures whatever your PC is currently playing (WASAPI loopback) plus your
mic, cuts it into speech segments (Silero VAD, or Discord's own speaking
events through the bridge plugin), transcribes them (Qwen3-ASR or Whisper)
and translates each into every chosen target language with Hy-MT2, a local
translation LLM (see translator.py).

Captions appear in a subtitle overlay per target language, the main
window's transcript, and a session log (see ui/ and captions.py).

Usage:
    python translate_vc.py                 # as last left (settings.json)
    python translate_vc.py --target en,vi

A single process handles capture/transcription once per speech segment
regardless of how many targets are chosen - only translation is repeated
per target.
"""

import argparse
import re
import time

from opencc import OpenCC

import discord_bridge
import settings
import translator
from lang_codes import TARGET_LANGUAGES
from overlay_core import capture_options, is_reliable_transcription, live_options, speech_context, speech_options

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


def make_process_segment(targets, out_queue):
    """process_segment for the pipeline: one transcription per speech
    segment, translated into every target in `targets` (the live list the
    UI edits) and put on `out_queue` as a single caption record for
    captions.CaptionBoard."""

    def process_segment(model, segment, source_tag="", speaker_hint=None, caption_group=None, final=False,
                        absorb=(), speech_end=None, live_section=0):
        """caption_group/final: a caption's live updates (final=False) share
        a group, and its final caption (final=True) replaces them - see
        overlay_core.LIVE_MIN_AUDIO_SEC. A final also replaces the captions
        of the groups in absorb - a speaker's back-to-back speech joined
        into one caption (see overlay_core.BRIDGE_JOIN_GAP_SEC). segment
        None with a group just removes that group's live caption. If the
        final pass yields nothing usable the earlier captions stay.
        speech_end: when the speech ended, for the end-to-end delay readout.
        live_section: which section of its growing live caption a live
        update is (see overlay_core.LIVE_SECTION_SEC)."""
        if caption_group and final and segment is None:
            out_queue.put({"group": caption_group, "final": True, "native": None})
            return
        if model is None or capture_options["paused"]:
            # Submitted while a speech-model swap had the old one unloaded
            # (see ModelHolder.request_reload), or before a pause.
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
                # Console-only - loopback speech getting cut by VAD/the
                # Discord bridge but then discarded here would otherwise be
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

        if info.language == "zh":
            native_text = _t2s.convert(native_text)  # normalize display text to Simplified
        elif info.language == "yue":
            native_text = _cantonese_script(native_text)  # Cantonese reads in Traditional

        translations = {}  # target -> (text to show, is it a translation)
        for target_lang in list(targets):
            if info.language == "zh" and target_lang in CHINESE_TARGETS:
                # Mandarin into Chinese: only the script, if anything, changes
                translations[target_lang] = (CHINESE_TARGETS[target_lang].convert(native_text), False)
            elif info.language == target_lang or (caption_group and not final and not live_options["translate"]):
                translations[target_lang] = (native_text, False)
            else:
                # Every target, English included, goes through the
                # translator - it replaced Whisper's own X->English
                # translate task too, which occasionally invented content
                # (see translator.py).
                translated = translator.translate(native_text, info.language, target_lang)
                if target_lang in CHINESE_TARGETS:
                    translated = CHINESE_TARGETS[target_lang].convert(translated)
                elif target_lang == "yue":
                    translated = _cantonese_script(translated)
                # The translation isn't reliability-checked like the
                # transcription above - it can come back empty (translator
                # still loading, or a failed request) or, rarely, as a
                # degenerate loop. Show the original either way rather than
                # dropping the caption or passing that off as a translation.
                if _is_degenerate(translated):
                    translated = ""
                translations[target_lang] = (translated, True) if translated else (native_text, False)

        out_queue.put({
            # Ground truth from the Discord speaking bridge - the only source
            # of speaker labels this app has; without it, captions are
            # unlabeled rather than guessed.
            "speaker": "You" if source_tag == "mic" else (speaker_hint[1] if speaker_hint else ""),
            "you": source_tag == "mic",
            "lang": info.language,
            "native": native_text,
            "translations": translations,
            "group": caption_group,
            "final": final,
            "absorb": list(absorb),
            "live_section": live_section,
            "speech_end": speech_end,
            "proc": time.time() - t0,
        })

    return process_segment


def main():
    saved = settings.load()
    parser = argparse.ArgumentParser(
        description="Live translator for Discord/system audio. Options left out come from the last run "
                    "(settings.json).")
    parser.add_argument(
        "--target", "-t", default=None,
        help="Comma-separated target language code(s), e.g. 'en,vi' - one subtitle overlay each. "
             f"Supported: {', '.join(sorted(TARGET_LANGUAGES))}",
    )
    parser.add_argument(
        "--model", "-m", default=None,
        help="Speech model: qwen3-asr-1.7b / qwen3-asr-0.6b, or a Whisper size "
             "(default: qwen3-asr-1.7b if installed, else large-v3)",
    )
    parser.add_argument("--no-mic", action="store_true", help="Don't also capture your microphone")
    parser.add_argument(
        "--discord-bridge-port", type=int, default=None,
        help=f"Port the optional Discord speaking-bridge WebSocket server listens on (default: {discord_bridge.DEFAULT_PORT})",
    )
    args = parser.parse_args()

    target_arg = args.target or ",".join(saved["targets"]) or "en"
    targets = list(dict.fromkeys(t.strip().lower() for t in target_arg.split(",") if t.strip()))
    for target in targets:
        if target not in TARGET_LANGUAGES:
            raise SystemExit(f"Unsupported target language '{target}'. Supported: {', '.join(sorted(TARGET_LANGUAGES))}")

    speech_options["language"] = saved["language"]
    speech_options["context"] = saved["context"]
    speech_options["auto_names"] = saved["auto_names"]
    live_options["interval"] = saved["live_interval"]
    live_options["translate"] = saved["translate_live"]
    capture_options["caption_mic"] = saved["caption_mic"]

    # Loads in the background; the main window's status tracks it.
    translator.start()

    from ui.app import run  # local import - Qt is only needed for the app itself
    run(
        make_process_segment,
        targets,
        model_size=args.model or saved["model"],
        capture_mic=saved["mic"] and not args.no_mic,
        bridge_port=args.discord_bridge_port or saved["bridge_port"],
    )


if __name__ == "__main__":
    main()
