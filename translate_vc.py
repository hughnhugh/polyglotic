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
import difflib
import os
import re
import sys
import time
import unicodedata
from collections import OrderedDict, deque

if sys.stdout is None:  # started without a console (pythonw - the Start Menu shortcut, start.bat)
    _logs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(_logs, exist_ok=True)
    # what it prints goes here instead - and libraries that write to stderr (torch.hub) don't crash
    sys.stdout = sys.stderr = open(os.path.join(_logs, "app.log"), "w", encoding="utf-8", buffering=1)

from opencc import OpenCC  # noqa: E402 - after the console check above

import aligner
import clauses
import dictionary
import discord_bridge
import vi_dictionary
import settings
import translator
from captions import join_text
from lang_codes import TARGET_LANGUAGES
from overlay_core import (VAD_SAMPLE_RATE, capture_options, is_reliable_transcription, link_options, live_options,
                          speech_context, speech_options)

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


_LATIN_WORD_RE = re.compile(r"[A-Za-z\u00c0-\u024f]{3,}")
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]")
CJK_LANGUAGES = {"zh", "zh-hant", "yue", "ja"}


def _spoken_languages(info, text):
    """Every language heard in a segment, main one first: the speech
    model's own report (Qwen3-ASR lists mixed speech, e.g. Chinese and
    English), plus what the text's scripts give away - Latin words in a
    Chinese line, Chinese characters in an English one - for models that
    only name one."""
    languages = list(getattr(info, "languages", None) or [info.language])
    if info.language in CJK_LANGUAGES and _LATIN_WORD_RE.search(text) and not any(
            lang not in CJK_LANGUAGES for lang in languages):
        languages.append("en")
    if info.language not in CJK_LANGUAGES and _CJK_RE.search(text) and not any(
            lang in CJK_LANGUAGES for lang in languages):
        languages.append("zh")
    return list(dict.fromkeys(languages))


_ENGLISH_WORD_RE = re.compile(r"[A-Za-z']{2,}")
# Combining marks Vietnamese spelling can't do without - the five tones plus
# the breve, circumflex and horn (ă â ê ô ơ ư) - and đ. English has none.
_VIETNAMESE_MARKS = {"̀", "́", "̃", "̉", "̣", "̆", "̂", "̛"}


def _looks_vietnamese(text):
    return any(c in "đĐ" or any(m in _VIETNAMESE_MARKS for m in unicodedata.normalize("NFD", c)) for c in text)


def _heard_language(forced, info, text):
    """The language a segment was really in. With the conversation
    language pinned the speech model labels every line with it, English
    included - which then gets "translated" English to English (a free
    paraphrase) and never into an overlay's second language. A pinned
    line whose text plainly isn't in that language is English."""
    if not forced or info.language != forced:
        return info.language
    english = len(_ENGLISH_WORD_RE.findall(text)) >= 2
    if forced == "vi" and english and not _looks_vietnamese(text):
        return "en"
    if forced in CJK_LANGUAGES and english and not _CJK_RE.search(text):
        return "en"
    return forced


ECHO_WINDOW_SEC = 8      # how far apart a "You" caption and someone else's can be and still be one echo
ECHO_SIMILARITY = 0.6


def _comparable(text):
    return "".join(c.lower() for c in text if c.isalnum())


def _same_words(a, b):
    """True if two captions are (nearly) the same speech - the mic echoing
    what Discord played."""
    a, b = _comparable(a), _comparable(b)
    # Short replies ("yeah", "好的") are said by everyone - only lines long
    # enough to be distinctive count as an echo.
    minimum = 4 if _CJK_RE.search(a + b) else 8
    if len(a) < minimum or len(b) < minimum:
        return False
    shorter, longer = sorted((a, b), key=len)
    return shorter in longer or difflib.SequenceMatcher(None, a, b).ratio() >= ECHO_SIMILARITY


# A caption is translated a clause at a time (see clauses.py and
# docs/line-by-line-captions.md): each clause with this many of the
# caption's clauses before it as context (earlier turns of the
# conversation - translator.translate's history), and each translation
# remembered by its clause's text, so a clause that comes out the same in
# the next live update - or the final pass - keeps its translation, and its
# row on the overlay doesn't move.
CLAUSE_HISTORY = 3
LINK_CACHE_SIZE = 4000        # clause/translation pairs whose word links are remembered
CLAUSE_CACHE_SIZE = 4000      # translations remembered (a long call says a lot of clauses)
LIVE_GROUPS_KEPT = 64         # captions whose live clauses are kept, as the next section's context


def _skip_pattern(words):
    """A regex matching text made only of `words` (comma-separated), each
    any number of times and with its last letter drawn out ("嗯嗯", "hmmm")
    - or None if there are none."""
    entries = sorted({_comparable(w) for w in words.split(",")} - {""}, key=len, reverse=True)
    if not entries:
        return None
    return re.compile("(?:" + "|".join(re.escape(e) + "+" for e in entries) + ")+")


_skip_patterns = {}


def _only_skip_words(text, words):
    """True if `text` is nothing but words from `words` - "嗯。", "嗯，嗯。",
    "Mm-hmm." - so it isn't worth a caption. Punctuation, spacing and case
    don't count; anything else in it (嗯，我觉得…) keeps the caption."""
    if words not in _skip_patterns:
        _skip_patterns.clear()
        _skip_patterns[words] = _skip_pattern(words)
    pattern, comparable = _skip_patterns[words], _comparable(text)
    return bool(pattern and comparable and pattern.fullmatch(comparable))


# Two Discord speakers flagged over the same speech - one's mic picking up
# the other, or speaking indicators that lag - get the same mixed audio cut
# twice, and the same caption twice. (discord_bridge.OVERLAP_SKIP_FRAC
# already skips an interval mostly covered by another's; this catches what
# gets past it, short replies especially.) Captions overlapping by this much
# of the shorter one's time, with the same words, are one: the speaker
# who's done more of the talking in the last DOMINANCE_WINDOW_SEC keeps it -
# or, within DOMINANCE_MARGIN_SEC of even, the one captioned last.
DUPLICATE_OVERLAP = 0.5
DOMINANCE_WINDOW_SEC = 30
DOMINANCE_MARGIN_SEC = 1.0


def _same_speech(a, b):
    """True if two captions of overlapping audio are the same speech: the
    same words, short ones included - two people don't say the same thing
    at the same moment, but one person's speech can be cut twice. Two cuts
    of the same speech mostly overlap, so a line much shorter than the
    other, inside it, isn't one: that's more likely someone echoing the
    end of what was said ("都是一模一样的" after "那个 background 都是一模一样的")."""
    a, b = _comparable(a), _comparable(b)
    if not a or not b:
        return False
    shorter, longer = sorted((a, b), key=len)
    return a == b or (len(shorter) >= len(longer) / 2 and _same_words(a, b))


def make_process_segment(targets, out_queue):
    """process_segment for the pipeline: one transcription per speech
    segment, translated into every target in `targets` (the live list the
    UI edits) and put on `out_queue` as a single caption record for
    captions.CaptionBoard."""

    # A backup to overlay_core.is_bleed, which catches mic echoes before
    # transcription: recent captions, to drop a "You" caption repeating
    # what someone else just said (or take it back when theirs arrives).
    recent_others = deque()   # (time, text)
    recent_mine = deque()     # (time, text, caption group)
    # caption group -> {"speaker" (user id), "name", "text", "start", "end", "at"}: Discord speakers'
    # recent captions, live and final - to tell one person's speech cut twice (see DUPLICATE_OVERLAP)
    others_recent: "OrderedDict[str, dict]" = OrderedDict()
    lost_to: "OrderedDict[str, str]" = OrderedDict()  # caption group -> the one it lost a duplicate to - that sticks
    clause_cache = OrderedDict()   # (clause, source language, target) -> translation, oldest first
    link_cache = OrderedDict()     # (clause, translation) -> its word links, oldest first
    # caption group -> live section -> [(clause, {target: (text, is_translation)})]: what a live
    # caption's earlier sections said, as context for its next one
    live_clauses: "dict[str, dict[int, list]]" = {}

    def talking(speaker, now, exclude):
        """Seconds of `speaker`'s captioned speech in the last DOMINANCE_WINDOW_SEC."""
        return sum(e["end"] - e["start"] for g, e in others_recent.items()
                   if e["speaker"] == speaker and g not in exclude and now - e["at"] <= DOMINANCE_WINDOW_SEC)

    def keeps_caption(group, speaker_hint, text, span, final, now):
        """False if this caption repeats another speaker's at the same
        moment and they keep it - else True, taking back theirs if it's
        this speaker who keeps it. A live update never takes back a
        finished caption (its own final can), and once one caption has
        lost to another, it stays lost - the decision doesn't flip with
        the next update."""
        for old in [g for g, e in others_recent.items() if now - e["at"] > DOMINANCE_WINDOW_SEC]:
            del others_recent[old]
        speaker, name = speaker_hint
        for other_group, other in list(others_recent.items()):
            if other_group == group or other["speaker"] == speaker or not _same_speech(text, other["text"]):
                continue
            overlap = min(span[1], other["end"]) - max(span[0], other["start"])
            if overlap < DUPLICATE_OVERLAP * min(span[1] - span[0], other["end"] - other["start"]):
                continue
            exclude = {group, other_group}
            mine, theirs = talking(speaker, now, exclude), talking(other["speaker"], now, exclude)
            if lost_to.get(group) == other_group or theirs > mine + DOMINANCE_MARGIN_SEC:
                print(f"[debug] dropped {name}'s {text!r} - the same speech as {other['name']}'s, "
                      f"who's done more of the talking ({theirs:.1f}s to {mine:.1f}s)")
                lost_to[group] = other_group
                if others_recent.pop(group, None) is not None:  # its live caption's showing - take it back
                    out_queue.put({"group": group, "final": True, "native": None,
                                   "reason": f"the same speech as {other['name']}'s"})
                return False
            if other["final"] and not final:
                continue  # theirs is finished - only this one's final may take it back
            print(f"[debug] took back {other['name']}'s {other['text']!r} - the same speech as {name}'s "
                  f"({mine:.1f}s of talking to {theirs:.1f}s)")
            lost_to[other_group] = group
            del others_recent[other_group]
            out_queue.put({"group": other_group, "final": True, "native": None,
                           "reason": f"the same speech as {name}'s"})
        while len(lost_to) > 200:
            lost_to.popitem(last=False)
        others_recent[group] = {"speaker": speaker, "name": name, "text": text, "start": span[0], "end": span[1],
                                "final": final, "at": now}
        return True

    def link_clause(text, translation):
        """Which words of `text` and `translation` say the same thing
        (aligner.links) - remembered, so a clause and translation that
        come out the same aren't aligned again."""
        key = (text, translation)
        if key in link_cache:
            link_cache.move_to_end(key)
            return link_cache[key]
        try:
            found = aligner.links(text, translation)
        except Exception as e:  # one bad pair mustn't cost the caption
            print(f"[aligner] {type(e).__name__}: {e}")
            found = []
        link_cache[key] = found
        if len(link_cache) > LINK_CACHE_SIZE:
            link_cache.popitem(last=False)
        return found

    def translate_clause(text, src_lang, target, history):
        """`text` (one clause) in `target` - remembered, or translated with
        `history` ([(clause, translation)] before it) as context. "" if the
        translator's unavailable or came back with nothing usable."""
        key = (text, src_lang, target)
        if key in clause_cache:
            clause_cache.move_to_end(key)
            return clause_cache[key]
        translated = translator.translate(text, src_lang, target, history[-CLAUSE_HISTORY:])
        if target in CHINESE_TARGETS:
            translated = CHINESE_TARGETS[target].convert(translated)
        elif target == "yue":
            translated = _cantonese_script(translated)
        # The translation isn't reliability-checked like the transcription
        # - it can come back empty (translator still loading, or a failed
        # request) or, rarely, as a degenerate loop. Shown as the original
        # either way rather than passed off as a translation.
        if _is_degenerate(translated):
            translated = ""
        if translated:
            clause_cache[key] = translated
            if len(clause_cache) > CLAUSE_CACHE_SIZE:
                clause_cache.popitem(last=False)
        return translated

    def is_echo_of_others(text, now):
        while recent_others and now - recent_others[0][0] > ECHO_WINDOW_SEC:
            recent_others.popleft()
        return any(_same_words(text, other) for _, other in recent_others)

    def take_back_my_echoes(text, now):
        while recent_mine and now - recent_mine[0][0] > ECHO_WINDOW_SEC:
            recent_mine.popleft()
        for entry in [e for e in recent_mine if _same_words(text, e[1])]:
            recent_mine.remove(entry)
            print(f"[debug] took back \"You\" caption {entry[1]!r} - it was someone else's speech in the mic")
            out_queue.put({"group": entry[2], "final": True, "native": None,
                           "reason": "someone else's speech in the mic"})

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
            others_recent.pop(caption_group, None)
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
        if _only_skip_words(native_text, speech_options["skip_words"]):
            # Just a "嗯" - not captioned (see settings' skip_words). One that
            # was showing live, before it turned out to be only that, goes.
            print(f"[debug] skipped {native_text!r} - only words not captioned on their own")
            if caption_group and final:
                out_queue.put({"group": caption_group, "final": True, "native": None,
                               "reason": "only words not captioned on their own"})
            return
        if options["initial_prompt"] and _echoes_context(native_text, options["initial_prompt"]):
            print(f"[debug] dropped a transcription echoing the context words (speaker_hint={speaker_hint}): "
                  f"{native_text!r}")
            return

        heard = _heard_language(options["language"], info, native_text)
        if heard != info.language:
            info.language, info.languages = heard, [heard]

        now = time.time()
        if source_tag == "mic":
            if is_echo_of_others(native_text, now):
                print(f"[debug] dropped \"You\" caption {native_text!r} - someone else's speech in the mic")
                if caption_group and final:
                    out_queue.put({"group": caption_group, "final": True, "native": None,
                                   "reason": "someone else's speech in the mic"})
                return
            if final and caption_group:
                recent_mine.append((now, native_text, caption_group))
        else:
            recent_others.append((now, native_text))
            take_back_my_echoes(native_text, now)

        if info.language == "zh":
            native_text = _t2s.convert(native_text)  # normalize display text to Simplified
        elif info.language == "yue":
            native_text = _cantonese_script(native_text)  # Cantonese reads in Traditional

        if source_tag != "mic" and speaker_hint and caption_group:
            for group in absorb:  # joined into this caption - no longer shown on their own
                others_recent.pop(group, None)
            span_end = speech_end if speech_end is not None else now
            span = (span_end - len(segment) / VAD_SAMPLE_RATE, span_end)
            if not keeps_caption(caption_group, speaker_hint, native_text, span, final, now):
                if final:  # its live caption, if it had one, goes too
                    out_queue.put({"group": caption_group, "final": True, "native": None,
                                   "reason": "the same speech as another speaker's"})
                return

        spoken = _spoken_languages(info, native_text)
        mixed = len(spoken) > 1
        parts = clauses.split(native_text)
        # A live caption's earlier sections, as context for this one's first clauses
        sections = live_clauses.get(caption_group, {}) if caption_group and not final else {}
        before = [clause for s in sorted(sections) if s < live_section for clause in sections[s]]
        per_clause = [{} for _ in parts]  # clause -> {target: (text to show, is it a translation)}
        for target_lang in list(targets):
            if caption_group and not final and not live_options["translate"]:
                shown = [(part, False) for part in parts]
            elif info.language == "zh" and target_lang in CHINESE_TARGETS and not mixed:
                # Mandarin into Chinese: only the script, if anything, changes
                shown = [(CHINESE_TARGETS[target_lang].convert(part), False) for part in parts]
            elif info.language == target_lang and not mixed:
                shown = [(part, False) for part in parts]
            else:
                # A line mixing languages is translated even into its main
                # one - the whole line rewritten in the target language, so
                # the parts already in it stay and the rest ("我觉得这个
                # culture concept 很 specific") is translated in context,
                # rather than left as it was.
                # Every target, English included, goes through the
                # translator - it replaced Whisper's own X->English
                # translate task too, which occasionally invented content
                # (see translator.py).
                history = [(said, versions[target_lang][0]) for said, versions in before
                           if versions.get(target_lang, ("", False))[1]]
                shown = []
                for part in parts:
                    translated = translate_clause(part, info.language, target_lang, history)
                    shown.append((translated, True) if translated else (part, False))
                    if translated:
                        history.append((part, translated))
            for versions, version in zip(per_clause, shown):
                versions[target_lang] = version
        # Word links, for the overlays (docs/word-links.md): each clause's
        # words linked to its translations'. Every clause, a live caption's
        # last too: once its section is done it's no longer last, but its
        # text stays - the overlay leaves out links on the clause still
        # being spoken.
        per_clause_links = [{} for _ in parts]
        if link_options["enabled"] and aligner.loaded():
            for n, (part, versions) in enumerate(zip(parts, per_clause)):
                per_clause_links[n] = {lang: link_clause(part, text) for lang, (text, is_translation) in versions.items()
                                       if is_translation and text}
        translations = {target_lang: (join_text(versions[target_lang][0] for versions in per_clause),
                                      any(versions[target_lang][1] for versions in per_clause))
                        for target_lang in (per_clause[0] if per_clause else {})}
        if caption_group and not final:
            live_clauses.setdefault(caption_group, {})[live_section] = list(zip(parts, per_clause))
            while len(live_clauses) > LIVE_GROUPS_KEPT:
                live_clauses.pop(next(iter(live_clauses)))
        elif final:
            for group in (caption_group, *absorb):
                live_clauses.pop(group, None)

        out_queue.put({
            # Ground truth from the Discord speaking bridge - the only source
            # of speaker labels this app has; without it, captions are
            # unlabeled rather than guessed.
            "speaker": "You" if source_tag == "mic" else (speaker_hint[1] if speaker_hint else ""),
            "you": source_tag == "mic",
            "lang": info.language,
            "langs": spoken,
            "native": native_text,
            "translations": translations,
            "clauses": [{"native": part, "translations": versions, "links": links}
                        for part, versions, links in zip(parts, per_clause, per_clause_links)],
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
        help="Speech model: qwen3-asr-0.6b / qwen3-asr-1.7b, or a Whisper size "
             "(default: the Qwen3-ASR in models/ - 1.7B if it's there, else 0.6B - else large-v3)",
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
    speech_options["skip_words"] = saved["skip_words"]
    live_options["interval"] = saved["live_interval"]
    live_options["translate"] = saved["translate_live"]
    capture_options["caption_mic"] = saved["caption_mic"]

    # Loads in the background; the main window's status tracks it.
    translator.start(saved["translator"])
    dictionary.start()  # the word lookup's - ~1s, also in the background
    vi_dictionary.start()  # Vietnamese lookups' - ~0.1s once its cache is made (~1.5s, the first time)

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
