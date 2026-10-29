"""
The caption model: every caption of the session, kept up to date as the
pipeline's records arrive, and shared by everything that shows captions -
each target language's overlay, the transcript page and the session log.

The pipeline (translate_vc.process_segment) produces one record per
transcribed segment, carrying the original text and its translation into
every target language at once. Records come in three kinds:

- a final caption, either standalone or ending a caption group - replacing
  the group's current caption, and any absorbed groups' (a source's
  back-to-back speech joined into one, see overlay_core.JOIN_GAP_SEC),
  in the earliest one's place;
- a live update of a caption group (final=False), shown in place until the
  final caption replaces it - long live captions grow in sections (see
  overlay_core.LIVE_SECTION_SEC), shown joined;
- a removal (final, native None) - a group's live caption taken back.

Plain strings are notices (model loading, audio device changes...) - logged,
and shown in the main window rather than in overlays.

Pure Python, no GUI - so it's tested directly (test_captions.py).
"""

import itertools
import time
from dataclasses import dataclass, field
from datetime import datetime

MAX_HISTORY = 300  # captions kept; older ones drop off the front


@dataclass
class Caption:
    key: int                  # unique, stable across updates - views use it to reuse widgets
    speaker: str              # who or what it came from - "You" for your mic, "" for system audio
    is_you: bool              # your own mic
    lang: str                 # the speech's (main) language code
    native: str               # what was said, in its own language
    translations: dict        # target -> (text, is_translation): text to show in that target's view
    group: "str | None"
    live: bool                # still being spoken - a live update, replaced later
    delay: "float | None"     # seconds from the end of the speech to this version appearing
    proc: "float | None"      # transcription + translation time alone
    langs: tuple = ()         # every language heard in it, main first - more than one when mixed
    updated_at: float = field(default_factory=time.time)
    time: datetime = field(default_factory=datetime.now)

    def lines_for(self, target, second=None):
        """(said, translation) for `target`'s overlay: the white line - what
        was said, in its own language (script-converted for the target where
        that's all it needs) - and the yellow one under it: its translation
        into `target`, or, when it was already said in the target's
        language, into `second`; None when there's nothing to translate."""
        text, is_translation = self.translations.get(target, (self.native, False))
        if is_translation and text:
            return self.native, text
        said = text or self.native
        if second and second != target:
            text2, is_translation2 = self.translations.get(second, ("", False))
            if is_translation2 and text2:
                return said, text2
        return said, None

    def text_for(self, target):
        """(primary, original): what to show in `target`'s view - the
        translation, with the original underneath; or just the original
        when it's already in that language (original then None)."""
        text, is_translation = self.translations.get(target, (self.native, False))
        return (text, self.native) if is_translation and text else (text or self.native, None)


def join_text(pieces):
    """Joins caption text pieces with a space - or without one between
    Chinese/Japanese characters, which aren't space-separated."""
    out = ""
    for piece in pieces:
        piece = (piece or "").strip()
        if not piece:
            continue
        cjk_join = out and "⺀" <= out[-1] <= "鿿" or "⺀" <= piece[0] <= "鿿"
        out += piece if not out or cjk_join else " " + piece
    return out


class CaptionBoard:
    """The session's captions in display order, oldest first. apply() takes
    the pipeline's records; views read `captions` and `version` (bumped on
    every change, so they know when to redraw)."""

    def __init__(self, log_file=None, max_history=MAX_HISTORY):
        self.captions: "list[Caption]" = []
        self.notices: "list[str]" = []
        self.version = 0
        self._log = log_file
        self._max_history = max_history
        self._keys = itertools.count(1)
        self._live_sections: "dict[str, dict[int, dict]]" = {}  # group -> section -> latest record
        self._live_counts: "dict[str, int]" = {}  # group -> live updates shown, for the log

    def apply(self, record):
        if isinstance(record, str):
            self.notices.append(record)
            self._write_log(record)
            self.version += 1
            return
        group = record.get("group")
        if group and record.get("native") is None:
            self._remove_groups([group])
        elif group and not record.get("final"):
            self._apply_live(group, record)
        else:
            self._apply_final(record)
        del self.captions[:-self._max_history]
        self.version += 1

    # -- record kinds -------------------------------------------------------

    def _apply_live(self, group, record):
        sections = self._live_sections.setdefault(group, {})
        sections[record.get("live_section", 0)] = record
        parts = [sections[n] for n in sorted(sections)]
        targets = {t for p in parts for t in p["translations"]}
        joined = dict(record, native=join_text(p["native"] for p in parts), translations={
            t: (join_text(p["translations"].get(t, (p["native"], False))[0] for p in parts),
                any(p["translations"].get(t, ("", False))[1] for p in parts))
            for t in targets})
        self._live_counts[group] = self._live_counts.get(group, 0) + 1
        self._place(self._caption(joined, live=True), [group])

    def _apply_final(self, record):
        group = record.get("group")
        groups = [g for g in (group, *(record.get("absorb") or ())) if g]
        live_updates = sum(self._live_counts.pop(g, 0) for g in groups)
        for g in groups:
            self._live_sections.pop(g, None)
        caption = self._caption(record, live=False)
        self._place(caption, groups)
        self._write_log(caption, live_updates)

    def _remove_groups(self, groups):
        for g in groups:
            self._live_sections.pop(g, None)
            self._live_counts.pop(g, None)
        self.captions = [c for c in self.captions if c.group not in groups]
        self._write_log(f"[removed {', '.join(groups)} - duplicate of another speaker]")

    # -- helpers --------------------------------------------------------------

    def _caption(self, record, live):
        speech_end = record.get("speech_end")
        return Caption(
            key=next(self._keys), speaker=record.get("speaker", ""), is_you=record.get("you", False),
            lang=record.get("lang", ""), native=record["native"], translations=dict(record.get("translations", {})),
            group=record.get("group"), live=live, langs=tuple(record.get("langs") or ()),
            delay=(time.time() - speech_end) if speech_end is not None else None, proc=record.get("proc"),
        )

    def _place(self, caption, groups):
        """Replace the captions of `groups` with `caption`, in the earliest
        one's place - or append it if there are none. A replaced caption's
        key carries over, so a view updates its widget instead of
        rebuilding it."""
        positions = [i for i, c in enumerate(self.captions) if c.group in groups] if groups else []
        if not positions:
            self.captions.append(caption)
            return
        first = positions[0]
        caption.key = self.captions[first].key
        caption.time = self.captions[first].time
        for i in reversed(positions[1:]):
            del self.captions[i]
        self.captions[first] = caption

    def set_log_file(self, log_file):
        """Starts (a file) or stops (None) writing finals to a session log."""
        if self._log is not None:
            self._log.close()
        self._log = log_file

    def _write_log(self, entry, live_updates=0):
        if self._log is None:
            return
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(entry, str):
            self._log.write(f"[{stamp}] {entry}\n")
        else:
            parts = [entry.speaker or "", f"[{entry.lang}, {entry.delay:.1f}s]" if entry.delay is not None
                     else f"[{entry.lang}]"]
            if entry.group:
                parts.append(f"(final {entry.group})")
            if entry.proc is not None:
                parts.append(f"(proc {entry.proc:.1f}s)")  # transcription + translation alone
            if live_updates:
                parts.append(f"(live x{live_updates})")
            lines = [f"[{stamp}] {' '.join(p for p in parts if p)} {entry.native}"]
            lines += [f"    -> [{target}] {text}" for target, (text, is_translation) in entry.translations.items()
                      if is_translation and text]
            self._log.write("\n".join(lines) + "\n")
        self._log.flush()
