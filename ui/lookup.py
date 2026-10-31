"""
The word lookup popup, shared by the overlays, the transcript and the
Vocabulary page. Hover a Chinese word - a whole word, set phrase or chengyu,
as the dictionary splits the line (dictionary.segment) - for its pinyin and
meaning. Click it to pin the lookup, which then:

- drills down into the shorter words and single characters inside it, each
  clickable, with Back to return;
- saves the word into one of your categories (vocab.py), so it's drawn in
  that category's colour wherever it turns up again;
- explains it in the sentence it was heard in, with the local assistant
  (assistant.py) - switching it on first, if it's off, and back off when
  the lookup closes - streamed in as it's written.

On an overlay you can also drag from a word across a longer phrase (in
the Transcript, select it) to look the whole phrase up and explain it.

A hover lookup lets the mouse through - it never takes the hover from the
word under it, which would hide it again; a pinned one is clickable, and
stays until closed - x, Esc, or a click anywhere outside it.
"""

import html
import re
import threading
from dataclasses import dataclass

from PySide6.QtCore import QEvent, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

import assistant
import dictionary
from ui import theme
from ui.widgets import bring_to_front

HOVER_DELAY_MS = 140     # resting on a word this long shows it - not every word the mouse crosses
HIDE_DELAY_MS = 150
WIDTH = 380
HOVER_MAX_HEIGHT = 320
PINNED_MAX_HEIGHT = 600
HOVER_READINGS = 2       # readings shown on hover; pinned shows them all (up to PINNED_READINGS)
PINNED_READINGS = 8
HOVER_DEFINITIONS = 3
HOVER_RESULTS = 3        # Chinese words for an English one: on hover, and pinned
OTHER_RESULTS = 8
HANZI_FAMILIES = {"simplified": ["Microsoft YaHei UI", "Microsoft YaHei"],
                  "traditional": ["Microsoft JhengHei UI", "Microsoft JhengHei"]}


@dataclass
class Context:
    sentence: str          # the line the word was in
    translation: str = ""  # that line in another language - Explain matches a word between the two


def context_for(caption, line):
    """The Context of a word in `line`, one of `caption`'s lines: the line
    itself, and the caption in another language - for a Chinese line, not
    Chinese (English first); for any other, Chinese if there is one (what
    was said, or a translation into it)."""
    sentence = line.removesuffix(" …")
    english = caption.translations.get("en")
    candidates = (([english[0]] if english and english[1] else []) + [caption.native]
                  + [text for text, is_translation in caption.translations.values() if is_translation])
    others = [t for t in candidates if t and t != sentence]
    if dictionary.has_han(sentence):
        translation = next((t for t in others if not dictionary.has_han(t)), "")
    else:
        translation = next((t for t in others if dictionary.has_han(t)), others[0] if others else "")
    return Context(sentence, translation)


_CLASSIFIER = re.compile(r"(?:([^|\[,]+)\|)?([^|\[,]+)\[([^\]]+)\]")


def _definitions(entry):
    """(definitions, measure words): CC-CEDICT's "CL:個|个[ge4]" entries
    pulled out as "个 gè"."""
    defs, measures = [], []
    for d in entry.definitions:
        if d.startswith("CL:"):
            measures += [f"{simp} {dictionary.mark_pinyin(pinyin)}" for _, simp, pinyin in _CLASSIFIER.findall(d[3:])]
        else:
            defs.append(d)
    return defs, measures


def meaning_of(word, limit=3):
    """A short meaning for `word` - its everyday reading's first few
    definitions - for saving it and for prompting Explain."""
    entries = dictionary.lookup(word)
    if entries:
        return "; ".join(_definitions(entries[0])[0][:limit])
    parts = [word[a:b] for a, b in dictionary.segment(word) if dictionary.has_han(word[a:b])]
    return " + ".join(meaning_of(p, 1) or "?" for p in parts) if len(parts) > 1 else ""


def _script(word, entries):
    traditional = any(e.traditional == word and e.simplified != word for e in entries)
    return "traditional" if traditional else "simplified"


def _hanzi_css(px, word, entries):
    return theme.font_css(HANZI_FAMILIES[_script(word, entries)], px, 500)


class _PartRow(QFrame):
    """A word or character inside the one looked up - click to drill in.
    A frame rather than a button: a button doesn't size to wrapped text."""

    def __init__(self, part, on_click):
        super().__init__()
        self.setObjectName("LookupPart")
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f"Look up {part}")
        self._on_click = lambda: on_click(part)
        entries = dictionary.lookup(part)
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 5, 10, 5)
        row.setSpacing(10)
        hanzi = QLabel(part)
        hanzi.setStyleSheet(_hanzi_css(17, part, entries) + " background: transparent;")
        row.addWidget(hanzi, 0, Qt.AlignmentFlag.AlignTop)
        if entries:
            readings = " / ".join(dict.fromkeys(e.pinyin for e in entries[:3]))
            meaning = "; ".join(_definitions(entries[0])[0][:2])
        else:
            readings, meaning = "", "not in the dictionary"
        text = QLabel(f'<span style="color:{theme.ACCENT}">{html.escape(readings)}</span>'
                      f'{"&nbsp;&nbsp;" if readings else ""}'
                      f'<span style="color:{theme.TEXT_DIM}">{html.escape(meaning)}</span>')
        text.setWordWrap(True)
        text.setStyleSheet("background: transparent; padding-top: 3px;")
        row.addWidget(text, 1)
        for child in (hanzi, text):
            child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self._on_click()


class LookupPopup(QWidget):
    _explained = Signal(int, str, str)   # request id, text, "partial" / "done" / "error"

    def __init__(self, controller):
        super().__init__(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.WindowTransparentForInput)
        self.c = controller
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedWidth(WIDTH)
        self.pinned = False
        self.word = None
        self.context = Context("")
        self._anchor = QRect()
        self._history: "list[tuple]" = []   # (word, context) drilled down from, for Back
        self._pending = None                # (word, context, anchor) waiting out the hover delay
        self._explain_id = 0
        self._explaining = None             # (word, sentence) being explained right now
        self._explanations: "dict[tuple, str]" = {}  # (word, sentence) -> finished explanation
        self._explain_error = None          # (word, sentence, message)
        self._partial = ""                  # the explanation streaming in, so far
        self._active = False                # pinned and has been the active window - leaving it closes it
        self._switched_on = False           # Explain switched the local assistant on - closing switches it off
        self._explanation_label = None

        self._hover_timer = QTimer(self, singleShot=True, interval=HOVER_DELAY_MS, timeout=self._show_pending)
        self._hide_timer = QTimer(self, singleShot=True, interval=HIDE_DELAY_MS, timeout=self._hide_hover)
        self._explained.connect(self._on_explained)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.viewport().setAutoFillBackground(False)
        outer.addWidget(self.scroll)
        self.body, self.content = None, None   # rebuilt on every render - see _render

        controller.vocab_changed.connect(self._refresh)
        controller.prefs_changed.connect(self._refresh)  # the local assistant switched on/off

    # -- showing ------------------------------------------------------------------

    def hover(self, word, context, anchor):
        """Looks up `word` beside `anchor` (its rect on screen) once the
        mouse has rested on it a moment - unless a lookup is pinned."""
        if self.pinned:
            return
        self._hide_timer.stop()
        if self.isVisible() and word == self.word and anchor == self._anchor:
            return
        self._pending = (word, context, anchor)
        if self.isVisible():
            self._show_pending()   # already showing one - follow the mouse to the next word straight away
        else:
            self._hover_timer.start()

    def unhover(self):
        """The mouse has left the word - a hover lookup goes."""
        self._hover_timer.stop()
        self._pending = None
        if not self.pinned and self.isVisible():
            self._hide_timer.start()

    def pin(self, word, context, anchor):
        """Looks up `word` and keeps it up - with the characters inside it,
        saving and Explain."""
        self._hover_timer.stop()
        self._hide_timer.stop()
        self._history = []
        self._show(word, context, anchor, pinned=True)

    def close_lookup(self):
        self.pinned = False
        self._active = False
        self._explain_id += 1   # stops any explanation still streaming
        self._explaining = None
        self.hide()
        if self._switched_on:
            # Explain switched the local assistant on - off again, freeing its
            # VRAM, unless the side chat has come to need it meanwhile.
            self._switched_on = False
            chat = getattr(self.c, "chat", None)
            if self.c.assistant_enabled and not (chat is not None and chat.isVisible()):
                self.c.set_assistant_enabled(False)

    def _show_pending(self):
        if self._pending is not None and not self.pinned:
            self._show(*self._pending, pinned=False)
        self._pending = None

    def _hide_hover(self):
        if not self.pinned:
            self.hide()

    def _show(self, word, context, anchor, pinned):
        if pinned != self.pinned or not self.isVisible():
            # Pinned: clickable. Hover: the mouse passes through to the word.
            self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, not pinned)
        self.pinned = pinned
        self.word, self.context, self._anchor = word, context, QRect(anchor)
        self._render()
        self.show()
        self.bring_to_front()
        if pinned:
            # Active, so that clicking anywhere else - the overlay, the game,
            # the desktop - deactivates it, which closes it (changeEvent).
            # The click that pinned it already took the focus from the game.
            self.activateWindow()
            QTimer.singleShot(0, self.bring_to_front)  # again once the click has raised the overlay clicked

    def changeEvent(self, event):  # noqa: N802
        if event.type() == QEvent.Type.ActivationChange:
            if self.isActiveWindow():
                self._active = True
            elif self._active and self.pinned:
                self.close_lookup()
        super().changeEvent(event)

    def bring_to_front(self):
        """Above the overlays - see widgets.bring_to_front."""
        bring_to_front(self)

    def _drill(self, part):
        self._history.append((self.word, self.context))
        if not dictionary.has_han(self.word) and dictionary.has_han(part):
            # From a translation's phrase to a Chinese word for it: that's
            # explained in the Chinese line - if it's there; anywhere else,
            # a small model takes it for a misheard word.
            chinese = self.context.translation
            self.context = Context(chinese if part in chinese else "", self.context.sentence)
        self.word = part
        self._render()

    def _back(self):
        if self._history:
            self.word, self.context = self._history.pop()
            self._render()

    def _refresh(self):
        if self.isVisible():
            self._render()

    # -- content ------------------------------------------------------------------

    def _render(self):
        """Rebuilds the content - into a new widget, swapped in once it's
        filled: widgets added to one already showing stay hidden until the
        next event loop pass, so they'd measure as nothing in _fit."""
        self._explanation_label = None
        self.body = QWidget()
        self.body.setObjectName("LookupBody")
        self.body.setStyleSheet(f"""
            #LookupBody {{ background: transparent; }}
            #LookupPart {{ background: transparent; border: 1px solid transparent; border-radius: 8px; }}
            #LookupPart:hover {{ background: {theme.SURFACE_2}; border-color: {theme.BORDER}; }}""")
        self.content = QVBoxLayout(self.body)
        self.content.setContentsMargins(18, 14, 18, 16)
        self.content.setSpacing(8)
        self._fill()
        self._fit()
        old = self.scroll.takeWidget()
        if old is not None:
            old.deleteLater()  # later: this may be one of its buttons' click
        self.scroll.setWidget(self.body)

    def _fill(self):
        word = self.word or ""
        entries = dictionary.lookup(word)

        if self.pinned:
            top = QHBoxLayout()
            if self._history:
                back = self._small_button(f"←  {self._history[-1][0]}", self._back)
                back.setToolTip("Back")
                top.addWidget(back)
            top.addStretch(1)
            top.addWidget(self._small_button("✕", self.close_lookup, "Close (Esc)"))
            self.content.addLayout(top)

        if not dictionary.has_han(word):
            self._fill_other(word)
            return
        self.content.addLayout(self._headline(word, entries))

        if not dictionary.ready():
            self.content.addWidget(_text(dictionary.failed() and f"Dictionary unavailable - {dictionary.failed()}"
                                         or "Loading the dictionary...", theme.TEXT_DIM))
        elif not entries:
            guess = dictionary.pinyin_of(word)
            if guess:
                self.content.addWidget(_text(guess, theme.ACCENT, 15))
            meaning = meaning_of(word)
            self.content.addWidget(_text(f"Not in the dictionary as a whole{f' - {meaning}' if meaning else ''}",
                                         theme.TEXT_DIM))
        else:
            limit = PINNED_READINGS if self.pinned else HOVER_READINGS
            for entry in entries[:limit]:
                self._reading(entry)
            if len(entries) > limit:
                self.content.addWidget(_text(f"+{len(entries) - limit} more readings - click to see them all"
                                             if not self.pinned else f"+{len(entries) - limit} more readings",
                                             theme.TEXT_FAINT, 12))

        if not self.pinned:
            hint = ("Click for the characters inside, to save it" + (" or explain it" if assistant.installed() else "")
                    + " - or drag across a longer phrase")
            self.content.addWidget(_text(hint, theme.TEXT_FAINT, 12))
        else:
            self._save_row(word, entries)
            if assistant.installed() and self.context.sentence:
                self._explain_section(word)
            parts = dictionary.inner_words(word) + list(dict.fromkeys(ch for ch in word if dictionary.is_han(ch)))
            if len(word) > 1 and parts:
                self.content.addSpacing(4)
                self.content.addWidget(_section("Inside this word" if entries else "Words and characters"))
                for part in parts:
                    self.content.addWidget(_PartRow(part, self._drill))

    def _fill_other(self, word):
        """A word or phrase not in Chinese - a translation's, usually: the
        Chinese words for it (English only - the dictionary's other side),
        and Explain, which finds it in the original line."""
        headline = QLabel(word)
        headline.setWordWrap(True)
        headline.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        headline.setStyleSheet(theme.font_css(theme.UI_FAMILIES, 22, 500))
        self.content.addWidget(headline)
        found = (dictionary.search_english(word, OTHER_RESULTS if self.pinned else HOVER_RESULTS)
                 if dictionary.ready() else [])
        if not self.pinned:
            for entry in [e for i, e in enumerate(found) if e.simplified not in {f.simplified for f in found[:i]}]:
                self.content.addWidget(_text(f"{entry.simplified}   {entry.pinyin}   "
                                             f"{'; '.join(_definitions(entry)[0][:2])}", theme.TEXT))
            if not found:
                self.content.addWidget(_text("No Chinese for it in the dictionary" if dictionary.ready()
                                             else "Loading the dictionary...", theme.TEXT_DIM))
            hint = "Click for more" + (" - and to find it in the Chinese" if assistant.installed() else "")
            self.content.addWidget(_text(hint + " - or drag across a longer phrase", theme.TEXT_FAINT, 12))
            return
        if assistant.installed() and self.context.sentence:
            self._explain_section(word)
        self.content.addSpacing(4)
        self.content.addWidget(_section("Chinese for it"))
        for chinese in dict.fromkeys(e.simplified for e in found):  # once each - 卡 is two readings
            self.content.addWidget(_PartRow(chinese, self._drill))
        if not found:
            self.content.addWidget(_text("Nothing in the dictionary - it only has English the other way."
                                         if dictionary.ready() else "Loading the dictionary...", theme.TEXT_DIM))

    def _headline(self, word, entries):
        row = QHBoxLayout()
        row.setSpacing(10)
        hanzi = QLabel(word)
        hanzi.setStyleSheet(_hanzi_css(26, word, entries))
        hanzi.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row.addWidget(hanzi, 0, Qt.AlignmentFlag.AlignVCenter)
        other = next((f for e in entries for f in (e.simplified, e.traditional) if f != word), None)
        if other:
            alt = QLabel(other)
            alt.setStyleSheet(_hanzi_css(16, other, entries) + f" color: {theme.TEXT_FAINT};")
            alt.setToolTip("Simplified" if _script(word, entries) == "traditional" else "Traditional")
            row.addWidget(alt, 0, Qt.AlignmentFlag.AlignVCenter)
        if any(e.idiom for e in entries):
            row.addWidget(_badge("chengyu", theme.WARNING), 0, Qt.AlignmentFlag.AlignVCenter)
        saved = self.c.vocab.get(word)
        category = self.c.vocab.category(saved["category"]) if saved else None
        if category:
            row.addWidget(_badge(category["name"], category["color"]), 0, Qt.AlignmentFlag.AlignVCenter)
        row.addStretch(1)
        return row

    def _reading(self, entry):
        defs, measures = _definitions(entry)
        if not self.pinned:
            defs = defs[:HOVER_DEFINITIONS]
        box = QVBoxLayout()
        box.setSpacing(1)
        box.addWidget(_text(entry.pinyin, theme.ACCENT, 15))
        box.addWidget(_text("; ".join(defs) or "-", theme.TEXT))
        if measures and self.pinned:
            box.addWidget(_text("Measure word: " + ", ".join(measures), theme.TEXT_FAINT, 12))
        self.content.addLayout(box)

    def _save_row(self, word, entries):
        saved = self.c.vocab.get(word)
        self.content.addSpacing(2)
        self.content.addWidget(_section("Save to"))
        row = QHBoxLayout()
        row.setSpacing(6)
        for category in self.c.vocab.categories:
            current = saved is not None and saved["category"] == category["id"]
            btn = QPushButton(category["name"])
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            color = category["color"]
            btn.setStyleSheet(f"""
                QPushButton {{ background: {theme.rgba(color, 0.22 if current else 0.08)}; color: {color};
                               border: 1px solid {theme.rgba(color, 0.9 if current else 0.35)}; border-radius: 13px;
                               padding: 4px 11px; font-weight: {600 if current else 500}; }}
                QPushButton:hover {{ background: {theme.rgba(color, 0.28)}; }}""")
            btn.setToolTip(f"Saved in {category['name']}" if current else f"Save to {category['name']}")
            btn.clicked.connect(lambda _, cid=category["id"]: self._save(word, entries, cid))
            row.addWidget(btn)
        row.addStretch(1)
        if saved is not None:
            row.addWidget(self._small_button("Remove", lambda: self.c.vocab.remove(word), "Remove from your words"))
        self.content.addWidget(_wrap(row))

    def _save(self, word, entries, category_id):
        entry = entries[0] if entries else None
        self.c.vocab.save(entry.simplified if entry else word, entry.traditional if entry else word,
                          entry.pinyin if entry else dictionary.pinyin_of(word), meaning_of(word), category_id,
                          self.context.sentence)

    def _explain_section(self, word):
        key = (word, self.context.sentence)
        self.content.addSpacing(2)
        head = QHBoxLayout()
        head.addWidget(_section("Explanation"))
        head.addStretch(1)
        done = key in self._explanations
        if not done and self._explaining != key:
            # a translation's phrase: Explain finds its Chinese in the other line
            find = not dictionary.has_han(word) and dictionary.has_han(self.context.translation)
            head.addWidget(self._small_button("Find it in the Chinese" if find else "Explain in this sentence",
                                              lambda: self._explain(word),
                                              "Ask the local assistant" if self.c.assistant_enabled else
                                              "Switches the local assistant on (~3 GB of GPU memory) while "
                                              "this lookup's open, and asks it"))
        self.content.addLayout(head)
        if done or self._explaining == key:
            waiting = "Thinking..." if assistant.ready() else "Loading the local assistant - a few seconds..."
            self._explanation_label = _text(self._explanations.get(key) or self._partial or waiting, theme.TEXT,
                                            markdown=True)
            self.content.addWidget(self._explanation_label)
        elif self._explain_error and self._explain_error[:2] == key:
            self.content.addWidget(_text(self._explain_error[2], theme.DANGER, 12))
        else:
            sentence = self.context.sentence
            self.content.addWidget(_text(f"In “{sentence if len(sentence) <= 40 else sentence[:40] + '…'}”",
                                         theme.TEXT_FAINT, 12))

    def _explain(self, word):
        self._explain_id += 1
        request = self._explain_id
        key = (word, self.context.sentence)
        self._explaining = key
        self._explain_error = None
        self._partial = ""
        context = self.context
        meanings = meaning_of(word, 4)
        if not self.c.assistant_enabled:
            self.c.set_assistant_enabled(True)  # before the request, which waits for it to load
            self._switched_on = True            # and off again when the lookup closes

        def work():
            try:
                text = assistant.explain(word, context.sentence, context.translation, meanings,
                                         lambda so_far: self._explained.emit(request, so_far, "partial"),
                                         cancelled=lambda: request != self._explain_id)
                self._explained.emit(request, text, "done")
            except RuntimeError as e:
                self._explained.emit(request, str(e), "error")

        threading.Thread(target=work, daemon=True, name="explain").start()
        self._render()

    def _on_explained(self, request, text, state):
        if request != self._explain_id or self._explaining is None:
            return
        key = self._explaining
        if state == "partial":
            self._partial = text  # kept for a redraw meanwhile (the model coming on redraws the popup)
            if self._explanation_label is not None and (self.word, self.context.sentence) == key:
                self._explanation_label.setText(text)
                self._fit()
            return
        self._explaining = None
        if state == "done" and text:
            self._explanations[key] = dictionary.with_bold_pinyin(text)
        else:
            self._explain_error = (*key, text or "No explanation came back.")
        self._refresh()

    def _small_button(self, text, on_click, tip=None):
        btn = QPushButton(text)
        btn.setProperty("kind", "ghost")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        if tip:
            btn.setToolTip(tip)
        btn.clicked.connect(on_click)
        return btn

    # -- size and place -------------------------------------------------------------

    def _fit(self):
        """Sizes the popup to its content (scrolling past the cap) and
        places it by the word: below it, or above if there's no room."""
        width = WIDTH - 2
        self.body.setFixedWidth(width)
        wanted = self.content.totalHeightForWidth(width)
        cap = PINNED_MAX_HEIGHT if self.pinned else HOVER_MAX_HEIGHT
        self.setFixedHeight(min(wanted, cap) + 2)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded if wanted > cap
                                               else Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.body.setFixedWidth(width - (10 if wanted > cap else 0))
        screen = QGuiApplication.screenAt(self._anchor.center()) or QGuiApplication.primaryScreen()
        area = screen.availableGeometry()
        x = min(max(self._anchor.left() - 14, area.left() + 8), area.right() - self.width() - 8)
        below = self._anchor.bottom() + 10
        if below + self.height() <= area.bottom() - 8:
            y = below
        else:
            y = max(area.top() + 8, self._anchor.top() - 10 - self.height())
        self.move(x, y)

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor(255, 255, 255, 34), 1))
        p.setBrush(QColor(theme.SURFACE))
        p.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 12, 12)

    def keyPressEvent(self, event):  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.close_lookup()
        elif event.key() == Qt.Key.Key_Backspace and self._history:
            self._back()
        else:
            super().keyPressEvent(event)


def _text(text, color, px=13, markdown=False):
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setTextFormat(Qt.TextFormat.MarkdownText if markdown else Qt.TextFormat.PlainText)
    lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    lbl.setStyleSheet(f"color: {color}; font-size: {px}px; background: transparent;")
    return lbl


def _section(title):
    lbl = QLabel(title.upper())
    lbl.setObjectName("SectionLabel")
    return lbl


def _badge(text, color):
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color: {color}; background: {theme.rgba(color, 0.14)}; border: 1px solid "
                      f"{theme.rgba(color, 0.45)}; border-radius: 9px; padding: 1px 8px; font-size: 11px; "
                      "font-weight: 600;")
    return lbl


def _wrap(layout):
    """A layout as a widget - so a row of buttons can sit in the content."""
    box = QWidget()
    box.setStyleSheet("background: transparent;")
    layout.setContentsMargins(0, 0, 0, 0)
    box.setLayout(layout)
    return box
