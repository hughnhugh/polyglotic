"""
The main window: a sidebar and six pages.

  Home        - is it working, pause, the everyday toggles, where captions go
  Transcript  - this session's captions, every language side by side
  Languages   - what to translate into, what's being spoken, names & slang
  Captions    - how the overlays look and when they update, with a preview
  Audio       - system audio, microphone, the Discord bridge
  Models      - speech model and translator choice, GPU memory

Closing it keeps the app running in the tray; Quit (sidebar or tray) exits.
"""

import html

from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QTextCursor
from PySide6.QtWidgets import (QApplication, QButtonGroup, QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMainWindow, QProgressBar, QPushButton, QScrollArea,
                               QSlider, QSpinBox, QStackedWidget, QTextBrowser, QVBoxLayout, QWidget)

import settings
import translator
from captions import Caption
from lang_codes import LANGUAGE_NAMES
from overlay_core import live_options, speech_context, speech_options
from qwen_asr_client import MODEL_DIRS, PYTHON as QWEN_PYTHON
from ui import theme
from ui.app import gpu_memory
from ui.overlay import CaptionBlock, spoken_languages_label
from ui.widgets import (Card, Chip, FlowRow, Segmented, SettingRow, StatusDot, ToggleSwitch, app_icon, button, glyph,
                        label)

PINNED_LANGUAGES = ["en", "zh", "zh-hant", "yue", "vi"]

MODEL_CHOICES = [
    ("qwen3-asr-1.7b", "Qwen3-ASR 1.7B", "Most accurate - especially Chinese, Cantonese and Vietnamese. Recommended.",
     "4.7 GB"),
    ("qwen3-asr-0.6b", "Qwen3-ASR 0.6B", "Nearly as good at under half the memory - when a game needs the GPU.",
     "2 GB"),
    ("large-v3", "Whisper large-v3", "OpenAI's model. Weaker on Cantonese, and invents text on noise more often.",
     "4 GB"),
    ("small", "Whisper small", "Lightest option, noticeably less accurate.", "1 GB"),
]

# The translator's quantizations (translator.MODELS): (name, label, what it's for)
TRANSLATOR_CHOICES = [
    ("Q4_K_M", "Hy-MT2-7B Q4_K_M (default)", "The most compressed - the least GPU memory."),
    ("Q8_0", "Hy-MT2-7B Q8_0", "The closest to the uncompressed model, for about 3.4 GB more than Q4_K_M."),
]

FADE_CHOICES = [(None, "Never"), (5, "5s"), (8, "8s"), (15, "15s"), (30, "30s")]
LIVE_CHOICES = [None, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6]


def _status_of(key, text):
    """(dot colour, short description) for a subsystem's raw status line."""
    text = text or ""
    if key == "discord_bridge":
        if text.startswith("Connected"):
            return theme.SUCCESS, "Connected"
        return theme.TEXT_FAINT, "Not connected - captions won't have names"
    if text.startswith("Capturing"):
        return theme.SUCCESS, text.split(": ", 1)[-1]
    if text.startswith("Ready"):
        return theme.SUCCESS, text
    if text.startswith(("Loading", "Reloading")):
        return theme.WARNING, text.rstrip(".") + "..."
    if text.startswith(("Failed", "Not installed")) or "ERROR" in text:
        return theme.DANGER, text
    if text == "Off":
        return theme.TEXT_FAINT, "Off"
    return theme.TEXT_FAINT, text if text not in ("...", "") else "Starting..."


def _qwen_installed(size):
    import os
    return os.path.exists(QWEN_PYTHON) and os.path.isdir(MODEL_DIRS[size])


def _page(title, subtitle):
    """A scrolling page with a title, returning (page, content layout)."""
    page = QScrollArea()
    page.setObjectName("Page")
    page.setWidgetResizable(True)
    page.setFrameShape(QFrame.Shape.NoFrame)
    inner = QWidget()
    inner.setObjectName("Page")
    content = QVBoxLayout(inner)
    content.setContentsMargins(36, 30, 36, 30)
    content.setSpacing(16)
    head = QVBoxLayout()
    head.setSpacing(4)
    head.addWidget(label(title, "PageTitle"))
    if subtitle:
        head.addWidget(label(subtitle, "PageSubtitle", wrap=True))
    content.addLayout(head)
    content.addSpacing(4)
    page.setWidget(inner)
    return page, content


class _StatusLine(QWidget):
    """A dot, a name and what it's doing - one subsystem."""

    def __init__(self, name):
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 3, 0, 3)
        row.setSpacing(10)
        self.dot = StatusDot()
        row.addWidget(self.dot)
        name_lbl = label(name)
        name_lbl.setFixedWidth(118)
        row.addWidget(name_lbl)
        self.detail = label("", "Dim")
        self.detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row.addWidget(self.detail, 1)

    def set(self, color, text):
        self.dot.set_color(color)
        self.detail.setText(text)


# =============================================================================
# Home
# =============================================================================

class HomePage(QWidget):
    def __init__(self, controller, go_to):
        super().__init__()
        self.c = controller
        page, content = _page("Home", None)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(page)

        # -- hero: is it working? ------------------------------------------------
        hero = Card()
        row = QHBoxLayout()
        row.setSpacing(14)
        self.state_dot = StatusDot(size=14)
        row.addWidget(self.state_dot, 0, Qt.AlignmentFlag.AlignVCenter)
        text = QVBoxLayout()
        text.setSpacing(2)
        self.state_title = label("Starting up", None)
        self.state_title.setStyleSheet("font-size: 20px; font-weight: 600;")
        self.state_detail = label("", "Dim", wrap=True)
        text.addWidget(self.state_title)
        text.addWidget(self.state_detail)
        row.addLayout(text, 1)
        self.pause_btn = button("Pause", "primary", lambda: controller.set_paused(not controller.paused))
        self.pause_btn.setToolTip("Ctrl+Alt+P - works while you're in a game")
        self.pause_btn.setMinimumWidth(110)
        row.addWidget(self.pause_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        hero.add(row)
        content.addWidget(hero)

        # -- everyday toggles ------------------------------------------------------
        quick = Card("Quick settings")
        self.mic_toggle = ToggleSwitch(controller.caption_mic)
        self.mic_toggle.toggled.connect(controller.set_caption_mic)
        quick.add(SettingRow("Caption my own speech", "Your microphone, labelled “You”.", self.mic_toggle))
        self.lock_toggle = ToggleSwitch(controller.locked)
        self.lock_toggle.toggled.connect(controller.set_locked)
        quick.add(SettingRow("Lock overlays", "Clicks pass through them to your game. Unlock to move or resize. "
                             "Ctrl+Alt+L", self.lock_toggle))
        self.hide_toggle = ToggleSwitch(not controller.overlays_hidden)
        self.hide_toggle.toggled.connect(lambda on: controller.set_overlays_hidden(not on))
        quick.add(SettingRow("Show overlays", "Ctrl+Alt+H", self.hide_toggle))
        self.original_toggle = ToggleSwitch(controller.prefs["show_original"])
        self.original_toggle.toggled.connect(lambda on: controller.set_pref("show_original", on))
        quick.add(SettingRow("Show what was said above translations", "White is what was said; yellow is "
                             "always a translation.", self.original_toggle))
        content.addWidget(quick)

        # -- where captions go -------------------------------------------------------
        langs = Card("Translating into", "Each language has its own subtitle overlay.")
        self.chips = FlowRow()
        langs.add(self.chips)
        add_row = QHBoxLayout()
        add_row.addWidget(button("Add a language", None, lambda: go_to("Languages")))
        add_row.addStretch(1)
        langs.add(add_row)
        content.addWidget(langs)

        # -- status ----------------------------------------------------------------------
        health = Card("Status")
        self.lines = {key: _StatusLine(name) for key, name in [
            ("whisper_model", "Speech model"), ("translator", "Translator"), ("loopback", "System audio"),
            ("mic", "Microphone"), ("discord_bridge", "Discord")]}
        for line in self.lines.values():
            health.add(line)
        gpu_row = QHBoxLayout()
        gpu_row.setSpacing(10)
        gpu_name = label("GPU memory")
        gpu_name.setFixedWidth(137)
        gpu_row.addWidget(gpu_name)
        self.gpu_bar = QProgressBar()
        self.gpu_bar.setTextVisible(False)
        self.gpu_bar.setFixedHeight(6)
        gpu_row.addWidget(self.gpu_bar, 1)
        self.gpu_text = label("", "Dim")
        self.gpu_text.setMinimumWidth(110)
        gpu_row.addWidget(self.gpu_text)
        health.add(gpu_row)
        self.notice = label("", "Faint", wrap=True)
        health.add(self.notice)
        content.addWidget(health)
        content.addStretch(1)

        controller.state_changed.connect(self.refresh)
        controller.status_changed.connect(lambda _: self.refresh())
        controller.targets_changed.connect(self._refresh_chips)
        controller.prefs_changed.connect(self.refresh)
        controller.notice.connect(lambda text: self.notice.setText(text.strip("[]")))
        QTimer(self, interval=2000, timeout=self._refresh_gpu).start()
        self._refresh_chips()
        self._refresh_gpu()
        self.refresh()

    def _refresh_chips(self):
        chips = []
        for code in self.c.targets:
            chip = Chip(LANGUAGE_NAMES.get(code, code))
            chip.removed.connect(lambda code=code: self.c.remove_target(code))
            chips.append(chip)
        if not chips:
            chips.append(label("No languages - captions are off.", "Dim"))
        self.chips.set_widgets(chips)

    def _refresh_gpu(self):
        mem = gpu_memory()
        if mem is None:
            self.gpu_text.setText("Unavailable")
            return
        used, total = mem
        self.gpu_bar.setRange(0, int(total * 10))
        self.gpu_bar.setValue(int(used * 10))
        self.gpu_text.setText(f"{used:.1f} of {total:.0f} GB")

    def refresh(self):
        c = self.c
        status = c.status
        for key, line in self.lines.items():
            line.set(*_status_of(key, status.get(key)))
        model_ready = str(status.get("whisper_model", "")).startswith("Ready")
        if c.paused:
            self.state_dot.set_color(theme.WARNING)
            self.state_title.setText("Paused")
            self.pause_btn.setText("Resume")
        elif not model_ready:
            self.state_dot.set_color(theme.WARNING)
            self.state_title.setText("Starting up")
            self.pause_btn.setText("Pause")
        else:
            self.state_dot.set_color(theme.SUCCESS)
            self.state_title.setText("Listening")
            self.pause_btn.setText("Pause")
        parts = []
        if c.targets:
            parts.append("Captions in " + ", ".join(LANGUAGE_NAMES.get(t, t) for t in c.targets))
        if str(status.get("discord_bridge", "")).startswith("Connected"):
            people = len(c.discord_people())
            parts.append(f"Discord connected{f' ({people} in call)' if people else ''}")
        if not c.caption_mic:
            parts.append("your speech not captioned")
        if c.overlays_hidden:
            parts.append("overlays hidden")
        self.state_detail.setText(" · ".join(parts) if parts else "")
        for toggle, value in ((self.mic_toggle, c.caption_mic), (self.lock_toggle, c.locked),
                              (self.hide_toggle, not c.overlays_hidden),
                              (self.original_toggle, c.prefs["show_original"])):
            if toggle.isChecked() != value:
                toggle.setChecked(value)


# =============================================================================
# Transcript
# =============================================================================

class TranscriptPage(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.c = controller
        layout = QVBoxLayout(self)
        layout.setContentsMargins(36, 30, 36, 24)
        layout.setSpacing(14)
        layout.addWidget(label("Transcript", "PageTitle"))
        layout.addWidget(label("Everything captioned this session, in every language. "
                               "Saved to the logs folder as it happens.", "PageSubtitle", wrap=True))
        bar = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search the transcript")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(lambda _: self.render(force=True))
        bar.addWidget(self.search, 1)
        bar.addWidget(button("Copy all", None, self._copy))
        bar.addWidget(button("Open logs folder", None, controller.open_logs_folder))
        layout.addLayout(bar)

        card = Card(spacing=0)
        card.body.setContentsMargins(6, 6, 6, 6)
        self.view = QTextBrowser()
        self.view.setOpenLinks(False)
        card.add(self.view)
        layout.addWidget(card, 1)

        self._rows = None  # what's shown: [(caption key, its html)]
        self._starts: "list[int]" = []  # where each shown caption starts in the document
        controller.board_changed.connect(self.render)
        controller.targets_changed.connect(lambda: self.render(force=True))

    def showEvent(self, event):  # noqa: N802
        self.render(force=True)

    def _matching(self):
        query = self.search.text().strip().lower()
        captions = [c for c in self.c.board.captions if not c.live]
        if query:
            captions = [c for c in captions if query in c.native.lower() or query in c.speaker.lower()
                        or any(query in text.lower() for text, _ in c.translations.values())]
        return captions

    def _row(self, cap):
        speaker = html.escape(cap.speaker) if cap.speaker else "Someone"
        color = theme.speaker_color(cap.speaker, cap.is_you) if cap.speaker else theme.TEXT_DIM
        lines = [f'<span style="color:{theme.TEXT_FAINT}">{cap.time:%H:%M:%S}</span>&nbsp;&nbsp;'
                 f'<b style="color:{color}">{speaker}</b>&nbsp;&nbsp;'
                 f'<span style="color:{theme.TEXT_FAINT}">{html.escape(spoken_languages_label(cap))}</span>',
                 f'<span style="color:{theme.TEXT}">{html.escape(cap.native)}</span>']
        for target in self.c.targets:
            text, is_translation = cap.translations.get(target, ("", False))
            if is_translation and text:
                lines.append(f'<span style="color:{theme.TEXT_FAINT}">{html.escape(LANGUAGE_NAMES.get(target, target))}'
                             f'</span>&nbsp;&nbsp;<span style="color:{theme.TEXT_DIM}">{html.escape(text)}</span>')
        return "<br>".join(lines) + "<br>"

    def render(self, force=False):
        """Shows the finished captions. Live updates don't touch it; a new
        caption is appended and a changed one (a join - nearly always the
        latest) is swapped in place, from it to the end, so the rest of the
        page never redraws. Your place is kept unless you're at the bottom."""
        if not self.isVisible():
            return
        rows = [(cap.key, self._row(cap)) for cap in self._matching()]
        old = [] if force or self._rows is None else self._rows
        if rows == old and not force:
            return
        first_change = next((i for i, (a, b) in enumerate(zip(rows, old)) if a != b), min(len(rows), len(old)))
        bar = self.view.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 4
        position = bar.value()
        self.view.setUpdatesEnabled(False)
        try:
            doc = self.view.document()
            cursor = QTextCursor(doc)
            if first_change == 0 or not self._starts:
                self.view.clear()
                self._starts = []
                cursor = QTextCursor(doc)
            elif first_change < len(self._starts):
                # Everything from the first changed caption on goes; the rest stays.
                cursor.setPosition(self._starts[first_change] - (1 if first_change else 0))
                cursor.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
                cursor.removeSelectedText()
                del self._starts[first_change:]
            cursor.movePosition(QTextCursor.MoveOperation.End)
            for index in range(len(self._starts), len(rows)):
                if index:
                    cursor.insertBlock()
                self._starts.append(cursor.position())
                cursor.insertHtml(rows[index][1])
            if not rows:
                self.view.setHtml(f'<p style="color:{theme.TEXT_FAINT}">'
                                  f'{"No matches." if self.search.text().strip() else "Nothing yet - captions will collect here."}</p>')
                self._starts = []
            self._rows = rows
            bar.setValue(bar.maximum() if at_bottom or force else position)
        finally:
            self.view.setUpdatesEnabled(True)

    def _copy(self):
        lines = []
        for cap in self._matching():
            lines.append(f"[{cap.time:%H:%M:%S}] {cap.speaker or 'Someone'} ({cap.lang}): {cap.native}")
            lines += [f"    {LANGUAGE_NAMES.get(t, t)}: {text}" for t, (text, is_tr) in cap.translations.items()
                      if is_tr and text]
        QApplication.clipboard().setText("\n".join(lines))


# =============================================================================
# Languages
# =============================================================================

class LanguagesPage(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.c = controller
        page, content = _page("Languages", None)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(page)

        targets = Card("Translate into", "Each language gets its own subtitle overlay. Speech in any language "
                                         "is translated into all of them.")
        self.chips = FlowRow()
        targets.add(self.chips)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Find a language")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        targets.add(self.search)
        self.list = QListWidget()
        self.list.setMinimumHeight(240)
        order = PINNED_LANGUAGES + sorted((c for c in LANGUAGE_NAMES if c not in PINNED_LANGUAGES),
                                          key=lambda c: LANGUAGE_NAMES[c])
        for code in order:
            item = QListWidgetItem(LANGUAGE_NAMES[code])
            item.setData(Qt.ItemDataRole.UserRole, code)
            # Not user-checkable: the click handler toggles it, wherever on
            # the row - a checkable box would toggle too and undo it.
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.list.addItem(item)
        self.list.itemChanged.connect(self._on_item_changed)
        self.list.itemClicked.connect(self._on_item_clicked)
        targets.add(self.list)
        content.addWidget(targets)

        spoken = Card("Spoken language", "Auto-detect handles mixed calls. For a call entirely in one language, "
                                         "choosing it stops short or mumbled speech being misheard as another "
                                         "language - but anything said in a different language is then shown "
                                         "untranslated.")
        self.spoken = Segmented([(None, "Auto-detect"), ("en", "English"), ("zh", "Mandarin"), ("yue", "Cantonese"),
                                 ("vi", "Vietnamese")], speech_options["language"])
        self.spoken.changed.connect(controller.set_language)
        row = QHBoxLayout()
        row.addWidget(self.spoken)
        row.addStretch(1)
        spoken.add(row)
        content.addWidget(spoken)

        second = Card("Second language", "Lines already in an overlay's language - usually your own - get "
                                         "no translation there. Pick a second language and they get a yellow "
                                         "translation into it too, so someone reading that overlay in another "
                                         "language can follow you.")
        self.second_rows = QVBoxLayout()
        self.second_rows.setSpacing(6)
        second.add(self.second_rows)
        content.addWidget(second)

        names = Card("Names & slang", "Words the speech model should spell your way - people's names, game "
                                      "terms, in-jokes. Separate them with commas.")
        self.context = QLineEdit(speech_options["context"])
        self.context.setPlaceholderText("e.g. Caleb, HSK, bún bò Huế, Valorant")
        self.context.editingFinished.connect(lambda: controller.set_context(self.context.text().strip()))
        names.add(self.context)
        self.auto_names = ToggleSwitch(speech_options["auto_names"])
        self.auto_names.toggled.connect(controller.set_auto_names)
        names.add(SettingRow("Add the names of people in the Discord call", "Needs the Discord plugin. "
                             "Names that read like a status rather than a name are skipped.", self.auto_names))
        self.preview = label("", "Faint", wrap=True)
        names.add(self.preview)
        content.addWidget(names)
        content.addStretch(1)

        controller.targets_changed.connect(self._refresh_targets)
        QTimer(self, interval=2000, timeout=self._refresh_preview).start()
        self._refresh_targets()
        self._refresh_preview()

    def _refresh_second_rows(self):
        while self.second_rows.count():
            item = self.second_rows.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if not self.c.targets:
            self.second_rows.addWidget(label("Choose a language above first.", "Dim"))
            return
        for target in self.c.targets:
            combo = QComboBox()
            combo.addItem("None", None)
            for code in PINNED_LANGUAGES + sorted((c for c in LANGUAGE_NAMES if c not in PINNED_LANGUAGES),
                                                  key=lambda c: LANGUAGE_NAMES[c]):
                if code != target:
                    combo.addItem(LANGUAGE_NAMES[code], code)
            combo.setCurrentIndex(max(0, combo.findData(self.c.second_languages.get(target))))
            combo.setFixedWidth(200)
            combo.currentIndexChanged.connect(lambda _, combo=combo, target=target:
                                              self.c.set_second_language(target, combo.currentData()))
            name = LANGUAGE_NAMES.get(target, target)
            self.second_rows.addWidget(SettingRow(f"{name} overlay", f"Lines already in {name} also translated "
                                                  "into:", combo))

    def _refresh_targets(self):
        self._refresh_second_rows()
        blocked = self.list.blockSignals(True)
        for i in range(self.list.count()):
            item = self.list.item(i)
            code = item.data(Qt.ItemDataRole.UserRole)
            item.setCheckState(Qt.CheckState.Checked if code in self.c.targets else Qt.CheckState.Unchecked)
        self.list.blockSignals(blocked)
        chips = []
        for code in self.c.targets:
            chip = Chip(LANGUAGE_NAMES.get(code, code))
            chip.removed.connect(lambda code=code: self.c.remove_target(code))
            chips.append(chip)
        if not chips:
            chips.append(label("None chosen - pick one below.", "Dim"))
        self.chips.set_widgets(chips)

    def _on_item_changed(self, item):
        code = item.data(Qt.ItemDataRole.UserRole)
        if item.checkState() == Qt.CheckState.Checked:
            self.c.add_target(code)
        else:
            self.c.remove_target(code)

    def _on_item_clicked(self, item):
        # The whole row toggles, not just the small checkbox.
        item.setCheckState(Qt.CheckState.Unchecked if item.checkState() == Qt.CheckState.Checked
                           else Qt.CheckState.Checked)

    def _filter(self, text):
        query = text.strip().lower()
        for i in range(self.list.count()):
            item = self.list.item(i)
            item.setHidden(bool(query) and query not in item.text().lower())

    def _refresh_preview(self):
        context = speech_context()
        self.preview.setText(f"The speech model is given: {context}" if context else "")


# =============================================================================
# Captions (appearance and timing)
# =============================================================================

class _PreviewPanel(QWidget):
    """A mock overlay over a game-like backdrop, drawn with the current look."""

    SAMPLES = [
        Caption(key=1, speaker="marina", is_you=False, lang="yue", native="唔知啊，睇下架車先。",
                translations={"en": ("No idea, check the car first.", True)}, group=None, live=False,
                delay=None, proc=None),
        Caption(key=2, speaker="Caleb", is_you=False, lang="yue", native="你將我嘅嘢放去邊啦？",
                translations={"en": ("Where did you put my stuff?", True)}, group=None, live=False,
                delay=None, proc=None),
    ]

    def __init__(self, controller):
        super().__init__()
        self.c = controller
        self.setMinimumHeight(220)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(60, 18, 60, 18)
        layout.addStretch(1)
        self.blocks = [CaptionBlock(), CaptionBlock()]
        self.box = QVBoxLayout()
        self.box.setSpacing(10)
        for block in self.blocks:
            self.box.addWidget(block)
        layout.addLayout(self.box)
        self.refresh()

    def refresh(self):
        for block, sample in zip(self.blocks, self.SAMPLES):
            block.show_caption(sample, "en", self.c.prefs)
        self.update()

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        # a stand-in "game" behind the captions
        from PySide6.QtGui import QLinearGradient
        grad = QLinearGradient(0, 0, self.width(), self.height())
        grad.setColorAt(0, QColor("#3b4f6b"))
        grad.setColorAt(0.55, QColor("#5b6f5a"))
        grad.setColorAt(1, QColor("#8a6a4a"))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(grad)
        p.drawRoundedRect(self.rect(), 10, 10)
        visible = [b for b in self.blocks if b.isVisible()]
        if visible:
            top = min(b.geometry().top() for b in visible) - 14
            bottom = max(b.geometry().bottom() for b in visible) + 14
            p.setBrush(QColor(10, 12, 16, int(255 * self.c.prefs["opacity"])))
            p.drawRoundedRect(QRect(42, top, self.width() - 84, bottom - top), 14, 14)


class CaptionsPage(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.c = controller
        prefs = controller.prefs
        page, content = _page("Captions", "How the subtitle overlays look and behave.")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(page)

        self.preview = _PreviewPanel(controller)
        preview_card = Card(spacing=0)
        preview_card.body.setContentsMargins(8, 8, 8, 8)
        preview_card.add(self.preview)
        content.addWidget(preview_card)

        look = Card("Look")
        look.add(SettingRow("Text size", None, self._slider(70, 200, int(prefs["text_scale"] * 100), "%",
                                                            lambda v: controller.set_pref("text_scale", v / 100))))
        look.add(SettingRow("Background", "How solid the panel behind the text is.",
                            self._slider(0, 100, int(prefs["opacity"] * 100), "%",
                                         lambda v: controller.set_pref("opacity", v / 100))))
        language = ToggleSwitch(prefs["show_language"])
        language.toggled.connect(lambda on: controller.set_pref("show_language", on))
        speakers = ToggleSwitch(prefs["show_speakers"])
        speakers.toggled.connect(lambda on: controller.set_pref("show_speakers", on))
        look.add(SettingRow("Speaker names", "From the Discord plugin; your own speech shows as “You”.",
                            speakers))
        look.add(SettingRow("Spoken language", "Which language each caption was spoken in, beside the name.",
                            language))
        self.original = ToggleSwitch(prefs["show_original"])
        self.original.toggled.connect(lambda on: controller.set_pref("show_original", on))
        look.add(SettingRow("What was said, above the translation", "White is what was said, in its own "
                            "language; yellow is always a translation. Off: just the yellow line.", self.original))
        content.addWidget(look)

        timing = Card("Timing")
        fade = Segmented(FADE_CHOICES, prefs["fade_after"])
        fade.changed.connect(lambda v: controller.set_pref("fade_after", v))
        timing.add(SettingRow("Fade out after silence", "Captions disappear when nobody's talking.", fade))
        self.live = QComboBox()
        self.live.setEditable(True)
        self.live.addItems(["Off"] + [f"{v:g}s" for v in LIVE_CHOICES if v])
        self.live.setCurrentText(self._format_live(live_options["interval"]))
        self.live.setFixedWidth(96)
        self.live.lineEdit().editingFinished.connect(self._on_live_changed)
        self.live.activated.connect(lambda _: self._on_live_changed())
        timing.add(SettingRow("Live updates every", "While someone's still talking, their caption updates "
                              "this often. Shorter feels more live but uses more GPU. Any value from 0.5 to 20 "
                              "seconds, or Off.", self.live))
        translate_live = ToggleSwitch(live_options["translate"])
        translate_live.toggled.connect(controller.set_translate_live)
        timing.add(SettingRow("Translate live updates", "Off: live captions show the original, and the "
                              "translation arrives when they pause.", translate_live))
        content.addWidget(timing)

        windows = Card("Overlay windows", "Unlock the overlays (Home, or Ctrl+Alt+L) to drag them anywhere, "
                                          "resize them from the corner - the bigger, the more history shows - and "
                                          "scroll back through earlier captions. Scrolled back, an overlay holds "
                                          "still until you go back to live. Positions are remembered.")
        row = QHBoxLayout()
        row.addWidget(button("Reset positions", None, controller.reset_positions))
        row.addStretch(1)
        windows.add(row)
        content.addWidget(windows)
        content.addStretch(1)

        controller.prefs_changed.connect(self._sync)

    def _sync(self):
        self.preview.refresh()
        if self.original.isChecked() != self.c.prefs["show_original"]:
            self.original.setChecked(self.c.prefs["show_original"])

    @staticmethod
    def _slider(lo, hi, value, suffix, on_change):
        box = QWidget()
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(value)
        slider.setFixedWidth(220)
        shown = label(f"{value}{suffix}", "Dim")
        shown.setFixedWidth(44)
        shown.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        def changed(v):
            shown.setText(f"{v}{suffix}")
            on_change(v)
        slider.valueChanged.connect(changed)
        row.addWidget(slider)
        row.addWidget(shown)
        return box

    @staticmethod
    def _format_live(seconds):
        return "Off" if seconds is None else f"{seconds:g}s"

    def _on_live_changed(self):
        text = self.live.currentText().strip().lower().removesuffix("s").strip()
        if text in ("off", "0", ""):
            seconds = None
        else:
            try:
                seconds = min(max(float(text), settings.LIVE_INTERVAL_RANGE[0]), settings.LIVE_INTERVAL_RANGE[1])
            except ValueError:
                seconds = live_options["interval"]
        self.live.setCurrentText(self._format_live(seconds))
        if seconds != live_options["interval"]:
            self.c.set_live_interval(seconds)


# =============================================================================
# Audio
# =============================================================================

class AudioPage(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.c = controller
        page, content = _page("Audio", "What's listened to.")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(page)

        system = Card("System audio", "Everything your PC plays - Discord, videos, games - from your default "
                                      "output device. Change that in Windows' sound settings.")
        self.system_line = _StatusLine("Device")
        system.add(self.system_line)
        content.addWidget(system)

        mic = Card("Microphone", "Your own speech.")
        self.mic_choice = QComboBox()
        self.mic_choice.setMinimumWidth(280)
        self.mic_choice.activated.connect(lambda _: controller.set_mic_device(self.mic_choice.currentData()))
        mic.add(SettingRow("Microphone", "Switches straight away. \u201cWindows default\u201d follows whatever "
                           "Windows' sound settings pick.", self.mic_choice))
        self.mic_on = ToggleSwitch(controller.mic_enabled)
        self.mic_on.toggled.connect(controller.set_mic_enabled)
        mic.add(SettingRow("Listen to my microphone", None, self.mic_on))
        self.caption_mic = ToggleSwitch(controller.caption_mic)
        self.caption_mic.toggled.connect(controller.set_caption_mic)
        mic.add(SettingRow("Caption my own speech", "Off saves GPU time - your speech isn't transcribed at all.",
                           self.caption_mic))
        self.mic_line = _StatusLine("Device")
        mic.add(self.mic_line)
        content.addWidget(mic)

        discord = Card("Discord", "With the companion Vencord plugin, captions get people's names, and are cut "
                                  "exactly where each person starts and stops talking. Without it, captions "
                                  "still work, unlabelled.")
        self.discord_line = _StatusLine("Plugin")
        discord.add(self.discord_line)
        self.people = label("", "Faint", wrap=True)
        discord.add(self.people)
        port_box = QWidget()
        port_row = QHBoxLayout(port_box)
        port_row.setContentsMargins(0, 0, 0, 0)
        self.port = QSpinBox()
        self.port.setRange(1024, 65535)
        self.port.setValue(controller.pipeline.bridge_port)
        self.port.setFixedWidth(90)
        port_row.addWidget(self.port)
        port_row.addWidget(button("Apply", None, lambda: controller.set_bridge_port(self.port.value())))
        discord.add(SettingRow("Plugin port", "Only if 8765 is taken - set the same port in the plugin's settings.",
                               port_box))
        content.addWidget(discord)
        content.addStretch(1)

        controller.status_changed.connect(lambda _: self.refresh())
        controller.state_changed.connect(self.refresh)
        QTimer(self, interval=3000, timeout=self.refresh).start()
        self.refresh()

    def showEvent(self, event):  # noqa: N802
        self._fill_mic_choice()  # a headset plugged in since shows up when you come back

    def _fill_mic_choice(self):
        devices = self.c.mic_devices()
        current = self.c.mic_device
        if current and current not in devices:
            devices.append(current)  # chosen but unplugged - keep it showing
        blocked = self.mic_choice.blockSignals(True)
        self.mic_choice.clear()
        self.mic_choice.addItem("Windows default", None)
        for name in devices:
            self.mic_choice.addItem(name, name)
        self.mic_choice.setCurrentIndex(max(0, self.mic_choice.findData(current)))
        self.mic_choice.blockSignals(blocked)

    def refresh(self):
        status = self.c.status
        self.system_line.set(*_status_of("loopback", status.get("loopback")))
        self.mic_line.set(*_status_of("mic", status.get("mic")))
        self.discord_line.set(*_status_of("discord_bridge", status.get("discord_bridge")))
        people = self.c.discord_people() if str(status.get("discord_bridge", "")).startswith("Connected") else []
        self.people.setText(f"In the call: {', '.join(people)}" if people else "")
        self.people.setVisible(bool(people))
        for toggle, value in ((self.mic_on, self.c.mic_enabled), (self.caption_mic, self.c.caption_mic)):
            if toggle.isChecked() != value:
                toggle.setChecked(value)


# =============================================================================
# Models
# =============================================================================

class _ModelOption(QPushButton):
    """A selectable card for one speech model."""

    def __init__(self, title, description, memory, available):
        super().__init__()
        self.setCheckable(True)
        self.setEnabled(available)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("ModelOption")
        self.setStyleSheet(f"""
            #ModelOption {{ background: {theme.SURFACE_2}; border: 1px solid {theme.BORDER}; border-radius: 10px;
                            text-align: left; padding: 0; }}
            #ModelOption:hover {{ border-color: {theme.TEXT_FAINT}; }}
            #ModelOption:checked {{ border: 1px solid {theme.ACCENT}; background: {theme.rgba(theme.ACCENT, 0.08)}; }}
            #ModelOption:disabled {{ background: transparent; }}""")
        row = QHBoxLayout(self)
        row.setContentsMargins(16, 12, 16, 12)
        text = QVBoxLayout()
        text.setSpacing(2)
        name = label(title)
        name.setStyleSheet("font-weight: 600; background: transparent;")
        desc = label(description if available else "Not installed - run setup.bat -QwenASR", "Faint", wrap=True)
        desc.setStyleSheet("background: transparent;")
        text.addWidget(name)
        text.addWidget(desc)
        row.addLayout(text, 1)
        mem = label(memory, "Dim")
        mem.setStyleSheet("background: transparent;")
        row.addWidget(mem)
        for child in (name, desc, mem):
            child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setMinimumHeight(64)


class ModelsPage(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.c = controller
        page, content = _page("Models", "Everything runs on your own GPU - nothing leaves your PC.")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(page)

        speech = Card("Speech recognition", "Turns speech into text. Switching takes a few seconds, during "
                                            "which captions pause.")
        self.group = QButtonGroup(self)
        self.options = {}
        current = controller.pipeline.model_holder.size
        choices = list(MODEL_CHOICES)
        if current not in [c[0] for c in choices]:
            choices.append((current, f"Whisper {current}", "Chosen from the command line.", ""))
        self.available = {}
        for size, title, description, memory in choices:
            available = not size.startswith("qwen") or _qwen_installed(size)
            self.available[size] = available
            option = _ModelOption(title, description, f"~{memory}" if memory else "", available)
            option.clicked.connect(lambda _, size=size: self._choose(size))
            self.group.addButton(option)
            self.options[size] = option
            speech.add(option)
        self.model_line = _StatusLine("Status")
        speech.add(self.model_line)
        content.addWidget(speech)

        translation = Card("Translation", "Hy-MT2-7B by Tencent - a translation model tuned for casual speech. "
                                          "Translates into every chosen language.")
        self.translator_choice = QComboBox()
        self.translator_choice.setMinimumWidth(280)
        for name, title, note in TRANSLATOR_CHOICES:
            available = translator.installed(name)
            size = translator.MODELS[name][1] / 1e9
            self.translator_choice.addItem(f"{title}  ·  {size:.1f} GB" + ("" if available else "  ·  not downloaded"),
                                           name)
            item = self.translator_choice.model().item(self.translator_choice.count() - 1)
            item.setEnabled(available)
            item.setToolTip(note if available else f"Run setup.bat -Translator {name} to download it ({size:.1f} GB).")
        self.translator_choice.activated.connect(
            lambda _: controller.set_translator(self.translator_choice.currentData()))
        translation.add(SettingRow("Model", "The same translator, less compressed: larger is a little more "
                                   "accurate with nuance and slang, and takes more GPU memory. Switching takes a few "
                                   "seconds, during which translations pause.", self.translator_choice))
        self.translator_line = _StatusLine("Status")
        translation.add(self.translator_line)
        content.addWidget(translation)
        content.addStretch(1)

        controller.status_changed.connect(lambda _: self.refresh())
        self.refresh()

    def _choose(self, size):
        if size != self.c.pipeline.model_holder.size:
            self.c.set_model(size)

    def refresh(self):
        holder = self.c.pipeline.model_holder
        option = self.options.get(holder.size)
        if option is not None and not option.isChecked():
            option.setChecked(True)
        model_status = str(self.c.status.get("whisper_model", ""))
        busy = model_status.startswith(("Loading", "Reloading"))
        for size, opt in self.options.items():
            opt.setEnabled(self.available[size] and not busy)
        self.model_line.set(*_status_of("whisper_model", model_status))
        translator_status = str(self.c.status.get("translator", ""))
        self.translator_line.set(*_status_of("translator", translator_status))
        self.translator_choice.setEnabled(not translator_status.startswith("Loading"))
        index = self.translator_choice.findData(translator.model)
        if index >= 0 and index != self.translator_choice.currentIndex():
            self.translator_choice.setCurrentIndex(index)


# =============================================================================
# Window
# =============================================================================

NAV = [
    ("Home", ""),
    ("Transcript", ""),
    ("Languages", ""),
    ("Captions", ""),
    ("Audio", ""),
    ("Models", ""),
]


class MainWindow(QMainWindow):
    def __init__(self, controller):
        super().__init__()
        self.c = controller
        self.tray = None
        self._told_about_tray = False
        self.setWindowTitle("vc-translator")
        self.setWindowIcon(app_icon())
        self.resize(980, 720)
        self.setMinimumSize(820, 560)

        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setCentralWidget(root)

        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(212)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(14, 18, 14, 14)
        side.setSpacing(2)
        brand = QHBoxLayout()
        brand.setSpacing(10)
        logo = QLabel()
        logo.setPixmap(app_icon().pixmap(28, 28))
        brand.addWidget(logo)
        name = label("vc-translator")
        name.setStyleSheet("font-size: 15px; font-weight: 600;")
        brand.addWidget(name)
        brand.addStretch(1)
        side.addLayout(brand)
        side.addSpacing(18)

        self.stack = QStackedWidget()
        self.pages = {
            "Home": HomePage(controller, self.go_to),
            "Transcript": TranscriptPage(controller),
            "Languages": LanguagesPage(controller),
            "Captions": CaptionsPage(controller),
            "Audio": AudioPage(controller),
            "Models": ModelsPage(controller),
        }
        self.nav = QButtonGroup(self)
        self.nav_buttons = {}
        for title, code in NAV:
            btn = QPushButton(f"  {title}")
            btn.setObjectName("NavButton")
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            icon_lbl = glyph(code, 15)
            icon_lbl.setParent(btn)
            icon_lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            icon_lbl.move(10, 9)
            btn.clicked.connect(lambda _, t=title: self.go_to(t))
            self.nav.addButton(btn)
            self.nav_buttons[title] = btn
            side.addWidget(btn)
            self.stack.addWidget(self.pages[title])
        side.addStretch(1)

        status_row = QHBoxLayout()
        status_row.setSpacing(8)
        self.side_dot = StatusDot()
        status_row.addWidget(self.side_dot)
        self.side_status = label("Starting up", "Dim")
        status_row.addWidget(self.side_status, 1)
        side.addLayout(status_row)
        side.addSpacing(8)
        quit_btn = button("Quit", None, QApplication.quit)
        quit_btn.setObjectName("QuitButton")
        quit_btn.setToolTip("Stop captioning and close vc-translator - closing this window keeps it running in the tray")
        side.addWidget(quit_btn)

        layout.addWidget(sidebar)
        layout.addWidget(self.stack, 1)

        controller.state_changed.connect(self._refresh_side)
        controller.status_changed.connect(lambda _: self._refresh_side())
        self._refresh_side()
        self.go_to("Home")

    def go_to(self, title):
        self.nav_buttons[title].setChecked(True)
        self.stack.setCurrentWidget(self.pages[title])

    def _refresh_side(self):
        if self.c.paused:
            self.side_dot.set_color(theme.WARNING)
            self.side_status.setText("Paused")
        elif str(self.c.status.get("whisper_model", "")).startswith("Ready"):
            self.side_dot.set_color(theme.SUCCESS)
            self.side_status.setText("Listening")
        else:
            self.side_dot.set_color(theme.WARNING)
            self.side_status.setText("Starting up")

    def closeEvent(self, event):  # noqa: N802
        # Keep running in the tray - captions are the point, not this window.
        if self.c.quitting or self.tray is None:
            event.accept()
            return
        event.ignore()
        self.hide()
        if not self._told_about_tray:
            self._told_about_tray = True
            self.tray.showMessage("vc-translator is still running",
                                  "Captions keep going. Open it again or quit from the tray icon.",
                                  app_icon(), 4000)
