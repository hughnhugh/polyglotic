"""
A target language's subtitle overlay: the session's captions, newest at the
bottom, over a soft translucent panel - the translation large, the
original speech small underneath, the speaker's name in their colour.

Built to stay out of the way in a game: no window chrome, fades out after
a few seconds of silence, and when locked, clicks pass straight through to
whatever's underneath. Unlocked (Ctrl+Alt+L, the tray, or the main
window), hovering shows its controls: drag anywhere to move, the corner to
resize, x to stop captioning that language - and the mouse wheel scrolls
back through earlier captions. Scrolled back, the overlay holds still
while new captions keep arriving, until "Back to live" (or scrolling to
the bottom) catches it up.
"""

import math
import time

from PySide6.QtCore import QPoint, QPointF, QRect, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QTextLayout, QTextOption
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizeGrip, QSizePolicy,
                               QVBoxLayout, QWidget)

from ui import theme

PAD = 14
FADE_STEP = 0.08          # per 30ms tick - about a third of a second to fade
HISTORY = 150             # captions an overlay keeps to scroll back through
BASE_PRIMARY_PX = 22
BASE_SECONDARY_PX = 15
BASE_SPEAKER_PX = 13


class OutlinedText(QWidget):
    """Word-wrapped text with a dark outline round every letter, like film
    subtitles - readable over any scene, even with no panel behind it.
    QLabel can't stroke text, so this lays it out and draws it itself."""

    def __init__(self):
        super().__init__()
        self._text = ""
        self._font = QFont()
        self._color = QColor("white")
        self._outline = 3.0
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def set(self, text, font, color, outline):
        color = QColor(color)
        if (text, font, color, outline) != (self._text, self._font, self._color, self._outline):
            self._text, self._font, self._color, self._outline = text, QFont(font), color, outline
            self.updateGeometry()
            self.update()

    def _lines(self, width):
        """[(text, x, baseline_y)] for the text wrapped at `width`, and the
        total height."""
        pad = math.ceil(self._outline)
        layout = QTextLayout(self._text, self._font)
        option = QTextOption()
        option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        layout.setTextOption(option)
        layout.beginLayout()
        lines, y = [], pad
        while True:
            line = layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(max(10, width - 2 * pad))
            line.setPosition(QPointF(pad, y))
            lines.append((self._text[line.textStart():line.textStart() + line.textLength()], pad, y + line.ascent()))
            y += line.height()
        layout.endLayout()
        return lines, math.ceil(y) + pad

    def hasHeightForWidth(self):  # noqa: N802
        return True

    def heightForWidth(self, width):  # noqa: N802
        return self._lines(width)[1]

    def sizeHint(self):  # noqa: N802
        width = self.width() if self.width() > 50 else 400
        return QSize(width, self.heightForWidth(width))

    def minimumSizeHint(self):  # noqa: N802
        return QSize(50, self.heightForWidth(max(self.width(), 50)))

    def paintEvent(self, event):  # noqa: N802
        lines, _ = self._lines(self.width())
        path = QPainterPath()
        for text, x, baseline in lines:
            path.addText(QPointF(x, baseline), self._font, text.rstrip())
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(0, 0, 0, 235), self._outline * 2)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.strokePath(path, pen)
        p.fillPath(path, self._color)


def _caption_font(px, weight):
    font = QFont()
    font.setFamilies(theme.CAPTION_FAMILIES)
    font.setPixelSize(max(8, round(px)))
    font.setWeight(weight)
    return font


class CaptionBlock(QWidget):
    """One caption: speaker, translation, original."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.speaker = OutlinedText()
        self.primary = OutlinedText()
        self.secondary = OutlinedText()
        for part in (self.speaker, self.primary, self.secondary):
            layout.addWidget(part)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def show_caption(self, caption, target, prefs):
        scale = prefs["text_scale"]
        primary, original = caption.text_for(target)

        show_speaker = prefs["show_speakers"] and bool(caption.speaker)
        self.speaker.setVisible(show_speaker)
        if show_speaker:
            self.speaker.set(caption.speaker, _caption_font(BASE_SPEAKER_PX * scale, QFont.Weight.DemiBold),
                             theme.speaker_color(caption.speaker, caption.is_you), 1.6 * scale)

        # A live caption is still being spoken - slightly softer, and it
        # trails off, until the final version replaces it.
        color = QColor("white")
        if caption.live:
            color.setAlphaF(0.85)
        self.primary.set(primary + (" \u2026" if caption.live else ""),
                         _caption_font(BASE_PRIMARY_PX * scale, QFont.Weight.Medium), color, 1.9 * scale)

        show_original = prefs["show_original"] and bool(original)
        self.secondary.setVisible(show_original)
        if show_original:
            self.secondary.set(original, _caption_font(BASE_SECONDARY_PX * scale, QFont.Weight.Normal), "#d3d8e2",
                               1.6 * scale)


class CaptionOverlay(QWidget):
    def __init__(self, controller, target, title):
        super().__init__(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint)
        self.controller = controller
        self.target = target
        self.setWindowTitle(f"{title} captions")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setMinimumSize(320, 120)

        self._hovered = False
        self._locked = False
        self._alpha = 1.0
        self._last_activity = 0.0
        self._last_seen = None    # (key, text) of the newest caption, to notice activity
        self._drag_from = None
        self._shown_version = None
        self._following = True    # showing live captions; False while scrolled back (frozen)
        self._frozen_at = None    # board.version when it froze, to count what's new since
        self._auto_scrolling = False
        self._blocks: "dict[int, CaptionBlock]" = {}  # caption key -> its block

        root = QVBoxLayout(self)
        root.setContentsMargins(PAD + 4, PAD, PAD + 4, PAD)
        root.setSpacing(8)

        self.header = QWidget()
        head = QHBoxLayout(self.header)
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(4)
        name = QLabel(title.upper())
        name.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 11px; font-weight: 600; letter-spacing: 1px;")
        head.addWidget(name)
        head.addStretch(1)
        hint = QLabel("Scroll for earlier captions")
        hint.setStyleSheet(f"color: {theme.TEXT_FAINT}; font-size: 11px;")
        head.addWidget(hint)
        head.addSpacing(6)
        head.addWidget(self._header_button("", "Lock - clicks pass through to what's underneath (Ctrl+Alt+L)",
                                           lambda: controller.set_locked(True)))
        head.addWidget(self._header_button("", f"Stop {title} captions", lambda: controller.remove_target(target)))
        root.addWidget(self.header)

        # Captions scroll; a stretch above them keeps a few sitting at the bottom.
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setStyleSheet(f"""
            QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; }}
            QScrollBar:vertical {{ background: transparent; width: 6px; margin: 2px 0; }}
            QScrollBar::handle:vertical {{ background: rgba(255,255,255,0.28); border-radius: 3px; min-height: 24px; }}
            QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{
                background: none; height: 0; }}""")
        self.content = QWidget()
        self.content.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.list = QVBoxLayout(self.content)
        self.list.setContentsMargins(0, 0, 8, 0)
        self.list.setSpacing(12)
        self.list.addStretch(1)
        self.placeholder = QLabel(f"{title} captions appear here")
        self.placeholder.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 15px;")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.list.addWidget(self.placeholder)
        self.scroll.setWidget(self.content)
        root.addWidget(self.scroll, 1)
        bar = self.scroll.verticalScrollBar()
        bar.valueChanged.connect(self._on_scrolled)
        bar.rangeChanged.connect(lambda *_: self._stick_to_bottom())

        # Floats over the captions while scrolled back.
        self.live_btn = QPushButton(self)
        self.live_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.live_btn.setStyleSheet(f"""
            QPushButton {{ background: {theme.ACCENT}; color: {theme.ACCENT_TEXT}; border: none; border-radius: 14px;
                           padding: 5px 14px; font-size: 12px; font-weight: 600; }}
            QPushButton:hover {{ background: {theme.ACCENT_HOVER}; }}""")
        self.live_btn.clicked.connect(self.go_live)
        self.live_btn.hide()

        self.grip = QSizeGrip(self)
        self.grip.setFixedSize(18, 18)
        self.grip.setToolTip("Drag to resize")

        self._save_timer = QTimer(self, singleShot=True, interval=400,
                                  timeout=lambda: controller.save_geometry(target, self.geometry()))
        self._tick_timer = QTimer(self, interval=30, timeout=self._tick)
        self._tick_timer.start()
        self.set_locked(controller.locked)

    def _header_button(self, code, tip, on_click):
        btn = QPushButton(code)
        btn.setToolTip(tip)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setFixedSize(26, 22)
        btn.setStyleSheet(f"""
            QPushButton {{ background: transparent; border: none; border-radius: 6px; color: {theme.TEXT_DIM};
                           {theme.font_css([theme.icon_family()], 12)} padding: 0; }}
            QPushButton:hover {{ background: rgba(255,255,255,0.12); color: white; }}""")
        btn.clicked.connect(on_click)
        return btn

    # -- state ----------------------------------------------------------------

    def set_locked(self, locked):
        self._locked = locked
        visible = self.isVisible()
        self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, locked)
        if visible:
            self.show()  # changing window flags hides it
        if locked:
            self._hovered = False
            self.go_live()  # can't scroll a click-through overlay - it follows live
        self._update_chrome()

    def _update_chrome(self):
        interactive = self._hovered and not self._locked
        self.header.setVisible(interactive)
        self.grip.setVisible(interactive)
        self.placeholder.setVisible(interactive and not self._blocks)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded if interactive
                                               else Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setCursor(Qt.CursorShape.SizeAllCursor if interactive else Qt.CursorShape.ArrowCursor)
        self._place_live_button()
        self.update()

    def refresh(self, force=False):
        """Redraws from the caption board if it (or the look) changed - unless
        scrolled back, where it holds still and just counts what's new."""
        board = self.controller.board
        if not self._following:
            new = sum(1 for c in board.captions if c.key not in self._blocks and self.target in c.translations)
            count = f"  ·  {new} new" if new else ""
            self.live_btn.setText(f"↓  Back to live{count}")
            self._place_live_button()
            return
        version = (board.version, self.controller.prefs_version)
        if version == self._shown_version and not force:
            return
        self._shown_version = version
        prefs = self.controller.prefs
        shown = [c for c in board.captions if self.target in c.translations or not c.translations][-HISTORY:]

        keep = {c.key for c in shown}
        for key in [k for k in self._blocks if k not in keep]:
            self._blocks.pop(key).deleteLater()
        for i, caption in enumerate(shown):
            block = self._blocks.get(caption.key)
            if block is None:
                block = self._blocks[caption.key] = CaptionBlock()
            # 2 + i: after the stretch and the placeholder. insertWidget
            # moves it if it's already there.
            if self.list.indexOf(block) != 2 + i:
                self.list.insertWidget(2 + i, block)
            block.show_caption(caption, self.target, prefs)

        newest = (shown[-1].key, shown[-1].text_for(self.target)[0]) if shown else None
        if newest != self._last_seen:
            self._last_seen = newest
            if newest is not None:
                self._last_activity = time.time()
        self._update_chrome()
        self._stick_to_bottom()

    # -- scrolling ------------------------------------------------------------

    def _stick_to_bottom(self):
        if self._following:
            bar = self.scroll.verticalScrollBar()
            self._auto_scrolling = True
            bar.setValue(bar.maximum())
            self._auto_scrolling = False

    def _on_scrolled(self, value):
        if self._auto_scrolling:
            return
        bar = self.scroll.verticalScrollBar()
        at_bottom = value >= bar.maximum() - 2
        if self._following and not at_bottom:
            self._following = False   # scrolled back - freeze here
            self._frozen_at = self.controller.board.version
            self.refresh()
            self.live_btn.show()
            self._place_live_button()
        elif not self._following and at_bottom:
            self.go_live()

    def go_live(self):
        if self._following:
            return
        self._following = True
        self.live_btn.hide()
        self.refresh(force=True)
        self._stick_to_bottom()

    def _place_live_button(self):
        if not self.live_btn.isVisible():
            return
        self.live_btn.adjustSize()
        area = self.scroll.geometry()
        self.live_btn.move(area.center().x() - self.live_btn.width() // 2, area.bottom() - self.live_btn.height() - 6)
        self.live_btn.raise_()

    def _tick(self):
        fade_after = self.controller.prefs["fade_after"]
        if (self._hovered and not self._locked) or not self._following:
            target = 1.0
        elif not self._blocks:
            target = 0.0
        elif fade_after and time.time() - self._last_activity > fade_after:
            target = 0.0
        else:
            target = 1.0
        if abs(target - self._alpha) > 0.001:
            self._alpha = min(1.0, self._alpha + FADE_STEP) if target > self._alpha else max(0.0, self._alpha - FADE_STEP)
            self.setWindowOpacity(self._alpha)

    # -- painting -------------------------------------------------------------

    def paintEvent(self, event):  # noqa: N802
        """The panel goes behind what's actually showing: the captions in
        view (a few sit at the bottom, not the whole window) and the header."""
        viewport = self.scroll.viewport()
        view_rect = QRect(viewport.mapTo(self, QPoint(0, 0)), viewport.size())
        tops, bottoms = [], []
        for w in (self.placeholder, *self._blocks.values()):
            if w.isVisible():
                r = QRect(w.mapTo(self, QPoint(0, 0)), w.size()).intersected(view_rect)
                if not r.isEmpty():
                    tops.append(r.top())
                    bottoms.append(r.bottom())
        if self.header.isVisible():
            tops.append(self.header.geometry().top())
            bottoms.append(self.header.geometry().bottom())
        if not tops:
            return
        top, bottom = min(tops), max(bottoms)
        panel = QRect(0, top - PAD, self.width(), bottom - top + 2 * PAD)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        opacity = self.controller.prefs["opacity"]
        if (self._hovered and not self._locked) or not self._following:
            opacity = max(opacity, 0.55)
            p.setPen(QPen(QColor(255, 255, 255, 40), 1))
        else:
            p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(10, 12, 16, int(255 * opacity)))
        p.drawRoundedRect(panel.adjusted(1, 1, -1, -1), 14, 14)

    def resizeEvent(self, event):  # noqa: N802
        self.grip.move(self.width() - self.grip.width() - 4, self.height() - self.grip.height() - 4)
        self._place_live_button()
        if event.oldSize().isValid() and self.isVisible():
            self._save_timer.start()

    # -- mouse (only reaches here unlocked) ---------------------------------------

    def enterEvent(self, event):  # noqa: N802
        self._hovered = True
        self._update_chrome()

    def leaveEvent(self, event):  # noqa: N802
        self._hovered = False
        self._update_chrome()

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_from = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._drag_from is not None:
            self.move(event.globalPosition().toPoint() - self._drag_from)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if self._drag_from is not None:
            self._drag_from = None
            self.controller.save_geometry(self.target, self.geometry())


def default_geometry(screen_rect, index, count):
    """Where the index-th overlay goes before it's been moved: centred
    near the bottom of the screen, later ones stacked above."""
    width = min(960, int(screen_rect.width() * 0.5))
    height = 260
    x = screen_rect.x() + (screen_rect.width() - width) // 2
    y = screen_rect.y() + screen_rect.height() - 90 - height - index * (height + 8)
    return QRect(QPoint(x, max(screen_rect.y(), y)), QPoint(x + width - 1, max(screen_rect.y(), y) + height - 1))
