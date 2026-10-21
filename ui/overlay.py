"""
A target language's subtitle overlay: the last few captions, newest at the
bottom, over a soft translucent panel - the translation large, the
original speech small underneath, the speaker's name in their colour.

Built to stay out of the way in a game: no window chrome, fades out after
a few seconds of silence, and when locked, clicks pass straight through to
whatever's underneath. Unlocked (Ctrl+Alt+L, the tray, or the main
window), hovering shows its controls: drag anywhere to move, the corner to
resize, x to stop captioning that language.
"""

import time

from PySide6.QtCore import QPoint, QRect, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (QGraphicsDropShadowEffect, QHBoxLayout, QLabel, QPushButton, QSizeGrip,
                               QVBoxLayout, QWidget)

from ui import theme

PAD = 14
FADE_STEP = 0.08          # per 30ms tick - about a third of a second to fade
OLDER_ALPHA = 0.55        # captions above the newest are dimmed
BASE_PRIMARY_PX = 22
BASE_SECONDARY_PX = 15
BASE_SPEAKER_PX = 13


class CaptionBlock(QWidget):
    """One caption: speaker, translation, original."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        self.speaker = QLabel()
        self.primary = QLabel()
        self.secondary = QLabel()
        for lbl in (self.speaker, self.primary, self.secondary):
            lbl.setTextFormat(Qt.TextFormat.PlainText)
            lbl.setWordWrap(True)
            lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            layout.addWidget(lbl)
        # Keeps text readable when the panel behind it is faint or off.
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(10)
        shadow.setOffset(0, 1)
        shadow.setColor(QColor(0, 0, 0, 220))
        self.setGraphicsEffect(shadow)

    def show_caption(self, caption, target, prefs, newest):
        scale = prefs["text_scale"]
        alpha = 1.0 if newest else OLDER_ALPHA
        primary, original = caption.text_for(target)

        families = theme.CAPTION_FAMILIES
        show_speaker = prefs["show_speakers"] and bool(caption.speaker)
        self.speaker.setVisible(show_speaker)
        if show_speaker:
            self.speaker.setText(caption.speaker)
            self.speaker.setStyleSheet(theme.font_css(families, BASE_SPEAKER_PX * scale, 600)
                                       + f" color: {theme.rgba(theme.speaker_color(caption.speaker, caption.is_you), alpha)};")

        # A live caption is still being spoken - slightly softer, and it
        # trails off, until the final version replaces it.
        self.primary.setText(primary + (" …" if caption.live else ""))
        self.primary.setStyleSheet(theme.font_css(families, BASE_PRIMARY_PX * scale, 500)
                                   + f" color: {theme.rgba('#ffffff', alpha * (0.82 if caption.live else 1.0))};")

        show_original = prefs["show_original"] and bool(original)
        self.secondary.setVisible(show_original)
        if show_original:
            self.secondary.setText(original)
            self.secondary.setStyleSheet(theme.font_css(families, BASE_SECONDARY_PX * scale)
                                         + f" color: {theme.rgba('#c7cdd8', alpha * 0.9)};")


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
        self._alpha = 1.0
        self._last_activity = 0.0
        self._last_seen = None   # (key, text) of the newest caption, to notice activity
        self._drag_from = None
        self._shown_version = None

        root = QVBoxLayout(self)
        root.setContentsMargins(PAD + 4, PAD, PAD + 4, PAD)
        root.setSpacing(10)

        self.header = QWidget()
        head = QHBoxLayout(self.header)
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(4)
        name = QLabel(title.upper())
        name.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 11px; font-weight: 600; letter-spacing: 1px;")
        head.addWidget(name)
        head.addStretch(1)
        self.lock_btn = self._header_button("", "Lock - clicks pass through to what's underneath (Ctrl+Alt+L)",
                                            lambda: controller.set_locked(True))
        close_btn = self._header_button("", f"Stop {title} captions", lambda: controller.remove_target(target))
        head.addWidget(self.lock_btn)
        head.addWidget(close_btn)
        root.addWidget(self.header)
        root.addStretch(1)

        self.body = QVBoxLayout()
        self.body.setSpacing(10)
        root.addLayout(self.body)
        self.placeholder = QLabel(f"{title} captions appear here")
        self.placeholder.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 15px;")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.body.addWidget(self.placeholder)
        self._blocks: "list[CaptionBlock]" = []

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
        self._hovered = False if locked else self._hovered
        self._update_chrome()

    def _update_chrome(self):
        interactive = self._hovered and not self._locked
        self.header.setVisible(interactive)
        self.grip.setVisible(interactive)
        self.placeholder.setVisible(interactive and not self._blocks)
        self.setCursor(Qt.CursorShape.SizeAllCursor if interactive else Qt.CursorShape.ArrowCursor)
        self._fit()
        self.update()

    def _fit(self):
        """Shows as many captions as fit, newest first - in a small window,
        or with the header showing, the oldest drop off rather than
        squeezing into each other."""
        margins = self.layout().contentsMargins()
        room = self.height() - margins.top() - margins.bottom()
        if self.header.isVisibleTo(self):
            room -= self.header.sizeHint().height() + self.layout().spacing()
        width = self.width() - margins.left() - margins.right()
        fits = True
        for i, block in enumerate(reversed(self._blocks)):
            # At the real width - sizeHint() guesses a narrower wrap, too tall
            need = block.heightForWidth(width) if block.hasHeightForWidth() else block.sizeHint().height()
            fits = fits and (i == 0 or need <= room)  # once one doesn't fit, nothing older shows either
            block.setVisible(fits)
            room -= need + self.body.spacing()

    def refresh(self, force=False):
        """Redraws from the caption board if it (or the look) changed."""
        version = (self.controller.board.version, self.controller.prefs_version)
        if version == self._shown_version and not force:
            return
        self._shown_version = version
        prefs = self.controller.prefs
        captions = [c for c in self.controller.board.captions if self.target in c.translations or not c.translations]
        shown = captions[-max(1, prefs["lines"]):]

        while len(self._blocks) < len(shown):
            block = CaptionBlock()
            self._blocks.append(block)
            self.body.addWidget(block)
        while len(self._blocks) > len(shown):
            self._blocks.pop().deleteLater()
        for i, (block, caption) in enumerate(zip(self._blocks, shown)):
            block.show_caption(caption, self.target, prefs, newest=i == len(shown) - 1)

        self._fit()

        newest = (shown[-1].key, shown[-1].text_for(self.target)[0]) if shown else None
        if newest != self._last_seen:
            self._last_seen = newest
            if newest is not None:
                self._last_activity = time.time()
        self._update_chrome()

    def _tick(self):
        fade_after = self.controller.prefs["fade_after"]
        if self._hovered and not self._locked:
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
        widgets = [w for w in (self.header, self.placeholder, *self._blocks) if w.isVisible()]
        if not widgets:
            return
        top = min(w.geometry().top() for w in widgets)
        bottom = max(w.geometry().bottom() for w in widgets)
        panel = QRect(0, top - PAD, self.width(), bottom - top + 2 * PAD)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        opacity = self.controller.prefs["opacity"]
        if self._hovered and not self._locked:
            opacity = max(opacity, 0.55)
            p.setPen(QPen(QColor(255, 255, 255, 40), 1))
        else:
            p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(10, 12, 16, int(255 * opacity)))
        p.drawRoundedRect(panel.adjusted(1, 1, -1, -1), 14, 14)

    def resizeEvent(self, event):  # noqa: N802
        self.grip.move(self.width() - self.grip.width() - 4, self.height() - self.grip.height() - 4)
        self._fit()
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
    height = 230
    x = screen_rect.x() + (screen_rect.width() - width) // 2
    y = screen_rect.y() + screen_rect.height() - 90 - height - index * (height + 8)
    return QRect(QPoint(x, max(screen_rect.y(), y)), QPoint(x + width - 1, max(screen_rect.y(), y) + height - 1))
