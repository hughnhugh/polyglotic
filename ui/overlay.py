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

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import (QColor, QFont, QFontMetricsF, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap,
                           QTextLayout, QTextOption)
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizeGrip, QSizePolicy, QVBoxLayout, QWidget

from ui import theme

PAD = 14
FADE_STEP = 0.08          # per 30ms tick - about a third of a second to fade
HISTORY = 150             # captions an overlay keeps to scroll back through
BASE_PRIMARY_PX = 22
BASE_SECONDARY_PX = 18    # the original - big enough to read at a glance, not to compete
BASE_SPEAKER_PX = 13
# The original speech under a translation, in classic subtitle yellow - set
# apart from the white translation, and easy to pick out.
ORIGINAL_COLOR = "#ffd76e"


class OutlinedText(QWidget):
    """Word-wrapped text with a dark outline round every letter, like film
    subtitles - readable over any scene, even with no panel behind it.
    QLabel can't stroke text, so this lays it out and draws it itself."""

    def __init__(self, wrap=True):
        """wrap=False: one line, as wide as its text (a label beside another)."""
        super().__init__()
        self._text = ""
        self._font = QFont()
        self._color = QColor("white")
        self._outline = 3.0
        self._wrap = wrap
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        if wrap:
            policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
            policy.setHeightForWidth(True)
        else:
            policy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
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
        return self._wrap

    def heightForWidth(self, width):  # noqa: N802
        return self._lines(width)[1]

    def sizeHint(self):  # noqa: N802
        if not self._wrap:
            pad = math.ceil(self._outline)
            width = math.ceil(QFontMetricsF(self._font).horizontalAdvance(self._text)) + 2 * pad + 2
            return QSize(width, self._lines(width)[1])
        width = self.width() if self.width() > 50 else 400
        return QSize(width, self.heightForWidth(width))

    def minimumSizeHint(self):  # noqa: N802
        if not self._wrap:
            return self.sizeHint()
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


# Chinese/Japanese characters pack far more strokes into each glyph, so
# at the same pixel size they read as smaller and denser than Latin text -
# subtitles in those scripts are set larger. Text that's mostly CJK gets
# this much more size, in the font drawn for its script.
CJK_SCALE = 1.18
CJK_FAMILIES = {
    "simplified": ["Microsoft YaHei UI", "Microsoft YaHei"],
    "traditional": ["Microsoft JhengHei UI", "Microsoft JhengHei"],
    "japanese": ["Yu Gothic UI", "Meiryo UI"],
}
SCRIPT_OF_LANGUAGE = {"zh": "simplified", "zh-hant": "traditional", "yue": "traditional", "ja": "japanese"}


def _cjk_share(text):
    letters = [c for c in text if not c.isspace()]
    return sum("\u2e80" <= c <= "\u9fff" or "\uf900" <= c <= "\ufaff" for c in letters) / len(letters) if letters else 0


def _caption_font(px, weight, text="", lang=None):
    """The caption font for `text` in language `lang`: the Latin caption
    families, with the right CJK font first in line for Chinese/Japanese
    (YaHei's glyphs for Simplified, JhengHei's for Traditional/Cantonese)
    - and mostly-CJK text set CJK_SCALE larger."""
    script = SCRIPT_OF_LANGUAGE.get(lang)
    cjk = _cjk_share(text) > 0.3
    if cjk and script is None:
        script = "simplified"
    families = list(theme.CAPTION_FAMILIES)
    if cjk:
        # First, not just ahead of the other fallbacks: Qt caches a font's
        # CJK fallback by its first family, so Simplified and Traditional
        # behind the same Latin face would share whichever loaded first.
        families = CJK_FAMILIES[script] + families
    font = QFont()
    font.setFamilies(families)
    font.setPixelSize(max(8, round(px * (CJK_SCALE if cjk else 1.0))))
    font.setWeight(weight)
    return font


# What was spoken, as the tag beside the speaker: Chinese speech is
# "Mandarin" (its target name, "Chinese (Simplified)", describes a script).
_SPOKEN_NAMES = {"zh": "Mandarin", "yue": "Cantonese"}


def spoken_language_name(code):
    from lang_codes import LANGUAGE_NAMES
    return _SPOKEN_NAMES.get(code) or LANGUAGE_NAMES.get(code) or code.upper()


class CaptionBlock(QWidget):
    """One caption: speaker, translation, original."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        # Who said it, and in what language - "marina  Mandarin"
        self.head = QWidget()
        head = QHBoxLayout(self.head)
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(6)
        self.speaker = OutlinedText(wrap=False)
        self.language = OutlinedText(wrap=False)
        head.addWidget(self.speaker)
        head.addWidget(self.language)
        head.addStretch(1)
        self.primary = OutlinedText()
        self.secondary = OutlinedText()
        for part in (self.head, self.primary, self.secondary):
            layout.addWidget(part)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def show_caption(self, caption, target, prefs):
        scale = prefs["text_scale"]
        primary, original = caption.text_for(target)

        show_speaker = prefs["show_speakers"] and bool(caption.speaker)
        show_language = prefs["show_language"] and bool(caption.lang)
        self.head.setVisible(show_speaker or show_language)
        self.speaker.setVisible(show_speaker)
        if show_speaker:
            self.speaker.set(caption.speaker, _caption_font(BASE_SPEAKER_PX * scale, QFont.Weight.DemiBold),
                             theme.speaker_color(caption.speaker, caption.is_you), 1.6 * scale)
        self.language.setVisible(show_language)
        if show_language:
            self.language.set(spoken_language_name(caption.lang), _caption_font(BASE_SPEAKER_PX * scale,
                                                                                QFont.Weight.Normal),
                              "#aab2c0", 1.6 * scale)

        # A live caption is still being spoken - slightly softer, and it
        # trails off, until the final version replaces it.
        color = QColor("white")
        if caption.live:
            color.setAlphaF(0.85)
        self.primary.set(primary + (" \u2026" if caption.live else ""),
                         _caption_font(BASE_PRIMARY_PX * scale, QFont.Weight.Medium, primary,
                                       target if original else caption.lang), color, 1.9 * scale)

        show_original = prefs["show_original"] and bool(original)
        self.secondary.setVisible(show_original)
        if show_original:
            self.secondary.set(original, _caption_font(BASE_SECONDARY_PX * scale, QFont.Weight.Medium, original,
                                                       caption.lang), ORIGINAL_COLOR,
                               1.6 * scale)


def _text_paths(text, font, width, x=0.0, y=0.0):
    """(path, height): `text` word-wrapped at `width` as one QPainterPath,
    its top-left at (x, y)."""
    layout = QTextLayout(text, font)
    option = QTextOption()
    option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
    layout.setTextOption(option)
    layout.beginLayout()
    path, h = QPainterPath(), 0.0
    while True:
        line = layout.createLine()
        if not line.isValid():
            break
        line.setLineWidth(max(10.0, width))
        line.setPosition(QPointF(0, h))
        segment = text[line.textStart():line.textStart() + line.textLength()].rstrip()
        if segment:
            path.addText(QPointF(x, y + h + line.ascent()), font, segment)
        h += line.height()
    layout.endLayout()
    return path, h


class _Item:
    """One caption as the canvas draws it: its text as outlined paths, and
    where it is / is heading on screen."""

    def __init__(self, key):
        self.key = key
        self.signature = None
        self.pixmap = None       # the caption drawn once (outlined text is costly) - frames just place it
        self.text = ""           # primary text, to tell growing (typed on) from rewritten (crossfaded)
        self.height = 0.0
        self.y = None            # on-screen top, sprung toward target_y
        self.vy = 0.0            # its velocity, px/s
        self.target_y = 0.0
        self.alpha = 0.0         # fades in on arrival, out on removal
        self.target_alpha = 1.0
        self.old_pixmap = None   # the previous text while crossfading to a rewrite
        self.cross = 1.0         # 0 -> 1: old text out, new text in
        self.ghost = False       # removed - fading out where it was, taking no space


class CaptionCanvas(QWidget):
    """Draws an overlay's captions and animates every change: captions ease
    to new positions when anything above or below them changes height (a
    new caption, a live caption growing a line, a join), arrive fading and
    rising into place, leave fading out while the rest close the gap, and
    crossfade when their text is rewritten (text that only grows just
    extends). The wheel scrolls smoothly too. One painted surface rather
    than a widget per caption - widgets resize in single frames, which is
    what read as flicker."""

    LINE_GAP = 4         # between a caption's speaker line, translation and original
    CAPTION_GAP = 14     # between captions
    # Positions move on a critically damped spring: they start from rest,
    # accelerate, and settle without overshoot - and keep their momentum
    # when the target changes mid-move (a live caption updating while
    # things are still sliding), so there's never a jerk. ~0.3s to settle.
    OMEGA = 20.0         # rad/s - the spring's stiffness
    FADE_TAU = 0.07      # seconds - fade in/out time constant
    CROSSFADE = 0.2      # seconds
    RISE = 14            # px a new caption rises as it fades in

    def __init__(self, overlay):
        super().__init__(overlay)
        self.overlay = overlay
        self.items: "dict[int, _Item]" = {}
        self.order: "list[int]" = []      # keys top to bottom (ghosts excluded)
        self.ghosts: "list[_Item]" = []
        self.scroll = 0.0                 # content scrolled off the top, animated toward target_scroll
        self.target_scroll = 0.0
        self.following = True
        self.on_scrolled_back = None      # set by the overlay
        self.on_back_at_bottom = None
        self._last_frame = time.monotonic()
        self._timer = QTimer(self, interval=8, timeout=self._animate)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)  # the default ticks unevenly on Windows
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    # -- content ----------------------------------------------------------------

    def set_captions(self, captions, target, prefs):
        """Brings the canvas up to date with `captions` (top to bottom)."""
        width = self.width()
        keys = [c.key for c in captions]
        for key in [k for k in self.order if k not in set(keys)]:
            item = self.items.pop(key)
            if item.y is not None and -item.height < item.y < self.height():
                item.ghost, item.target_alpha = True, 0.0   # visible - fade it out
                self.ghosts.append(item)
        for caption in captions:
            item = self.items.get(caption.key)
            if item is None:
                item = self.items[caption.key] = _Item(caption.key)
            signature = (caption.speaker, caption.lang, caption.live, caption.text_for(target), width,
                         tuple(sorted(prefs.items())))
            if signature != item.signature:
                primary = caption.text_for(target)[0]
                rewritten = item.signature is not None and not primary.startswith(item.text.rstrip(" …"))
                if rewritten:
                    item.old_pixmap, item.cross = item.pixmap, 0.0
                item.signature, item.text = signature, primary
                parts, item.height = self._build(caption, target, prefs, width)
                item.pixmap = self._render(parts, width, item.height)
        self.order = keys
        self._layout()

    def _build(self, caption, target, prefs, width):
        scale = prefs["text_scale"]
        primary, original = caption.text_for(target)
        parts, y = [], 0.0
        pad = 3  # room for the outline
        show_speaker = prefs["show_speakers"] and bool(caption.speaker)
        show_language = prefs["show_language"] and bool(caption.lang)
        if show_speaker or show_language:
            font = _caption_font(BASE_SPEAKER_PX * scale, QFont.Weight.DemiBold)
            x = pad
            if show_speaker:
                path, h = _text_paths(caption.speaker, font, width, x, y)
                parts.append((path, QColor(theme.speaker_color(caption.speaker, caption.is_you)), 1.6 * scale))
                x += QFontMetricsF(font).horizontalAdvance(caption.speaker) + 8 * scale
            if show_language:
                lang_font = _caption_font(BASE_SPEAKER_PX * scale, QFont.Weight.Normal)
                path, h = _text_paths(spoken_language_name(caption.lang), lang_font, width, x, y)
                parts.append((path, QColor("#aab2c0"), 1.6 * scale))
            y += QFontMetricsF(font).height() + self.LINE_GAP * 0.5
        color = QColor("white")
        if caption.live:
            color.setAlphaF(0.85)
        text = primary + (" …" if caption.live else "")
        font = _caption_font(BASE_PRIMARY_PX * scale, QFont.Weight.Medium, primary, target if original else caption.lang)
        path, h = _text_paths(text, font, width - 2 * pad, pad, y)
        parts.append((path, color, 1.9 * scale))
        y += h
        if prefs["show_original"] and original:
            y += self.LINE_GAP
            font = _caption_font(BASE_SECONDARY_PX * scale, QFont.Weight.Medium, original, caption.lang)
            path, h = _text_paths(original, font, width - 2 * pad, pad, y)
            parts.append((path, QColor(ORIGINAL_COLOR), 1.6 * scale))
            y += h
        return parts, y + pad

    def _render(self, parts, width, height):
        """The caption's outlined text, drawn once into a pixmap."""
        ratio = self.devicePixelRatioF()
        pixmap = QPixmap(max(1, math.ceil(width * ratio)), max(1, math.ceil(height * ratio)))
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(Qt.GlobalColor.transparent)
        p = QPainter(pixmap)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for path, color, outline in parts:
            pen = QPen(QColor(0, 0, 0, 235), outline * 2)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.strokePath(path, pen)
            p.fillPath(path, color)
        p.end()
        return pixmap

    def _content_height(self):
        heights = [self.items[k].height for k in self.order]
        return sum(heights) + self.CAPTION_GAP * max(0, len(heights) - 1)

    def max_scroll(self):
        return max(0.0, self._content_height() - self.height())

    def _layout(self):
        """Where every caption should be: stacked top to bottom, the whole
        stack sitting at the bottom when it's shorter than the canvas."""
        total = self._content_height()
        if self.following:
            self.target_scroll = max(0.0, total - self.height())
        self.target_scroll = min(max(0.0, self.target_scroll), self.max_scroll())
        top = max(0.0, self.height() - total) - self.target_scroll
        for key in self.order:
            item = self.items[key]
            item.target_y = top
            if item.y is None:  # arriving: fade in, rising into place
                item.y, item.alpha = top + self.RISE, 0.0
            top += item.height + self.CAPTION_GAP
        self._start()

    # -- scrolling ----------------------------------------------------------------

    def wheelEvent(self, event):  # noqa: N802
        self.scroll_by(-event.angleDelta().y() / 120 * 60)
        event.accept()

    def scroll_by(self, dy):
        self.target_scroll = min(max(0.0, self.target_scroll + dy), self.max_scroll())
        at_bottom = self.target_scroll >= self.max_scroll() - 2
        if self.following and not at_bottom:
            self.following = False
            if self.on_scrolled_back:
                self.on_scrolled_back()
        elif not self.following and at_bottom and self.on_back_at_bottom:
            self.on_back_at_bottom()
        self._layout()

    def follow(self):
        self.following = True
        self._layout()

    # -- animation ----------------------------------------------------------------

    def _start(self):
        if not self._timer.isActive():
            self._last_frame = time.monotonic()
            self._timer.start()

    def _spring(self, y, v, target, dt):
        """One step of a critically damped spring - (y, v) after dt."""
        steps = max(1, math.ceil(dt / 0.004))  # small substeps keep it stable whatever the frame time
        h = dt / steps
        for _ in range(steps):
            v += (-self.OMEGA ** 2 * (y - target) - 2 * self.OMEGA * v) * h
            y += v * h
        return y, v

    def _animate(self):
        now = time.monotonic()
        dt = min(0.1, now - self._last_frame)
        self._last_frame = now
        moving = False
        old_scroll = self.scroll
        self.scroll, self._scroll_v = self._spring(self.scroll, getattr(self, "_scroll_v", 0.0), self.target_scroll, dt)
        if abs(self.target_scroll - self.scroll) < 0.25 and abs(self._scroll_v) < 2:
            self.scroll, self._scroll_v = self.target_scroll, 0.0
        else:
            moving = True
        fade = 1 - math.exp(-dt / self.FADE_TAU)
        for item in list(self.items.values()) + self.ghosts:
            if item.ghost:
                item.y -= self.scroll - old_scroll  # stays put on screen while the rest move
            elif item.y is not None:
                item.y, item.vy = self._spring(item.y, item.vy, item.target_y, dt)
                if abs(item.target_y - item.y) < 0.25 and abs(item.vy) < 2:
                    item.y, item.vy = item.target_y, 0.0
                else:
                    moving = True
            if abs(item.target_alpha - item.alpha) > 0.01:
                item.alpha += (item.target_alpha - item.alpha) * fade
                moving = True
            else:
                item.alpha = item.target_alpha
            if item.cross < 1.0:
                item.cross = min(1.0, item.cross + dt / self.CROSSFADE)
                moving = True
                if item.cross >= 1.0:
                    item.old_pixmap = None
        self.ghosts = [g for g in self.ghosts if g.alpha > 0.01]
        if not moving and not self.ghosts:
            self._timer.stop()
        self.overlay.update()  # the panel behind follows the captions

    # -- drawing ----------------------------------------------------------------

    def visible_extent(self):
        """(top, bottom) of what's drawn, in canvas coordinates, or None."""
        spans = [(max(0.0, i.y), min(float(self.height()), i.y + i.height))
                 for i in [*self.items.values(), *self.ghosts]
                 if i.y is not None and i.alpha > 0.02 and i.y + i.height > 0 and i.y < self.height()]
        if not spans:
            return None
        return min(s[0] for s in spans), max(s[1] for s in spans)

    FADE_EDGE = 26  # px over which captions scrolled off the top dissolve

    def paintEvent(self, event):  # noqa: N802
        # Drawn offscreen first, so the top edge can be faded without also
        # erasing the panel behind.
        ratio = self.devicePixelRatioF()
        frame = QPixmap(max(1, math.ceil(self.width() * ratio)), max(1, math.ceil(self.height() * ratio)))
        frame.setDevicePixelRatio(ratio)
        frame.fill(Qt.GlobalColor.transparent)
        p = QPainter(frame)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        cut_at_top = False
        for item in [*self.ghosts, *(self.items[k] for k in self.order)]:
            if (item.y is None or item.pixmap is None or item.y + item.height < 0 or item.y > self.height()
                    or item.alpha <= 0.01):
                continue
            # Whole pixels: a pixmap at a fractional offset is resampled - softer text.
            y = round(item.y)
            cut_at_top = cut_at_top or y < 0
            if item.old_pixmap is not None:
                p.setOpacity(item.alpha * (1 - item.cross))
                p.drawPixmap(0, y, item.old_pixmap)
                p.setOpacity(item.alpha * item.cross)
            else:
                p.setOpacity(item.alpha)
            p.drawPixmap(0, y, item.pixmap)
        if cut_at_top:
            fade = QLinearGradient(0, 0, 0, self.FADE_EDGE)
            fade.setColorAt(0, QColor(0, 0, 0, 0))
            fade.setColorAt(1, QColor(0, 0, 0, 255))
            p.setOpacity(1.0)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
            p.fillRect(QRectF(0, 0, self.width(), self.FADE_EDGE), fade)
        p.end()
        QPainter(self).drawPixmap(0, 0, frame)


class CaptionOverlay(QWidget):
    def __init__(self, controller, target, title):
        super().__init__(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint)
        self.controller = controller
        self.target = target
        self.title = title
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
        self._shown_keys: "set[int]" = set()

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

        self.canvas = CaptionCanvas(self)
        self.canvas.on_scrolled_back = self._scrolled_back
        self.canvas.on_back_at_bottom = self.go_live
        root.addWidget(self.canvas, 1)

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
        self._relayout_timer = QTimer(self, singleShot=True, interval=60, timeout=lambda: self.refresh(force=True))
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
        if self.header.isVisible() != interactive:
            self.header.setVisible(interactive)
            self._relayout_timer.start()  # the canvas changed height
        self.grip.setVisible(interactive)
        self.setCursor(Qt.CursorShape.SizeAllCursor if interactive else Qt.CursorShape.ArrowCursor)
        self._place_live_button()
        self.update()

    def _captions(self):
        return [c for c in self.controller.board.captions
                if self.target in c.translations or not c.translations][-HISTORY:]

    def refresh(self, force=False):
        """Brings the captions up to date - unless scrolled back, where it
        holds still and just counts what's new."""
        board = self.controller.board
        if not self.canvas.following:
            new = sum(1 for c in board.captions if c.key not in self._shown_keys and self.target in c.translations)
            count = f"  ·  {new} new" if new else ""
            self.live_btn.setText(f"↓  Back to live{count}")
            self._place_live_button()
            return
        version = (board.version, self.controller.prefs_version, self.canvas.width(), self.canvas.height())
        if version == self._shown_version and not force:
            return
        self._shown_version = version
        shown = self._captions()
        self._shown_keys = {c.key for c in shown}
        self.canvas.set_captions(shown, self.target, self.controller.prefs)
        newest = (shown[-1].key, shown[-1].text_for(self.target)[0]) if shown else None
        if newest != self._last_seen:
            self._last_seen = newest
            if newest is not None:
                self._last_activity = time.time()
        self.update()

    # -- scrolling ------------------------------------------------------------

    def _scrolled_back(self):
        self.live_btn.show()
        self.refresh()
        self._place_live_button()

    def go_live(self):
        if self.canvas.following and not self.live_btn.isVisible():
            return
        self.live_btn.hide()
        self.canvas.follow()
        self.refresh(force=True)

    def _place_live_button(self):
        if not self.live_btn.isVisible():
            return
        self.live_btn.adjustSize()
        area = self.canvas.geometry()
        self.live_btn.move(area.center().x() - self.live_btn.width() // 2, area.bottom() - self.live_btn.height() - 6)
        self.live_btn.raise_()

    def _tick(self):
        fade_after = self.controller.prefs["fade_after"]
        if (self._hovered and not self._locked) or not self.canvas.following:
            target = 1.0
        elif not self.canvas.order:
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
        """The panel goes behind what's actually showing - the captions in
        view (moving with them as they animate) and the header."""
        interactive = (self._hovered and not self._locked) or not self.canvas.following
        tops, bottoms = [], []
        extent = self.canvas.visible_extent()
        if extent is not None:
            offset = self.canvas.geometry().top()
            tops.append(offset + extent[0])
            bottoms.append(offset + extent[1])
        elif interactive:
            area = self.canvas.geometry()
            tops.append(area.center().y() - 12)
            bottoms.append(area.center().y() + 12)
        if self.header.isVisible():
            tops.append(self.header.geometry().top())
            bottoms.append(self.header.geometry().bottom())
        if not tops:
            return
        top, bottom = min(tops), max(bottoms)
        panel = QRectF(0, top - PAD, self.width(), bottom - top + 2 * PAD)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        opacity = self.controller.prefs["opacity"]
        if interactive:
            opacity = max(opacity, 0.55)
            p.setPen(QPen(QColor(255, 255, 255, 40), 1))
        else:
            p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(10, 12, 16, int(255 * opacity)))
        p.drawRoundedRect(panel.adjusted(1, 1, -1, -1), 14, 14)
        if extent is None and interactive:
            p.setPen(QColor(theme.TEXT_DIM))
            p.drawText(self.canvas.geometry(), Qt.AlignmentFlag.AlignCenter, f"{self.title} captions appear here")

    def resizeEvent(self, event):  # noqa: N802
        self.grip.move(self.width() - self.grip.width() - 4, self.height() - self.grip.height() - 4)
        self._place_live_button()
        self._relayout_timer.start()  # rewrap the text once the resize settles
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
