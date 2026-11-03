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

Unlocked, words can be looked up too, in any line: hover one - a Chinese
word for its pinyin and meaning, an English one for the Chinese for it -
click to pin the lookup, or drag from it across a longer phrase to look
that up whole (see ui/lookup.py). Words you've saved are drawn in their
category's colour (see vocab.py).
"""

import math
import time
import unicodedata

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import (QColor, QFont, QFontMetricsF, QLinearGradient, QPainter, QPainterPath, QPainterPathStroker,
                           QPen, QPixmap, QTextLayout, QTextLine, QTextOption)
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizeGrip, QSizePolicy, QVBoxLayout, QWidget

import dictionary
from lang_codes import spoken_language_name
from ui import theme
from ui.lookup import context_for

PAD = 14
FADE_STEP = 0.08          # per 30ms tick - about a third of a second to fade
HISTORY = 150             # captions an overlay keeps to scroll back through
# One colour rule everywhere: white is what was said, in its own language;
# yellow, under it, is always a translation. Both the same size - the
# yellow line is the one most readers need, so it isn't demoted.
BASE_TEXT_PX = 21
BASE_SPEAKER_PX = 13
TRANSLATION_COLOR = "#ffd76e"   # classic subtitle yellow
# Word links (docs/word-links.md): a clause's words and the words of its
# translation that say the same thing, underlined in the same colour -
# and, as arrows, joined by a curve. Not text colour: white/yellow and
# saved words' colours already mean something.
LINK_COLORS = ["#ff9f43", "#b388ff", "#4dd0e1", "#ff6e9c", "#a5d66f",   # orange, violet, cyan, pink, green
               "#5c9dff", "#ff5252", "#3ddc97", "#e040fb", "#c8925a"]   # blue, red, mint, magenta, copper -
# - 10, so a pair's links rarely share one (they cycle past that); neighbours far apart in hue
# At most this many links per said/translated pair - the surest kept, see
# caption_lines - or None for all of them: for now, to see everything the
# model links. (Was 6 for key words and arrows, 10 for most words - but a
# cap kept the first in the line, so a plain teach - 教 could be crowded out.)
MAX_LINKS = None
ARROW_GAP = 18            # px between a clause and its translation, for the links to cross
ARROW_CLAUSE_GAP = 28     # px between pairs then - clearly more, so each pair still reads as one
_SELECTION_COLOR = QColor(124, 156, 255, 105)   # a phrase being dragged across - theme.ACCENT, see-through


def caption_lines(caption, target, second, prefs):
    """[(text, colour, language, starts a clause, links)] for a caption in
    `target`'s overlay - a clause at a time (see clauses.py), so a long
    caption reads line by line and a clause whose words didn't change
    keeps its rows as they were: what was said (white) - left out if the
    overlay hides originals and there's a translation - then its
    translation (yellow), into the target or, for a clause already in it,
    into the overlay's second language. A live caption's last line trails
    off. links: ((start, end, link number), ...) - the row's words linked
    to words of the other row of its pair (prefs["word_links"]), on
    clauses that have settled: not a live caption's last, whose words -
    and links - are still changing."""
    white = QColor("white")
    if caption.live:
        white.setAlphaF(0.85)
    linking = prefs.get("word_links", "off") in ("hover", "colours", "arrows") and prefs["show_original"]
    detail = prefs.get("link_detail", "most")
    cap = MAX_LINKS
    clauses = caption.clauses or ((caption.native, None),)
    lines = []
    for n, ((native, _), (said, translation, in_target)) in enumerate(zip(clauses, caption.clause_lines(target,
                                                                                                         second))):
        said_links = translated_links = ()
        settled = not (caption.live and n == len(clauses) - 1)
        if linking and translation is not None and settled and n < len(caption.clause_links) \
                and len(said) == len(native):  # a script conversion keeps the positions
            # One link - one colour - per counterpart: a word said twice
            # for the same thing (佛印 ... 佛印 - "Foyin") shares it. With
            # more than `cap`, the surest stay - key words first - so a
            # plain match like teach - 教 isn't crowded out by "some" - 一下
            # for coming later in the line; then they're numbered (coloured)
            # in reading order.
            groups = {}  # counterpart span -> [its words' spans, key, surety]
            for link in caption.clause_links[n].get(target if in_target else second) or ():
                key = link[2] if len(link) > 2 else True
                if detail == "key" and not key:
                    continue  # a pronoun, 是, 这个... - only at "most words"
                group = groups.setdefault(tuple(link[1]), [[], False, 0.0])
                group[0].append(tuple(link[0]))
                group[1] = group[1] or key
                group[2] = max(group[2], link[3] if len(link) > 3 else 1.0)
            kept = sorted(groups.items(), key=lambda g: (not g[1][1], -g[1][2]))[:cap]  # cap None: all
            kept.sort(key=lambda g: min(g[1][0]))  # reading order, by the first word said
            said_links = tuple(sorted((a, b, k) for k, (_, (spans, _, _)) in enumerate(kept) for a, b in spans))
            translated_links = tuple(sorted((c, d, k) for k, ((c, d), _) in enumerate(kept)))
        first = True
        if translation is None or prefs["show_original"]:
            # said only differs from what was said when it's been put in the target's script (zh -> zh-hant)
            lines.append([said, white, target if said != native else caption.lang, first, said_links])
            first = False
        if translation is not None:
            lines.append([translation, QColor(TRANSLATION_COLOR), target if in_target else second, first,
                          translated_links])
    if caption.live:
        lines[-1][0] += " \u2026"
    return [tuple(line) for line in lines]


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


# Chinese/Japanese next to Latin text at the same pixel size: in its own
# script's font (YaHei, JhengHei, Yu Gothic - below) it already reads about
# as large; a touch more keeps its denser strokes as easy to read without
# it towering over the English line beside it.
CJK_SCALE = 1.05
CJK_FAMILIES = {
    "simplified": ["Microsoft YaHei UI", "Microsoft YaHei"],
    "traditional": ["Microsoft JhengHei UI", "Microsoft JhengHei"],
    "japanese": ["Yu Gothic UI", "Meiryo UI"],
}
SCRIPT_OF_LANGUAGE = {"zh": "simplified", "zh-hant": "traditional", "yue": "traditional", "ja": "japanese"}


def _wordlike(char):
    """Not a space or punctuation - what a dragged-out phrase is trimmed to."""
    return not char.isspace() and not unicodedata.category(char).startswith("P") and char != "…"


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
def spoken_languages_label(caption):
    """"Mandarin", or "Mandarin + English" for a line mixing languages."""
    return " + ".join(spoken_language_name(code) for code in (caption.langs or (caption.lang,)))


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
        for part in (self.head, self.primary, self.secondary):  # primary: white (said), secondary: yellow
            layout.addWidget(part)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def show_caption(self, caption, target, prefs, second=None):
        scale = prefs["text_scale"]

        show_speaker = prefs["show_speakers"] and bool(caption.speaker)
        show_language = prefs["show_language"] and bool(caption.lang)
        self.head.setVisible(show_speaker or show_language)
        self.speaker.setVisible(show_speaker)
        if show_speaker:
            self.speaker.set(caption.speaker, _caption_font(BASE_SPEAKER_PX * scale, QFont.Weight.DemiBold),
                             theme.speaker_color(caption.speaker, caption.is_you), 1.6 * scale)
        self.language.setVisible(show_language)
        if show_language:
            self.language.set(spoken_languages_label(caption), _caption_font(BASE_SPEAKER_PX * scale,
                                                                                QFont.Weight.Normal),
                              "#aab2c0", 1.6 * scale)

        lines = caption_lines(caption, target, second, prefs)
        for widget, line in zip((self.primary, self.secondary), [*lines, None]):
            widget.setVisible(line is not None)
            if line is not None:
                text, color, lang, *_ = line
                widget.set(text, _caption_font(BASE_TEXT_PX * scale, QFont.Weight.Medium, text, lang), color,
                           1.8 * scale)


def _text_layout(text, font, width):
    """(layout, height): `text` word-wrapped at `width`."""
    layout = QTextLayout(text, font)
    option = QTextOption()
    option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
    layout.setTextOption(option)
    layout.beginLayout()
    h = 0.0
    while True:
        line = layout.createLine()
        if not line.isValid():
            break
        line.setLineWidth(max(10.0, width))
        line.setPosition(QPointF(0, h))
        h += line.height()
    layout.endLayout()
    return layout, h


def _text_paths(text, font, width, x=0.0, y=0.0):
    """(path, height): `text` word-wrapped at `width` as one QPainterPath,
    its top-left at (x, y)."""
    paths, _, h = _caption_text(text, font, width, x, y, None)
    return (paths[0][0] if paths else QPainterPath()), h


def _caption_text(text, font, width, x, y, color, spans=()):
    """([(path, colour)], layout, height): `text` word-wrapped at `width`,
    its top-left at (x, y) - in `color`, except the (start, end, colour)
    spans, each drawn in its own. The layout is kept to find the character
    under the mouse."""
    layout, h = _text_layout(text, font, width)
    paths = {}  # colour -> (path, colour)

    def add(start, end, run_color, line):
        segment = text[start:end]
        if not segment.strip():
            return
        dx = line.cursorToX(start)[0] if start > line.textStart() else 0.0
        key = run_color.name(QColor.NameFormat.HexArgb) if run_color is not None else None
        if key not in paths:
            paths[key] = (QPainterPath(), run_color)
        paths[key][0].addText(QPointF(x + dx, y + line.y() + line.ascent()), font, segment)

    for i in range(layout.lineCount()):
        line = layout.lineAt(i)
        start = line.textStart()
        end = start + len(text[start:start + line.textLength()].rstrip())
        at = start
        for a, b, run_color in spans:
            a, b = max(a, start), min(b, end)
            if a < b:
                add(at, a, color, line)
                add(a, b, run_color, line)
                at = b
        add(at, end, color, line)
    return list(paths.values()), layout, h


_underline_drops = {}


def _underline_drop(font):
    """How far below the baseline `font`'s marks under letters reach - the
    dot of Vietnamese's heavy tone (ạ, ọ, ệ) - so a link's underline goes
    under them rather than through them."""
    key = font.key()
    if key not in _underline_drops:
        path = QPainterPath()
        path.addText(0, 0, font, "ạẹịọụậệộợự")  # no descenders (g, y): underlines cross those, as usual
        _underline_drops[key] = max(0.0, path.boundingRect().bottom())
    return _underline_drops[key]


def _first_rewritten(old, new):
    """The index of the first row of `new` that rewrites its row in `old`,
    rather than only growing it (a live caption's last row) - len(new) if
    only rows at the end went - or None if nothing was rewritten."""
    for i, (a, b) in enumerate(zip(old, new)):
        if a != b and not (i == len(old) - 1 and b.startswith(a.removesuffix(" \u2026"))):
            return i
    return len(new) if len(new) < len(old) else None


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
        self.cross_from = 0      # where the crossfade starts - rows above it weren't rewritten
        self.ghost = False       # removed - fading out where it was, taking no space
        self.lines = []          # [(text, layout, x, y)] its caption lines, to find the word under the mouse
        self.live = False        # its caption is still being spoken
        self.links = []          # per caption line: its word links, ((start, end, link number), ...)
        self.clause_starts = []  # per caption line: whether it starts a clause's pair


class CaptionCanvas(QWidget):
    """Draws an overlay's captions and animates every change: captions ease
    to new positions when anything changes height (a new caption, a live
    caption growing a line, a join), arrive fading and rising into place,
    leave fading out while the rest close the gap, and crossfade when their
    text is rewritten (text that only grows just extends). The wheel
    scrolls smoothly too. One painted surface rather than a widget per
    caption - widgets resize in single frames, which is what read as
    flicker.

    Following the live end, captions only ever move up - the transcript
    reads in one direction (see _layout)."""

    LINE_GAP = 4         # between a caption's speaker line, translation and original
    CLAUSE_GAP = 9       # before each clause's rows after the first - each said/translated pair reads as one
    CAPTION_GAP = 14     # between captions
    # Positions move on a critically damped spring: they start from rest,
    # accelerate, and settle without overshoot - and keep their momentum
    # when the target changes mid-move (a live caption updating while
    # things are still sliding), so there's never a jerk. ~0.3s to settle.
    OMEGA = 20.0         # rad/s - the spring's stiffness
    FADE_TAU = 0.07      # seconds - fade in/out time constant
    CROSSFADE = 0.2      # seconds
    RISE = 14            # px a new caption rises as it fades in
    # A live caption at the bottom that has to push the rest up pushes this
    # many lines further, and grows into them - the transcript steps up now
    # and then rather than with every update.
    HEADROOM_LINES = 2

    def __init__(self, overlay):
        super().__init__(overlay)
        self.overlay = overlay
        self.items: "dict[int, _Item]" = {}
        self.order: "list[int]" = []      # keys top to bottom (ghosts excluded)
        self.ghosts: "list[_Item]" = []
        self.scroll = 0.0                 # content scrolled off the top, animated toward target_scroll
        self.target_scroll = 0.0
        self.following = True
        self._reanchor = True             # next layout: settle at the bottom afresh (see _layout)
        self._scroll_hi = 0.0             # scrolled back from below max_scroll() - space left at the bottom
        self._anchor_key = None           # (width, height, prefs) the layout was last made for
        self._line_height = 28.0          # a caption line's, at the current text size
        self.on_scrolled_back = None      # set by the overlay
        self.on_back_at_bottom = None
        self._last_frame = time.monotonic()
        self._timer = QTimer(self, interval=8, timeout=self._animate)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)  # the default ticks unevenly on Windows
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMouseTracking(True)       # hovering a word looks it up
        self.hovered = None               # (key, line index, start, end) of the word under the mouse
        self._selecting = None            # (key, line index, first character, word) from a press on a word
        self.selection = None             # (key, line index, start, end) of a phrase being dragged across

    # -- content ----------------------------------------------------------------

    def set_captions(self, captions, target, prefs, second=None):
        """Brings the canvas up to date with `captions` (top to bottom)."""
        width = self.width()
        anchor_key = (width, self.height(), tuple(sorted(prefs.items())))
        if anchor_key != self._anchor_key:  # resized or restyled - everything reflows anyway
            self._anchor_key, self._reanchor = anchor_key, True
            self._line_height = QFontMetricsF(_caption_font(BASE_TEXT_PX * prefs["text_scale"],
                                                            QFont.Weight.Medium)).height()
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
            item.live = caption.live
            lines = caption_lines(caption, target, second, prefs)
            signature = (caption.speaker, caption.lang, caption.langs, caption.live,
                         tuple((text, color.name(QColor.NameFormat.HexArgb), clause_start, links)
                               for text, color, _, clause_start, links in lines), width,
                         tuple(sorted(prefs.items())), self.overlay.controller.vocab.version)
            if signature != item.signature:
                old_rows = [text for text, *_ in item.lines] if item.signature is not None else None
                item.signature, item.text = signature, " ".join(text for text, *_ in lines)
                old_pixmap = item.pixmap
                parts, item.lines, item.height = self._build(caption, lines, prefs, width)
                item.links = [line[4] for line in lines]
                item.clause_starts = [line[3] for line in lines]
                item.pixmap = self._render(parts, width, item.height)
                rewritten = _first_rewritten(old_rows, [text for text, *_ in item.lines]) if old_rows else None
                if rewritten is not None:
                    # Only from the first rewritten row down - the rows above
                    # are unchanged, and fading them through themselves would
                    # dim them for a moment.
                    item.old_pixmap, item.cross = old_pixmap, 0.0
                    item.cross_from = (round(item.lines[rewritten][3] - self.LINE_GAP / 2)
                                       if rewritten < len(item.lines) else math.ceil(item.height))
                if self.hovered and self.hovered[0] == caption.key:
                    self.hovered = None  # its text moved - found again on the next mouse move
        self.order = keys
        self._layout()

    def _build(self, caption, lines, prefs, width):
        scale = prefs["text_scale"]
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
                path, h = _text_paths(spoken_languages_label(caption), lang_font, width, x, y)
                parts.append((path, QColor("#aab2c0"), 1.6 * scale))
            y += QFontMetricsF(font).height() + self.LINE_GAP * 0.5
        vocab = self.overlay.controller.vocab
        text_lines = []
        arrows = prefs.get("word_links", "off") == "arrows" and prefs["show_original"]
        above = {}  # link number -> where its curves start: under its words on the row above's last line
        for i, (text, color, lang, clause_start, links) in enumerate(lines):
            if i:
                # Arrows: every pair gets the gap, links or not, so a row
                # doesn't jump when its links arrive.
                if clause_start:
                    y += ARROW_CLAUSE_GAP * scale if arrows else self.CLAUSE_GAP
                else:
                    y += ARROW_GAP * scale if arrows else self.LINE_GAP
            if clause_start:
                above = {}
            font = _caption_font(BASE_TEXT_PX * scale, QFont.Weight.Medium, text, lang)
            spans = []
            for start, end, saved_color in vocab.spans(text):  # saved words, in their category's colour
                saved_color = QColor(saved_color)
                saved_color.setAlphaF(color.alphaF())
                spans.append((start, end, saved_color))
            paths, layout, h = _caption_text(text, font, width - 2 * pad, pad, y, color, spans)
            parts += [(path, path_color, 1.8 * scale) for path, path_color in paths]
            text_lines.append((text, layout, pad, y))
            if links and prefs.get("word_links") in ("colours", "arrows"):  # on hover: drawn only when pointed at
                parts += self._link_parts(layout, pad, y, links, scale, arrows, clause_start, above)
            y += h
        return parts, text_lines, y + pad

    @staticmethod
    def _link_parts(layout, x, y, links, scale, arrows, upper, above):
        """Paths for a row's links: each linked word underlined in its
        link's colour; and with arrows, from the lower row of a pair, a
        curve from where the word above starts (`above`, filled in by the
        upper row) down to its counterpart here. Only words on the rows'
        facing lines get a curve - one from an upper line would cut through
        the text; the rest are underlined alone."""
        parts = []
        last = layout.lineCount() - 1
        for start, end, number in links:
            color = QColor(LINK_COLORS[number % len(LINK_COLORS)])
            for i in range(layout.lineCount()):
                line = layout.lineAt(i)
                a, b = max(start, line.textStart()), min(end, line.textStart() + line.textLength())
                if a >= b:
                    continue
                x1, x2 = x + line.cursorToX(a)[0], x + line.cursorToX(b)[0]
                base = y + line.y() + line.ascent()
                drop = _underline_drop(layout.font()) + 1.5 * scale
                underline = QPainterPath()
                underline.addRoundedRect(QRectF(x1, base + drop, x2 - x1, 2.2 * scale), 1.1 * scale, 1.1 * scale)
                parts.append((underline, color, 1.2 * scale))
                if not arrows:
                    continue
                middle = (x1 + x2) / 2
                if upper and i == last:
                    above.setdefault(number, []).append(QPointF(middle, base + drop + 3 * scale))
                elif not upper and i == 0:
                    end_at = QPointF(middle, y + line.y() + 1 * scale)
                    for start_at in above.get(number, ()):
                        bend = (end_at.y() - start_at.y()) * 0.6
                        curve = QPainterPath(start_at)
                        curve.cubicTo(QPointF(start_at.x(), start_at.y() + bend),
                                      QPointF(end_at.x(), end_at.y() - bend), end_at)
                        stroker = QPainterPathStroker()
                        stroker.setWidth(1.7 * scale)
                        stroker.setCapStyle(Qt.PenCapStyle.RoundCap)
                        faint = QColor(color)
                        faint.setAlphaF(0.9)
                        parts.append((stroker.createStroke(curve), faint, 1.0 * scale))
        return parts

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
        stack sitting at the bottom when it's shorter than the canvas.
        Following the live end (see _following_top), captions only move up."""
        total = self._content_height()
        if self.following:
            top = self._following_top(total)
            self.target_scroll = max(0.0, self.height() - total) - top
        else:
            self.target_scroll = min(max(0.0, self.target_scroll), max(self.max_scroll(), self._scroll_hi))
            top = max(0.0, self.height() - total) - self.target_scroll
        self._reanchor = False
        for key in self.order:
            item = self.items[key]
            item.target_y = top
            if item.y is None:  # arriving: fade in, rising into place
                item.y, item.alpha = top + self.RISE, 0.0
            top += item.height + self.CAPTION_GAP
        self._start()

    def _following_top(self, total):
        """Where the stack's top goes, following the live end: as low as it
        can be while (a) it fits on the canvas and (b) no caption already
        placed moves down. So a caption shrinking - the final pass merging
        rows, a duplicate taken back - doesn't pull the ones above down:
        those below it move up, and blank space is left at the bottom. A
        caption growing pushes the ones above up - or, at the bottom, first
        grows into that space. A live caption at the bottom that has to
        push leaves HEADROOM_LINES spare to grow into. Settles at the
        bottom afresh after a resize, a restyle, or scrolling back."""
        bottom = self.height() - total  # the stack's top when it sits at the bottom
        if self._reanchor or not self.order:
            return bottom
        placed, offset = math.inf, 0.0  # the lowest top that moves no placed caption down
        for key in self.order:
            item = self.items[key]
            if item.y is not None:  # placed before - arrivals have no place yet
                placed = min(placed, item.target_y - offset)
            offset += item.height + self.CAPTION_GAP
        if placed <= bottom:
            return placed  # fits - with whatever space was left below it
        if self.items[self.order[-1]].live:
            return bottom - self.HEADROOM_LINES * (self._line_height + self.LINE_GAP)
        return bottom

    # -- scrolling ----------------------------------------------------------------

    def wheelEvent(self, event):  # noqa: N802
        self.scroll_by(-event.angleDelta().y() / 120 * 60)
        event.accept()

    def scroll_by(self, dy):
        """Scrolls `dy` px (up < 0) from where the captions are now. At the
        live end that can be below the bottom-aligned position - the one-way
        layout leaves space at the bottom (see _following_top) - so that's
        where scrolling back starts from, not max_scroll()."""
        if self._content_height() <= self.height():
            # It all fits - nothing to scroll back to. (A caption that shrank
            # may have left the oldest partly off the top: this brings it back.)
            if self.following:
                self._reanchor = True
                self._layout()
            return
        current = self.target_scroll
        target = min(max(0.0, current + dy), max(self.max_scroll(), self._scroll_hi if not self.following else current))
        if self.following:
            if target >= current - 0.5:
                return  # already at the live end
            self.following, self._scroll_hi = False, current
            self.target_scroll = target
            if self.on_scrolled_back:
                self.on_scrolled_back()
        else:
            self.target_scroll = target
            if dy > 0 and target >= self.max_scroll() - 2 and self.on_back_at_bottom:  # scrolled down to it
                self.on_back_at_bottom()
                return
        self._layout()

    def follow(self):
        self.following, self._reanchor, self._scroll_hi = True, True, 0.0
        self._layout()

    # -- word lookup ----------------------------------------------------------------

    def _word_at(self, pos):
        """(item, line index, start, end) of the word under `pos` - a
        dictionary word in Chinese, a run of letters in other languages - or
        None."""
        for key in self.order:
            item = self.items[key]
            if item.y is None or item.alpha < 0.5:
                continue
            top = round(item.y)
            if not top <= pos.y() < top + item.height:
                continue
            for n, (text, layout, x, y) in enumerate(item.lines):
                for i in range(layout.lineCount()):
                    line = layout.lineAt(i)
                    if line.naturalTextRect().translated(x, top + y).contains(pos):
                        index = line.xToCursor(pos.x() - x, QTextLine.CursorPosition.CursorOnCharacter)
                        span = dictionary.word_span_at(text, index)
                        return (item, n, *span) if span else None
            return None
        return None

    def _word_rects(self, item, n, start, end):
        """Where characters start-end of the item's n-th line are drawn, in
        canvas coordinates - more than one rect if the word wraps."""
        _, layout, x, y = item.lines[n]
        rects = []
        for i in range(layout.lineCount()):
            line = layout.lineAt(i)
            a, b = max(start, line.textStart()), min(end, line.textStart() + line.textLength())
            if a < b:
                x1, x2 = line.cursorToX(a)[0], line.cursorToX(b)[0]
                rects.append(QRectF(x + x1, round(item.y) + y + line.y(), x2 - x1, line.height()))
        return rects

    def _link_partners(self, item, n, start, end):
        """[(line index, start, end, link number)]: the word links touching
        characters start-end of the item's n-th line - on that line and on
        the other line of its pair - or [] if there are none (or word links
        are off)."""
        if self.overlay.controller.prefs.get("word_links", "off") == "off" or n >= len(item.links):
            return []
        numbers = {number for a, b, number in item.links[n] if a < end and start < b}
        if not numbers:
            return []
        # A pair is a line starting a clause and the one after it
        other = n + 1 if item.clause_starts[n] else n - 1
        lines = [n] + ([other] if 0 <= other < len(item.links) and item.clause_starts[other] != item.clause_starts[n]
                       else [])
        return [(m, a, b, number) for m in lines for a, b, number in item.links[m] if number in numbers]

    def _char_at(self, item, n, pos):
        """The character of the item's n-th line nearest `pos` - dragging
        across a phrase needn't stay exactly on the text."""
        text, layout, x, y = item.lines[n]
        top = round(item.y) + y
        lines = [layout.lineAt(i) for i in range(layout.lineCount())]
        line = next((l for l in lines if pos.y() < top + l.y() + l.height()), lines[-1])
        index = line.xToCursor(pos.x() - x, QTextLine.CursorPosition.CursorOnCharacter)
        return min(max(index, 0), len(text) - 1)

    def _lookup_for(self, item, n, start, end):
        """(text, context, anchor on screen) for characters start-end of the
        item's n-th line - None if its caption has gone."""
        caption = next((c for c in self.overlay.shown if c.key == item.key), None)
        if caption is None:
            return None
        text = item.lines[n][0]
        rects = self._word_rects(item, n, start, end)
        box = rects[0]
        for rect in rects[1:]:
            box = box.united(rect)
        anchor = QRect(self.mapToGlobal(box.topLeft().toPoint()), box.size().toSize())
        return text[start:end], context_for(caption, text), anchor

    def clear_hover(self):
        if self.hovered is not None:
            self.hovered = None
            self.unsetCursor()
            self.update()
        self.overlay.controller.lookup.unhover()

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._selecting is not None:
            self._drag_select(event.position())
            return
        if event.buttons():
            event.ignore()
            return
        hit = self._word_at(event.position())
        lookup = self._lookup_for(*hit) if hit else None
        if lookup is None:
            self.clear_hover()
            return
        hovered = (hit[0].key, *hit[1:])
        if hovered != self.hovered:
            self.hovered = hovered
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.update()
        self.overlay.controller.lookup.hover(*lookup)

    def mousePressEvent(self, event):  # noqa: N802
        """On a word: a click looks it up, a drag selects a phrase from it
        (see mouseReleaseEvent). Anywhere else: the overlay's drag."""
        hit = self._word_at(event.position()) if event.button() == Qt.MouseButton.LeftButton else None
        if hit is None:
            event.ignore()
            return
        item, n = hit[0], hit[1]
        self._selecting = (item.key, n, self._char_at(item, n, event.position()), hit)
        self.clear_hover()
        event.accept()

    def _drag_select(self, pos):
        key, n, anchor, _ = self._selecting
        item = self.items.get(key)
        if item is None or n >= len(item.lines):
            return
        text = item.lines[n][0]
        current = self._char_at(item, n, pos)
        low, high = sorted((anchor, current))
        start, end = dictionary.snap_to_words(text, low, high + 1)  # English by the whole word
        if self.selection is None and (start, end) == dictionary.snap_to_words(text, anchor, anchor + 1):
            return  # not off the first character (or English word) yet - still a click
        self.selection = (key, n, start, end)
        self.setCursor(Qt.CursorShape.IBeamCursor)
        self.update()

    def mouseReleaseEvent(self, event):  # noqa: N802
        """Pins the lookup of the phrase dragged across - its ends trimmed of
        spaces and punctuation - or, after a plain click, the word."""
        if self._selecting is None:
            event.ignore()
            return
        target = self._selecting[3]
        selection, self._selecting, self.selection = self.selection, None, None
        self.unsetCursor()
        self.update()
        item = self.items.get(selection[0]) if selection else None
        if item is not None and selection[1] < len(item.lines):
            _, n, start, end = selection
            text = item.lines[n][0]
            while start < end and not _wordlike(text[start]):
                start += 1
            while end > start and not _wordlike(text[end - 1]):
                end -= 1
            if start < end:
                target = (item, n, start, end)
        lookup = self._lookup_for(*target)
        if lookup is not None:
            self.overlay.controller.lookup.pin(*lookup)
        event.accept()

    def leaveEvent(self, event):  # noqa: N802
        self.clear_hover()

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
            for marked, color in ((self.hovered, QColor(255, 255, 255, 46)), (self.selection, _SELECTION_COLOR)):
                if marked and marked[0] == item.key and not item.ghost and marked[1] < len(item.lines):
                    p.setOpacity(item.alpha)
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(color)
                    for rect in self._word_rects(item, *marked[1:]):
                        p.drawRoundedRect(rect.adjusted(-3, 0, 3, 0), 5, 5)
            if self.hovered and self.hovered[0] == item.key and not item.ghost:
                # The hovered word's links: it and its counterpart in the
                # other line of the pair, tinted in the link's colour.
                p.setOpacity(item.alpha)
                p.setPen(Qt.PenStyle.NoPen)
                for n, start, end, number in self._link_partners(item, *self.hovered[1:]):
                    tint = QColor(LINK_COLORS[number % len(LINK_COLORS)])
                    tint.setAlphaF(0.4)
                    p.setBrush(tint)
                    for rect in self._word_rects(item, n, start, end):
                        p.drawRoundedRect(rect.adjusted(-3, 0, 3, 0), 5, 5)
            if item.old_pixmap is not None:
                # Rows above cross_from as they are; the rest crossfades.
                split = y + item.cross_from
                p.save()
                p.setClipRect(QRect(0, y - 8, self.width(), split - y + 8), Qt.ClipOperation.IntersectClip)
                p.setOpacity(item.alpha)
                p.drawPixmap(0, y, item.pixmap)
                p.restore()
                p.save()
                p.setClipRect(QRect(0, split, self.width(), max(0, self.height() - split + 8)),
                              Qt.ClipOperation.IntersectClip)
                p.setOpacity(item.alpha * (1 - item.cross))
                p.drawPixmap(0, y, item.old_pixmap)
                p.setOpacity(item.alpha * item.cross)
                p.drawPixmap(0, y, item.pixmap)
                p.restore()
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
        self.shown = []           # the captions on the canvas, as they were when drawn

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
        head.addWidget(self._header_button("", "Assistant - chat about the call, summarise it, help replying. "
                                           "Opening it switches the local assistant on, closing it off",
                                           lambda: controller.toggle_chat(self)))
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
            self.canvas.hovered = None
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
        self.shown = shown
        self._shown_keys = {c.key for c in shown}
        self.canvas.set_captions(shown, self.target, self.controller.prefs,
                                 self.controller.second_languages.get(self.target))
        newest = (shown[-1].key, shown[-1].text_for(self.target)[0]) if shown else None
        if newest != self._last_seen:
            self._last_seen = newest
            if newest is not None:
                self._last_activity = time.time()
        self.update()

    def recolor(self):
        """Redraws the captions showing - a saved word or its colour changed
        - without catching up, if scrolled back."""
        self.canvas.set_captions(self.shown, self.target, self.controller.prefs,
                                 self.controller.second_languages.get(self.target))
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
        self.controller.lookup.close_lookup()  # clicked off a word - usually closed already, by losing focus
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_from = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._drag_from is not None:
            self.move(event.globalPosition().toPoint() - self._drag_from)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if self._drag_from is not None:
            self._drag_from = None
            self.controller.save_geometry(self.target, self.geometry())
        self.controller.lookup.bring_to_front()  # the click brought this overlay up over it


def default_geometry(screen_rect, index, count):
    """Where the index-th overlay goes before it's been moved: centred
    near the bottom of the screen, later ones stacked above."""
    width = min(960, int(screen_rect.width() * 0.5))
    height = 260
    x = screen_rect.x() + (screen_rect.width() - width) // 2
    y = screen_rect.y() + screen_rect.height() - 90 - height - index * (height + 8)
    return QRect(QPoint(x, max(screen_rect.y(), y)), QPoint(x + width - 1, max(screen_rect.y(), y) + height - 1))
