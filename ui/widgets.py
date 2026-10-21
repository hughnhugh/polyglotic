"""Small reusable widgets: cards, setting rows, a toggle switch, segmented
choices, chips, status dots and the app icon."""

from PySide6.QtCore import Property, QEasingCurve, QPropertyAnimation, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractButton, QButtonGroup, QFrame, QHBoxLayout, QLabel, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

from ui import theme


def label(text="", name=None, wrap=False, selectable=False):
    lbl = QLabel(text)
    if name:
        lbl.setObjectName(name)
    lbl.setWordWrap(wrap)
    if selectable:
        lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return lbl


def glyph(code, size=16, color=None):
    """An icon from Windows' icon font, as a label."""
    lbl = QLabel(code)
    lbl.setFixedWidth(size + 6)
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setStyleSheet(theme.font_css([theme.icon_family()], size) + (f" color: {color};" if color else "")
                      + " background: transparent;")
    return lbl


def button(text, kind=None, on_click=None):
    btn = QPushButton(text)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    if kind:
        btn.setProperty("kind", kind)
    if on_click:
        btn.clicked.connect(on_click)
    return btn


class Card(QFrame):
    """A rounded surface grouping related content, optionally titled."""

    def __init__(self, title=None, subtitle=None, spacing=12):
        super().__init__()
        self.setObjectName("Card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(20, 18, 20, 18)
        self.body.setSpacing(spacing)
        if title:
            head = QVBoxLayout()
            head.setSpacing(3)
            head.addWidget(label(title, "CardTitle"))
            if subtitle:
                head.addWidget(label(subtitle, "Dim", wrap=True))
            self.body.addLayout(head)

    def add(self, widget_or_layout):
        if isinstance(widget_or_layout, QWidget):
            self.body.addWidget(widget_or_layout)
        else:
            self.body.addLayout(widget_or_layout)
        return widget_or_layout


class SettingRow(QWidget):
    """Title and description on the left, the control on the right."""

    def __init__(self, title, description=None, control=None):
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(16)
        text = QVBoxLayout()
        text.setSpacing(2)
        text.addWidget(label(title))
        if description:
            self.description = label(description, "Faint", wrap=True)
            text.addWidget(self.description)
        row.addLayout(text, 1)
        if control is not None:
            row.addWidget(control, 0, Qt.AlignmentFlag.AlignVCenter)


class ToggleSwitch(QAbstractButton):
    """An animated on/off switch."""

    def __init__(self, checked=False):
        super().__init__()
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(42, 24)
        self._offset = 1.0 if checked else 0.0
        self._anim = QPropertyAnimation(self, b"offset", self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.toggled.connect(self._animate)

    def _animate(self, on):
        self._anim.stop()
        self._anim.setStartValue(self._offset)
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()

    def setChecked(self, on):  # noqa: N802 - Qt naming
        blocked = self.blockSignals(True)
        super().setChecked(on)
        self.blockSignals(blocked)
        self._offset = 1.0 if on else 0.0
        self.update()

    def get_offset(self):
        return self._offset

    def set_offset(self, value):
        self._offset = value
        self.update()

    offset = Property(float, get_offset, set_offset)

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        track = QColor(theme.SURFACE_3).toRgb()
        on = QColor(theme.ACCENT)
        t = self._offset
        color = QColor(int(track.red() + (on.red() - track.red()) * t), int(track.green() + (on.green() - track.green()) * t),
                       int(track.blue() + (on.blue() - track.blue()) * t))
        if not self.isEnabled():
            color.setAlpha(110)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawRoundedRect(QRectF(0, 0, self.width(), self.height()), 12, 12)
        p.setBrush(QColor(theme.ACCENT_TEXT if t > 0.5 else theme.TEXT))
        x = 3 + t * (self.width() - self.height())
        p.drawEllipse(QRectF(x, 3, self.height() - 6, self.height() - 6))

    def sizeHint(self):  # noqa: N802
        return QSize(42, 24)


class Segmented(QWidget):
    """A row of mutually exclusive choices - for a handful of options that
    should all stay visible, rather than hidden in a dropdown."""

    changed = Signal(object)

    def __init__(self, options, value=None):
        """options: [(value, label)]."""
        super().__init__()
        self._values = [v for v, _ in options]
        row = QHBoxLayout(self)
        row.setContentsMargins(3, 3, 3, 3)
        row.setSpacing(2)
        self.setObjectName("Segmented")
        self.setStyleSheet(f"""
            #Segmented {{ background: {theme.SURFACE_2}; border: 1px solid {theme.BORDER}; border-radius: 9px; }}
            #Segmented QPushButton {{ background: transparent; border: none; border-radius: 6px; padding: 6px 12px;
                                      color: {theme.TEXT_DIM}; }}
            #Segmented QPushButton:hover {{ color: {theme.TEXT}; }}
            #Segmented QPushButton:checked {{ background: {theme.SURFACE_3}; color: {theme.TEXT}; }}
        """)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        for i, (_, text) in enumerate(options):
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._group.addButton(btn, i)
            row.addWidget(btn)
        self.set_value(value)
        self._group.idClicked.connect(lambda i: self.changed.emit(self._values[i]))

    def set_value(self, value):
        if value in self._values:
            self._group.button(self._values.index(value)).setChecked(True)

    def value(self):
        return self._values[self._group.checkedId()] if self._group.checkedId() >= 0 else None


class Chip(QFrame):
    """A removable tag."""

    removed = Signal()

    def __init__(self, text, removable=True):
        super().__init__()
        self.setObjectName("Chip")
        self.setStyleSheet(f"""
            #Chip {{ background: {theme.SURFACE_2}; border: 1px solid {theme.BORDER}; border-radius: 15px; }}
            #Chip QPushButton {{ background: transparent; border: none; color: {theme.TEXT_FAINT}; padding: 0 2px;
                                 font-size: 15px; }}
            #Chip QPushButton:hover {{ color: {theme.TEXT}; }}
        """)
        row = QHBoxLayout(self)
        row.setContentsMargins(12, 4, 6 if removable else 12, 4)
        row.setSpacing(4)
        row.addWidget(QLabel(text))
        if removable:
            x = QPushButton("×")
            x.setCursor(Qt.CursorShape.PointingHandCursor)
            x.setToolTip(f"Remove {text}")
            x.clicked.connect(self.removed.emit)
            row.addWidget(x)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)


class StatusDot(QWidget):
    def __init__(self, color=theme.TEXT_FAINT, size=9):
        super().__init__()
        self._color = QColor(color)
        self.setFixedSize(size, size)

    def set_color(self, color):
        self._color = QColor(color)
        self.update()

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self._color)
        p.drawEllipse(self.rect())


class FlowRow(QWidget):
    """Chips laid out left to right, wrapping onto new lines."""

    def __init__(self, spacing=8):
        super().__init__()
        self._spacing = spacing
        self._items: "list[QWidget]" = []

    def set_widgets(self, widgets):
        for w in self._items:
            w.setParent(None)
            w.deleteLater()
        self._items = list(widgets)
        for w in self._items:
            w.setParent(self)
            w.show()
        self._relayout()

    def _relayout(self):
        x = y = row_h = 0
        width = max(self.width(), 1)
        for w in self._items:
            hint = w.sizeHint()
            if x and x + hint.width() > width:
                x, y = 0, y + row_h + self._spacing
                row_h = 0
            w.setGeometry(x, y, hint.width(), hint.height())
            x += hint.width() + self._spacing
            row_h = max(row_h, hint.height())
        self.setMinimumHeight(y + row_h)
        self.updateGeometry()

    def resizeEvent(self, event):  # noqa: N802
        self._relayout()


def app_icon():
    """A speech bubble with a translation mark - drawn, so no image file."""
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        s = size / 64
        bubble = QPainterPath()
        bubble.addRoundedRect(QRectF(4 * s, 6 * s, 56 * s, 42 * s), 12 * s, 12 * s)
        tail = QPainterPath()
        tail.moveTo(16 * s, 44 * s)
        tail.lineTo(14 * s, 58 * s)
        tail.lineTo(30 * s, 46 * s)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(theme.ACCENT))
        p.drawPath(bubble.united(tail))
        pen = QPen(QColor(theme.ACCENT_TEXT), max(1.5, 4.2 * s))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        # two caption lines, the second shorter
        p.drawLine(int(16 * s), int(21 * s), int(48 * s), int(21 * s))
        p.drawLine(int(16 * s), int(33 * s), int(38 * s), int(33 * s))
        p.end()
        icon.addPixmap(pm)
    return icon
