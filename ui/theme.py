"""Colours, fonts and the stylesheet - one dark theme, used everywhere."""

import hashlib

from PySide6.QtGui import QColor, QFont, QFontDatabase

BG = "#0f1115"           # window
SIDEBAR = "#13161b"
SURFACE = "#191d24"      # cards
SURFACE_2 = "#20252e"    # inputs, hover
SURFACE_3 = "#2a303b"    # pressed, tracks
BORDER = "#262b34"
TEXT = "#e9ecf2"
TEXT_DIM = "#a0a8b6"
TEXT_FAINT = "#6c7484"
ACCENT = "#7c9cff"
ACCENT_HOVER = "#95afff"
ACCENT_TEXT = "#0d1220"
SUCCESS = "#4ade80"
WARNING = "#fbbf24"
DANGER = "#f87171"

# Speaker name colours on the overlay - distinct, readable on dark, and
# stable per name (a speaker keeps theirs all session, like on Discord).
SPEAKER_COLORS = ["#7cc4ff", "#ffb86b", "#c9a7ff", "#7ee0a8", "#ff8fa3", "#f5d76e", "#6fe0e0", "#ffa3e0"]


def speaker_color(name, is_you=False):
    if is_you:
        return ACCENT
    digest = hashlib.md5(name.encode("utf-8")).digest()
    return SPEAKER_COLORS[digest[0] % len(SPEAKER_COLORS)]


UI_FAMILIES = ["Segoe UI Variable Text", "Segoe UI", "Microsoft YaHei UI", "Microsoft JhengHei UI"]
# Captions mix scripts in one line ("你可以 just 讀呢個") - list a CJK
# family after the Latin one so every character gets a proper glyph.
CAPTION_FAMILIES = ["Segoe UI Variable Display", "Segoe UI", "Microsoft JhengHei UI", "Microsoft YaHei UI"]


def font_css(families, px, weight=400, italic=False):
    """A font as stylesheet properties - widgets under the app stylesheet
    take their font from it, so a font set in code doesn't stick."""
    names = ", ".join(f"'{f}'" for f in families)
    return f"font-family: {names}; font-size: {int(px)}px; font-weight: {weight};" + (" font-style: italic;" if italic else "")


def ui_font(px=13, weight=QFont.Weight.Normal):
    font = QFont()
    font.setFamilies(UI_FAMILIES)
    font.setPixelSize(px)
    font.setWeight(weight)
    return font


def caption_font(px, weight=QFont.Weight.Normal, italic=False):
    font = QFont()
    font.setFamilies(CAPTION_FAMILIES)
    font.setPixelSize(max(8, int(px)))
    font.setWeight(weight)
    font.setItalic(italic)
    return font


def icon_family():
    """Windows' own icon font - Fluent on Windows 11, MDL2 on 10."""
    families = set(QFontDatabase.families())
    return "Segoe Fluent Icons" if "Segoe Fluent Icons" in families else "Segoe MDL2 Assets"


def rgba(hex_color, alpha):
    c = QColor(hex_color)
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {alpha:.3f})"


STYLESHEET = f"""
* {{ outline: none; }}
QWidget {{ color: {TEXT}; font-size: 13px; }}
QMainWindow, #Page {{ background: {BG}; }}
#Sidebar {{ background: {SIDEBAR}; border-right: 1px solid {BORDER}; }}
QToolTip {{ background: {SURFACE_2}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px 6px; }}

#Card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px; }}
#PageTitle {{ font-size: 24px; font-weight: 600; }}
#PageSubtitle, #Dim {{ color: {TEXT_DIM}; }}
#Faint {{ color: {TEXT_FAINT}; font-size: 12px; }}
#CardTitle {{ font-size: 14px; font-weight: 600; }}
#SectionLabel {{ color: {TEXT_FAINT}; font-size: 11px; font-weight: 600; letter-spacing: 0.6px; }}

QPushButton {{
    background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 7px 14px; color: {TEXT};
}}
QPushButton:hover {{ background: {SURFACE_3}; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton:disabled {{ color: {TEXT_FAINT}; }}
QPushButton[kind="primary"] {{ background: {ACCENT}; color: {ACCENT_TEXT}; border: none; font-weight: 600; }}
QPushButton[kind="primary"]:hover {{ background: {ACCENT_HOVER}; }}
QPushButton[kind="ghost"] {{ background: transparent; border: none; color: {TEXT_DIM}; padding: 6px 10px; }}
QPushButton[kind="ghost"]:hover {{ background: {SURFACE_2}; color: {TEXT}; }}

#NavButton {{
    background: transparent; border: none; border-radius: 8px; text-align: left;
    padding: 9px 12px 9px 38px; color: {TEXT_DIM}; font-size: 13px;
}}
#NavButton:hover {{ background: {SURFACE_2}; color: {TEXT}; }}
#NavButton:checked {{ background: {SURFACE_2}; color: {TEXT}; font-weight: 600; }}
#QuitButton {{ background: transparent; border: none; border-radius: 8px; text-align: left; padding: 8px 12px;
              color: {TEXT_FAINT}; }}
#QuitButton:hover {{ background: {SURFACE_2}; color: {TEXT}; }}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 6px 10px; selection-background-color: {ACCENT}; selection-color: {ACCENT_TEXT};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {{ border: 1px solid {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {SURFACE_2}; border: 1px solid {BORDER}; selection-background-color: {SURFACE_3};
    selection-color: {TEXT}; padding: 4px;
}}
QSpinBox::up-button, QSpinBox::down-button {{ width: 0; border: none; }}

QSlider::groove:horizontal {{ height: 4px; background: {SURFACE_3}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 16px; height: 16px; margin: -6px 0; border-radius: 8px; background: {TEXT};
}}

QScrollArea {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {SURFACE_3}; border-radius: 3px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {TEXT_FAINT}; }}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{ background: none; height: 0; }}

QListWidget {{ background: transparent; border: none; }}
QListWidget::item {{ padding: 7px 8px; border-radius: 6px; }}
QListWidget::item:hover {{ background: {SURFACE_2}; }}
QListWidget::item:selected {{ background: {SURFACE_2}; color: {TEXT}; }}
QListWidget::indicator {{ width: 16px; height: 16px; border-radius: 5px; border: 1px solid {TEXT_FAINT}; }}
QListWidget::indicator:checked {{ background: {ACCENT}; border: 1px solid {ACCENT}; }}

QTextBrowser {{ background: transparent; border: none; }}
QProgressBar {{ background: {SURFACE_3}; border: none; border-radius: 3px; height: 6px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}

QMenu {{ background: {SURFACE}; border: 1px solid {BORDER}; padding: 6px; border-radius: 8px; }}
QMenu::item {{ padding: 7px 22px 7px 12px; border-radius: 6px; }}
QMenu::item:selected {{ background: {SURFACE_2}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 5px 4px; }}
"""
