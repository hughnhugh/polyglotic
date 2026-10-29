"""
Runs the app: the capture pipeline, the main window, a subtitle overlay
per target language, the tray icon and global hotkeys - all wired through
one Controller, which owns the live state and remembers it in
settings.json.

Global hotkeys (work over games): Ctrl+Alt+L lock/unlock the overlays,
Ctrl+Alt+P pause/resume captioning, Ctrl+Alt+H hide/show the overlays.
"""

import ctypes
import ctypes.wintypes
import os
import queue
import sys
from datetime import datetime

from PySide6.QtCore import QAbstractNativeEventFilter, QObject, QRect, QTimer, Signal
from PySide6.QtGui import QAction, QGuiApplication
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

import overlay_core
import settings
from captions import CaptionBoard
from lang_codes import LANGUAGE_NAMES
from overlay_core import LOG_DIR, capture_options, live_options, speech_options
from ui import theme
from ui.overlay import CaptionOverlay, default_geometry
from ui.widgets import app_icon

try:  # GPU memory readout - optional (nvidia-ml-py)
    import pynvml
    pynvml.nvmlInit()
    _NVML = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    _NVML = None


def gpu_memory():
    """(used_gb, total_gb) for the whole GPU - every process, the models'
    included - or None without NVML."""
    if _NVML is None:
        return None
    info = pynvml.nvmlDeviceGetMemoryInfo(_NVML)
    return info.used / 1024 ** 3, info.total / 1024 ** 3


class Controller(QObject):
    """The app's live state and every action the UI can take on it."""

    status_changed = Signal(dict)
    board_changed = Signal()
    targets_changed = Signal()
    state_changed = Signal()     # paused / locked / hidden / mic toggles
    prefs_changed = Signal()     # overlay look, timing, speech options
    notice = Signal(str)

    def __init__(self, targets, pipeline_factory, model_size, capture_mic):
        super().__init__()
        saved = settings.load()
        self.targets = targets                    # languages with an overlay
        # target -> its overlay's second language (see Caption.lines_for)
        self.second_languages = {t: c for t, c in saved["second_languages"].items() if c}
        self.translate_into: "list[str]" = []     # shared with process_segment - edited in place
        self._sync_translate_into()
        self.prefs = saved["overlay"]
        self.prefs_version = 0
        self.locked = saved["locked"]
        self.overlays_hidden = False
        self.status = dict(overlay_core.status.values)
        self.overlays: "dict[str, CaptionOverlay]" = {}
        self.quitting = False

        self.board = CaptionBoard()
        self.log_path = None  # this session's transcript file, while saving is on
        if saved["save_transcript"]:
            self._start_log()

        self.caption_queue = queue.Queue()
        overlay_core.register_broadcast_queues([self.caption_queue])  # notices arrive with the captions
        self._status_queue = queue.Queue()
        overlay_core.status.subscribe(self._status_queue)
        self.pipeline = pipeline_factory(self.caption_queue, self.translate_into, model_size, capture_mic,
                                         saved["mic_device"])

        self._poll = QTimer(self, interval=40, timeout=self._drain)

    def start(self):
        for target in self.targets:
            self._open_overlay(target)
        self.pipeline.start()
        self._poll.start()

    def _drain(self):
        changed = False
        while True:
            try:
                record = self.caption_queue.get_nowait()
            except queue.Empty:
                break
            self.board.apply(record)
            changed = True
            if isinstance(record, str):
                self.notice.emit(record)
        if changed:
            for overlay in self.overlays.values():
                overlay.refresh()
            self.board_changed.emit()
        latest = None
        while True:
            try:
                latest = self._status_queue.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            self.status = latest
            self.status_changed.emit(latest)

    # -- targets / overlays -----------------------------------------------------

    def _sync_translate_into(self):
        """Every language captions are translated into: the overlays' and
        their second languages."""
        wanted = list(dict.fromkeys([*self.targets, *(self.second_languages[t] for t in self.targets
                                                      if t in self.second_languages)]))
        self.translate_into[:] = wanted

    def set_second_language(self, target, code):
        if code:
            self.second_languages[target] = code
        else:
            self.second_languages.pop(target, None)
        self._sync_translate_into()
        settings.update(second_languages=dict(self.second_languages))
        if target in self.overlays:
            self.overlays[target].refresh(force=True)
        self.prefs_changed.emit()

    def add_target(self, code):
        if code in self.targets:
            return
        self.targets.append(code)
        self._sync_translate_into()
        self._open_overlay(code)
        settings.update(targets=list(self.targets))
        self.targets_changed.emit()

    def remove_target(self, code):
        if code not in self.targets:
            return
        self.targets.remove(code)
        self._sync_translate_into()
        overlay = self.overlays.pop(code, None)
        if overlay is not None:
            overlay.close()
            overlay.deleteLater()
        settings.update(targets=list(self.targets))
        self.targets_changed.emit()

    def _open_overlay(self, code):
        overlay = CaptionOverlay(self, code, LANGUAGE_NAMES.get(code, code))
        overlay.setGeometry(self._geometry_for(code))
        self.overlays[code] = overlay
        overlay.refresh(force=True)
        if not self.overlays_hidden:
            overlay.show()

    def _geometry_for(self, code):
        saved = settings.load()["overlay_geometry"].get(code)
        if saved:
            rect = QRect(*saved)
            if any(screen.availableGeometry().intersects(rect) for screen in QGuiApplication.screens()):
                return rect
        screen = QGuiApplication.primaryScreen().availableGeometry()
        index = [t for t in self.targets if t not in settings.load()["overlay_geometry"]].index(code) \
            if code in self.targets else 0
        return default_geometry(screen, index, len(self.targets))

    def save_geometry(self, code, rect):
        geometry = dict(settings.load()["overlay_geometry"])
        geometry[code] = [rect.x(), rect.y(), rect.width(), rect.height()]
        settings.update(overlay_geometry=geometry)

    def reset_positions(self):
        settings.update(overlay_geometry={})
        screen = QGuiApplication.primaryScreen().availableGeometry()
        for i, code in enumerate(self.targets):
            if code in self.overlays:
                self.overlays[code].setGeometry(default_geometry(screen, i, len(self.targets)))

    # -- toggles --------------------------------------------------------------------

    def set_locked(self, locked):
        self.locked = locked
        for overlay in self.overlays.values():
            overlay.set_locked(locked)
        settings.update(locked=locked)
        self.state_changed.emit()

    def set_overlays_hidden(self, hidden):
        self.overlays_hidden = hidden
        for overlay in self.overlays.values():
            overlay.setVisible(not hidden)
        self.state_changed.emit()

    @property
    def paused(self):
        return capture_options["paused"]

    def set_paused(self, paused):
        capture_options["paused"] = paused
        self.state_changed.emit()

    @property
    def caption_mic(self):
        return capture_options["caption_mic"]

    def set_caption_mic(self, on):
        capture_options["caption_mic"] = on
        settings.update(caption_mic=on)
        self.state_changed.emit()

    @property
    def mic_enabled(self):
        return self.pipeline.mic_capture.enabled

    def mic_devices(self):
        return self.pipeline.mic_capture.devices()

    @property
    def mic_device(self):
        return self.pipeline.mic_capture.device_name

    def set_mic_device(self, name):
        self.pipeline.mic_capture.set_device(name)
        settings.update(mic_device=name)
        self.state_changed.emit()

    def set_mic_enabled(self, on):
        self.pipeline.mic_capture.set_enabled(on)
        settings.update(mic=on)
        self.state_changed.emit()

    # -- preferences -------------------------------------------------------------------

    def set_pref(self, key, value):
        self.prefs[key] = value
        self.prefs_version += 1
        settings.update(overlay=dict(self.prefs))
        for overlay in self.overlays.values():
            overlay.refresh()
            overlay.update()
        self.prefs_changed.emit()

    def set_language(self, code):
        speech_options["language"] = code
        settings.update(language=code)
        self.prefs_changed.emit()

    def set_context(self, text):
        speech_options["context"] = text
        settings.update(context=text)
        self.prefs_changed.emit()

    def set_live_interval(self, seconds):
        live_options["interval"] = seconds
        settings.update(live_interval=seconds)
        self.prefs_changed.emit()

    def set_translate_live(self, on):
        live_options["translate"] = on
        settings.update(translate_live=on)
        self.prefs_changed.emit()

    def set_model(self, size):
        self.pipeline.model_holder.request_reload(size)

    @property
    def save_transcript(self):
        return self.log_path is not None

    def set_save_transcript(self, on):
        if on and self.log_path is None:
            self._start_log()
        elif not on and self.log_path is not None:
            self.board.set_log_file(None)
            self.log_path = None
        settings.update(save_transcript=on)
        self.state_changed.emit()

    def _start_log(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        self.log_path = os.path.join(LOG_DIR, f"Session_{datetime.now():%Y%m%d_%H%M%S}.log")
        self.board.set_log_file(open(self.log_path, "a", encoding="utf-8"))

    def open_logs_folder(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        os.startfile(LOG_DIR)  # noqa: S606 - opens Explorer on our own folder

    def shutdown(self):
        self.quitting = True
        self._poll.stop()
        self.pipeline.shutdown()
        for overlay in self.overlays.values():
            overlay.close()


class _Hotkeys(QAbstractNativeEventFilter):
    """System-wide hotkeys through Windows' RegisterHotKey - they fire even
    while a game has focus, which a Qt shortcut can't."""

    WM_HOTKEY = 0x0312
    MOD_ALT, MOD_CONTROL, MOD_NOREPEAT = 0x0001, 0x0002, 0x4000

    def __init__(self, bindings):
        """bindings: [(key letter, callback)] - all Ctrl+Alt+key."""
        super().__init__()
        self._handlers = {}
        self.failed = []
        user32 = ctypes.windll.user32
        for i, (key, callback) in enumerate(bindings, start=1):
            if user32.RegisterHotKey(None, i, self.MOD_CONTROL | self.MOD_ALT | self.MOD_NOREPEAT, ord(key)):
                self._handlers[i] = callback
            else:
                self.failed.append(f"Ctrl+Alt+{key}")  # another app already has it

    def nativeEventFilter(self, event_type, message):  # noqa: N802
        kind = event_type.data() if hasattr(event_type, "data") else bytes(event_type)
        if kind == b"windows_generic_MSG":
            msg = ctypes.wintypes.MSG.from_address(int(message))
            if msg.message == self.WM_HOTKEY and msg.wParam in self._handlers:
                QTimer.singleShot(0, self._handlers[msg.wParam])
                return True, 0
        return False, 0

    def unregister(self):
        for i in self._handlers:
            ctypes.windll.user32.UnregisterHotKey(None, i)


def _tray(controller, show_main):
    tray = QSystemTrayIcon(app_icon())
    tray.setToolTip("Polyglotic")
    menu = QMenu()
    open_action = menu.addAction("Open Polyglotic", show_main)
    menu.setDefaultAction(open_action)
    menu.addSeparator()
    toggles = [
        ("Pause captions\tCtrl+Alt+P", lambda: controller.paused, controller.set_paused),
        ("Lock overlays\tCtrl+Alt+L", lambda: controller.locked, controller.set_locked),
        ("Hide overlays\tCtrl+Alt+H", lambda: controller.overlays_hidden, controller.set_overlays_hidden),
        ("Caption my own speech", lambda: controller.caption_mic, controller.set_caption_mic),
    ]
    actions = []
    for text, getter, setter in toggles:
        action = QAction(text, menu, checkable=True)
        action.toggled.connect(setter)
        menu.addAction(action)
        actions.append((action, getter))
    menu.addSeparator()
    menu.addAction("Quit", QApplication.quit)

    def sync():
        for action, getter in actions:
            blocked = action.blockSignals(True)
            action.setChecked(getter())
            action.blockSignals(blocked)
        tray.setToolTip("Polyglotic - paused" if controller.paused else "Polyglotic - listening")

    controller.state_changed.connect(sync)
    sync()
    tray.setContextMenu(menu)
    tray.activated.connect(lambda reason: show_main() if reason in (
        QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick) else None)
    tray._menu = menu  # keep the menu alive with the icon
    tray.show()
    return tray


def run(make_process_segment, targets, model_size, capture_mic):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    # Its own taskbar identity, so Windows shows the app's icon rather than Python's.
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("polyglotic")

    app = QApplication(sys.argv)
    app.setApplicationName("Polyglotic")
    app.setQuitOnLastWindowClosed(False)  # closing the main window keeps it running in the tray
    app.setWindowIcon(app_icon())
    app.setFont(theme.ui_font())
    app.setStyleSheet(theme.STYLESHEET)

    def pipeline_factory(caption_queue, translate_into, model, mic, mic_device):
        return overlay_core.Pipeline(make_process_segment(translate_into, caption_queue), model, mic, mic_device)

    controller = Controller(targets, pipeline_factory, model_size, capture_mic)

    from ui.main_window import MainWindow  # local import - it imports this module
    window = MainWindow(controller)

    def show_main():
        window.showNormal()
        window.raise_()
        window.activateWindow()

    tray = _tray(controller, show_main)
    window.tray = tray
    hotkeys = _Hotkeys([
        ("L", lambda: controller.set_locked(not controller.locked)),
        ("P", lambda: controller.set_paused(not controller.paused)),
        ("H", lambda: controller.set_overlays_hidden(not controller.overlays_hidden)),
    ])
    app.installNativeEventFilter(hotkeys)
    if hotkeys.failed:
        controller.board.apply(f"[Hotkeys already taken by another app: {', '.join(hotkeys.failed)}]")

    app.aboutToQuit.connect(controller.shutdown)
    app.aboutToQuit.connect(hotkeys.unregister)
    app.aboutToQuit.connect(tray.hide)

    controller.start()
    window.show()
    app.exec()
