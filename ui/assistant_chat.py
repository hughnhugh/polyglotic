"""
The side chat: a window beside the overlays for talking to the local
assistant model (assistant.py) about the call, opened with an overlay's
chat button. Ask it anything - with the conversation so far as context, or
without (the switch under the messages) - or, in one click:

- Summarise - the conversation so far: topics, what matters, anything
  asked of you;
- Explain the last line - the latest Chinese line someone said, broken
  down for a learner;
- Help me reply - a few natural replies in the conversation's language,
  the dictionary's pinyin under each Mandarin one.

Pinyin is always the dictionary's, never the model's (a 4B model's tones are
often wrong): it's asked not to write any, and the dictionary's is added
after each **bold** Chinese word in its answers.

Opening it switches the local assistant on (it loads in a few seconds);
closing it - x, Esc or the overlay's chat button again - switches it off,
freeing its VRAM. Like the overlays it stays on top and has no window
frame - drag the header to move it, the corner to resize; where it was is
remembered. Unlike them it takes the keyboard when you click into it, to
type.
"""

import re
import threading

from PySide6.QtCore import QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QLineEdit, QScrollArea, QSizeGrip, QVBoxLayout, QWidget

import assistant
import dictionary
import settings
from ui import theme
from ui.widgets import ToggleSwitch, bring_to_front, button, label

WIDTH = 400
HISTORY_MESSAGES = 12     # of the chat sent back to the model with each message
# Cooler than Qwen's suggested 0.7: on a replayed call, 0.3 kept to what was
# said and who said it, where 0.7 sometimes made the story the user's own.
TEMPERATURE = 0.3

SYSTEM = ("You are a helpful assistant built into vc-translator, a live translator for voice chats, helping an "
          "English speaker who's learning Chinese. Answer in English unless asked otherwise, briefly and plainly. "
          "Use **bold** for Chinese words and phrases. Don't write pinyin - the app adds it from a dictionary.")
# Goes in the latest message, right before the question - a small model
# loses track of a transcript far back in the system prompt, and tells the
# story as the user's own.
CONVERSATION = (f"Here's my voice chat so far, transcribed by speech recognition (it may contain mistakes), one line "
                f"per turn, with English translations of lines that aren't in English.\n"
                f"- People are labelled by their Discord names, or as Speaker A, Speaker B and so on: just labels "
                f"for who's talking. Use them as they're written.\n"
                f"- Lines from {assistant.ME} are mine. Everyone else is someone else in the call: their \"I\", "
                f"\"me\" and \"my\" mean themselves, and their \"you\" means whoever they're talking to - not "
                f"necessarily me.\n"
                f"- Say who said or did what by their name, and call me \"you\". Never describe what happened to "
                f"someone else as happening to me.\n\n"
                "[Conversation]\n{}\n[End of conversation]\n\n{}")


def build_messages(question, conversation, history):
    """The model's messages for `question`: the earlier chat (`history`,
    [{role, content}]), then the question - after the conversation
    transcript, if there is one. Earlier questions go back without theirs:
    only the latest transcript is current."""
    content = CONVERSATION.format(conversation, question) if conversation else question
    return ([{"role": "system", "content": SYSTEM}] + history[-HISTORY_MESSAGES:]
            + [{"role": "user", "content": content}])


def with_pinyin(text):
    """Suggested replies ("- <Chinese> — <meaning>" lines) laid out one
    paragraph each: the Chinese in bold, the dictionary's pinyin under it in
    italics, then the meaning. Paragraphs, not a list: Qt gives each line of
    a list item its own bullet."""
    out = []
    for line in text.splitlines():
        reply = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line)  # its list marker
        match = re.match(r"(.*?)\s[—–-]\s(.*)", reply)
        chinese, meaning = (match.group(1), match.group(2).strip()) if match else (reply, "")
        chinese = chinese.strip().strip("*").strip()
        if dictionary.has_han(chinese):
            parts = [f"**{chinese}**", f"*{dictionary.pinyin_of(chinese)}*"] + ([meaning] if meaning else [])
            out.append("  \n".join(parts))  # two spaces: a line break
        elif line.strip():
            out.append(line)
    return "\n\n".join(out)


class _Bubble(QFrame):
    """One message: yours tinted with the accent, the assistant's plain."""

    def __init__(self, text, mine, error=False):
        super().__init__()
        self.setObjectName("ChatMine" if mine else "ChatTheirs")
        box = QVBoxLayout(self)
        box.setContentsMargins(12, 8, 12, 9)
        self.text = QLabel(text)
        self.text.setWordWrap(True)
        self.text.setTextFormat(Qt.TextFormat.PlainText if mine else Qt.TextFormat.MarkdownText)
        self.text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.text.setStyleSheet(f"background: transparent; color: {theme.DANGER if error else theme.TEXT};")
        box.addWidget(self.text)


class AssistantChat(QWidget):
    _streamed = Signal(int, str, str)   # request id, text, "partial" / "done" / "error"

    def __init__(self, controller):
        super().__init__(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint)
        self.c = controller
        self.setWindowTitle("vc-translator assistant")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumSize(320, 360)
        self.setStyleSheet(f"""
            #ChatMine {{ background: {theme.rgba(theme.ACCENT, 0.16)}; border: 1px solid
                         {theme.rgba(theme.ACCENT, 0.3)}; border-radius: 10px; }}
            #ChatTheirs {{ background: {theme.SURFACE_2}; border: 1px solid {theme.BORDER}; border-radius: 10px; }}
            #ChatMessages {{ background: transparent; }}""")
        self.messages: "list[dict]" = []  # {"role", "content" (sent to the model), "bubble"}
        self._request = 0
        self._streaming = None            # the message being written, while it is
        self._labels = {}                 # speaker -> "Speaker A"..., for the session (see assistant.conversation)
        self._drag_from = None

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 14)
        root.setSpacing(10)

        self.header = QWidget()
        head = QHBoxLayout(self.header)
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(6)
        title = label("Assistant", "CardTitle")
        head.addWidget(title)
        self.status = label("", "Faint")
        head.addWidget(self.status)
        head.addStretch(1)
        clear = button("Clear", "ghost", self.clear)
        clear.setToolTip("Start a new chat")
        head.addWidget(clear)
        close = button("✕", "ghost", controller.close_chat)
        close.setToolTip("Close - switches the local assistant off (Esc)")
        head.addWidget(close)
        root.addWidget(self.header)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        for text, tip, run in [
                ("Summarise", "Summarise the conversation so far", self._summarise),
                ("Explain the last line", "Break down the latest Chinese line someone said", self._explain_last),
                ("Help me reply", "Suggest replies in the conversation's language", self._help_reply)]:
            btn = button(text, None, run)
            btn.setToolTip(tip)
            btn.setStyleSheet("padding: 5px 10px; font-size: 12px;")
            actions.addWidget(btn)
        actions.addStretch(1)
        root.addLayout(actions)

        # Switched off or not installed: what to do about it, in place of the messages' empty space.
        self.banner = QFrame()
        self.banner.setObjectName("ChatTheirs")
        banner = QHBoxLayout(self.banner)
        banner.setContentsMargins(12, 8, 8, 8)
        self.banner_text = label("", "Dim", wrap=True)
        banner.addWidget(self.banner_text, 1)
        self.switch_on = button("Switch on", "primary", lambda: controller.set_assistant_enabled(True))
        banner.addWidget(self.switch_on)
        root.addWidget(self.banner)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.viewport().setAutoFillBackground(False)
        inner = QWidget()
        inner.setObjectName("ChatMessages")
        self.column = QVBoxLayout(inner)
        self.column.setContentsMargins(0, 0, 6, 0)
        self.column.setSpacing(8)
        self.column.addStretch(1)   # messages sit at the bottom, newest last
        self.hint = label("Ask anything about the call, or pick one of the above. The conversation so far "
                          "goes along as context.", "Faint", wrap=True)
        self.column.addWidget(self.hint)
        self.scroll.setWidget(inner)
        root.addWidget(self.scroll, 1)

        context_row = QHBoxLayout()
        self.use_conversation = ToggleSwitch(settings.load()["chat_context"])
        self.use_conversation.toggled.connect(lambda on: settings.update(chat_context=on))
        context_row.addWidget(self.use_conversation)
        context_row.addWidget(label("Give it the conversation so far", "Dim"))
        context_row.addStretch(1)
        root.addLayout(context_row)

        send_row = QHBoxLayout()
        send_row.setSpacing(8)
        self.input = QLineEdit()
        self.input.setPlaceholderText("Ask about the call, a word, how to say something...")
        self.input.returnPressed.connect(self._send)
        send_row.addWidget(self.input, 1)
        self.send_btn = button("Send", "primary", self._send_or_stop)
        self.send_btn.setMinimumWidth(68)
        send_row.addWidget(self.send_btn)
        root.addLayout(send_row)

        self.grip = QSizeGrip(self)
        self.grip.setFixedSize(16, 16)
        self._save_timer = QTimer(self, singleShot=True, interval=400, timeout=self._save_geometry)
        self._streamed.connect(self._on_streamed)
        controller.prefs_changed.connect(self._refresh_state)
        controller.status_changed.connect(lambda _: self._refresh_state())
        self._refresh_state()

    # -- showing ------------------------------------------------------------------

    def open_beside(self, overlay_rect):
        """Shows the chat where it was last left - or, the first time, beside
        `overlay_rect` (the overlay its button was clicked on)."""
        saved = settings.load()["chat_geometry"]
        rect = QRect(*saved) if saved else None
        screens = QGuiApplication.screens()
        if rect is None or not any(s.availableGeometry().intersects(rect) for s in screens):
            screen = QGuiApplication.screenAt(overlay_rect.center()) or QGuiApplication.primaryScreen()
            area = screen.availableGeometry()
            height = min(max(overlay_rect.height(), 480), area.height() - 16)
            x = overlay_rect.right() + 10
            if x + WIDTH > area.right():
                x = max(area.left() + 8, overlay_rect.left() - WIDTH - 10)
            y = min(max(area.top() + 8, overlay_rect.bottom() - height), area.bottom() - height)
            rect = QRect(x, y, WIDTH, height)
        self.setGeometry(rect)
        self.show()
        self.activateWindow()   # opened to type in
        self.input.setFocus()
        bring_to_front(self)

    def _save_geometry(self):
        r = self.geometry()
        settings.update(chat_geometry=[r.x(), r.y(), r.width(), r.height()])

    def _refresh_state(self):
        enabled, text = self.c.assistant_enabled, str(self.c.status.get("assistant", ""))
        loading = enabled and text.startswith("Loading")
        failed = enabled and text.startswith("Failed")
        self.status.setText("loading..." if loading else assistant.MODEL_NAME if enabled else "off")
        self.banner.setVisible(not enabled or loading or failed)
        self.switch_on.setVisible(not enabled and assistant.installed())
        if not assistant.installed():
            self.banner_text.setText("The local assistant model isn't installed - run setup.bat -Assistant "
                                     "(2.7 GB download).")
        elif not enabled:
            self.banner_text.setText("The local assistant model is off. Switching it on uses ~3 GB of GPU memory.")
        elif loading:
            self.banner_text.setText("Loading the model - a few seconds...")
        elif failed:
            self.banner_text.setText(f"The model didn't load: {text}")

    # -- messages -----------------------------------------------------------------

    def _add(self, role, text, content=None, error=False):
        """Adds a message - `text` shown, `content` (if different) what the
        model is given."""
        self.hint.hide()
        bubble = _Bubble(text, role == "user", error)
        self.column.addWidget(bubble)
        message = {"role": role, "content": content or text, "bubble": bubble, "error": error}
        self.messages.append(message)
        self._scroll_to_end()
        return message

    def _scroll_to_end(self):
        bar = self.scroll.verticalScrollBar()
        QTimer.singleShot(0, lambda: bar.setValue(bar.maximum()))

    def clear(self):
        self._stop()
        for message in self.messages:
            message["bubble"].deleteLater()
        self.messages = []
        self.hint.show()

    def _conversation(self):
        return assistant.conversation(self.c.board.captions, self._labels)

    def _ask(self, shown, prompt=None, with_conversation=None, pinyin=False):
        """Sends a message - `shown` in the chat, `prompt` (if different)
        to the model - and streams the reply in."""
        if self._streaming is not None:
            return
        conversation = self._conversation() if (self.use_conversation.isChecked() if with_conversation is None
                                                else with_conversation) else ""
        history = [{"role": m["role"], "content": m["content"]} for m in self.messages if not m["error"]]
        self._add("user", shown, prompt)
        payload = build_messages(prompt or shown, conversation, history)
        reply = self._add("assistant", "...")
        reply["pinyin"] = pinyin
        self._streaming = reply
        self._request += 1
        request = self._request
        self.send_btn.setText("Stop")

        def work():
            try:
                text = assistant.chat(payload, lambda so_far: self._streamed.emit(request, so_far, "partial"),
                                      cancelled=lambda: request != self._request, temperature=TEMPERATURE)
                self._streamed.emit(request, text, "done")
            except RuntimeError as e:
                self._streamed.emit(request, str(e), "error")

        threading.Thread(target=work, daemon=True, name="assistant-chat").start()

    def _on_streamed(self, request, text, state):
        reply = self._streaming
        if request != self._request or reply is None:
            return
        bar = self.scroll.verticalScrollBar()
        at_end = bar.value() >= bar.maximum() - 30
        if state == "error":
            reply["error"] = True
            reply["bubble"].text.setTextFormat(Qt.TextFormat.PlainText)
            reply["bubble"].text.setStyleSheet(f"background: transparent; color: {theme.DANGER};")
        reply["content"] = text   # as the model wrote it, labels and all - it gets it back as the chat so far
        if state == "done":
            text = with_pinyin(text) if reply.get("pinyin") else dictionary.with_bold_pinyin(text)
        if state != "error":
            text = assistant.restore_names(text, self._labels)  # after the pinyin: none for names
        reply["bubble"].text.setText(text or "(no answer)")
        if state != "partial":
            self._streaming = None
            self.send_btn.setText("Send")
        if at_end:
            self._scroll_to_end()

    def _stop(self):
        self._request += 1   # the worker sees it and stops reading
        if self._streaming is not None:
            if self._streaming["content"] == "...":
                self._streaming["bubble"].text.setText("(stopped)")
                self._streaming["content"] = "(stopped)"
            self._streaming = None
        self.send_btn.setText("Send")

    def _send_or_stop(self):
        if self._streaming is not None:
            self._stop()
        else:
            self._send()

    def _send(self):
        text = self.input.text().strip()
        if text and self._streaming is None:
            self.input.clear()
            self._ask(text)

    # -- one-click actions ------------------------------------------------------------

    def _need_conversation(self):
        if self._conversation():
            return True
        self.hint.hide()
        self._add("assistant", "Nothing's been said yet - once there are captions, this works on them.")
        return False

    def _summarise(self):
        if self._need_conversation():
            self._ask("Summarise the conversation", "Summarise the conversation so far in a few short bullet "
                      "points: the main topics, what each person said that matters, and anything asked or decided "
                      "- who asked whom. Only say I was asked something if a line clearly speaks to me.",
                      with_conversation=True)

    def _last_chinese(self):
        captions = [c for c in self.c.board.captions if not c.live and dictionary.has_han(c.native)]
        others = [c for c in captions if not c.is_you]
        return (others or captions or [None])[-1]

    def _explain_last(self):
        caption = self._last_chinese()
        if caption is None:
            self.hint.hide()
            self._add("assistant", "Nobody's said anything in Chinese yet.")
            return
        self._conversation()  # labels everyone so far
        who = self._labels.get(caption.speaker) or caption.speaker or "Someone"
        self._ask(f"Explain the last line: “{caption.native}”",
                  f"Explain this line {who} just said: “{caption.native}”. First what it means, then the key words "
                  "and any grammar a learner should notice. Keep it short.", with_conversation=True)

    def _help_reply(self):
        if not self._need_conversation():
            return
        last = self._last_chinese()
        language = "Cantonese" if last is not None and last.lang == "yue" else "Mandarin Chinese"
        self._ask("Help me reply", f"Suggest 3 short, natural, casual things I could say next in {language}, "
                  f"fitting the conversation. One per line, exactly like: - <{language}> — <English meaning>. No "
                  "pinyin, nothing else.", with_conversation=True, pinyin=language.startswith("Mandarin"))

    # -- window -------------------------------------------------------------------

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor(255, 255, 255, 34), 1))
        p.setBrush(QColor(theme.SURFACE))
        p.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 14, 14)

    def resizeEvent(self, event):  # noqa: N802
        self.grip.move(self.width() - self.grip.width() - 4, self.height() - self.grip.height() - 4)
        if self.isVisible():
            self._save_timer.start()

    def moveEvent(self, event):  # noqa: N802
        if self.isVisible():
            self._save_timer.start()

    def keyPressEvent(self, event):  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.c.close_chat()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and event.position().y() < self.header.geometry().bottom() + 4:
            self._drag_from = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._drag_from is not None:
            self.move(event.globalPosition().toPoint() - self._drag_from)

    def mouseReleaseEvent(self, event):  # noqa: N802
        self._drag_from = None
