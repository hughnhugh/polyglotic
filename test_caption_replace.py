"""Verifies OverlayApp swaps a long Discord interval's live captions for its
final one in place (see overlay_core.BRIDGE_FINAL_MAX_SEC), with a real
(hidden) Tk window - no GPU/audio needed."""

import io
import threading
import tkinter as tk

import overlay_core


def _caption(speaker, native, group=None, final=False):
    return {"speaker": speaker, "meta": "[en, 0.3s]", "native": native, "translated": None,
            "group": group, "final": final}


def _shown(app, *items):
    for item in items:
        app.caption_queue.put(item)
    app.poll_queue()
    return app.text.get("1.0", "end-1c")


def main():
    root = tk.Tk()
    root.withdraw()
    log = io.StringIO()
    app = overlay_core.OverlayApp(root, threading.Event(), log_file=log)

    text = _shown(app, _caption("A", "before"), _caption("B", "live one", "g1"), _caption("C", "interjection"),
                  _caption("B", "live two", "g1"))
    assert text.count("\n\n") == 3 and "live one" in text and "…" in text, text

    text = _shown(app, _caption("B", "the whole sentence", "g1", final=True))
    assert text == "A [en, 0.3s] before\n\nB [en, 0.3s] the whole sentence\n\nC [en, 0.3s] interjection", repr(text)
    print("OK: final caption replaced both live pieces in the first one's place")

    text = _shown(app, _caption("D", "dup live", "g2"))
    text = _shown(app, {"group": "g2", "final": True, "native": None})
    assert text == "A [en, 0.3s] before\n\nB [en, 0.3s] the whole sentence\n\nC [en, 0.3s] interjection", repr(text)
    print("OK: removing a group takes its live caption back out cleanly")

    text = _shown(app, _caption("E", "after"))
    assert text.endswith("interjection\n\nE [en, 0.3s] after"), repr(text)
    assert list(app.entry_marks) and len(app.entry_marks) == 4
    print("OK: later captions still append normally")

    assert "(live g1)" in log.getvalue() and "(final g1)" in log.getvalue() and "[removed g2" in log.getvalue()
    print("OK: log keeps the live and final captions, tagged")

    # A join: g1's final caption plus a later live group g3 replaced by one
    # caption in g1's place; then a further join replaces that again.
    text = _shown(app, _caption("B", "more live", "g3"))
    joined = dict(_caption("B", "the whole sentence and more", "g1", final=True), absorb=["g3"])
    text = _shown(app, joined)
    assert text == ("A [en, 0.3s] before\n\nB [en, 0.3s] the whole sentence and more\n\nC [en, 0.3s] interjection"
                    "\n\nE [en, 0.3s] after"), repr(text)
    text = _shown(app, _caption("B", "the whole sentence and even more", "g1", final=True))
    assert "and even more\n\nC [en, 0.3s] interjection" in text and text.count("the whole sentence") == 1, repr(text)
    print("OK: joins replace the earlier final caption and absorb live ones, in place")

    overlay_core.MAX_HISTORY = 3
    text = _shown(app, _caption("F", "newest"))
    assert text == "C [en, 0.3s] interjection\n\nE [en, 0.3s] after\n\nF [en, 0.3s] newest", repr(text)
    print("OK: history trimming drops exactly the oldest entries")
    root.destroy()
    print("ALL OK")


if __name__ == "__main__":
    main()
