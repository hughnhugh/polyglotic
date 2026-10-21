"""Verifies OverlayApp updates a caption in place - its growing live
caption, then the final one, then joins (see overlay_core.LIVE_MIN_AUDIO_SEC
and BRIDGE_JOIN_GAP_SEC) - with a real (hidden) Tk window, no GPU/audio
needed."""

import io
import threading
import tkinter as tk

import overlay_core


def _caption(speaker, native, group=None, final=False, **extra):
    return dict({"speaker": speaker, "meta": "[en, 0.3s]", "native": native, "translated": None,
                 "group": group, "final": final}, **extra)


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

    text = _shown(app, _caption("A", "before"), _caption("B", "the", "g1"), _caption("C", "interjection"))
    assert text == "A [en, 0.3s] before\n\nB [en, 0.3s] … the\n\nC [en, 0.3s] interjection", repr(text)
    text = _shown(app, _caption("B", "the whole", "g1"))
    assert text == "A [en, 0.3s] before\n\nB [en, 0.3s] … the whole\n\nC [en, 0.3s] interjection", repr(text)
    print("OK: live updates replace the growing caption in place, marked …")

    text = _shown(app, _caption("B", "the whole sentence", "g1", final=True))
    assert text == "A [en, 0.3s] before\n\nB [en, 0.3s] the whole sentence\n\nC [en, 0.3s] interjection", repr(text)
    print("OK: final caption replaces the live one in its place")

    text = _shown(app, _caption("D", "dup live", "g2"))
    text = _shown(app, {"group": "g2", "final": True, "native": None})
    assert text == "A [en, 0.3s] before\n\nB [en, 0.3s] the whole sentence\n\nC [en, 0.3s] interjection", repr(text)
    print("OK: removing a group takes its live caption back out cleanly")

    text = _shown(app, _caption("E", "after"))
    assert text.endswith("interjection\n\nE [en, 0.3s] after"), repr(text)
    assert len(app.entry_marks) == 4
    print("OK: later captions still append normally")

    # A join: g1's final caption plus a later live caption g3, replaced by
    # one caption in g1's place; then a further join replaces that again.
    text = _shown(app, _caption("B", "more live", "g3"))
    text = _shown(app, _caption("B", "the whole sentence and more", "g1", final=True, absorb=["g3"]))
    assert text == ("A [en, 0.3s] before\n\nB [en, 0.3s] the whole sentence and more\n\nC [en, 0.3s] interjection"
                    "\n\nE [en, 0.3s] after"), repr(text)
    text = _shown(app, _caption("B", "the whole sentence and even more", "g1", final=True))
    assert "and even more\n\nC [en, 0.3s] interjection" in text and text.count("the whole sentence") == 1, repr(text)
    print("OK: joins replace the earlier final caption and absorb live ones, in place")

    # A long live caption grows in sections (see LIVE_SECTION_SEC), shown joined.
    text = _shown(app, _caption("H", "first part so", "g5", translated="1a"),
                  _caption("H", "first part so far", "g5", translated="1b"),
                  _caption("H", "and then", "g5", live_section=1, translated="2a"))
    assert "H [en, 0.3s] … first part so far and then\n    -> 1b 2a" in text, repr(text)
    text = _shown(app, _caption("H", "and then more", "g5", live_section=1, translated="2b"))
    assert "… first part so far and then more\n    -> 1b 2b" in text, repr(text)
    text = _shown(app, _caption("H", "the whole thing", "g5", final=True))
    assert "H [en, 0.3s] the whole thing" in text and "first part" not in text, repr(text)
    print("OK: live caption sections shown joined, then replaced by the final")

    logged = log.getvalue()
    assert "…" not in logged and "dup live" not in logged, logged
    assert "(final g1) (live x2) the whole sentence\n" in logged, logged
    assert "(final g1) (live x1) the whole sentence and more" in logged, logged
    assert "[removed g2" in logged
    print("OK: log keeps final captions only, with how many live updates each had")

    item = _caption("F", "timed", "g9", final=True, lang="vi", speech_end=overlay_core.time.time() - 1.25, proc=0.4)
    text = _shown(app, item)
    assert "F [vi, 1.2s] timed" in text or "F [vi, 1.3s] timed" in text, repr(text)
    assert "(proc 0.4s)" in log.getvalue()
    print("OK: shown delay measured from the end of speech; log adds processing time")

    overlay_core.MAX_HISTORY = 3
    text = _shown(app, _caption("G", "newest"))
    assert text.startswith("H [en, 0.3s] the whole thing") and text.endswith("G [en, 0.3s] newest"), repr(text)
    print("OK: history trimming drops exactly the oldest entries")
    root.destroy()
    print("ALL OK")


if __name__ == "__main__":
    main()
