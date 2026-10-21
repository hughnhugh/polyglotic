"""Verifies captions.CaptionBoard, the caption model every view shares:
live updates replace a caption in place, sections join, finals and joins
replace earlier versions in the earliest one's place, removals, history
trimming and the session log - no GUI needed."""

import io
import time

import captions


def _record(speaker, native, group=None, final=True, en=None, **extra):
    translations = {"en": (en, True)} if en else {"en": (native, False)}
    return dict({"speaker": speaker, "lang": "zh" if en else "en", "native": native, "translations": translations,
                 "group": group, "final": final, "absorb": [], "speech_end": time.time() - 1.0, "proc": 0.3}, **extra)


def _texts(board, target="en"):
    return [(c.speaker, c.text_for(target)[0], c.live) for c in board.captions]


def main():
    log = io.StringIO()
    board = captions.CaptionBoard(log_file=log)

    board.apply(_record("A", "before"))
    board.apply(_record("B", "the", "g1", final=False))
    board.apply(_record("C", "interjection"))
    board.apply(_record("B", "the whole", "g1", final=False))
    assert _texts(board) == [("A", "before", False), ("B", "the whole", True), ("C", "interjection", False)], _texts(board)
    key = board.captions[1].key
    print("OK: live updates replace the growing caption in place")

    board.apply(_record("B", "the whole sentence", "g1"))
    assert _texts(board)[1] == ("B", "the whole sentence", False) and board.captions[1].key == key
    print("OK: final caption replaces the live one in its place, keeping its key")

    board.apply(_record("D", "echoed", "g2", final=False))
    board.apply({"group": "g2", "final": True, "native": None})
    assert [s for s, _, _ in _texts(board)] == ["A", "B", "C"]
    print("OK: a removal takes a group's live caption back out")

    board.apply(_record("B", "more", "g3", final=False))
    board.apply(dict(_record("B", "the whole sentence and more", "g1"), absorb=["g3"]))
    assert _texts(board) == [("A", "before", False), ("B", "the whole sentence and more", False),
                             ("C", "interjection", False)], _texts(board)
    print("OK: a join replaces the earlier final and absorbs the newer live caption, in place")

    board.apply(_record("E", "我觉得", "g5", final=False, en="I think", live_section=0))
    board.apply(_record("E", "这个地方", "g5", final=False, en="this place", live_section=1))
    cap = board.captions[-1]
    assert cap.native == "我觉得这个地方" and cap.text_for("en") == ("I think this place", "我觉得这个地方"), cap
    print("OK: live sections joined - Chinese without a space, translations with one")

    logged = log.getvalue()
    assert "(final g1) (proc 0.3s) (live x2) the whole sentence\n" in logged, logged
    assert "(final g1) (proc 0.3s) (live x1) the whole sentence and more" in logged, logged
    assert "echoed" not in logged and "[removed g2" in logged and "…" not in logged
    board.apply(_record("E", "我觉得这个地方很好", "g5", en="I think this place is great"))
    assert "    -> [en] I think this place is great" in log.getvalue()
    print("OK: log keeps finals only, with live update counts and translations")

    board.apply("[Discord bridge: connected]")
    assert board.notices[-1] == "[Discord bridge: connected]" and len(board.captions) == 4
    print("OK: notices kept apart from captions")

    small = captions.CaptionBoard(max_history=3)
    for i in range(5):
        small.apply(_record("X", f"line {i}"))
    assert [t for _, t, _ in _texts(small)] == ["line 2", "line 3", "line 4"]
    print("OK: history trimmed to the newest captions")

    assert captions.join_text(["Hello there.", "how are you"]) == "Hello there. how are you"
    assert captions.join_text(["OK", "我们走"]) == "OK我们走"
    print("ALL OK")


if __name__ == "__main__":
    main()
