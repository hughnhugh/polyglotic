"""Verifies a caption made only of words not captioned on their own (the
skip_words setting - 嗯, mm-hmm...) isn't shown, live or final, while one
with anything else in it is; and that a caption taken back is logged with
why, only if it was showing. No GPU needed."""

import io
import queue
import types

import captions
import settings
import translate_vc
import translator
from overlay_core import speech_options

WORDS = settings.DEFAULTS["skip_words"]
SKIPPED = ["嗯。", "嗯", "嗯嗯嗯。", "嗯，嗯。", "嗯哼。", "呃。", "对。", "对对对。", "哦。", "啊。", "哎。", "嗯，对。",
           "Um.", "Hmm. Hmm. Hmm.", "hmmm", "Uh huh.", "Mm-hmm.", "ừ.", "Ừ."]
KEPT = ["嗯，我觉得这个可以。", "对，就是这样。", "哎呀。", "好的。", "Um, okay.", "Hmm, interesting.", "Yeah.",
        "humm... what?", "ừ, đi đi.", "Mum.", ""]


def main():
    for text in SKIPPED:
        assert translate_vc._only_skip_words(text, WORDS), text
    for text in KEPT:
        assert not translate_vc._only_skip_words(text, WORDS), text
    assert not translate_vc._only_skip_words("嗯。", "") and not translate_vc._only_skip_words("嗯。", " , ")
    assert translate_vc._only_skip_words("好的。", WORDS + ", 好的") and not translate_vc._only_skip_words("好的。", WORDS)
    print(f"OK: {len(SKIPPED)} lines of only those words skipped, {len(KEPT)} with anything else kept")

    translator.translate = lambda text, src, target, history=(): f"<{text}>"
    speech_options["skip_words"] = WORDS
    out = queue.Queue()
    process = translate_vc.make_process_segment(["en"], out)

    def run(text, **kwargs):
        segment = types.SimpleNamespace(text=text, no_speech_prob=0.0, avg_logprob=-0.2)
        info = types.SimpleNamespace(language="zh", language_probability=0.99)
        model = types.SimpleNamespace(transcribe=lambda *a, **k: (iter([segment]), info))
        process(model, "AUDIO", "", ("1", "marina"), **kwargs)
        records = []
        while not out.empty():
            records.append(out.get())
        return records

    assert run("嗯。", caption_group="g1", final=False) == [], "a live 嗯 isn't shown"
    final = run("嗯嗯。", caption_group="g1", final=True)
    assert final == [{"group": "g1", "final": True, "native": None,
                      "reason": "only words not captioned on their own"}], final
    assert run("嗯，我觉得这个可以。", caption_group="g2", final=True)[0]["native"] == "嗯，我觉得这个可以。"
    speech_options["skip_words"] = ""
    assert run("嗯。", caption_group="g3", final=True)[0]["native"] == "嗯。", "an empty list skips nothing"
    print("OK: a 嗯 isn't captioned, live or final - its live caption taken back; a longer line is")

    log = io.StringIO()
    board = captions.CaptionBoard(log_file=log)
    board.apply({"speaker": "marina", "lang": "zh", "native": "嗯", "translations": {}, "group": "g4",
                 "final": False})
    board.apply({"group": "g4", "final": True, "native": None, "reason": "only words not captioned on their own"})
    board.apply({"group": "g5", "final": True, "native": None, "reason": "only words not captioned on their own"})
    assert board.captions == [] and log.getvalue().count("[removed") == 1, log.getvalue()
    assert "[removed g4 - only words not captioned on their own]" in log.getvalue(), log.getvalue()
    print("OK: a caption taken back is logged with why - and only if it was showing")
    print("ALL OK")


if __name__ == "__main__":
    main()
