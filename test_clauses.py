"""Verifies captions are split into clauses (clauses.split) and translated
a clause at a time (translate_vc.process_segment): each clause with the
caption's earlier ones as context - a live caption's earlier sections too
- and remembered, so a clause that comes out the same isn't translated
again and keeps its translation. No GPU needed."""

import queue
import types

import captions
import clauses
import translate_vc
import translator
from captions import join_text
from overlay_core import live_options

SPLITS = [
    ("嗯。", ["嗯。"]),
    ("心动了呢，他还他还给了谁？他还给了这个佛印禅师。", ["心动了呢，他还他还给了谁？", "他还给了这个佛印禅师。"]),
    # a comma only ends a clause once it's long enough; "嗯，" alone joins what follows
    ("嗯，我觉得这个地方很好，我们明天再去一次吧。", ["嗯，我觉得这个地方很好，", "我们明天再去一次吧。"]),
    ("对，好。", ["对，好。"]),
    ("I mean, I think it's just like it really depends, like your habits, like your system.",
     ["I mean, I think it's just like it really depends,", "like your habits, like your system."]),
    ("And you know everything. It's supposed to be AI, but like, man, this already beats it, for sure.",
     ["And you know everything.", "It's supposed to be AI,", "but like, man, this already beats it, for sure."]),
    ("3.5 dollars is fine. That's it.", ["3.5 dollars is fine. That's it."]),   # not "3." / "5 dollars"
    ("他说：“我靠！我写这么好的诗，你告诉我我放屁？”然后走了。",
     ["他说：“我靠！", "我写这么好的诗，你告诉我我放屁？”", "然后走了。"]),
    ("no punctuation at all in this one", ["no punctuation at all in this one"]),
    ("我觉得这个 culture concept 很 specific，你懂吗？", ["我觉得这个 culture concept 很 specific，你懂吗？"]),
]


def main():
    for text, expected in SPLITS:
        got = clauses.split(text)
        assert got == expected, (text, got)
        assert join_text(got) == text, (text, join_text(got))
    assert clauses.split("") == [] and clauses.split("   ") == []
    assert clauses.size("I'm") == 2 and clauses.size("我觉得") == 3 and clauses.size("我觉得 OK") == 5
    print(f"OK: {len(SPLITS)} lines split into clauses at sentence ends and long-enough commas, joining back up")

    calls = []

    def fake_translate(text, src, target, history=()):
        calls.append((text, target, tuple(history)))
        return "" if "FAIL" in text else f"<{target}:{text}>"

    translator.translate = fake_translate
    out = queue.Queue()
    process = translate_vc.make_process_segment(["en", "zh"], out)

    def model(text, lang):
        segment = types.SimpleNamespace(text=text, no_speech_prob=0.0, avg_logprob=-0.2)
        info = types.SimpleNamespace(language=lang, language_probability=0.99)
        return types.SimpleNamespace(transcribe=lambda *a, **k: (iter([segment]), info))

    def run(text, lang="zh", **kwargs):
        calls.clear()
        process(model(text, lang), "AUDIO", "", ("1", "marina"), **kwargs)
        return out.get_nowait()

    first = "我觉得这个地方很好，我们明天再去一次吧。你觉得怎么样？要不要叫上他们？"
    record = run(first, caption_group="g1", final=True)
    parts = [c["native"] for c in record["clauses"]]
    assert parts == ["我觉得这个地方很好，", "我们明天再去一次吧。", "你觉得怎么样？", "要不要叫上他们？"], parts
    assert [c["translations"]["en"] for c in record["clauses"]] == [(f"<en:{p}>", True) for p in parts]
    assert all(c["translations"]["zh"] == (p, False) for c, p in zip(record["clauses"], parts)), "Mandarin into zh"
    assert record["translations"]["en"] == (join_text(f"<en:{p}>" for p in parts), True)
    assert record["translations"]["zh"] == (first, False) and record["native"] == first
    assert [c[0] for c in calls] == parts and all(c[1] == "en" for c in calls)
    assert calls[0][2] == () and calls[1][2] == ((parts[0], f"<en:{parts[0]}>"),)
    assert len(calls[3][2]) == translate_vc.CLAUSE_HISTORY and calls[3][2][0][0] == parts[0], "the last 3 as context"
    print("OK: a caption translated a clause at a time, each with the clauses before it as context")

    record = run(first.replace("要不要叫上他们？", "要不要叫上小王？"), caption_group="g2", final=True)
    assert [c[0] for c in calls] == ["要不要叫上小王？"], calls
    assert record["clauses"][0]["translations"]["en"] == (f"<en:{parts[0]}>", True)
    print("OK: clauses that came out the same keep their translation - only the changed one's translated")

    record = run("嗯，我觉得FAIL这个不行。但是另外一个可以。", caption_group="g3", final=True)
    versions = [c["translations"]["en"] for c in record["clauses"]]
    assert versions == [("嗯，我觉得FAIL这个不行。", False), ("<en:但是另外一个可以。>", True)], versions
    assert record["translations"]["en"][1] is True
    run("嗯，我觉得FAIL这个不行。", caption_group="g4", final=True)
    assert calls, "a failed translation isn't remembered - it's tried again"
    print("OK: a clause the translator failed on shows as said, and is tried again next time")

    # A live caption's sections: section 1's first clause gets section 0's clauses as context.
    run("今天天气真的很不错，我们出去走走吧。", caption_group="g5", final=False, live_section=0)
    record = run("公园里人多不多？", caption_group="g5", final=False, live_section=1)
    assert calls[0][2] == (("今天天气真的很不错，", "<en:今天天气真的很不错，>"), ("我们出去走走吧。", "<en:我们出去走走吧。>")), calls
    run("今天天气真的很不错，我们出去走走吧。公园里人多不多？", caption_group="g5", final=True)
    assert calls == [], "the final pass's clauses were all translated live already"
    run("公园里人多不多？", caption_group="g5", final=False, live_section=1)
    assert calls == [], "remembered, even with the live sections gone"
    print("OK: a live caption's later section has the earlier ones as context; the final pass reuses them")

    live_options["translate"] = False
    record = run("这个不用翻译，等停下来再说。", caption_group="g6", final=False)
    assert all(c["translations"]["en"] == (c["native"], False) for c in record["clauses"]) and not calls
    live_options["translate"] = True
    print("OK: live updates left untranslated when that's switched off")

    # The board puts a live caption's sections' clauses end to end.
    board = captions.CaptionBoard()
    board.apply(run("今天天气真的很不错，我们出去走走吧。", caption_group="g7", final=False, live_section=0))
    board.apply(run("公园里人多不多？", caption_group="g7", final=False, live_section=1))
    caption = board.captions[-1]
    assert [native for native, _ in caption.clauses] == ["今天天气真的很不错，", "我们出去走走吧。", "公园里人多不多？"]
    assert caption.clause_lines("en") == [(n, f"<en:{n}>", True) for n, _ in caption.clauses]
    assert caption.native == "今天天气真的很不错，我们出去走走吧。公园里人多不多？"
    board.apply({"speaker": "x", "native": "Old record.", "translations": {"en": ("Old record.", False)},
                 "final": True})
    assert board.captions[-1].clauses == (("Old record.", {"en": ("Old record.", False)}),)
    print("OK: on the board, a live caption's sections' clauses end to end; a record without clauses is one")
    print("ALL OK")


if __name__ == "__main__":
    main()
