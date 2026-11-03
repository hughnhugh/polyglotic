"""Verifies captions get word links (docs/word-links.md) in the pipeline:
each clause's words linked to each of its translations while word links
are on and the model's loaded - a live caption's clauses too (the overlay
leaves out the one still being spoken); a clause/translation pair aligned
once; none when off - and that the board keeps them with each clause. No
model needed: a fake aligner records what it's asked."""

import queue
import types

import aligner
import captions
import translate_vc
import translator
from overlay_core import link_options


def main():
    calls = []

    def fake_links(source, target):
        calls.append((source, target))
        return [((0, 2), (0, len(target.split()[0])))]  # first two characters - first word

    translator.translate = lambda text, src, target, history=(): f"Said {text}"
    aligner.loaded = lambda: True
    aligner.links = fake_links
    out = queue.Queue()
    process = translate_vc.make_process_segment(["en", "zh"], out)

    def run(text, **kwargs):
        calls.clear()
        segment = types.SimpleNamespace(text=text, no_speech_prob=0.0, avg_logprob=-0.2)
        info = types.SimpleNamespace(language="zh", language_probability=0.99)
        model = types.SimpleNamespace(transcribe=lambda *a, **k: (iter([segment]), info))
        process(model, "AUDIO", "", ("1", "marina"), **kwargs)
        return out.get_nowait()

    link_options["enabled"] = True
    text = "我昨天去了那个新开的火锅店，人特别多，排了一个多小时的队。"
    record = run(text, caption_group="g1", final=True)
    parts = [c["native"] for c in record["clauses"]]
    assert len(parts) == 2, parts
    for clause in record["clauses"]:
        assert set(clause["links"]) == {"en"}, clause["links"]  # zh: Mandarin into Chinese - no translation to link
        assert clause["links"]["en"] == [((0, 2), (0, 4))], clause["links"]
    assert calls == [(p, f"Said {p}") for p in parts], calls
    print("OK: each clause's words linked to its translation - not to an untranslated line")

    record = run(text, caption_group="g2", final=True)
    assert calls == [] and record["clauses"][0]["links"]["en"], "aligned once, remembered"
    print("OK: a clause and translation that come out the same aren't aligned again")

    record = run("今天我们讲一下这个问题，然后大家可以提问。", caption_group="g3", final=False)
    assert all(c["links"]["en"] for c in record["clauses"]) and len(calls) == 2, (record["clauses"], calls)
    print("OK: a live caption's clauses are linked too - its last, once its section's done, stays as it is")

    link_options["enabled"] = False
    record = run("这个不用对齐，因为现在是关着的。", caption_group="g4", final=True)
    assert calls == [] and all(c["links"] == {} for c in record["clauses"]), record["clauses"]
    print("OK: word links off - nothing aligned")

    board = captions.CaptionBoard()
    link_options["enabled"] = True
    board.apply(run(text, caption_group="g5", final=False, live_section=0))
    board.apply(run("后来我们就回家了，挺累的。", caption_group="g5", final=False, live_section=1))
    caption = board.captions[-1]
    assert len(caption.clause_links) == len(caption.clauses) == 3, caption.clause_links
    assert all(links["en"] for links in caption.clause_links), caption.clause_links
    print("OK: the board keeps each clause's links, a live caption's sections end to end")
    print("ALL OK")


if __name__ == "__main__":
    main()
