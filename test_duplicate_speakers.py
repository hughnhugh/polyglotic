"""Verifies one person's speech cut twice - two Discord speakers flagged over
it - is captioned once (translate_vc.DUPLICATE_OVERLAP): the speaker who's
done more of the talking keeps it, whichever caption comes first; about
even, the one captioned last. Different words, or the same words said
again later, are two captions. No GPU needed."""

import queue
import time
import types

import numpy as np

import translate_vc
import translator

SR = 16000


def main():
    translator.translate = lambda text, src, target, history=(): f"<{text}>"
    out = queue.Queue()
    process = translate_vc.make_process_segment(["en"], out)
    t0 = time.time() - 60

    def say(who, text, start, seconds, group, final=True):
        """`who` ("1:小豆漿") captioned `text` for the audio from t0+start, `seconds` long."""
        user_id, name = who.split(":")
        segment = types.SimpleNamespace(text=text, no_speech_prob=0.0, avg_logprob=-0.2)
        info = types.SimpleNamespace(language="zh", language_probability=0.99)
        model = types.SimpleNamespace(transcribe=lambda *a, **k: (iter([segment]), info))
        process(model, np.zeros(int(seconds * SR), dtype=np.float32), "", (user_id, name),
                caption_group=group, final=final, speech_end=t0 + start + seconds)
        records = []
        while not out.empty():
            records.append(out.get())
        return records

    def shown(records):
        return [(r["speaker"], r["native"]) for r in records if r.get("native")]

    def taken_back(records):
        return [r["group"] for r in records if r.get("native") is None]

    XIAO, KIM, PEDRO = "1:小豆漿", "2:kimchie.", "3:Pedro"
    say(XIAO, "呃，可以讲中文，我觉得没问题。", 0, 2.5, "g1")
    say(KIM, "One.", 2.0, 0.5, "g2")
    first = say(XIAO, "挺好。", 3.0, 0.6, "g3")
    second = say(KIM, "挺好。", 3.1, 0.5, "g4")
    assert shown(first) == [("小豆漿", "挺好。")], first
    assert shown(second) == [] and taken_back(second) == ["g4"], second
    print("OK: kimchie's copy of 小豆漿's line dropped - 小豆漿's done more of the talking")

    first = say(KIM, "都是大人。", 5.0, 0.7, "g5")
    second = say(XIAO, "都是大人。", 4.9, 0.8, "g6")
    assert shown(first) == [("kimchie.", "都是大人。")], first
    assert shown(second) == [("小豆漿", "都是大人。")] and taken_back(second) == ["g5"], second
    print("OK: ...whichever comes first - kimchie's, already shown, is taken back")

    later = say(KIM, "都是大人。", 9.0, 0.7, "g7")
    assert shown(later) == [("kimchie.", "都是大人。")], later
    print("OK: the same words said again a few seconds later are their own caption")

    both = say(PEDRO, "我也觉得。", 12.0, 0.8, "g8") + say(XIAO, "那我们走吧。", 12.1, 0.8, "g9")
    assert shown(both) == [("Pedro", "我也觉得。"), ("小豆漿", "那我们走吧。")], both
    print("OK: two people talking over each other with different words - both captioned")

    out2 = queue.Queue()
    process = translate_vc.make_process_segment(["en"], out2)
    out = out2
    say("4:A", "我们先吃饭吧。", 20.0, 1.0, "h1")
    say("5:B", "好啊，没问题的。", 21.5, 1.2, "h2")
    first = say("4:A", "走吧。", 23.0, 0.5, "h3")
    second = say("5:B", "走吧。", 23.0, 0.5, "h4")
    assert shown(second) == [("B", "走吧。")] and taken_back(second) == ["h3"], second
    print("OK: about even (1.0s to 1.2s of talking) - the one captioned last keeps it")

    say("4:A", "这是一个测试。", 30.0, 1.0, "h5", final=False)
    process(None, None, "", ("4", "A"), caption_group="h5", final=True)  # taken back for another reason
    out.get()
    kept = say("5:B", "这是一个测试。", 30.1, 1.0, "h6")
    assert shown(kept) == [("B", "这是一个测试。")] and taken_back(kept) == [], kept
    print("OK: a caption taken back for another reason isn't counted as a duplicate later")

    # From a real call: DurianMT's "都是一模一样的" over the end of 小豆漿's line - an echo of its end, not the
    # same audio cut twice (much shorter) - so both stay, and no later update flips anything.
    out = queue.Queue()
    process = translate_vc.make_process_segment(["en"], out)
    say("6:小豆漿", "我们班的那个同学，你知道吗？", 40.0, 3.0, "k1")
    say("7:DurianMT", "对啊。", 43.5, 0.5, "k2")
    first = say("6:小豆漿", "那个 background 都是一模一样的。", 44.0, 2.5, "k3")
    echo = say("7:DurianMT", "都是一模一样的", 45.6, 1.2, "k4", final=False)
    again = say("7:DurianMT", "都是一模一样的，嗯。", 45.6, 1.6, "k4")
    assert shown(first) and shown(echo) and shown(again) and not taken_back(echo + again), (echo, again)
    print("OK: someone echoing the end of another's line isn't a duplicate - both kept")

    # A live update never takes back a finished caption; its own final can.
    out = queue.Queue()
    process = translate_vc.make_process_segment(["en"], out)
    say("8:A", "今天我们讲一下这个问题，好不好？", 50.0, 4.0, "m1")
    say("9:B", "我们开始吧。", 54.5, 1.0, "m2")  # B has said less
    live = say("8:A", "我们开始吧", 54.6, 0.9, "m3", final=False)
    assert shown(live) and taken_back(live) == [], "A's live copy shows, B's finished one stays for now"
    final = say("8:A", "我们开始吧。", 54.6, 0.9, "m3")
    assert taken_back(final) == ["m2"], final
    print("OK: a live update doesn't take back a finished caption - its final does")

    # Once lost, stays lost: C's caption lost to D's; C then talks more, but its update of that line stays dropped.
    out = queue.Queue()
    process = translate_vc.make_process_segment(["en"], out)
    say("10:D", "这个问题我想了很久，其实很简单。", 60.0, 4.0, "n1")
    say("10:D", "就是这样。", 64.5, 0.8, "n2")
    lost = say("11:C", "就是这样。", 64.5, 0.8, "n3", final=False)
    assert shown(lost) == [], lost
    say("11:C", "我觉得你说得对，但是还有一点我要补充一下，就是那个价格。", 66.0, 6.0, "n4")
    still = say("11:C", "就是这样。", 64.5, 0.8, "n3")
    assert shown(still) == [] and taken_back(still) == ["n3"], still
    print("OK: once a caption has lost to another, its later updates stay dropped")
    print("ALL OK")


if __name__ == "__main__":
    main()
