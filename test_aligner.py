"""Verifies word alignment (aligner.py) on real clause pairs: content words
linked to the words that translate them, as key links; pronouns, 是 and
the like linked too, but not as key links; particles, fillers and
articles never; neighbouring words joined into one link. Needs the model
(models/awesome-align-with-co) - skipped without it. Runs on the CPU."""

import aligner
import dictionary


def shown(source, target, key_only=False):
    return {source[a:b]: target[c:d] for (a, b), (c, d), key, *_ in aligner.links(source, target) if key or not key_only}


def main():
    if not aligner.installed():
        print("SKIP: the alignment model isn't installed (models/awesome-align-with-co)")
        return
    dictionary.load()
    aligner.load("cpu")

    got = shown("昨天我去了那个新开的火锅店，", "Yesterday I went to that new hot pot place,", key_only=True)
    assert got.get("昨天") == "Yesterday" and got.get("火锅") == "hot pot" and got.get("新") == "new", got
    print("OK: key links -", got)

    got = shown("是阳澄湖大闸蟹吗？", "Are they Yangcheng Lake hairy crabs?")
    assert got.get("阳澄") == "Yangcheng" and got.get("大闸蟹") == "hairy crabs" and got.get("湖") == "Lake", got
    assert "吗" not in got, "particles never linked"
    print("OK:", got, "- neighbouring characters linked to one word are one link")

    links = aligner.links("他们都是台湾人，你不用在意。", "They're all from Taiwan, you don't need to worry.")
    by_word = {"他们都是台湾人，你不用在意。"[a:b]: key for (a, b), _, key, *_ in links}
    assert by_word.get("台湾") is True and by_word.get("在意") is True, by_word
    assert by_word.get("他们") is False or by_word.get("你") is False, by_word
    print("OK: content words are key links; pronouns linked, but not as key links -", by_word)

    got = shown("就很好玩儿啊！", "That's so much fun!")
    assert "啊" not in got and "fun" in got.get("好玩儿", ""), got
    assert shown("就很好玩儿啊！", "That's so much fun!", key_only=True).keys() <= {"好玩儿"}
    print("OK:", got, "- 啊 never; 就/很 only beyond key words")

    got = shown("I'm not sure I'd dare handle it yet.", "我不太确定自己敢不敢处理。", key_only=True)
    assert got.get("handle") == "处理" and "敢" in got.get("dare", ""), got
    print("OK:", got, "- English to Chinese too")
    assert aligner.links("。", "Hmm.") == [] and aligner.links("", "") == []

    import time
    from overlay_core import status
    aligner.stop()
    assert not aligner.loaded() and status.values["aligner"] == "Off", status.values["aligner"]
    assert aligner.links("昨天我去了火锅店。", "Yesterday I went to a hot pot place.") == [], "unloaded - no links"
    aligner.start()
    assert status.values["aligner"] == "Loading...", status.values["aligner"]
    deadline = time.time() + 60
    while aligner.loading() and time.time() < deadline:
        time.sleep(0.2)
    assert aligner.loaded() and status.values["aligner"] == "Ready (on the CPU)", status.values["aligner"]
    aligner.stop()
    aligner.start()
    assert aligner.loading()
    aligner.stop()  # switched off while it's loading: let go of once it's loaded
    deadline = time.time() + 60
    while aligner.loading() and time.time() < deadline:
        time.sleep(0.2)
    assert not aligner.loaded() and status.values["aligner"] == "Off", status.values["aligner"]
    print("OK: off lets go of the model (no links meanwhile), on loads it again - its status saying which")
    print("ALL OK")


if __name__ == "__main__":
    main()
