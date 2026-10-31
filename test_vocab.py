"""Verifies vocab.py, your saved words: saving into categories, matching
saved words in either script and inside longer words but never across a
word boundary, categories added/changed/deleted, and the file surviving
a reload - no GUI needed."""

import os
import tempfile

import dictionary
import test_dictionary
import vocab


def main():
    dictionary._index = dictionary._parse(test_dictionary.SAMPLE.splitlines())
    path = os.path.join(tempfile.mkdtemp(), "vocab.json")
    v = vocab.Vocab(path)
    changes = []
    v.on_change = lambda: changes.append(v.version)
    assert [c["id"] for c in v.categories] == ["learning", "tricky", "known"]

    v.save("电脑", "電腦", "diàn nǎo", "computer", "learning", "我的电脑又卡了")
    v.save("游戏", "遊戲", "yóu xì", "game", "tricky")
    v.save("个人", "個人", "gè rén", "personal", "known")
    learning, tricky = v.category("learning")["color"], v.category("tricky")["color"]
    assert v.spans("我的电脑") == [(2, 4, learning)]
    assert v.spans("我的電腦") == [(2, 4, learning)]
    assert v.spans("一起打游戏") == [(3, 5, tricky)]   # inside 打游戏
    assert v.spans("这个人") == []                      # 这个/人 - 个人 would cut across 这个
    assert v.spans("English only") == []
    assert changes == [1, 2, 3]
    print("OK: saved words matched in either script, inside words, not across them")

    added = v.words["电脑"]["added"]
    v.move("電腦", "tricky")
    assert v.words["电脑"]["category"] == "tricky" and v.words["电脑"]["added"] == added
    assert v.words["电脑"]["sentence"] == "我的电脑又卡了"
    v.remove("遊戲")
    assert "游戏" not in v.words and v.spans("打游戏") == []
    print("OK: moved between categories, removed by either script")

    new = v.add_category("Food")
    assert new["color"] not in {c["color"] for c in vocab.DEFAULT_CATEGORIES}
    v.update_category(new["id"], name="Food words", color="#123456")
    v.save("足球", "足球", "zú qiú", "football", new["id"])
    assert v.spans("足球") == [(0, 2, "#123456")]
    v.remove_category("tricky")
    assert "电脑" not in v.words and v.category("tricky") is None
    print("OK: categories added, renamed, recoloured, deleted with their words")

    reloaded = vocab.Vocab(path)
    assert reloaded.words == v.words and reloaded.categories == v.categories
    assert reloaded.spans("我们踢足球") == [(3, 5, "#123456")]
    print("OK: kept across a reload")
    print("ALL OK")


if __name__ == "__main__":
    main()
