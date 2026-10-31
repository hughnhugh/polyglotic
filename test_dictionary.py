"""Verifies dictionary.py, the word lookup's CC-CEDICT reader: tone marks,
splitting a line into the longest words the dictionary knows (chengyu kept
whole), which word a character belongs to (in any language), English to
Chinese, and everyday readings first - on a small sample of real CC-CEDICT
lines, so no download is needed."""

import dictionary

SAMPLE = """\
# CC-CEDICT
電腦 电脑 [dian4 nao3] /computer/CL:臺|台[tai2]/
電 电 [dian4] /electric; electricity; electrical/
腦 脑 [nao3] /brain; mind; head/
畫蛇添足 画蛇添足 [hua4 she2 tian1 zu2] /lit. draw legs on a snake (idiom); fig. to ruin the effect by adding sth superfluous/
畫 画 [hua4] /to draw; to paint/picture; painting/
蛇 蛇 [she2] /snake; serpent/
添 添 [tian1] /to add; to increase; to replenish/
足 足 [ju4] /excessive/
足 足 [zu2] /foot; to be sufficient; ample/
足球 足球 [zu2 qiu2] /soccer ball; football/
滿足 满足 [man3 zu2] /to satisfy; to meet (the needs of)/
研究 研究 [yan2 jiu1] /research; study/
研究生 研究生 [yan2 jiu1 sheng1] /graduate student/
生命 生命 [sheng1 ming4] /life (as the characteristic of living beings)/
起源 起源 [qi3 yuan2] /origin/
這個 这个 [zhe4 ge5] /this; this one/
個人 个人 [ge4 ren2] /individual; personal/
這 这 [zhe4] /this; these/
個 个 [ge4] /individual; this; that; size/
人 人 [ren2] /person; people/
打遊戲 打游戏 [da3 you2 xi4] /to play a game/
遊戲 游戏 [you2 xi4] /game/
還 还 [huan2] /to pay back; to return/
還 还 [hai2] /still; still in progress; yet/
還款 还款 [huan2 kuan3] /repayment/
還錢 还钱 [huan2 qian2] /to pay back money/
瞭 了 [liao3] /variant of 了[liao3]/
了 了 [le5] /(completed action marker)/
女 女 [nu:3] /female; woman/
北京 北京 [Bei3 jing1] /Beijing/
"""


def main():
    assert dictionary.mark_pinyin("dian4 nao3") == "diàn nǎo"
    assert dictionary.mark_pinyin("nu:3 lu:4") == "nǚ lǜ"
    assert dictionary.mark_pinyin("xiong2 gui4 liu2 hou4") == "xióng guì liú hòu"   # i/u pairs: the last vowel
    assert dictionary.mark_pinyin("Bei3 jing1 r5 m2") == "Běi jīng r m"
    print("OK: numbered pinyin gets tone marks")

    dictionary._index = dictionary._parse(SAMPLE.splitlines())
    assert [e.pinyin for e in dictionary.lookup("电脑")] == ["diàn nǎo"]
    assert dictionary.lookup("電腦") == dictionary.lookup("电脑")
    assert dictionary.lookup("画蛇添足")[0].idiom and not dictionary.lookup("电脑")[0].idiom
    print("OK: lookups in either script, idioms flagged")

    def words(text):
        return [text[a:b] for a, b in dictionary.segment(text)]

    assert words("他画蛇添足了") == ["他", "画蛇添足", "了"]
    assert words("研究生命起源") == ["研究", "生命", "起源"]
    assert words("这个人") == ["这个", "人"]
    assert words("我们一起打游戏OK吧！") == ["我", "们", "一", "起", "打游戏", "OK", "吧", "！"]
    assert words("用電腦") == ["用", "電腦"]
    print("OK: lines split into the fewest, most even words - chengyu whole, Latin and punctuation apart")

    assert dictionary.word_at("他画蛇添足了", 3) == (1, 5)
    assert dictionary.word_at("OK电脑", 0) is None and dictionary.word_at("电脑", 5) is None
    assert dictionary.inner_words("打游戏") == ["游戏"]
    assert dictionary.pinyin_of("电脑游戏") == "diàn nǎo yóu xì"
    print("OK: the word under a character, the words inside one, pinyin for a phrase")

    answer = "**电脑 (diǎn nāo)** means computer; **游戏 (game)** too. **Bold** English stays."
    assert dictionary.with_bold_pinyin(answer) == ("**电脑** *diàn nǎo* means computer; **游戏 (game)** *yóu xì* "
                                                   "too. **Bold** English stays."), dictionary.with_bold_pinyin(answer)
    print("OK: a model's own pinyin swapped for the dictionary's, English glosses kept")

    assert dictionary.word_span_at("Doing it like this", 10) == (9, 13)
    assert dictionary.word_span_at("I don't know", 3) == (2, 7)
    assert dictionary.word_span_at("tiếng Việt", 2) == (0, 5)
    assert dictionary.word_span_at("OK电脑游戏", 3) == (2, 4) and dictionary.word_span_at("a, b", 1) is None
    print("OK: the word under a character in any language - Chinese words, runs of letters")

    line = "Doing it like this is overdoing it"
    assert dictionary.snap_to_words(line, 25, 33) == (22, 34)        # "rdoing i" -> "overdoing it"
    assert dictionary.snap_to_words(line, 21, 23) == (21, 31)        # " o" -> " overdoing", trimmed later
    assert dictionary.snap_to_words("我们一起打游戏吧", 2, 5) == (2, 5)   # Chinese by the character
    assert dictionary.snap_to_words("play 游戏 games", 2, 11) == (0, 13)  # "ay 游戏 gam" -> the whole of both
    print("OK: selections snapped to whole English words, Chinese by the character")

    dictionary._frequency = {"游戏": 2917, "打游戏": 3, "人": 20000}
    assert [e.simplified for e in dictionary.search_english("games")][:1] == ["游戏"]
    assert dictionary.search_english("play games")[0].simplified == "打游戏"   # "to play a game"
    assert dictionary.search_english("computers")[0].simplified == "电脑"
    assert [e.simplified for e in dictionary.search_english("Beijing")] == ["北京"]
    assert [e.simplified for e in dictionary.search_english("a person")][:1] == ["人"]
    assert dictionary.search_english("xyzzy") == []
    print("OK: English to Chinese - exact meanings first, stems and plurals, without the little words")

    assert [e.pinyin for e in dictionary.lookup("足")] == ["zú", "jù"]    # zú is in more words
    assert [e.pinyin for e in dictionary.lookup("还")][0] == "hái"        # though huán is in more
    assert [e.pinyin for e in dictionary.lookup("了")] == ["le", "liǎo"]  # "variant of" last
    print("OK: everyday readings first")
    print("ALL OK")


if __name__ == "__main__":
    main()
