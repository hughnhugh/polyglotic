"""Verifies the Vietnamese dictionary (vi_dictionary.py) on a few real
Wiktionary entries: what's kept of each, the cut-down cache read back the
same, lookups whatever the case or accent encoding, a hovered syllable
looked up as the longest word around it (not across punctuation), and
which lines read as Vietnamese. No download needed."""

import json
import os
import tempfile
import unicodedata

import dictionary
import vi_dictionary as vd

RECORDS = [
    {"word": "học sinh", "pos": "noun", "lang_code": "vi", "senses": [{"glosses": ["a student"]},
     {"glosses": ["a pupil; a schoolchild/schoolkid"]}],
     "etymology_text": "Sino-Vietnamese word from 學生, composed of 學 (“to learn”) and 生."},
    {"word": "học", "pos": "verb", "lang_code": "vi", "senses": [{"glosses": ["to study; to learn"]}],
     "forms": [{"form": "學", "tags": ["CJK"]}, {"form": "学", "tags": ["CJK"]}],
     "etymology_text": "Sino-Vietnamese word from 學."},
    {"word": "bánh mì", "pos": "noun", "lang_code": "vi",
     "senses": [{"glosses": ["bread, especially Saigon-style short baguettes with filling"]}],
     "forms": [{"form": "餅麵", "tags": ["CJK"]}, {"form": "chiếc", "tags": ["classifier"]},
               {"form": "ổ", "tags": ["classifier"]}]},
    {"word": "đi", "pos": "verb", "lang_code": "vi", "senses": [{"glosses": ["to go"]}, {"glosses": ["to leave"]}]},
    {"word": "đi", "pos": "adv", "lang_code": "vi", "senses": [{"glosses": ["away; out of sight"]}]},
    {"word": "hôm nay", "pos": "noun", "lang_code": "vi", "senses": [{"glosses": ["today"]}]},
    {"word": "學", "pos": "character", "lang_code": "vi", "senses": [{"glosses": ["chữ Hán form of học"]}]},
    {"word": "hoc", "pos": "romanization", "lang_code": "vi", "senses": [{"glosses": ["romanization of học"]}]},
    {"word": "trống", "pos": "noun", "lang_code": "vi", "senses": [{}]},  # no glosses - not shown
    {"word": "tao", "pos": "pron", "lang_code": "vi",
     "senses": [{"glosses": ["I, me (used to inferiors)"], "tags": ["colloquial", "derogatory"]},
                {"glosses": ["I, me"], "tags": ["Southern", "transitive"]}]},
    {"word": "ăn quả nhớ kẻ trồng cây", "pos": "proverb", "lang_code": "vi",
     "senses": [{"glosses": ["when eating a fruit, remember who planted the tree"]}]},
    {"word": "mèo mả gà đồng", "pos": "phrase", "lang_code": "vi",
     "senses": [{"glosses": ["an illicit couple"], "tags": ["idiomatic"]}]},
]


def main():
    index = vd.parse(json.dumps(r, ensure_ascii=False) for r in RECORDS)
    assert set(index) == {"học sinh", "học", "bánh mì", "đi", "hôm nay", "tao", "ăn quả nhớ kẻ trồng cây",
                          "mèo mả gà đồng"}, set(index)
    tao = index["tao"][0]
    assert tao.sense_labels == (("colloquial", "derogatory"), ("Southern",)), tao.sense_labels  # not "transitive"
    assert index["ăn quả nhớ kẻ trồng cây"][0].idiom == "proverb" and index["mèo mả gà đồng"][0].idiom == "idiom"
    assert index["học"][0].idiom == "" and index["học"][0].sense_labels == ((),)
    student = index["học sinh"][0]
    assert student.senses == ("a student", "a pupil; a schoolchild/schoolkid") and student.chinese == "學生"
    assert index["học"][0].hanzi == ("學", "学") and index["bánh mì"][0].classifiers == ("chiếc", "ổ")
    assert [e.pos for e in index["đi"]] == ["verb", "adv"]
    print("OK: entries kept with their senses and their labels (Southern, slang...), Hán-Nôm, classifiers, "
          "Chinese origin, idiom/proverb; characters, romanizations and sense-less entries left out")

    folder = tempfile.mkdtemp()
    cache = os.path.join(folder, "vi-dict.json.gz")
    vd._write_cache(index, cache)
    assert vd._read_cache(cache) == index
    print("OK: the cut-down cache reads back the same")

    vd._index = index
    vd._loaded.set()
    assert vd.lookup("HỌC SINH") and vd.lookup(unicodedata.normalize("NFD", "học sinh")), "case, accent encoding"
    assert vd.lookup("học  sinh") and not vd.lookup("học sinh viên")
    print("OK: lookups whatever the case, accent encoding or spacing")

    line = "Hôm nay em là học sinh, đi ăn bánh mì."
    for hovered, word in [("sinh", "học sinh"), ("học", "học sinh"), ("mì", "bánh mì"), ("nay", "Hôm nay"),
                          ("em", "em"), ("đi", "đi")]:
        a, b = vd.word_span_at(line, line.index(hovered))
        assert line[a:b] == word, (hovered, line[a:b])
    assert vd.word_span_at(line, line.index(",")) is None and vd.word_span_at(line, line.index(" ")) is None
    across = "Em là học, sinh viên."  # punctuation between: not one word
    a, b = vd.word_span_at(across, across.index("sinh"))
    assert across[a:b] == "sinh", across[a:b]
    print("OK: a syllable looked up as the longest word around it - never across punctuation")

    assert vd.words_in("học sinh đi ăn bánh mì") == ["học sinh", "đi", "ăn", "bánh mì"], vd.words_in("học sinh đi ăn bánh mì")
    school = "Sáng nay em đi học rồi."
    a, b = vd.word_span_at(school, school.index("học"))
    assert school[a:b] == "học", "not a dictionary compound - just the syllable"
    vd.set_saved({"đi học"})
    a, b = vd.word_span_at(school, school.index("học"))
    assert school[a:b] == "đi học", "a saved compound is hovered whole"
    vd.set_saved(set())
    print("OK: a phrase split into its words; a compound you saved hovered whole, dictionary or not")

    assert vd.is_vietnamese(line) and vd.is_vietnamese("Ừ.") and not vd.is_vietnamese("I went to a café")
    assert not vd.is_vietnamese("anh oi")  # no letter only Vietnamese uses - not recognised
    a, b = dictionary.word_span_at(line, line.index("sinh"))
    assert line[a:b] == "học sinh", "the shared word_span_at uses it for Vietnamese lines"
    english = "I'm a student, I go to class."
    a, b = dictionary.word_span_at(english, english.index("student"))
    assert english[a:b] == "student", "English lines unchanged"
    print("OK: Vietnamese lines get Vietnamese words - through dictionary.word_span_at too; English unchanged")
    print("ALL OK")


if __name__ == "__main__":
    main()
