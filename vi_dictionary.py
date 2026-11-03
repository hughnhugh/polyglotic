"""
A Vietnamese dictionary for word lookups - for Vietnamese what dictionary.py
(CC-CEDICT) is for Chinese: Wiktionary's Vietnamese entries, from English
Wiktionary (CC BY-SA), as exported by kaikki.org - data/vi-wiktionary.jsonl.gz,
downloaded the first time if it's missing.

Vietnamese is written a syllable at a time, spaces between, but a word can
be several syllables (học sinh, "student") - so a syllable is looked up as
the longest dictionary word around it (word_span_at), the way a Chinese
character is looked up as the dictionary word it's in.

Each entry has its part of speech, its senses in English, its Hán-Nôm
characters (học - 學) and, for a Sino-Vietnamese word, the Chinese it comes
from (học sinh - 學生): a bridge to Chinese for someone learning both.

The export (13MB, 83MB unpacked) is cut down once to what's shown, in
data/vi-dict.json.gz, and that's what's loaded after - quicker.

Pure Python, no GUI - tested in test_vi_dictionary.py.
"""

import gzip
import json
import os
import re
import threading
import unicodedata
import urllib.request
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(ROOT, "data", "vi-wiktionary.jsonl.gz")
URL = "https://kaikki.org/dictionary/Vietnamese/kaikki.org-dictionary-Vietnamese.jsonl.gz"
CACHE = os.path.join(ROOT, "data", "vi-dict.json.gz")
CACHE_VERSION = 2       # 2: senses' labels, idioms
MAX_SYLLABLES = 4       # the longest word looked for around a syllable
MAX_SENSES = 8          # per entry, kept from the export
SKIPPED_POS = {"character", "romanization"}  # a Hán-Nôm character's own entry; a transliteration
# Wiktionary's sense tags worth a badge in a lookup - where a word's said, and
# how it comes across - by how they're shown. The rest (transitive,
# alt-of...) are grammar or bookkeeping.
SENSE_LABELS = {
    "Southern": "Southern", "Northern": "Northern", "North": "Northern", "Central": "Central",
    "colloquial": "colloquial", "informal": "informal", "slang": "slang", "Internet": "Internet",
    "formal": "formal", "literary": "literary", "dated": "dated", "archaic": "archaic", "obsolete": "obsolete",
    "rare": "rare", "humorous": "humorous", "euphemistic": "euphemistic", "figuratively": "figurative",
    "neologism": "new word", "derogatory": "derogatory", "vulgar": "vulgar",
}

# Letters only Vietnamese uses - a line with one is read as Vietnamese.
# Not á/à/é...: French and Spanish have those too.
_VIETNAMESE_LETTERS = set("ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ")
_SYLLABLE = re.compile(r"[^\W\d_]+")
_CJK = r"[㐀-䶿一-鿿豈-﫿\U00020000-\U0003ffff]"
_SINO = re.compile(rf"Sino-Vietnamese word from ({_CJK}+)")


@dataclass(frozen=True)
class Entry:
    word: str
    pos: str                  # noun, verb, adj... - Wiktionary's
    senses: tuple             # in English
    hanzi: tuple = ()         # its Hán-Nôm characters
    chinese: str = ""         # the Chinese a Sino-Vietnamese word comes from
    classifiers: tuple = ()   # for a noun: chiếc, cái...
    sense_labels: tuple = ()  # per sense: its SENSE_LABELS ("Southern", "slang"...)
    idiom: str = ""           # "idiom" or "proverb", if it's one


_index: "dict[str, list[Entry]]" = {}
_saved: "set[str]" = set()   # your saved words' keys (vocab.py): words too, when hovering, dictionary or not
_loaded = threading.Event()
_failed = None


def set_saved(keys):
    """Your saved Vietnamese words - a compound you saved that the
    dictionary hasn't got (đi học) is then hovered whole, like one it has."""
    global _saved
    _saved = set(keys)


def key(text):
    """How a word is looked up: composed (NFC), lower case, single spaces."""
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


def _entry(record):
    """An Entry from one line of the export, or None for one not shown."""
    if record.get("pos") in SKIPPED_POS or record.get("lang_code") not in (None, "vi"):
        return None
    senses, labels, idiomatic = [], [], False
    for sense in record.get("senses", []):
        tags = sense.get("tags") or []
        idiomatic = idiomatic or "idiomatic" in tags
        glosses = sense.get("glosses") or sense.get("raw_glosses")
        if glosses:
            gloss = glosses[-1].strip()  # the most specific - a sub-sense's own
            if gloss and gloss not in senses:
                senses.append(gloss)
                labels.append(tuple(dict.fromkeys(SENSE_LABELS[tag] for tag in tags if tag in SENSE_LABELS)))
    if not senses:
        return None
    pos = record.get("pos", "")
    forms = record.get("forms", [])
    sino = _SINO.search(record.get("etymology_text") or "")
    return Entry(
        word=unicodedata.normalize("NFC", record["word"]), pos=record.get("pos", ""),
        senses=tuple(senses[:MAX_SENSES]),
        hanzi=tuple(dict.fromkeys(f["form"] for f in forms if "CJK" in (f.get("tags") or []))),
        chinese=sino.group(1) if sino else "",
        classifiers=tuple(dict.fromkeys(f["form"] for f in forms if "classifier" in (f.get("tags") or []))),
        sense_labels=tuple(labels[:MAX_SENSES]),
        idiom="proverb" if pos == "proverb" else "idiom" if idiomatic else "",
    )


def parse(lines):
    """{key: [Entry]} from the export's lines (JSON, one entry each)."""
    index = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = _entry(json.loads(line))
        except (ValueError, KeyError):
            continue
        if entry:
            index.setdefault(key(entry.word), []).append(entry)
    return index


def _write_cache(index, path=CACHE):
    data = {"version": CACHE_VERSION,
            "entries": [[e.word, e.pos, list(e.senses), list(e.hanzi), e.chinese, list(e.classifiers),
                         [list(labels) for labels in e.sense_labels], e.idiom]
                        for entries in index.values() for e in entries]}
    with gzip.open(path + ".part", "wt", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(path + ".part", path)


def _read_cache(path=CACHE):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("version") != CACHE_VERSION:
        return None
    index = {}
    for word, pos, senses, hanzi, chinese, classifiers, labels, idiom in data["entries"]:
        index.setdefault(key(word), []).append(Entry(word, pos, tuple(senses), tuple(hanzi), chinese,
                                                     tuple(classifiers), tuple(tuple(x) for x in labels), idiom))
    return index


def load(path=PATH, cache=CACHE):
    """Reads the dictionary - from the cut-down cache if it's there and up
    to date, else from the export (downloading it if it's missing), writing
    the cache. Call from a background thread; ready() says when it's done."""
    global _index, _failed
    try:
        index = None
        if os.path.exists(cache) and (not os.path.exists(path) or os.path.getmtime(cache) >= os.path.getmtime(path)):
            index = _read_cache(cache)
        if index is None:
            if not os.path.exists(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                urllib.request.urlretrieve(URL, path + ".part")
                os.replace(path + ".part", path)
            with gzip.open(path, "rt", encoding="utf-8") as f:
                index = parse(f)
            _write_cache(index, cache)
        _index = index
    except Exception as e:  # noqa: BLE001 - offline, corrupt file... lookups just find nothing
        _failed = f"{type(e).__name__}: {e}"
        print(f"[vi_dictionary] couldn't load the Vietnamese dictionary: {_failed}")
    finally:
        _loaded.set()


def start():
    threading.Thread(target=load, daemon=True, name="vi-dictionary").start()


def ready():
    return _loaded.is_set() and bool(_index)


def failed():
    """Why the dictionary isn't available, or None."""
    return _failed


# -- lookup ---------------------------------------------------------------------

def is_vietnamese(text):
    """True if `text` has a letter only Vietnamese uses (ơ, ư, đ, ạ...) - a
    line without one ("anh oi") isn't recognised, but nearly every line has
    one."""
    return any(c in _VIETNAMESE_LETTERS for c in unicodedata.normalize("NFC", text).lower())


def lookup(word):
    """Every entry for `word` - each part of speech it's used as."""
    return _index.get(key(word), [])


def word_span_at(text, index):
    """(start, end) of the Vietnamese word at `index`: the longest
    dictionary word - up to MAX_SYLLABLES syllables, separated only by
    spaces - around the syllable there ("học sinh" for "sinh"), else that
    syllable alone. None on a space, digit or punctuation."""
    syllables = [m.span() for m in _SYLLABLE.finditer(text)]
    k = next((n for n, (a, b) in enumerate(syllables) if a <= index < b), None)
    if k is None:
        return None
    best = None
    for first in range(max(0, k - MAX_SYLLABLES + 1), k + 1):
        for last in range(min(len(syllables), first + MAX_SYLLABLES) - 1, k - 1, -1):
            if last > first and any(text[syllables[n][1]:syllables[n + 1][0]].strip()
                                    for n in range(first, last)):
                continue  # punctuation between - not one word
            phrase = key(text[syllables[first][0]:syllables[last][1]])
            if (last - first) and (phrase in _index or phrase in _saved):
                if best is None or last - first > best[1] - best[0]:
                    best = (first, last)
                break  # the longest from this first syllable
    first, last = best or (k, k)
    return syllables[first][0], syllables[last][1]


def words_in(phrase):
    """The words a phrase is made of, left to right - the longest
    dictionary word at each point (bánh mì / thịt), a syllable on its own
    where there's none: ["bánh mì", "thịt"]."""
    syllables = phrase.split()
    out, i = [], 0
    while i < len(syllables):
        for n in range(min(MAX_SYLLABLES, len(syllables) - i), 0, -1):
            if n == 1 or key(" ".join(syllables[i:i + n])) in _index:
                out.append(" ".join(syllables[i:i + n]))
                i += n
                break
    return out
