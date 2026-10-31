"""
Chinese word lookup from CC-CEDICT (https://cc-cedict.org - CC BY-SA 4.0),
the free community Chinese-English dictionary: ~125k entries, each with its
Traditional and Simplified forms, pinyin and English definitions. Words,
set phrases and chengyu (4-character idioms) are all entries, so a caption
line is split into the longest words the dictionary knows (segment()) and
hovering any character looks up the whole word it's part of.

Fetched into data/ by setup.ps1 - or on first use, if it's missing (4MB).
Loaded in the background at startup (~1s); lookups before then find
nothing rather than block.

Pure Python, no GUI - tested directly (test_dictionary.py).
"""

import gzip
import itertools
import math
import os
import re
import threading
import urllib.request
from collections import Counter
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(ROOT, "data", "cedict.txt.gz")
URL = "https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz"
# Word frequencies (jieba's dictionary, MIT) - CC-CEDICT has none, and an
# English search needs them to put 支持 before 赡 for "support".
FREQUENCY_PATH = os.path.join(ROOT, "data", "word-frequency.txt")
FREQUENCY_URL = "https://raw.githubusercontent.com/fxsjy/jieba/master/jieba/dict.txt"

MAX_WORD = 20  # characters - CC-CEDICT's longest entries are whole proverbs

# Function words read differently alone than in the words that contain
# them - their everyday reading first (see _parse).
FUNCTION_READINGS = {"还": "hái", "還": "hái", "都": "dōu", "了": "le", "着": "zhe", "著": "zhe", "得": "de",
                     "的": "de", "吧": "ba", "呢": "ne", "啊": "a", "吗": "ma", "嗎": "ma", "嘛": "ma", "呀": "ya",
                     "么": "me", "麼": "me", "哪": "nǎ"}

_LINE = re.compile(r"^(\S+) (\S+) \[([^\]]*)\] /(.*)/\s*$")
_VARIANT = re.compile(r"^(old |archaic |unofficial |Japanese |erroneous )?variant of ")


@dataclass(frozen=True)
class Entry:
    traditional: str
    simplified: str
    pinyin: str        # with tone marks: "diàn nǎo"
    definitions: tuple

    @property
    def idiom(self):
        return any("(idiom)" in d for d in self.definitions)

    @property
    def variant_only(self):
        """Just "variant of X" - a form you'd look up under X."""
        return all(_VARIANT.match(d) for d in self.definitions)


# -- pinyin ---------------------------------------------------------------------

_MARKS = {"a": "āáǎà", "e": "ēéěè", "i": "īíǐì", "o": "ōóǒò", "u": "ūúǔù", "ü": "ǖǘǚǜ"}


def _mark_syllable(syllable):
    """"nao3" -> "nǎo", "lu:4" -> "lǜ", "r5" -> "r". The mark goes on a or e
    if there is one, on the o of "ou", and otherwise on the last vowel."""
    match = re.fullmatch(r"([A-Za-z:]+)([1-5])", syllable)
    if not match:
        return syllable.replace("u:", "ü").replace("U:", "Ü")
    letters, tone = match.group(1).replace("u:", "ü").replace("U:", "Ü"), int(match.group(2))
    if tone == 5:
        return letters
    lower = letters.lower()
    if "a" in lower:
        at = lower.index("a")
    elif "e" in lower:
        at = lower.index("e")
    elif "ou" in lower:
        at = lower.index("o")
    else:
        at = max((i for i, c in enumerate(lower) if c in "iouü"), default=-1)
        if at < 0:
            return letters  # "m2", "ng2" - no vowel to mark
    marked = _MARKS[lower[at]][tone - 1]
    return letters[:at] + (marked.upper() if letters[at].isupper() else marked) + letters[at + 1:]


def mark_pinyin(numbered):
    """CC-CEDICT's numbered pinyin ("dian4 nao3") with tone marks
    ("diàn nǎo")."""
    return " ".join(_mark_syllable(s) for s in numbered.split())


# -- loading --------------------------------------------------------------------

_index: "dict[str, list[Entry]]" = {}   # simplified and traditional forms -> entries
_loaded = threading.Event()
_failed = None


def _parse(lines):
    index = {}
    for line in lines:
        if line.startswith("#"):
            continue
        match = _LINE.match(line)
        if not match:
            continue
        trad, simp, pinyin, defs = match.groups()
        entry = Entry(trad, simp, mark_pinyin(pinyin), tuple(d for d in defs.split("/") if d))
        index.setdefault(simp, []).append(entry)
        if trad != simp:
            index.setdefault(trad, []).append(entry)
    # A character's everyday reading: CC-CEDICT has no frequencies, but how
    # many words use each reading is a good stand-in (足: zú in 329 words,
    # jù in none) - except for function words, said alone in their own
    # reading (还 hái, though huán is in more words).
    uses = Counter()
    for key, entries in index.items():
        for e in entries:
            syllables = e.pinyin.lower().split()
            # each entry once - under its simplified form
            if key == e.simplified and len(key) > 1 and len(syllables) == len(key):
                for form in {e.simplified, e.traditional}:
                    uses.update(zip(form, syllables))
    for key, entries in index.items():
        # Real definitions before "variant of", common words before proper
        # nouns (capitalized pinyin - names, places), then the reading
        # likeliest in speech.
        entries.sort(key=lambda e: (e.variant_only, e.pinyin[:1].isupper(),
                                    e.pinyin != FUNCTION_READINGS.get(key, e.pinyin),
                                    -uses[key, e.pinyin.lower()]))
    return index


def load(path=PATH):
    """Reads the dictionary - downloading it first if it's missing. Call
    from a background thread; ready() says when it's done."""
    global _index, _failed
    try:
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            urllib.request.urlretrieve(URL, path + ".part")
            os.replace(path + ".part", path)
        with gzip.open(path, "rt", encoding="utf-8") as f:
            _index = _parse(f)
        _english_index()  # built here, in the background, rather than on the first search
        _load_frequency()
    except Exception as e:  # noqa: BLE001 - offline, corrupt file... lookups just find nothing
        _failed = f"{type(e).__name__}: {e}"
        print(f"[dictionary] couldn't load CC-CEDICT: {_failed}")
    finally:
        _loaded.set()


def start():
    threading.Thread(target=load, daemon=True, name="dictionary").start()


def ready():
    return _loaded.is_set() and bool(_index)


def failed():
    """Why the dictionary isn't available, or None."""
    return _failed


# -- lookup ---------------------------------------------------------------------

def is_han(char):
    return ("㐀" <= char <= "鿿" or "豈" <= char <= "﫿" or "\U00020000" <= char <= "\U0002ffff"
            or char == "〇")


def has_han(text):
    return any(is_han(c) for c in text)


def lookup(word):
    """Every entry for `word` (in either script), everyday readings first."""
    return _index.get(word, [])


def segment(text):
    """[(start, end)] covering `text`: each run of Chinese characters split
    into dictionary words - as few as possible, so whole phrases and chengyu
    stay together; among equally few, the most evenly sized (研究/生命/起源,
    not 研究生/命/起源), then the longest first (这个/人, not 这/个人).
    Everything else (Latin, digits, punctuation) is one span per run, not
    looked up."""
    spans, i, n = [], 0, len(text)
    while i < n:
        j = i
        if is_han(text[i]):
            while j < n and is_han(text[j]):
                j += 1
            spans += [(i + a, i + b) for a, b in _segment_run(text[i:j])]
        else:
            while j < n and not is_han(text[j]):
                j += 1
            spans.append((i, j))
        i = j
    return spans


def _segment_run(run):
    n = len(run)
    # best[i]: (words, sum of squared lengths, previous cut) for run[:i]
    best = [(0, 0, -1)] + [None] * n
    for end in range(1, n + 1):
        for start in range(max(0, end - MAX_WORD), end):
            if best[start] is None or (end - start > 1 and run[start:end] not in _index):
                continue
            words, squares, _ = best[start]
            candidate = (words + 1, squares + (end - start) ** 2, start)
            # <=: on a tie the later start wins - the longer earlier word
            if best[end] is None or candidate[:2] <= best[end][:2]:
                best[end] = candidate
    cuts, end = [], n
    while end > 0:
        start = best[end][2]
        cuts.append((start, end))
        end = start
    return cuts[::-1]


def word_at(text, index):
    """(start, end) of the dictionary word in `text` that the character at
    `index` belongs to - None for anything but a Chinese character."""
    if not (0 <= index < len(text)) or not is_han(text[index]):
        return None
    for start, end in segment(text):
        if start <= index < end:
            return start, end
    return None


def inner_words(word):
    """The shorter dictionary words inside `word` (游戏 in 打游戏), longest
    first - each once, in order of where they start."""
    found = []
    for length in range(len(word) - 1, 1, -1):
        for start in range(len(word) - length + 1):
            part = word[start:start + length]
            if part in _index and part not in found:
                found.append(part)
    return found


_WORD = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*")   # a word in a spaced language: letters, it's, well-known


def word_span_at(text, index):
    """(start, end) of the word at `index` in any language: a dictionary
    word for Chinese (word_at), a run of letters for the rest - None on a
    space, digit or punctuation."""
    if not 0 <= index < len(text):
        return None
    if is_han(text[index]):
        return word_at(text, index)
    for match in _WORD.finditer(text):
        if match.start() <= index < match.end():
            return None if any(is_han(c) for c in match.group()) else match.span()
    return None


def snap_to_words(text, start, end):
    """start-end widened to whole words at either end where it cuts into a
    word in a spaced language ("rdoing i" -> "overdoing it"); Chinese stays
    by the character - part of a word is a fair thing to look up."""
    if start < end:
        span = word_span_at(text, start)
        if span and not is_han(text[start]):
            start = span[0]
        span = word_span_at(text, end - 1)
        if span and not is_han(text[end - 1]):
            end = span[1]
    return start, end


# -- English to Chinese -------------------------------------------------------------

_STOP = {"a", "an", "the", "to", "of", "it", "its", "is", "am", "are", "was", "were", "be", "been", "will", "would",
         "have", "has", "and", "or", "in", "on", "at", "for", "with", "this", "that", "i", "you", "me", "my", "your",
         "s", "so", "just", "like", "do"}
# A reading's first meaning starting so: rarely the one being said
_OBSCURE = ("(archaic)", "(literary)", "(old)", "(dialect)", "(Cantonese)", "old variant", "surname", "used in")
_english = {}      # English word -> entries whose definitions use it
_english_of = None  # the _index it was built from
_frequency = {}    # Simplified word -> how often it's used (jieba's counts), for words CC-CEDICT has


def _load_frequency(path=FREQUENCY_PATH):
    """Reads the word frequencies - downloading them first if they're
    missing. Without them, searches still work, just ranked less well."""
    global _frequency
    try:
        if not os.path.exists(path):
            urllib.request.urlretrieve(FREQUENCY_URL, path + ".part")
            os.replace(path + ".part", path)
        frequency = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                word, count = line.split(" ", 2)[:2]
                if word in _index:
                    frequency[word] = int(count)
        _frequency = frequency
    except (OSError, ValueError) as e:
        print(f"[dictionary] no word frequencies ({type(e).__name__}: {e}) - English searches ranked by meaning only")


def _english_index():
    global _english, _english_of
    if _english_of is not _index:
        english = {}
        for key, entries in _index.items():
            for e in entries:
                if key != e.simplified or e.variant_only:  # each entry once; "variant of" never
                    continue
                words = set()
                for d in e.definitions:
                    if not d.startswith("CL:"):
                        words.update(re.findall(r"[a-z]+", d.lower()))
                for word in words:
                    english.setdefault(word, []).append(e)
        _english, _english_of = english, _index
    return _english


def _plain(definition):
    """A definition (or English query) compared plainly: lower case, no
    "(coll.)"-style notes, no articles, no leading "to"."""
    text = re.sub(r"\([^)]*\)", " ", definition.lower())
    text = " ".join(w for w in re.findall(r"[a-z']+", text) if w not in ("a", "an", "the"))
    return re.sub(r"^to ", "", text)


def _stems(word):
    """The word and what it may be in a definition: games -> game,
    overdoing -> overdo, lagging -> lag."""
    forms = [word]
    for suffix, back in (("ies", "y"), ("es", ""), ("s", ""), ("ing", ""), ("ing", "e"), ("ed", ""), ("ed", "e")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            stem = word[:-len(suffix)] + back
            forms.append(stem)
            if len(stem) > 3 and stem[-1] == stem[-2]:  # lagging -> lagg -> lag
                forms.append(stem[:-1])
    return forms


def search_english(query, limit=8):
    """Chinese words for an English word or phrase, from CC-CEDICT's
    definitions: a definition that's exactly the phrase first ("game" -
    游戏), then ones starting or ending with it, then containing it, then
    with all its words; everyday words before names; then the most used
    words (_frequency - 累 before 骀 for "tired", and before 辋 "tire",
    which "tired" also tries), favouring a word's main meaning over a side
    one; then the phrase as given before a stemmed form of it."""
    index = _english_index()
    text = query.lower().replace("’", "'")
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"'(re|m|s|ll|ve|d)\b", lambda m: {"re": " are", "m": " am", "ll": " will", "ve": " have",
                                                       "d": " would"}.get(m.group(1), ""), text)
    words = re.findall(r"[a-z]+", text)
    content = [w for w in words if w not in _STOP] or words
    if not content:
        return []
    found = None
    for word in content:
        entries = {id(e): e for form in _stems(word) for e in index.get(form, ())}
        found = entries if found is None else {k: e for k, e in found.items() if k in entries}
    # The phrase, and without what it starts with before its first real word
    # ("I don't know" -> "not know"), each also with its words stemmed.
    first = words.index(content[0])
    phrases = {}
    for start in (0, first):
        tail = words[start:]
        options = [_stems(w) if w in content else [w] for w in tail][:6]
        for variant in itertools.islice(itertools.product(*options), 64):
            phrase = _plain(" ".join(variant))
            stemmed = any(form != word for form, word in zip(variant, tail))
            if phrase and (phrase not in phrases or not stemmed):
                phrases[phrase] = stemmed
    ranked = []
    for e in (found or {}).values():
        best = None
        # each sense on its own - a definition is often several: "safe; secure"
        senses = [(position, sense) for position, d in enumerate(e.definitions) for sense in d.split(";")]
        for position, sense in senses:
            plain = _plain(sense)
            if not plain:
                continue
            for phrase, stemmed in phrases.items():
                if plain == phrase:
                    score = 0
                elif plain.startswith(phrase + " ") or plain.endswith(" " + phrase):
                    score = 1
                elif f" {phrase} " in f" {plain} ":
                    score = 2
                elif all(any(f" {form} " in f" {plain} " for form in _stems(w)) for w in content):
                    score = 3
                else:
                    continue
                key = (score, stemmed, position)
                best = key if best is None or key < best else best
        if best is not None:
            # How common, less a little for each meaning before the match:
            # 支持 (to support, its 2nd) still before 赡 (to support, its 1st).
            # The frequency is the word's, so not an obscure reading's - not
            # 那 nuó's (archaic: beautiful), which 那 nà is so common for.
            obscure = e.definitions[0].startswith(_OBSCURE)
            common = math.log2((0 if obscure else _frequency.get(e.simplified, 0)) + 1) - 1.5 * best[2]
            ranked.append(((best[0], e.pinyin[:1].isupper(), -common, best[1], len(e.simplified)), e))
    ranked.sort(key=lambda r: r[0])
    return [e for _, e in ranked[:limit]]


_BOLD = re.compile(r"\*\*([^*\n]+)\*\*")
# "(nǐ hǎo)" - tone marks tell pinyin from an English gloss like "(support)"
_PINYIN_GLOSS = re.compile(r"\s*\((?=[^)]*[āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ])[A-Za-züÜāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ\s'’-]+\)")


def with_bold_pinyin(markdown):
    """A model's markdown answer with the dictionary's pinyin, in italics,
    after each **bold** Chinese word - and any pinyin it wrote itself
    (in brackets) taken out: a small model's tones are often wrong."""
    def add(match):
        inner = match.group(1)
        return f"{match.group(0)} *{pinyin_of(inner)}*" if has_han(inner) else match.group(0)
    return _BOLD.sub(add, _PINYIN_GLOSS.sub("", markdown))


def pinyin_of(text):
    """Best-guess pinyin for any Chinese text: each word's everyday
    reading - for a phrase the dictionary doesn't have as a whole."""
    out = []
    for start, end in segment(text):
        part = text[start:end]
        if not has_han(part):
            continue
        entries = lookup(part)
        out.append(entries[0].pinyin if entries else "?" * len(part))
    return " ".join(out)
