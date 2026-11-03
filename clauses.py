"""
Splits a caption's text into clauses: the rows a caption is shown in on the
overlays, and the units it's translated in, so each row's translation is
its own and a row whose words didn't change keeps its translation (see
docs/line-by-line-captions.md).

A clause ends at a sentence's end (。！？!? or a full stop), and at a comma,
semicolon or colon once it's long enough to stand as a line - "嗯，" or
"So," on its own isn't one, and joins what follows. Punctuation stays with
its clause, so the clauses join back into the text (captions.join_text).

Pure Python, no GUI - tested directly (test_clauses.py).
"""

import re

# Sizes in rough units of meaning: a Chinese/Japanese/Korean character is
# one, a word in a spaced language two (an English word carries about as
# much as two characters).
MIN_SOFT_BREAK = 8   # a comma ends a clause only once it's this long - 8 characters, 4 words
# Anything shorter joins the next clause (or, at the end, the one before):
MIN_CLAUSE = 4       # characters, in Chinese/Japanese/Korean - "然后走了。" stands, "对对对。" joins
MIN_CLAUSE_WORDS = 3  # words, in a clause without any - "for sure." joins

_HARD = "。！？!?"
_SOFT = "，,；;：:"
_CLOSERS = "\"'”’」』）)】]》"
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]")
_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")  # "I'm" is one word
# A break: a run of ending punctuation (?! ... 。」), or a full stop that
# isn't inside a number (3.5) - each with any closing quotes/brackets after it.
_BREAK = re.compile(rf"(?:[{_HARD}{_SOFT}…]+|\.+(?!\d))[{re.escape(_CLOSERS)}]*")


def size(text):
    """How much `text` says, in units: CJK characters, plus two per word of
    anything else."""
    cjk = len(_CJK.findall(text))
    return cjk + 2 * len(_WORD.findall(_CJK.sub(" ", text)))


def _short(clause):
    if _CJK.search(clause):
        return size(clause) < MIN_CLAUSE
    return len(_WORD.findall(clause)) < MIN_CLAUSE_WORDS


def _pieces(text):
    """(piece, kind) for `text` cut after every break - kind "hard" (a
    sentence's end), "soft" (a comma and the like) or None (the rest)."""
    out, at = [], 0
    for match in _BREAK.finditer(text):
        mark = match.group().rstrip(_CLOSERS)
        if mark.startswith(".") and match.end() < len(text) and not text[match.end()].isspace():
            continue  # "e.g", "U.S" - a full stop only ends one before a space (or the end)
        kind = "hard" if any(ch in _HARD or ch == "." for ch in mark) else "soft"
        if "…" in mark and not any(ch in _HARD for ch in mark):
            kind = "soft"  # trailing off - a pause, not an end
        out.append((text[at:match.end()], kind))
        at = match.end()
    if at < len(text):
        out.append((text[at:], None))
    return out


def split(text):
    """`text` as clauses, in order - one, when it's short or has no
    punctuation to split at. Never empty for non-empty text."""
    text = (text or "").strip()
    if not text:
        return []
    clauses, current = [], ""
    for piece, kind in _pieces(text):
        current += piece
        if kind == "hard" or (kind == "soft" and size(current) >= MIN_SOFT_BREAK):
            clauses.append(current.strip())
            current = ""
    if current.strip():
        clauses.append(current.strip())
    merged = []
    for clause in clauses:  # a short one joins the next
        if merged and _short(merged[-1]):
            merged[-1] = _join(merged[-1], clause)
        else:
            merged.append(clause)
    if len(merged) > 1 and _short(merged[-1]):  # ...or, last, the one before
        merged[-2:] = [_join(merged[-2], merged[-1])]
    return merged


def _join(a, b):
    from captions import join_text
    return join_text([a, b])
