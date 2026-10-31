"""
Your saved words: Chinese words saved from a lookup into a category - each
a name and a colour of your choosing - so wherever a saved word turns up
again, on an overlay or in the transcript, it's drawn in its category's
colour. Kept in vocab.json next to the app.

A saved word matches in either script (电脑 and 電腦), and inside longer
words too (游戏 in 打游戏) - but not across a word boundary: 个人 isn't
coloured in 这个人, which reads 这个/人 (see dictionary.segment).

Pure Python, no GUI - tested directly (test_vocab.py).
"""

import json
import os
import uuid
from datetime import date

import dictionary

ROOT = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(ROOT, "vocab.json")

# Clear of the overlays' own colours: white (what was said) and yellow
# (a translation).
DEFAULT_CATEGORIES = [
    {"id": "learning", "name": "Learning", "color": "#6fc3ff"},
    {"id": "tricky", "name": "Tricky", "color": "#ff7a9a"},
    {"id": "known", "name": "Know it", "color": "#8be38f"},
]
NEW_CATEGORY_COLORS = ["#c9a7ff", "#ffa25c", "#5fe0d0", "#ff9de2"]


class Vocab:
    def __init__(self, path=PATH):
        self.path = path
        self.categories: "list[dict]" = [dict(c) for c in DEFAULT_CATEGORIES]
        self.words: "dict[str, dict]" = {}   # simplified -> its record
        self.version = 0                     # bumped on every change, so views know to redraw
        self.on_change = None                # called after every change (the app redraws captions)
        try:
            with open(path, encoding="utf-8") as f:
                saved = json.load(f)
            self.categories = saved.get("categories") or self.categories
            self.words = saved.get("words") or {}
        except (OSError, ValueError):
            pass
        self._index()

    # -- reading ------------------------------------------------------------------

    def category(self, category_id):
        return next((c for c in self.categories if c["id"] == category_id), None)

    def get(self, word):
        """The saved record for `word` (in either script), or None."""
        key = self._forms.get(word)
        return self.words.get(key) if key else None

    def color_of(self, word):
        record = self.get(word)
        category = self.category(record["category"]) if record else None
        return category["color"] if category else None

    def in_category(self, category_id):
        return [w for w in self.words.values() if w["category"] == category_id]

    def spans(self, text):
        """[(start, end, colour)] for the saved words in `text` - the longest
        match at each point, left to right, never across a word boundary."""
        if not self._forms or not dictionary.has_han(text):
            return []
        words = dictionary.segment(text)
        out, i, n = [], 0, len(text)
        while i < n:
            for length in range(min(self._longest, n - i), 0, -1):
                end = i + length
                if text[i:end] in self._forms and _within_words(words, i, end):
                    color = self.color_of(text[i:end])
                    if color:
                        out.append((i, end, color))
                    i = end
                    break
            else:
                i += 1
        return out

    # -- changing -----------------------------------------------------------------

    def save(self, simplified, traditional, pinyin, meaning, category_id, sentence=""):
        """Saves a word into a category - or moves it there, if it's saved
        already (keeping when it was first added)."""
        old = self.words.get(simplified, {})
        self.words[simplified] = {
            "simplified": simplified, "traditional": traditional or simplified, "pinyin": pinyin,
            "meaning": meaning, "category": category_id, "sentence": sentence or old.get("sentence", ""),
            "added": old.get("added") or date.today().isoformat(),
        }
        self._changed()

    def move(self, word, category_id):
        record = self.get(word)
        if record is not None and record["category"] != category_id:
            record["category"] = category_id
            self._changed()

    def remove(self, word):
        key = self._forms.get(word, word)
        if self.words.pop(key, None) is not None:
            self._changed()

    def add_category(self, name, color=None):
        used = {c["color"] for c in self.categories}
        color = color or next((c for c in NEW_CATEGORY_COLORS if c not in used), NEW_CATEGORY_COLORS[0])
        category = {"id": uuid.uuid4().hex[:8], "name": name, "color": color}
        self.categories.append(category)
        self._changed()
        return category

    def update_category(self, category_id, name=None, color=None):
        category = self.category(category_id)
        if category is None:
            return
        if name is not None:
            category["name"] = name
        if color is not None:
            category["color"] = color
        self._changed()

    def remove_category(self, category_id):
        """Deletes a category and the words saved in it."""
        self.categories = [c for c in self.categories if c["id"] != category_id]
        self.words = {k: w for k, w in self.words.items() if w["category"] != category_id}
        self._changed()

    # -- internals ----------------------------------------------------------------

    def _index(self):
        self._forms = {}
        for key, record in self.words.items():
            self._forms[key] = key
            self._forms.setdefault(record.get("traditional") or key, key)
        self._longest = max((len(f) for f in self._forms), default=0)

    def _changed(self):
        self._index()
        self.version += 1
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"categories": self.categories, "words": self.words}, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except OSError as e:
            print(f"[vocab] couldn't save {self.path}: {e}")
        if self.on_change:
            self.on_change()


def _within_words(words, start, end):
    """Whether [start, end) doesn't cut across a word: every word it
    touches is either wholly inside it or wholly contains it."""
    for a, b in words:
        if b <= start or a >= end:
            continue
        if not (start <= a and b <= end) and not (a <= start and end <= b):
            return False
    return True
