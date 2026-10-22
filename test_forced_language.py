"""Verifies a pinned conversation language doesn't stick to lines plainly
in English (translate_vc._heard_language) - the speech model labels every
line with the pinned one, so English got "translated" English to English
and never into a second language. Lines from a real session; no GPU needed."""

from types import SimpleNamespace

import translate_vc

CASES = [
    # (pinned, text, heard)
    ("vi", "This is crazy. We can't be that good, bro. You're glazing.", "en"),
    ("vi", "Yeah, you guys.", "en"),
    ("vi", "The translation.", "en"),
    ("vi", "Maybe check out your language resource of the accents.", "en"),
    ("vi", "thì anh hơn mình có nửa chấm.", "vi"),
    ("vi", "dạ.", "vi"),
    ("vi", "Ừ.", "vi"),
    ("vi", "haha.", "vi"),                       # one word - not enough to go on
    ("vi", "à, wait wait wait wait wait", "vi"),  # mixed - Vietnamese, translated as a whole
    ("vi", "thường thôi, man, eva nó cũng kinda bad from char", "vi"),
    ("vi", "đi đi", "vi"),
    ("zh", "Wait, who took the car", "en"),
    ("zh", "我觉得这个 culture concept 很 specific", "zh"),
    ("yue", "OK la", "en"),
]


def main():
    for pinned, text, expected in CASES:
        info = SimpleNamespace(language=pinned, languages=[pinned])
        got = translate_vc._heard_language(pinned, info, text)
        assert got == expected, (pinned, text, got, expected)
    detected = SimpleNamespace(language="en", languages=["en"])
    assert translate_vc._heard_language(None, detected, "thì anh hơn mình") == "en"  # nothing pinned: as detected
    print(f"OK: {len(CASES)} lines under a pinned language labelled by what was actually said")
    print("ALL OK")


if __name__ == "__main__":
    main()
