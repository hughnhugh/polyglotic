"""Verifies Cantonese is shown in Hong Kong Traditional characters
(translate_vc._cantonese_script) - including 系 -> 係 for "is", which a
plain Simplified-to-Traditional conversion gets wrong - no GPU needed."""

import translate_vc

CASES = [
    ("我觉得呢度真系好正", "我覺得呢度真係好正"),     # s2hk alone gives 真繫
    ("哦，亦都对呀，但系我个读错", "哦，亦都對呀，但係我個讀錯"),  # ...and leaves 但系
    ("你系唔系傻㗎", "你係唔係傻㗎"),
    ("佢系咪唔记得咗？", "佢係咪唔記得咗？"),
    ("我哋嘅关系好好", "我哋嘅關係好好"),
    ("我哋保持联系啦", "我哋保持聯繫啦"),            # 繫 (tie) kept
    ("绑紧鞋带系住佢", "綁緊鞋帶繫住佢"),
    ("呢个系统好差", "呢個系統好差"),                # 系 (system) kept
    ("中文系嘅学生", "中文系嘅學生"),
    ("佢系日系风格", "佢係日系風格"),
    ("搞掂咗未啊？你将我嘅嘢放去边啦？", "搞掂咗未啊？你將我嘅嘢放去邊啦？"),  # colloquial characters untouched
]


def main():
    for simplified, expected in CASES:
        got = translate_vc._cantonese_script(simplified)
        assert got == expected, (simplified, got, expected)
    already = "我唔知佢做緊咩，佢仲未食飯嚟㗎。"
    assert translate_vc._cantonese_script(already) == already
    print(f"OK: {len(CASES)} Cantonese lines converted to HK Traditional, 係/繫/系 right; Traditional input unchanged")
    print("ALL OK")


if __name__ == "__main__":
    main()
