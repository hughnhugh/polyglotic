"""
Target languages the translator (Hy-MT2, see translator.py) supports, keyed
by the app's language codes - the same codes faster-whisper reports for a
detected source language, plus "zh-hant" for Traditional Chinese, which
Whisper itself never reports (it only ever says "zh").

Each entry is (display name, English prompt name, Chinese prompt name) -
Hy-MT2 is prompted in Chinese whenever Chinese is on either side of a
translation, in English otherwise, and expects the target language named
in that prompt's language (names as listed on the model card).

Source languages aren't restricted: Whisper can detect ~99 languages and
the translator's prompt doesn't name the source, so any detected language
is passed through as-is - only the target needs to be one of these.
"""

TARGET_LANGUAGES = {
    "en": ("English", "English", "英语"),
    "zh": ("Chinese (Simplified)", "Simplified Chinese", "简体中文"),
    "zh-hant": ("Chinese (Traditional)", "Traditional Chinese", "繁体中文"),
    # Spelled out so Hy-MT2 writes Cantonese the way it's read - Hong Kong
    # Traditional characters with the colloquial ones (嘅、咗、唔...) - rather
    # than in Simplified about half the time, as plain "粤语" got it to.
    "yue": ("Cantonese", "Cantonese, written in Hong Kong Traditional Chinese characters with colloquial "
            "Cantonese characters such as 嘅, 咗, 唔, 佢, 喺, 啲, 冇 and 嘢",
            "粤语，用香港繁体字书写，并使用粤语口语字（如嘅、咗、唔、佢、喺、啲、冇、嘢、乜、點解）"),
    "fr": ("French", "French", "法语"),
    "pt": ("Portuguese", "Portuguese", "葡萄牙语"),
    "es": ("Spanish", "Spanish", "西班牙语"),
    "ja": ("Japanese", "Japanese", "日语"),
    "tr": ("Turkish", "Turkish", "土耳其语"),
    "ru": ("Russian", "Russian", "俄语"),
    "ar": ("Arabic", "Arabic", "阿拉伯语"),
    "ko": ("Korean", "Korean", "韩语"),
    "th": ("Thai", "Thai", "泰语"),
    "it": ("Italian", "Italian", "意大利语"),
    "de": ("German", "German", "德语"),
    "vi": ("Vietnamese", "Vietnamese", "越南语"),
    "ms": ("Malay", "Malay", "马来语"),
    "id": ("Indonesian", "Indonesian", "印尼语"),
    "tl": ("Filipino", "Filipino", "菲律宾语"),
    "hi": ("Hindi", "Hindi", "印地语"),
    "pl": ("Polish", "Polish", "波兰语"),
    "cs": ("Czech", "Czech", "捷克语"),
    "nl": ("Dutch", "Dutch", "荷兰语"),
    "km": ("Khmer", "Khmer", "高棉语"),
    "my": ("Burmese", "Burmese", "缅甸语"),
    "fa": ("Persian", "Persian", "波斯语"),
    "gu": ("Gujarati", "Gujarati", "古吉拉特语"),
    "ur": ("Urdu", "Urdu", "乌尔都语"),
    "te": ("Telugu", "Telugu", "泰卢固语"),
    "mr": ("Marathi", "Marathi", "马拉地语"),
    "he": ("Hebrew", "Hebrew", "希伯来语"),
    "bn": ("Bengali", "Bengali", "孟加拉语"),
    "ta": ("Tamil", "Tamil", "泰米尔语"),
    "uk": ("Ukrainian", "Ukrainian", "乌克兰语"),
    "bo": ("Tibetan", "Tibetan", "藏语"),
    "kk": ("Kazakh", "Kazakh", "哈萨克语"),
    "mn": ("Mongolian", "Mongolian", "蒙古语"),
    "ug": ("Uyghur", "Uyghur", "维吾尔语"),
}

LANGUAGE_NAMES = {code: names[0] for code, names in TARGET_LANGUAGES.items()}

CHINESE_CODES = {"zh", "zh-hant", "yue"}
