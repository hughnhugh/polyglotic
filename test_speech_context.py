"""Verifies the speech model's context words: Settings' "Names & slang"
(overlay_core.speech_context), and catching a transcription that's just
those words read back - no GPU needed."""

import overlay_core
import translate_vc


def main():
    overlay_core.speech_options["context"] = " HSK,, Mai ,bún bò Huế, "
    assert overlay_core.speech_context() == "HSK, Mai, bún bò Huế", overlay_core.speech_context()
    overlay_core.speech_options["context"] = ""
    assert overlay_core.speech_context() == ""
    print("OK: typed words passed on tidied - blanks and stray spaces dropped")

    # The speech model echoing its context back instead of transcribing.
    context = "Alex, Valorant, HSK, Kenji, Mai, 水瓶"
    assert translate_vc._echoes_context("Alex, Valorant, HSK, Kenji, Mai, 水瓶", context)
    assert translate_vc._echoes_context("Mai Kenji Valorant", context)
    assert not translate_vc._echoes_context("Alex!", context)
    assert not translate_vc._echoes_context("Hey Alex, did Mai go to the shops already?", context)
    assert not translate_vc._echoes_context("水瓶，你好", context)
    print("OK: a transcription that's just the context read back is caught; real mentions aren't")
    print("ALL OK")


if __name__ == "__main__":
    main()
