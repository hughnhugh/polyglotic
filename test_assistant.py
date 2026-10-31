"""Verifies the side chat's plumbing without the model: the conversation
transcript it's given (a line per turn, Chinese names as Speaker A..., Me
for your own lines, the newest within the limit, no live captions), real
names back in the answer, where the transcript goes in the messages, and
the dictionary's pinyin under suggested replies."""

import time

import assistant
import captions
import dictionary
import test_dictionary
from ui.assistant_chat import build_messages, with_pinyin


def _record(speaker, native, en=None, you=False, group=None, final=True):
    translations = {"en": (en, True)} if en else {"en": (native, False)}
    return {"speaker": speaker, "you": you, "lang": "zh" if en else "en", "native": native,
            "translations": translations, "group": group, "final": final, "speech_end": time.time()}


def main():
    dictionary._index = dictionary._parse(test_dictionary.SAMPLE.splitlines() + [
        "從 从 [cong2] /from/", "來 来 [lai2] /to come/", "暖 暖 [nuan3] /warm/", "火 火 [huo3] /fire/"])
    board = captions.CaptionBoard()
    board.apply(_record("我在东北玩泥巴  wǒ zài dōngběi wán níba", "他今天早上给我打电话了",
                        en="He called me this morning"))
    board.apply(_record("我在东北玩泥巴  wǒ zài dōngběi wán níba", "I'm freaking out"))
    board.apply(_record("從火來暖", "This is so creepy"))
    board.apply(_record("我在东北玩泥巴  wǒ zài dōngběi wán níba", "从火来暖，你说呢", en="It warms me from the inside"))
    board.apply(_record("", "Sure, what time?", you=True))
    board.apply(_record("kaektus", "我还在说", en="I'm still talking", group="g1", final=False))
    labels = {}
    text = assistant.conversation(board.captions, labels)
    assert labels == {"我在东北玩泥巴  wǒ zài dōngběi wán níba": "Speaker A", "從火來暖": "Speaker B"}, labels
    head, body = text.split("\n\n")
    assert head == "People talking: Speaker A, Speaker B, Me", head
    lines = body.splitlines()
    assert len(lines) == 4, text
    assert lines[0].endswith("Speaker A: 他今天早上给我打电话了 (English: He called me this morning) I'm freaking out"), \
        lines[0]
    assert lines[2].endswith("Speaker A: Speaker B，你说呢"), lines[2]  # their name said - its garbled translation dropped
    assert lines[3].endswith("Me: Sure, what time?"), lines[3]
    assert assistant.conversation(board.captions, labels, limit=len(lines[3]) + 5) == f"People talking: Me\n\n{lines[3]}"
    print("OK: the transcript - a turn per line, Chinese names as labels (said ones too), Me, newest within the "
          "limit, no live captions")

    answer = "**Speaker A** told a story, and speaker b found it creepy. Speaker AB isn't anyone. **Me** asked me."
    assert assistant.restore_names(answer, labels) == ("**我在东北玩泥巴  wǒ zài dōngběi wán níba** told a story, and "
                                                       "從火來暖 found it creepy. Speaker AB isn't anyone. **You** asked "
                                                       "me."), assistant.restore_names(answer, labels)
    print("OK: real names back in the answer")

    messages = build_messages("what's going on?", text, [{"role": "user", "content": "earlier"},
                                                         {"role": "assistant", "content": "reply"}])
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert "[Conversation]" not in messages[0]["content"] and messages[1]["content"] == "earlier"
    assert text in messages[-1]["content"] and messages[-1]["content"].endswith("what's going on?")
    assert build_messages("hi", "", [])[-1]["content"] == "hi"
    print("OK: the transcript goes with the latest question, right before it")

    replies = "Here you go:\n- 这个电脑 — this computer\n2. **电脑** - computer\n- 人"
    assert with_pinyin(replies) == ("Here you go:\n\n**这个电脑**  \n*zhè ge diàn nǎo*  \nthis computer\n\n"
                                    "**电脑**  \n*diàn nǎo*  \ncomputer\n\n**人**  \n*rén*"), with_pinyin(replies)
    print("OK: each suggested reply - Chinese, the dictionary's pinyin, its meaning")
    print("ALL OK")


if __name__ == "__main__":
    main()
