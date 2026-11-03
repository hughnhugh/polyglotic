"""Verifies the side chat's plumbing without the model: the conversation
transcript it's given (the time now and who's spoken, a timestamped line
per turn, silences marked, Chinese names as Speaker A..., Me for your own
lines, the newest within the limit, no live captions), real
names back in the answer, where the transcript goes in the messages, and
the dictionary's pinyin under suggested replies."""

import time
from datetime import datetime, timedelta

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
    base = datetime(2026, 11, 2, 21, 0, 0)
    for caption, seconds in zip(board.captions, [0, 10, 20, 240, 285]):  # the live one's left as it is
        caption.time = base + timedelta(seconds=seconds)
    now = base + timedelta(minutes=5)
    labels = {}
    text = assistant.conversation(board.captions, labels, now=now)
    assert labels == {"我在东北玩泥巴  wǒ zài dōngběi wán níba": "Speaker A", "從火來暖": "Speaker B"}, labels
    clock, people, body = text.split("\n\n")
    assert clock == "Time now: Monday 2 November 2026, 21:05:00", clock
    assert people.splitlines()[1:] == [  # the latest to speak first
        "- Me: spoke once, at 21:04:45 (just now): “Sure, what time?”; English",
        "- Speaker A: last spoke at 21:04:00 (1 minute ago): “Speaker B，你说呢”; first spoke at 21:00:00; 3 lines; "
        "Mandarin and English",
        "- Speaker B: spoke once, at 21:00:20 (5 minutes ago): “This is so creepy”; English"], people
    lines = body.splitlines()
    assert lines == [
        "[21:00:00] Speaker A: 他今天早上给我打电话了 (English: He called me this morning) I'm freaking out",
        "[21:00:20] Speaker B: This is so creepy",
        "(4 minutes of silence)",
        "[21:04:00] Speaker A: Speaker B，你说呢",  # their name said - its garbled translation dropped
        "[21:04:45] Me: Sure, what time?"], body
    print("OK: the transcript - the time now, who's spoken (latest first: when, last words, lines, languages), a "
          "timestamped line per turn, silences marked, Chinese names as labels (said ones too), Me, no live captions")

    short = assistant.conversation(board.captions, labels, limit=len(lines[4]) + 5, now=now)
    assert short.split("\n\n")[1] == people, short  # everyone's still listed
    assert short.split("\n\n")[2:] == ["Only the latest part fits below: it starts at 21:04:45, and the earlier "
                                        "lines, from 21:00:00, are left out.", lines[4]], short
    fits_silence = len(lines[2]) + len(lines[3]) + len(lines[4]) + 3
    assert short != assistant.conversation(board.captions, labels, limit=fits_silence, now=now)
    assert assistant.conversation(board.captions, labels, limit=fits_silence, now=now).endswith(
        "left out.\n\n" + "\n".join(lines[3:])), "never starts on a silence"
    print("OK: the newest turns within the limit, with a note of what's left out")

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
