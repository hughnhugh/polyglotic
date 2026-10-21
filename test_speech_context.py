"""Verifies the speech model's context words: Settings' "Names & slang" plus
the Discord channel members' names (overlay_core.speech_context), following
people joining, leaving and speaking - no GPU/Discord needed."""

import time

import discord_bridge
import overlay_core
import translate_vc


def _sync(members):
    discord_bridge._handle_message({"event": "channel_sync", "members": members})


def main():
    discord_bridge._connected_clients = 1
    overlay_core.speech_options.update(context="HSK, marina", auto_names=True)

    _sync({"1": "Caleb", "2": "marina", "3": "🍁Kayo🍁", "4": "?"})
    assert overlay_core.speech_context() == "HSK, marina, Caleb, Kayo", overlay_core.speech_context()
    print("OK: channel names added after the typed words - deduped, decorations stripped, '?' left out")

    discord_bridge._handle_message({"event": "speaking_start", "userId": "9", "username": "StinkyTofu",
                                    "ts": time.time() * 1000})
    assert "StinkyTofu" in overlay_core.speech_context()
    print("OK: someone heard speaking who joined after the last member list is added")

    _sync({"1": "Caleb"})
    context = overlay_core.speech_context()
    assert "Kayo" not in context and "StinkyTofu" in context, context
    print("OK: a refreshed member list drops who left; recent speakers stay for a while")

    discord_bridge._last_heard["9"] = time.time() - 3600
    assert "StinkyTofu" not in overlay_core.speech_context()
    print("OK: a speaker not heard for over 10 minutes drops off")

    overlay_core.speech_options["auto_names"] = False
    assert overlay_core.speech_context() == "HSK, marina"
    discord_bridge._connected_clients = 0
    overlay_core.speech_options["auto_names"] = True
    assert overlay_core.speech_context() == "HSK, marina"
    print("OK: no names when turned off, or when the plugin isn't connected")

    # Display names that read like statuses are left out; at most
    # MAX_CONTEXT_NAMES are used, most recently heard first.
    for name, expected in [("silly student try hard", None), ("gap life 躺平 chữa lành", None),
                           ("蟹黄宝神秘配方", None), ("水瓶", "水瓶"), ("CrèmeBrûlée", "CrèmeBrûlée")]:
        assert overlay_core._speakable_name(name) == expected, (name, overlay_core._speakable_name(name))
    discord_bridge._connected_clients = 1
    overlay_core.speech_options["context"] = ""
    _sync({str(i): f"Person{i:02d}" for i in range(20)})
    discord_bridge._handle_message({"event": "speaking_start", "userId": "19", "ts": time.time() * 1000})
    names = overlay_core.speech_context().split(", ")
    assert len(names) == overlay_core.MAX_CONTEXT_NAMES and names[0] == "Person19", names
    discord_bridge._connected_clients = 0
    print("OK: status-like names left out; capped, most recent speaker first")

    # The speech model echoing its context back instead of transcribing.
    context = "Caleb, CrèmeBrûlée, JuGGerNauT, kaektus, marina, 水瓶"
    assert translate_vc._echoes_context("Caleb, CrèmeBrûlée, JuGGerNauT, kaektus, marina, 水瓶", context)
    assert translate_vc._echoes_context("marina kaektus JuGGerNauT", context)
    assert not translate_vc._echoes_context("Caleb!", context)
    assert not translate_vc._echoes_context("Hey Caleb, did marina go to the shops already?", context)
    assert not translate_vc._echoes_context("水瓶，你好", context)
    print("OK: a transcription that's just the context read back is caught; real mentions aren't")
    print("ALL OK")


if __name__ == "__main__":
    main()
