"""Verifies the speech model's context words: Settings' "Names & slang" plus
the Discord channel members' names (overlay_core.speech_context), following
people joining, leaving and speaking - no GPU/Discord needed."""

import time

import discord_bridge
import overlay_core


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
    print("ALL OK")


if __name__ == "__main__":
    main()
