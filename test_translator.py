import sys

import translator

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Smoke test for the translator end to end: starts llama-server with
# Hy-MT2 (needs bin/ and models/ from setup.ps1, and the GPU), then
# translates a few casual lines in the app's common directions.
CASES = [
    ("Bro, you're my friend now.", "en", "zh"),
    ("It'll take time, it'll take time.", "en", "vi"),
    ("I love this car.", "en", "yue"),
    ("太酷了", "zh", "en"),
    ("Nhưng mà chú lại nói là chú không biết đúng không?", "vi", "en"),
]

print("Starting translator...")
translator.start()
if not translator._ready.wait(translator.LOAD_TIMEOUT_SEC):
    raise SystemExit("Translator failed to load - see logs/llama-server.log")

for text, src, target in CASES:
    print(f"{src}->{target}: {text}")
    print(f"    {translator.translate(text, src, target)}")

translator.stop()
