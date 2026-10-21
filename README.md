# vc-translator

Live translator for Discord (or any Windows system audio). Captures your PC's
audio output via WASAPI loopback plus your microphone, detects speech with
Silero VAD, transcribes it with `faster-whisper`, and translates it to a
target language of your choosing with Hy-MT2-7B, a translation-tuned LLM
served by a local llama.cpp server — all running locally on your GPU, no
cloud APIs.

Shows a small always-on-top overlay with the original text and its
translation. Pass a comma-separated `--target` list to get multiple
simultaneous overlays (e.g. one for English, one for Chinese) from a single
process - audio capture and Whisper transcription happen once per spoken
segment and are shared across every target; only the (cheap) translation
step runs once per target.

## Requirements

- **Windows 10/11.** This uses WASAPI loopback audio capture and Win32-specific
  packages - it will not run on macOS/Linux.
- **An NVIDIA GPU with CUDA support, and a recent driver (580 or newer -
  CUDA 13).** Whisper and the translation model are heavy - this isn't
  optional, there's no CPU fallback mode. **12GB+ VRAM recommended**: the
  translator takes ~5GB and Whisper `large-v3` ~4GB, plus whatever else
  (games, browser, Discord) is using the GPU. Smaller Whisper models (see
  Options below) free up some of that.
- **Python 3.12.** Get it from [python.org/downloads](https://www.python.org/downloads/)
  - during install, check **"Add python.exe to PATH"**, and check
  **"py launcher"** (on by default) - both are just checkboxes on the first
  install screen.
- **~15GB free disk space** - dependencies (~4-5GB, mostly PyTorch+CUDA), the
  translator (~5GB, downloaded by setup) and the Whisper/VAD models downloaded
  on first run (~3GB - see Notes below).
- **Internet connection** for setup and for the one-time Whisper download on
  first launch. Not needed after that.

## Setup

**Option A - automated (recommended):** double-click **`setup.bat`** in this
folder. It finds your Python install, creates the project's virtual
environment (`venv\`), installs everything, and downloads the translator - a
prebuilt llama.cpp server into `bin\` and the Hy-MT2-7B model (4.6GB) into
`models\` - printing clear errors if something's missing (e.g. no Python
found, or no NVIDIA GPU detected). Safe to re-run any time - it reuses an
existing `venv\` and existing downloads rather than redoing them, so
re-running after a `git pull` just catches up on anything new, and
re-running after an interrupted model download resumes it.

If Windows won't run `setup.ps1` directly (only relevant if you run the
`.ps1` yourself instead of double-clicking `setup.bat`), that's PowerShell's
script-execution policy blocking it by default - `setup.bat` already works
around this for you by invoking it with the restriction bypassed for just
that one run, without changing any system-wide setting.

**Option B - manual**, if you'd rather see/control each step:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
.\venv\Scripts\python.exe -m pip install faster-whisper PyAudioWPatch numpy opencc-python-reimplemented websockets
.\venv\Scripts\python.exe -m pip install nvidia-ml-py
```

Then the translator: unzip llama.cpp's
`llama-b11796-bin-win-cuda-13.4-x64.zip` and
`cudart-llama-bin-win-cuda-13.4-x64.zip` from its
[releases page](https://github.com/ggml-org/llama.cpp/releases/tag/b11796)
into `bin\`, and download
[`Hy-MT2-7B-Q4_K_M.gguf`](https://huggingface.co/tencent/Hy-MT2-7B-GGUF) into
`models\`.

`nvidia-ml-py` (the last line) is optional (Settings' GPU memory readout
only) - queries the driver directly via NVML so it reflects everything using
VRAM, including Whisper (which runs through ctranslate2, invisible to
PyTorch's own memory stats). Without it, the readout falls back to a
PyTorch-only number that misses Whisper's usage entirely.

## Run

```powershell
.\venv\Scripts\python.exe translate_vc.py
.\venv\Scripts\python.exe translate_vc.py --target en
.\venv\Scripts\python.exe translate_vc.py --target zh
.\venv\Scripts\python.exe translate_vc.py --target en,vi
```

This is a persistent app, not a one-shot script: a **Settings window** is
always available - a normal, taskbar-visible window (alt-tabbable like any
other app) where you pick target languages (by name, e.g. "Vietnamese", not
code), speech model, conversation language, names & slang, overlay
position, mic capture, and the Discord bridge port, all editable live while
it keeps running - checking/unchecking a language opens/closes its overlay
immediately, no restart needed. All of it is remembered in `settings.json`
for the next launch, so a plain `translate_vc.py` comes back the way you
left it; command-line options override it for that run.

- **Conversation language** - "Auto-detect" (default), or English,
  Mandarin, Cantonese or Vietnamese only. Forcing one skips the speech
  model's language detection, so short or mumbled speech can't be misread
  as some other language (Thai, Malay...). Use it for calls that are
  entirely in one language: anything said in another language is still
  written down, but labeled as the forced one - so it isn't translated in
  that language's own window.
- **Names & slang** - comma-separated words the speech model should spell
  the way you wrote them (people's names, game terms, "HSK", "bún bò"...).
  Applied from the next caption.

The Settings window always starts visible right now (whether or not you
pass `--target`) - it's meant to stay hidden for scripted `--target`
launches with the tray as the way back in, but the tray is currently
disabled (see below), so hiding it would strand it unreachable.

**Window behavior:** closing Settings (✕) quits the whole app - streams,
models, the Discord bridge, everything. Minimizing behaves like a normal
window for now - a system tray/minimize-to-tray integration was tried
(`tray.py`) but pystray's native Win32 message-loop thread turned out to
segfault against Tkinter's own message loop on Windows, so it's disabled
until a safer approach is found.

An overlay window appears per active target (bottom for `en`, top for other
targets by default; windows sharing an anchor stack instead of overlapping).
Drag one to reposition, click its ✕ or press Escape to close just that
window (unchecks it in Settings too), drag the bottom-right corner to
resize, `_` to minimize it to a small tab, `T` to toggle that window's
translation off/on (dims when off) - handy if you just want the native
transcription without a second line underneath - and `Me` to toggle your
own mic captioning off/on (shared across every window, since it's one mic
feed) - handy for muting yourself out of the captions without disabling
your mic in Discord itself. Have Discord (or whatever app) playing audio
through your default output device — your microphone is captured too,
tagged `[You]`, unless toggled off.

**How captions update:** while someone keeps talking, a live caption
(marked `…`) appears every 6 seconds; when they pause, the whole stretch
(up to 20s) is transcribed again in one go and replaces those live pieces
in place. If the same person carries on within 2 seconds, with nobody else
captioned in between, their new speech is joined onto the previous caption
and the whole run re-transcribed and re-translated as one. The time in each
caption's `[lang, 1.2s]` tag is the delay from the end of the speech to the
caption appearing - waiting and processing included (the log files also
record processing time alone as `proc`).

### Options

- `--target LANG[,LANG...]` - one or more comma-separated target language
  codes to start captioning immediately (default: the last run's, or `en`). 38 target languages - the ones Hy-MT2 supports
  (see `lang_codes.py`), including `zh` (Simplified Chinese), `zh-hant`
  (Traditional Chinese) and `yue` (Cantonese) as separate targets. Speech in
  any language Whisper recognizes can be translated *from*; only the target
  is limited to this list. More targets can always be added live from
  Settings too.
- `--model SIZE` - initial speech model: a Whisper size, or
  `qwen3-asr-1.7b` / `qwen3-asr-0.6b` (see below) (default: the last run's;
  first run `qwen3-asr-1.7b` if it's installed, else `large-v3`) -
  changeable live from Settings afterward (briefly pauses captioning while
  it swaps).
- `--position top|bottom` - initial overlay screen anchor for every window
  (default: `bottom` for `en`, `top` otherwise, chosen per target) -
  changeable live from Settings afterward.
- `--no-mic` - don't capture your microphone at startup - changeable live
  from Settings afterward.

## Qwen3-ASR speech model (optional)

An alternative to Whisper for transcription, picked from Settings' "Speech
model" dropdown (`qwen3-asr-1.7b` or `qwen3-asr-0.6b`). Alibaba's
Qwen3-ASR is much stronger than Whisper on Cantonese and other Chinese
dialects (by Qwen's own benchmarks - roughly half Whisper's error rate on
Cantonese, far less on conversational Cantonese), and reports Cantonese as
its own language instead of lumping it in with Mandarin. It also stays
silent on silence and background noise, where Whisper tends to invent
text that the app then has to filter out.

Setup is opt-in, since it's ~9GB more on disk - run `setup.bat -QwenASR`.
It needs its own Python environment (`venv-qwen-asr\`): the `qwen-asr`
package pins library versions that conflict with faster-whisper's. The
app runs it as a separate background process, started when you pick it
and stopped when you switch back.

Trade-offs, measured on this setup:

| | VRAM | Time per ~4s segment |
|---|---|---|
| Whisper `large-v3` | ~4.0GB | ~0.4s |
| `qwen3-asr-0.6b` | ~2.0GB | ~0.5s |
| `qwen3-asr-1.7b` | ~4.9GB | ~0.5s |

Whisper's confidence scores, which the app uses to catch made-up text, don't
exist for Qwen3-ASR - it relies on the VAD gate and Qwen's own "no speech"
signal instead.

## Speaker identification (optional, Discord only, advanced setup)

This part is meaningfully harder to set up than everything above - it needs
a full Vencord *source* build (there's no marketplace install, since this
plugin isn't an official Vencord one), which means Git, Node.js, and `pnpm`,
not just Python. `vc-translator` is fully usable without it - you only need
this if you specifically want real Discord usernames attached to captions
instead of unlabeled ones. See `vencord-plugin/README.md` for the full
build/install steps if you want it.

Speaker labels come only from ground truth, never a guess: a companion
Vencord plugin (`vencord-plugin/discordSpeakingBridge.ts`) reads Discord's
own "who is currently speaking" state (the same signal behind the green
speaking ring in the UI) and the real username, and pushes that to a small
local WebSocket server `translate_vc.py` runs (`--discord-bridge-port`,
default `8765`, also changeable live from Settings). When connected,
loopback-audio segment boundaries and labels come straight from Discord's
own speaking start/stop events. See `vencord-plugin/README.md` for
build/install steps. The overlay shows a one-line status whenever the
bridge connects or drops, so it's always clear which segmentation path is
active. Note changing the port from Settings only affects the Python side -
the Vencord plugin has its own separate `port` setting to match.

Without the plugin connected, `vc-translator` works exactly the same, just
with unlabeled captions for system audio (your own `[You]` mic captions are
always labeled). Note the bridge can't isolate simultaneous speakers' audio
from each other (that would need a bot with real per-user voice-receive) -
it only improves *identity and segment timing* on top of the existing
mixed loopback audio. Some rules keep that mixing from producing wrong
captions:

- When two people are flagged as speaking over the same audio (say, one's
  mic picking up the other), it's captioned once, not twice.
- Someone flagged as speaking for over 20 seconds straight - usually an
  open mic held on by background noise - can't cause anyone else's speech
  to be skipped that way, and their live captions are skipped while the
  audio is mostly other people talking.
- A reply that starts in someone's last second or so is cut off the end of
  their caption, and a joined caption never spans a gap someone else spoke
  in.

## Notes

- First run downloads the Silero VAD model and the Whisper model (~3GB for
  `large-v3`). The translation model is downloaded by setup instead (see
  Setup above).
- Whisper transcribes the native speech, then Hy-MT2-7B translates that text
  into each target language - English included. It runs in a separate
  `llama-server` process the app starts and stops itself (also killed
  automatically if the app crashes, so it can't leave VRAM tied up); its
  log goes to `logs/llama-server.log`. It loads in the background at
  startup - captions show the native text alone until it's ready.
- The translator is told it's working on casual, speech-recognized voice
  chat, so it keeps a spoken register (slang, fillers, reactions) rather
  than formalizing it. Hy-MT2 replaced NLLB-200 after a side-by-side on real
  captions - `eval/compare_translators.py` reruns that comparison.
- Chinese speech is normalized to Simplified for display; pick `--target
  zh-hant` if you want Traditional output instead.
- Every session's captions are logged to `logs/` (gitignored) with
  timestamps, one file per overlay window (target) per run.

## Troubleshooting

- **"No Python install found" from `setup.bat`/`setup.ps1`.** Install Python
  3.12 from [python.org/downloads](https://www.python.org/downloads/),
  checking "Add python.exe to PATH" during install, then re-run.
- **"'nvidia-smi' not found" warning during setup.** Means either there's no
  NVIDIA GPU, or its driver isn't installed - install the latest driver from
  [nvidia.com/drivers](https://www.nvidia.com/download/index.aspx). The app
  will not run without a working NVIDIA GPU + driver.
- **PowerShell won't run `setup.ps1` directly** ("running scripts is disabled
  on this system"). Use `setup.bat` instead - it invokes the `.ps1` with the
  restriction bypassed for just that one run. (Or manually:
  `powershell -ExecutionPolicy Bypass -File setup.ps1`.)
- **First launch seems stuck / slow.** It's downloading the Whisper and VAD
  models (~3GB total, one-time) - check the Settings window's status
  panel, which shows "Loading..." per component. A slow/metered connection
  can make this take several minutes.
- **Captions show only the original text, no translation.** Check the
  Settings window's "Translation (Hy-MT2)" row. "Not installed" means
  `bin\` or `models\` is missing - re-run `setup.bat`. "Failed" means the
  server didn't start - see `logs/llama-server.log`; the usual causes are an
  NVIDIA driver too old for CUDA 13, or not enough free VRAM (close other
  GPU-heavy apps, or pick a smaller Whisper model).
- **Symlink warning from `huggingface_hub` on first run.** Harmless - Windows
  needs Developer Mode or admin rights for symlinks; without them, model
  files are just copied instead, using a bit more disk space.
- **No captions ever appear for other people talking.** Check the overlay's
  status line for `[Capturing system audio: ...]` - it should name your
  actual output device (headphones/speakers). If Discord's output device
  differs from your Windows default output device, loopback capture won't
  hear it - either match them, or see PyAudioWPatch's docs for capturing a
  specific device.
- **Model reload / GPU memory reading not moving.** The Settings window's
  "Speech model" status row is the source of truth for whether a model
  switch actually completed (`Ready (size)`); the GPU row needs
  `nvidia-ml-py` installed to see Whisper's memory at all (see Setup above).
  A failed switch to Qwen3-ASR reverts to the previous model - its server
  log is `logs/qwen-asr-server.log`.
