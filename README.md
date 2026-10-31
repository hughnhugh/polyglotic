# vc-translator

Live translator for Discord (or any Windows system audio). Captures your PC's
audio output via WASAPI loopback plus your microphone, detects speech with
Silero VAD (or Discord's own speaking events), transcribes it with
Qwen3-ASR or Whisper, and translates it into the languages of your choosing with Hy-MT2-7B, a translation-tuned LLM
served by a local llama.cpp server — all running locally on your GPU, no
cloud APIs.

Captions appear as subtitles over your game or video - the translation,
the original underneath, and who said it - one overlay per language you
translate into (e.g. English and Chinese at once). Audio capture and
transcription happen once per spoken segment and are shared across every
language; only the (cheap) translation step runs once per language. A main
window holds the settings and a searchable transcript of the session.

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
.\venv\Scripts\python.exe -m pip install faster-whisper PyAudioWPatch numpy opencc-python-reimplemented websockets PySide6
.\venv\Scripts\python.exe -m pip install nvidia-ml-py
```

Then the translator: unzip llama.cpp's
`llama-b11796-bin-win-cuda-13.4-x64.zip` and
`cudart-llama-bin-win-cuda-13.4-x64.zip` from its
[releases page](https://github.com/ggml-org/llama.cpp/releases/tag/b11796)
into `bin\`, and download
[`Hy-MT2-7B-Q4_K_M.gguf`](https://huggingface.co/tencent/Hy-MT2-7B-GGUF) into
`models\`.

`nvidia-ml-py` (the last line) is optional (Home's GPU memory readout
only) - queries the driver directly via NVML so it reflects everything using
VRAM, every model included. Without it, the readout shows as unavailable.

## Run

```powershell
.\venv\Scripts\python.exe translate_vc.py
```

It comes back the way you last left it - target languages, speech model,
caption look and the rest are remembered in `settings.json`. Captions
appear as subtitles over whatever you're doing; the **main window** is for
changing things and reading back. Closing it keeps the app running in the
**system tray** - quit from the tray icon or the main window's sidebar.

### The subtitle overlays

One per language you translate into, centred near the bottom of the screen
to start with. Captions run newest at the bottom, with one colour rule:
**white is what was said**, in its own language, and **yellow, under it,
is always a translation** - into the overlay's language. Above each is who
said it, in their own colour (names come from the Discord plugin; your own
speech shows as "You"), and the language it was spoken in. Lines already in
the overlay's language have no yellow line - unless you give the overlay a
**second language** (Languages page): then they're translated into that,
so one overlay serves readers of both. E.g. an English overlay with
Chinese as its second language shows Chinese speech with English under it,
and your English with Chinese under it.

Text has a dark outline, like film subtitles, so it stays readable over
bright scenes even with the background panel turned right down. A caption
still being spoken updates live, trailing off with `…`, and settles when
the speaker pauses. When nobody's talking for a few seconds the captions
fade away. The bigger you make an overlay, the more history it shows.

- **Locked** (the default once you're set up): clicks pass straight
  through to your game. **Ctrl+Alt+L** toggles it from anywhere, as does
  the tray menu and Home.
- **Unlocked**: hover an overlay for its controls - drag anywhere to move
  it, drag the corner to resize, the lock icon to lock it again, `x` to
  stop that language. Positions and sizes are remembered. The mouse wheel
  scrolls back through earlier captions (the last 150): scrolled back, the
  overlay holds still while new captions keep arriving, and a **Back to
  live** button shows how many are waiting - click it, or scroll to the
  bottom, to catch up. Locking always goes back to live.
- **Ctrl+Alt+H** hides/shows all overlays; **Ctrl+Alt+P** pauses/resumes
  captioning (nothing's transcribed while paused). These hotkeys work over
  games.

Overlays sit on top of windowed and borderless-fullscreen games. In
exclusive fullscreen, Windows puts the game above everything - switch the
game to borderless.

### The main window

- **Home** - whether it's listening, a Pause button, the everyday toggles
  (caption your own speech, lock and show overlays, show the original
  under translations), the languages being captioned, and the status of
  each part - speech model, translator, audio, Discord - with GPU memory.
- **Transcript** - everything captioned this session, every language side
  by side, searchable, with Copy all and a link to the logs folder.
- **Languages** - what to translate into (each gets an overlay), the
  spoken language, and names & slang:
  - **Spoken language** - Auto-detect (default), or English, Mandarin,
    Cantonese or Vietnamese. Choosing one skips the speech model's
    language detection, so short or mumbled speech can't be misread as
    another language (Thai, Malay...). Use it for calls entirely in one
    language: anything said in another is still written down, but labeled
    as the chosen one - so it isn't translated in that language's overlay.
  - **Names & slang** - comma-separated words the speech model should
    spell your way (people's names, game terms, "HSK", "bún bò"...).
    With "Add the names of people in the Discord call" on (the default)
    and the Discord plugin connected, the names of the people in your
    voice channel are added automatically, following people joining and
    leaving - the line underneath shows exactly what the model is given.
    Emoji and decorations are stripped from names, ones that read more
    like a status than a name (just symbols, over two words or 16
    characters, Chinese over 4 characters) are left out, and at most 12
    are used, most recent speakers first. Given a short or unclear clip,
    the speech model sometimes "transcribes" the list itself - a caption
    that's just names read back is dropped.
- **Captions** - how the overlays look and behave, with a live preview:
  text size, background opacity, speaker names, the white line above
  translations, how soon they fade, and live
  update timing (below). Also resets overlay positions.
- **Audio** - which microphone (Windows' default, or pick one - it
  switches straight away and is remembered), system audio (Windows'
  default output), whether your own speech is captioned (off saves GPU
  time), and the Discord plugin's connection, who's in the call, and its
  port.
- **Models** - the speech model (switching takes a few seconds, during
  which captions pause), and the translator - its compression and status
  (see [Translator quality](#translator-quality-optional)).

### Mixed languages and your own mic

A line mixing languages ("反正就是一个非常specific的这种culture concept")
is tagged with both ("Mandarin + English") and translated even into its
main language's overlay - the whole line rewritten in that language, so
the English bits come out in Chinese too ("反正这就是个非常具体的文化概念啦"),
translated in context rather than piece by piece.

Your mic only captions you. With headphones, Discord can leak faintly into
the mic, and a small noise of yours (a click, a tap, a breath) can open a
mic segment that then picks up whoever's talking. Before transcribing, the
app checks whether the mic's loudness follows what the PC was playing - a
leaked copy does, your voice doesn't (even talking over someone) - and
drops leaks. As a backup, a "You" caption repeating what someone else said
in the last few seconds is dropped, or taken back if theirs arrives after
it.

### How captions update

While someone's talking, their caption appears live and grows as they go -
every 1.5 seconds by default, what they've said is transcribed again and
the caption updates in place, extending and correcting itself as more
context arrives. When they pause, the whole stretch is transcribed once
more in one go and replaces it. If the same person carries on within 2
seconds, with nobody else captioned in between, their new speech is
joined onto that caption and the whole run re-transcribed and
re-translated as one. Speech running over 20 seconds without a pause is
captioned in 20-second pieces.

Captions' **"Live updates every"** box sets the update interval: pick a
suggestion or type any number of seconds from 0.5 to 20, or `Off` for no
live captions (each appears only once the speaker pauses). Shorter feels
more live but uses more GPU time; it can't build a backlog, though - when
the GPU's busy, waiting live updates are replaced by newer ones rather
than queued, and captions finishing a pause always go first. **"Translate
live updates"** (on by default) translates them as they come; off, they
show the original and translate on the pause. Long live captions grow in
6-second sections, so each update stays quick (transcription time grows
with the audio: ~0.35s for 1.5s of speech, ~1.1s for 6s, ~3.5s for 20s).

Each session is logged to `logs/Session_<date>_<time>.log`: every final
caption with its translations, the delay from the end of the speech to it
appearing, processing time alone (`proc`) and how many live updates it had
(`live x3`).

### Options

Anything left out comes from the last run.

- `--target LANG[,LANG...]` - target language codes, e.g. `en,vi` (first
  run: `en`). 38 target languages - the ones Hy-MT2 supports (see
  `lang_codes.py`), including `zh` (Simplified Chinese), `zh-hant`
  (Traditional Chinese) and `yue` (Cantonese) as separate targets -
  Cantonese written in Hong Kong Traditional characters with the
  colloquial Cantonese ones (嘅、咗、唔、佢、喺...), as is Cantonese speech
  itself. Speech in any language the speech model recognizes can be
  translated *from*; only the target is limited to this list.
- `--model SIZE` - speech model: `qwen3-asr-1.7b` / `qwen3-asr-0.6b` (see
  below) or a Whisper size (first run: `qwen3-asr-1.7b` if it's installed,
  else `large-v3`).
- `--no-mic` - don't capture your microphone.
- `--discord-bridge-port PORT` - the Discord plugin's port (default 8765).

## Translator quality (optional)

The translator, Hy-MT2-7B, comes compressed (quantized) to two sizes,
picked from the dropdown on the Models page - switching takes a few
seconds, during which translations pause:

| | Download / VRAM | |
|---|---|---|
| `Q4_K_M` (default) | ~4.6GB | the most compressed - always installed |
| `Q8_0` | ~8.0GB | the closest to uncompressed |

`Q8_0` is opt-in: `setup.bat -Translator Q8_0`. If it isn't downloaded the
dropdown says so, and the app falls back to `Q4_K_M` if it goes missing.
Don't expect much from it: `Q6_K`, nearly as close to uncompressed, was
compared with `Q4_K_M` on 640 real captions (`eval/compare_translators.py`)
and translated no better - about half the lines came out word for word the
same, and the rest split evenly.

## Qwen3-ASR speech model (optional)

An alternative to Whisper for transcription, picked on the main window's
Models page (`qwen3-asr-1.7b` or `qwen3-asr-0.6b`), and the default once
installed. Alibaba's
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
default `8765`, also changeable on the Audio page). When connected,
loopback-audio segment boundaries and labels come straight from Discord's
own speaking start/stop events. See `vencord-plugin/README.md` for
build/install steps. Home and the Audio page show whether it's connected
and who's in the call. Note changing the port in the app only affects the
Python side -
the Vencord plugin has its own separate `port` setting to match.

Without the plugin connected, `vc-translator` works exactly the same, just
with unlabeled captions for system audio (your own mic captions, "You", are
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
- Every session's captions are logged to `logs/Session_<date>_<time>.log`
  (gitignored), every target language together.

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
  models (~3GB total, one-time) - Home's Status panel shows "Loading..."
  per component. A slow/metered connection
  can make this take several minutes.
- **Captions show only the original text, no translation.** Check the
  Translator row on Home (or the Models page). "Not installed" means
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
- **Model switch / GPU memory reading not moving.** The Models page's
  status line is the source of truth for whether a model switch actually
  completed (`Ready (size)`); Home's GPU readout needs `nvidia-ml-py`
  installed (see Setup above).
  A failed switch to Qwen3-ASR reverts to the previous model - its server
  log is `logs/qwen-asr-server.log`.
