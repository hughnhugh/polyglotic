# models

setup.bat downloads the models here. Already have them - from another PC,
a USB drive, a download of your own? Copy them in, laid out like this, and
setup uses them instead of downloading (or run `setup.bat -ModelsFrom
<folder>` to have it copy them from a folder laid out the same way). The app
finds them by these names - nothing else to wire up.

## Default - an 8GB GPU (4.2GB of GPU memory)

| Put here | What it is | Get it from |
|---|---|---|
| `Qwen3-ASR-0.6B\` (the whole folder) | Speech recognition, ~2.0GB | [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) |
| `Hy-MT2-1.8B-Q8_0.gguf` | Translation, ~1.9GB | [tencent/Hy-MT2-1.8B-GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF) |
| `awesome-align-with-co\` (the whole folder) | Word links, 1.1GB - runs on the CPU | [aneuraz/awesome-align-with-co](https://huggingface.co/aneuraz/awesome-align-with-co) |

## Better on Cantonese, tight on 8GB (6.2GB) - `setup.bat -LargerSpeechModel`

| Put here | What it is |
|---|---|
| `Qwen3-ASR-1.7B\` (the whole folder) | Speech recognition, ~4.7GB |
| `Hy-MT2-1.8B-Q4_K_M.gguf` | Translation, ~1.1GB - more compressed, to make room |

With both speech models here, the app starts with the 1.7B and the Q4_K_M
translator; either can be switched on the Models page.

## Bigger GPUs (12GB+) - optional

| Put here | What it is |
|---|---|
| `Hy-MT2-7B-Q4_K_M.gguf` / `HY-MT2-7B-Q8_0.gguf` | The larger translator, 4.6 / 8.0GB - picked on the Models page |
| `Qwen3.5-4B-Q4_K_M.gguf` | The local assistant (side chat, Explain), ~3GB - `setup.bat -Assistant` |

The names and folders the app looks for are in `model_files.py`.
