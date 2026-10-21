"""
Side-by-side comparison of candidate translation models on real captions
pulled from logs/, focused on casual voice-chat speech and on whether
passing the previous few captions as context helps.

Each candidate runs through a local llama-server (prebuilt llama.cpp in
eval/bin/), one model at a time, against the same sampled lines. The
current NLLB-600M runs as the baseline, and whatever the app actually
showed at the time (from the logs) is included for reference.

Prompt variants per LLM:
    plain       - the model's standard translate prompt, line on its own
    casual      - casual-speech instructions, line on its own
    casual+ctx  - casual-speech instructions plus the previous captions

Usage:
    .\\venv\\Scripts\\python.exe eval\\compare_translators.py
    .\\venv\\Scripts\\python.exe eval\\compare_translators.py --per-lang 20 --models hymt2-7b

Writes eval/results/<timestamp>/results.jsonl and report.html. Both contain
real conversation text from logs/ - they're gitignored, keep them local.
"""

import argparse
import glob
import html
import json
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta

import requests
from opencc import OpenCC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from translate_vc import _is_degenerate  # noqa: E402
from lang_codes import WHISPER_TO_FLORES  # noqa: E402

EVAL_DIR = os.path.join(ROOT, "eval")
LLAMA_SERVER = os.path.join(EVAL_DIR, "bin", "llama-server.exe")
PORT = 8091

MODELS = {
    "hymt2-7b": dict(repo="tencent/Hy-MT2-7B-GGUF", file="Hy-MT2-7B-Q4_K_M.gguf", family="hymt"),
    "hymt2-1.8b": dict(repo="tencent/Hy-MT2-1.8B-GGUF", file="Hy-MT2-1.8B-Q8_0.gguf", family="hymt"),
    "translategemma-4b": dict(repo="bullerwins/translategemma-4b-it-GGUF", file="translategemma-4b-it-Q8_0.gguf",
                              family="translategemma"),
    "gemma4-e4b": dict(repo="google/gemma-4-E4B-it-qat-q4_0-gguf", file="gemma-4-E4B_q4_0-it.gguf", family="gemma"),
}
VARIANTS = ["plain", "casual", "casual+ctx"]

# Which targets each source language gets translated into - the app's
# common pairs (Chinese/Cantonese, Vietnamese, English).
DIRECTIONS = {"en": ["zh", "vi", "yue"], "zh": ["en", "vi"], "vi": ["en", "zh"]}

LANG_EN = {"en": "English", "zh": "Simplified Chinese", "vi": "Vietnamese", "yue": "Cantonese"}
LANG_ZH = {"en": "英语", "zh": "简体中文", "vi": "越南语", "yue": "粤语"}

# Log filename target -> language code, for picking up what the app showed.
LOG_TARGETS = {"EN": "en", "English": "en", "ZH": "zh", "Chinese__Simplified": "zh",
               "VI": "vi", "Vietnamese": "vi", "Cantonese": "yue"}

CONTEXT_LINES = 4
CONTEXT_WINDOW_SEC = 90
SESSION_GAP_SEC = 300

_t2s = OpenCC("t2s")
_CJK_RE = re.compile(r"[一-鿿㐀-䶿]")
_LATIN_RE = re.compile(r"[A-Za-zÀ-ỹ]")
_CANTONESE_MARKERS = set("係嘅唔咗哋喺冇啲嚟咁噉乜嘢佢睇揾搵畀俾啱嗰架隻幾")


# ---------------------------------------------------------------- log parsing

_LINE_RE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] (.*?)\[([a-z]{2,3}), [\d.]+s\] (.*)$")
_FILE_RE = re.compile(r"Live_Captions__(.+)_\d{8}_\d{6}\.log$")


def load_segments():
    """Every caption across logs/, deduplicated (each segment is logged once
    per open overlay window), in time order, with the translations the app
    showed collected per target."""
    raw = []
    for path in glob.glob(os.path.join(ROOT, "logs", "*.log")):
        m = _FILE_RE.search(os.path.basename(path))
        target = LOG_TARGETS.get(m.group(1)) if m else None
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        for i, line in enumerate(lines):
            lm = _LINE_RE.match(line)
            if not lm:
                continue
            ts, speaker, lang, native = lm.groups()
            translated = None
            if i + 1 < len(lines) and lines[i + 1].startswith("    -> "):
                translated = lines[i + 1][len("    -> "):].strip()
            raw.append((datetime.strptime(ts, "%Y-%m-%d %H:%M:%S"), speaker.strip(), lang, native.strip(),
                        target, translated))

    raw.sort(key=lambda r: r[0])
    segments = []
    recent = {}  # (speaker, native) -> segment, for merging duplicates logged by sibling windows
    for ts, speaker, lang, native, target, translated in raw:
        key = (speaker, native)
        seg = recent.get(key)
        if seg is None or ts - seg["ts"] > timedelta(seconds=5):
            seg = {"ts": ts, "speaker": speaker, "lang": lang, "native": native, "app": {}}
            segments.append(seg)
            recent[key] = seg
        if target and translated and target != lang:
            seg["app"].setdefault(target, translated)

    for i, seg in enumerate(segments):
        seg["idx"] = i
    return segments


def context_for(segments, seg):
    ctx = []
    j = seg["idx"] - 1
    while j >= 0 and len(ctx) < CONTEXT_LINES:
        prev = segments[j]
        if (seg["ts"] - prev["ts"]).total_seconds() > CONTEXT_WINDOW_SEC:
            break
        ctx.append(prev)
        j -= 1
    return list(reversed(ctx))


def _is_short(seg):
    if seg["lang"] == "zh":
        return len(seg["native"]) <= 10
    return len(seg["native"].split()) <= 6


def sample_items(segments, per_lang, seed):
    """Half short/casual lines (fillers, fragments, reactions - where NLLB
    struggles most), half everything else, per source language."""
    rng = random.Random(seed)
    items = []
    for src, targets in DIRECTIONS.items():
        pool = [s for s in segments if s["lang"] == src and 2 <= len(s["native"]) <= 300]
        short = [s for s in pool if _is_short(s)]
        rest = [s for s in pool if not _is_short(s)]
        picked = rng.sample(short, min(len(short), per_lang // 2))
        picked += rng.sample(rest, min(len(rest), per_lang - len(picked)))
        for seg in picked:
            ctx = context_for(segments, seg)
            for tgt in targets:
                items.append({
                    "id": f"{seg['idx']}-{tgt}",
                    "src": src, "tgt": tgt,
                    "speaker": seg["speaker"], "text": seg["native"],
                    "short": _is_short(seg),
                    "ts": seg["ts"].isoformat(sep=" "),
                    "context": [{"speaker": c["speaker"], "lang": c["lang"], "text": c["native"]} for c in ctx],
                    "app": seg["app"].get(tgt),
                })
    return items


# ------------------------------------------------------------------- prompts

def _ctx_block(item):
    return "\n".join(f"{c['speaker'] or '(unknown)'}: {c['text']}" for c in item["context"])


def _zh_prompt(item):
    # Tencent recommends Chinese prompts whenever Chinese is on either side.
    return "zh" in (item["src"], item["tgt"]) or "yue" in (item["src"], item["tgt"])


def hymt_messages(item, variant):
    tgt, text = item["tgt"], item["text"]
    if variant == "plain":
        if _zh_prompt(item):
            p = f"将以下文本翻译为{LANG_ZH[tgt]}，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
        else:
            p = (f"Translate the following text into {LANG_EN[tgt]}. Note that you should only output the "
                 f"translated result without any additional explanation:\n\n{text}")
        return [{"role": "user", "content": p}]

    use_ctx = variant == "casual+ctx" and item["context"]
    if _zh_prompt(item):
        parts = []
        if use_ctx:
            parts.append(f"【背景信息】\n以下是这段语音聊天中前面的几句话：\n{_ctx_block(item)}\n")
        parts.append(f"【待翻译文本】\n{text}\n")
        tasks = ["这是一段轻松随意的实时语音聊天中的一句话，由语音识别转写而来，可能有识别错误或不完整的句子。",
                 "保持口语化、随意的语气，俚语、语气词和感叹词按意思自然地翻译，不要变得书面或正式。"]
        if use_ctx:
            tasks.append("背景信息只用于理解意思，不要翻译背景信息。")
        tasks.append("只输出【待翻译文本】的译文，不要额外解释。")
        tasks.append(f"将【待翻译文本】翻译为{LANG_ZH[tgt]}。")
        parts.append("【翻译任务】\n" + "\n".join(f"{i}、{t}" for i, t in enumerate(tasks, 1)))
    else:
        parts = []
        if use_ctx:
            parts.append(f"[Background Information]\nEarlier lines of this voice chat:\n{_ctx_block(item)}\n")
        parts.append(f"[Source Text]\n{text}\n")
        tasks = ["The text is one line from a casual live voice chat, transcribed by speech recognition. It may "
                 "contain recognition errors or be an incomplete sentence.",
                 "Keep the casual, spoken tone. Translate slang, filler words and interjections naturally by "
                 "meaning; do not make it more formal."]
        if use_ctx:
            tasks.append("Use the background information only to understand the meaning; do not translate it.")
        tasks.append("Only output the translation of the [Source Text], without any additional explanation.")
        tasks.append(f"Translate the [Source Text] into {LANG_EN[tgt]}.")
        parts.append("[Translation Tasks]\n" + "\n".join(f"{i}. {t}" for i, t in enumerate(tasks, 1)))
    return [{"role": "user", "content": "\n".join(parts)}]


_CASUAL_EN = ("The text is one line from a casual live voice chat, transcribed by speech recognition, so it may "
              "contain recognition errors or be an incomplete sentence. Keep the casual, spoken tone: translate "
              "slang, filler words and interjections naturally by meaning, and do not make it more formal.")


def gemma_messages(item, variant):
    src, tgt, text = item["src"], item["tgt"], item["text"]
    if variant == "plain":
        return [{"role": "user", "content": f"Translate the following {LANG_EN[src]} text into {LANG_EN[tgt]}. "
                                           f"Output only the translation.\n\n{text}"}]
    system = (f"You translate live voice-chat captions from {LANG_EN[src]} into {LANG_EN[tgt]}. {_CASUAL_EN} "
              f"Output only the translation of the line you are given - no quotes, notes or explanations.")
    if tgt == "yue":
        system += " Write natural spoken Cantonese in Traditional characters, not Mandarin."
    user = text
    if variant == "casual+ctx" and item["context"]:
        system += (" You are also given the previous lines of the conversation for context; use them only to "
                   "understand the meaning, and do not translate them.")
        user = f"Previous lines:\n{_ctx_block(item)}\n\nLine to translate:\n{text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def translategemma_prompt(item, variant):
    """TranslateGemma's chat template only takes a bare segment plus language
    codes, so it can't go through the chat endpoint with extra
    instructions. This reproduces its template text (see
    chat_template.jinja on the model card) as a raw prompt; the casual/ctx
    variants add to it, which the model card notes is unofficial."""
    src, tgt, text = item["src"], item["tgt"], item["text"]
    s, t = LANG_EN[src], LANG_EN[tgt]
    head = (f"You are a professional {s} ({src}) to {t} ({tgt}) translator. Your goal is to accurately convey "
            f"the meaning and nuances of the original {s} text while adhering to {t} grammar, vocabulary, and "
            f"cultural sensitivities.\n")
    if variant != "plain":
        head += _CASUAL_EN + "\n"
    if variant == "casual+ctx" and item["context"]:
        head += (f"For context only (do not translate these), the previous lines of the conversation were:\n"
                 f"{_ctx_block(item)}\n")
    body = (f"Produce only the {t} translation, without any additional explanations or commentary. Please "
            f"translate the following {s} text into {t}:\n\n\n{text}")
    return f"<start_of_turn>user\n{head}{body}<end_of_turn>\n<start_of_turn>model\n"


# ------------------------------------------------------------------- servers

class VramMonitor:
    """Polls nvidia-smi for whole-GPU memory use; peak minus the reading
    before the model loaded is that model's footprint."""

    def __init__(self):
        self.baseline = self.read()
        self.peak = self.baseline
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    @staticmethod
    def read():
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        return int(out.strip().splitlines()[0])

    def _run(self):
        while not self._stop.wait(0.5):
            self.peak = max(self.peak, self.read())

    def stop(self):
        self._stop.set()
        self._t.join()
        return self.peak - self.baseline


class LlamaServer:
    def __init__(self, model_path, log_path):
        self.log = open(log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [LLAMA_SERVER, "-m", model_path, "--port", str(PORT), "-ngl", "999", "-c", "4096", "-np", "1",
             "-fa", "on", "--jinja", "--no-webui"],
            stdout=self.log, stderr=subprocess.STDOUT)
        self.base = f"http://127.0.0.1:{PORT}"
        deadline = time.time() + 300
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early, see {log_path}")
            try:
                if requests.get(f"{self.base}/health", timeout=2).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        raise RuntimeError(f"llama-server didn't become ready, see {log_path}")

    def chat(self, messages, **sampling):
        r = requests.post(f"{self.base}/v1/chat/completions", json={
            "messages": messages, "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False}, **sampling}, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"].get("content") or ""

    def complete(self, prompt, **sampling):
        r = requests.post(f"{self.base}/completion", json={
            "prompt": prompt, "n_predict": 256, "stop": ["<end_of_turn>"], **sampling}, timeout=120)
        r.raise_for_status()
        return r.json()["content"]

    def chat_template(self):
        try:
            return requests.get(f"{self.base}/props", timeout=5).json().get("chat_template", "")
        except requests.RequestException:
            return ""

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


def run_llm(name, items, variants, out_dir, write):
    from huggingface_hub import hf_hub_download

    spec = MODELS[name]
    # A file already fetched into eval/models/ (e.g. with a resumable curl,
    # when hf-xet keeps dropping on a slow connection) takes precedence.
    local = os.path.join(EVAL_DIR, "models", spec["file"])
    path = local if os.path.exists(local) else hf_hub_download(spec["repo"], spec["file"])
    print(f"\n== {name} ({spec['file']})", flush=True)
    vram = VramMonitor()
    server = LlamaServer(path, os.path.join(out_dir, f"server_{name}.log"))
    family = spec["family"]
    try:
        with open(os.path.join(out_dir, f"chat_template_{name}.txt"), "w", encoding="utf-8") as f:
            f.write(server.chat_template())

        # Greedy decoding throughout so runs are repeatable and comparable;
        # Tencent's light repetition penalty kept for Hy-MT.
        sampling = {"temperature": 0}
        if family == "hymt":
            sampling["repeat_penalty"] = 1.05

        def call(item, variant):
            if family == "translategemma":
                return server.complete(translategemma_prompt(item, variant), **sampling)
            msgs = hymt_messages(item, variant) if family == "hymt" else gemma_messages(item, variant)
            return server.chat(msgs, **sampling)

        call(items[0], "plain")  # warm-up, untimed

        for variant in variants:
            for n, item in enumerate(items, 1):
                t0 = time.perf_counter()
                try:
                    out, err = call(item, variant).strip(), None
                except Exception as e:  # keep going - one bad request shouldn't sink the run
                    out, err = "", str(e)
                write({"item": item["id"], "system": name, "variant": variant, "output": out,
                       "ms": round((time.perf_counter() - t0) * 1000), "error": err})
                if n % 50 == 0:
                    print(f"  {variant}: {n}/{len(items)}", flush=True)
    finally:
        server.close()
        delta = vram.stop()
    print(f"  peak VRAM +{delta} MB", flush=True)
    return delta


def run_nllb(items, write):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print("\n== nllb-600m (current app)", flush=True)
    vram = VramMonitor()
    name = "facebook/nllb-200-distilled-600M"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name).to("cuda").half()
    for item in items:
        t0 = time.perf_counter()
        tok.src_lang = WHISPER_TO_FLORES[item["src"]]
        inputs = tok(item["text"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, forced_bos_token_id=tok.convert_tokens_to_ids(WHISPER_TO_FLORES[item["tgt"]]),
                                 max_new_tokens=200)
        text = tok.batch_decode(out, skip_special_tokens=True)[0].strip()
        write({"item": item["id"], "system": "nllb-600m", "variant": "plain", "output": text,
               "ms": round((time.perf_counter() - t0) * 1000), "error": None})
    del model
    torch.cuda.empty_cache()
    delta = vram.stop()
    print(f"  peak VRAM +{delta} MB", flush=True)
    return delta


# ------------------------------------------------------------------- scoring

def _units(text):
    """Rough length that's comparable across scripts: CJK characters count
    as ~0.7 of a Latin-script word."""
    return len(_CJK_RE.findall(text)) * 0.7 + len(_CJK_RE.sub(" ", text).split())


def flags_for(item, out, nocontext_out=None):
    f = []
    if not out:
        return ["empty"]
    if _is_degenerate(out):
        f.append("loop")
    cjk = len(_CJK_RE.findall(out))
    latin = len(_LATIN_RE.findall(out))
    if item["tgt"] in ("zh", "yue") and cjk < latin:
        f.append("wrong-lang")
    if item["tgt"] in ("en", "vi") and cjk > 0.3 * max(1, latin):
        f.append("wrong-lang")
    if out.strip().lower() == item["text"].strip().lower():
        f.append("untranslated")
    if item["tgt"] == "zh" and _t2s.convert(out) != out:
        f.append("traditional")
    if item["tgt"] == "yue" and cjk >= 5 and not (_CANTONESE_MARKERS & set(out)):
        f.append("mandarin?")
    if "\n" in out and "\n" not in item["text"]:
        f.append("multiline")
    if _units(out) > 3 * _units(item["text"]) + 8:
        f.append("too-long")
    if nocontext_out is not None and len(out) > 2.5 * len(nocontext_out) + 10:
        f.append("ctx-leak?")
    return f


# -------------------------------------------------------------------- report

REPORT_CSS = """
:root{--bg:#fbfbfa;--panel:#fff;--ink:#1d1d1b;--mute:#6b6b66;--line:#e4e3df;--bad:#b42318;--badbg:#fdecea;
--warn:#8a5a00;--warnbg:#fff4dc;--accent:#2f5bd3}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--panel:#1f1f1d;--ink:#ecebe7;--mute:#9b9a94;--line:#33322f;
--bad:#ff8a80;--badbg:#3a1d1a;--warn:#f2c46d;--warnbg:#33290f;--accent:#8fb0ff}}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"Segoe UI","Microsoft YaHei",sans-serif}
main{max-width:1200px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px}
.sub{color:var(--mute);margin:0 0 16px}
table{border-collapse:collapse;width:100%;background:var(--panel);font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{font-weight:600;color:var(--mute);font-size:12px;text-transform:uppercase;letter-spacing:.03em}
.wrap{overflow-x:auto}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:12px 0;position:sticky;top:0;
background:var(--bg);padding:8px 0;z-index:1;border-bottom:1px solid var(--line)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;margin:12px 0;padding:12px}
.ctx{color:var(--mute);font-size:12.5px;margin-bottom:6px;white-space:pre-wrap}
.src{font-size:15px;margin-bottom:8px}.src b{color:var(--accent)}
.meta{color:var(--mute);font-size:12px}
.row td:first-child{white-space:nowrap;color:var(--mute);font-size:12.5px;width:210px}
.flag{display:inline-block;font-size:11px;padding:0 6px;border-radius:9px;margin-left:6px;background:var(--warnbg);
color:var(--warn)}
.flag.bad{background:var(--badbg);color:var(--bad)}
.ms{color:var(--mute);font-size:12px;white-space:nowrap;width:60px;text-align:right}
.hidden{display:none}
"""

REPORT_JS = """
const $=s=>document.querySelectorAll(s);
function apply(){
  const dir=document.getElementById('dir').value, flagged=document.getElementById('flagged').checked,
        short=document.getElementById('short').checked;
  const shown=new Set([...$('.sys:checked')].map(c=>c.value));
  $('.row').forEach(r=>r.classList.toggle('hidden',!shown.has(r.dataset.sys)));
  $('.card').forEach(c=>{
    let ok=(dir==='all'||c.dataset.dir===dir)&&(!short||c.dataset.short==='1');
    if(ok&&flagged) ok=[...c.querySelectorAll('.row:not(.hidden)')].some(r=>r.dataset.flagged==='1');
    c.classList.toggle('hidden',!ok);
  });
}
document.addEventListener('change',apply);
"""

BAD_FLAGS = {"empty", "loop", "wrong-lang", "untranslated", "ctx-leak?", "multiline"}


def write_report(out_dir, items, results, vram, meta):
    by_item = {}
    for r in results:
        by_item.setdefault(r["item"], {})[(r["system"], r["variant"])] = r
    systems = []
    for r in results:
        key = (r["system"], r["variant"])
        if key not in systems:
            systems.append(key)

    # Summary
    rows = []
    for sysname, variant in systems:
        rs = [by_item[i["id"]][(sysname, variant)] for i in items if (sysname, variant) in by_item.get(i["id"], {})]
        ms = sorted(r["ms"] for r in rs)
        flagged = sum(1 for r in rs if set(r["flags"]) & BAD_FLAGS)
        counts = {}
        for r in rs:
            for f in r["flags"]:
                counts[f] = counts.get(f, 0) + 1
        p95 = ms[min(len(ms) - 1, int(len(ms) * 0.95))] if ms else 0
        rows.append(f"<tr><td>{html.escape(sysname)}</td><td>{variant}</td><td>{len(rs)}</td>"
                    f"<td>{statistics.median(ms) if ms else 0:.0f}</td><td>{p95}</td>"
                    f"<td>{vram.get(sysname, '')}</td><td>{100 * flagged / max(1, len(rs)):.1f}%</td>"
                    f"<td>{', '.join(f'{k} {v}' for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))}</td></tr>")

    sys_boxes = "".join(
        f'<label><input type="checkbox" class="sys" value="{html.escape(s)}|{v}" checked> {html.escape(s)} · {v}</label>'
        for s, v in [("app (logged)", "-")] + systems)
    dirs = sorted({f"{i['src']}→{i['tgt']}" for i in items})
    dir_opts = '<option value="all">all directions</option>' + "".join(f"<option>{d}</option>" for d in dirs)

    cards = []
    for item in items:
        ctx = "\n".join(f"{c['speaker'] or '·'} [{c['lang']}]: {c['text']}" for c in item["context"])
        trs = []
        if item["app"]:
            trs.append(f'<tr class="row" data-sys="app (logged)|-" data-flagged="0"><td>app (logged)</td>'
                       f'<td>{html.escape(item["app"])}</td><td class="ms"></td></tr>')
        for sysname, variant in systems:
            r = by_item.get(item["id"], {}).get((sysname, variant))
            if not r:
                continue
            badges = "".join(f'<span class="flag{" bad" if f in BAD_FLAGS else ""}">{f}</span>' for f in r["flags"])
            text = html.escape(r["output"]) if r["output"] else f'<i>{html.escape(r["error"] or "")}</i>'
            trs.append(f'<tr class="row" data-sys="{html.escape(sysname)}|{variant}" '
                       f'data-flagged="{int(bool(set(r["flags"]) & BAD_FLAGS))}"><td>{html.escape(sysname)} · {variant}</td>'
                       f'<td>{text}{badges}</td><td class="ms">{r["ms"]} ms</td></tr>')
        cards.append(
            f'<div class="card" data-dir="{item["src"]}→{item["tgt"]}" data-short="{int(item["short"])}">'
            f'<div class="meta">{item["src"]}→{item["tgt"]} · {item["ts"]}{" · short" if item["short"] else ""}</div>'
            + (f'<div class="ctx">{html.escape(ctx)}</div>' if ctx else "")
            + f'<div class="src"><b>{html.escape(item["speaker"] or "·")}</b> {html.escape(item["text"])}</div>'
            f'<div class="wrap"><table>{"".join(trs)}</table></div></div>')

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Translator Comparison</title>
<style>{REPORT_CSS}</style></head><body><main>
<h1>Translator comparison</h1>
<p class="sub">{len(items)} lines from logs/ · seed {meta['seed']} · {meta['started']} ·
greedy decoding · VRAM is each model's peak on top of what was already in use ({meta['vram_before']} MB)</p>
<div class="wrap"><table><tr><th>System</th><th>Variant</th><th>n</th><th>Median ms</th><th>p95 ms</th>
<th>VRAM +MB</th><th>Flagged</th><th>Flags</th></tr>{''.join(rows)}</table></div>
<p class="meta">Red flags (counted in "Flagged"): {', '.join(sorted(BAD_FLAGS))}. Amber flags are softer hints worth
a look: traditional (non-Simplified in a zh target), mandarin? (no Cantonese particles in a yue target), too-long.</p>
<h2>Lines</h2>
<div class="controls"><select id="dir">{dir_opts}</select>
<label><input type="checkbox" id="short"> short lines only</label>
<label><input type="checkbox" id="flagged"> only lines with a red flag</label></div>
<details><summary>Systems shown</summary><div class="controls" style="position:static">{sys_boxes}</div></details>
{''.join(cards)}
</main><script>{REPORT_JS}</script></body></html>"""
    path = os.path.join(out_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-lang", type=int, default=40, help="source lines sampled per source language")
    ap.add_argument("--models", default=",".join(MODELS), help="comma-separated subset of: " + ", ".join(MODELS))
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--no-nllb", action="store_true", help="skip the NLLB-600M baseline")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    started = datetime.now()
    out_dir = os.path.join(EVAL_DIR, "results", started.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir)

    segments = load_segments()
    items = sample_items(segments, args.per_lang, args.seed)
    print(f"{len(segments)} unique captions in logs/, sampled {len(items)} translation items -> {out_dir}")
    with open(os.path.join(out_dir, "items.json"), "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=1)

    results = []
    jsonl = open(os.path.join(out_dir, "results.jsonl"), "w", encoding="utf-8")

    def write(r):
        results.append(r)
        jsonl.write(json.dumps(r, ensure_ascii=False) + "\n")
        jsonl.flush()

    vram_before = VramMonitor.read()
    vram = {}
    variants = [v for v in args.variants.split(",") if v]
    if not args.no_nllb:
        vram["nllb-600m"] = run_nllb(items, write)
    for name in [m for m in args.models.split(",") if m]:
        try:
            vram[name] = run_llm(name, items, variants, out_dir, write)
        except Exception as e:
            print(f"  {name} failed: {e}", flush=True)
    jsonl.close()

    item_by_id = {i["id"]: i for i in items}
    nocontext = {(r["item"], r["system"]): r["output"] for r in results if r["variant"] == "casual"}
    for r in results:
        item = item_by_id[r["item"]]
        base = nocontext.get((r["item"], r["system"])) if r["variant"] == "casual+ctx" else None
        r["flags"] = flags_for(item, r["output"], base)

    path = write_report(out_dir, items, results, vram,
                        {"seed": args.seed, "started": started.strftime("%Y-%m-%d %H:%M"), "vram_before": vram_before})
    print(f"\nReport: {path}")


if __name__ == "__main__":
    main()
