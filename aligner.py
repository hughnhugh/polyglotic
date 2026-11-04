"""
Word alignment: which words of a caption's clause and its translation say
the same thing (火锅店 - "hot pot place"), for the overlays' word links -
colours, arrows (see docs/word-links.md).

awesome-align (Dou & Neubig, "Word Alignment by Fine-tuning Embeddings on
Parallel Corpora", 2021): multilingual BERT fine-tuned for alignment,
Chinese-English among its evaluated pairs - the co-trained checkpoint
aneuraz/awesome-align-with-co, BSD-3, in models/awesome-align-with-co
(1.1GB). Its method, as published: both lines through the model, their
8th layer's token vectors compared, and a pair of tokens kept where each
is the other's likely match - a softmax over the similarities, both ways,
both over ALIGN_THRESHOLD. Tokens are then mapped back to words: Chinese
by the dictionary's segmentation (mBERT reads Chinese a character at a
time), other languages by spaces and punctuation.

Pure model code, no GUI - tested in test_aligner.py (needs the model).
"""

import os
import re
import threading

import dictionary

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models", "awesome-align-with-co")
ALIGN_LAYER = 8          # awesome-align's: the layer its fine-tuning aligns
ALIGN_THRESHOLD = 1e-3   # its default - both ways' softmax over this

_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")
_lock = threading.Lock()
_model = _tokenizer = _device = None
_loading = threading.Event()
_wanted = False  # on - so a load that finishes after stop() is let go of
error = None  # why it couldn't load, if it couldn't


def _status(text):
    """The main window's status for the model ("aligner" - Home, Models)."""
    from overlay_core import status
    status.set("aligner", text)


def show_status():
    """Sets the status to what the model's doing now."""
    if not installed():
        _status("Not installed - run setup.bat")
    elif _model is not None:
        _status(f"Ready (on the {_device.upper()})")
    elif _loading.is_set():
        _status("Loading...")
    elif error:
        _status(f"Failed - {error}")
    else:
        _status("Off")


def installed():
    return all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in ("config.json", "vocab.txt")) and any(
        os.path.exists(os.path.join(MODEL_DIR, f)) for f in ("model.safetensors", "pytorch_model.bin"))


def start():
    """Loads the model in the background (see load()) - no-op if it's
    loaded or loading, or isn't installed."""
    global _wanted
    _wanted = True
    if _model is not None or _loading.is_set() or not installed():
        show_status()
        return
    _loading.set()
    show_status()

    def run():
        global error
        try:
            load(DEVICE)
            error = None
        except Exception as e:  # missing transformers, a broken download...
            error = f"{type(e).__name__}: {e}"
            print(f"[aligner] couldn't load the word-alignment model: {error}")
        finally:
            _loading.clear()
        if not _wanted:  # switched off while it loaded
            stop()
        else:
            show_status()

    threading.Thread(target=run, daemon=True, name="aligner-load").start()


def stop():
    """Lets go of the model - its ~1.2GB of memory - when word links are
    switched off. Captions in the meantime just get no links."""
    global _model, _tokenizer, _wanted
    _wanted = False
    with _lock:  # not while a caption's being aligned
        released = _model is not None
        _model = _tokenizer = None
    if released:
        import gc
        gc.collect()
    show_status()


def loading():
    return _loading.is_set()


# On the CPU: ~30ms a clause pair there (see docs/word-links.md) - fast
# enough, and it leaves the GPU, and its memory, to speech and translation.
DEVICE = "cpu"


def load(device=None):
    """Loads the model (once) - on `device`, or the GPU if there is one, at
    half precision there. Blocks for a few seconds; call from a worker."""
    global _model, _tokenizer, _device
    with _lock:
        if _model is not None:
            return
        import torch
        import transformers
        from transformers import BertModel, BertTokenizerFast
        transformers.logging.set_verbosity_error()  # its notes on unused weights (the fine-tuning heads)
        transformers.logging.disable_progress_bar()
        _device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
        model = BertModel.from_pretrained(MODEL_DIR, add_pooling_layer=False)
        if _device == "cuda":
            model = model.half()
        _model = model.to(_device).eval()


def loaded():
    return _model is not None


def words(text):
    """[(start, end)] of `text`'s words: dictionary words where it's
    Chinese, runs of letters/digits elsewhere - punctuation left out."""
    out = []
    for start, end in dictionary.segment(text):
        piece = text[start:end]
        if dictionary.has_han(piece):
            out.append((start, end))
        else:
            out += [(start + m.start(), start + m.end()) for m in _WORD.finditer(piece)]
    return out


def _vectors(text, spans):
    """The model's ALIGN_LAYER vectors for `text`'s words' tokens, and which
    word each token is."""
    import torch
    ids, owner = [], []
    for n, (start, end) in enumerate(spans):
        pieces = _tokenizer.encode(text[start:end], add_special_tokens=False) or [_tokenizer.unk_token_id]
        ids += pieces
        owner += [n] * len(pieces)
    ids = ids[:510]  # BERT's limit - far beyond a clause
    owner = owner[:510]
    batch = torch.tensor([[_tokenizer.cls_token_id] + ids + [_tokenizer.sep_token_id]], device=_device)
    with torch.no_grad():
        hidden = _model(batch, output_hidden_states=True).hidden_states[ALIGN_LAYER][0, 1:-1]
    return hidden.float(), owner


def align(source, target):
    """[(i, j)]: word i of `source` (words(source)) and word j of `target`
    say the same thing - each pair once, in source order. [] if either
    has no words, or the model isn't loaded (see load(), start())."""
    return sorted(_scored(source, target))


def _scored(source, target):
    """{(i, j): how sure}: align()'s pairs, each with the weaker of its two
    ways' probabilities (each word's softmax over the other line), at its
    surest token pair - 1.0 for teach - 教, nearer the threshold for a
    looser match like some - 一下."""
    import torch
    src_spans, tgt_spans = words(source), words(target)
    if not src_spans or not tgt_spans:
        return {}
    with _lock:
        if _model is None:
            return {}
        src, src_owner = _vectors(source, src_spans)
        tgt, tgt_owner = _vectors(target, tgt_spans)
    similarity = src @ tgt.T
    surety = torch.minimum(torch.softmax(similarity, dim=-1), torch.softmax(similarity, dim=-2))
    scored = {}
    for i, j in torch.nonzero(surety > ALIGN_THRESHOLD).tolist():
        key = (src_owner[i], tgt_owner[j])
        scored[key] = max(scored.get(key, 0.0), float(surety[i, j]))
    return scored


# Never linked: particles, fillers, articles - a link to them says nothing.
_GRAMMAR_HAN = set("的了吗呢吧啊呀嘛哦嗯呃哎哈")
_GRAMMAR_EN = {"a", "an", "the", "of", "to", "s"}
# Linked, but not key words (a link's `key` is False - the overlay's "Key
# words" leaves them out): pronouns, 是/在/有, 这个/那个, 然后, 也... and
# their English. Worth seeing for a beginner (他们 - "they"), noise later on.
_MINOR_HAN = set("就是这那个也都还在有和跟把被给让对很太我你他她它们说然后其实")
_MINOR_EN = dictionary._STOP


def _is_grammar(word):
    if dictionary.has_han(word):
        return all(ch in _GRAMMAR_HAN for ch in word if dictionary.is_han(ch))
    return word.lower() in _GRAMMAR_EN


def _is_minor(word):
    """A grammar word, or one that isn't a key word (see _MINOR_HAN)."""
    if dictionary.has_han(word):
        return all(ch in _GRAMMAR_HAN or ch in _MINOR_HAN for ch in word if dictionary.is_han(ch))
    return word.lower() in _GRAMMAR_EN or word.lower() in _MINOR_EN


def links(source, target):
    """[((start, end) in source, (start, end) in target, key, surety)]: the
    aligned words as character spans, in source order - particles, fillers
    and articles left out on both sides, a word aligned to neighbouring
    words joined into one span (火锅 - "hot pot"), and neighbouring words
    aligned to the same span one link (阳澄 - "Yangcheng"). key: a link
    between content words, not pronouns, 是, 这个 and the like (see
    _MINOR_HAN). surety: how sure the model is (see _scored) - so when
    there's room for only some, the overlay keeps the surest."""
    src_spans, tgt_spans = words(source), words(target)
    keep_tgt = [not _is_grammar(target[a:b]) for a, b in tgt_spans]
    scored = _scored(source, target)
    by_source = {}
    for (i, j), surety in sorted(scored.items()):
        if keep_tgt[j] and not _is_grammar(source[src_spans[i][0]:src_spans[i][1]]):
            by_source.setdefault(i, {})[j] = surety
    found = []  # (source word index, target span, whether its target words include a key word, surety)
    for i in sorted(by_source):
        js = sorted(by_source[i])
        run = [js[0]]
        for j in js[1:] + [None]:
            if j is not None and j == run[-1] + 1:
                run.append(j)
                continue
            key_target = any(not _is_minor(target[slice(*tgt_spans[k])]) for k in run)
            found.append((i, (tgt_spans[run[0]][0], tgt_spans[run[-1]][1]), key_target,
                          max(by_source[i][k] for k in run)))
            if j is not None:
                run = [j]
    out = []  # [source span, target span, key, surety, last source word index]
    for i, span, key_target, surety in found:
        key = key_target and not _is_minor(source[slice(*src_spans[i])])
        if out and out[-1][1] == span and out[-1][4] == i - 1:
            out[-1] = [(out[-1][0][0], src_spans[i][1]), span, out[-1][2] or key, max(out[-1][3], surety), i]
        else:
            out.append([src_spans[i], span, key, surety, i])
    return [(src, tgt, key, round(surety, 3)) for src, tgt, key, surety, _ in out]
