# Word links

Design notes for showing which words of a caption's original and its
translation say the same thing - so you can learn on the spot, in real
time, without hovering each word to find its counterpart. Started
2026-11-03, from a suggestion by a friend testing the app ("colour-matching
words in the target language to the translation would make that easy").
Builds on the clause rows (docs/line-by-line-captions.md): each said /
translated pair is short, so a word's counterpart is close by and easier
to find.

## Modes

One setting, **Word links: Off / On hover / Colours / Arrows**:

- **On hover** - hovering a word highlights the word(s) it links to in the
  other row, next to the lookup popup. Least visual noise. Needs the
  overlay unlocked (locked, it's click-through for games) - or the
  Transcript page.
- **Colours** - linked words share a coloured underline, always on; the
  layout doesn't change. Underlines, not text colour: text colour already
  means something (white said, yellow translation, saved words in their
  category's colour). Rows are short, so 4-5 colours cycling per row pair
  are enough.
- **Arrows** - a study mode that changes the layout: a gap of ~16-20px
  between each original and its translation, thin curved links across it
  from word to word, each with its colour, and the words underlined in it
  too so crossing curves can still be followed.

Decided for arrows:

- Only on rows that have settled - not the clause still being spoken,
  whose words (and links) keep changing.
- Links only between the original's last line and the translation's first
  line when a row wraps (a link from an upper line would cut through the
  text); matched words on other lines get just the underline.
- Content words only, confident links only, at most ~6 per row pair.
- Needs originals shown.

## Finding the links

Options considered:

1. **The dictionary** (CC-CEDICT definitions matched against the
   translation's words). Instant, offline, nothing new - but on 160 real
   clause pairs only 49% of content words found their counterpart, missing
   paraphrase ("就很好玩儿啊！" - "That's so much fun!"), names (阳澄湖 -
   "Yangcheng Lake") and idioms, with some false matches from loose
   definitions (讲 - "about", 拜师 - "become").
2. **A word-alignment model** - **chosen**. awesome-align (Dou & Neubig,
   2021): multilingual BERT fine-tuned for alignment, Chinese-English among
   its evaluated pairs; checkpoint `aneuraz/awesome-align-with-co`, BSD-3,
   1.1GB, in `models/awesome-align-with-co`. Needs `transformers` in the
   main venv. `aligner.py`.
3. **The local assistant** - on demand only, ~1s, and needs it switched on;
   Explain's "Find it in the Chinese" already does this for one phrase.

## Review log

### Cycle 1 - 2026-11-03: the aligner, offline

Same 160 clause pairs (Mandarin clauses and their clause-by-clause English
translations from the line-by-line evaluation), CPU:

| | awesome-align | dictionary |
|---|---|---|
| content words linked | 610 / 789 (77%) | 384 / 789 (49%) |
| of the dictionary's matches, linked to the same word | 81% | - |
| time per clause pair | median 30ms, slowest 42ms | ~0 |

Where the two disagree the model is often the one that's right (讲 -
"talking", where the dictionary had "about"). Examples: 火锅 - "hot pot",
大闸蟹 - "hairy crabs", 阳澄 - "Yangcheng", 好玩儿 - "much fun", 拜 -
"worship". Mistakes seen: a word linked to scattered words ("说不通" - "None
... say ... sense"), 无性 once linked to "self".

On the CPU it needs no VRAM; 30ms per clause per target is affordable in
the pipeline (only changed clauses need it - cache by clause and
translation). `transformers` 5.22 warns about the tokenizer's "regex
pattern" - spurious here: its tokens match BERT's reference tokenizer.

`aligner.links()` leaves out grammar words on both sides (的/了/就/很,
"the"), joins a word's neighbouring counterparts into one span (火锅 -
"hot pot"), and neighbouring words with the same counterpart into one link
(阳+澄 - "Yangcheng").

**Next:** an offscreen prototype of the Colours and Arrows modes, rendered
from real clause pairs, to look at before wiring anything in. Then: links
computed in the pipeline and carried with each clause; the setting; setup
(`setup.bat -WordLinks`: `transformers` and the model, opt-in like the
assistant).

### Cycle 2 - 2026-11-03: Colours and Arrows, prototyped offscreen

Drawn by the real overlay code (`CaptionCanvas._link_parts`), behind the
overlay setting `word_links` (off / colours / arrows; off by default),
from real clause pairs aligned by `aligner.links()` - nothing computes
links live yet. Captions carry them per clause (`Caption.clause_links`,
`{language: [(span in the clause, span in its translation)]}`).

- **Colours:** each link's words underlined in one of 5 colours
  (`LINK_COLORS`: orange, violet, cyan, pink, green - none white or the
  translation's yellow), at most `MAX_LINKS` (6) per pair. The layout
  doesn't change.
- **Arrows:** the underlines plus an outlined curve from each word to its
  counterpart, across an `ARROW_GAP` (18px) inside every pair - links or
  not, so a row doesn't jump when its links arrive. Pairs then needed more
  space between them (`ARROW_CLAUSE_GAP`, 28px): with the old 9px a
  translation sat closer to the next clause than to its own.
- Only settled clauses get links - not a live caption's last, whose words
  are still changing. On a wrapped row only words on the two facing lines
  get a curve ("Go ahead and chat." wrapping: 聊 and "chat" underlined
  alike, no curve).
- Links seen: 金山寺 - "Jinshan Temple", 庭院 - "courtyard", 两句 - "two
  sentences", 台湾 - "Taiwan", 在意 - "worry", 晚餐 - "dinner"; looser:
  要求 - "asking", 剑 (of 螯剑, "pincer") - "pincers".

**To decide:** the palette (orange under the yellow translation is the
weakest contrast); whether 18px is enough room for the curves.

### Cycle 5 - 2026-11-03: status, colours, no cap

From testing live:

- **Status:** the alignment model has a status like the other models
  ("aligner": not installed / off / loading / ready / failed), shown on
  Home and in a Word alignment card on the Models page - next to the local
  assistant, now on Home too. Switching word links off lets go of the model
  (`aligner.stop`), freeing its ~1.2GB of memory; on loads it again.
- **10 colours** (`LINK_COLORS`): with 5, a pair's 6th link shared the 1st's
  colour. Ordered so neighbouring links get far-apart hues; past 10 they
  cycle.
- **"Why didn't teach match?"** - "I myself will need to do some deep dive
  if I teach it to you." / "要是我教你的话，我自己也得好好研究一下才行。": the
  model linked teach - 教 at full confidence, but the arrows cap of 6 kept
  the first six links in the line (I, myself, need, some, deep, if) and
  dropped it. Links now carry the model's confidence (`aligner._scored`:
  the weaker of the two ways' probabilities), and a cap keeps the surest,
  key words first, coloured in reading order.
- **No cap, for now** (`MAX_LINKS = None`): the user wants to see the
  model's full capability. The surest-first ranking stays, for if a cap
  comes back.
