# Line-by-line captions

Design notes for showing a speaker's live caption as lines - one per
clause - rather than one block of text that rewrites itself. Started
2026-11-03. The options we didn't take are kept here in case we come back
to this feature.

## The problem

During a monologue the whole caption keeps changing, and it's hard to read
along with either the transcription or the translation. Three things
rewrite text you've already read:

1. **Live updates.** Every `live_options["interval"]` (1.5s) the current
   section - up to `LIVE_SECTION_SEC` (6s) of audio - is transcribed again,
   and its *whole* translation is redone from scratch. Even when the
   transcription only adds words at the end, the translator often
   rephrases the whole sentence, so the yellow line moves even when the
   white one barely did.
2. **The final pass.** When they pause, up to `BRIDGE_FINAL_MAX_SEC` (20s)
   of speech is transcribed and translated again as one block, and the
   whole caption changes at once.
3. **Layout.** The caption is one block of wrapped text, so any change
   near its start re-wraps every line below it.

The mechanics themselves (sections, the final pass, Discord joins) work
well and stay. The fix is in what gets translated as a unit and how it's
shown.

The hard part is the translation. The transcription splits easily at
punctuation (Qwen3-ASR punctuates), but a translation of the whole
paragraph won't split into lines that match.

## Phases

1. **Line by line (option B, below).** Everything is still re-evaluated on
   every update. But each clause is its own row, translated on its own, and
   a clause whose transcription didn't change keeps its row unchanged. In
   practice only the last line or two moves.
2. **Locking clauses (option D).** Clauses that have settled stop being
   re-evaluated at all - or are re-evaluated only when there's reason to.

## Options

### A. Split the display only - not taken

Leave the pipeline alone and break both texts into lines at punctuation
when drawing them.

- Cheapest - overlay-only.
- The translation lines won't match the original lines (the translator
  merges and reorders clauses), and the translation churns exactly as
  much as now. It looks tidier without fixing the problem.

**Revisit if:** we only want to fix the re-wrapping (source 3) - e.g. as a
fallback layout for languages B's splitting handles badly.

### B. Translate clause by clause - **chosen**

The pipeline splits each transcription into clauses, translates each one
on its own, and the overlay shows a row per clause: what was said (white),
its translation (yellow).

- Rows match by construction - each translation is its clause's.
- Translations are cached by clause text: a clause whose transcription
  didn't change gets the identical translation back, so its row doesn't
  move even though everything was re-evaluated.
- Each clause can be given the speaker's preceding clauses as context, so
  its translation doesn't lose the thread.
- Cost: short translator requests, mostly cache hits - similar to now.
- Risk: translating clause by clause can lose quality - Chinese to English
  especially, where meaning runs across commas. Measured before building
  the UI (see Evaluation).

### C. Translation appears only when a clause is finished - deferred

On top of B: the clause still being spoken shows only its (dimmed)
original. Its translation appears once the clause is complete - more speech
came after it, or the speaker paused.

- The yellow lines never rewrite, only grow - the steadiest to read.
- Each translation arrives one clause later.

**Revisit if:** B's last line still churns too much to read. It's cheap once
B exists - mostly a display rule, maybe a setting.

### D. Locking clauses - phase 2

A clause that comes out the same in two consecutive updates is locked: its
text and translation are fixed, and the final pass only replaces the
unlocked tail (or a locked clause only if it's now very different). The
standard approach for streaming speech recognition ("local agreement" -
Macháček et al., *Turning Whisper into Real-Time Transcription System*,
2023).

- Builds directly on B's clause rows.
- Saves no GPU time by itself - locked clauses are still in the audio
  that's re-transcribed. To stop transcribing them, we need where each
  clause ends in the audio. Qwen3-ASR doesn't give word timings without
  its separate forced aligner (Qwen3-ForcedAligner); the short pauses the
  VAD sees are a cheaper approximation. Sections could then start at
  locked clause boundaries instead of every 6s.
- "Re-evaluate situationally": re-check a locked clause only when the new
  transcription of it differs a lot, or with more context. Qwen3-ASR
  reports no confidence (`avg_logprob` is None), so agreement between
  updates is the practical stand-in for confidence.

### E. Cut the audio at short pauses - not taken

Split the audio at short pauses (~250ms, under the 0.8s that ends an
utterance) and transcribe each chunk as its own line, with only the last
chunk live.

- Very stable, and no text alignment problem.
- Accuracy drops at every cut (words clipped, no context across chunks),
  and it replaces the section/final-pass mechanics that work well.

**Revisit if:** B and D can't get earlier lines stable enough - or as the
audio side of D, cutting only at pauses where a clause has been locked.

## B in detail

- **Splitting** (`clauses.py`): always after 。！？!? and a sentence-ending
  `.`; after ，,；;：: only once the clause so far is long enough
  (`MIN_SOFT_HAN` Han characters, `MIN_SOFT_WORDS` words); fragments shorter
  than a minimum ("嗯，", "Yeah,") merge into their neighbour. Punctuation
  stays with its clause, so the clauses join back into the text.
- **Translating** (`translate_vc.process_segment`, `translator.py`): each
  clause separately, with the caption's previous clauses and their
  translations as earlier chat turns (see the review log for why that and
  not a context section), cached by (clause, source language, target). The
  cache key doesn't include the context, so a clause keeps its first
  translation even if what came before it changes.
- **Data**: a record carries `clauses` - `[{"native", "translations"}]` -
  next to the joined `native` / `translations`, which the session log,
  transcript page and side chat keep using. A live caption's sections
  (see `LIVE_SECTION_SEC`) put their clause lists end to end, so a clause
  cut by a section boundary shows as two rows until the final pass.
- **Overlay**: a row per clause - what was said (white), then its
  translation (yellow), the existing colour rule applied per clause, a
  little more space between clauses than within one. A live caption's last
  row trails off "…". A rewrite crossfades only from the first rewritten
  row down.
- **The final pass** still redoes the whole stretch, as now. Clauses that
  come out the same keep their translations (the cache), so only changed
  rows change.

## Evaluation

1. **Translation quality, offline.** On real session logs: lines that split
   into more than one clause, translated whole (the logged translation),
   clause by clause, and clause by clause with the preceding clauses as
   context. Judged side by side.
2. **Churn, live.** How much already-shown text changes per update - before
   and after. Needs a trace of every live update (today only finals are
   logged).

## Open questions

- Row layout: a translation under each clause (chosen to start), or all the
  translation rows then all the original rows (smoother as subtitles)?
- Should the final pass be allowed to rewrite earlier rows, once D exists?
- Clause length limits on a narrow overlay: split long unpunctuated runs?

## Review log

(Each implement-and-review cycle: what changed, what we saw, what's next.)

### Cycle 1 - 2026-11-03: splitting, and how to give a clause its context

**Splitting.** `clauses.split` over 4,571 logged lines: 58% stay one clause,
the rest split into 2-6+. Every line joins back to its text exactly.
Counting "I'm" as two words made English rows too short, and 2-word tails
("for sure.") stood alone - fixed (`MIN_CLAUSE` 5).

**Translation, offline** - 55 logged lines with more than one clause (40
Mandarin to English, 15 English to Chinese), the app's Hy-MT2 Q4_K_M,
greedy. Four ways to translate a clause, against the logged whole-line
translation:

- **Isolated** - each clause alone. Clearly worse: lost subjects ("I
  started cleaning" for "they"), short fragments turned to nonsense ("Eat,
  eat, eat more methylene red!", "This one's totally you"), a Chinese phrase
  left untranslated.
- **Preceding clauses in the prompt** (a 【上文】 / `[Preceding Text]`
  section, "don't translate it"). Fixes most of isolated's mistakes, but in
  12 of 55 lines a row also translated - or only translated - the context,
  which would show up as a row repeating earlier ones. Unusable as is.
- **Hy-MT's own contextual template** ("参考上面的信息…注意不需要翻译上文"). No
  repeats, accurate, but stiff and formal ("我确实有过连续三晚都做的经历",
  "您") - the register the app's prompt deliberately avoids.
- **Earlier clauses as previous chat turns** - the last 3 clauses and their
  translations as user/assistant turns, then the clause, each turn in the
  app's usual prompt. No repeats, keeps the casual tone, consistent from row
  to row. **Chosen.**

Whole-line translation is still somewhat better where meaning runs across
clauses (who "he" is, a subject dropped in Chinese) - the price of rows
that don't move. Worth watching live.

Scripts: `clause_translation_eval.py` / `clause_translation_eval2.py`
(kept outside the repo for now - move into `eval/` if we iterate on this).

**Built:**
- `clauses.py` splits; `translate_vc.process_segment` translates per clause
  with the last `CLAUSE_HISTORY` (3) clauses as chat turns, through a
  `CLAUSE_CACHE_SIZE` cache, and a live caption's later sections get its
  earlier sections' clauses as context.
- Records and `Caption`s carry `clauses`.
- The overlay draws a row pair per clause, with `CLAUSE_GAP` between pairs.
- A rewrite now crossfades only from the first rewritten row down. Before,
  the whole caption crossfaded through itself, which dims unchanged text to
  about 75% mid-fade - every tail change would have flickered the rows above.
- Word lookups pair a row with its own clause.

**Measured - churn, simulated.** Six long logged Mandarin monologues,
revealed ~6 characters (about 1.5s of speech) per update, real translator,
no speech-recognition revisions (the simulation's limit):

| | whole line (before) | clause by clause |
|---|---|---|
| shown translation rewritten per update (from the first changed character on) | 50% | 24% |
| updates rewriting more than half of it | 33 / 70 | 9 / 70 |
| translator requests per update | 1.0 | 1.3 |
| time translating per update | 0.35s | 0.24s |

What's left is mostly the row still being spoken.

**Seen in the clause translations:** a row ending "…" where its clause
ends at a comma (the model marking it unfinished), and a capital at a row's
start mid-sentence ("In a big city like Shanghai, / We need to..."). On
separate rows both read like subtitle line breaks; in the joined text
(transcript, log, side chat) they're a little odd.

**To watch live:**
- how much the speech model's own revisions move earlier rows - the
  simulation has none;
- the final pass's latency on long monologues - each changed clause is its
  own request now;
- whether a two-language overlay (a second language) gets too tall with a
  row pair per clause.

### Cycle 2 - 2026-11-03: movement in one direction

**Feedback from the first live run:** still too much pushing around -
things went up and down. Prefer the transcript to move strictly one way;
maybe fixed boxes for lines that aren't settled yet, or some buffer space.
Also: the same short line captioned twice under two speakers ("挺好。" as
小豆漿 and as kimchie.).

**Cause of the up-and-down:** the stack was anchored to the overlay's
bottom, so a caption growing pushed everything above it up and one
shrinking pulled everything above it back down. Shrinks are common now -
the final pass tidies a live caption's rows, duplicates are taken back.

**Built:**
- **One-way layout** (`CaptionCanvas._following_top`). While following the
  live end, the stack's top is as low as it can be while it fits *and* no
  caption already placed moves down. A shrink moves only what's below it,
  up, leaving blank space at the bottom; growth uses that space first (at
  the bottom) or pushes what's above up. It settles at the bottom afresh
  only after a resize, a restyle, or coming back from scrolling.
- **Headroom.** A live caption at the bottom that has to push leaves
  `HEADROOM_LINES` (2) spare lines below it, so it pushes now and then
  rather than on every update - the "fixed box" for unsettled lines.
- **Duplicate speakers** (separate from this feature, `translate_vc`
  `DUPLICATE_OVERLAP`): two Discord speakers' captions overlapping by half
  their time with the same words - short ones included - are one; whoever's
  done more talking in the last 30s keeps it, about even the later one.

**Checked offscreen:** 18 steps - a monologue growing, its shorter final,
overlapping speakers, a duplicate taken back, a run of new captions -
with no caption on screen moving down at any step.

**Still moves down:** inside a caption, when a row grows a line, the rows
under it in the same caption shift down (e.g. the live clause's white row
wrapping pushes its yellow row down a line). It's crossfaded, not slid. A
fixed minimum height per live row would stop it, if it shows.

**To watch:** whether the blank space a shrink leaves at the bottom looks
odd; whether 2 lines of headroom is the right amount.

### Cycle 3 - 2026-11-03: scrolling, and lines not worth a caption

**Feedback:** "looks good". Remove small agreement lines - a "don't
transcribe" list, 嗯 when it's said on its own. And scrolling stopped
working in some states.

**Scrolling** was broken by cycle 2: at the live end the stack can sit
above the bottom-aligned position (space left under it), but the wheel
still clamped to that position - so with more than a notch (60px) of space
left, a notch up counted as "at the bottom" and nothing moved. Headroom
under a live caption (~66px) was always enough to trigger it. Now
`CaptionCanvas.scroll_by` scrolls from where the stack actually is, only
going back to live when scrolled *down* to the bottom; and when everything
fits, a scroll settles it at the bottom (bringing back the oldest caption
if a shrink had left it partly off the top). Checked offscreen: several
notches up under a live caption and after a big shrink, then back down to
live.

**Not captioned on their own** - a setting (`skip_words`, on the Languages
page): a caption made only of these words, repeated or drawn out, isn't
shown, live or final (`translate_vc._only_skip_words`). Default: 嗯, 嗯哼,
呃, 额, ừ, ờ, mhm, mm, mm-hmm, hmm, uh-huh, um, uh. On the last 8 session
logs that drops 232 of 5,449 captions - all of them fillers (嗯。 165 times,
呃。 16, Um. 12, Hmm. 10...). Left off at first as saying a little more
than 嗯 - 对 (32), 哦 (31), 啊 (29), 哎 (15) - then added at the user's
request (Oh. (12) still isn't). Also: a caption taken back is now logged with why,
and only if it was showing.

**Duplicate speakers, tightened** after the first live run of cycle 2's
rule: DurianMT's live "都是一模一样的" was taken back as a copy of 小豆漿's
final "那个 background 都是一模一样的。" (it was *inside* it), then a later
DurianMT update flipped the decision and took back 小豆漿's final. Now:
a line much shorter than the other (under half) isn't the same speech even
inside it - two cuts of one stretch of audio mostly overlap, so that's more
likely an echo of its end; a live update never takes back a finished
caption (its own final can); and once a caption has lost to another, it
stays lost.
