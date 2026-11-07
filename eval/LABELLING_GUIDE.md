# Step 2: hand-pick the human gold set (about 40 minutes)

Do this **before** running the full tool, so your picks are not influenced by its output.

## 1. Make the reading pack

```bash
python -m eval.make_reading_pack data
```

This writes 12 episodes (2, 3, 5, 8, 11, 15, 20, 25, 30, 36, 42, 48) to `outputs/reading_pack/` as
readable timed scripts. Open them in Antigravity's Markdown preview. Episode 1 is excluded because
example promos from it were discussed while designing the tool, so it would not be a blind pick.

## 2. Pick promos like an editor

Aim for **about 15 promos** in total, roughly 1–2 per episode. Skip an episode if nothing in it would work.

A promo must:

- be **20–60 seconds** in total
- **hook in the first 3 seconds** (slap, reveal, shocking line)
- **end on a cliffhanger** (unresolved: a threat, a question, someone bursting in)
- **make sense to a stranger** who has never seen the show

Most should be one continuous stretch. You may join up to 3 pieces, including across episodes
(e.g. a hook in Ep 2 and the cliffhanger in Ep 3). Start and end on a `[shot]` line's time where you can.

## 3. Fill `eval/gold_human.csv`

Copy `eval/gold_human_template.csv` to `eval/gold_human.csv`. **One row per segment**; a 2-piece promo is 2 rows
with the same `gold_id`.

| Column | Example | Notes |
|---|---|---|
| gold_id | G01 | same id for all segments of one promo |
| segment | 1 | order within the promo |
| episode | 1 | (example from the fictional demo show) |
| start | 00:09.00 | mm:ss.ss from the script |
| end | 00:35.00 | |
| hook | "Vivienne kicks the bucket over." | quote or short description |
| cliffhanger | "He is asking for you. Only you." | |
| why | Public humiliation flips into a mystery summons | one line |
| confidence | 5 | 1–5: how sure you'd post it |

Write the row as soon as you decide; don't revise picks after seeing tool output (note second thoughts in a
`notes` column instead). That honesty is what makes the comparison credible.
