# Method

How Kwan works, in four parts: the curriculum map, the learner model, the recommender and the practice settings.

The formulas are here for readers who want them. You can skip them and still follow the page.

## 1. The curriculum map

### Skills

Kwan's skills are the **learning indicators** of the NaCCA Standards-Based Curriculum for Mathematics, Basic 1 to Basic 6. Each indicator has a code printed in the curriculum. The code says where the skill sits:

```
B4.1.1.1.2
│  │ │ │ └─ indicator 2
│  │ │ └─── content standard 1
│  │ └───── sub-strand 1
│  └─────── strand 1 (Number)
└────────── Basic 4
```

The four strands are Number, Algebra, Geometry and Measurement, and Data.

Inside Kwan the code gets a subject prefix, `math:B4.1.1.1.2`, so that other subjects can be added later without a clash. The plain code is still accepted.

For each skill Kwan holds the code, the class, the strand and sub-strand names, and a **short label** written by Kwan contributors (60 characters or fewer). The full description and the exemplars are NaCCA's text. Kwan reads them from the official PDFs when you build it and does not store them in the repository. See [Governance](governance.md).

### Misprinted codes

The printed curriculum has a few indicator codes that do not fit their place: a code from another class, a code used twice, or a heading that is not an indicator at all. Kwan keeps a **corrections file** (`data/curriculum/corrections.json`) with one entry per misprint: the code as printed, the corrected code, what the parser does about it (recode, merge or drop the heading), the document and page, and the reason. The corrected code is the one Kwan uses everywhere. The printed code is kept as an alias where no other indicator owns it, so a request that uses the printed code still works. The [curriculum QA report](qa/curriculum-math.html) lists the corrections applied in the last build.

### Links

A link from skill A to skill B means "learn A before B". There are four kinds.

| Kind | What it joins | Where it comes from |
|---|---|---|
| `sequence` | One indicator to the next indicator in the same content standard | A rule |
| `vertical` | A content standard in one class to the same content standard in the next class | A rule |
| `bridge` | One content standard to the next content standard in the same sub-strand | A rule |
| `augmented` | Two skills that the rules do not join, often across strands (for example Number to Measurement) | A decision model, then a teacher |

Every link records its **provenance**: what made it, when, and for augmented links, which reviewer accepted or rejected it. A rule link says which rule (`rule:sequence`, `rule:vertical` or `rule:bridge`) and the date it was built. An augmented link says which decision model proposed it, the probability the model gave, and a review record with a status (`pending`, `accepted` or `rejected`), the reviewer id and the date. The graph file is the single source of this record, and the [skill map report](qa/graph-math.md) counts the links by type and by provenance.

### Augmented links

The three rules follow the order of the curriculum. They miss links that cross strands. To find those, Kwan asks a decision model one question about a pair of skills: *"Must a pupil be able to do A before they can learn B?"* The model answers with a probability.

- Only pairs that make sense are asked: A is in the same class as B or up to two classes below, the pair is not already linked, and the two skills are in the same strand or in a listed pair of strands.
- The method is first tested on Junyi Academy data, which has prerequisite links drawn by experts. The threshold is set there.
- Links are added from the most likely down. A link that would make a loop, or that only repeats a short path that already exists, is skipped.
- Each suggested link goes to a teacher on a review sheet. **Only accepted links ship.**

This work is done offline, before a release. No model is called when Kwan recommends.

### Checks

The map must be a directed graph with no loops: you can never follow links from a skill and arrive back at it. Kwan also checks that every link joins two real skills and that every skill can be reached. The [skill map report](qa/graph-math.md) shows these checks.

## 2. The learner model

The learner model answers one question: *how likely is this pupil to get this skill right, now?*

### The prediction

Kwan keeps two kinds of numbers for a pupil:

- `θ_u`: the pupil's overall ability;
- `θ_u,k`: an offset for each skill `k` the pupil has tried.

A question can also have its own difficulty, `β_i`. The chance of a correct answer is

```
p = σ(θ_u + θ_u,k − β_i)        where σ(z) = 1 / (1 + e^(−z))
```

`σ` squeezes any number into a chance between 0 and 1. More ability or an easier question pushes `p` up.

This is a **multi-level Elo** model. Elo is the rating system used in chess: after each game the rating moves by how surprising the result was.

### The update after an answer

Let `c` be 1 for a correct answer and 0 for a wrong one, and let `e = c − p` be the surprise. Then

```
θ_u   ← θ_u   + k_s / (1 + d · n_all) · e
θ_u,k ← θ_u,k + k_k / (1 + d · n_k)   · e
```

- `k_s` and `k_k` set how big a step is.
- `n_all` is the number of answers the pupil has given so far, and `n_k` the number on this skill.
- `d` makes the steps smaller as answers add up, so early answers move the estimate a lot and later ones fine-tune it.
- On a pupil's first answer to a skill, `θ_u,k` starts from the prior described in the next section.

A correct answer that the model did not expect (`p` low) moves the numbers up a lot. A correct answer it expected moves them only a little.

### Skills the pupil has not tried: graph transfer

Before a pupil has tried skill `k`, Kwan does not start from nothing. It uses a prior built from the map:

```
prior_k = −g · (class_k − 1) + t · mean(θ_u,j for tried prerequisites j of k)
```

- The first part lowers the starting point for each class level above Basic 1, so doing well on Basic 1 skills does not make Basic 6 skills look known. `g` is the step per class.
- The second part borrows evidence from the skills just before `k` on the map. If the pupil is strong on them, `k` starts higher. `t` sets how much is borrowed.
- When none of the prerequisites has been tried, a small fixed penalty is used in place of the second part.

This is why one answer changes more than one skill: evidence flows along the links.

### Forgetting

When answers come with a time, a skill that is not practised fades back towards its prior:

```
θ_eff = prior_k + (θ_u,k − prior_k) · exp(−Δdays / τ)
```

- `Δdays` is the number of days since the skill was last practised.
- `τ` sets how fast skills fade: after `τ` days without practice, about a third of what was learned above the prior is left. It is fitted on datasets with timestamps, and the value that ships comes from the results file:

<!-- RESULTS:START id=method-forgetting-tau source=data/eval_results.json -->
The shipped value of τ is 30 days, chosen on eedi_2020_task34, eedi_2020_task12. A streak mastery expires after 30 days without practice.
<!-- RESULTS:END -->

- Mastery is worked out from `θ_eff`. A mastery earned by a streak of right answers also expires, after the number of days given above.

Without times, nothing fades and the model gives the same results as the version without forgetting. The forgetting is applied when Kwan reads the state, not when it writes it, so a learner's stored numbers never change by themselves.

### Item difficulty

`β_i` comes from an **item bank** that is learned offline. A question with no entry has `β_i = 0`. If a question only has a label, the label gives the starting value: easy −0.5, medium 0, hard +0.5. The item banks shipped with Kwan are learned only from datasets that allow it. See the [Data card](data-card.md).

### Mastery

A skill counts as **mastered** when either is true:

- the predicted chance of a correct answer on an ordinary question (`β = 0`) is **0.85 or higher**;
- the pupil has answered it correctly **three times in a row**. A streak like this stops counting after the expiry period shown in the forgetting section above.

### Parameters

`k_s`, `k_k`, `d`, `g`, `t` and `τ` are chosen by a grid search on training data only. The values shipped with Kwan are stored in the bundle under `params`, and the datasets they were chosen on are listed in the [Model card](model-card.md). Parameters fitted on non-commercial datasets are never shipped.

## 3. The recommender

The recommender turns the mastery estimates into a short list of next skills.

### Which skills are candidates

A skill is a candidate when:

- it is **not yet mastered**;
- it is **ready**: every skill just before it on the map has an estimated mastery of at least 0.70 (a skill with nothing before it is always ready);
- it is at or below the class cap, if the teacher set one.

### The score

Each candidate gets a score from five signals, each between 0 and 1.

| Signal | Weight | What it measures |
|---|---|---|
| Readiness | 0.30 | How solid the foundations are: the lowest mastery among the skills just before it |
| Challenge | 0.25 | How close the pupil is to a 60% chance of success, where practice pays most: `1 − |m − 0.6| / 0.6` |
| Goal | 0.25 | How close the skill is to the goal: `1 / (1 + steps to the goal)` |
| Opens the way | 0.15 | How many later skills build on it, as a share of the largest such count |
| Time since practice | 0.05 | Rises as more steps pass since the skill was last practised |

```
score = 0.30 · readiness + 0.25 · challenge + 0.25 · goal + 0.15 · opens + 0.05 · recency
```

On top of the score, Kwan applies four rules. Each one is a change from the Round 1 engine, and each can be switched off: with `goal_mode="soft"`, `eps=0`, `variety=0` and `review=False`, Kwan gives exactly the Round 1 output, which the tests pin.

**Strict goal (the default).** When a goal is set, candidates on the path to the goal rank above all others: the skills that lead to the goal and the goal itself. If fewer than `k` skills on the path are ready, the rest of the list is filled from other candidates, so a pupil is never left with an empty list. Setting `goal_mode="soft"` turns this off, and the goal then counts only through its weight in the score. The [Evidence](evidence.md) page measures how many picks stay on the path in each mode.

**Exploration.** A little noise can be added so that Kwan sometimes tries a skill it knows less about:

```
score + ε · N(0,1) / sqrt(1 + n_k)
```

The noise is larger for skills with few answers (`n_k` small). `ε` is 0 in the library by default and 0.05 in the demo. The random numbers come from a small generator (Mulberry32) seeded with the CRC32 checksum of the text `learner_id:step`, so any recommendation can be reproduced exactly, on any machine and in any language a port is written in.

**Review slot.** When three or more skills are asked for, one place can go to review: a skill that was mastered before, was last answered at least 7 days ago, and whose mastery with forgetting applied has dropped below the mastery line. Among the skills that are due, the one that has faded most is chosen. If no skill fits, the place goes to the next candidate. The slot needs times: without `at` on the answers and `now` on the request, nothing is ever due. A review pick is marked `is_review` and its reason says so.

**Strand variety.** After each pick, other candidates from the same content standard lose 0.05, so the list is not three versions of the same thing. The penalty is small on purpose: it breaks near-ties, it does not push a weak foundation off the list.

Ties are broken by indicator code, so the order is always the same.

### The reason

Each recommendation comes with a reason in plain language, built only from the signals that produced the score. For example, it says which earlier skill the new one builds on and how the pupil is doing on it, how close the pupil is to mastering the new skill, how many steps remain to the goal, and how many later skills it opens.

The reason is returned twice: as an English sentence (`rationale`), and as a list of parts (`rationale_parts`), each with a key and its values. A sentence template such as

```
Builds on {indicator}, where you are at {pct}.
```

is kept in a language file, so the same parts can be shown in another language.

## 4. Practice settings

Each skill comes with a **practice setting**: one short line, shown as "Try it in Ghana", that places the skill in everyday Ghanaian life.

- **Written to a guide.** Real places in all 16 regions, names from many groups, real coins and notes, believable prices, 220 characters at most, no brand names and no stereotypes. See the [context style guide](context-style-guide.md).
- **Kwan-written.** Settings are never copied from NaCCA exemplars. They are shared under CC BY 4.0.
- **Checked twice.** Automatic checks cover length, reading level, the way money is written and banned words. Then a teacher reviews each setting. Only reviewed settings ship. See the [teacher review guide](teacher-review-guide.md).
- **Picked the same way every time.** If a skill has several settings, Kwan picks one from a checksum of the indicator code (CRC32), so the same skill always shows the same setting. If a skill has no setting of its own, Kwan falls back to a general setting for its strand and level.
- **Local languages.** For Basic 1 to Basic 3, settings can be shown in Twi, Ewe or Ga. Only translations approved by a fluent teacher are shown.

## What makes Kwan predictable

- **Deterministic.** The same bundle, the same answers and the same `now` always give the same output.
- **Portable.** Rounding is half-up and picks use CRC32, so a future version in another language can match Kwan exactly.
- **Offline.** No network call and no AI model call when Kwan recommends.
- **One file.** The map, the parameters, the settings and the text templates for a subject travel together in one bundle file, `kwan-bundle.math.json`, with a version number.
- **No personal data.** Learner state holds ability numbers, counts, streaks and times, keyed by an opaque id.

How well all of this works is on the [Evidence](evidence.md) page. The limits are on the [Model card](model-card.md).

---

Kwan uses the NaCCA Standards-Based Curriculum to organise skills. Kwan is not reviewed, approved or endorsed by NaCCA or the Ministry of Education.
