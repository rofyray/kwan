# Model card

What Kwan's model is for, how it works, how well it does, and where it falls short.

!!! note "Where the numbers come from"
    No number on this page is typed by hand. Each results block is filled from `data/eval_results.json`, which is written by the evaluation scripts in the Kwan repository, by `uv run scripts/fill_docs.py`. A block whose results are not in the file yet says so. The [Evidence](evidence.md) page explains how to read each table.

## Summary

| | |
|---|---|
| Name | Kwan learner model and recommender |
| Version | Package `kwan` 1.0.x; the bundle version is `math-` plus the package version (for example `math-1.0.1`) |
| Type | Multi-level Elo learner model with evidence passed along a prerequisite graph, a forgetting curve and an item-difficulty bank, feeding a rule-based recommender |
| Subject | NaCCA Mathematics, Basic 1 to Basic 6 |
| Input | A stream of answers: which skill, right or wrong, and optionally which question and when |
| Output | The chance that a pupil gets each skill right, a mastery estimate per skill, and a ranked list of next skills with reasons |
| Runs on | One CPU core, with no network call when it recommends |
| Licence | Code MIT. Kwan-written content CC BY 4.0. |
| Maintainer | Rofy Ray ([@rofyray](https://github.com/rofyray)) |

## Intended use

**Kwan is built to:**

- suggest which skill a pupil should practise next, and say why (in 1.0: primary Mathematics);
- help a teacher or a learning app see which earlier skills a pupil may have missed;
- give developers and researchers an open engine that follows the Ghanaian curriculum.

**Kwan is for** developers of learning tools, teachers and teacher trainers, and education researchers.

**Kwan is not built to:**

- grade pupils, rank them, place them in classes or decide who moves up;
- stand in for an examination or for a teacher's judgement;
- diagnose a learning difficulty;
- teach by itself. Kwan chooses the skill and the setting. It does not write lessons or mark answers;
- cover subjects or classes outside NaCCA Mathematics, Basic 1 to Basic 6.

A mastery estimate from Kwan is a best guess from the answers it has seen. Treat it as advice for the next practice session, never as a mark.

## The model

The [Method](method.md) page has the full formulas. In short:

**Learner model: multi-level Elo with graph transfer.** Kwan keeps one number for the pupil's overall ability and one number for each skill the pupil has tried. The chance of a correct answer is

```
p = σ(θ_pupil + θ_pupil,skill − β_item)
```

After each answer, both numbers move a little towards what the answer showed. The steps get smaller as more answers come in. For a skill the pupil has not tried yet, Kwan borrows evidence from the skills that come just before it in the curriculum map, and lowers the estimate a little for each class level, so that doing well in Basic 1 does not make Basic 6 look easy.

**Forgetting.** When answers come with dates, a skill's estimate drifts back towards its starting value as days pass without practice, at a rate set by the forgetting time τ. A skill mastered by a streak of correct answers also stops counting as mastered after a fixed number of days without practice. Both values are in the shipped parameters below. Without dates, the model behaves exactly as it did before forgetting was added.

**Item bank.** Each question can have its own difficulty, learned offline. With no information, a question has difficulty 0. A label of easy, medium or hard moves it by a fixed step. Shipped item banks come only from datasets that allow it.

**Mastery.** A skill counts as mastered when the predicted chance of a correct answer is 0.85 or higher, or after three correct answers in a row.

**Recommender.** Kwan looks at skills that are not yet mastered and whose prerequisites are in place. It scores them on how solid the foundations are, how close the skill is to a productive level of challenge, how close it is to the goal, how many later skills it opens, and how long since it was last practised. With a goal set, skills on the path to the goal come first. One slot can go to a skill that is due for review. No language model is called when Kwan recommends.

**The skill map.** Links come from curriculum order (rules) and from an offline decision model whose suggestions ship only after a teacher accepts them.

## Data

The model is tested on public research datasets. None of them is stored in the repository. See the [Data card](data-card.md) for the source, size, licence and role of each.

Two datasets (AMMORE and Junyi Academy) have non-commercial licences. Kwan uses them only to measure the model. The parameters shipped in a Kwan bundle are fitted on the other datasets, and the bundle builder enforces this.

## How it is evaluated

- **Predicting the next answer.** Students are split 80/20, by student, with a fixed seed. The model reads each test student's answers in order and predicts each one before seeing it. Reported: AUC, RMSE, accuracy and calibration error (ECE, 10 bins).
- **Baselines.** The average per skill, Bayesian Knowledge Tracing (BKT) and a classic single-level Elo.
- **Deep knowledge tracing.** DKT, SAKT, simpleKT and AKT from the pyKT toolkit, on 5 folds by student, compared with Kwan's model on the same folds.
- **Simulation.** Virtual pupils with hidden true skills follow different practice policies. This checks the recommender's logic. It is not evidence about real pupils.
- **Speed.** The time for one top-3 recommendation on one CPU core.

## Results

### Predicting the next answer

One row per dataset, with the AUC of each model on held-out students (Kwan in bold), then Kwan's other measures on the same runs.

<!-- RESULTS:START id=model-card-kt-table source=data/eval_results.json -->
| Dataset | Answers | Students | Skill average | BKT | Classic Elo | **Kwan** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AMMORE, all pupils (Ghana, Nigeria, Kenya, South Africa) | 50,507 | 2,388 | 0.6314 | not run | 0.6904 | **0.7740** |
| AMMORE, Ghanaian pupils only | 16,706 | 730 | 0.6388 | not run | 0.6902 | **0.7639** |
| ASSISTments 2009 | 259,399 | 4,163 | 0.6244 | 0.7001 | 0.6682 | **0.7545** |
| Eedi 2020, task 3/4 | 1,382,727 | 4,918 | 0.5762 | not run | 0.6842 | **0.7764** |
| Eedi 2020, task 1/2 | 15,867,850 | 118,971 | 0.5716 | not run | 0.6780 | **0.7946** |
| ASSISTments 2012-13 | 2,711,602 | 29,018 | 0.6093 | not run | 0.6541 | **0.7472** |
| ASSISTments 2015 | 683,801 | 19,840 | 0.6401 | not run | 0.6567 | **0.6835** |
| ASSISTments 2017 | 942,816 | 1,709 | 0.5856 | not run | 0.6278 | **0.7103** |
| XES3G5M | 5,549,184 | 18,066 | 0.6189 | not run | 0.6468 | **0.8008** |
| Junyi Academy 2015 (evaluation only) | 25,925,992 | 247,606 | 0.7188 | not run | 0.7542 | **0.7899** |

Kwan's model, the same runs, other measures:

| Dataset | RMSE | Accuracy | ECE |
| --- | ---: | ---: | ---: |
| AMMORE, all pupils (Ghana, Nigeria, Kenya, South Africa) | 0.4103 | 0.7513 | 0.0312 |
| AMMORE, Ghanaian pupils only | 0.4109 | 0.7577 | 0.0509 |
| ASSISTments 2009 | 0.4271 | 0.7300 | 0.0254 |
| Eedi 2020, task 3/4 | 0.4390 | 0.7080 | 0.0402 |
| Eedi 2020, task 1/2 | 0.4185 | 0.7375 | 0.0292 |
| ASSISTments 2012-13 | 0.4204 | 0.7411 | 0.0248 |
| ASSISTments 2015 | 0.4235 | 0.7382 | 0.0236 |
| ASSISTments 2017 | 0.4510 | 0.6931 | 0.0223 |
| XES3G5M | 0.3580 | 0.8221 | 0.0224 |
| Junyi Academy 2015 (evaluation only) | 0.3369 | 0.8497 | 0.0163 |
<!-- RESULTS:END -->

### Ghanaian pupils only

The same model on all AMMORE pupils and on the Ghanaian pupils alone.

<!-- RESULTS:START id=model-card-ghana source=data/eval_results.json -->
Kwan's AUC is 0.7740 on all AMMORE pupils and 0.7639 on the 730 Ghanaian pupils (16,706 answers).

| AMMORE | Pupils | Answers | Skills | Skill average | Classic Elo | **Kwan** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All pupils | 2,388 | 50,507 | 151 | 0.6314 | 0.6904 | **0.7740** |
| **Ghanaian pupils only** | 730 | 16,706 | 144 | 0.6388 | 0.6902 | **0.7639** |
<!-- RESULTS:END -->

### Forgetting

The forgetting time τ chosen on each dataset with timestamps, and the AUC with and without forgetting, followed by the held-out gain of the shipped parameters over the Round 1 defaults.

<!-- RESULTS:START id=model-card-forgetting source=data/eval_results.json -->
| Dataset | Has times | Best τ | AUC, v1 steps, no forgetting | AUC, with forgetting | Change | AUC, tuned steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ASSISTments 2009 | no | none | 0.7524 | 0.7524 | 0.0000 | 0.7537 |
| Eedi 2020, task 3/4 | yes | 30 days | 0.7763 | 0.7779 | 0.0016 | 0.7803 |
| Eedi 2020, task 1/2 | yes | 45 days | 0.7859 | 0.7869 | 0.0010 | 0.7908 |
| AMMORE (evaluation only) | yes | 14 days | 0.7708 | 0.7708 | 0.0001 | 0.7689 |
| ASSISTments 2015 | no | none | 0.6836 | 0.6836 | 0.0000 | 0.6872 |
| ASSISTments 2017 | yes | 45 days | 0.7097 | 0.7115 | 0.0018 | 0.7151 |
| XES3G5M | yes | 7 days | 0.7976 | 0.7985 | 0.0009 | 0.8041 |

Shipped in the maths bundle: `k_s` 0.50, `k_k` 0.30, `decay` 0.02, τ 30 days (step sizes chosen on assistments_2009, eedi_2020_task34, eedi_2020_task12; τ chosen on eedi_2020_task34, eedi_2020_task12).

Held-out students, v1 defaults against the shipped parameters:

| Dataset | AUC, v1 defaults | AUC, shipped | Gain |
| --- | ---: | ---: | ---: |
| ASSISTments 2009 | 0.7524 | 0.7528 | 0.0004 |
| Eedi 2020, task 3/4 | 0.7763 | 0.7803 | 0.0040 |
| Eedi 2020, task 1/2 | 0.7859 | 0.7905 | 0.0046 |
<!-- RESULTS:END -->

### Deep knowledge tracing, and the decision to keep Elo as the default

Kwan adopts a deep model only if it beats the Elo model by at least 0.03 AUC on at least two of AMMORE, Eedi and ASSISTments 2009, and stays under 20 ms per update on a CPU. This rule (decision D7 in the project's planning) was set before the benchmark was run.

**The decision, in words.** The benchmark was run with four pyKT models (DKT, SAKT, simpleKT and AKT) on the three datasets the rule names, with Kwan's Elo scored on the same folds and the same answers. No deep model cleared the margin on any of the three datasets, let alone on two, and the largest of the four was also over the latency limit. So Kwan keeps the multi-level Elo as its only learner model, and no deep-model backend is built into the package. The training budget was smaller than pyKT's reference runs, so a deep model with full training may come closer; the rule would then be applied again, with the same thresholds. The figures behind each sentence are in the block below.

<!-- RESULTS:START id=model-card-deep-kt source=data/eval_results.json -->
| Model | ASSISTments 2009 AUC | Eedi task 3/4 AUC | AMMORE AUC | XES3G5M AUC | ms per update (CPU) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Kwan (multi-level Elo)** | 0.7583 ± 0.0047 | 0.7795 ± 0.0035 | 0.7734 ± 0.0090 | 0.8008 ± 0.0009 | under 0.01 |
| DKT | 0.7496 ± 0.0088 | 0.7591 ± 0.0041 | 0.7245 ± 0.0122 | not run | 1.76 |
| SAKT | 0.7128 ± 0.0075 | 0.7390 ± 0.0047 | 0.6957 ± 0.0118 | not run | 0.77 |
| simpleKT | 0.7390 ± 0.0095 | 0.7884 ± 0.0040 | 0.7369 ± 0.0095 | not run | 1.36 |
| AKT | 0.7118 ± 0.0098 | 0.7299 ± 0.0048 | 0.7166 ± 0.0091 | not run | 21.87 |

**Decision: Elo stays the default learner model.** No deep model meets rule D7. The closest is simplekt: at least 0.03 AUC ahead of elo-multi (Kwan) on 0 of the 3 D7 datasets it was run on (needs 2); its best margin is +0.0089 on eedi_34. Elo stays the only learner model.

Rule: D7: a deep model is adopted only if it beats elo-multi by at least 0.03 AUC on at least two of AMMORE, Eedi and ASSISTments 2009, and stays under 20 ms per update on CPU.

Protocol: 5 folds by student, seed 42; each fold is held out once and no student is in two folds. 10% of each fold's training students (seed 42 + fold) are used only to pick the best epoch of the deep models; the held-out fold is never used for model selection

Training budget: Reduced budget: every fit was capped at far fewer epochs than pyKT's reference runs. Where a fit was stopped by the cap, the model's AUC is likely below what full training would reach. The cap, the epochs run and the count of folds stopped by the cap are under datasets.<name>.models.<model>. train_minutes_total adds up fits that ran side by side on one shared GPU.

Not run:

- `xes3g5m/dkt`, `xes3g5m/sakt`, `xes3g5m/simplekt`, `xes3g5m/akt`: Dropped for time: XES3G5M (5.5M answers) is four times the Eedi subset and the training budget went to the three D7 datasets. Only elo-multi was run on it.
- `onnx_backend`: not built: no deep model met rule D7, so Elo stays the only learner model
<!-- RESULTS:END -->

### Shipped parameters

The bundle carries one set of learner-model parameters. The step sizes and the forgetting time were chosen by grid search on the validation students of the commercially usable datasets, then checked against the Round 1 defaults on held-out students: a parameter that made things worse would have gone back to its old value. The two graph parameters keep their Round 1 values, because no public dataset has a prerequisite map to fit them on. AMMORE and Junyi informed nothing that ships.

<!-- RESULTS:START id=model-card-params source=data/eval_results.json -->
| Parameter | Value |
| --- | ---: |
| `k_s`, step size for the pupil's overall ability | 0.50 |
| `k_k`, step size for a skill offset | 0.30 |
| `decay`, how fast the steps shrink | 0.02 |
| `grade_step`, prior drop per class level | 0.40 |
| `transfer`, share borrowed from prerequisites | 0.50 |
| `streak_to_master`, right answers in a row that count as mastery | 3 |
| `tau_days`, the forgetting time constant (days) | 30 |
| `streak_expiry_days`, life of a streak mastery (days) | 30 |

Chosen on: assistments_2009, eedi_2020_task34, eedi_2020_task12. τ chosen on: eedi_2020_task34, eedi_2020_task12.

Rule: tau: the value with the best mean validation AUC at the v1 step sizes on eedi_2020_task34, eedi_2020_task12; k_s, k_k, decay: the triple with the best mean validation AUC across assistments_2009, eedi_2020_task34, eedi_2020_task12 at that tau; then never worse than the v1 defaults by more than 0.002 AUC on the held-out students of assistments_2009, eedi_2020_task34 (a parameter that breaks this goes back to its v1 value); grade_step and transfer keep the v1 values because the public datasets have no prerequisite graph; streak_to_master and streak_expiry_days are not fitted (PRD defaults).

| Held-out students | AUC, v1 defaults | AUC, shipped | Gain |
| --- | ---: | ---: | ---: |
| ASSISTments 2009 | 0.7524 | 0.7528 | 0.0004 |
| Eedi 2020, task 3/4 | 0.7763 | 0.7803 | 0.0040 |
| Eedi 2020, task 1/2 | 0.7859 | 0.7905 | 0.0046 |

- grade_step and transfer act only through a prerequisite graph; the public datasets have none, so they keep the v1 values
- the grid's banks are fitted on the inner fit students; the saved banks on all the training students
- the maths bundle ships without an item bank: Eedi and ASSISTments item ids do not map to Kwan items
<!-- RESULTS:END -->

### Simulation

New skills mastered, skills retained after a rest, and steps to a goal, for each simulated pupil profile and policy. Read with the caution in the Limitations below.

<!-- RESULTS:START id=model-card-simulation source=data/eval_results.json -->
30 simulated learners per profile and policy (seeds 0 to 29), 150 practice steps at 5 a day, then 30 days without practice; goal runs stop at 300 steps

New skills mastered after 150 practice steps (mean over 30 simulated pupils):

| Pupil profile | **Kwan** | Kwan + exploration | Kwan, top pick only | Syllabus order | Random | Remedial first |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Beginner in Basic 1 | **23.2** | 22.0 | 19.9 | 17.6 | 6.9 | 19.1 |
| Basic 4 pupil with gaps in B1 to B3 | **11.1** | 10.4 | 12.7 | 6.3 | 7.2 | 15.4 |
| Basic 6 pupil with gaps that grew | **9.4** | 8.4 | 8.6 | 5.6 | 8.1 | 12.9 |
| Basic 4 pupil, strong in Number, weak in Geometry | **10.1** | 9.9 | 8.6 | 8.0 | 7.6 | 5.0 |

Of those, skills still known after 30 days without practice:

| Pupil profile | **Kwan** | Kwan + exploration | Kwan, top pick only | Syllabus order | Random | Remedial first |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Beginner in Basic 1 | **17.7** | 17.4 | 14.1 | 10.4 | 4.1 | 11.6 |
| Basic 4 pupil with gaps in B1 to B3 | **8.2** | 7.5 | 9.1 | 4.4 | 3.6 | 9.5 |
| Basic 6 pupil with gaps that grew | **7.2** | 6.0 | 5.9 | 3.2 | 4.4 | 7.3 |
| Basic 4 pupil, strong in Number, weak in Geometry | **7.7** | 7.1 | 6.1 | 5.3 | 4.3 | 3.3 |

Steps to reach a goal skill, with the share of runs that reached it within 300 steps:

| Pupil profile | **Kwan** | Kwan + exploration | Kwan, top pick only | Syllabus order | Random | Remedial first |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Beginner in Basic 1 | 79.7 (100%) | 82.2 (100%) | 28.4 (100%) | 264.7 (27%) | 290.2 (3%) | 27.1 (100%) |
| Basic 4 pupil with gaps in B1 to B3 | 170.1 (83%) | 155.8 (90%) | 85.2 (97%) | 300.0 (0%) | 300.0 (0%) | 75.7 (100%) |
| Basic 6 pupil with gaps that grew | 197.4 (73%) | 201.2 (67%) | 101.0 (93%) | 300.0 (0%) | 299.2 (3%) | 84.3 (100%) |
| Basic 4 pupil, strong in Number, weak in Geometry | 194.6 (83%) | 194.9 (83%) | 87.4 (100%) | 300.0 (0%) | 296.1 (3%) | 102.9 (93%) |

Checks set before the run: kwan strict beats syllabus and random on gapped profiles: yes; kwan strict within 1 skill of syllabus on every profile: yes. All met: yes.

Profiles:

- `beginner_p1`: ASSUMPTION, not data. A pupil starting Basic 1: knows 15% of the B1 and B2 skills (from home and kindergarten) and 3% of everything later. Same as the v1.0 profile.
- `gapped_p4`: ASSUMPTION, not data. A Basic 4 pupil with foundational gaps: knows 75% of the B1 to B3 skills, so one lower-primary skill in four is missing, and 5% of B4 to B6. Same as the v1.0 profile.
- `gapped_p6`: ASSUMPTION, not data. A Basic 6 pupil with gaps that grew over the years: knows 80% of B1 to B3, 50% of B4 and B5, and 5% of B6.
- `uneven_strands`: ASSUMPTION, not data. A Basic 4 pupil who is strong in Number and weak in Geometry and Measurement: of the B1 to B4 skills, knows 85% in Number, 15% in Geometry and Measurement and 50% in Algebra and Data; knows 5% of B5 and B6 in every strand.

Policies:

- `kwan-strict`: Kwan with the library defaults (strict goal mode, no exploration, variety 0.05, review slot on). The pupil gets a list of 3 skills, practises each once in the order given, then gets a new list.
- `kwan-explore`: The same as kwan-strict with exploration switched on (eps 0.05, as in the demo).
- `kwan-top1`: Kwan used as in the v1.0 simulator, added as a sensitivity check: the pupil practises only the top pick and asks again after every answer. With a list of 1 there is no review slot.
- `syllabus`: Syllabus order from Basic 1: always the first skill, by class and then by code, that the learner model does not count as mastered. It does not use the prerequisite map.
- `random`: A skill drawn at random from the whole curriculum at every step.
- `remedial-first`: The pupil's class syllabus with remediation first: take the first skill of the pupil's class (then later classes, then earlier ones) that is not mastered; if any skill that leads to it is below 70%, practise the earliest such skill, otherwise practise the skill itself. It uses the prerequisite map but none of Kwan's scoring. With a goal, the goal is the skill it works towards.
<!-- RESULTS:END -->

### Speed

<!-- RESULTS:START id=model-card-latency source=data/eval_results.json -->
A top-3 recommendation takes 0.10 ms on average and 0.13 ms at the 95th percentile (single CPU core, pure Python; 3,000 timed calls per learner state, in-process, no network).
<!-- RESULTS:END -->

## Limitations

- **Most of the test data is not Ghanaian.** The large datasets come from the United States, the United Kingdom, China and Taiwan. They show that the model predicts answers well in general, not that it fits Ghanaian classrooms.
- **The African data is from older pupils.** AMMORE covers grades 6 to 9 on a WhatsApp tutor. Kwan's subject is Basic 1 to Basic 6. There are no public answer logs from Ghanaian primary classrooms yet.
- **Skills in the test data are not NaCCA skills.** Each dataset has its own skill tags. The evaluation tests the learner model. It does not test Kwan's NaCCA skill map directly.
- **The skill map is built by rule, so far.** Links from curriculum order are sensible but not perfect. The decision model that proposes cross-strand links is built and tested, but the local model tried first sits below the precision bar and its few proposals are still pending review, so no augmented link ships yet. A reviewer can be wrong too.
- **No practice setting has been reviewed yet.** Every setting and short label is a draft written for Kwan. By default Kwan shows the general setting for the skill's strand and class band until a teacher reviews the specific one. No Twi, Ewe or Ga translation ships yet.
- **Untried skills are estimated from nearby skills.** A pupil who does very well on early skills raises the estimate for later skills that have not been tried. In rare cases an untried skill can already count as mastered and is then not recommended.
- **Item difficulty is unknown for new questions.** A question with no history is treated as average.
- **The forgetting rate is one number for everyone.** Real pupils forget at different speeds, and some skills fade faster than others.
- **The simulation rests on assumptions.** The virtual pupils follow rules chosen by the authors. The simulation checks the policy's logic and should not be read as a measured learning gain.
- **Right or wrong only.** Kwan does not read working, partial credit, time taken or the kind of mistake.
- **A short history gives a rough estimate.** After a handful of answers the model knows little about a pupil. Its first suggestions lean on the curriculum order.

## Fairness

- **Kwan does not use personal attributes.** The model never sees gender, age, region, language, religion, school or family background. Two pupils with the same answers get the same recommendations.
- **Results are reported for Ghanaian pupils on their own**, not only as part of a larger average, so that a weaker fit for Ghana cannot hide behind other countries.
- **Who is missing from the data matters.** Pupils in AMMORE had access to a phone with WhatsApp. Pupils without that access are not represented. Kwan has no evidence yet about how well the model fits them.
- **Language.** The practice settings are in English, with Twi, Ewe and Ga for lower primary where a teacher has reviewed the translation. Pupils who learn in other Ghanaian languages are not yet served equally.
- **Practice settings are written to a style guide** that asks for places in all 16 regions, names from many groups, girls and boys equally, and no stereotypes. A teacher reviews every setting before it ships. See the [context style guide](context-style-guide.md).
- **What we do not know.** Kwan has not been tested for differences in accuracy between girls and boys, between regions, or between urban and rural pupils, because the public data does not carry that information.

## Children's data

Kwan is made for tools that children use, so it is built to hold as little as possible.

- **No personal data.** Kwan never stores a pupil's name, email address or any free text about a pupil.
- **Opaque ids.** Learner state is keyed by an id chosen by whoever uses Kwan. Choose an id that means nothing outside your own system. Do not use a name, a phone number or an index number.
- **What the state holds.** Ability numbers, attempt counts, streaks and the times of last practice for each skill. Nothing else.
- **Who holds it.** The state lives wherever the tool that uses Kwan puts it: in memory, in a local SQLite file or in that tool's own database. Kwan sends nothing to the Kwan project or to anyone else, and makes no network call when it recommends.
- **Delete.** Every learner-state store has a delete operation, and the REST API has a route to delete a learner.
- **Your duties.** If you build a tool on Kwan, you decide what else is collected. You are responsible for consent from parents or guardians and schools, and for following the law where you work, including Ghana's Data Protection Act, 2012 (Act 843).
- **The public demo** is for teachers, funders and contributors to try Kwan. It is not meant for collecting pupils' answers.

## NaCCA

Kwan uses the NaCCA Standards-Based Curriculum to organise skills. Kwan is not reviewed, approved or endorsed by NaCCA or the Ministry of Education.

## How to cite and where to ask

Cite Kwan with the `CITATION.cff` file in the [repository](https://github.com/rofyray/kwan). Open an issue there for questions, or to report a problem with the model.
