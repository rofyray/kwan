# Evidence

What has been measured about Kwan, and how to read it.

!!! note "Where the numbers come from"
    No number on this page is typed by hand. Each results block is filled from `data/eval_results.json`, which is written by the evaluation scripts in the Kwan repository. Each script owns one part of that file, and `uv run scripts/fill_docs.py` renders the blocks on this page from it. A block whose results are not in the file says "not run yet". You can rebuild the file yourself with `make all`.

## How to read these results

- **AUC** tells how well the model separates answers that will be right from answers that will be wrong. 0.5 is a coin flip. 1.0 is perfect.
- **RMSE** and **ECE** tell how close the predicted chances are to what happened. Lower is better.
- **Accuracy** is the share of answers the model called correctly.
- Every result is on **students the model never saw** while its settings were chosen.
- Each block also gives the number of answers and students, so you can see how much data stands behind it.

## 1. Does the learner model predict the next answer?

The model reads a pupil's answers one by one and predicts each answer before seeing it. Kwan's model is compared with simpler ones: the average for each skill, Bayesian Knowledge Tracing (BKT) and a classic Elo.

**How to read the table.** One row per dataset. The first two rows are AMMORE, the only data from African pupils, with the Ghanaian pupils on their own. "Answers" and "Students" say how much data stands behind the row. The four right-hand columns are the AUC of each model on the held-out students; higher is better, and Kwan's column is in bold. "Not run" means that baseline was not run on that dataset (BKT is only run on ASSISTments 2009). The second table gives Kwan's RMSE, accuracy and ECE on the same runs; lower is better for RMSE and ECE. Junyi is evaluation only, because of its licence.

<!-- RESULTS:START id=evidence-kt-table source=data/eval_results.json -->
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

## 2. Does it work for Ghanaian pupils?

AMMORE holds answers from pupils in Ghana, Nigeria, Kenya and South Africa. Kwan reports the result for Ghanaian pupils on their own, next to the result for all pupils.

**How to read the table.** The two rows are the same model, evaluated once on every pupil and once on the Ghanaian pupils only. The Ghana row has fewer pupils and answers, so its AUC is a little less certain. What matters is that the two rows are close: a model that fits the other countries but not Ghana would show a clear drop in the second row.

<!-- RESULTS:START id=evidence-ghana source=data/eval_results.json -->
Kwan's AUC is 0.7740 on all AMMORE pupils and 0.7639 on the 730 Ghanaian pupils (16,706 answers).

| AMMORE | Pupils | Answers | Skills | Skill average | Classic Elo | **Kwan** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All pupils | 2,388 | 50,507 | 151 | 0.6314 | 0.6904 | **0.7740** |
| **Ghanaian pupils only** | 730 | 16,706 | 144 | 0.6388 | 0.6902 | **0.7639** |
<!-- RESULTS:END -->

These pupils are in grades 6 to 9, older than the primary pupils Kwan is built for. See the limits in the [Model card](model-card.md).

## 3. Does forgetting help?

Where answers have dates, Kwan lets a skill's estimate fade when it is not practised. This block compares the model with and without forgetting, and the Round 1 step sizes with the tuned ones.

**How to read the table.** One row per dataset. "Has times" says whether the dataset carries timestamps; without them, forgetting cannot act, and the two AUC columns must be equal (a check on the code). "Best τ" is the forgetting time chosen on that dataset's validation students. "Change" is the AUC gained from forgetting: small positive numbers are the honest result here. The last column is the AUC after also tuning the step sizes. Below the table, the values that ship in the bundle and the held-out gains they give over the Round 1 defaults. AMMORE informs the choice but never sets it, because its licence is non-commercial.

<!-- RESULTS:START id=evidence-forgetting source=data/eval_results.json -->
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

Check: forgetting never lowers AUC by more than 0.002 on datasets without timestamps. Largest drop seen: 0.0000. Passed: yes.

Protocol: outer 80/20 by student (seed 42); chosen on an inner 80/20 by student split (seed 43) of the training students.
<!-- RESULTS:END -->

## 4. Would a deep model do better?

Kwan's Elo model is small and fast. Deep knowledge tracing models (DKT, SAKT, simpleKT and AKT, from the pyKT toolkit) are larger. They were trained on the same data and compared on the same 5 folds.

The rule, set before the benchmark: a deep model is adopted only if it beats Elo by at least 0.03 AUC on at least two of AMMORE, Eedi and ASSISTments 2009, and stays under 20 ms per update on a CPU.

**How to read the table.** One row per model, Kwan first. Each AUC is the mean over the 5 folds, with the standard deviation across folds after the ± sign; two models whose ranges overlap are not clearly different. The last column is the CPU time for one update, measured in the most favourable way for the deep models (one prediction for one learner). The decision below the table follows from the rule mechanically. The training budget was reduced, so the deep models may do better with full training; the note says where each fit was stopped by the cap.

<!-- RESULTS:START id=evidence-deep-kt source=data/eval_results.json -->
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

## 5. Does following Kwan help a pupil learn more?

There are no classroom trials of Kwan yet. What exists is a simulation: virtual pupils with hidden true skills practise for a fixed number of steps under different policies (Kwan with a strict goal, Kwan with a little exploration, Kwan asked for one skill at a time, syllabus order, random practice, and remedial-first), then rest, and the simulator counts what they learned and what they kept.

**How to read the tables.** Rows are pupil profiles, columns are policies, and every cell is a mean over the simulated pupils. The first table counts new skills truly learned; the second counts how many of those survived the rest; the third gives the mean number of steps to reach a goal skill, with the share of runs that reached it in brackets (a run that never got there counts as the maximum). Higher is better in the first two tables, lower in the third. The checks that follow were written before the run. The profiles and policies are described in the simulator's own words, and the assumptions at the end are the rules the virtual pupils follow; none of them is fitted to Ghanaian pupils.

<!-- RESULTS:START id=evidence-simulation source=data/eval_results.json -->
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

Each simulated pupil has a hidden true state for every skill. One practice step on a skill the pupil does not know teaches it with probability 0.35 when every prerequisite is truly known and 0.03 otherwise. A known skill is answered wrongly 10% of the time and an unknown one is guessed right 20% of the time. The pupil does 5 steps a day. Each night every known skill can be forgotten: a skill with memory strength S days survives one day with probability exp(-1/S). S is 45 days for a skill just learned and 365 days for what the pupil knew at the start; practising a known skill again on a later day doubles S, up to 365 days. Practising a skill does not refresh its prerequisites. These are modelling assumptions, not measurements of Ghanaian pupils.

- `beginner_p1`: ASSUMPTION, not data. A pupil starting Basic 1: knows 15% of the B1 and B2 skills (from home and kindergarten) and 3% of everything later. Same as the v1.0 profile.
- `gapped_p4`: ASSUMPTION, not data. A Basic 4 pupil with foundational gaps: knows 75% of the B1 to B3 skills, so one lower-primary skill in four is missing, and 5% of B4 to B6. Same as the v1.0 profile.
- `gapped_p6`: ASSUMPTION, not data. A Basic 6 pupil with gaps that grew over the years: knows 80% of B1 to B3, 50% of B4 and B5, and 5% of B6.
- `uneven_strands`: ASSUMPTION, not data. A Basic 4 pupil who is strong in Number and weak in Geometry and Measurement: of the B1 to B4 skills, knows 85% in Number, 15% in Geometry and Measurement and 50% in Algebra and Data; knows 5% of B5 and B6 in every strand.
<!-- RESULTS:END -->

The simulation is a check on the recommender's logic. Its pupils follow rules chosen by the authors, so it cannot show what real pupils would gain. The Round 1 table (three policies, no forgetting) is kept in the README for comparison.

## 6. Does the goal path work?

When a teacher sets a goal skill, Kwan should keep its picks on the path to that goal.

**How to read it.** The share counts the picks that sit on the path to the goal (the goal itself or a skill that leads to it) across seeded pupils. It cannot reach 100%: when fewer skills on the path are ready than the list asks for, the rest of the list is filled from other ready skills. The soft figure is the Round 1 behaviour, kept as `goal_mode="soft"`.

<!-- RESULTS:START id=evidence-goal-path source=data/eval_results.json -->
With a goal set, 76.7% of the top-3 picks sit on the path to the goal in strict mode (the default) and 36.0% in soft mode (the Round 1 behaviour). Setup: 50 seeded pupils, each with up to 60 answers on B1 to B3 skills (85% right) and a goal drawn from B4 to B6; share of the top 3 recommendations that are on the path to the goal.
<!-- RESULTS:END -->

## 7. Is the curriculum data complete?

The parser's count of indicators is compared with a hand count from the PDFs, for every class and strand.

**How to read it.** "Expected" is the hand count kept in `data/curriculum/expected_counts.json`; the parse is complete when the two match. "Cut short" counts descriptions that end mid-sentence, a sign that the parser lost a wrapped line. Exemplars are NaCCA's worked examples, counted but not committed. "Misprinted codes corrected" counts the entries of the corrections file applied in this build. The class-by-strand table shows where the indicators sit. Short labels are Kwan's own names for skills; a label is "reviewed" only after a teacher's decision is imported.

<!-- RESULTS:START id=evidence-curriculum source=data/eval_results.json -->
Indicators parsed: 276 of 276 expected from the hand count (counts match: yes). Descriptions that look cut short: 1 (0.4%). Indicators with at least one exemplar: 276 (100%), 673 exemplars in all, 43 of them set in Ghana. Misprinted codes corrected: 13.

Short labels: 276 drafted, 0 reviewed by a teacher, 0 missing, 0 over 60 characters.

| Class | Number | Algebra | Geometry and Measurement | Data | All |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 | 16 | 1 | 5 | 2 | 24 |
| B2 | 16 | 2 | 9 | 2 | 29 |
| B3 | 24 | 1 | 13 | 2 | 40 |
| B4 | 34 | 16 | 15 | 6 | 71 |
| B5 | 34 | 12 | 14 | 7 | 67 |
| B6 | 25 | 6 | 5 | 9 | 45 |

Cut short: `B3.1.2.2.1`.

NaCCA text committed to the repository: no.
<!-- RESULTS:END -->

The full check is the [curriculum QA report](qa/curriculum-math.html), published with this site.

## 8. Are the skill links right?

Links suggested by the decision model are tested in two ways. First, the same method is run on Junyi Academy data, which comes with links drawn by experts. Then teachers review the links suggested for the NaCCA skills.

**How to read it.** The first lines describe the map that ships: its size, the links by type, and the structural checks. Then the decision model: on Junyi, "precision" is the share of proposed links that are expert links, and "recall" the share of expert links that were proposed; the bar is on precision, because a wrong link sends a pupil the wrong way while a missing one only costs a suggestion. The NaCCA lines count what the model proposed and what teachers have decided so far. Only accepted links ship, and the count of accepted links says how many did.

<!-- RESULTS:START id=evidence-graph source=data/eval_results.json -->
Skill map: 276 skills and 295 prerequisite links (54 bridge, 159 sequence, 82 vertical). No cycles: yes. Every skill reachable: yes. Longest chain: 36 links.

| Strand | Skills |
| --- | ---: |
| Number | 149 |
| Algebra | 38 |
| Geometry and Measurement | 61 |
| Data | 28 |

Decision model: local-encoder (all-MiniLM-L6-v2+logreg@50721cd7). Precision bar: 80%. On the Junyi benchmark (held-out exercises) it proposed 13 links, 4 of them expert links: precision 30.8%, recall 1.6%. Meets the bar: no.

On NaCCA Mathematics it scored 5,403 candidate pairs and proposed 4 links. Teacher review: 4 pending, 0 accepted, 0 rejected. Accepted links shipped in the graph: 0.

Teacher acceptance: not yet measured: no teacher has reviewed an edge.

Counting a source that is an earlier prerequisite of the target as correct, precision is 53.8%; ignoring direction, 46.2%.

Proposed links by strand pair: Geometry and Measurement -> Geometry and Measurement: 4.

Cost of the NaCCA run: 0.00 USD.
<!-- RESULTS:END -->

The [skill map report](qa/graph-math.md) shows that the map has no cycles, that every skill can be reached, and how the links spread across classes.

## 9. Does every skill have a reviewed practice setting?

**How to read it.** Settings are drafted for Kwan, checked automatically, then reviewed by a teacher. The coverage figure counts indicators with at least one reviewed English setting, which is what ships by default. The translation table counts, for each language, the Basic 1 to 3 indicators, the machine drafts waiting for review, the reviewed translations, and the indicators whose translation ships.

<!-- RESULTS:START id=evidence-contexts source=data/eval_results.json -->
Practice settings: 828 drafted for 276 of 276 indicators; 828 pass the automatic checks. Reviewed by a teacher: 0 settings, covering 0 indicators (0.0% coverage). Reviewers so far: 0.

Translations for Basic 1 to Basic 3:

| Language | B1 to B3 indicators | Drafts | Reviewed | Indicators shipping |
| --- | ---: | ---: | ---: | ---: |
| Twi | 93 | 0 | 0 | 0 |
| Ewe | 93 | 0 | 0 | 0 |
| Ga | 93 | 0 | 0 | 0 |

Short labels: 276, 0 reviewed. Reviewed means a teacher's decision was imported from a review sheet. Drafts are not shown by default.
<!-- RESULTS:END -->

## 10. Is it fast enough?

Kwan is meant to run on modest hardware, with no network.

**How to read it.** The time is for one call that returns the top three skills, measured in-process on one CPU core, over many calls on three seeded learner states. The mean is the usual case; the 95th percentile is what the slowest one call in twenty takes.

<!-- RESULTS:START id=evidence-latency source=data/eval_results.json -->
A top-3 recommendation takes 0.10 ms on average and 0.13 ms at the 95th percentile (single CPU core, pure Python; 3,000 timed calls per learner state, in-process, no network).
<!-- RESULTS:END -->

## What this evidence does not show

- It does not show that pupils who use Kwan learn more. That needs a classroom study.
- It does not show how the model does for Ghanaian pupils in Basic 1 to Basic 6. No public data of that kind exists yet.
- It does not test the NaCCA skill map against real pupils' answers.

The [Model card](model-card.md) lists the limits in full. The [Data card](data-card.md) describes every dataset.

## Reproduce it

```bash
uv sync --extra eval
make data     # downloads the public datasets
make all      # rebuilds every committed result
```

The slow evaluations on full datasets also run every week in CI, and a pull request is opened when `data/eval_results.json` changes. After the file changes, `uv run scripts/fill_docs.py` refreshes this page, the model card and the README; `uv run scripts/fill_docs.py --check` is the test that keeps them in step.
