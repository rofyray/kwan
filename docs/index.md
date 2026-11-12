# Kwan

<img class="kwan-wordmark" src="assets/kwan-logo-light.png#only-light" alt="Kwan logo: a green path through gold steps to a red goal under a star">
<img class="kwan-wordmark" src="assets/kwan-logo-dark.png#only-dark" alt="Kwan logo: a green path through gold steps to a red goal under a star">

**Kwan** is Twi for *the path* (*ɔkwan*). It tells a pupil which maths skill to learn next, explains why, and sets the practice in everyday Ghanaian life.

Kwan is an open-source learning path engine for Ghanaian primary maths. It follows the NaCCA Standards-Based Curriculum for Mathematics, Basic 1 to Basic 6.

## Why Kwan exists

Most learning apps and AI tutors used across Africa were designed somewhere else. Their word problems feature snow days and prices in dollars, and their learning paths follow a foreign syllabus. A pupil in a village in the Volta Region has to decode the story before she can start on the maths.

Kwan starts from what Ghanaian pupils study: the curriculum their teachers plan from. Every recommendation lines up with the lesson on the blackboard and comes with a practice setting a Ghanaian child will recognise.

## What Kwan does

1. **Maps the curriculum.** Every learning indicator in NaCCA Mathematics for Basic 1 to Basic 6 is a skill on a map. Links show which skills come before which. Each link records where it came from.
2. **Tracks the pupil.** After each answer, Kwan updates its estimate of how well the pupil knows every skill. Evidence flows along the map, and skills fade a little when they are not practised.
3. **Picks the next skills.** Kwan looks for skills whose foundations are in place and that are neither too easy nor too hard. If the teacher sets a goal, Kwan follows the path to it. It also brings back old skills for review.
4. **Explains each pick** in plain language, built from the same signals that produced the ranking.
5. **Grounds it in Ghana.** Each skill comes with a "Try it in Ghana" setting, written to a style guide and reviewed by a teacher.

Kwan makes no network call and no AI model call when it recommends. It runs on one CPU core, which matters for schools with patchy internet.

## Who it is for

| You are | Start here |
|---|---|
| A **teacher or education specialist** who wants to check skill links, practice settings or translations | [Teacher review guide](teacher-review-guide.md) and [Context style guide](context-style-guide.md) |
| A **developer** who wants to use Kwan in a learning tool | [Quickstart](quickstart.md) |
| A **researcher or funder** who wants to know how Kwan works and how well | [Method](method.md), [Evidence](evidence.md), [Model card](model-card.md) and [Data card](data-card.md) |
| Someone who wants to **help build Kwan** | [CONTRIBUTING.md](https://github.com/rofyray/kwan/blob/main/CONTRIBUTING.md) |

## What is in Kwan 1.0

- **Curriculum:** NaCCA Mathematics, Basic 1 to Basic 6: every indicator, with strand and sub-strand names.
- **Map:** a prerequisite graph with the source of every link, extended by an offline decision model and checked by teachers.
- **Learner model:** multi-level Elo with forgetting and an item-difficulty bank, tested on public datasets that include answers from African pupils.
- **Recommender:** strict goal paths, gentle exploration, spaced review, variety across strands, and reasons built from parts so they can be translated.
- **Ghana context:** a teacher-reviewed practice setting for each indicator, with Twi, Ewe and Ga versions for lower primary as reviewers allow.
- **Package:** `kwan` for Python, with a command-line tool, a learner-state store and an optional REST API.
- **Demo:** a public demo for teachers, funders and contributors.

English Language, JHS Mathematics and other work come after 1.0. See the [Roadmap](roadmap.md).

## Numbers

Every number about Kwan's results comes from one file, `data/eval_results.json`, written by scripts in the repository. Nothing is typed by hand. See [Evidence](evidence.md).

## Privacy

Kwan never stores a pupil's name, email or any free text about a pupil. Learner state is keyed by an opaque id chosen by whoever uses Kwan, and every store can delete a learner.

## Licences

- **Code:** MIT.
- **Practice settings, short labels, translations and documentation written for Kwan:** CC BY 4.0.
- **Curriculum text** belongs to NaCCA. Kwan downloads it from the official PDFs when you build it. The repository stores the indicator codes and Kwan's own short labels.
- **Datasets** keep their own licences. See the [Data card](data-card.md). Parameters fitted on non-commercial datasets are never shipped in Kwan's bundles.

## NaCCA

Kwan uses the NaCCA Standards-Based Curriculum to organise skills. Kwan is not reviewed, approved or endorsed by NaCCA or the Ministry of Education. See [Governance](governance.md) for the permission status.
