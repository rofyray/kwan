---
title: Kwan
emoji: 🧭
colorFrom: yellow
colorTo: green
sdk: streamlit
sdk_version: 1.67.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
---

# Kwan 🧭

**Kwan** is Twi for *the path*. It reads Ghana's Basic 1 to Basic 6 maths curriculum, keeps track of what a pupil has mastered, and picks the next skill to learn, with a reason and an example from everyday Ghanaian life.

This Space is the live demo. It has three tabs:

- **Pupil view.** Three skills to learn next, each with a "Mastery so far" bar, the reason Kwan picked it, and a practice setting from everyday Ghanaian life. Mark an answer right or wrong and the path changes straight away. You can load a P4 pupil with gaps in earlier classes, cap the recommendations at a class level, or set a goal skill such as next term's topic.
- **Curriculum map.** How the NaCCA learning indicators are linked into a map of prerequisite skills.
- **Evidence.** How well the learner model predicts a pupil's next answer on two public datasets of real student answers, and a simulation you can run that compares following Kwan with following the syllabus order and with random practice.

Everything runs on one CPU core, with no network calls and no language model. The numbers on the Evidence tab are read from `data/eval_results.json`, which is written by the evaluation scripts in the source repository, or come from running the simulation in front of you.

## Data and credits

- NaCCA Standards-Based Mathematics Curriculum, Basic 1 to 3 and Basic 4 to 6 (Ministry of Education, Ghana).
- ASSISTments 2009 skill-builder data: Feng, Heffernan and Koedinger (2009).
- Eedi NeurIPS 2020 Education Challenge: Wang et al. (2020), arXiv:2007.12061.

Built for the DataNova 2026 Global Data-thon (problem DN-ED-01, Adaptive Learning Path Recommendation). MIT License.
