---
title: Kwan
emoji: 🧭
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Next maths skill for Ghanaian pupils, with a reason.
---

# Kwan demo

Kwan is Twi for "the path". It reads Ghana's Basic 1 to Basic 6 mathematics curriculum (NaCCA), keeps track of
what a pupil has mastered, and picks the next skill to learn, with a reason and a practice setting from
everyday Ghanaian life.

The demo has three tabs:

- **Pupil view:** mark answers right or wrong and watch the next three skills change. Pick a goal skill by
  its name, mix in other skills, and load a Basic 4 pupil with gaps in earlier classes.
- **Curriculum map:** every indicator as a node, linked to the skills that come before it.
- **Evidence:** how well the learner model predicts real pupils' answers on public datasets, and what the
  simulator says about following Kwan.

Nothing a visitor does here is stored. The pupil lives in the browser session and is gone when the tab closes.

This Space is deployed from https://github.com/rofyray/kwan by the `space-deploy` workflow: `app.py` is
`demo/gradio_app.py` and `kwan/` is the package unpacked from the release wheel. Do not edit files here; change
them in the repository.

Kwan uses the NaCCA Standards-Based Curriculum to organise skills. Kwan is not reviewed, approved or endorsed by NaCCA or the Ministry of Education.

Code is MIT licensed; the practice settings and labels written for Kwan are CC BY 4.0.
