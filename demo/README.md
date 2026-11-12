---
title: Kwan
emoji: ⭐
colorFrom: yellow
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Next maths skill for Ghanaian pupils, with a reason.
---

# Kwan demo

Live demo: https://huggingface.co/spaces/rofyray/kwan

Kwan is Twi for "the path". It reads Ghana's Basic 1 to Basic 6 mathematics curriculum (NaCCA), keeps track of
what a pupil has mastered, and picks the next skill to learn, with a reason and a practice setting from
everyday Ghanaian life.

The demo has three tabs:

- **Pupil view:** mark answers right or wrong and watch the next three skills change. Set a goal skill, mix in
  other skills, and load a Basic 4 pupil with gaps in earlier classes.
- **Curriculum map:** every indicator as a node, linked to the skills that come before it.
- **Evidence:** how well the learner model predicts real pupils' answers on public datasets, and what the
  simulator says about following Kwan.

Nothing a visitor does here is stored. The pupil lives in the browser session and is gone when the tab closes.

## Run it yourself

```bash
uv run --with "kwan[demo]" streamlit run app.py
```

The public Space runs the Gradio version of this demo (`demo/gradio_app.py`, folder files in `demo/space/`),
because Docker Spaces are a paid option. This Streamlit version stays for Docker hosting:

```bash
docker build -t kwan-demo .
docker run --rm -p 7860:7860 kwan-demo
```

Source, documentation and the Python package: https://github.com/rofyray/kwan

Kwan uses the NaCCA Standards-Based Curriculum to organise skills. Kwan is not reviewed, approved or endorsed by NaCCA or the Ministry of Education.

Code is MIT licensed; the practice settings and labels written for Kwan are CC BY 4.0.
