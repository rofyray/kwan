PY ?= python
RESULTS := data/eval_results.json

.PHONY: data curriculum graph eval sim bench test api app space all
data:        ; bash scripts/download_data.sh
curriculum:  ; $(PY) parse_curriculum.py raw/m13.txt raw/m46.txt data/indicators.json
graph:       ; $(PY) graph.py data/indicators.json data/graph.json
eval:        ; $(PY) eval_kt.py raw/skill_builder_data_corrected.csv --bkt --out $(RESULTS) && $(PY) eval_eedi.py raw/eedi/data --out $(RESULTS)
sim:         ; $(PY) simulate.py data/graph.json --out $(RESULTS)
bench:       ; $(PY) bench_latency.py --out $(RESULTS)
test:        ; $(PY) -m pytest -q
api:         ; uvicorn api:app --reload --port 8000
app:         ; streamlit run app.py
space:       ; bash scripts/build_space.sh
all: data curriculum graph eval sim bench test
