PY ?= uv run python
RESULTS := data/eval_results.json

.PHONY: app-gradio install data curriculum graph eval sim bench fit bundle test lint typecheck api app demo-docker docs all fill-docs
install:     ; uv sync --extra api --extra demo --extra eval --extra curriculum --extra gradio
data:        ; bash scripts/download_data.sh
curriculum:  ; $(PY) -m kwan.cli curriculum fetch && $(PY) -m kwan.cli curriculum parse && $(PY) -m kwan.cli curriculum qa
graph:       ; $(PY) -m kwan.cli graph build && $(PY) -m kwan.cli graph validate && $(PY) -m kwan.cli graph export --out build/graph/graph.html
eval:        ; $(PY) -m kwan.cli eval all --out $(RESULTS)
sim:         ; $(PY) -m kwan.cli sim --v2 --out $(RESULTS)
bench:       ; $(PY) -m kwan.cli bench --out $(RESULTS)
fit:         ; $(PY) -m kwan.cli fit all && $(PY) -m kwan.cli fit report --out $(RESULTS)
bundle:      ; $(PY) -m kwan.cli bundle --out src/kwan/bundles/
test:        ; $(PY) -m pytest -q
lint:        ; $(PY) -m ruff check . && $(PY) -m ruff format --check .
typecheck:   ; $(PY) -m mypy
api:         ; $(PY) -m uvicorn kwan.api.app:app --reload --port 8000
app:         ; $(PY) -m streamlit run demo/app.py
app-gradio:  ; $(PY) demo/gradio_app.py
demo-docker: ; docker build -f demo/Dockerfile -t kwan-demo .
docs:        ; uv run --group docs mkdocs build --strict
fill-docs:   ; $(PY) scripts/fill_docs.py
all: data curriculum graph eval fit sim bench fill-docs bundle test
