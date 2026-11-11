# Kwan demo image for a Hugging Face Docker Space (PRD Appendix A.3). The same file builds locally:
#
#   docker build -f demo/Dockerfile -t kwan-demo .
#   docker run --rm -p 7860:7860 kwan-demo
#
# It accepts two build contexts. Locally and in CI the context is the repository root, so the package is
# built from source. On the Space the context is the folder that space-deploy.yml uploads: app.py,
# README.md, requirements-demo.txt, data/eval_results.json and wheels/kwan-<version>-py3-none-any.whl.
# Stage 1 copies the wheel it finds or builds one; stage 2 installs it with the demo extra and nothing else,
# so the final image carries no source tree, no raw data and no build tools.

FROM python:3.12-slim AS build
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
COPY . .
RUN set -eu; \
    mkdir -p /out/app/data /out/wheels; \
    if [ -d demo ]; then app=demo; else app=.; fi; \
    cp "$app/app.py" "$app/requirements-demo.txt" /out/app/; \
    if [ -f "$app/data/eval_results.json" ]; then cp "$app/data/eval_results.json" /out/app/data/; \
    elif [ -f data/eval_results.json ]; then cp data/eval_results.json /out/app/data/; fi; \
    if ls "$app"/wheels/kwan-*.whl >/dev/null 2>&1; then cp "$app"/wheels/kwan-*.whl /out/wheels/; \
    elif [ -f pyproject.toml ]; then pip wheel --no-deps --wheel-dir /out/wheels .; \
    else echo "no wheel under $app/wheels and no pyproject.toml in the build context" >&2; exit 1; fi

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false STREAMLIT_SERVER_HEADLESS=true
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
# the pinned Streamlit stack first, so that layer is reused when only the kwan wheel changes
COPY --from=build /out/app/requirements-demo.txt .
RUN pip install -r requirements-demo.txt
COPY --from=build /out/wheels /tmp/wheels
RUN pip install "$(ls /tmp/wheels/kwan-*.whl)[demo]" && rm -rf /tmp/wheels
COPY --from=build --chown=user /out/app .
USER user
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/_stcore/health', timeout=4)"
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
