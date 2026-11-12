"""Smoke test for the Gradio demo (the Hugging Face Space): open it in a real browser and check that it works.

    uv run --with playwright demo/gradio_smoke_test.py http://127.0.0.1:7860
    uv run --with playwright demo/gradio_smoke_test.py https://rofyray-kwan.hf.space --wait 600

Checks, in order: `<url>/config` answers with the Kwan title, the page shows all three tabs, the pupil view
shows recommendation cards, one click on "Got it right" changes them, the curriculum map draws its graph,
the evidence tab shows the AUC table, and the NaCCA disclaimer is on the page. Any failure prints
"FAIL: ..." and exits with status 1. With `--health-only` it checks `/config` alone and needs nothing but
the standard library. Playwright needs Chromium once: `uv run --with playwright playwright install chromium`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

TABS = ["Pupil view", "Curriculum map", "Evidence"]
DISCLAIMER = "not reviewed, approved or endorsed by NaCCA"
CARD_TITLES_JS = "() => Array.from(document.querySelectorAll('h3')).map(h => h.innerText.trim())"
WAIT_MS = 30_000


class Failure(Exception):
    pass


def health(url: str, wait: float) -> None:
    """Poll `<url>/config` until it answers 200 with the Kwan title, for up to `wait` seconds."""
    target = url.rstrip("/") + "/config"
    deadline = time.monotonic() + wait
    last = "no answer yet"
    while True:
        try:
            with urllib.request.urlopen(target, timeout=15) as resp:
                config = json.loads(resp.read().decode("utf-8", "replace"))
                if resp.status == 200 and "Kwan" in str(config.get("title", "")):
                    print(f"ok: {target} answers with the Kwan title")
                    return
                last = f"status {resp.status}, title {config.get('title')!r}"
        except (urllib.error.URLError, OSError, ValueError) as e:
            last = str(e)
        if time.monotonic() >= deadline:
            raise Failure(f"health endpoint {target}: {last}")
        time.sleep(5)


def browser_checks(url: str, screenshot: str | None) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise Failure("playwright is not installed: run with `uv run --with playwright`") from e
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=120_000)
            for name in TABS:
                page.get_by_role("tab", name=name).first.wait_for(timeout=WAIT_MS)
            print("ok: three tabs")
            right = page.get_by_role("button", name="Got it right").first
            right.wait_for(timeout=WAIT_MS)
            before = page.evaluate(CARD_TITLES_JS)
            if not before:
                raise Failure("the pupil view shows no recommendation card")
            print(f"ok: {len(before)} cards, first {before[0]!r}")
            right.click()
            deadline = time.monotonic() + WAIT_MS / 1000
            while page.evaluate(CARD_TITLES_JS) == before or "Mastery so far" not in page.inner_text("body"):
                if time.monotonic() > deadline:
                    raise Failure("the cards did not change after one right answer")
                time.sleep(0.5)
            print("ok: one right answer changes the cards")
            page.get_by_role("tab", name="Curriculum map").first.click()
            frame = page.frame_locator("iframe[title^='Interactive map']")
            frame.locator("canvas").first.wait_for(timeout=WAIT_MS)
            print("ok: the curriculum map draws")
            page.get_by_role("tab", name="Evidence").first.click()
            page.get_by_text("AMMORE 2024 (Ghana)").first.wait_for(timeout=WAIT_MS)
            print("ok: the evidence tab shows the AUC table")
            if DISCLAIMER not in page.inner_text("body"):
                raise Failure("the NaCCA disclaimer is missing")
            print("ok: disclaimer")
        except Failure:
            if screenshot:
                page.screenshot(path=screenshot, full_page=True)
            raise
        except Exception as e:  # a Playwright timeout or a page error, reported the same way
            if screenshot:
                page.screenshot(path=screenshot, full_page=True)
            raise Failure(f"{type(e).__name__}: {e}") from e
        finally:
            browser.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("url", help="where the demo runs, e.g. http://127.0.0.1:7860")
    ap.add_argument("--wait", type=float, default=60, help="seconds to wait for the app to answer (default 60)")
    ap.add_argument("--health-only", action="store_true", help="check /config only")
    ap.add_argument("--screenshot", metavar="PNG", help="save a full-page screenshot here when a check fails")
    args = ap.parse_args(argv)
    try:
        health(args.url, args.wait)
        if not args.health_only:
            browser_checks(args.url, args.screenshot)
    except Failure as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
