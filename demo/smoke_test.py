"""Smoke test for the Kwan demo (PRD 10.5): open the app in a real browser and check that it works.

    python demo/smoke_test.py http://localhost:7860
    python demo/smoke_test.py https://rofyray-kwan.hf.space --wait 600

Checks, in order: the Streamlit health endpoint answers, the page renders all three tabs, each tab shows
its content (the curriculum map as an embedded graph page), one click on "Got it right" changes the cards,
and the NaCCA disclaimer is in the footer. Any failure prints "FAIL: ..." and exits with status 1.

It needs Playwright with Chromium (`pip install playwright && playwright install chromium`). With
`--health-only` it checks the health endpoint alone and needs nothing but the standard library.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.request

TABS = ["Pupil view", "Curriculum map", "Evidence"]
RIGHT = re.compile(r"Got it right")
DISCLAIMER = "not reviewed, approved or endorsed by NaCCA"
PANEL = '[data-testid="stTabs"] [role="tabpanel"]:not([hidden])'
PROGRESS = f'{PANEL} [data-testid="stProgress"]'
# a card is the nearest vertical block around a mastery bar; its text holds the skill line and the bar's label
CARDS_JS = (
    "() => Array.from(document.querySelectorAll('" + PROGRESS + "'))"
    ".map(p => (p.closest('[data-testid=\"stVerticalBlock\"]') || p).innerText.trim())"
)
RERUN_TIMEOUT = 30.0  # seconds for a click to change the page


class Failure(Exception):
    pass


def health(url: str, wait: float) -> None:
    """Poll `<url>/_stcore/health` until it answers 200 "ok", for up to `wait` seconds."""
    target = url.rstrip("/") + "/_stcore/health"
    deadline = time.monotonic() + wait
    last = "no answer yet"
    while True:
        try:
            with urllib.request.urlopen(target, timeout=10) as resp:
                body = resp.read(64).decode("utf-8", "replace").strip()
                if resp.status == 200 and body == "ok":
                    print(f"ok: health endpoint answers at {target}")
                    return
                last = f"status {resp.status}, body {body!r}"
        except (urllib.error.URLError, OSError, ValueError) as e:
            last = str(e)
        if time.monotonic() >= deadline:
            raise Failure(f"health endpoint {target}: {last}")
        time.sleep(5)


def visible_panel(page):
    """The tab panel that is shown now."""
    return page.locator(PANEL).first


def open_tab(page, name: str):
    page.get_by_role("tab", name=name).click()
    panel = visible_panel(page)
    panel.wait_for(state="visible", timeout=20_000)
    return panel


def cards_snapshot(page) -> list[str]:
    """The text of every card on the shown tab (the block around each mastery bar), in page order."""
    return page.evaluate(CARDS_JS)


def browser_checks(url: str, screenshot: str | None) -> None:
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeout
        from playwright.sync_api import sync_playwright
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise Failure(f"Playwright is not installed ({e}); run with --health-only or install it") from e

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            page.locator('[data-testid="stAppViewContainer"]').wait_for(timeout=60_000)
            tabs = page.get_by_role("tab")
            tabs.first.wait_for(timeout=60_000)
            names = [t.strip() for t in tabs.all_inner_texts()]
            if names != TABS:
                raise Failure(f"tabs are {names}, expected {TABS}")
            print(f"ok: page renders the tabs {names}")

            pupil = open_tab(page, "Pupil view")
            pupil.get_by_text("What to learn next").wait_for(timeout=30_000)
            page.locator(PROGRESS).first.wait_for(timeout=30_000)
            before = cards_snapshot(page)
            if len(before) != 3:
                raise Failure(f"expected 3 cards with a mastery bar, found {len(before)}")
            print("ok: Pupil view shows 3 cards")

            graph = open_tab(page, "Curriculum map")
            frame = graph.locator("iframe").first
            frame.wait_for(state="visible", timeout=30_000)
            page.frame_locator(f"{PANEL} iframe").locator("#mynetwork").wait_for(timeout=30_000)
            print("ok: Curriculum map embeds the interactive graph")

            evidence = open_tab(page, "Evidence")
            evidence.get_by_text("AUC", exact=False).first.wait_for(timeout=30_000)
            evidence.locator('[data-testid="stDataFrame"]').first.wait_for(timeout=30_000)
            print("ok: Evidence tab shows its tables")

            pupil = open_tab(page, "Pupil view")
            pupil.get_by_role("button", name=RIGHT).first.click()
            deadline = time.monotonic() + RERUN_TIMEOUT
            after = cards_snapshot(page)
            while after == before or len(after) != 3:
                if time.monotonic() > deadline:
                    raise Failure("the cards did not change after one right answer")
                time.sleep(0.5)
                after = cards_snapshot(page)
            print("ok: one right answer changes the cards")

            page.get_by_text(DISCLAIMER, exact=False).first.wait_for(timeout=20_000)
            print("ok: NaCCA disclaimer is on the page")
        except PlaywrightTimeout as e:
            if screenshot:
                page.screenshot(path=screenshot, full_page=True)
            raise Failure(f"timed out waiting for the page: {str(e).splitlines()[0]}") from e
        except Failure:
            if screenshot:
                page.screenshot(path=screenshot, full_page=True)
            raise
        finally:
            browser.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Smoke test for the Kwan demo.")
    ap.add_argument("url", help="where the demo runs, for example http://localhost:7860")
    ap.add_argument("--wait", type=float, default=60.0, help="seconds to wait for the health endpoint (default 60)")
    ap.add_argument("--health-only", action="store_true", help="check the health endpoint and stop")
    ap.add_argument("--screenshot", help="write a full-page screenshot here when a browser check fails")
    args = ap.parse_args(argv)
    try:
        health(args.url, args.wait)
        if not args.health_only:
            browser_checks(args.url, args.screenshot)
    except Failure as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
