"""Browser regression test for the front-end (the layer the audit found broken).

Runs the app in a real Chromium engine with the mock backends
(``MDA_USE_REAL_BACKENDS=0``) so it is deterministic and needs no network, then
asserts the observable outcomes the audit named:

  * zero console errors and zero uncaught page errors after selecting a market;
  * all five chart canvases contain drawn (non-blank) pixels — this is the
    original "every chart renders blank" bug (Chart.js threw on load because the
    animation default object was replaced);
  * the "Pick a market" empty state is not visible once a market is chosen;
  * the payoff chart carries the break-even / current-price / max-gain / max-loss
    guides *and* draws them (guide-coloured pixels are present on the canvas);
  * each chart's x-axis shows several labels (e.g. 0%, 20% … 100%), not just 0%;
  * a Greek value that is small but non-zero is not displayed as "0.00";
  * the Delta caption states a correct unit (not the wrong "+1%");
  * at 894px width the market list shows several rows;
  * rapid market clicks never leave the page showing an older market's data;
  * the page makes zero requests to any external host (fonts / Chart.js are
    vendored locally).

The test is skipped only if the Chromium browser binary is unavailable, so a
developer without it installed is not blocked; CI installs it and runs it for
real. Chromium's first launch on Windows is slow — that is not a hang.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import pytest

playwright_sync = pytest.importorskip("playwright.sync_api")
from playwright.sync_api import Error as PWError  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
_PORT = 8101
_BASE = f"http://127.0.0.1:{_PORT}"


def _free_port(preferred: int) -> int:
    """Use the preferred port if free, else let the OS pick one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
                s2.bind(("127.0.0.1", 0))
                return s2.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    """Serve the app with the mock backends (offline, deterministic)."""
    port = _free_port(_PORT)
    base = f"http://127.0.0.1:{port}"
    env = dict(os.environ)
    env["MDA_USE_REAL_BACKENDS"] = "0"
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=str(_ROOT), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                with urlopen(base + "/api/health", timeout=1) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(0.25)
        else:
            raise RuntimeError("app server did not start")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as pw:
        try:
            b = pw.chromium.launch()
        except PWError as exc:
            pytest.skip(f"Chromium not available: {exc}")
        try:
            yield b
        finally:
            b.close()


def _collect_page(context):
    """Wire console/page-error/request listeners onto a fresh page."""
    page = context.new_page()
    console_errors: list[str] = []
    page_errors: list[str] = []
    external: list[str] = []
    page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: page_errors.append(str(e)))

    def _on_request(req):
        host = req.url.split("/")[2] if "//" in req.url else ""
        # Anything not served by our own origin (or a data: URL) is external.
        if req.url.startswith("data:"):
            return
        if not req.url.startswith(_SELF[0]):
            external.append(req.url)

    page.on("request", _on_request)
    return page, console_errors, page_errors, external


# filled per-test with the live base url (module fixture gives a dynamic port)
_SELF = [""]


def _select_first_market(page):
    page.wait_for_selector(".market-item", timeout=15000)
    page.locator(".market-item").first.click()
    page.wait_for_selector("#positionBody:not([hidden])", timeout=15000)
    # all five canvases exist and Chart.js has built instances for them
    page.wait_for_function(
        "window.__charts && Object.keys(window.__charts).length >= 5",
        timeout=15000,
    )


def _canvas_is_drawn(page, canvas_id: str) -> bool:
    """True if the canvas has any non-transparent pixel (i.e. it drew)."""
    return page.evaluate(
        """(id) => {
            const c = document.getElementById(id);
            if (!c || !c.width || !c.height) return false;
            const ctx = c.getContext('2d');
            const { data } = ctx.getImageData(0, 0, c.width, c.height);
            for (let i = 3; i < data.length; i += 4) {
                if (data[i] !== 0) return true;   // alpha channel non-zero
            }
            return false;
        }""",
        canvas_id,
    )


def _count_pixels_near(page, canvas_id: str, rgb: tuple[int, int, int], tol: int = 24) -> int:
    """Count canvas pixels close to an RGB colour (used to prove guide lines)."""
    return page.evaluate(
        """([id, r, g, b, tol]) => {
            const c = document.getElementById(id);
            const ctx = c.getContext('2d');
            const { data } = ctx.getImageData(0, 0, c.width, c.height);
            let n = 0;
            for (let i = 0; i < data.length; i += 4) {
                if (data[i+3] === 0) continue;
                if (Math.abs(data[i]-r) <= tol &&
                    Math.abs(data[i+1]-g) <= tol &&
                    Math.abs(data[i+2]-b) <= tol) n++;
            }
            return n;
        }""",
        [canvas_id, rgb[0], rgb[1], rgb[2], tol],
    )


_CANVASES = ["payoffChart", "deltaChart", "gammaChart", "vegaChart", "thetaChart"]


def test_full_ui_regression(server, browser):
    _SELF[0] = server
    context = browser.new_context(viewport={"width": 1280, "height": 900})
    page, console_errors, page_errors, external = _collect_page(context)
    try:
        page.goto(server + "/", wait_until="networkidle")
        _select_first_market(page)
        # let the (short) draw animation settle
        page.wait_for_timeout(600)

        # --- item 1: every chart drew non-blank pixels ---
        for cid in _CANVASES:
            assert _canvas_is_drawn(page, cid), f"canvas {cid} rendered blank"

        # --- item 1 (root cause): configureCharts() must mutate Chart.js's own
        # animation config in place, not replace it with a bare POJO. The
        # original bug did `Chart.defaults.animation = {duration: 200}`, which
        # drops the native fields (easing / fn / type / ...) Chart.js's animator
        # needs and throws "this._fn is not a function" on some builds, blanking
        # every chart. Asserting the native keys survive catches that revert on
        # every browser, even ones that happen to tolerate the blank. ---
        anim_keys = page.evaluate("() => Object.keys(Chart.defaults.animation)")
        for native in ("easing", "fn", "type"):
            assert native in anim_keys, (
                "Chart.defaults.animation was replaced with a bare object "
                f"(keys={anim_keys}); it must be mutated in place so the native "
                "animator fields survive"
            )

        # --- item 2: the 'Pick a market' empty state is gone ---
        assert page.locator("#emptyState").is_hidden(), "empty state still visible"
        assert page.locator("#positionBody").is_visible()

        # --- item 3: payoff guides are configured AND drawn ---
        guides = page.evaluate(
            "() => (window.__charts.payoffChart.options.plugins.payoffGuides) || null"
        )
        assert guides is not None, "payoff guides not passed through plugin options"
        for key in ("breakEven", "underlying", "maxGain", "maxLoss"):
            assert key in guides and guides[key] is not None, f"guide {key} missing"
        # navy break-even line (#1b2a52) and gold current-price line (#c8a951)
        assert _count_pixels_near(page, "payoffChart", (0x1b, 0x2a, 0x52)) > 0, \
            "break-even guide line not drawn"
        assert _count_pixels_near(page, "payoffChart", (0xc8, 0xa9, 0x51)) > 0, \
            "current-price guide line not drawn"

        # --- item 4: x-axis shows several labels, not just one ---
        for cid in _CANVASES:
            labels = page.evaluate(
                """(id) => {
                    const sc = window.__charts[id].scales.x;
                    return sc.ticks
                        .map(t => (t.label == null ? '' : String(t.label)))
                        .filter(s => s.length > 0);
                }""",
                cid,
            )
            assert len(labels) >= 2, f"{cid} x-axis shows only {labels}"

        # --- item 5: a small non-zero Greek is not shown as 0.00 ---
        # At the default "held to resolution" horizon (tau = 1) the premium is the
        # exact scaled-binary value, which is linear in the probability, so gamma
        # and theta are genuinely 0 there (the correct display, not a bug). The
        # "not mis-rendered as 0.00" check targets a genuinely non-zero Greek, so
        # first move the time-to-resolution slider to a modeled horizon (tau < 1)
        # where gamma/vega are non-zero, then assert the formatting.
        page.eval_on_selector(
            "#tauRange",
            """(el) => {
                el.value = '0.5';
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
            }""",
        )
        # Wait until the derivative has refreshed at the new horizon: gamma is a
        # pure-model Greek, so it becomes non-zero once tau < 1 takes effect.
        page.wait_for_function(
            """() => {
                const t = document.querySelector('#gvGamma');
                if (!t) return false;
                const s = t.textContent.trim();
                return s && !['0.00','-0.00','\u22120.00','0.000','-0.000'].includes(s);
            }""",
            timeout=15000,
        )
        for sel in ("#gvDelta", "#gvGamma", "#gvVega", "#gvTheta"):
            txt = page.locator(sel).inner_text().strip()
            assert txt not in ("0.00", "-0.00", "−0.00", "0.000", "-0.000"), \
                f"{sel} shows a zero-looking value: {txt!r}"
        # the compact scalar row too (these previously printed at 2 dp)
        scalars = page.locator("#greekScalars").inner_text()
        assert "-0.00" not in scalars and "−0.00" not in scalars

        # --- item 5b: 'Max you'd lose' is never negative zero ---
        strip = page.locator("#summaryStrip").inner_text()
        assert "−$0.000" not in strip and "-$0.000" not in strip

        # --- item 6: Delta caption states a correct unit ---
        delta_gloss = page.evaluate(
            """() => {
                for (const cell of document.querySelectorAll('.greek-cell')) {
                    const name = cell.querySelector('.gc-name');
                    if (name && name.textContent.trim() === 'Delta') {
                        return cell.querySelector('.gc-gloss').textContent;
                    }
                }
                return '';
            }"""
        )
        assert "+1%" not in delta_gloss, f"Delta caption still wrong: {delta_gloss!r}"
        assert ("1.00" in delta_gloss) or ("percentage point" in delta_gloss), \
            f"Delta caption unit unclear: {delta_gloss!r}"

        # --- item 17: rapid market clicks leave the newest market showing ---
        items = page.locator(".market-item")
        n = min(items.count(), 4)
        assert n >= 2
        titles = [items.nth(i).get_attribute("title") for i in range(n)]
        for i in range(n):
            items.nth(i).click()  # click fast, no waiting between
        last_title = titles[n - 1]
        # the shown position must settle on the LAST clicked market, not a
        # slower earlier response overwriting it
        page.wait_for_function(
            "(t) => document.getElementById('posTitle').textContent.trim() === t",
            arg=last_title, timeout=15000,
        )

        # --- no console errors, no uncaught page errors ---
        assert console_errors == [], f"console errors: {console_errors}"
        assert page_errors == [], f"page errors: {page_errors}"

        # --- items 18/(c): zero external-host network requests ---
        assert external == [], f"external requests made: {external}"
    finally:
        context.close()


def test_market_list_usable_at_894px(server, browser):
    """At a 894px-wide viewport (Chrome side panel) the market list shows
    several rows, not just one or two (audit item 7)."""
    _SELF[0] = server
    context = browser.new_context(viewport={"width": 894, "height": 820})
    page, console_errors, page_errors, external = _collect_page(context)
    try:
        page.goto(server + "/", wait_until="networkidle")
        page.wait_for_selector(".market-item", timeout=15000)
        # count how many market rows are visible within the scrollable list box
        visible = page.evaluate(
            """() => {
                const box = document.getElementById('marketList').getBoundingClientRect();
                let count = 0;
                for (const li of document.querySelectorAll('.market-item')) {
                    const r = li.getBoundingClientRect();
                    // row at least half inside the list viewport
                    if (r.top >= box.top - 1 && r.top < box.bottom - (r.height / 2)) count++;
                }
                return count;
            }"""
        )
        assert visible >= 3, f"only {visible} market rows visible at 894px width"
        assert external == [], f"external requests made: {external}"
    finally:
        context.close()


def test_reduced_motion_charts_still_draw(server, browser):
    """With prefers-reduced-motion the charts must still draw (the fix must not
    reintroduce the blank-chart bug by nulling the animation object)."""
    _SELF[0] = server
    context = browser.new_context(
        viewport={"width": 1280, "height": 900}, reduced_motion="reduce"
    )
    page, console_errors, page_errors, external = _collect_page(context)
    try:
        page.goto(server + "/", wait_until="networkidle")
        _select_first_market(page)
        page.wait_for_timeout(300)
        for cid in _CANVASES:
            assert _canvas_is_drawn(page, cid), f"{cid} blank under reduced motion"
        assert console_errors == [], f"console errors: {console_errors}"
        assert page_errors == [], f"page errors: {page_errors}"
        assert external == [], f"external requests: {external}"
    finally:
        context.close()
