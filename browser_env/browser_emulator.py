"""
BrowserEnv — Playwright-based browser environment for HTML5/WebGL games.

Playwright's sync API uses greenlets and must be called from the same thread
that created the browser.  Since the game server dispatches FastAPI endpoint
handlers on arbitrary threads, we run Playwright in a dedicated background
thread and marshal every call through a simple request/response queue.
"""

import io
import logging
import queue
import threading
import time
from typing import Any, Dict, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)


class BrowserEnv:
    """Playwright-based browser environment for HTML5/WebGL games.

    All Playwright operations run on a single dedicated thread to avoid
    greenlet/thread affinity issues.
    """

    def __init__(self, game_url: str, headless: bool = True):
        self.game_url = game_url
        self.headless = headless
        self.width = 800
        self.height = 600
        self._initialized = False

        # Thread communication
        self._cmd_q: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API (thread-safe — dispatches to the Playwright thread)
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        self._thread = threading.Thread(target=self._pw_thread, daemon=True)
        self._thread.start()
        # Block until the Playwright thread finishes setup
        result = self._call("_init")
        if isinstance(result, Exception):
            raise result
        self._initialized = True

    def stop(self) -> None:
        if self._initialized:
            self._call("_stop")
        self._initialized = False

    def get_screenshot(self) -> Image.Image:
        return self._call("get_screenshot")

    def get_page_text(self) -> str:
        return self._call("get_page_text")

    def get_game_info(self) -> Dict[str, Any]:
        return {
            "url": self.game_url,
            "title": self._call("get_title"),
            "canvas_width": self.width,
            "canvas_height": self.height,
        }

    def focus_game(self) -> None:
        self._call("focus_game")

    def press_key(self, key: str, duration_ms: int = 100) -> None:
        self._call("press_key", key, duration_ms)

    def press_keys_sequence(self, keys: List[str], delay_ms: int = 100) -> None:
        self._call("press_keys_sequence", keys, delay_ms)

    def hold_key(self, key: str, duration_ms: int = 500) -> None:
        self._call("hold_key", key, duration_ms)

    def click_at(self, x: int, y: int) -> None:
        self._call("click_at", x, y)

    def double_click_at(self, x: int, y: int) -> None:
        self._call("double_click_at", x, y)

    # ------------------------------------------------------------------
    # Dispatch helpers
    # ------------------------------------------------------------------

    def _call(self, method: str, *args, timeout: float = 60.0):
        """Send a command to the Playwright thread and wait for the result."""
        result_q: queue.Queue = queue.Queue()
        self._cmd_q.put((method, args, result_q))
        try:
            result = result_q.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"BrowserEnv.{method} timed out after {timeout}s")
        if isinstance(result, Exception):
            raise result
        return result

    # ------------------------------------------------------------------
    # Playwright thread — all browser interaction happens here
    # ------------------------------------------------------------------

    def _pw_thread(self) -> None:
        """Background thread that owns the Playwright browser."""
        from playwright.sync_api import sync_playwright

        pw = sync_playwright().start()
        browser = pw.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                # Enable GPU acceleration for WebGL games (Unity, etc.)
                "--enable-gpu",
                "--enable-webgl",
                "--use-gl=angle",
                "--enable-unsafe-swiftshader",
                "--ignore-gpu-blocklist",
            ],
        )
        ctx = browser.new_context(
            viewport={"width": 960, "height": 600},
            device_scale_factor=1,  # Prevent retina 2x scaling
            ignore_https_errors=True,
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = ctx.new_page()

        # Internal state (only accessed from this thread)
        game_frame = None   # FrameLocator for itch.io iframe
        canvas = None       # Locator for the <canvas>

        # ---- helpers (closures over page/game_frame/canvas) ----

        def _input_target():
            if canvas is not None:
                return canvas
            if game_frame is not None:
                return page.locator("iframe#game_drop, .game_frame iframe").first
            return page.locator("body")

        def _do_init():
            nonlocal game_frame, canvas

            logger.info("Navigating to %s", self.game_url)
            page.goto(self.game_url, wait_until="domcontentloaded", timeout=60_000)

            if "itch.io" in self.game_url:
                # Click "Run game" button
                run_btn = page.locator("button.load_iframe_btn, .above_game_frame button")
                try:
                    run_btn.wait_for(state="visible", timeout=10_000)
                    logger.info("Clicking 'Run game' button")
                    run_btn.click()
                except Exception:
                    logger.info("No 'Run game' button — game may already be loaded")

                # Wait for iframe
                iframe_sel = "iframe#game_drop"
                try:
                    page.wait_for_selector(iframe_sel, state="attached", timeout=30_000)
                except Exception:
                    iframe_sel = ".game_frame iframe, #game_drop"
                    page.wait_for_selector(iframe_sel, state="attached", timeout=15_000)

                logger.info("Game iframe found")
                game_frame = page.frame_locator(iframe_sel).first

                # Wait for canvas inside iframe
                try:
                    canvas = game_frame.locator("canvas").first
                    canvas.wait_for(state="visible", timeout=30_000)
                    logger.info("Canvas found inside iframe")
                except Exception:
                    logger.warning("No <canvas> in iframe — will screenshot iframe")
                    canvas = None
            else:
                # Direct page
                try:
                    canvas = page.locator("canvas").first
                    canvas.wait_for(state="visible", timeout=15_000)
                    logger.info("Canvas found on page")
                except Exception:
                    logger.warning("No <canvas> — will screenshot full page")
                    canvas = None

            # Read canvas dimensions
            if canvas is not None:
                try:
                    box = canvas.bounding_box()
                    if box:
                        self.width = int(box["width"])
                        self.height = int(box["height"])
                except Exception:
                    pass

            # Wait for game to finish loading (detect visual change in canvas)
            logger.info("Waiting for game to finish loading...")
            import numpy as np
            prev_mean = 0.0
            stable_count = 0
            for _ in range(30):  # up to 30 seconds
                time.sleep(1)
                try:
                    img = _do_screenshot()
                    cur_mean = float(np.array(img).mean())
                    # Consider loaded when brightness changes from splash and stabilizes
                    if abs(cur_mean - prev_mean) < 1.0 and cur_mean > 40:
                        stable_count += 1
                        if stable_count >= 3:
                            logger.info("Game appears loaded (stable frame detected)")
                            break
                    else:
                        stable_count = 0
                    prev_mean = cur_mean
                except Exception:
                    pass
            else:
                logger.info("Loading wait finished (timeout — proceeding anyway)")

            # Focus
            _do_focus()
            logger.info("BrowserEnv ready — canvas %dx%d", self.width, self.height)

        def _do_focus():
            try:
                if canvas is not None:
                    canvas.click(force=True)
                elif game_frame is not None:
                    page.locator("iframe#game_drop, .game_frame iframe").first.click(force=True)
                else:
                    page.click("body")
            except Exception as e:
                logger.warning("focus_game failed: %s", e)

        def _do_screenshot() -> Image.Image:
            nonlocal canvas, game_frame
            for attempt in range(2):
                try:
                    if canvas is not None:
                        png = canvas.screenshot(type="png", timeout=10_000)
                    elif game_frame is not None:
                        png = page.locator(
                            "iframe#game_drop, .game_frame iframe"
                        ).first.screenshot(type="png", timeout=10_000)
                    else:
                        png = page.screenshot(type="png", timeout=10_000)
                    return Image.open(io.BytesIO(png)).convert("RGB")
                except Exception as e:
                    logger.warning("Screenshot attempt %d failed: %s", attempt + 1, e)
                    if attempt == 0:
                        # Canvas locator may be stale (game iframe reloaded).
                        # Try to re-acquire the canvas before giving up.
                        try:
                            if game_frame is not None:
                                canvas = game_frame.locator("canvas").first
                                canvas.wait_for(state="visible", timeout=5_000)
                                logger.info("Re-acquired canvas after screenshot failure")
                        except Exception:
                            canvas = None
            logger.error("Screenshot failed after retry, returning blank")
            return Image.new("RGB", (self.width, self.height), (0, 0, 0))

        def _do_page_text() -> str:
            try:
                if game_frame is not None:
                    texts = game_frame.locator("body").all_inner_texts()
                    return "\n".join(texts).strip()
                return page.inner_text("body")
            except Exception:
                return ""

        def _do_press_key(key: str, duration_ms: int):
            try:
                _input_target().press(key, delay=duration_ms)
            except Exception as e:
                logger.error("press_key(%s) failed: %s", key, e)

        def _do_press_keys_sequence(keys: list, delay_ms: int):
            for key in keys:
                _do_press_key(key, delay_ms)
                time.sleep(delay_ms / 1000.0)

        def _do_hold_key(key: str, duration_ms: int):
            try:
                page.keyboard.down(key)
                time.sleep(duration_ms / 1000.0)
                page.keyboard.up(key)
            except Exception as e:
                logger.error("hold_key(%s) failed: %s", key, e)

        def _do_click_at(x: int, y: int):
            try:
                if canvas is not None:
                    canvas.click(position={"x": x, "y": y})
                elif game_frame is not None:
                    page.locator("iframe#game_drop, .game_frame iframe").first.click(position={"x": x, "y": y})
                else:
                    page.mouse.click(x, y)
            except Exception as e:
                logger.error("click_at(%d, %d) failed: %s", x, y, e)

        def _do_double_click_at(x: int, y: int):
            try:
                if canvas is not None:
                    canvas.dblclick(position={"x": x, "y": y})
                elif game_frame is not None:
                    page.locator("iframe#game_drop, .game_frame iframe").first.dblclick(position={"x": x, "y": y})
                else:
                    page.mouse.dblclick(x, y)
            except Exception as e:
                logger.error("double_click_at(%d, %d) failed: %s", x, y, e)

        # ---- dispatch map ----
        dispatch = {
            "_init": lambda: _do_init(),
            "_stop": lambda: None,  # handled specially below
            "get_screenshot": lambda: _do_screenshot(),
            "get_page_text": lambda: _do_page_text(),
            "get_title": lambda: page.title(),
            "focus_game": lambda: _do_focus(),
            "press_key": lambda k, d: _do_press_key(k, d),
            "press_keys_sequence": lambda ks, d: _do_press_keys_sequence(ks, d),
            "hold_key": lambda k, d: _do_hold_key(k, d),
            "click_at": lambda x, y: _do_click_at(x, y),
            "double_click_at": lambda x, y: _do_double_click_at(x, y),
        }

        # ---- event loop ----
        while True:
            try:
                method, args, result_q = self._cmd_q.get(timeout=1.0)
            except queue.Empty:
                continue

            if method == "_stop":
                result_q.put(None)
                break

            fn = dispatch.get(method)
            if fn is None:
                result_q.put(ValueError(f"Unknown method: {method}"))
                continue

            try:
                result = fn(*args)
                result_q.put(result)
            except Exception as e:
                logger.error("BrowserEnv.%s failed: %s", method, e)
                result_q.put(e)

        # Cleanup
        try:
            browser.close()
        except Exception:
            pass
        try:
            pw.stop()
        except Exception:
            pass
        logger.info("BrowserEnv Playwright thread stopped")
