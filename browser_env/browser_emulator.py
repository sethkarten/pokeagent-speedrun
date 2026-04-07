"""
BrowserEnv — Playwright-based browser environment for HTML5/WebGL games.

Playwright's sync API uses greenlets and must be called from the same thread
that created the browser.  Since the game server dispatches FastAPI endpoint
handlers on arbitrary threads, we run Playwright in a dedicated background
thread and marshal every call through a simple request/response queue.
"""

import base64
import io
import json
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

    def __init__(
        self,
        game_url: str,
        headless: bool = True,
        virtual_time: bool = True,
        step_budget_ms: int = 200,
        video_dir: Optional[str] = None,
    ):
        self.game_url = game_url
        self.headless = headless
        self.width = 800
        self.height = 600
        self._initialized = False

        # Video recording (Playwright native, .webm). When ``video_dir``
        # is set, the browser context is created with
        # ``record_video_dir=video_dir`` and Playwright flushes a single
        # .webm file when we close the context on stop. Recording is
        # off by default — opt in via the constructor or
        # BROWSER_VIDEO_DIR env var on the server side.
        self.video_dir: Optional[str] = video_dir
        # Set after _stop completes — the absolute path to the .webm
        # file Playwright wrote, or None if recording was disabled or
        # the flush failed.
        self.video_path: Optional[str] = None

        # Last known cursor position in canvas-relative coords (x, y).
        # None means "cursor has not been moved by the agent yet" — we
        # don't try to read the real OS cursor because the agent's only
        # interaction is via the dispatch primitives below. Updated by
        # _do_click_at / _do_double_click_at / _do_move_to / _do_drag_to.
        self.mouse_x: Optional[int] = None
        self.mouse_y: Optional[int] = None

        # Real-time game support: when virtual_time=True, BrowserEnv uses
        # Chrome DevTools Protocol Emulation.setVirtualTimePolicy to
        # freeze game time during the agent's VLM call and advance it by
        # ``step_budget_ms`` after each action. This makes Flappy-Bird-
        # class real-time games tractable: the bird stops falling while
        # the agent thinks. For turn-based games (Folder Dungeon style)
        # the pause is harmless and animations stop pixel-popping the
        # screenshot mid-frame.
        self.virtual_time = virtual_time
        self.step_budget_ms = step_budget_ms
        self.held_keys: set[str] = set()  # tracked across steps

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
            "mouse_x": self.mouse_x,
            "mouse_y": self.mouse_y,
            "held_keys": sorted(self.held_keys),
            "virtual_time": self.virtual_time,
            "step_budget_ms": self.step_budget_ms,
        }

    def focus_game(self) -> None:
        self._call("focus_game")

    def press_key(self, key: str, duration_ms: int = 100) -> None:
        self._call("press_key", key, duration_ms)

    def press_keys_sequence(self, keys: List[str], delay_ms: int = 100) -> None:
        self._call("press_keys_sequence", keys, delay_ms)

    def hold_key(self, key: str, duration_ms: int = 500) -> None:
        self._call("hold_key", key, duration_ms)

    def key_down(self, key: str) -> None:
        """Press a key WITHOUT releasing it. The key stays held across
        agent steps until a matching key_up is issued (or the env is
        stopped, or navigation recovery fires). The set of currently
        held keys is exposed via get_game_info()['held_keys']."""
        self._call("key_down", key)

    def key_up(self, key: str) -> None:
        """Release a key that was previously held with key_down."""
        self._call("key_up", key)

    def wait_ms(self, duration_ms: int) -> None:
        """Let game time pass without taking any other action.

        In virtual-time mode this advances virtual time by duration_ms.
        In wall-clock mode this just sleeps the dispatch thread for
        duration_ms. Useful for waiting on animations, falling
        platforms, enemy approach windows, etc — the agent should
        articulate WHY it's waiting in the tool's reasoning field.
        """
        self._call("wait_ms", duration_ms)

    def pause_virtual_time(self) -> None:
        """Freeze game virtual time (CDP). No-op if virtual_time=False."""
        self._call("pause_virtual_time")

    def advance_virtual_time(self, duration_ms: int) -> None:
        """Advance game virtual time by ``duration_ms`` then re-pause.
        No-op if virtual_time=False."""
        self._call("advance_virtual_time", duration_ms)

    def click_at(self, x: int, y: int) -> None:
        self._call("click_at", x, y)

    def double_click_at(self, x: int, y: int) -> None:
        self._call("double_click_at", x, y)

    def move_to(self, x: int, y: int, steps: int = 8) -> None:
        """Move the mouse cursor to (x, y) without clicking.

        Coordinates are canvas-relative (same convention as click_at).
        ``steps`` controls the number of intermediate mousemove events
        Playwright dispatches — useful for hover-driven UI that animates
        during cursor motion.
        """
        self._call("move_to", x, y, steps)

    def evaluate(self, expression: str) -> Any:
        """Evaluate a JS expression in the top-level page context.

        Mainly useful for tests that need to force a navigation away from
        the game URL to exercise the recovery path.
        """
        return self._call("evaluate", expression)

    def drag_to(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        steps: int = 12,
        hold_ms: int = 50,
    ) -> None:
        """Press at (x1, y1), drag to (x2, y2), release.

        Both endpoints are canvas-relative. ``steps`` controls how many
        mousemove events are issued during the drag (matters for games
        that sample the cursor position continuously). ``hold_ms`` is a
        small delay between mousedown and the first move so games that
        debounce can register the press.
        """
        self._call("drag_to", x1, y1, x2, y2, steps, hold_ms)

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
        # WebGL2 is required by Unity 2021+ WebGL builds. Playwright's headless
        # Chromium ships without real GPU support — it can only do WebGL1 via
        # SwiftShader, so Unity loaders silently stall on the splash screen.
        # The fix is to run headed against a virtual display (Xvfb), which gives
        # us WebGL2 via ANGLE+SwiftShader on Vulkan. `run_browser.sh` wraps the
        # whole process in `xvfb-run` and passes --headed.
        browser = pw.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                # Enable GPU acceleration for WebGL games (Unity, etc.)
                "--enable-gpu",
                "--enable-webgl",
                "--use-gl=angle",
                "--enable-unsafe-swiftshader",
                "--ignore-gpu-blocklist",
            ],
        )
        # Build context kwargs. record_video_dir is only set when video
        # recording is requested — Playwright otherwise doesn't write
        # any video and there's zero overhead. The recording size
        # matches the viewport so the .webm is exactly what the agent
        # sees through its screenshots.
        ctx_kwargs = dict(
            viewport={"width": 960, "height": 600},
            device_scale_factor=1,  # Prevent retina 2x scaling
            ignore_https_errors=True,
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        if self.video_dir:
            try:
                import os as _os_local
                _os_local.makedirs(self.video_dir, exist_ok=True)
                ctx_kwargs["record_video_dir"] = self.video_dir
                ctx_kwargs["record_video_size"] = {"width": 960, "height": 600}
                logger.info("Video recording enabled — dir: %s", self.video_dir)
            except Exception as e:
                logger.warning("could not enable video recording: %s", e)
        ctx = browser.new_context(**ctx_kwargs)
        page = ctx.new_page()

        # Internal state (only accessed from this thread)
        game_frame = None   # FrameLocator for itch.io iframe
        canvas = None       # Locator for the <canvas>
        cdp_session = None  # Lazy CDP session for virtual time (Chromium-only)
        # Tracks whether we've explicitly paused virtual time at least
        # once. Init runs without virtual-time control (we want the
        # "wait for splash screen to load" loop to actually let game
        # time pass) and only flips to true after the first explicit
        # pause at the end of _do_init.
        vt_state = {"paused": False}

        # ---- helpers (closures over page/game_frame/canvas) ----

        def _input_target():
            if canvas is not None:
                return canvas
            if game_frame is not None:
                return page.locator("iframe#game_drop, .game_frame iframe").first
            return page.locator("body")

        def _setup_game_frame():
            """Locate the iframe + canvas after a fresh page.goto.

            Extracted from _do_init so the navigation recovery path can
            re-run the same setup without repeating itself.
            """
            nonlocal game_frame, canvas
            game_frame = None
            canvas = None

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

        def _ensure_cdp_session():
            """Lazily attach a CDP session for Emulation.* commands.

            Also subscribes to ``Emulation.virtualTimeBudgetExpired`` so
            ``_do_advance_virtual_time`` can wait synchronously for
            advances to complete. Idempotent. Returns None if CDP isn't
            available (non-Chromium browsers, future Playwright that
            strips CDP, etc) — callers should treat None as "virtual
            time disabled" and fall back to wall-clock sleeps.
            """
            nonlocal cdp_session
            if cdp_session is not None:
                return cdp_session
            try:
                cdp_session = ctx.new_cdp_session(page)
                # NB: subscribe BEFORE any setVirtualTimePolicy call so
                # we never miss the budget-expired event.
                try:
                    cdp_session.on(
                        "Emulation.virtualTimeBudgetExpired",
                        _on_virtual_time_budget_expired,
                    )
                except Exception as e:
                    logger.warning(
                        "could not subscribe to virtualTimeBudgetExpired: %s", e,
                    )
                return cdp_session
            except Exception as e:
                logger.warning("CDP session unavailable: %s", e)
                return None

        # Virtual time via injected JS shim — passthrough state machine.
        #
        # We tried CDP Emulation.setVirtualTimePolicy first, but it's
        # unrecoverably broken in Chromium when the main thread sleeps
        # wall-clock for >1s with virtual time paused — the renderer's
        # internal clock stops responding to advance commands and
        # virtualTimeBudgetExpired never fires again.
        #
        # Then we tried a "queue everything" shim: override RAF and the
        # timer family to push into a JS-side queue, drain only when
        # __advance(ms) is called. That worked for itch.io games (which
        # wrap the game in an iframe — the shim ran in the parent only
        # and silently no-op'd) but DEADLOCKED direct-page games like
        # flappybird.io because the game's RAF loop was captured before
        # any rendering could happen and the canvas stayed at its
        # uninitialized black state forever.
        #
        # This version is a state machine:
        #
        #   live mode (default, paused=false):
        #     RAF and the timers run NATIVELY — full passthrough. The
        #     game loop ticks normally on wall clock. Date.now and
        #     performance.now also pass through. The page loads and
        #     animates exactly as if the shim weren't there.
        #
        #   paused mode (after __pause()):
        #     Newly-registered RAF/timers go into our queue instead of
        #     native APIs. Date.now and performance.now return the
        #     frozen virtualNow. Already-pending native callbacks (the
        #     ones registered before pause()) still fire — that's an
        #     unavoidable single-frame escape, fine in practice.
        #
        #   __advance(ms) (only meaningful while paused):
        #     Walk the queue forward by ms of virtual time, firing any
        #     callbacks whose deadlines fall in the window. Newly-
        #     registered RAF/timers from those callbacks land back in
        #     the queue. Returns to the paused snapshot at the end.
        #
        # Injected via page.add_init_script so it runs on every
        # navigation, including after _check_and_recover_nav fires.
        VIRTUAL_TIME_SHIM = r"""
        (() => {
          if (window.__vtShim) return;  // idempotent
          const origRAF = window.requestAnimationFrame.bind(window);
          const origCAF = window.cancelAnimationFrame.bind(window);
          const origST = window.setTimeout.bind(window);
          const origCT = window.clearTimeout.bind(window);
          const origSI = window.setInterval.bind(window);
          const origCI = window.clearInterval.bind(window);
          const origDateNow = Date.now.bind(Date);
          const origPerfNow = performance.now.bind(performance);

          // Mode flag. While false the overrides are passthroughs and
          // the page runs exactly as if no shim were installed.
          let paused = false;

          // Virtual clock — only meaningful while paused. Snapshotted
          // at pause() time so Date.now is frozen during the agent's
          // VLM call. advance(ms) bumps it forward.
          let virtualNow = origDateNow();
          let virtualPerfBase = origPerfNow();
          let virtualPerfNow = origPerfNow();

          // Negative ids for our queue so they can never collide with
          // native ids while in paused mode (some games store the id
          // and pass it to clearTimeout later).
          let nextQueueId = -1;
          // Pending timer entries: {id, deadline, cb, interval, args}
          const timers = new Map();
          // Pending RAF callbacks: [{id, cb}]
          let rafQueue = [];

          window.requestAnimationFrame = function(cb) {
            if (!paused) return origRAF(cb);
            const id = nextQueueId--;
            rafQueue.push({id, cb});
            return id;
          };
          window.cancelAnimationFrame = function(id) {
            if (id >= 0) return origCAF(id);
            rafQueue = rafQueue.filter(e => e.id !== id);
          };
          window.setTimeout = function(cb, delay, ...args) {
            if (!paused) return origST(cb, delay, ...args);
            const id = nextQueueId--;
            timers.set(id, {
              id, cb, args,
              deadline: virtualNow + (delay || 0),
              interval: 0,
            });
            return id;
          };
          window.clearTimeout = function(id) {
            if (id >= 0) return origCT(id);
            timers.delete(id);
          };
          window.setInterval = function(cb, delay, ...args) {
            if (!paused) return origSI(cb, delay, ...args);
            const id = nextQueueId--;
            timers.set(id, {
              id, cb, args,
              deadline: virtualNow + (delay || 0),
              interval: delay || 0,
            });
            return id;
          };
          window.clearInterval = function(id) {
            if (id >= 0) return origCI(id);
            timers.delete(id);
          };

          // Date.now and performance.now: live = native, paused =
          // frozen. The frozen value is the snapshot taken at pause()
          // time, advanced only by __advance(ms).
          Date.now = function() {
            return paused ? virtualNow : origDateNow();
          };
          performance.now = function() {
            return paused ? virtualPerfNow : origPerfNow();
          };

          window.__vtShim = {
            pause() {
              if (paused) return;
              // Snapshot the wall clock so future Date.now/performance.now
              // calls return a stable value.
              virtualNow = origDateNow();
              virtualPerfNow = origPerfNow();
              paused = true;
            },
            resume() {
              // Drain whatever's queued back into native APIs so the
              // game continues without losing work, then go live.
              const t = Array.from(timers.values());
              timers.clear();
              for (const e of t) {
                const remaining = Math.max(0, e.deadline - virtualNow);
                if (e.interval > 0) {
                  origSI(e.cb, e.interval, ...(e.args || []));
                } else {
                  origST(e.cb, remaining, ...(e.args || []));
                }
              }
              const q = rafQueue;
              rafQueue = [];
              for (const e of q) {
                origRAF(e.cb);
              }
              paused = false;
            },
            isPaused() { return paused; },
            now() { return paused ? virtualNow : origDateNow(); },
            queueSizes() {
              return {raf: rafQueue.length, timers: timers.size};
            },
            // Advance virtual time by ms. Only meaningful while paused.
            // Fires any queued callbacks whose deadlines fall in the
            // window, in deadline order. Returns the number of
            // callbacks fired.
            advance(ms) {
              if (!paused) return 0;
              const target = virtualNow + ms;
              let fired = 0;
              const maxIter = 5000;  // safety: setTimeout(0) chains
              for (let iter = 0; iter < maxIter; iter++) {
                // Find earliest-deadline timer.
                let nextEntry = null;
                for (const e of timers.values()) {
                  if (!nextEntry || e.deadline < nextEntry.deadline) {
                    nextEntry = e;
                  }
                }
                if (nextEntry && nextEntry.deadline <= target) {
                  virtualNow = nextEntry.deadline;
                  virtualPerfNow += (nextEntry.deadline - virtualNow);
                  if (nextEntry.interval > 0) {
                    nextEntry.deadline += nextEntry.interval;
                  } else {
                    timers.delete(nextEntry.id);
                  }
                  try { nextEntry.cb.apply(null, nextEntry.args || []); }
                  catch (e) { console.error("vtShim timer error:", e); }
                  fired++;
                  continue;
                }
                break;
              }
              // Drain RAF queue (RAF callbacks all run together at
              // the "start" of each frame in real browsers, so we fire
              // them after timers but before final virtualNow bump).
              if (rafQueue.length > 0) {
                const q = rafQueue;
                rafQueue = [];
                for (const e of q) {
                  try { e.cb(virtualNow); fired++; }
                  catch (err) { console.error("vtShim raf error:", err); }
                }
              }
              const dt = target - virtualNow;
              virtualNow = target;
              virtualPerfNow += dt;
              return fired;
            },
          };
        })();
        """

        def _inject_virtual_time_shim():
            """Inject the virtual time shim into the current page.

            Called from _do_init and _check_and_recover_nav. We use
            page.add_init_script so the shim runs on every subsequent
            navigation too — but we also evaluate it immediately on the
            current page so it's active right now.
            """
            if not self.virtual_time:
                return
            try:
                page.add_init_script(VIRTUAL_TIME_SHIM)
            except Exception as e:
                logger.debug("add_init_script(vt shim) failed: %s", e)
            try:
                page.evaluate(VIRTUAL_TIME_SHIM)
            except Exception as e:
                logger.debug("evaluate(vt shim) failed: %s", e)

        def _do_pause_virtual_time():
            """Pause game time by telling the JS shim to stop firing
            queued callbacks. Game animations/setTimeout/setInterval
            stop. The renderer compositor still runs (so screenshots
            work) but no game logic executes.
            """
            if not self.virtual_time:
                return
            try:
                page.evaluate("window.__vtShim && window.__vtShim.pause()")
                vt_state["paused"] = True
            except Exception as e:
                logger.debug("pause_virtual_time failed: %s", e)

        def _do_advance_virtual_time(duration_ms: int):
            """Advance virtual time by ``duration_ms`` and fire any
            timers/RAF callbacks whose deadlines fall in the window.
            Stable across long wall sleeps because the shim is
            stateful in user-space JS, not in the Chromium renderer.
            """
            ms = max(0, int(duration_ms))
            if not self.virtual_time:
                time.sleep(ms / 1000.0)
                return
            try:
                fired = page.evaluate(
                    "(ms) => window.__vtShim ? window.__vtShim.advance(ms) : 0",
                    ms,
                )
                logger.debug("vt advance(%dms) fired %s callbacks", ms, fired)
                vt_state["paused"] = True  # advance returns to paused state
            except Exception as e:
                logger.debug("advance_virtual_time failed: %s", e)
                time.sleep(ms / 1000.0)

        def _do_wait_ms(duration_ms: int):
            """Let game time pass without doing anything else."""
            _do_advance_virtual_time(duration_ms)

        def _do_key_down(key: str):
            """Press a key without releasing it. Tracks the held key in
            self.held_keys so we can release it on stop / nav recovery."""
            try:
                page.keyboard.down(key)
                self.held_keys.add(key)
            except Exception as e:
                logger.error("key_down(%s) failed: %s", key, e)

        def _do_key_up(key: str):
            """Release a previously-held key."""
            try:
                page.keyboard.up(key)
            except Exception as e:
                logger.error("key_up(%s) failed: %s", key, e)
            finally:
                self.held_keys.discard(key)

        def _release_all_held_keys(reason: str = ""):
            """Best-effort release of every key the agent has been
            holding. Called on _stop and on navigation recovery — a
            fresh page has no held keys, and we don't want a leaked
            ArrowRight to make the agent's next browser session
            mysterious."""
            if not self.held_keys:
                return
            keys = list(self.held_keys)
            logger.info("Releasing %d held keys (%s): %s",
                        len(keys), reason or "cleanup", keys)
            for k in keys:
                try:
                    page.keyboard.up(k)
                except Exception:
                    pass
            self.held_keys.clear()

        def _do_init():
            # Register the virtual-time shim BEFORE navigating so it
            # runs on the very first page load (and any subsequent ones
            # via add_init_script). The shim is harmless when
            # virtual_time=False.
            _inject_virtual_time_shim()
            logger.info("Navigating to %s", self.game_url)
            page.goto(self.game_url, wait_until="domcontentloaded", timeout=60_000)
            _setup_game_frame()

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
            # Freeze virtual time so the first agent decision sees a
            # static game state. Subsequent commands re-pause
            # implicitly (advance returns to pause, init->pause is
            # explicit). No-op when self.virtual_time is False.
            _do_pause_virtual_time()
            logger.info(
                "BrowserEnv ready — canvas %dx%d  virtual_time=%s  step_budget=%dms",
                self.width, self.height, self.virtual_time, self.step_budget_ms,
            )

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
            """Screenshot the game canvas.

            Strategy: take a full-page screenshot and clip to the iframe's
            bounding box. This avoids stale canvas locators and never times
            out — page.screenshot always succeeds. If we have no iframe
            (direct game URL), screenshot the page or canvas directly.

            The JS-shim virtual time approach (see _do_pause_virtual_time)
            doesn't break page.screenshot the way CDP virtual time pause
            did, because the renderer compositor still ticks — only
            game-side timers/RAF are gated. So the normal Playwright
            screenshot path works even while paused.
            """
            nonlocal canvas, game_frame
            try:
                if game_frame is not None:
                    # Get the iframe's current bounding box (refreshes each call)
                    iframe_el = page.locator("iframe#game_drop, .game_frame iframe").first
                    box = iframe_el.bounding_box(timeout=5_000)
                    if box and box["width"] > 0 and box["height"] > 0:
                        png = page.screenshot(
                            type="png",
                            clip={
                                "x": box["x"],
                                "y": box["y"],
                                "width": box["width"],
                                "height": box["height"],
                            },
                            timeout=10_000,
                        )
                        return Image.open(io.BytesIO(png)).convert("RGB")
                    raise RuntimeError("iframe has zero dimensions")
                elif canvas is not None:
                    # Direct canvas (non-iframe game)
                    box = canvas.bounding_box(timeout=5_000)
                    if box and box["width"] > 0 and box["height"] > 0:
                        png = page.screenshot(
                            type="png",
                            clip={
                                "x": box["x"],
                                "y": box["y"],
                                "width": box["width"],
                                "height": box["height"],
                            },
                            timeout=10_000,
                        )
                        return Image.open(io.BytesIO(png)).convert("RGB")
                    raise RuntimeError("canvas has zero dimensions")
                else:
                    png = page.screenshot(type="png", timeout=10_000)
                    return Image.open(io.BytesIO(png)).convert("RGB")
            except Exception as e:
                logger.error("Screenshot failed: %s — falling back to full page", e)
                try:
                    png = page.screenshot(type="png", timeout=10_000, full_page=False)
                    return Image.open(io.BytesIO(png)).convert("RGB")
                except Exception as e2:
                    logger.error("Full-page screenshot also failed: %s", e2)
                    raise RuntimeError(f"Screenshot completely failed: {e2}") from e2

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

        def _clamp_to_canvas(x: int, y: int) -> tuple:
            """Clamp (x, y) into the canvas bounding box.

            Playwright's locator.click(position={x, y}) does an
            actionability check that includes "the click target is
            inside the element box". When the position is outside the
            element, Playwright tries to scroll it into view and waits
            up to 30s before timing out — this happens constantly when
            VLMs hallucinate coordinates that are slightly off-canvas
            (e.g. clicking the page chrome under the canvas at y=784
            for a 600-px-tall game). Each timeout costs 30s of wall
            clock and accomplishes nothing. We clamp here so off-
            canvas clicks land on the nearest edge instead.
            """
            cx = max(0, min(int(x), max(0, self.width - 1)))
            cy = max(0, min(int(y), max(0, self.height - 1)))
            if (cx, cy) != (int(x), int(y)):
                logger.debug(
                    "clamped click (%d,%d) -> (%d,%d) [canvas %dx%d]",
                    x, y, cx, cy, self.width, self.height,
                )
            return cx, cy

        def _do_click_at(x: int, y: int):
            """Click at canvas-relative (x, y) using page.mouse.click.

            We deliberately use page.mouse.click instead of
            locator.click(position=...) because the locator path runs
            an actionability check that often fails on game canvases:
            it requires the click target to be the topmost element at
            that pixel, but games frequently have transparent overlays
            (HUD, ad iframes, focus rings) on top of the canvas. The
            check then waits up to its timeout and bails out, costing
            seconds per failed click and accomplishing nothing.

            page.mouse.click dispatches the mousedown/mouseup events
            at the absolute page coordinate without any actionability
            wait. We translate canvas-relative -> page-absolute via
            _canvas_to_page (which already exists for the
            move/drag primitives).
            """
            cx, cy = _clamp_to_canvas(x, y)
            try:
                page_xy = _canvas_to_page(cx, cy)
                if page_xy is None:
                    # No canvas/iframe — treat input as page-absolute.
                    page_xy = (cx, cy)
                page.mouse.click(page_xy[0], page_xy[1])
                self.mouse_x, self.mouse_y = cx, cy
            except Exception as e:
                logger.error("click_at(%d, %d) failed: %s", x, y, e)

        def _do_double_click_at(x: int, y: int):
            cx, cy = _clamp_to_canvas(x, y)
            try:
                page_xy = _canvas_to_page(cx, cy)
                if page_xy is None:
                    page_xy = (cx, cy)
                page.mouse.dblclick(page_xy[0], page_xy[1])
                self.mouse_x, self.mouse_y = cx, cy
            except Exception as e:
                logger.error("double_click_at(%d, %d) failed: %s", x, y, e)

        def _canvas_to_page(x: int, y: int) -> Optional[tuple]:
            """Convert canvas-relative (x, y) to page-absolute coordinates.

            Locator.click(position=...) handles the offset for us, but
            page.mouse.move()/down()/up() take page-absolute coordinates,
            so for the move/drag primitives we have to add the canvas
            (or iframe) origin ourselves. Returns None if neither the
            canvas nor the iframe has a bounding box yet.
            """
            try:
                if canvas is not None:
                    box = canvas.bounding_box(timeout=2_000)
                    if box:
                        return (box["x"] + x, box["y"] + y)
                if game_frame is not None:
                    iframe_el = page.locator(
                        "iframe#game_drop, .game_frame iframe"
                    ).first
                    box = iframe_el.bounding_box(timeout=2_000)
                    if box:
                        return (box["x"] + x, box["y"] + y)
            except Exception as e:
                logger.warning("_canvas_to_page(%d, %d) failed: %s", x, y, e)
            return None

        def _do_move_to(x: int, y: int, steps: int):
            try:
                page_xy = _canvas_to_page(x, y)
                if page_xy is None:
                    # Fall back to treating coordinates as page-absolute
                    page_xy = (x, y)
                page.mouse.move(page_xy[0], page_xy[1], steps=max(1, int(steps)))
                self.mouse_x, self.mouse_y = int(x), int(y)
            except Exception as e:
                logger.error("move_to(%d, %d) failed: %s", x, y, e)

        def _do_drag_to(x1: int, y1: int, x2: int, y2: int, steps: int, hold_ms: int):
            try:
                start = _canvas_to_page(x1, y1) or (x1, y1)
                end = _canvas_to_page(x2, y2) or (x2, y2)
                page.mouse.move(start[0], start[1], steps=1)
                page.mouse.down()
                if hold_ms > 0:
                    time.sleep(hold_ms / 1000.0)
                page.mouse.move(end[0], end[1], steps=max(1, int(steps)))
                page.mouse.up()
                self.mouse_x, self.mouse_y = int(x2), int(y2)
            except Exception as e:
                logger.error(
                    "drag_to((%d,%d)->(%d,%d)) failed: %s", x1, y1, x2, y2, e
                )

        # ---- dispatch map ----
        # Track navigation state so we can detect when the agent
        # accidentally clicks something that takes the page off the game
        # (e.g. an in-game Discord button, an "X" close icon, an itch.io
        # popup link). The dispatch loop calls _check_and_recover_nav
        # before every command so we recover proactively instead of
        # spinning forever waiting for an iframe that's no longer there.
        from urllib.parse import urlsplit

        def _url_key(url: str) -> tuple[str, str, str]:
            """Compare URLs by (scheme, host, path) — ignore query/fragment."""
            try:
                p = urlsplit(url)
                return (p.scheme, p.netloc, p.path.rstrip("/"))
            except Exception:
                return ("", "", "")

        game_url_key = _url_key(self.game_url)
        nav_recoveries = {"count": 0}

        def _check_and_recover_nav():
            """If the page has navigated off the game URL, navigate back."""
            try:
                current = page.url or ""
            except Exception:
                return
            if not current or current == "about:blank":
                return
            if _url_key(current) == game_url_key:
                return
            # Navigation away from the game detected.
            nav_recoveries["count"] += 1
            logger.warning(
                "External navigation detected (#%d): %s — recovering to %s",
                nav_recoveries["count"], current, self.game_url,
            )
            # Drop any held-key state — the fresh page can't possibly
            # have OS-level held keys, so let the agent re-press them
            # explicitly if it still wants them held.
            _release_all_held_keys(reason="nav recovery")
            try:
                page.goto(self.game_url, wait_until="domcontentloaded", timeout=60_000)
                _setup_game_frame()
                _do_focus()
                # The init-script form of the shim auto-injects on
                # every navigation, but evaluate it explicitly too in
                # case Chrome dropped the init script for any reason.
                _inject_virtual_time_shim()
                _do_pause_virtual_time()
                logger.info("Navigation recovery complete")
            except Exception as e:
                logger.error("Navigation recovery failed: %s", e)

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
            "move_to": lambda x, y, s: _do_move_to(x, y, s),
            "drag_to": lambda x1, y1, x2, y2, s, h: _do_drag_to(x1, y1, x2, y2, s, h),
            "evaluate": lambda expr: page.evaluate(expr),
            "key_down": lambda k: _do_key_down(k),
            "key_up": lambda k: _do_key_up(k),
            "wait_ms": lambda ms: _do_wait_ms(ms),
            "pause_virtual_time": lambda: _do_pause_virtual_time(),
            "advance_virtual_time": lambda ms: _do_advance_virtual_time(ms),
        }

        # ---- event loop ----
        while True:
            try:
                method, args, result_q = self._cmd_q.get(timeout=1.0)
            except queue.Empty:
                continue

            if method == "_stop":
                _release_all_held_keys(reason="stop")
                # Snapshot the page video reference BEFORE we close the
                # context. Playwright finalizes the .webm during
                # context.close(); afterwards page.video.path() returns
                # the on-disk path. We grab the reference here, then
                # break out and let the cleanup block below do the
                # actual close + path resolution.
                result_q.put(None)
                break

            fn = dispatch.get(method)
            if fn is None:
                result_q.put(ValueError(f"Unknown method: {method}"))
                continue

            # Recover from any navigation that happened since the last
            # command (e.g. agent accidentally clicked an external link).
            # Skip for _init (which does its own navigation) — every other
            # command assumes we're on the game page.
            if method != "_init":
                try:
                    _check_and_recover_nav()
                except Exception as e:
                    logger.warning("nav check failed: %s", e)

            try:
                result = fn(*args)
                result_q.put(result)
            except Exception as e:
                logger.error("BrowserEnv.%s failed: %s", method, e)
                result_q.put(e)

        # Cleanup. When video recording is enabled, we MUST close the
        # context (not just the browser) for Playwright to flush the
        # .webm to disk and let page.video.path() return a valid path.
        # Browser.close() alone leaves the file truncated.
        if self.video_dir:
            try:
                video = page.video
            except Exception:
                video = None
            try:
                ctx.close()
            except Exception as e:
                logger.warning("ctx.close() failed: %s", e)
            try:
                if video is not None:
                    self.video_path = video.path()
                    logger.info("Video recording saved: %s", self.video_path)
            except Exception as e:
                logger.warning("could not resolve video path: %s", e)
        try:
            browser.close()
        except Exception:
            pass
        try:
            pw.stop()
        except Exception:
            pass
        logger.info("BrowserEnv Playwright thread stopped")
