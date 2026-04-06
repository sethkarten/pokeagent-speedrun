# pokeagent-speedrun — agent context

This repo contains a game-playing agent harness. Originally built for Pokemon Emerald/Red, now generalized to play arbitrary browser-based games (e.g. itch.io game jams).

## Branches

- `main` — stable Pokemon harness
- `dev` — Pokemon multi-game development (red + emerald)
- `feature/generalized-harness` — **current branch**: adds browser game support via Playwright

## High-level architecture

```
run.py (orchestrator)
  ├── starts server subprocess (server/app.py)
  │     ├── pokemon_env/emulator.py     (mGBA, when --game=red|emerald)
  │     ├── pokemon_red_env/...         (Game Boy)
  │     └── browser_env/browser_emulator.py  (Playwright, when --game=browser)
  └── starts agent (agents/PokeAgent.py or agents/browser_game_agent.py)
        └── calls /mcp/* HTTP tools on the server
```

The server is FastAPI. It exposes `/mcp/*` endpoints that the agent calls via HTTP. This is the same harness pattern across emulator and browser games — the only difference is the env class behind `env`.

## Browser game support (this branch)

### Components

- **`browser_env/browser_emulator.py`** — `BrowserEnv` class. Wraps Playwright in a dedicated background thread (greenlets need thread affinity). Handles itch.io embed flow: navigate → click "Run game" → switch into `iframe#game_drop` → find `<canvas>` → focus. Public methods: `get_screenshot`, `press_key`, `press_keys_sequence`, `hold_key`, `click_at`, `double_click_at`, `get_page_text`, `get_game_info`, `stop`.

- **`agents/browser_game_agent.py`** — `BrowserGameAgent`. Auto-evolve style agent modeled after `PokeAgent`'s `autoevolve` scaffold but stripped of Pokemon-specific logic. Reuses `VLM`, `SubagentExecutor`, `HarnessEvolver`, stores. Tools: `press_keys`, `mouse_click`, `double_click`, `hold_key`, `process_memory`, `process_skill`, `run_skill`, `run_code`, `process_subagent`, `execute_custom_subagent`, `replan_objectives`, `evolve_harness`.

- **`agents/prompts/browser-game-directives/`** — `SYSTEM_PROMPT.md` (full harness instructions) + `BASE_ORCHESTRATOR_POLICY.md` (evolvable seed prompt).

- **`server/app.py` browser endpoints** — `/mcp/press_keys`, `/mcp/mouse_click`, `/mcp/double_click`, `/mcp/hold_key`. The existing `/mcp/get_game_state` branches on `game_type == "browser"` to return screenshot + page text + canvas dimensions + last action info. Pokemon-specific endpoints (`/state`, `/milestones`, `/screenshot`, `/status`) are also branched to no-op gracefully for browser games.

- **`run.py`** — added `--game browser`, `--game-url`, `--headed`, `--max-steps`. Auto-sets scaffold to `browser_autoevolve` and enables prompt optimization when `--game=browser`. Polls `/health` for up to 60s when starting browser server (Playwright init takes ~10-15s).

### Running

```bash
# Set up
uv sync
uv run playwright install chromium

# Add GEMINI_API_KEY to .env

# Run
./run_browser.sh
# or
.venv/bin/python run.py --game browser \
  --game-url "https://ravernt.itch.io/folder-dungeon" \
  --max-steps 50

# Watch live: http://localhost:8000/stream
```

### Default model
`gemini-3-flash-preview` (set in `run.py` argparse default).

## Run artifacts

Each run creates `run_data/run_<timestamp>/`:
- `screenshots/step_NNNN.png` — saved per step
- `prompt_evolution/llm_traces/llm_log.jsonl` — pointer to LLM logs
- `agent_scratch_space/`, `end_state/` — internal state

LLM call traces go to `llm_logs/llm_log_<timestamp>.jsonl` (full prompt + response + duration + tokens). Use this for debugging step durations and detecting hangs.

`run_browser.sh` also writes a tee'd console log to `run_data/browser_logs/<timestamp>.log`.

## Critical bugs that have been fixed (do not reintroduce)

1. **Playwright thread affinity** — `BrowserEnv` runs Playwright in its own thread and dispatches via queue. Don't call Playwright methods from FastAPI handler threads directly.

2. **Text-only fallback in Gemini backend was sending image-referencing prompts without the image, causing 180s hangs.** Removed the fallback in `vlm_backends.py` line ~2438 (`get_query` for Gemini, after image error). Do NOT re-add it. If the image query errors, the caller's outer retry handles it.

3. **`process_trajectory_history` is excluded from browser agent tools** because it runs without screenshots and the VLM hallucinates. The agent has the screenshot in its main prompt every step anyway.

4. **`get_comprehensive_state`, `get_milestones`, `get_status`, `get_screenshot` endpoints** are guarded with `if game_type == "browser"` returning safe defaults — `BrowserEnv` doesn't have these emulator methods.

5. **`save_state` checkpoint** is guarded with `hasattr(env, "save_state")` — browser games skip checkpointing.

6. **Game loading wait** — `BrowserEnv.initialize` polls screenshots for stable brightness for up to 30s before returning, so the agent never sees the Unity splash screen.

7. **Canvas size mismatch / flicker** — viewport is 960x600 with `device_scale_factor=1`. Don't resize the page after navigation (it triggers reload).

8. **Frame streaming for browser games** — `update_frame_cache(screenshot)` is called in browser MCP endpoints. The 50% frame skip (`% 2`) is bypassed for browser games since each agent step produces only one frame.

## Timeouts

- Gemini SDK timeout: 60s for stable models, 90s for `*-preview`, 180s for `gemini-3-pro`. (`vlm_backends.py` `_call_generate_content`)
- Agent VLM call timeout: 60s (`browser_game_agent.py` `run_step`)
- These are coordinated so the agent retry fires before the SDK gives up.

## Open items / next work

- **Local LLM backend**: Benchmark and integrate Gemma 4 (26B MoE / 31B dense) on a 5090 to compare tok/s vs Gemini API. Want a local backend in `utils/agent_infrastructure/vlm_backends.py` (probably vLLM-based) so the agent can run on local Gemma instead of Gemini API.

- **Agent quality**: The agent is currently spending too many steps on `replan_objectives` and not enough on actual game progression. The auto-evolve loop may help but the seed prompt could be tightened.

- **More games**: Currently tested against `https://ravernt.itch.io/folder-dungeon` (Brackeys 10 game jam). The harness should work for any HTML5 itch.io game.
