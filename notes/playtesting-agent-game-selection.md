# Playtesting agent — game selection brainstorm

> Captured from a chat session on 2026-04-06 so we can pick this up later.
> The framing: this harness isn't really an "AI plays games" benchmark — it's
> a **playtesting agent for indie game devs**. That recasts every game-choice
> question.

## The reframe

The wrong question: *"What's the hardest game we can solve?"*
The right question: *"If a fresh player drops into this game cold, what
happens, and what would the dev want to know about that experience?"*

What a playtesting agent actually answers:

| Question a dev asks | What the agent reveals |
|---|---|
| Can a fresh player figure out the controls in 5 min? | Memory entries about controls in steps 1–100 |
| Where does the player get stuck? | Repeated identical actions, replan_objectives loops |
| Does the tutorial teach the right things? | Compare what the agent *learned* vs what the tutorial *showed* |
| What does the player think the game IS? | The agent's own description in objectives & memory |
| Are there UX dead-ends I didn't notice? | Agent traps in popups, modal dialogs, wrong UI states |
| Do the mechanics emerge or do you have to be told? | Whether the agent discovers them vs needs the dev to explain |

The Folder Dungeon run already produced a real signal on three of these.
The agent's evolved skills (`force_close_window_sequence`,
`interact_with_file`) are *literally* a playtest report saying "your popups
trap players and your icon hitboxes are unclear." Real data points an indie
dev could act on.

## What we explicitly do NOT want

- **Sokoban / Picross / "AI olympics" puzzles.** No indie dev needs help
  knowing whether their puzzle is solvable — they made it. These are pure
  AI benchmarks with zero dev relevance.
- **Twitch platformers (Fancy Pants, Spelunky, N+, agar.io, Robot Unicorn
  Attack).** Our agent runs at 2.5 s/step on gemma4:26b (or 13s on Gemini).
  By the time the screenshot arrives the player character is already dead.
  *Maybe* worth one run on Fancy Pants just to *measure* the failure mode
  ("agent loses 3 lives in 10 steps" is itself a playtesting-limit data
  point), but not as a recurring benchmark.
- **Modern 3D Unity games.** Our Xvfb stack uses ANGLE+SwiftShader software
  rasterization. WebGL2 *works* but big 3D scenes are too slow to render.
- **Multiplayer / login-required.** Out of scope for reproducible runs.
- **Real Flash without Ruffle.** Dead in modern browsers. Ruffle ports of
  classic Flash games are fine, when they work.

## Candidate games — by what they stress for a playtester

### Tier 1: Discoverability (does the player figure out the rules?)

1. **Hempuli's PuzzleScript prototypes** — `hempuli.com/baba/` and similar.
   Arvi Teikari (Baba Is You's creator) ships rule-discovery prototypes for
   free in the browser. PuzzleScript runs in pure HTML5 canvas, simple
   keyboard controls (4 arrows + Z + R). "No tutorial, figure it out." A
   playtesting agent that solves even 2–3 levels is *literally* what Hempuli
   would want to see — would tell him whether his rule-discovery hooks work
   for someone seeing them fresh.

2. **Baba Is You demo** — official browser demo. Same family but with the
   rule-pushing meta-mechanic. The "rules ARE blocks you can manipulate" idea
   is the canonical example of a game where playtesters tell you whether
   your central mechanic is communicable. Real research interest *and* real
   game-dev relevance. Caveat: Baba is well-studied as an AI benchmark
   already, so a paper on it would be the 50th — but as a *playtest* angle
   it's still novel.

### Tier 2: Real indie targets (devs who would actually use a playtest report)

3. **Sokpop games** — `sokpop.co`, also on itch. Sokpop releases a new tiny
   game every two weeks for years. Each one is unpolished by design, has
   weird controls, and the team famously says they don't have time to
   playtest internally. Pick 2–3 recent ones (e.g. *Bun Bun*, *skiii*,
   *Spaghetti*) and run the agent on each. **If we ever publish results,
   these are devs who would actually retweet it.** Small enough that 5000
   steps is enough to see real signal. Not all Sokpop games are browser-
   playable — need to verify per-game.

4. **Brackeys / GMTK / LD jam winners.** Jam games are made in 48–72 hours
   by people who couldn't possibly playtest them. Folder Dungeon is from
   Brackeys 10. The Brackeys 13 winners (most recent jam) would let us run
   the playtest agent across **5 comparable games from one cohort**, which
   is a much stronger story than five disconnected games.

   Best jams to mine:
   | Jam | Cadence | Volume | Quality |
   |---|---|---|---|
   | **GMTK Jam** | annual | ~5000 entries | highest. Mark Brown's audience is hardcore game-dev nerds. |
   | **Brackeys Jam** | biannual | ~2000 entries | very high. Folder Dungeon is from Brackeys 10. |
   | **Ludum Dare** | 3×/yr | ~1500 entries | variable but the top 100 are famous. |
   | **js13kgames** | annual | ~250 entries | niche, JS-only, 13kb max — browser-perfect. |
   | **GitHub Game Off** | annual Nov | ~500 entries | smaller, repo-driven. |

### Tier 3: Genre-specific dev pain points

5. **A Bitsy narrative game** — Bitsy is the most popular tool for tiny
   narrative games on itch. Tests dialogue navigation, choice navigation,
   and most importantly *does the VLM actually read the text?* Devs who use
   Bitsy worry constantly about whether players read their writing or just
   mash through it. The agent's behaviour would directly answer that.

6. **A 5-minute idle game** (Cookie Clicker / Universal Paperclips style).
   Specifically tests *upgrade tree discoverability*. Indie idle-game devs
   ship these all the time and the #1 complaint is "players don't notice
   the prestige mechanic until step 200". A playtesting agent that ignores
   half the upgrade tree for 100 steps is exactly what they need to know.

### Bonus: Flash games via Ruffle that ARE good fits

Most Flash games are too twitchy, but the *point-and-click* Flash genre is
perfect — slow, visual, puzzle-driven, no time pressure.

- **Submachine series** (Mateusz Skutnik) — pure point-and-click escape rooms.
  Famously inscrutable. *Submachine 1: The Basement* and *Submachine 2: The
  Lighthouse* are both on itch.io with Ruffle. Real game devs making escape
  rooms would love to know whether the agent figures out that the rusty key
  in room 3 unlocks the panel in room 7.
- **Bonte puzzle games** (`bartbonte.com/games`) — Bart Bonte makes
  one-mechanic color/click puzzles. *Yellow*, *Red*, *Blue*, *Green*,
  *Orange*, *Pink* — series of 30–50 levels each. Tests pure visual reasoning
  + hypothesis generation. Each game has zero tutorial; you have to figure
  out the mechanic from the title and the screen.
- **Achievement Unlocked** (jmtb02) — meta-game about getting achievements.
  The whole game is "can you discover the 100 things to do?" which is
  *literally a playtest*.
- **Sushi Cat** — physics puzzle, drop a cat through pegs to eat sushi. No
  twitch, infinite time per shot. Could be a good gentle test.
- **Kingdom Rush / Bloons TD** — pausable tower defense. AS3 so Ruffle
  support is iffier; would need to verify they actually load.

## Recommended first session

If the goal is "show this is useful for playtesting indie games", I'd run,
in this order:

1. **Hempuli's PuzzleScript prototype** — canonical rule-discovery test. If
   this works at all it's already publishable.
2. **One Sokpop game** (recent + browser-playable) — tests against a *real*
   shipping indie game whose dev wouldn't dismiss it.
3. **A Bitsy narrative game** — totally different genre, tests reading +
   decision-making.
4. **Folder Dungeon (existing baseline)** — keep for comparison.

Three new games × 5000 steps each ≈ 10 hours of agent time on gemma4:26b
(at 2.5 s/step). We get a data point about three completely different player
experiences.

Optionally: one **Submachine** run as the "Flash escape room" representative.

## What needs to be built before a real session

A **playtest report deliverable**. After each run, dump a markdown file with:

- Summary stats (steps, total time, tool calls breakdown)
- "What the agent learned" — top memory entries by importance
- "What the agent built" — evolved skills + subagents (and crucially,
  whether they got USED)
- "Where the agent got stuck" — repeat-action sequences, replan loops,
  same-screenshot streaks
- "What the agent thinks the game is" — the orchestrator policy after
  evolution
- "Onboarding curve" — time-to-first-meaningful-action, time-to-first-skill,
  time-to-first-objective-completion
- 5–10 representative screenshots with the action that was taken

This is the actual *product* of the playtesting agent. Without it the runs
are just "the agent did stuff for 3 hours" which is useless for a dev.

## Verification checklist (run before committing to a real session)

For each candidate game, three checks:

1. **Does it actually load in our Xvfb + ANGLE/SwiftShader stack?**
   Some Unity WebGL games crash on software-rasterized WebGL2.
2. **Is the canvas size something the agent can read?**
   Some games have tiny canvases or weird DPI scaling.
3. **Is there a "click to start" gate beyond itch.io's "Run game" button?**
   Some games have a second "click anywhere to begin" overlay that may
   trip up our init code.

A 1-step "does it boot" test takes ~30s per game and is much cheaper than
discovering a problem 200 steps into a 5000-step run.

## Open questions for next session

1. Pick a specific jam to mine? Brackeys 13 vs GMTK 2024 vs Ludum Dare 56.
   Lean Brackeys 13 since we already have Brackeys 10 as a comparison
   point.
2. Is the playtest report a deliverable I should design now? If so, lock
   down the schema before any new runs so they collect the right data.
3. Which Sokpop game first? They have ~150 games — need to pull their itch
   page and pick. Verify each is browser-playable and not too action-y.
4. Are there any specific games the *user* has been wanting to test that
   I should add to this list? (Answer pending — the user mentioned Fancy
   Pants and game jam games but said Fancy Pants only as a question.)

## Active findings about the harness, not the games (as of run_20260406_181137 ~step 1285)

These are about the harness itself, not game selection, but they're relevant
because they affect every game we benchmark on. They block "useful playtest
reports" until fixed.

- **`run_skill` is called ZERO times** despite the autoevolver building
  89 executable skills. The whole skill machinery is dead weight in the
  current orchestrator prompt. Either the prompt doesn't push hard enough
  toward `run_skill`, the skill tree overview hides them, or the skills
  are being created with descriptions the orchestrator can't match against
  its current objective. Investigate before any real benchmark.
- **`execute_custom_subagent` is called ZERO times** despite the
  autoevolver building 90 subagents. Same root cause likely.
- **`get_game_state` is called 2.4 times per step on average** (2 automatic
  calls in the agent loop + ~0.4 LLM-emitted calls). The screenshot is
  already in the prompt every step, so the LLM-emitted ones are pure waste.
  And one of the two automatic calls is just to extract `game_info` for
  trajectory logging — should be combined with the prompt-building call.
  Each redundant call costs a Playwright screenshot (~100-200 ms) plus
  ~500 KB of base64 over HTTP.

The dead skills/subagents are the most important finding because they
mean **the autoevolve loop is currently writing artifacts no one reads**.
Until that's fixed, the playtesting agent's output is just trajectory +
memory entries — the skill library is decorative.
