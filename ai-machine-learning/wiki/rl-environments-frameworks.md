# RL Environment Frameworks for LLMs

Practical guide to building and scaling RL environments in the LLM era — covering the six major frameworks, three reference environments (Jupyter agent, Wordle, Desktop computer-use), and a framework-agnostic design methodology. By Adithya S K and Sergio Paniego (May 5, 2026).

Source: [The Ultimate Guide to RL Environments: Building and Scaling Them in the LLM Era](https://huggingface.co/spaces/AdithyaSK/rl-environments-guide) · [GitHub companion repo](https://github.com/adithya-s-k/RL_Envs_101)

Closely related: [[rl-training-systems]] (taxonomy E = {T, H, V, S, C}; OpenReward / ORS ecosystem context), [[cloud-agent-infrastructure]] (E2B sandboxes), [[agent-harness-engineering]] (harness engineering around rollouts), [[swe-agent-benchmarks]] (code execution environments).

---

## The Six Frameworks at a Glance

| Framework | Type | Tool syntax | Reward model | Deployable | Best for |
|-----------|------|-------------|--------------|------------|----------|
| **OpenEnv** (Meta) | HTTP / MCP | `@mcp.tool` | External | ✅ Docker / HF Space | Long-running sandboxes; MCP ecosystem |
| **ORS / OpenReward** (General Reasoning) | HTTP / REST+SSE | `@tool` + Pydantic | Per-tool-call | ✅ Docker / HF Space / OpenReward marketplace | Server-decided rewards; sharing envs publicly |
| **NeMo Gym** (NVIDIA) | HTTP / REST | `app.post()` | Post-episode `/verify` | ✅ Docker / HF Space | NVIDIA/Ray stack; LLM-as-judge or unit-test scoring |
| **Verifiers** (PrimeIntellect) | In-process | plain Python `def` | `Rubric` system | ⚙️ | Fast prototyping; bundled dataset + rubric |
| **SkyRL Gym** (NovaSky-AI) | In-process | inside `step()` | `step()` returns | ⚙️ | Gym-style RL; SkyRL training stack |
| **GEM** (Axon-RL) | In-process | inside `step()` | `step()` returns | ⚙️ | Gymnasium API; `make_vec()` parallel rollouts |

**Key split:** HTTP frameworks (OpenEnv, ORS, NeMo Gym) wrap a remote server — they're deployable and language-agnostic. In-process frameworks (Verifiers, SkyRL, GEM) run the env class in the same Python process as the trainer — lower overhead, easier to iterate.

**Rule of thumb:** prototype in Verifiers (fastest), productionize in OpenEnv or ORS (deployable).

---

## The Three Reference Environments

The repo uses the same three environments reimplemented across all six frameworks — a "Rosetta stone" approach so you can compare how each framework handles the same logic.

### 1. Wordle Solver (multi-turn, deterministic, no external backend)

The cleanest entry point: pure Python, one tool (`guess(word)`), persistent state across turns, no external services. Every concurrent rollout needs its own isolated `WordleGame` instance — simple session management. Used as a cross-domain proof that the same training and rollout patterns work without modification.

- Deployed HTTP variants: [OpenEnv](https://huggingface.co/spaces/AdithyaSK/wordle-openenv) · [ORS](https://huggingface.co/spaces/AdithyaSK/wordle-ors) · [NeMo Gym](https://huggingface.co/spaces/AdithyaSK/wordle-nemo-gym)

### 2. Jupyter Agent (multi-turn, real code execution, external backend)

The model writes and executes Python in a real Jupyter kernel inside an E2B cloud sandbox. Four tools: `add_and_execute_code_cell`, `edit_and_execute_current_cell`, `execute_shell_command`, `get_notebook_state`. Key complexity: persistent state across turns, real external backend, session isolation per rollout.

- Deployed HTTP variants: [OpenEnv](https://huggingface.co/spaces/AdithyaSK/jupyter-agent-openenv) · [ORS](https://huggingface.co/spaces/AdithyaSK/jupyter-agent-ors) · [NeMo Gym](https://huggingface.co/spaces/AdithyaSK/jupyter-agent-nemo-gym)

### 3. Desktop Computer-Use (multi-turn, vision-driven, full Linux VM)

The model sees a screenshot of a full Linux desktop and drives mouse/keyboard with tool calls. 19 tools mirroring Anthropic's `computer_20251124` schema (screenshot, clicks, scroll, type, key, hold_key, wait, terminate, run_command, cursor_position, get_screen_size). Coordinates are `[x, y]` pixel arrays — compatible with OpenAI Operator and Qwen3-VL output with minimal adaptation.

Screenshots return as **MCP image blocks** (model sees pixels, not base64 text). Terminal reward via `terminate(status)`.

> Caveat from rollout testing: Qwen3-VL emits coordinates outside the configured display (e.g. y≈965 in a 768-px screen), suggesting an internal normalized scale — a rescaling adapter is needed before training.

- Deployed HTTP variants: [OpenEnv](https://huggingface.co/spaces/AdithyaSK/desktop-openenv) · [ORS](https://huggingface.co/spaces/AdithyaSK/desktop-ors)

---

## How to Build an RL Environment (Framework-Agnostic)

Before picking a framework, answer eight structural questions and make four key decisions.

### Step 1: Write the loop in 10 lines

1. What is the model trying to do?
2. What can it DO? (tools/actions)
3. What does it SEE back? (observation format)
4. When is it done? (termination condition)
5. How do you score it? (reward sketch)

If you can't write this in 10 lines, you have an idea, not an environment.

### Step 2: Identify the Eight Components

| Component | Question | Decide before coding |
|-----------|----------|----------------------|
| **Tasks / Dataset** | What problems should the model solve? | List 5–10 example tasks by hand |
| **Prompt template** | How is the task presented? | Write the system + user prompt |
| **Tools / Actions** | What can the model DO? | Sketch function signatures |
| **Observations** | What does the model SEE back? | Raw string? Structured JSON? Image? |
| **Execution backend** | Where do actions actually run? | Sandbox? In-process Python? None? |
| **State** | What persists across turns? | Session-scoped dict? File system? |
| **Reward / Rubric** | How is success measured? | Exact match? LLM-as-judge? Unit tests? |
| **Termination** | When does it end? | Max turns? `done` from a tool? |

Picking a framework before writing these down is putting the cart before the horse.

### Step 3: Make Four Key Decisions

**Decision A: In-process or HTTP server?**

| Factor | In-process if… | HTTP server if… |
|--------|---------------|-----------------|
| Backend | Pure Python logic | Sandbox / Docker / external service |
| Scale | <100 parallel rollouts | 100s–1000s concurrent sessions |
| Iteration | Prototyping | Production deployment |
| Languages | Python only | Mixed language env |

Start in-process. Move to HTTP only when you outgrow it.

**Decision B: Single-turn or multi-turn?**

Single-turn = one prompt → one response → score. Much simpler. If you can frame your task as single-turn, do it. Multi-turn requires state persistence and a decision about who controls the loop (trainer, framework, or env).

**Decision C: Where does reward come from?**

| Pattern | When to use | Frameworks |
|---------|-------------|------------|
| External (training script from final output) | Reward depends on full trajectory | OpenEnv, Verifiers, SkyRL, GEM |
| Per tool call (env returns reward with each action) | Each step independently scoreable | ORS |
| Post-episode `/verify` (separate endpoint) | Holistic LLM-as-judge or unit-test | NeMo Gym |

If unsure: start with **external**. Most flexible, easiest to debug.

**Decision D: Stateless or stateful tools?**

Stateless tools (`add(a, b)`) need no session management. Stateful tools (`run_code(...)` in a Jupyter kernel) need isolated state per concurrent rollout. If your tools are stateful, plan for session management early — it will dominate half the engineering effort.

### Step 4: Match Framework to Decisions

| Decided… | Best match |
|----------|-----------|
| In-process + bundled dataset + rubric | **Verifiers** |
| In-process + Gymnasium API + parallel `make_vec()` | **GEM** |
| In-process + Gym-style + SkyRL training stack | **SkyRL Gym** |
| HTTP + MCP / community + HF Spaces | **OpenEnv** |
| HTTP + per-call rewards + OpenReward marketplace | **ORS** |
| HTTP + post-episode verify + NVIDIA stack | **NeMo Gym** |

### Step 5: Build the Smallest Possible Version First

1. One task. Hardcoded.
2. One tool. Even if the real env has ten.
3. No reward. Just print "got result: X".
4. One rollout. With a known model (`Qwen3-4B`), no training.

Get that working end-to-end, then layer in: more tasks, more tools, real rewards, batching, async, deployment.

### Step 6: Validate with Rollouts, Not Training

Training is a slow, expensive way to discover your environment is broken. Before any training run:

- Manually call `env.reset()`, call each tool, then `env.close()`
- Run a single LLM rollout and **read the trajectory by hand** — did the model see what you expected? Did tool returns make sense? Did reward fire correctly?
- If a human can't read the trajectory and tell whether the model did well, neither can a reward function

The biggest mistakes in RL env design are caught by reading 5 trajectories. They will not be caught by 1000 training steps.

---

## Common Pitfalls

- **Reward too sparse.** Every rollout returns 0.0; GRPO has no signal. Fix: design partial credit, or pick easier tasks for the smoke test.
- **Reward too dense or leaky.** Model gets reward for behaviors that don't generalize (shortcuts). Fix: read trajectories and look for exploits.
- **Tasks too easy.** Model solves in one tool call; no learning signal in multi-turn settings.
- **Tools too powerful.** One tool solves everything; no exploration and no interesting behavior.
- **State leaks across rollouts.** Same sandbox or dict reused without reset; episodes contaminate each other.
- **No timeout or max turns.** A buggy model loops forever and stalls training.
- **Observation format the model can't parse.** Huge JSON dumps or stack traces longer than the context window.

---

## Key Observations on Framework Comparison

After implementing all three reference environments across all six frameworks, the patterns that emerge:

- **MCP is the right abstraction for HTTP envs.** OpenEnv's `MCPToolClient` auto-discovers tools via `list_tools()` and converts them to OpenAI schemas — no env-specific package installs for the rollout client.
- **ORS per-call rewards are architecturally interesting** but require that every tool call in isolation be meaningful to score. Most envs don't have this property — terminal reward (pass/fail at episode end) is more natural.
- **NeMo Gym's Ray dependency** is a recurring ops problem on shared cluster nodes where `gcs_server` can't bind. Local development works only against the deployed HF Space.
- **Verifiers is the fastest for prototyping** because it's entirely in-process and auto-generates tool schemas from function signatures + docstrings via `inspect`. No server, no containerization, no MCP wiring.
- **Tag-based action parsing** (SkyRL, GEM) trades structured tool-calling for free-text generation with XML-style tags (`<guess>word</guess>`, `<click x="100" y="200"/>`). This matches how models generate text naturally and can simplify rollout engineering, at the cost of parse robustness.
- **In-process vs HTTP is the load-bearing decision.** The six frameworks are really two clusters with different engineering profiles — not six equal choices.

---

## Agent Skills (`.claude/skills/`)

The repo ships 5 agent skills written to the [SKILL.md spec](https://github.com/anthropics/skills):

| Skill | What it builds |
|-------|----------------|
| `rl-env-from-description` | Orchestrator — interview, archetype selection, shared domain module, all 4 framework variants, smoke-test rollouts |
| `generate-openenv-env` | OpenEnv (Meta) MCP variant |
| `generate-ors-env` | OpenReward (ORS) per-call-reward variant |
| `generate-verifiers-env` | Verifiers (PrimeIntellect) in-process variant |
| `generate-nemo-gym-env` | NeMo Gym (NVIDIA) Resources Server variant |

Install into any project: `npx skills add adithya-s-k/RL_Envs_101`

These skills work with Claude Code, Cursor, Codex, OpenCode, Gemini CLI, and any spec-compliant agent. They're folder-agnostic and ask where you want files written.

---

## Framework Links

- [OpenEnv](https://github.com/meta-pytorch/OpenEnv) (Meta)
- [ORS / OpenReward](https://openrewardstandard.io/) (General Reasoning)
- [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) (NVIDIA)
- [Verifiers](https://github.com/PrimeIntellect-ai/verifiers) (PrimeIntellect)
- [SkyRL Gym](https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-gym) (NovaSky-AI)
- [GEM](https://github.com/axon-rl/gem) (Axon-RL)

## Sources

- [The Ultimate Guide to RL Environments: Building and Scaling Them in the LLM Era](https://huggingface.co/spaces/AdithyaSK/rl-environments-guide) — Adithya S K & Sergio Paniego (May 5, 2026)
- [RL_Envs_101 GitHub repo](https://github.com/adithya-s-k/RL_Envs_101) — companion code with runnable implementations across all 6 frameworks
- [openreward Python package](https://pypi.org/project/openreward/) — ORS client
- [OpenReward Standard](https://openrewardstandard.io/) — ORS specification
