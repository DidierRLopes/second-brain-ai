# Scaling Long-Running Autonomous Coding (Cursor)

**Source:** https://cursor.com/blog/scaling-agents
**Author:** Wilson Lin
**Published:** January 14, 2026
**Filed under:** Research
**Companion:** github.com/wilsonzlin/fastrender (the browser project), with related Java LSP, Windows 7 emulator, and Excel projects

## Premise

Cursor has been running coding agents autonomously for *weeks* to see how far agentic coding can be pushed for projects that would take human teams months. The post reports on running hundreds of concurrent agents on a single project — over a million lines of code, trillions of tokens — and what they learned about coordination.

## The Core Problem

Single agents are good at focused tasks but slow on complex projects. The natural move is to run them in parallel. The hard part is coordinating them. The post is a tour through Cursor's iterations on that coordination problem.

## Iteration 1: Flat Self-Coordination via Shared File

First attempt: peer agents with equal status, coordinating through a shared file. Each agent checks what others are doing, claims a task, updates its status. Locking prevents two agents from grabbing the same task.

Failure modes:

1. **Lock-holding pathologies.** Agents held locks too long or forgot to release them. Even when locking worked, it became a throughput bottleneck — twenty agents would slow to the effective throughput of two or three, with most time spent waiting.
2. **Brittleness.** Agents could fail while holding locks, try to acquire locks they already held, or update the coordination file without acquiring the lock at all.

## Iteration 2: Optimistic Concurrency Control

Replaced locks with OCC: agents read state freely, writes fail if state changed since the last read. Simpler and more robust, but a deeper problem surfaced.

**With no hierarchy, agents became risk-averse.** They avoided difficult tasks and made small, safe changes. No agent took responsibility for hard problems or end-to-end implementation. Work churned for long periods without progress.

## Iteration 3: Planners + Workers + Judge

Separate roles into a pipeline:

- **Planners** continuously explore the codebase and create tasks. They can spawn sub-planners for specific areas, making planning itself parallel and recursive.
- **Workers** pick up tasks and focus entirely on completing them. They don't coordinate with other workers or worry about the big picture. They just grind on their assigned task until it's done, then push their changes.
- **Judge agent** at the end of each cycle determines whether to continue. The next iteration starts fresh.

This solved most of the coordination problems and let the system scale to very large projects without any single agent getting tunnel vision.

## Results — Concrete Projects

- **Browser from scratch (`fastrender`):** ~1 week of agent time, >1M lines of code, ~1,000 files. Hundreds of workers running concurrently, pushing to the same branch with minimal conflicts. New agents joining can still understand the codebase and make meaningful progress.
- **Solid → React migration in the Cursor codebase:** ~3 weeks, +266K / −193K edits. Passed CI and early checks, still needs careful review.
- **Video rendering rewrite:** A long-running agent made video rendering 25× faster with an efficient Rust version, plus zoom/pan with spring transitions and motion blur. Code merged, going to production.
- **Other ongoing experiments (still running at time of writing):**
  - Java LSP: 7.4K commits, 550K LoC
  - Windows 7 emulator: 14.6K commits, 1.2M LoC
  - Excel: 12K commits, 1.6M LoC

## What They Learned

### Model choice matters for extreme long-running tasks

- **GPT-5.2** is much better at extended autonomous work: following instructions, keeping focus, avoiding drift, implementing things precisely and completely.
- **Opus 4.5** tends to stop earlier and take shortcuts when convenient, yielding back control quickly.
- **Different models for different roles.** GPT-5.2 is a better *planner* than GPT-5.1-Codex, even though the latter is trained specifically for coding. Cursor now uses the model best suited for each role rather than one universal model.

### Removing complexity beat adding it

They initially built an **integrator** role for quality control and conflict resolution. It created more bottlenecks than it solved — workers were already capable of handling conflicts themselves. Killing the integrator role was a net win.

### The best system is often simpler than expected

Cursor initially tried to import patterns from distributed computing and organizational design. Not all of them transfer to agents. The right amount of structure is *in the middle*: too little, and agents conflict / duplicate / drift; too much, and the system becomes fragile.

### Prompts > harness > models

A surprising amount of the system's behavior comes down to how the agents are prompted. Getting them to coordinate well, avoid pathological behaviors, and maintain focus over long periods required extensive experimentation. The harness and models matter, but **the prompts matter more.**

## Open Problems

- Planners should wake up when their tasks complete to plan the next step (currently they don't).
- Agents occasionally run for far too long.
- Periodic fresh starts are still required to combat drift and tunnel vision.

## Bottom Line

The optimistic answer to "can we scale autonomous coding by throwing more agents at a problem?" is yes — hundreds of agents can work together on a single codebase for weeks and make real progress on ambitious projects. The system isn't perfectly efficient but it's far more effective than the team expected.
