# Cloud Agent Infrastructure

Cloud agent infrastructure is the compute, isolation, state, and orchestration layer that lets AI agents operate securely and autonomously in production. It is distinct from the [[agent-harness-engineering|harness]] (the scaffolding around the model) and from the model itself. The foundational challenge is that agents run arbitrary code, operate across async gaps (minutes or days), and need to scale to hundreds or thousands of concurrent sessions — requirements that the infrastructure cloud originally designed for web workloads does not meet.

Two reference accounts define the space: Cognition's post-mortem on building Devin's cloud infrastructure (Apr 2026), and Han Lee's "Hidden Technical Debt of AI Systems: Agent Runtime" (Apr 2026). Stripe's "Minions" case study (Feb 2026) is the production-at-scale data point.

## Stripe Minions: Production Cloud Agents at Scale

Stripe's "Minions" are homegrown coding agents built by the internal Leverage team, responsible for more than **1,000 pull requests merged per week**. Though humans review every PR, Minions write code from start to finish — one-shot, end-to-end. The design is deliberately "one-shot": the agent takes a task and completes it without iterative back-and-forth with the requester. Stripe backs these agents with an internal MCP server containing **400+ tools** to keep agents connected to CI, monitoring, package registries, documentation, and source control. The Minions case study is the clearest public proof that end-to-end autonomous coding agents can work at enterprise scale without full autonomy — human review remains the acceptance gate.

Key design choice: one-shot + human review at the end, not human-in-the-loop throughout. This maximizes agent autonomy while keeping a hard quality gate, and avoids the "interrupt-driven agent" failure mode where constant check-ins fragment the agent's context.

## Why Containers Are Not Enough

The natural starting point for cloud agents — containerize a CLI agent and give it repo access — hits three structural problems immediately.

**Shared kernel = security threat.** Containers share the host kernel. Agents generate their own code, run arbitrary shell commands, and probe the environment in unpredictable ways. A kernel-level escape from one session reaches every other container's filesystem, credentials, and network connections. For agents that execute attacker-controlled instructions embedded in tool outputs (web pages, PDFs, emails), this is not a theoretical risk — it is the working assumption. The industry consensus for untrusted code is VM-level isolation, where each workload gets its own kernel with no shared attack surface.

**Containers cannot survive async gaps.** Real engineering work has gaps: open a PR, wait on CI, respond to a review comment, rerun tests, push a follow-up commit. Between each step there are minutes, hours, sometimes days. A containerized agent can only survive these gaps by burning compute to stay alive; if the container is rescheduled or times out, the session is lost. This makes containerized agents suitable only for bounded, single-pass tasks like dependency upgrades, not for work that spans the full SDLC.

**Scaling to hundreds of sessions requires purpose-built orchestration.** Each agent session is unique — tied to a specific task and the dispatching engineer's permissions. Running hundreds concurrently requires provisioning the right environment per session, routing correctly, predicting demand to keep warm VM pools ready, and keeping every environment current as codebases change daily. This is its own multi-quarter engineering project.

Cognition's conclusion, after over a year of hypervisor engineering for Devin: VM-level isolation (microVMs) is mandatory, and hypervisor-level snapshotting of full machine state — memory, process trees, filesystem — is what lets agents survive async gaps. Compute shuts down while idle; sessions resume exactly where they left off when a CI result or review comment arrives. Building this reliably across thousands of concurrent sessions took longer than any other piece of infrastructure Cognition had shipped.

## The Isolation Primitive Stack

Five primitives are in serious use:

| Primitive | Isolation model | Cold start | Workload fit |
|-----------|----------------|------------|--------------|
| **Linux containers (runc)** | Shared host kernel, namespaces + cgroups + seccomp | ~100ms | Trusted code, internal CI |
| **Firecracker** | KVM-based microVM, dedicated kernel per VM | ~125ms boot, sub-second from snapshot | Untrusted code at high density |
| **gVisor** | Userspace kernel intercepting syscalls | Container-class | Defense in depth without full VM cost |
| **Kata Containers** | Lightweight VM per pod, OCI-compatible | Few hundred ms | Multi-tenant Kubernetes |
| **V8 isolates** | Per-tenant JS heap inside a single process | Sub-millisecond | JavaScript-only |

**Firecracker is the de facto standard for agent sandboxes.** AWS open-sourced it in 2018 to power Lambda and Fargate. It boots a stripped-down Linux kernel inside KVM in ~125ms with ~5MB VMM memory. Almost all agent-sandbox startups (E2B, Fly.io, Vercel Sandbox) run on top of it. The combination of strong isolation, fast boot, and high density makes thousands of concurrent rollouts economically viable.

**gVisor** (Google's userspace kernel) is the middle path — stronger than containers without the full VM cost. Google uses it for Cloud Run and App Engine.

**Containers are not a sandbox for agent code** — the shared kernel is the problem, not the packaging. The question is whether you've accepted the threat model, not whether you've "secured the container."

## The Sandbox-as-a-Service Landscape

A category of managed sandbox providers has emerged, all building on the primitives above:

| Vendor | Isolation | Snapshot model | Notable design |
|--------|-----------|---------------|----------------|
| **E2B** | Firecracker microVM | Full VM snapshot | Open-source SDK, popular in agent dev community |
| **Modal** | gVisor | Filesystem diffs | GPU support |
| **Daytona** | Containers/VMs | Forkable workspaces | OCI-compatible |
| **Cloudflare Workers Sandbox** | V8 isolates + containers | Object snapshots | Edge-first |
| **Vercel Sandbox** | Firecracker | Snapshot from build | Tied to Vercel deploy model |
| **AWS Bedrock AgentCore** | microVM-class | Per-session | Integrated with AWS data plane |
| **Azure Container Apps Dynamic Sessions** | Hyper-V | Sub-second | Strongest if already on Azure OpenAI |
| **GCP Cloud Run Sandboxes** | gVisor | Cloud Run-style | Most flexible hyperscaler offering |

The differentiation between vendors is developer ergonomics, snapshot model, and pre-wired tool mix — not the isolation primitive underneath.

**Reference implementation:** Ramp's background coding agent runs on Modal, with each session containing Postgres, Redis, Temporal, RabbitMQ, a VS Code server, and Chromium. Filesystem snapshots refresh every 30 minutes. A new session reaches prompt-ready state in seconds. Key lesson: agent productivity is bounded by runtime startup time, not model tokens-per-second.

## Experimentation vs. Production Runtimes

Training/eval and production have fundamentally different requirements:

| Dimension | Experimentation / Training | Production |
|-----------|--------------------------|------------|
| **Concurrency** | Thousands of parallel rollouts, bursty | One session per user, steady-state |
| **Cold start** | Critical (5s × 10k rollouts is real money) | Forgivable — users wait |
| **State model** | Fork, branch, replay, snapshot | Durable, per-user, auditable |
| **Network** | Often offline/recorded for determinism | Live internet, real APIs |
| **Failure model** | Drop the rollout, sample more | Retry, degrade, page someone |
| **Lifetime** | Seconds to minutes | Minutes to hours |

Optimizing the same runtime for both is how you get a system that's too slow for training and too brittle for production.

## The Runtime Shift Problem (Dev/Prod Parity)

Agents learn the runtime. Tool latencies, failure modes, shell quirks, filesystem layout, the exact way `ls` formats output — the model picks all of this up during training and bakes it into the policy. Move to a different runtime and behavior shifts silently: tools that were idempotent become flaky, commands that were instant now block.

This is **runtime shift** — a new flavor of distributional shift that no standard eval catches because the eval runs in the training runtime.

Three paths through it:
1. **Co-locate train and prod on the same runtime.** Tightest coupling, most predictable behavior.
2. **Define a runtime contract.** A small, versioned interface — shell semantics, tool schemas, failure modes — implemented on both sides. Harder than it sounds because latency and failure semantics matter, not just the API surface.
3. **Train against production noise.** Inject 5–10% tool errors during training (Step-DeepResearch). The policy becomes robust to runtime variance instead of dependent on a specific runtime.

The wrong answer — picked by default by most teams — is to treat the sandbox as a software engineering decision and ignore the ML requirements, then spend quarters chasing agent performance flakiness.

## The Three-Layer Orchestration Challenge (Cognition)

Running cloud agents at enterprise scale requires three things that each become multi-quarter engineering projects:

- **Orchestration:** Provisioning the right environment per session, routing sessions, predicting demand for warm VM pools, keeping provisioned environments current as codebases change daily.
- **Governance:** Each session inherits the dispatching engineer's permissions across every system it touches, with every action in a tamper-evident audit trail. Identity chaining, access scoping, and audit logging at enterprise scale.
- **Integrations:** Each tool (CI, monitoring, package registries, documentation, source control) has its own authentication model, permission scoping, and maintenance burden. Stripe's 400+ tool MCP server is the scale reference.

The pattern Cognition observed: the combined surface area is what becomes untenable. Not any single piece, but the fact that all three must be built, integrated, and maintained indefinitely. Their orchestration layer alone took over three quarters of dedicated engineering to build.

## Engineering Org Change Management

Building the infrastructure is only phase one. Phase two is transforming how engineers work with agents:

- **Engineer fluency:** Which work to delegate, how to define tasks precisely enough that agents execute without correction, managing concurrent sessions. Takes months of practice on real projects.
- **Planning and resource allocation:** Team sizing, sprint capacity, and project staffing assumptions all change when agent capacity enters the equation.
- **Review at volume:** The volume of code needing review increases dramatically; the review process designed for human-authored code doesn't transfer at higher volume.

Itaú (17,000 engineers, 11 months in): migrations 5-6× faster, 70% of static-analysis security vulnerabilities auto-remediated, test coverage doubled.

## Related Topics

- [[agent-harness-engineering]] — The scaffolding layer above the runtime: prompts, tools, context strategy, training vs. production harness asymmetry
- [[llm-agents]] — Agent architectures and the engineering patterns that turn models into agents
- [[swe-agent-benchmarks]] — Executable environments for training and evaluating coding agents
- [[rl-training-systems]] — RL training systems and the environment taxonomy (T, H, V, S, C) that training harnesses implement

## Sources

- "What We Learned Building Cloud Agents" — Cognition Team (Apr 23, 2026): https://cognition.ai/blog/what-we-learned-building-cloud-agents
- "Minions: Stripe's one-shot, end-to-end coding agents" — Alistair Gray, Stripe (Feb 9, 2026): https://stripe.dev/blog/minions-stripes-one-shot-end-to-end-coding-agents
- "Minions: Stripe's one-shot, end-to-end coding agents — Part 2" — Alistair Gray, Stripe (Feb 19, 2026): https://stripe.dev/blog/minions-stripes-one-shot-end-to-end-coding-agents-part-2
- "Hidden Technical Debt of AI Systems: Agent Runtime" — Han Lee (Apr 24, 2026): https://leehanchung.github.io/blogs/2026/04/24/hidden-technical-debt-agent-runtime/
- Firecracker microVM: https://firecracker-microvm.github.io/
- E2B: https://e2b.dev/
- Modal Sandboxes: https://modal.com/docs/guide/sandbox
