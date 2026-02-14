# DyTopo Multi-Agent Swarm Reference

## What is DyTopo and how does dynamic topology routing work?

<!-- Verified: 2026-02-13 -->

DyTopo (Dynamic Topology) is a multi-agent collaboration framework based on arXiv 2602.06039 that creates a team of specialized AI agents and routes messages between them using semantically-matched descriptors. The communication graph is rebuilt every round, adapting dynamically based on what agents actually produce and need. Each agent generates short text descriptors: keys describing what it can offer and queries describing what it needs. MiniLM-L6-v2 embeds these descriptors into 384-dimensional vectors on CPU, then a cosine similarity matrix determines which agents should communicate. Edges where similarity exceeds the routing threshold τ (tau) form a directed graph, which is cycle-broken by greedy removal of the lowest-weight edge per cycle and topologically sorted using Kahn's algorithm with alphabetical tiebreaking to establish a deterministic execution order. Round 1 is always a broadcast where all agents see all outputs; starting from round 2, the three-phase descriptor routing takes effect and agents receive only messages from their connected neighbors.

<!-- Related: DyTopo, dynamic topology, multi-agent, routing, arXiv 2602.06039, MiniLM, descriptors, cosine similarity, topological sort -->

## What agent teams are available in each DyTopo domain?

<!-- Verified: 2026-02-13 -->

DyTopo organizes agents into three predefined domains, each tailored to a category of problem. The **code** domain contains a Manager plus four workers: Developer, Researcher, Tester, and Designer — suited for code generation, debugging, algorithm design, software architecture, and code review tasks. The **math** domain contains a Manager plus three workers: ProblemParser, Solver, and Verifier — designed for proofs, numerical calculations, optimization problems, and multi-step mathematical reasoning where cross-verification catches errors. The **general** domain contains a Manager plus three workers: Analyst, Critic, and Synthesizer — intended for open-ended analysis, multi-perspective reasoning, evaluating trade-offs, and tasks requiring balanced critique and synthesis. Every domain includes a Manager agent responsible for defining round goals, deciding when to terminate (setting `terminate=true` when the team consensus is reached or after 3+ rounds without progress), and extracting the final solution from the collaborative output.

<!-- Related: domains, code, math, general, Manager, Developer, Researcher, Tester, Designer, ProblemParser, Solver, Verifier, Analyst, Critic, Synthesizer -->

## How does the three-phase round structure work in DyTopo?

<!-- Verified: 2026-02-13 -->

Starting from round 2, each DyTopo round executes three distinct phases. In **Phase A (descriptor generation)**, each agent generates short key and query descriptors using Qwen3-30B-A3B at temperature 0.1 with the `/no_think` flag and a 256-token output limit — this keeps descriptor generation fast, cheap, and deterministic. In **Phase B (topology routing)**, MiniLM-L6-v2 embeds all descriptors into 384-dimensional vectors on CPU, computes a cosine similarity matrix S where S[i][j] = cosine(query_i, key_j), applies the adjacency threshold `A = (S >= tau)`, enforces k_in by keeping only the top-K highest-similarity inbound edges per agent, breaks cycles by greedy removal of the lowest-weight edge, and performs topological sort via Kahn's algorithm with alphabetical tiebreaking for deterministic execution order. In **Phase C (work execution)**, agents execute their main work in the topological order from Phase B, with routed messages from connected agents injected into their context. Phase C uses temperature 0.3 with a 4096-token output limit. The Manager agent runs at temperature 0.1 with a 2000-token limit to make termination decisions.

<!-- Related: Phase A, Phase B, Phase C, descriptor generation, topology routing, work execution, three-phase, temperature, MiniLM routing -->

## What parameters control DyTopo swarm behavior and how should they be tuned?

<!-- Verified: 2026-02-13 -->

Three main parameters control how a DyTopo swarm operates. The **tau** parameter (routing threshold τ, default 0.3) sets the cosine similarity threshold — edges above tau create communication links between agents. A tau of 0.2 creates dense connections where most agents communicate with most others, producing richer collaboration at higher token cost. A tau of 0.3 (default) balances breadth of communication with focused relevance. A tau of 0.5 creates sparse connections where agents only link when descriptors are highly similar, producing faster convergence with more independent agents. The **k_in** parameter (default 3, range 1-5) caps the maximum inbound messages any single agent receives per round, preventing information overload on popular agents. The **max_rounds** parameter (default 5, range 1-10) limits total swarm rounds. A convergence check also runs each round: if agent outputs stop changing meaningfully, or the Manager sets `terminate=true`, the swarm terminates early regardless of max_rounds. Increase max_rounds to 8-10 for problems requiring iterative refinement where early rounds build foundations for later solutions.

<!-- Related: tau, k_in, max_rounds, routing threshold, parameter tuning, convergence, termination, DyTopo configuration -->

## How should a DyTopo swarm be launched and monitored?

<!-- Verified: 2026-02-13 -->

DyTopo is controlled through three MCP tools. Call `swarm_start(task, domain, tau, k_in, max_rounds)` to launch — it returns a `task_id` immediately without blocking. Only `task` is required; `domain` defaults to "code", and tau, k_in, max_rounds use defaults of 0.3, 3, and 5 respectively. Poll `swarm_status(task_id)` every 15-30 seconds to check progress — it returns the current round number, which agent is active, elapsed wall-clock time, and overall status (running, completed, or failed). When status shows completed, call `swarm_result(task_id, include_topology)` to retrieve the final output. Set `include_topology=true` for debugging or analysis — it adds a per-round log showing which agents connected, the similarity scores that created each edge, and the topological execution order. All agent inference runs through Qwen3-30B-A3B at localhost:1234 via AsyncOpenAI. The server stores up to 20 concurrent tasks and evicts the oldest half of completed tasks when the limit is reached.

<!-- Related: swarm_start, swarm_status, swarm_result, task_id, polling, monitoring, launch workflow, AsyncOpenAI -->

## When should a DyTopo swarm be used instead of solving directly?

<!-- Verified: 2026-02-13 -->

DyTopo swarms are valuable for complex, multi-perspective problems where specialist viewpoints improve the result through iterative collaboration. Good use cases include code review and debugging (Developer + Tester + Researcher collaborate on finding and fixing issues), multi-step mathematical reasoning (Parser + Solver + Verifier cross-check each other's work), architecture design (multiple specialists contribute domain expertise), and open-ended analysis that benefits from a critic-synthesizer dynamic. DyTopo adds meaningful overhead — multiple rounds of inference at localhost:1234, descriptor generation, and routing computation — so it should be reserved for tasks where that overhead produces better results than a single-pass response. Handle simple lookups, single-tool tasks, factual questions answerable from RAG or Memory, and time-sensitive queries directly. A direct response from the agent is faster and sufficient for straightforward requests. Launch a swarm when the problem genuinely benefits from multiple agents iterating toward a refined, cross-verified answer.

<!-- Related: when to use DyTopo, use cases, overhead, code review, math proofs, architecture, multi-perspective analysis -->

## How is DyTopo isolated from the RAG pipeline?

<!-- Verified: 2026-02-13 -->

DyTopo swarms run via `asyncio.create_task()` in the qdrant-rag server, which isolates them from the main server event loop. If a swarm crashes or encounters an error, it returns an error string to the caller — the exception is contained and does not propagate to the RAG pipeline or affect `rag_search` operations. The three core server singletons — the BGE-M3 FlagEmbedding model for RAG embedding, the Qdrant client connected to port 6334, and the MiniLM-L6-v2 routing model — are independent of the swarm lifecycle. DyTopo accesses MiniLM-L6-v2 in read-only mode for descriptor embedding and accesses the LM Studio API at localhost:1234 for agent inference, but it does not modify any shared state. A swarm failure cannot corrupt the Qdrant collection, disrupt embedding generation, or block incoming `rag_search` requests. This isolation-by-design means the RAG server remains stable regardless of swarm behavior, and multiple swarms can run concurrently without interfering with each other or with the search pipeline.

<!-- Related: isolation, asyncio, crash safety, BGE-M3, MiniLM, Qdrant client, RAG stability, concurrent swarms -->

## What models does DyTopo use and why was MiniLM chosen for routing?

<!-- Verified: 2026-02-13 -->

DyTopo uses two models with distinct roles. **Qwen3-30B-A3B-Instruct-2507** (Q6_K) at localhost:1234 handles all agent inference — descriptor generation in Phase A runs at temperature 0.1 with `/no_think` and a 256-token limit for fast deterministic output, while work execution in Phase C runs at temperature 0.3 with a 4096-token limit. Manager decisions use temperature 0.1 with a 2000-token limit. All inference calls go through AsyncOpenAI pointed at the LM Studio API with additional sampling parameters top_p=0.85, top_k=20, and min_p=0.05. **MiniLM-L6-v2** (22M parameters, 384-dimensional output, ~80 MB RAM on CPU) handles descriptor routing in Phase B — it embeds key and query vectors used to build the communication graph via cosine similarity. MiniLM-L6-v2 was chosen over BGE-M3 for routing because its wider similarity spread provides better threshold discrimination at the tau boundary. When tau is 0.3, the system needs to cleanly separate "should connect" from "should disconnect" — MiniLM's output distribution makes this boundary more reliable than BGE-M3's tighter clustering.

<!-- Related: Qwen3, MiniLM-L6-v2, inference model, routing model, temperature, similarity spread, tau discrimination, model choice -->
