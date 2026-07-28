# TorchTitan RL Experiment — Visual Diagrams

> **Source of truth: the code (`torchtitan/experiments/rl/`).** These diagrams are hand-authored renderings of that code. When in doubt, trust the code first. Between the two diagram formats, **Mermaid is canonical** (diffable, version-controlled, renders on GitHub); the **Excalidraw files are the editable rendering** derived from it. They are not auto-synced, so minor wording/layout differences between a Mermaid diagram and its Excalidraw twin are cosmetic, not factual.
>
> Companion to `RL_EXPERIMENT_DEEP_DIVE.md`. Read this bottom-up: each diagram builds on the one before.
> Diagrams are [Mermaid](https://mermaid.js.org/) — they render automatically on GitHub. Editable Excalidraw versions live in `diagrams/*.excalidraw` (open at excalidraw.com or with the VS Code Excalidraw extension).
>
> **Editable Excalidraw files available** (`diagrams/`):
> | # | Diagram | File |
> |---|---------|------|
> | 1 | System architecture | `01_system_architecture.excalidraw` |
> | 2 | Async pipeline | `02_async_pipeline.excalidraw` |
> | 3 | Work-buffer state machine | `03_buffer_state_machine.excalidraw` |
> | 5 | Prompt → gradient | `05_prompt_to_gradient.excalidraw` |
> | 6 | GRPO advantage | `06_grpo_advantage.excalidraw` |
> | 8 | On-policy vs off-policy | `08_on_vs_off_policy.excalidraw` |
> | 9 | Bitwise parity | `09_bitwise_parity.excalidraw` |

---

## Diagram 1 — System architecture (the 30,000-ft view)

Who owns whom, and where the GPUs are. The Controller is the brain; the Trainer and Generators run on **disjoint GPU meshes**; weights flow one way (trainer → generators) through TorchStore.

```mermaid
flowchart TB
    subgraph DRIVER["Driver process"]
        C["<b>Controller</b><br/>controller.py<br/>async orchestrator"]
        RO["<b>Rollouter</b><br/>dataset + env + rubric"]
        WB[("<b>Work Buffer</b><br/>bounds off-policy staleness")]
        MP["<b>MetricsProcessor</b><br/>W&B / TensorBoard / console"]
    end

    subgraph TMESH["Trainer GPU mesh"]
        T["<b>PolicyTrainer</b> (TorchTitan)<br/>forward_backward · optim_step<br/>FSDP / TP / EP"]
    end

    subgraph GMESH["Generator GPU mesh(es)"]
        G1["<b>VLLMGenerator #0</b><br/>vLLM engine + TorchTitan model"]
        G2["<b>VLLMGenerator #1</b><br/>(optional replica)"]
    end

    TS[("<b>TorchStore</b><br/>weight staging<br/>(RDMA-capable)")]

    C -->|owns| RO
    C -->|owns| WB
    C -->|owns| MP
    C -->|spawns via Monarch| T
    C -->|spawns via Monarch| G1
    C -->|spawns via Monarch| G2
    C -->|"generate() calls"| G1
    RO -.->|drives rollouts through| G1

    T -->|"push_model_state_dict"| TS
    TS -->|"pull_model_state_dict"| G1
    TS -->|"pull_model_state_dict"| G2

    classDef ctrl fill:#e3f2fd,stroke:#1565c0,color:#0d47a1;
    classDef train fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20;
    classDef gen fill:#fff3e0,stroke:#ef6c00,color:#e65100;
    classDef store fill:#f3e5f5,stroke:#7b1fa2,color:#4a148c;
    class C,RO,WB,MP ctrl;
    class T train;
    class G1,G2 gen;
    class TS,WB store;
```

**Read it as:** the Controller spawns actors on separate meshes (Monarch), asks generators to `generate`, feeds results through the buffer to the trainer, and after each optimizer step ships new weights to the generators via TorchStore.

---

## Diagram 2 — The async pipeline (the beating heart)

Five `asyncio` loops form a producer→consumer pipeline glued together by the shared work buffer. This is the single most important picture in the whole system.

```mermaid
flowchart LR
    DS[("Dataset")] --> DIL

    subgraph LOOPS["Controller async loops"]
        direction LR
        DIL["<b>_data_input_loop</b><br/>get sample<br/>add_work()"]
        RL["<b>_rollout_loop × N</b><br/>generate + score<br/>a whole group"]
        BL["<b>_batcher_loop</b><br/>build + pack<br/>training samples"]
        TL["<b>_trainer_loop</b><br/>fwd/bwd · optim<br/>· weight sync"]
    end

    DIL -->|"RolloutGroupWork"| WB
    WB -->|"claim_next()"| RL
    RL -->|"finalize_work()"| WB
    WB -->|"take_finalized()"| BL
    BL -->|"TrainingBatch"| Q[["training_batch_queue<br/>(maxsize=1)"]]
    Q -->|"get()"| TL

    WB[("<b>RolloutGroupWorkBuffer</b><br/>active slots =<br/>(max_offpolicy_steps + 1)<br/>× num_groups_per_train_step")]

    TL -.->|"release_active_groups()<br/>AFTER weight pull"| WB
    TL ==>|"generate() during rollout"| GEN(["Generators"])
    RL ==>|"generate()"| GEN

    classDef loop fill:#e3f2fd,stroke:#1565c0,color:#0d47a1;
    classDef buf fill:#f3e5f5,stroke:#7b1fa2,color:#4a148c;
    class DIL,RL,BL,TL loop;
    class WB,Q buf;
```

**Backpressure is the trick:** the data-input loop can only add work when the buffer has a free slot, and slots are released **only after the trainer's weight pull** — so generation can run ahead, but never far enough to produce a "born-stale" sample.

---

## Diagram 3 — Work buffer: one group's lifecycle (state machine)

Every prompt group is a `RolloutGroupWork` that walks this state machine. Note the subtle-but-critical detail: **the active slot is held until the trainer explicitly releases it**, not when the group is finalized or taken.

```mermaid
stateDiagram-v2
    [*] --> WAITING: _data_input_loop<br/>add_work() (slot acquired)
    WAITING --> INFLIGHT: _rollout_loop<br/>claim_next()
    INFLIGHT --> FINALIZED: _rollout_loop<br/>finalize_work()
    FINALIZED --> Taken: _batcher_loop<br/>take_finalized()<br/>(slot STILL held)

    Taken --> Released_trained: _trainer_loop<br/>release_active_groups(..., "trained")<br/>AFTER weight pull
    Taken --> Released_untrainable: _batcher_loop<br/>release(1, "untrainable_group")<br/>(empty / zero-std group)

    Released_trained --> [*]
    Released_untrainable --> [*]

    note right of Taken
        Slot NOT freed on finalize or take.
        This is what bounds off-policiness:
        at most max_active_rollout_groups
        live in the whole pipeline.
    end note
```

---

## Diagram 4 — One rollout, turn by turn (sequence)

Zooming into a single rollout: the Rollouter alternates between the environment and the generator until the env says `done` (or a limit trips). `TokenEnv` is the translator that keeps everything else in message-space.

```mermaid
sequenceDiagram
    autonumber
    participant R as Rollouter<br/>(_run_single_rollout)
    participant TE as TokenEnv<br/>(token ↔ message)
    participant ME as MessageEnv<br/>(your task logic)
    participant G as Generator<br/>(vLLM)

    R->>TE: init()
    TE->>ME: init()
    ME-->>TE: opening messages + tool specs
    TE-->>R: prompt_token_ids (turn 0)

    loop until env terminal
        R->>G: generate(prompt_token_ids)
        G-->>R: Completion (tokens + logprobs)
        R->>TE: step(completion)
        TE->>ME: step(completion_message)
        ME-->>TE: env reply / done / rewards
        TE-->>R: next_prompt_token_ids OR terminal
    end

    Note over R,G: turns collected → Rollout<br/>(scored later by the Rubric)
```

---

## Diagram 5 — How a dataset prompt becomes a gradient (data transformation)

The end-to-end transformation of *one prompt* into *training signal*. This is the GRPO pipeline made concrete.

```mermaid
flowchart TD
    S["Dataset sample<br/>(one prompt)"] --> ENVS["group_size sibling envs<br/>(the GRPO group)"]
    ENVS --> RG["group_size rollouts<br/>(each: multi-turn play-out)"]
    RG --> RUB["<b>Rubric.score_group</b><br/>weighted reward fns → r_i"]
    RUB --> ADV["<b>AdvantageEstimator</b><br/>A_i = (r_i − mean(r)) / denom"]
    ADV --> TSB["<b>TrainingSampleBuilder</b><br/>tokens + loss_mask +<br/>logprobs + advantages<br/>(+ drop zero-std groups)"]
    TSB --> BAT["<b>Batcher</b><br/>next-fit pack → microbatches"]
    BAT --> LOSS["<b>GRPO / DAPO loss</b><br/>clipped surrogate<br/>Σ token_loss / global_valid_tokens"]
    LOSS --> STEP["optim_step →<br/>new policy weights"]

    classDef roll fill:#fff3e0,stroke:#ef6c00,color:#e65100;
    classDef build fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20;
    classDef math fill:#e3f2fd,stroke:#1565c0,color:#0d47a1;
    class S,ENVS,RG roll;
    class RUB,ADV,TSB,BAT build;
    class LOSS,STEP math;
```

---

## Diagram 6 — GRPO advantage intuition (why "group relative")

GRPO has **no value network**. The baseline is just the *mean reward of the sibling group*. A rollout that beats its siblings gets a positive advantage; one that lags gets negative. Simple and cheap.

```mermaid
flowchart LR
    subgraph GRP["One prompt group (group_size = 4)"]
        r0["rollout 0<br/>r = 1.0"]
        r1["rollout 1<br/>r = 0.0"]
        r2["rollout 2<br/>r = 0.5"]
        r3["rollout 3<br/>r = 0.5"]
    end
    GRP --> MEAN["mean(r) = 0.5<br/>(the baseline)"]
    MEAN --> A0["A0 = +0.5 ✅ reinforce"]
    MEAN --> A1["A1 = −0.5 ❌ suppress"]
    MEAN --> A2["A2 = 0.0 neutral"]
    MEAN --> A3["A3 = 0.0 neutral"]

    classDef pos fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20;
    classDef neg fill:#ffebee,stroke:#c62828,color:#b71c1c;
    classDef neu fill:#eceff1,stroke:#546e7a,color:#263238;
    class A0 pos;
    class A1 neg;
    class A2,A3 neu;
```

> `denom = 1.0` → Dr.GRPO (mean baseline only). `denom = std(r) + eps` → standard GRPO.

---

## Diagram 7 — Weight sync overlapped with the next step (timeline)

The trainer never sits idle waiting for weights to ship. Push→pull→slot-release for step *N* run **in the background, overlapped with step N+1's forward/backward**. The trainer only *blocks* on the previous sync at the last safe moment.

```mermaid
gantt
    title Weight sync overlaps compute (per training step)
    dateFormat X
    axisFormat %s
    section Step N
    forward_backward         :a1, 0, 3
    wait prev push (block)   :a2, 3, 1
    optim_step               :a3, 4, 1
    wait prev pull (block)   :a4, 5, 1
    section Background sync (N)
    push weights → TorchStore :crit, b1, 5, 2
    pull weights → generators :crit, b2, 7, 2
    release buffer slots      :b3, 9, 1
    section Step N+1
    forward_backward          :c1, 6, 3
```

**Takeaway:** the striped background bars (push/pull) run concurrently with step N+1's compute, so weight sync is mostly hidden.

---

## Diagram 8 — On-policy vs off-policy (the staleness window)

The `max_offpolicy_steps` knob sets how far generation may run ahead of training. `0` = strict lockstep (needed for bitwise parity); `>0` = generators stay busy while the trainer works.

```mermaid
flowchart TB
    subgraph SYNC["max_offpolicy_steps = 0 (on-policy / sync)"]
        direction LR
        s1["gen v1"] --> s2["train → v2"] --> s3["gen v2"] --> s4["train → v3"]
    end

    subgraph ASYNC["max_offpolicy_steps = 2 (async)"]
        direction LR
        a1["gen (v1)"] --> a2["gen (v1)"] --> a3["gen (v1)"]
        a2 -.-> t1["train → v2"]
        a3 -.-> t2["train → v3"]
    end

    SYNC -.->|"generator idles<br/>while trainer runs"| NOTE1["✅ bitwise-verifiable<br/>❌ lower GPU utilization"]
    ASYNC -.->|"generator runs ahead,<br/>staleness bounded by buffer"| NOTE2["✅ high GPU utilization<br/>⚠️ samples up to 2 steps stale"]

    classDef good fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20;
    class NOTE1,NOTE2 good;
```

---

## Diagram 9 — Bitwise parity: why the unified model matters

The correctness bug this whole experiment is designed to kill. Two code paths compute logprobs; if they disagree, the importance ratio `π_θ/π_old` is wrong. The unified model + batch-invariant mode drives the difference to **exactly zero**.

```mermaid
flowchart TB
    M["<b>ONE TorchTitan model definition</b>"]
    M --> GP["Generator path<br/>(inside vLLM, bf16)<br/>→ logprobs_gen"]
    M --> TP["Trainer path<br/>(FSDP bf16 forward)<br/>→ logprobs_train"]

    GP --> CMP{"logprobs_gen<br/>vs<br/>logprobs_train"}
    TP --> CMP

    CMP -->|"naive: different batch<br/>composition & kernels"| DRIFT["⚠️ logprob_diff > 0<br/>biased objective,<br/>can flip MoE routing"]
    CMP -->|"batch-invariant mode:<br/>fixed-order kernels,<br/>deterministic NCCL,<br/>num_splits=1, fp32 RoPE"| ZERO["✅ logprob_diff == 0<br/>(bitwise identical)"]

    classDef bad fill:#ffebee,stroke:#c62828,color:#b71c1c;
    classDef good fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20;
    class DRIFT bad;
    class ZERO good;
```

> Cost (Qwen3-8B, Search-R1, 8×H100): raw compute ~2.4–2.9× slower, but end-to-end wall-clock ≈ unchanged for orchestration-bound workloads. Requires matched TP; no sequence parallelism.

---

## Diagram 10 — Generator internals (continuous batching)

Inside one `VLLMGenerator`: rank 0 accepts requests and enqueues decisions; a background engine loop runs `engine.step()` bursts so new requests join mid-flight instead of waiting for the batch to drain. Weight pulls ride the same loop.

```mermaid
sequenceDiagram
    autonumber
    participant CT as Controller
    participant R0 as Rank 0 (intake + futures)
    participant EL as _engine_loop (per rank)
    participant VE as vLLM engine.step()

    CT->>R0: generate(prompt_0)
    R0->>R0: enqueue LoopDecision, await future_0
    CT->>R0: generate(prompt_1)
    R0->>R0: enqueue LoopDecision, await future_1

    loop engine loop (continuous)
        EL->>EL: _decide_next_action()<br/>broadcast to followers
        EL->>VE: add_request(prompt_0, prompt_1)
        EL->>VE: engine.step() × N (burst)
        VE-->>R0: finished request(s)
        R0-->>CT: resolve future → Completion
    end

    CT->>R0: pull_model_state_dict(v)
    R0->>EL: enqueue PULL decision
    EL->>VE: apply new weights between bursts
```

---

*These diagrams are a learning aid. The code is the source of truth — cross-check against `torchtitan/experiments/rl/` (it moves fast).*
