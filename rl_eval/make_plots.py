#!/usr/bin/env python3
"""Generate all TitanRL eval plots from rl_eval/logs/*.log. Saves PNGs to rl_eval/plots/."""
import re, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGS = "rl_eval/logs"
OUT = "rl_eval/plots"

def steps(path):
    """Return list of per-step dicts of metrics parsed from 'Train | Step:' lines."""
    rows = []
    for line in open(path, errors="ignore"):
        if "Train | Step:" not in line:
            continue
        d = {}
        m = re.search(r"Train \| Step:\s*(\d+)", line); d["step"] = int(m.group(1)) if m else None
        for key in ["rollout_reward/_mean","loss/mean","bit_wise/logprob_diff/max",
                    "perf/trainer/tokens_per_second_full_step","perf/trainer/tokens_per_second_fwd_bwd",
                    "generator/decode_time_ms/mean","generator/inter_token_latency_ms/mean",
                    "trainer/grad_norm/mean","trainer/entropy/mean"]:
            mm = re.search(rf"{re.escape(key)}:\s*([0-9.eE+-]+)", line)
            if mm: d[key] = float(mm.group(1))
        rows.append(d)
    return rows

def val(path):
    """(pre,post) validation_reward/_mean."""
    t = open(path, errors="ignore").read()
    v = re.findall(r"validation_reward/_mean:\s*\+?([0-9.]+)\s*/\s*\+?([0-9.]+)", t)
    return (float(v[-1][0]), float(v[-1][1])) if v else (None, None)

def med(xs): 
    xs=[x for x in xs if x is not None]; return st.median(xs) if xs else 0

saved=[]
def save(fig, name):
    p=f"{OUT}/{name}.png"; fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig); saved.append(p); print("saved", p)

# ---- Fig 1: Tier 0 learning curve (rollout_reward + logprob_diff per step) ----
r = steps(f"{LOGS}/tier0_smoke.log")
if r:
    fig, ax1 = plt.subplots(figsize=(7,4))
    xs=[d["step"] for d in r]
    ax1.plot(xs, [d.get("rollout_reward/_mean") for d in r], "o-", color="tab:green", label="rollout_reward/_mean")
    ax1.set_xlabel("training step"); ax1.set_ylabel("rollout_reward/_mean", color="tab:green")
    ax1.tick_params(axis="y", labelcolor="tab:green")
    pre,post=val(f"{LOGS}/tier0_smoke.log")
    ax2=ax1.twinx()
    ax2.plot(xs,[d.get("bit_wise/logprob_diff/max") for d in r],"s--",color="tab:red",label="logprob_diff/max")
    ax2.set_ylabel("bit_wise/logprob_diff/max (drift)", color="tab:red"); ax2.tick_params(axis="y",labelcolor="tab:red")
    plt.title(f"Tier 0 — alphabet_sort smoke (Qwen3-0.6B)\nvalidation_reward/_mean {pre} → {post} (non-batch-invariant)")
    save(fig,"tier0_learning_curve")

# ---- Fig 2: Tier 1 BI ON vs OFF cost bars ----
def med_metrics(path):
    r=steps(path)
    return dict(
        itl=med([d.get("generator/inter_token_latency_ms/mean") for d in r]),
        dec=med([d.get("generator/decode_time_ms/mean") for d in r]),
        fbw=med([d.get("perf/trainer/tokens_per_second_fwd_bwd") for d in r]),
        full=med([d.get("perf/trainer/tokens_per_second_full_step") for d in r]),
    )
on=med_metrics(f"{LOGS}/tier1_cost_ON.log"); off=med_metrics(f"{LOGS}/tier1_cost_OFF.log")
if on["itl"] and off["itl"]:
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    # latency-type (lower=better): ITL, decode
    labels=["gen ITL (ms/tok)","gen decode (ms)"]
    axes[0].bar([i-0.2 for i in range(2)],[off["itl"],off["dec"]],0.4,label="BI OFF",color="tab:blue")
    axes[0].bar([i+0.2 for i in range(2)],[on["itl"],on["dec"]],0.4,label="BI ON",color="tab:orange")
    axes[0].set_xticks(range(2)); axes[0].set_xticklabels(labels); axes[0].set_title("Latency (lower=better)"); axes[0].legend()
    # throughput (higher=better)
    labels2=["fwd/bwd tok/s","full-step tok/s"]
    axes[1].bar([i-0.2 for i in range(2)],[off["fbw"],off["full"]],0.4,label="BI OFF",color="tab:blue")
    axes[1].bar([i+0.2 for i in range(2)],[on["fbw"],on["full"]],0.4,label="BI ON",color="tab:orange")
    axes[1].set_xticks(range(2)); axes[1].set_xticklabels(labels2); axes[1].set_title("Throughput (higher=better)"); axes[1].legend()
    fig.suptitle("Tier 1 — Batch-invariant ON vs OFF cost (Qwen3-0.6B, TP2/TP2)\nBI compute cost ~1.4–1.5× ; logprob_diff: ON=0.0 exact, OFF≈3e-5", y=1.08)
    fig.tight_layout()
    save(fig,"tier1_bi_cost")

# ---- Fig 3: Tier 3 async off-policy tradeoff ----
ops_vals=[0,1,3]; tp=[]; stale=[]; rew=[]
for o in ops_vals:
    r=steps(f"{LOGS}/tier3_offpolicy_{o}.log")
    tp.append(med([d.get("perf/trainer/tokens_per_second_full_step") for d in r]))
    stale.append(max([d.get("bit_wise/logprob_diff/max") for d in r if d.get("bit_wise/logprob_diff/max") is not None]))
    rew.append(val(f"{LOGS}/tier3_offpolicy_{o}.log")[1])
fig,ax1=plt.subplots(figsize=(7,4.5))
ax1.bar([str(o) for o in ops_vals], tp, color="tab:blue", alpha=0.7, label="throughput (tok/s)")
ax1.set_xlabel("max_offpolicy_steps"); ax1.set_ylabel("median full-step tok/s", color="tab:blue")
ax1.tick_params(axis="y",labelcolor="tab:blue")
for i,v in enumerate(tp): ax1.text(i, v+10, f"{v:.0f}", ha="center")
ax2=ax1.twinx()
ax2.plot([str(o) for o in ops_vals], stale, "s--", color="tab:red", label="logprob_diff/max (staleness)")
ax2.set_ylabel("logprob_diff/max (staleness)", color="tab:red"); ax2.tick_params(axis="y",labelcolor="tab:red")
plt.title("Tier 3 — Async off-policy tradeoff (Qwen3-0.6B)\nthroughput +55% at ops=3, staleness 0.66→5.51, reward stable ~0.40")
save(fig,"tier3_offpolicy_tradeoff")

# ---- Fig 4: Tier 3 compile ON vs OFF speedup ----
c_on=med_metrics(f"{LOGS}/tier3_compile_on.log"); c_off=med_metrics(f"{LOGS}/tier3_compile_off.log")
if c_on["full"] and c_off["full"]:
    fig,ax=plt.subplots(figsize=(7,4))
    labels=["full-step tok/s","fwd/bwd tok/s"]
    ax.bar([i-0.2 for i in range(2)],[c_off["full"],c_off["fbw"]],0.4,label="compile OFF",color="tab:gray")
    ax.bar([i+0.2 for i in range(2)],[c_on["full"],c_on["fbw"]],0.4,label="compile ON",color="tab:green")
    ax.set_xticks(range(2)); ax.set_xticklabels(labels); ax.set_ylabel("tokens/s (higher=better)")
    ax.legend(); ax.set_title("Tier 3 — torch.compile impact (Qwen3-0.6B)\n~1.36× end-to-end, ~2.14× trainer fwd/bwd")
    save(fig,"tier3_compile_speedup")

# ---- Fig 5: Tier 2 custom-task learning curve ----
r=steps(f"{LOGS}/tier2_count_letters.log")
if r:
    fig,ax=plt.subplots(figsize=(7,4))
    xs=[d["step"] for d in r]
    ax.plot(xs,[d.get("rollout_reward/_mean") for d in r],"o-",color="tab:purple",label="rollout_reward/_mean")
    pre,post=val(f"{LOGS}/tier2_count_letters.log")
    ax.set_xlabel("training step"); ax.set_ylabel("rollout_reward/_mean")
    ax.set_title(f"Tier 2 — custom count_letters task (Qwen3-0.6B)\nvalidation_reward/_mean {pre} → {post} (custom rubric shapes gradient)")
    ax.legend(); save(fig,"tier2_count_letters_curve")

# ---- Fig 6: Tier 2 GRPO vs DAPO reward curves ----
rg=steps(f"{LOGS}/tier2_count_letters.log"); rd=steps(f"{LOGS}/tier2_dapo.log")
if rg and rd:
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot([d["step"] for d in rg],[d.get("rollout_reward/_mean") for d in rg],"o-",label="GRPO",color="tab:blue")
    ax.plot([d["step"] for d in rd],[d.get("rollout_reward/_mean") for d in rd],"s-",label="DAPO (clip-higher 0.2/0.28)",color="tab:orange")
    ax.set_xlabel("training step"); ax.set_ylabel("rollout_reward/_mean")
    ax.set_title("Tier 2 — GRPO vs DAPO loss on count_letters\nboth reach val 0.786; loss swap is config-only")
    ax.legend(); save(fig,"tier2_grpo_vs_dapo")

print("\nTOTAL PLOTS:", len(saved))
for p in saved: print(" ", p)
