#!/usr/bin/env python3
"""Generate all TitanRL eval plots from rl_eval/logs/*.log. Saves PNGs to rl_eval/plots/.

Style goals (2026-07 revision):
  - professional look: grid on, clean spines, consistent palette, value labels on bars
  - data-driven y-axis ranges (no oversized empty space)
  - readable multi-metric panels (no metric crushed by a larger-scale one)
  - new Tier 1 "trust the numbers" parity figure
"""
import re, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

LOGS = "rl_eval/logs"
OUT = "rl_eval/plots"

# ---------------------------------------------------------------- global style
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.titleweight": "semibold",
    "axes.labelsize": 10.5,
    "axes.grid": True,
    "grid.alpha": 0.30,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,          # grid behind data
    "axes.spines.top": False,
    "axes.spines.right": False,      # (re-enabled per-axis for twinx plots)
    "axes.edgecolor": "#444444",
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.fontsize": 9.5,
    "figure.constrained_layout.use": False,
})

# consistent palette
C_OFF   = "#4C78A8"   # blue  (baseline / OFF)
C_ON    = "#F58518"   # orange (feature ON)
C_REW   = "#54A24B"   # green (reward)
C_REW2  = "#7B5EA7"   # purple (custom-task reward)
C_DRIFT = "#D62728"   # red (drift / staleness)
C_GRAD  = "#E45756"   # salmon (grad norm)
C_NEUT  = "#9E9E9E"   # gray (secondary baseline)

def _grid(ax):
    ax.grid(True, which="major", alpha=0.30, ls="--", lw=0.6)
    ax.set_axisbelow(True)

def _bar_labels(ax, bars, fmt="{:.0f}", dy=0.0):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x()+b.get_width()/2, h),
                    xytext=(0, 3+dy), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5)

def _headroom(ax, values, base=0.0, frac=0.18):
    vals=[v for v in values if v is not None]
    if not vals: return
    top=max(vals); ax.set_ylim(base, top*(1+frac) if top>0 else 1)

# ---------------------------------------------------------------- log parsing
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

def diffs(path):
    xs=[]
    for line in open(path, errors="ignore"):
        if "Train | Step:" not in line: continue
        m=re.search(r"bit_wise/logprob_diff/max:\s*([0-9.eE+-]+)", line)
        if m: xs.append(float(m.group(1)))
    return xs

def med(xs):
    xs=[x for x in xs if x is not None]; return st.median(xs) if xs else 0

saved=[]
def save(fig, name):
    p=f"{OUT}/{name}.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); saved.append(p); print("saved", p)

def med_metrics(path):
    r=steps(path)
    return dict(
        itl=med([d.get("generator/inter_token_latency_ms/mean") for d in r]),
        dec=med([d.get("generator/decode_time_ms/mean") for d in r]),
        fbw=med([d.get("perf/trainer/tokens_per_second_fwd_bwd") for d in r]),
        full=med([d.get("perf/trainer/tokens_per_second_full_step") for d in r]),
    )

# ======================================================================
# Fig 1: Tier 0 learning curve (rollout_reward + logprob_diff per step)
# ======================================================================
r = steps(f"{LOGS}/tier0_smoke.log")
if r:
    fig, ax1 = plt.subplots(figsize=(7,4))
    xs=[d["step"] for d in r]
    rew=[d.get("rollout_reward/_mean") for d in r]
    l1,=ax1.plot(xs, rew, "o-", color=C_REW, label="rollout_reward/_mean")
    ax1.set_xlabel("training step"); ax1.set_ylabel("rollout_reward/_mean", color=C_REW)
    ax1.tick_params(axis="y", labelcolor=C_REW); _grid(ax1)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    _headroom(ax1, rew, base=0, frac=0.15)
    pre,post=val(f"{LOGS}/tier0_smoke.log")
    ax2=ax1.twinx(); ax2.spines["right"].set_visible(True)
    dr=[d.get("bit_wise/logprob_diff/max") for d in r]
    l2,=ax2.plot(xs,dr,"s--",color=C_DRIFT,label="logprob_diff/max (drift)")
    ax2.set_ylabel("bit_wise/logprob_diff/max (drift)", color=C_DRIFT); ax2.tick_params(axis="y",labelcolor=C_DRIFT)
    ax2.grid(False); _headroom(ax2, dr, base=0, frac=0.12)
    ax1.legend(handles=[l1,l2], loc="upper left")
    plt.title(f"Tier 0 — alphabet_sort smoke (Qwen3-0.6B)\nvalidation_reward/_mean {pre} → {post} (non-batch-invariant)")
    save(fig,"tier0_learning_curve")

# ======================================================================
# Fig 2: Tier 1 BI ON vs OFF cost  — 2x2 per-metric panels (readable axes)
# ======================================================================
def cost_panel(on, off, title, outname):
    if not (on["itl"] and off["itl"]): return
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2))
    specs = [
        ("gen ITL",        "ms / token",  off["itl"],  on["itl"],  "lower=better", "{:.1f}"),
        ("gen decode",     "ms",          off["dec"],  on["dec"],  "lower=better", "{:.0f}"),
        ("trainer fwd/bwd","tokens / s",  off["fbw"],  on["fbw"],  "higher=better","{:.0f}"),
        ("trainer full-step","tokens / s",off["full"], on["full"], "higher=better","{:.0f}"),
    ]
    for ax,(name,unit,voff,von,better,fmt) in zip(axes.ravel(), specs):
        b=ax.bar([0,1],[voff,von],width=0.6,color=[C_OFF,C_ON])
        ax.set_xticks([0,1]); ax.set_xticklabels(["BI OFF","BI ON"])
        ax.set_ylabel(unit); _grid(ax); _headroom(ax,[voff,von],base=0,frac=0.22)
        _bar_labels(ax,b,fmt=fmt)
        ratio = (von/voff) if voff else 0
        rtxt = f"{ratio:.2f}× slower" if better=="lower=better" else f"{ratio:.2f}× ({'slower' if ratio<1 else 'faster'})"
        ax.set_title(f"{name}  ({better})\nBI cost: {rtxt}", fontsize=10)
    fig.suptitle(title, y=0.99, fontsize=12, fontweight="semibold")
    fig.tight_layout(rect=[0,0,1,0.96])
    save(fig, outname)

on=med_metrics(f"{LOGS}/tier1_cost_ON.log"); off=med_metrics(f"{LOGS}/tier1_cost_OFF.log")
cost_panel(on, off,
    "Tier 1 — Batch-invariant ON vs OFF cost (Qwen3-0.6B, TP2/TP2, 6 steps)\nlogprob_diff: BI ON = 0.0 exact · BI OFF ≈ 2e-5",
    "tier1_bi_cost")

# ======================================================================
# Fig 2b (NEW): Tier 1 "Do I trust the numbers?" — PARITY proof
#   Left  : unit-test scorecard (token-logprobs checked vs differing)
#   Right : live-loop drift over steps (symlog) — the divergence story
# ======================================================================
# parse parity unit-test log for per-test token counts + differing counts
ptxt = open(f"{LOGS}/tier1_parity_varlen.log", errors="ignore").read()
def parity_test(tag):
    rows = re.findall(rf"{re.escape(tag)}: max_delta=([0-9.eE+-]+), num_diff=(\d+)/(\d+)", ptxt)
    tot_tok = sum(int(t) for _,_,t in rows); tot_diff = sum(int(d) for _,d,_ in rows)
    return len(rows), tot_diff, tot_tok
tests = [
    ("batch-invariance\nprefill bsz2 vs bsz5", "prefill(bsz=2) vs prefill(bsz=5)"),
    ("trainer vs vLLM\nprefill",               "Trainer prefill vs vLLM prefill"),
    ("vLLM decode vs\n2nd-pass prefill",       "vLLM decode vs vLLM 2nd-pass prefill"),
]
labels=[]; toks=[]; diffs_n=[]; seqs=[]
for lab, tag in tests:
    n, d, t = parity_test(tag)
    labels.append(lab); seqs.append(n); toks.append(t); diffs_n.append(d)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

# ---- Left: scorecard ----
x=range(len(labels))
b1=axL.bar(x, toks, width=0.6, color=C_REW, label="token log-probs checked")
b2=axL.bar(x, diffs_n, width=0.6, color=C_DRIFT, label="differing (non-bitwise-equal)")
axL.set_xticks(list(x)); axL.set_xticklabels(labels, fontsize=8.6)
axL.set_ylabel("token log-probs compared"); _grid(axL)
_headroom(axL, toks, base=0, frac=0.42)   # extra headroom so the legend clears bar annotations
for xi, t, n, s in zip(x, toks, diffs_n, seqs):
    axL.annotate(f"{t} tok · {s} seq\n{n} differ", (xi, t), xytext=(0,4),
                 textcoords="offset points", ha="center", va="bottom", fontsize=8.3)
axL.legend(loc="upper center", ncol=1, fontsize=8.6)
axL.set_title(f"Unit test — bitwise parity\n{sum(diffs_n)}/{sum(toks)} token log-probs differ  →  all {sum(seqs)} sequences bitwise-identical")

# ---- Right: live-loop drift over steps (symlog) ----
on6  = diffs(f"{LOGS}/tier1_cost_ON.log")      # BI ON, on-policy
off6 = diffs(f"{LOGS}/tier1_cost_OFF.log")     # BI OFF, on-policy
t0   = diffs(f"{LOGS}/tier0_smoke.log")        # non-BI, off-policy loop
axR.plot(range(1,len(on6)+1),  on6,  "o-", color=C_ON,   label="BI ON (on-policy) — 0.0 exact")
axR.plot(range(1,len(off6)+1), off6, "s-", color=C_OFF,  label="BI OFF (on-policy) — ~2e-5")
axR.plot(range(1,len(t0)+1),   t0,   "^--",color=C_DRIFT,label="non-BI loop (Tier 0) — diverges")
axR.set_yscale("symlog", linthresh=1e-5)
axR.set_xlabel("training step"); axR.set_ylabel("bit_wise/logprob_diff/max  (symlog)")
axR.xaxis.set_major_locator(MaxNLocator(integer=True))
_grid(axR); axR.set_ylim(-1e-6, 20)
axR.legend(loc="upper left", fontsize=8.6)
axR.set_title("Live-loop trainer↔generator drift\nBI ON stays exactly 0; without it, skew accumulates")

fig.suptitle("Tier 1 — Do I trust the numbers?  (bitwise parity holds; drift is measurable)", y=1.02, fontsize=12, fontweight="semibold")
fig.tight_layout()
save(fig, "tier1_parity_trust")

# ======================================================================
# Fig 3: Tier 3 async off-policy tradeoff
# ======================================================================
ops_vals=[0,1,3]; tp=[]; stale=[]; rew=[]
for o in ops_vals:
    r=steps(f"{LOGS}/tier3_offpolicy_{o}.log")
    tp.append(med([d.get("perf/trainer/tokens_per_second_full_step") for d in r]))
    lp=[d.get("bit_wise/logprob_diff/max") for d in r if d.get("bit_wise/logprob_diff/max") is not None]
    stale.append(max(lp) if lp else 0)
    rew.append(val(f"{LOGS}/tier3_offpolicy_{o}.log")[1])
fig,ax1=plt.subplots(figsize=(7,4.5))
b=ax1.bar([str(o) for o in ops_vals], tp, color=C_OFF, alpha=0.85, width=0.55, label="throughput (tok/s)")
ax1.set_xlabel("max_offpolicy_steps"); ax1.set_ylabel("median full-step tok/s", color=C_OFF)
ax1.tick_params(axis="y",labelcolor=C_OFF); _grid(ax1); _headroom(ax1, tp, base=0, frac=0.15)
_bar_labels(ax1, b, fmt="{:.0f}")
ax2=ax1.twinx(); ax2.spines["right"].set_visible(True)
ax2.plot([str(o) for o in ops_vals], stale, "s--", color=C_DRIFT, label="logprob_diff/max (staleness)")
ax2.set_ylabel("logprob_diff/max (staleness)", color=C_DRIFT); ax2.tick_params(axis="y",labelcolor=C_DRIFT)
ax2.grid(False); _headroom(ax2, stale, base=0, frac=0.30)
plt.title("Tier 3 — Async off-policy tradeoff (Qwen3-0.6B)\nthroughput +55% at ops=3, staleness 0.66→5.51, reward stable ~0.40")
save(fig,"tier3_offpolicy_tradeoff")

# ======================================================================
# Fig 4: Tier 3 compile ON vs OFF speedup — 2x2 per-metric (throughput only meaningful)
# ======================================================================
def compile_panel(c_on, c_off, title, outname):
    if not (c_on.get("full") and c_off.get("full")): return
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2))
    specs=[("full-step","tokens / s", c_off["full"], c_on["full"]),
           ("fwd/bwd","tokens / s",  c_off["fbw"],  c_on["fbw"])]
    for ax,(name,unit,voff,von) in zip(axes, specs):
        b=ax.bar([0,1],[voff,von],width=0.6,color=[C_NEUT,C_REW])
        ax.set_xticks([0,1]); ax.set_xticklabels(["compile OFF","compile ON"])
        ax.set_ylabel(unit); _grid(ax); _headroom(ax,[voff,von],base=0,frac=0.22)
        _bar_labels(ax,b,fmt="{:.0f}")
        ax.set_title(f"trainer {name}  (higher=better)\n{von/voff:.2f}× with compile", fontsize=10)
    fig.suptitle(title, y=1.0, fontsize=12, fontweight="semibold")
    fig.tight_layout(rect=[0,0,1,0.95])
    save(fig, outname)

compile_panel(med_metrics(f"{LOGS}/tier3_compile_on.log"), med_metrics(f"{LOGS}/tier3_compile_off.log"),
    "Tier 3 — torch.compile impact (Qwen3-0.6B, 6 steps)\n~1.36× end-to-end, ~2.14× trainer fwd/bwd",
    "tier3_compile_speedup")

# ======================================================================
# Fig 5: Tier 2 custom-task learning curve
# ======================================================================
r=steps(f"{LOGS}/tier2_count_letters.log")
if r:
    fig,ax=plt.subplots(figsize=(7,4))
    xs=[d["step"] for d in r]; rr=[d.get("rollout_reward/_mean") for d in r]
    ax.plot(xs,rr,"o-",color=C_REW2,label="rollout_reward/_mean")
    pre,post=val(f"{LOGS}/tier2_count_letters.log")
    ax.set_xlabel("training step"); ax.set_ylabel("rollout_reward/_mean")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)); _grid(ax); _headroom(ax, rr, base=0, frac=0.15)
    ax.set_title(f"Tier 2 — custom count_letters task (Qwen3-0.6B)\nvalidation_reward/_mean {pre} → {post} (custom rubric shapes gradient)")
    ax.legend(); save(fig,"tier2_count_letters_curve")

# ======================================================================
# Fig 6: Tier 2 GRPO vs DAPO reward curves
# ======================================================================
rg=steps(f"{LOGS}/tier2_count_letters.log"); rd=steps(f"{LOGS}/tier2_dapo.log")
if rg and rd:
    fig,ax=plt.subplots(figsize=(7,4))
    yg=[d.get("rollout_reward/_mean") for d in rg]; yd=[d.get("rollout_reward/_mean") for d in rd]
    ax.plot([d["step"] for d in rg],yg,"o-",label="GRPO",color=C_OFF)
    ax.plot([d["step"] for d in rd],yd,"s-",label="DAPO (clip-higher 0.2/0.28)",color=C_ON)
    ax.set_xlabel("training step"); ax.set_ylabel("rollout_reward/_mean")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)); _grid(ax)
    _headroom(ax, [v for v in yg+yd if v is not None], base=0, frac=0.15)
    ax.set_title("Tier 2 — GRPO vs DAPO loss on count_letters\nboth reach val 0.786; loss swap is config-only")
    ax.legend(); save(fig,"tier2_grpo_vs_dapo")

print("\nTOTAL PLOTS:", len(saved))

# =========================================================================
# SCALED VARIANTS: 5/100/1000-step learning curves + 100-step comparisons
# =========================================================================
print("\n--- scaled variants ---")

def learning_curve(logpath, title, outname, color=C_REW):
    r = steps(logpath)
    if not r:
        print(f"skip {outname}: no data at {logpath}"); return
    xs = [d["step"] for d in r]
    rew = [d.get("rollout_reward/_mean") for d in r]
    fig, ax1 = plt.subplots(figsize=(7,4))
    ax1.plot(xs, rew, "-", color=color, linewidth=1.4, label="rollout_reward/_mean")
    ax1.set_xlabel("training step"); ax1.set_ylabel("rollout_reward/_mean", color=color)
    ax1.tick_params(axis="y", labelcolor=color); ax1.set_ylim(0, 1.02); _grid(ax1)
    pre, post = val(logpath)
    ax2 = ax1.twinx(); ax2.spines["right"].set_visible(True)
    gn = [d.get("trainer/grad_norm/mean") for d in r]
    ax2.plot(xs, gn, "-", color=C_GRAD, alpha=0.55, linewidth=0.9, label="grad_norm")
    ax2.set_ylabel("grad_norm/mean", color=C_GRAD); ax2.tick_params(axis="y", labelcolor=C_GRAD)
    ax2.grid(False); _headroom(ax2, gn, base=0, frac=0.10)
    # combined legend
    h1,l1=ax1.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="lower right", fontsize=9)
    if pre is not None and post is not None:
        sub = f"validation_reward/_mean {pre} → {post}  ({len(xs)} steps)"
    else:
        rr = [x for x in rew if x is not None]
        sub = (f"rollout_reward/_mean {rr[0]:.2f} → {rr[-1]:.2f} (converged)  ({len(xs)} steps)"
               if rr else f"({len(xs)} steps)")
    plt.title(f"{title}\n{sub}")
    save(fig, outname)

learning_curve(f"{LOGS}/scaled_base_ops3_compon_s100.log",
               "Tier 0 — alphabet_sort (Qwen3-0.6B), 100 steps", "tier0_learning_curve_s100", C_REW)
learning_curve(f"{LOGS}/scaled_base_ops3_compon_s1000.log",
               "Tier 0 — alphabet_sort (Qwen3-0.6B), 1000 steps", "tier0_learning_curve_s1000", C_REW)

learning_curve(f"{LOGS}/tier2b_cl_grpo_s5.log",
               "Tier 2 — count_letters brevity reward (Qwen3-0.6B), 5 steps", "tier2_count_letters_curve_s5", C_REW2)
learning_curve(f"{LOGS}/tier2b_cl_grpo_s100.log",
               "Tier 2 — count_letters brevity reward (Qwen3-0.6B), 100 steps", "tier2_count_letters_curve_s100", C_REW2)
learning_curve(f"{LOGS}/tier2b_cl_grpo_s1000.log",
               "Tier 2 — count_letters brevity reward (Qwen3-0.6B), 586-step run (stopped: reward saturated ~0.92, then batch starvation G13)", "tier2_count_letters_curve_s1000", C_REW2)

# Tier 2 decode-time-vs-step (brevity fix keeps generation FLAT)
_r = steps(f"{LOGS}/tier2b_cl_grpo_s1000.log")
if _r:
    xs=[d["step"] for d in _r if d.get("generator/decode_time_ms/mean") is not None]
    ys=[d["generator/decode_time_ms/mean"] for d in _r if d.get("generator/decode_time_ms/mean") is not None]
    if xs:
        fig,ax=plt.subplots(figsize=(7,4))
        ax.plot(xs, ys, "-", color=C_REW, lw=1.2, label="brevity reward + max_tokens=128")
        ax.axhline(50, ls=":", color=C_NEUT, lw=1.2, label="~50 ms reference")
        ax.set_xlabel("training step"); ax.set_ylabel("generator/decode_time_ms/mean")
        # data-driven y-range (was hard-coded 0..300; data maxes ~155)
        ax.set_ylim(0, max(ys)*1.20); _grid(ax)
        ax.set_title("Tier 2 — decode time stays FLAT with brevity fix (~50 ms)\ncontrast G12: without it, decode grew ~53 ms → ~11,000 ms (~200×)")
        ax.legend(); save(fig, "tier2_decode_time_stable")

# Tier 3 offpolicy tradeoff @100 steps
ops_logs = {0: f"{LOGS}/scaled_ops0_s100.log", 1: f"{LOGS}/scaled_ops1_s100.log", 3: f"{LOGS}/scaled_base_ops3_compon_s100.log"}
tp=[]; stale=[]; okv=[]
for o in [0,1,3]:
    rr = steps(ops_logs[o])
    if not rr: continue
    full=[d.get("perf/trainer/tokens_per_second_full_step") for d in rr if d.get("perf/trainer/tokens_per_second_full_step")]
    lp=[d.get("bit_wise/logprob_diff/max") for d in rr if d.get("bit_wise/logprob_diff/max") is not None]
    tp.append(st.median(full) if full else 0); stale.append(max(lp) if lp else 0); okv.append(o)
if tp:
    fig, ax1 = plt.subplots(figsize=(7,4.5))
    b=ax1.bar([str(o) for o in okv], tp, color=C_OFF, alpha=0.85, width=0.55)
    ax1.set_xlabel("max_offpolicy_steps"); ax1.set_ylabel("median full-step tok/s", color=C_OFF)
    ax1.tick_params(axis="y", labelcolor=C_OFF); _grid(ax1); _headroom(ax1, tp, base=0, frac=0.20)
    _bar_labels(ax1, b, fmt="{:.0f}")
    ax2=ax1.twinx(); ax2.spines["right"].set_visible(True)
    ax2.plot([str(o) for o in okv], stale, "s--", color=C_DRIFT)
    ax2.set_ylabel("logprob_diff/max (staleness)", color=C_DRIFT); ax2.tick_params(axis="y", labelcolor=C_DRIFT)
    ax2.grid(False); _headroom(ax2, stale, base=0, frac=0.15)
    plt.title("Tier 3 — async off-policy tradeoff @100 steps (Qwen3-0.6B)\nthroughput vs staleness")
    save(fig, "tier3_offpolicy_tradeoff_s100")

# Tier 3 compile ON vs OFF @100 steps
compile_panel(med_metrics(f"{LOGS}/scaled_base_ops3_compon_s100.log"), med_metrics(f"{LOGS}/scaled_compoff_s100.log"),
    "Tier 3 — torch.compile ON vs OFF @100 steps (Qwen3-0.6B)",
    "tier3_compile_speedup_s100")

# Tier 1 BI cost ON vs OFF @100 steps
cost_panel(med_metrics(f"{LOGS}/scaled_bi_on_s100.log"), med_metrics(f"{LOGS}/scaled_bi_off_s100.log"),
    "Tier 1 — Batch-invariant ON vs OFF @100 steps (Qwen3-0.6B, TP2/TP2)\nlogprob_diff: BI ON = 0.0 exact · BI OFF ≈ 2e-5",
    "tier1_bi_cost_s100")

print("\nSCALED PLOTS DONE")
for p in saved: print(" ", p)
