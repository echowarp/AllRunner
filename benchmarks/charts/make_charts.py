#!/usr/bin/env python3
"""Render the public benchmark charts from data/published.json (others, as published) and data/ours.json (our runs).
  python3 make_charts.py [--out out] [--only name,...]
Add a chart: write a function chart_<name>(P, O, out) that calls style.frame(...) and style.save(...), and it is
picked up automatically. Add data: edit the two JSON files (keep the source keys)."""
import argparse, json, os, statistics as st
import matplotlib
import numpy as np
import style as S
from style import plt

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    return (json.load(open(os.path.join(HERE, "data", "published.json"))),
            json.load(open(os.path.join(HERE, "data", "ours.json"))))


def _hero_ax(fig, rect):
    ax = fig.add_axes(rect, zorder=3); ax.set_facecolor("none"); ax.grid(False); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.spines["bottom"].set_visible(True); ax.spines["bottom"].set_color(S.HERO_OTHER)
    ax.tick_params(colors=S.HERO_MUTED, length=0, pad=6)
    return ax


def _hero_label(ax, bb, v, fmt, ours):
    ax.annotate(fmt.format(v), (bb.get_x() + bb.get_width() / 2, bb.get_y() + bb.get_height()), xytext=(0, 4),
                textcoords="offset points", ha="center", fontfamily=S.DISPLAY, fontsize=14, fontweight="bold",
                color=S.HERO_TEXT if ours else S.HERO_MUTED)


def hero_bars(fig, rect, labels, vals, ours_mask, fmt, ymin=0):
    ax = _hero_ax(fig, rect)
    cols = [S.HERO_OURS if o else S.HERO_OTHER for o in ours_mask]
    bars = ax.bar(range(len(vals)), [v - ymin for v in vals], 0.62, bottom=ymin, color=cols)
    for bb, v, o in zip(bars, vals, ours_mask):
        _hero_label(ax, bb, v, fmt, o)
    ax.set_xticks(range(len(vals)), labels, fontsize=11.5); ax.set_ylim(ymin, ymin + (max(vals) - ymin) * 1.3)
    return ax


def hero_hbars(fig, rect, labels, vals, ours_mask, fmt, xmin=0):
    """horizontal mini bars for narrow hero cards: label on the left, value at the bar end"""
    ax = _hero_ax(fig, rect); ax.spines["bottom"].set_visible(False); ax.set_xticks([])
    n = len(vals); y = np.arange(n)[::-1]
    hi = max(vals)
    for yy, lab, v, o in zip(y, labels, vals, ours_mask):
        ax.barh([yy], [v - xmin], left=xmin, height=0.62, color=S.HERO_OURS if o else S.HERO_OTHER)
        ax.text(xmin + (v - xmin) + (hi - xmin) * 0.03, yy, fmt.format(v), va="center", fontfamily=S.DISPLAY, fontsize=14,
                fontweight="bold", color=S.HERO_TEXT if o else S.HERO_MUTED)
    ax.set_yticks(y, labels, fontsize=11.5); ax.set_xlim(xmin, xmin + (hi - xmin) * 1.32); ax.set_ylim(-0.6, n - 0.4)
    return ax


def hero_groups(fig, rect, labels, series):
    ax = _hero_ax(fig, rect); x = np.arange(len(labels)); bw = 0.86 / len(series)
    top = max(max(v) for _, v, _ in series)
    for k, (lab, vals, ours) in enumerate(series):
        bars = ax.bar(x + (k - (len(series) - 1) / 2) * bw, vals, bw * 0.9, color=S.HERO_OURS if ours else S.HERO_OTHER, label=lab)
        for bb, v in zip(bars, vals):
            ax.annotate(f"{v:.0f}", (bb.get_x() + bb.get_width() / 2, bb.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", fontfamily=S.DISPLAY, fontsize=11.5, fontweight="bold", color=S.HERO_TEXT if ours else S.HERO_MUTED)
    ax.set_xticks(x, labels, fontsize=11.5); ax.set_ylim(0, top * 1.35)
    return ax


def chart_headline(P, O, out):
    """Hero graphic (dark violet): four claims, each a big number + one line + a mini chart against the published number."""
    import matplotlib.image as mpimg
    fig = plt.figure(figsize=(16, 9))
    S.background(fig)
    logo = mpimg.imread(os.path.join(HERE, "assets", "qwen-logo.png"))
    lh = 0.075; lw = lh * logo.shape[1] / logo.shape[0] * 9 / 16
    la = fig.add_axes([0.04, 0.855, lw, lh]); la.imshow(logo); la.axis("off")
    fig.text(0.04 + lw + 0.012, 0.872, "3.8-27B", fontfamily=S.DISPLAY, fontsize=44, fontweight="bold", color=S.HERO_TEXT)
    fig.text(0.04, 0.815, "At Q4_K_M quality · our runtime on Intel Arc Pro B70 and NVIDIA RTX 5090",
             fontsize=14.5, color=S.HERO_MUTED)
    fig.text(0.04, 0.025, "Qwen and the Qwen logo are trademarks of Alibaba Cloud; not affiliated with or endorsed by Alibaba Cloud or the Qwen team. "
             "Sources and methods: public-benchmarks/SOURCES.md", fontsize=9.5, color=S.HERO_MUTED)

    def tile(x, y, kicker, big, line, w=0.44, bigsize=46):
        h = 0.34
        fig.patches.append(matplotlib.patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.014",
                           transform=fig.transFigure, facecolor=S.HERO_TILE, alpha=S.HERO_TILE_ALPHA, edgecolor="none", zorder=1))
        fig.patches.append(matplotlib.patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.014",
                           transform=fig.transFigure, facecolor="none", edgecolor=S.HERO_OURS, alpha=0.35, lw=1.2, zorder=1))
        fig.text(x + 0.025, y + h - 0.055, kicker.upper(), fontsize=12, fontweight="bold", color=S.HERO_MUTED)
        small = w < 0.4
        fig.text(x + 0.025, y + h - (0.13 if small else 0.155), big, fontfamily=S.DISPLAY, fontsize=bigsize, fontweight="bold", color=S.HERO_TEXT)
        fig.text(x + 0.025, y + h - (0.163 if small else 0.19), line, fontsize=13.5 if small else 14.5, color=S.HERO_OURS_2)

    pub = P["aggregate"]; pc = pub["concurrency"]; a = O["aggregate_arc"]; ac = a["concurrency"]; lv = [1, 2, 4]
    best = [max(v for v in (pub["SparkInfer v0.5.5"][pc.index(u)], pub["vLLM 0.28"][pc.index(u)]) if v) for u in lv]
    ours = [a["tok_s"][ac.index(u)] for u in lv]
    tile(0.04, 0.43, "Value", "B70 beats 5090", "1–4 users · at ~20 % of the price")
    ax = hero_groups(fig, [0.34, 0.47, 0.13, 0.23], [f"{u}" for u in lv],
                     [("best published RTX 5090", best, False), ("ours · Arc B70", ours, True)])
    ax.text(-0.04, -0.075, "users", transform=ax.transAxes, ha="right", va="center", fontsize=10.5, color=S.HERO_MUTED)
    h, l = ax.get_legend_handles_labels()
    leg = fig.legend(h, l, loc="lower left", bbox_to_anchor=(0.062, 0.45), fontsize=11, handlelength=1.2, frameon=False)
    for t in leg.get_texts():
        t.set_color(S.HERO_MUTED)

    agg = O["aggregate_5090"]; t16 = agg["tok_s"][agg["concurrency"].index(16)]
    b16 = [pub["SparkInfer v0.5.5"][pc.index(16)], pub["vLLM 0.28"][pc.index(16)]]
    tile(0.52, 0.43, "NVIDIA RTX 5090", f"{t16:,.0f} tok/s", f"16 users · {t16 / max(b16):.1f}× the best published")
    hero_bars(fig, [0.79, 0.455, 0.15, 0.245], ["SparkInfer", "vLLM", "ours"], b16 + [t16], [False, False, True], "{:,.0f}")

    qt = P["quality_table"]; m = lambda v: st.mean(v)
    W3, X3 = 0.287, [0.04, 0.3565, 0.673]           # bottom row: three narrower cards
    tile(X3[0], 0.06, "Accuracy", "≈ Q4_K_M quality", "same harness · 5-task mean, % of BF16", w=W3, bigsize=34)
    bf = m(qt["BF16"])                                                   # bars start at 0: no exaggerated differences
    hero_hbars(fig, [X3[0] + 0.1, 0.063, 0.155, 0.17], ["BF16", "Q4_K_M", "NVFP4", "ours", "IQ3_XXS (3-bit)"],
               [100.0, 100 * m(qt["Q4_K_M"]) / bf, 100 * m(qt["NVFP4"]) / bf, 100 * m(O["quality_table"]["RTX 5090 (ours)"]) / bf,
                100 * m(qt["UD-IQ3_XXS"]) / bf],
               [False, False, False, True, False], "{:.1f}", xmin=0)

    r = O["agent_replay"]
    tile(X3[1], 0.06, "Real agent traffic", f"{r['stock']['agent_median_s'] / r['ours']['agent_median_s']:.0f}× shorter wait",
         "per agent turn", w=W3, bigsize=34)
    hero_hbars(fig, [X3[1] + 0.085, 0.09, 0.17, 0.09], ["stock vLLM", "ours"], [r["stock"]["agent_median_s"], r["ours"]["agent_median_s"]],
               [False, True], "{:.2f} s")

    g = r["stall_on_others_agent_h"]
    tile(X3[2], 0.06, "Parallel agents", f"{g['stock'] / g['ours']:.0f}× less stalling", "decode lost to others' prefill", w=W3, bigsize=34)
    hero_hbars(fig, [X3[2] + 0.085, 0.08, 0.17, 0.12], ["no cache", "stock vLLM", "ours"], [g["no_cache"], g["stock"], g["ours"]],
               [False, False, True], "{:.1f} h")
    return S.save(fig, out, "headline")


def chart_5090_single_stream(P, O, out):
    w = ["chat", "code", "math", "json"]
    ours = [O["single_5090"]["tok_s"][k] for k in w]; sp = [P["sparkinfer_single"]["tok_s"][k] for k in w]
    fig, ax = S.frame("RTX 5090 · single stream, by workload",
                      "SparkInfer's harness shape: 256 tokens, greedy, thinking off · tokens/s, higher is better",
                      source="ours: RTX 5090 shipped defaults (public prompts, median of 3 reps) · SparkInfer: gittensor model card")
    x = np.arange(len(w)); bw = 0.36
    b1 = ax.bar(x - bw / 2, ours, bw, color=S.OURS, label="Ours (EXL3 4.0bpw + MTP, shipped defaults)")
    b2 = ax.bar(x + bw / 2, sp, bw, color=S.OTHERS[0], label="SparkInfer v0.5.5 + DSpark v2 (NVFP4)")
    S.bar_labels(ax, b1, bold=True, color=S.OURS, size=14); S.bar_labels(ax, b2, color=S.MUTED, size=12)
    for i, k in enumerate(w):
        gain = ours[i] / sp[i] - 1
        ax.annotate(f"+{gain:.0%}", (x[i], max(ours[i], sp[i])), xytext=(0, 30), textcoords="offset points", ha="center",
                    fontfamily=S.DISPLAY, fontsize=16, fontweight="bold", color=S.INK,
                    bbox=dict(boxstyle="round,pad=0.25", fc=S.BADGE, ec="none"))
    ax.set_xticks(x, [k.upper() if k == "json" else k.capitalize() for k in w], fontfamily=S.DISPLAY, fontsize=15)
    ax.set_ylim(0, max(ours) * 1.22); ax.set_ylabel("tokens / s"); ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2)
    return S.save(fig, out, "5090-single-stream")


def chart_5090_aggregate(P, O, out):
    c = O["aggregate_5090"]["concurrency"]
    fig, ax = S.frame("Total throughput vs. concurrent users",
                      "Distinct prompts, 128 output tokens each, 32 requests per level · aggregate tokens/s",
                      source="ours: RTX 5090 shipped defaults (median of 6 reps), Arc B70 shipped defaults (median of 3 reps) · others (RTX 5090): gittensor model card (same test shape)")
    ax.plot(c, O["aggregate_5090"]["tok_s"], "-o", color=S.OURS, lw=3.2, ms=8, label="Ours · RTX 5090", zorder=5)
    if "aggregate_arc" in O:
        ax.plot(O["aggregate_arc"]["concurrency"], O["aggregate_arc"]["tok_s"], "o", ls=(0, (5, 3)), color=S.OURS_2, lw=2.4, ms=6,
                label="Ours · Arc Pro B70 (for reference)", zorder=4)
    for (name, col) in (("vLLM 0.28", S.OTHERS[0]), ("SparkInfer v0.5.5", S.OTHERS[3])):
        ys = P["aggregate"][name]; xs = [a for a, b in zip(c, ys) if b is not None]; ys = [b for b in ys if b is not None]
        ax.plot(xs, ys, "-o", color=col, lw=2.2, ms=6, label=f"{name} · RTX 5090 (published)")
    ax.set_xscale("log", base=2); ax.set_xticks(c, [str(v) for v in c]); ax.set_xlabel("concurrent requests")
    ax.set_ylabel("tokens / s (all users)"); ax.set_ylim(0, 1500)
    i16 = c.index(16); v16 = O["aggregate_5090"]["tok_s"][i16]
    best = max(v for v in (P["aggregate"]["vLLM 0.28"][i16], P["aggregate"]["SparkInfer v0.5.5"][i16]) if v)
    ax.annotate(f"{v16:,.0f} tok/s at 16 users", (16, v16), xytext=(12, -22), textcoords="offset points", ha="left", va="top",
                fontfamily=S.DISPLAY, fontsize=18, fontweight="bold", color=S.OURS)
    ax.annotate(f"{v16 / best:.1f}× the best published", (16, v16), xytext=(12, -52), textcoords="offset points",
                ha="left", va="top", fontsize=12, color=S.INK)
    ax.legend(loc="upper left", handlelength=4)
    return S.save(fig, out, "5090-aggregate")


def chart_arc_panel(P, O, out):
    c = O["arc_panel"]["concurrency"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 6.2), sharey=True)
    plt.close(fig)
    fig, ax0 = S.frame("Arc Pro B70 · upstream exl3xpu's own benchmark",
                       "Their script and prompts (sweep.py --corpus real, thinking on, T=0.7) · aggregate decode tokens/s",
                       source="ours: runs xpub-curve-20261112 + -r2 (production engine, mean of 2 reps) · upstream: 0xSero/exl3xpu README (C1-C16 published)")
    ax0.remove()
    for j, cls in enumerate(("prose", "code")):
        ax = fig.add_axes([0.075 + j * 0.47, 0.17, 0.42, 0.59])
        x = np.arange(len(c)); bw = 0.38
        b1 = ax.bar(x - bw / 2, O["arc_panel"][cls], bw, color=S.OURS, label="Ours")
        b2 = ax.bar(x + bw / 2, [v if v is not None else np.nan for v in P["exl3xpu_panel"][cls]], bw, color=S.OTHERS[0], label="exl3xpu upstream (published)")
        S.bar_labels(ax, b1, bold=True, color=S.OURS, size=12); S.bar_labels(ax, b2, color=S.MUTED, size=11)
        ax.set_xticks(x, [str(v) for v in c], fontsize=11.5); ax.set_xlabel("concurrent streams")
        ax.set_title(cls.capitalize(), loc="left", fontfamily=S.DISPLAY, fontsize=17, fontweight="bold", color=S.INK)
        ax.set_ylim(0, 600); ax.grid(axis="x", visible=False)
        if j == 0:
            ax.set_ylabel("tokens / s"); ax.legend(loc="upper left")
        else:
            ax.tick_params(labelleft=False)
    return S.save(fig, out, "arc-upstream-panel")


def _msd(v, n_items=None):
    """mean of the repeats and its error bar: ±1 standard error of the mean, binomial over all answered items
    (n_items x repeats); run-to-run spread is too few runs to estimate (2 identical GPQA runs would show 0)."""
    if not v:
        return (None, 0)
    m = st.mean(v)
    if not n_items:
        return (m, 0)
    p = m / 100
    return (m, 100 * (p * (1 - p) / (n_items * len(v))) ** 0.5)


N_ITEMS = {"GPQA Diamond": 198, "AIME 2025": 30}


def chart_arc_prefill(P, O, out):
    pub = P["arc_prefill_32k"]["tok_s"]; ours = O["arc_panel"]["prefill"]["32k"]
    rows = [("ours", ours, S.OURS)] + [(k, v, S.OTHERS[i]) for i, (k, v) in enumerate(pub.items())]
    fig, ax = S.frame("Arc Pro B70 · prefill a 32k-token prompt",
                      "Cold prompt processing, tokens/s · higher is better",
                      source="ours: production launcher, upstream exl3xpu sweep.py prefill phase · others as published (SOURCES.md, Arc table)")
    y = np.arange(len(rows))[::-1]
    for (k, v, c), yv in zip(rows, y):
        ax.barh([yv], [v], color=c, height=0.62)
        ax.text(v + 40, yv, f"{v:,.0f}", va="center", fontfamily=S.DISPLAY, fontsize=24, fontweight="bold",
                color=S.OURS if k == "ours" else S.MUTED)
        ax.text(-60, yv, k, va="center", ha="right", fontsize=14, color=S.INK)
    ax.set_yticks([]); ax.set_xticks([]); ax.grid(False); ax.spines["bottom"].set_visible(False)
    ax.set_xlim(0, ours * 1.18); ax.set_position([0.30, 0.16, 0.66, 0.56])
    up = pub["exl3xpu upstream (EXL3 4.0bpw)"]; lc = pub["llama.cpp SYCL (Q4_K_M)"]
    ax.text(ours, y[0] - 0.5, f"{ours / up:.2f}× upstream exl3xpu · {ours / lc:.1f}× llama.cpp",
            ha="right", va="top", fontsize=13, color=S.OURS_2)
    return S.save(fig, out, "arc-prefill")


def chart_quality(P, O, out):
    """The one quality card: same-harness accuracy, distance from BF16 (KL), Qwen's own benchmarks."""
    qt = P["quality_table"]; m = st.mean
    fig, ax0 = S.frame("Q4_K_M-level quality",
                       "Three independent checks · EXL3 4.0 bpw weights · int4 KV = 1.8× the cache of int8 (480k vs 264k tokens on a 32 GB RTX 5090)",
                       figsize=(16, 9),
                       source="left: llm-bench-rig, same harness as the published rows · middle: our 24-chat set vs BF16 · right: Qwen3.8-27B card (BF16 CI = one 198-question run); ours = RTX 5090 + Arc B70 runs pooled")
    ax0.remove()

    def panel(x, title, sub):
        ax = fig.add_axes([x, 0.16, 0.27, 0.54])
        ax.set_title(title, loc="left", fontfamily=S.DISPLAY, fontsize=24, fontweight="bold", color=S.INK, pad=34)
        ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=12, color=S.MUTED)
        ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        return ax

    def hbars(ax, rows, fmt, lo, hi, errs=None):
        y = np.arange(len(rows))[::-1] * 1.0
        errs = errs or [0] * len(rows)
        for (k, v, ours), yv, e in zip(rows, y, errs):
            col = (S.OURS if ours == 1 else S.OURS_2) if ours else S.OTHERS[0]
            ax.barh([yv], [v - lo], left=lo, color=col, height=0.5)
            if e:
                ax.errorbar([v], [yv], xerr=[e], fmt="none", capsize=6, ecolor=S.INK, elinewidth=1.6, alpha=0.9 if ours else 0.6)
            ax.text(v + e + (hi - lo) * 0.02, yv, fmt.format(v) + (f" ± {e:.1f}" if e else ""), va="center", fontfamily=S.DISPLAY,
                    fontsize=18, fontweight="bold", color=S.INK if ours else S.MUTED)
            ax.text(lo, yv + 0.3, k, va="bottom", fontsize=11.5, color=S.INK if ours else S.MUTED)
        ax.set_xlim(lo, hi + (hi - lo) * 0.12); ax.set_ylim(-0.5, len(rows) - 0.2)

    a = panel(0.05, "Accuracy", "5-task mean, same harness · axis from 90")
    kv = O["kv_capacity"]["RTX 5090"]; k8 = f"{kv['int8'] / 1000:.0f}k"; k4 = f"{kv['int4'] / 1000:.0f}k"
    hbars(a, [("BF16", m(qt["BF16"]), 0), ("Q4_K_M", m(qt["Q4_K_M"]), 0), ("NVFP4", m(qt["NVFP4"]), 0),
              (f"ours · int8 KV · {k8}-token cache", m(O["quality_table"]["RTX 5090 int8 KV (ours)"]), 2),
              (f"ours · int4 KV (production) · {k4}-token cache", m(O["quality_table"]["RTX 5090 (ours)"]), 1)], "{:.1f}", 90, 94)

    f = O["fidelity_private_set"]["rows"]
    b = panel(0.375, "Distance from BF16", "mean KL divergence · lower is closer")
    hbars(b, [("Q4_K_M · 16-bit KV (llama.cpp default)", f["Q4_K_M (llama.cpp)"], 0), (f"ours · int4 KV (production) · {k4}-token cache", f["Ours, int4 KV (production)"], 1),
              (f"ours · int8 KV · {k8}-token cache", f["Ours, int8 KV"], 2)], "{:.4f}", 0, 0.015)

    c = panel(0.70, "Reasoning & instructions", "GPQA Diamond, IFBench · thinking on · 95 % CI · axis from 60")
    ob = P["official_bf16"]
    ci = lambda p, n: 196 * (p / 100 * (1 - p / 100) / n) ** 0.5
    gr = O["official"]["GPQA Diamond"]["repeats"] + O["uncapped_arc"]["GPQA Diamond"]["repeats"]; g = m(gr)
    il = m([O["official"]["IFBench 5090"]["loose"], O["official"]["IFBench"]["loose"]])
    is_ = m([O["official"]["IFBench 5090"]["strict"], O["official"]["IFBench"]["strict"]])
    hbars(c, [("GPQA Diamond · official BF16", ob["GPQA Diamond"], 0), ("GPQA Diamond · ours", g, 1),
              ("IFBench · official BF16 (metric not stated)", ob["IFBench"], 0), ("IFBench · ours, loose", il, 1),
              ("IFBench · ours, strict", is_, 1)], "{:.1f}", 60, 104,
          errs=[ci(ob["GPQA Diamond"], 198), ci(g, 198 * len(gr)), 0, ci(il, 600), ci(is_, 600)])
    return S.save(fig, out, "quality")


def chart_quality_32k(P, O, out):
    pub = P["gittensor_quality_32k"]; ours = O["matched_32k"]
    names = ["GPQA Diamond", "AIME 2025"]; builds = ["gittensor NVFP4", "NVIDIA NVFP4", "QUASAR NVFP4"]
    fig, ax = S.frame("Same 32k thinking budget as the published NVFP4 runs",
                      "RTX 5090, thinking on, T=1.0 / top-k 20 / top-p 0.95, 32,768-token budget per answer · accuracy %",
                      source=f"ours: RTX 5090 production engine (EXL3 4.0bpw, int4 KV), GPQA x{len(ours['GPQA Diamond']['repeats'])}, "
                             f"AIME x{len(ours['AIME 2025']['repeats'])}, mean ±1 s.e. (binomial, all answers) · NVFP4: gittensor-model-hub/Qwen3.8-27B-NVFP4 card")
    x = np.arange(len(names)); bw = 0.2
    rows = [(b, [pub[n][b] for n in names], S.OTHERS[k]) for k, b in enumerate(builds)]
    rows.append(("Ours · RTX 5090", [_msd(ours[n]["repeats"], N_ITEMS[n])[0] for n in names], S.OURS))
    for k, (lab, vals, col) in enumerate(rows):
        xs = x + (k - (len(rows) - 1) / 2) * bw
        ax.bar(xs, vals, bw * 0.92, color=col, label=lab)
        for xi, v, n in zip(xs, vals, names):
            e = _msd(ours[n]["repeats"], N_ITEMS[n])[1] if "Ours" in lab else 0
            if e:
                ax.errorbar([xi], [v], yerr=[e], fmt="none", capsize=5, ecolor=S.INK, elinewidth=1.4)
            ax.annotate(f"{v:.1f}", (xi, v + e), xytext=(0, 4), textcoords="offset points", ha="center", fontfamily=S.DISPLAY,
                        fontsize=14 if "Ours" in lab else 12, fontweight="bold" if "Ours" in lab else "normal",
                        color=S.OURS if "Ours" in lab else S.MUTED)
    ax.set_xticks(x, names, fontfamily=S.DISPLAY, fontsize=16); ax.set_ylim(60, 90); ax.grid(axis="x", visible=False)
    ax.set_ylabel("accuracy % (axis starts at 60)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=4)
    ax.set_position([0.075, 0.16, 0.9, 0.6])
    return S.save(fig, out, "quality-32k-vs-nvfp4")


def chart_task_time(P, O, out):
    r = O["task_replay"]; m = r["median_min"]
    sp = m["RTX 5090 · best published"] / m["RTX 5090 · ours"]
    fig, ax0 = S.frame(f"Multi-agent tasks finish {sp:.1f}× sooner",
                       f"{r['tasks']} real multi-agent tasks replayed on each engine · median time the task waits on the model",
                       figsize=(16, 9),
                       source="modeled replay: same tasks, same tool/user gaps; decode from each engine's measured or published throughput curve, prefill from its cache rules\n"
                              f"calibration: the model gives {m['RTX 5090 · ours']:.1f} min for our RTX 5090, measured {r['measured_5090_ours']['median_min']:.1f} min · "
                              "best published = best published RTX 5090 result at each concurrency (SGLang + DFlash2, SparkInfer, vLLM 0.28)")
    ax0.remove()

    def panel(x, title, rows):
        ax = fig.add_axes([x, 0.2, 0.4, 0.48])
        ax.set_title(title, loc="left", fontfamily=S.DISPLAY, fontsize=22, fontweight="bold", color=S.INK, pad=14)
        ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        hi = max(v for _, v, _ in rows); y = np.arange(len(rows))[::-1]
        for (k, v, ours), yv in zip(rows, y):
            ax.barh([yv], [v], color=S.OURS if ours else S.OTHERS[0], height=0.5)
            ax.text(v + hi * 0.02, yv, f"{v:.1f} min", va="center", fontfamily=S.DISPLAY, fontsize=22, fontweight="bold",
                    color=S.INK if ours else S.MUTED)
            ax.text(0, yv + 0.31, k, va="bottom", fontsize=13, color=S.INK if ours else S.MUTED)
        ax.set_xlim(0, hi * 1.3); ax.set_ylim(-0.5, len(rows) - 0.2)

    panel(0.06, "NVIDIA RTX 5090", [("best published", m["RTX 5090 · best published"], 0), ("vLLM 0.28", m["RTX 5090 · vLLM 0.28"], 0),
                                    ("ours", m["RTX 5090 · ours"], 1)])
    panel(0.54, "Intel Arc Pro B70", [("exl3xpu upstream", m["Arc B70 · exl3xpu upstream"], 0), ("ours", m["Arc B70 · ours"], 1)])
    return S.save(fig, out, "task-time")


def chart_system(P, O, out):
    """Stylized system diagram: two cartoon GPUs, the shared RAM tier between them, combined KV capacity, highlights."""
    from matplotlib.patches import FancyBboxPatch, Circle, Wedge, Rectangle, FancyArrowPatch
    kv = O["kv_capacity"]; k5 = kv["RTX 5090"]["int4_production"]; ka = kv["Arc B70"]["int4_production"]   # today's production engines
    rt = O["ram_tier"]; ram_tok = rt["slots"] * rt["tokens_per_slot"]       # pool slots x tokens per slot (upper bound)
    total = k5 + ka + ram_tok
    fig = plt.figure(figsize=(16, 9)); S.background(fig)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")
    W = S.INK; M = S.MUTED; V = S.OURS; V2 = S.OURS_2
    ax.text(0.65, 8.2, "Two GPUs, one cache", fontfamily=S.DISPLAY, fontsize=40, fontweight="bold", color=W)
    ax.text(0.65, 7.72, "Qwen3.8-27B · our runtime on an NVIDIA RTX 5090 and an Intel Arc Pro B70, sharing one RAM-tier KV cache",
            fontsize=14.5, color=M)

    def card(x, y, w, h, bracket_left=True):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.18", fc="#1E1838", ec="#4B4290", lw=2.2, zorder=3))
        ax.add_patch(Rectangle((x + 0.15, y + h), w - 0.3, 0.1, fc="#3A3270", ec="none", zorder=3))
        # PCIe x16 edge connector (gold fingers + key notch) under the card
        fx = x + 0.9
        ax.add_patch(Rectangle((fx - 0.08, y - 0.06), 2.7, 0.06, fc="#3A3270", ec="none", zorder=3))
        for i in range(26):
            if i == 4: continue                                              # key notch
            ax.add_patch(Rectangle((fx + i * 0.1, y - 0.24), 0.065, 0.18, fc="#C9A94A", ec="none", zorder=3))
        # I/O bracket at the end of the card
        bx = x - 0.16 if bracket_left else x + w + 0.04
        ax.add_patch(Rectangle((bx, y - 0.35), 0.12, h + 0.55, fc="#8F8AB8", ec="none", zorder=2))
        for j in range(4):
            ax.add_patch(Rectangle((bx + 0.02, y + 0.2 + j * (h - 0.3) / 4), 0.08, 0.16, fc="#1E1838", ec="none", zorder=3))
        return fx + 1.3, y - 0.3                                             # connector centre (arrow anchor)

    def fan(cx, cy, r, glow):
        ax.add_patch(Circle((cx, cy), r * 1.08, fc="none", ec=glow, lw=3, zorder=4))
        ax.add_patch(Circle((cx, cy), r, fc="#161126", ec="#3A3270", lw=1.5, zorder=4))
        for k in range(7):
            ax.add_patch(Wedge((cx, cy), r * 0.92, k * 360 / 7, k * 360 / 7 + 30, width=r * 0.62, fc="#2A2350", ec="none", zorder=5))
        ax.add_patch(Circle((cx, cy), r * 0.28, fc="#0E0A24", ec=glow, lw=1.6, zorder=6))

    def gpu_text(x, name, tok, col, eng):
        ax.text(x, 6.75, name, fontfamily=S.DISPLAY, fontsize=20, fontweight="bold", color=W)
        ax.text(x, 3.95, f"{tok / 1000:,.0f}k", fontfamily=S.DISPLAY, fontsize=34, fontweight="bold", color=col)
        ax.text(x, 3.6, "tokens of KV cache in VRAM", fontsize=12.5, color=M)
        ax.text(x, 3.25, eng, fontsize=11.5, color=M)

    # RTX 5090: chunky triple-fan card (left)
    c5 = card(0.75, 4.65, 4.9, 1.85)
    for i in range(3):
        fan(1.6 + i * 1.6, 5.57, 0.68, V)
    gpu_text(0.75, "NVIDIA RTX 5090 · 32 GB", k5, V, "vLLM CUDA + exl3cuda")
    # Arc Pro B70: slim blower card (right)
    ca = card(10.35, 4.85, 4.9, 1.35)
    for i in range(6):
        ax.plot([10.7 + i * 0.36, 11.2 + i * 0.36], [5.05, 6.0], color="#2E2758", lw=5, solid_capstyle="round", zorder=4)
    fan(14.4, 5.52, 0.52, V2)
    gpu_text(10.35, "Intel Arc Pro B70 · 32 GB", ka, V2, "vLLM XPU + exl3xpu (custom ESIMD kernels)")

    # combined capacity badge (centre, between the cards)
    ax.add_patch(FancyBboxPatch((6.3, 4.75), 3.4, 1.65, boxstyle="round,pad=0,rounding_size=0.2", fc="#2A1F5E", ec=V, lw=2, zorder=3))
    ax.text(8.0, 5.65, f"{total / 1e6:.1f}M tok", ha="center", fontfamily=S.DISPLAY, fontsize=42, fontweight="bold", color=W, zorder=4)
    ax.text(8.0, 5.0, "KV cache, combined", ha="center", fontsize=12.5, color=M, zorder=4)

    # shared RAM tier: a DIMM, centre, right under the combined badge
    rx, ry, rw, rh = 5.9, 2.75, 4.2, 0.8
    ax.add_patch(FancyBboxPatch((rx, ry), rw, rh, boxstyle="round,pad=0,rounding_size=0.08", fc="#2A1F5E", ec=V, lw=2.2, zorder=3))
    for i in range(7):
        ax.add_patch(Rectangle((rx + 0.27 + i * 0.56, ry + 0.17), 0.4, 0.46, fc="#120D2B", ec="#4B4290", lw=1, zorder=4))
    for i in range(19):
        ax.add_patch(Rectangle((rx + 0.17 + i * 0.205, ry - 0.14), 0.11, 0.14, fc="#C9A94A", ec="none", zorder=3))
    ax.text(rx + rw / 2, 4.25, f"Shared RAM tier · {rt['pool_gib']} GiB · ~{ram_tok / 1e6:.2f}M tokens", ha="center",
            fontfamily=S.DISPLAY, fontsize=19, fontweight="bold", color=W)
    ax.text(rx + rw / 2, 3.88, "a prompt prefilled on one GPU is reused by the other", ha="center", fontsize=12, color=V2)
    # PCIe links GPU <-> RAM
    ax.add_patch(FancyArrowPatch((2.6, 3.0), (rx - 0.05, ry + rh / 2), arrowstyle="<|-|>", mutation_scale=22, lw=3, color=V,
                                 connectionstyle="arc3,rad=0.3", zorder=2))
    ax.add_patch(FancyArrowPatch((13.4, 3.0), (rx + rw + 0.05, ry + rh / 2), arrowstyle="<|-|>", mutation_scale=22, lw=3, color=V2,
                                 connectionstyle="arc3,rad=-0.3", zorder=2))
    ax.text(3.7, 2.35, "PCIe", fontsize=12, color=M); ax.text(11.85, 2.35, "PCIe", fontsize=12, color=M)
    # SSD tier: an M.2 2280 stick below the RAM (off on today's int4 broker, planned again -> tagged "next")
    mx, my, mw, mh = 6.5, 0.98, 3.0, 0.83
    ax.add_patch(FancyBboxPatch((mx, my), mw, mh, boxstyle="round,pad=0,rounding_size=0.04", fc="#1F3A2E", ec="#3E6B55", lw=1.4, zorder=3))
    # edge connector on the left short side: gold pads with the M-key notch
    for i in range(12):
        if i == 8: continue
        ax.add_patch(Rectangle((mx - 0.16, my + 0.06 + i * 0.062), 0.16, 0.038, fc="#C9A94A", ec="none", zorder=3))
    # half-moon mounting notch on the right short side
    ax.add_patch(Wedge((mx + mw, my + mh / 2), 0.13, 90, 270, fc=S.BG_BOTTOM, ec="none", zorder=4))
    # controller, DRAM, two NAND packages, label sticker
    ax.add_patch(Rectangle((mx + 0.22, my + 0.2), 0.5, 0.43, fc="#121212", ec="#3E6B55", lw=0.8, zorder=4))     # controller
    ax.add_patch(Rectangle((mx + 0.84, my + 0.27), 0.3, 0.29, fc="#121212", ec="#3E6B55", lw=0.8, zorder=4))    # DRAM
    for k in range(2):                                                                                          # NAND
        ax.add_patch(Rectangle((mx + 1.27 + k * 0.72, my + 0.13), 0.62, 0.57, fc="#121212", ec="#3E6B55", lw=0.8, zorder=4))
    ax.add_patch(FancyArrowPatch((rx + rw / 2, ry - 0.2), (rx + rw / 2, my + mh + 0.06), arrowstyle="<|-|>", mutation_scale=14, lw=1.6,
                                 color=M, zorder=2))
    ax.text(mx + mw + 0.3, my + mh / 2, "SSD: keeps system prompts hot across restarts", va="center", fontsize=12.5, color=W)
    ax.add_patch(FancyBboxPatch((mx - 1.0, my + 0.27), 0.62, 0.3, boxstyle="round,pad=0,rounding_size=0.1", fc=V, ec="none", zorder=5))
    ax.text(mx - 0.69, my + 0.42, "next", ha="center", va="center", fontsize=10, color="#140E33", fontweight="bold", zorder=6)

    # highlights strip (fixed-width chips)
    hl = ["int4 KV: 1.8× int8's tokens", "token-exact tail reuse", "vision + LoRA on demand",
          "MTP speculation", "Q4_K_M quality", "one endpoint, both GPUs"]
    cw = 2.35
    for i, h in enumerate(hl):
        x = 0.65 + i * (cw + 0.12)
        ax.add_patch(FancyBboxPatch((x, 0.25), cw, 0.5, boxstyle="round,pad=0,rounding_size=0.2", fc="#2A1F5E", ec="none", zorder=3))
        ax.text(x + cw / 2, 0.5, h, ha="center", va="center", fontsize=11, color=W, zorder=4)
    return S.save(fig, out, "system")


def chart_tail_reuse(P, O, out):
    """Explainer: what a tool-result turn recomputes under stock vLLM's block-rounded cache vs our token-exact tail hold."""
    from matplotlib.patches import FancyBboxPatch, Rectangle
    r = O["agent_replay"]
    fig = plt.figure(figsize=(16, 9)); S.background(fig)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")
    W, M, V, V2 = S.INK, S.MUTED, S.OURS, S.OURS_2
    RED = "#FF7A9A"
    ax.text(0.65, 8.2, "Tool results: prefill only what's new", fontfamily=S.DISPLAY, fontsize=40, fontweight="bold", color=W)
    ax.text(0.65, 7.72, "One agent turn after a tool call · what must be prefilled before the first token", fontsize=14.5, color=M)

    # illustrative turn (tokens): history before the previous prompt's end, previous reply, new tool result
    B = 832; hist = 30000; reply = 1152; tool = 1899        # a median real tool-result turn: 3,931 / 1,899 recomputed
    L1 = hist                                  # previous request's prompt length
    stock_keep = (L1 // B - 1) * B             # stock: last full block of the previous prompt, minus the drafter's block
    zoom0 = stock_keep - 2 * B                 # zoom window start (tokens)
    total = hist + reply + tool
    X0, X1 = 3.6, 15.3                         # zoomed axis span (figure units)
    sx = lambda tkn: X0 + (tkn - zoom0) / (total - zoom0) * (X1 - X0)

    HELD = "#6A62F2"; CACHED = "#3B3480"
    ys, yo = 6.15, 4.3
    Tend = total + 700                                          # a little open space after the tool result

    def head(y, label, note, col):
        ax.text(0.65, y + 0.6, label, fontfamily=S.DISPLAY, fontsize=22, fontweight="bold", color=W)
        ax.add_patch(FancyBboxPatch((0.65, y - 0.35), 2.7, 0.7, boxstyle="round,pad=0,rounding_size=0.08", fc=CACHED, ec="none", zorder=3))
        ax.text(2.0, y, "cached · system prompt, tools, history", ha="center", va="center", fontsize=10.5, color=W, zorder=4)
        ax.text(3.47, y, "···", ha="center", va="center", fontsize=16, color=M)
        ax.text(2.6, y + 0.6, note, fontfamily=S.DISPLAY, fontsize=19, fontweight="bold", color=col)

    def blocks(y, bsz, segs, partial_at, partial_label, partial_col):
        """block containers of bsz tokens (outlined, separated); token segments [(a, b, colour)] filled inside them"""
        k = (zoom0 // bsz) * bsz
        while k < Tend:
            ka, kb = max(k, zoom0), min(k + bsz, Tend)
            x0, x1 = sxe(ka) + 0.04, sxe(kb) - 0.04
            open_end = k + bsz > Tend
            ax.add_patch(FancyBboxPatch((x0, y - 0.36), x1 - x0, 0.72, boxstyle="round,pad=0,rounding_size=0.07", fc="none",
                                        ec="#8F88FF", lw=1.4, ls="--" if open_end else "-", zorder=5))
            for sa, sb, col in segs:
                fa, fb = max(sa, ka), min(sb, kb)
                if fa < fb:
                    ax.add_patch(Rectangle((max(x0, sxe(fa)) + 0.03, y - 0.3), min(x1, sxe(fb)) - max(x0, sxe(fa)) - 0.06, 0.6,
                                           fc=col, ec="none", zorder=4))
            if k <= partial_at < k + bsz:
                ax.annotate(partial_label, (sxe(partial_at), y + 0.36), xytext=(0, 14), textcoords="offset points", ha="center",
                            fontsize=11.5, color=partial_col, arrowprops=dict(arrowstyle="-", color=partial_col, lw=1.2), zorder=6)
            k += bsz

    sxe = lambda tkn: X0 + (tkn - zoom0) / (Tend - zoom0) * (X1 - X0)
    # stock vLLM: 832-token blocks; reuse stops at stock_keep, the previous prompt's partial last block can't be reused
    head(ys, "stock vLLM", f"recompute {hist + reply - stock_keep:,}", RED)
    t0_ = ax.texts[-1]; fig.canvas.draw()
    bb = t0_.get_window_extent().transformed(ax.transData.inverted())
    ax.text(bb.x1 + 0.12, ys + 0.6, f"+ new {tool:,} tokens", fontfamily=S.DISPLAY, fontsize=19, fontweight="bold", color=V)
    blocks(ys, B, [(zoom0, stock_keep, CACHED), (stock_keep, hist + reply, RED), (hist + reply, total, V)], -1, "", RED)
    # ours: 3,200-token blocks; the partial block at the end of the previous turn is held to the exact token
    BO = 3200
    head(yo, "ours", f"recompute 0 + new {tool:,} tokens", V)
    blocks(yo, BO, [(zoom0, hist + reply, CACHED), (hist + reply, total, V)], -1, "", V2)
    sx = sxe

    def flag(tkn, y, col, label, dx=0, ha="center"):
        """GDN state checkpoint: a flag on top of the bar at the exact token where decoding can resume"""
        x = sxe(tkn)
        ax.plot([x, x], [y - 0.36, y + 0.62], color=col, lw=2.2, zorder=6)
        ax.add_patch(matplotlib.patches.Polygon([(x, y + 0.62), (x + 0.32, y + 0.52), (x, y + 0.42)], closed=True, fc=col, ec="none", zorder=6))
        ax.text(x + 0.42 + dx, y + 0.52, label, ha=ha, va="center", fontsize=11.5, color=col, zorder=6) if ha == "left" else None

    saved = (hist // B) * B                                     # stock: GDN state only at the previous prompt's last full block
    flag(saved, ys, RED, "GDN checkpoint: previous prompt's last full block", ha="left")
    ax.annotate("", (sxe(stock_keep), ys - 0.42), (sxe(saved), ys - 0.42), arrowprops=dict(arrowstyle="->", color=RED, lw=1.5), zorder=6)
    ax.text((sxe(stock_keep) + sxe(saved)) / 2, ys - 0.52, "−1 block (drafter)", ha="center", va="top", fontsize=10.5, color=RED, fontfamily=S.BODY)
    flag(hist + reply, yo, V2, "GDN checkpoint at the exact last token (tail hold)", ha="left")
    for yy in (ys, yo):                                          # where the history, reply and tool result meet
        for tkn in (hist, hist + reply):
            ax.plot([sxe(tkn)] * 2, [yy - 0.3, yy + 0.3], color="#0E0A24", lw=2.5, zorder=4.5)
    # segment labels between the rows
    yl = ys - 0.66
    for a, b_, lab in ((zoom0, hist, "history"), (hist, hist + reply, "previous reply"), (hist + reply, total, "new tool result")):
        ax.text((sx(a) + sx(b_)) / 2, yl, lab, ha="center", va="center", fontsize=12.5, color=W)

    # measured-on-traffic anchor

    # how the cached head stays cached: prefix chains in the RAM tier, only tips evicted, tool-wait holds
    LEAF2 = RED
    ax.add_patch(FancyBboxPatch((0.65, 1.45), 2.9, 1.5, boxstyle="round,pad=0,rounding_size=0.14", fc=V, ec="none", zorder=3))
    ax.text(2.1, 2.3, "shared baseline", ha="center", fontfamily=S.DISPLAY, fontsize=21, fontweight="bold", color="#140E33", zorder=4)
    ax.text(2.1, 1.82, "system prompt · tools · docs", ha="center", fontsize=12, color="#140E33", zorder=4)
    for (name, y, n) in (("agent A", 2.75, 6), ("agent B", 2.2, 4), ("agent C", 1.65, 7)):
        x0 = 4.15
        ax.plot([3.55, x0 - 0.05], [2.2, y], color=V2, lw=1.8, zorder=2)
        for i in range(n):
            x = x0 + i * 0.52; tip = i == n - 1; held = tip and name == "agent B"
            if tip:                                                   # partial tail: outlined block, partly filled
                col = V2 if held else LEAF2
                ax.add_patch(FancyBboxPatch((x, y - 0.2), 0.42, 0.4, boxstyle="round,pad=0,rounding_size=0.06",
                                            fc="none", ec=col, lw=1.8, zorder=4))
                ax.add_patch(Rectangle((x + 0.04, y - 0.16), 0.42 * (0.55 if held else 0.4), 0.32, fc=col, ec="none", zorder=3))
            else:
                ax.add_patch(FancyBboxPatch((x, y - 0.2), 0.42, 0.4, boxstyle="round,pad=0,rounding_size=0.06",
                                            fc="#3B3480", ec="none", zorder=3))
            if held:
                ax.text(x + 0.6, y, "held partial tail: awaiting tool result", va="center", fontsize=12, color=V2)
            if tip and name == "agent C":
                ax.text(x + 0.6, y, "evictable partial tail", va="center", fontsize=12, color=LEAF2)
    ax.text(9.7, 2.6, "Cached prompts are chains of blocks.", fontfamily=S.DISPLAY, fontsize=21, fontweight="bold", color=W)
    ax.text(9.7, 2.15, "Only tips are evicted: a shared baseline is never dropped.", fontsize=13.5, color=M)
    ax.text(9.7, 1.78, "A turn ending in a tool call is held 5 min for its result", fontsize=13.5, color=M)
    ax.text(9.7, 1.45, "(1 min otherwise), then spills to the RAM tier.", fontsize=13.5, color=M)

    hl = ["token-exact tail hold", "GDN state saved at the exact token", "chain-aware eviction + tool-wait holds",
          "RAM tier: evicted prefixes load over PCIe"]
    cw = 3.55
    for i, h in enumerate(hl):
        x = 0.65 + i * (cw + 0.12)
        ax.add_patch(FancyBboxPatch((x, 0.62), cw, 0.5, boxstyle="round,pad=0,rounding_size=0.2", fc="#2A1F5E", ec="none", zorder=3))
        ax.text(x + cw / 2, 0.87, h, ha="center", va="center", fontsize=11.5, color=W, zorder=4)
    ax.text(0.65, 0.25, f"a median tool-result turn (modeled on {r['tool_turns']:,} real turns): {reply:,}-token reply, {tool:,}-token tool result · stock block size "
            f"{B} tokens (16-bit KV, estimated): reuse stops one block before the previous prompt's last full block", fontsize=10, color=M)
    return S.save(fig, out, "tail-reuse")


def chart_cache_production(P, O, out):
    r = O["agent_replay"]; st_, ou = r["stock"], r["ours"]
    fig, ax0 = S.frame(f"Real agent traffic: {st_['agent_median_s'] / ou['agent_median_s']:.0f}× shorter wait per turn",
                       f"{r['agent_requests']:,} production agent requests · median prefill wait, and the stall it causes for other agents",
                       figsize=(16, 9),
                       source="same replay for both: stock vLLM prefix-cache rules + 16-bit KV vs ours (token-exact tail reuse, int4 KV; RAM tier not modeled)\n"
                              "wait from the measured cold-prefill curve; stall = wait x agents in flight at the time (modeled, not timed)")
    ax0.remove()

    def panel(x, title, rows, hi, fmt="{:.2f} s"):
        ax = fig.add_axes([x, 0.2, 0.25, 0.48])
        ax.set_title(title, loc="left", fontfamily=S.DISPLAY, fontsize=22, fontweight="bold", color=S.INK, pad=14)
        ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        y = np.arange(len(rows))[::-1]
        for (k, v, ours), yv in zip(rows, y):
            col = S.OURS if ours else S.OTHERS[0]
            ax.barh([yv], [v], color=col, height=0.5)
            ax.text(v + hi * 0.02, yv, fmt.format(v), va="center", fontfamily=S.DISPLAY, fontsize=22, fontweight="bold",
                    color=S.INK if ours else S.MUTED)
            ax.text(0, yv + 0.31, k, va="bottom", fontsize=13, color=S.INK if ours else S.MUTED)
        ax.set_xlim(0, hi * 1.3); ax.set_ylim(-0.5, len(rows) - 0.2)

    hi = max(st_["agent_median_s"], st_["tool_median_s"])
    panel(0.05, "Every agent turn", [("stock vLLM", st_["agent_median_s"], 0), ("ours", ou["agent_median_s"], 1)], hi)
    panel(0.37, "After a tool result", [("stock vLLM", st_["tool_median_s"], 0), ("ours", ou["tool_median_s"], 1)], hi)
    g = r["stall_on_others_agent_h"]
    rows = [("no cache", g["no_cache"], 0), ("stock vLLM", g["stock"], 0), ("ours", g["ours"], 1)]
    panel(0.69, "Stall on other agents", rows, g["no_cache"], "{:.1f} h")
    fig.text(0.69, 0.165, "agent-hours of decode lost to others' prefill", fontsize=13, color=S.OURS_2)
    return S.save(fig, out, "cache-production")


def chart_on_demand(P, O, out):
    """Reserve nothing, borrow everything: a typical static layout (vision + LoRA loaded at start, peak reserves) vs one
    KV pool that elastic memory borrows from (schematic, RTX 5090 budget to scale), and how an image borrows KV."""
    from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
    m = O["memory_pool"]
    fig = plt.figure(figsize=(16, 9)); S.background(fig)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")
    W = S.INK; M = S.MUTED; V = S.OURS; V2 = S.OURS_2; HOT = "#FFB45C"; LORA = "#5BD1B5"; PRE = "#FF8FA3"
    WASTE = "#4A4475"; FLOOR = "#2E2758"; WT = "#3A3270"; GR = "#2A2350"
    ax.text(0.65, 8.2, "Reserve nothing, borrow everything", fontfamily=S.DISPLAY, fontsize=40, fontweight="bold", color=W)
    ax.text(0.65, 7.72, "KV gets every byte that isn't weights; vision and adapters borrow KV blocks only while they're needed",
            fontsize=14.5, color=M)
    x0, X1 = 3.1, 12.6; k = (X1 - x0) / m["total"]

    def seg(x, y, gib, fc, txt="", tc=S.ON_LIGHT, h=0.55, hatch=None, ec="none", ls="-", size=11, z=3):
        w = gib * k
        ax.add_patch(FancyBboxPatch((x, y), w - 0.03, h, boxstyle="round,pad=0,rounding_size=0.04", fc=fc, ec=ec, ls=ls,
                                    lw=1.4, hatch=hatch, zorder=z))
        if txt:
            ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=size, color=tc, zorder=z + 1)
        return x + w

    def label(y, a, b=None, size=19):
        ax.text(0.65, y + 0.36, a, fontfamily=S.DISPLAY, fontsize=size, fontweight="bold", color=W, va="center")
        if b: ax.text(0.65, y + 0.03, b, fontsize=11, color=M, va="center")

    def under(x, gib, y, txt, col, dy=0.22):
        ax.text(x + gib * k / 2, y - dy, txt, ha="center", fontsize=9.5, color=col, va="baseline" if dy < 0 else "baseline")

    floor = m["act_floor"] + 1.0                                             # activation floor + device floor
    kv_typ = m["kv"] + (m["lora_slots"] - m["lora_slot"])                   # 1 adapter loaded instead of 3: KV gets the rest
    kv_after = kv_typ + m["encoder"] + m["lora_slot"]   # KV grows by exactly the width of the vision + LoRA boxes it replaces
    margin = m["total"] - (m["weights"] + m["encoder"] + m["lora_slot"] + m["graphs"] + kv_typ + m["act_reserve"])

    # A. typical static layout (the runtime floor exists here too: part of the prefill reserve and the margin)
    yA = 6.62; label(yA, "Typical", "everything loaded at start")
    x = seg(x0, yA, m["weights"], WT, "model weights", tc=W)
    xg = x; x = seg(x, yA, m["graphs"] + floor, GR, "graphs + runtime", tc=M, size=9.5)
    xe = x; x = seg(x, yA, m["encoder"], HOT); ax.text(x - 0.02, yA + 0.68, "vision", ha="right", fontsize=9.5, color=HOT)
    xl = x
    x = seg(x, yA, m["lora_slot"], LORA)
    ax.text(xl + 0.05, yA + 0.68, "LoRA", ha="left", fontsize=9.5, color=LORA)
    xa0 = x; x = seg(x, yA, kv_typ, V, "KV cache"); xa1 = x
    peaks = m["total"] - (x - x0) / k
    pbuf = m["act_reserve"] - m["act_floor"]
    xp = x; x = seg(x, yA, pbuf, WASTE, "", hatch="///", ec=PRE)
    xm = x; x = seg(x, yA, peaks - pbuf, WASTE, "", hatch="///", ec="#7D77A8")
    assert abs((x - x0) / k - m["total"]) < 1e-6
    ax.text(X1 + 0.5, yA + 0.27, "vision + LoRA loaded at start,\nidle most of the time", fontsize=10.5, color=M, va="center", linespacing=1.15)

    # B. one pool: same weights and runtime; KV takes the rest, from both ends
    yB = 5.82; label(yB, "One pool", "KV takes the rest")
    x = seg(x0, yB, m["weights"], WT, "model weights", tc=W)
    xg2 = x; x = seg(x, yB, m["graphs"] + floor, GR, "graphs + runtime", tc=M, size=9.5)
    xk = x; x = seg(x, yB, kv_after, V, "KV cache"); xk1 = x
    x = seg(x, yB, pbuf, WASTE, "", hatch="///", ec=PRE)
    x = seg(x, yB, peaks - pbuf, WASTE, "", hatch="///", ec="#7D77A8")
    assert abs((x - x0) / k - m["total"]) < 1e-6
    ax.plot([xa0, xa0], [yB - 0.2, yB + 0.55], color=V2, lw=1.4, ls="--", zorder=6)   # where KV used to start
    lgain = (xa0 - xk) / k
    gain = kv_after - kv_typ
    ax.annotate("", (xk, yB - 0.16), (xa0, yB - 0.16), arrowprops=dict(arrowstyle="<->", color=V2, lw=1.2))
    ax.text(xk - 0.1, yB - 0.16, f"+{lgain:.1f} GiB: vision + LoRA", ha="right", va="center", fontsize=11, color=V2)
    ax.text(X1 + 0.5, yB + 0.27, f"+{lgain:.1f} GiB of KV: vision + LoRA\nborrow it only while they run", fontsize=10.5, color=V2, va="center", linespacing=1.15)
    # legend of the hatched boxes (the same two in both rows)
    for lx, lc, ltxt in ((8.6, PRE, "prefill reserve (fixed in both)"), (11.9, "#7D77A8", "safety margin")):
        ax.add_patch(Rectangle((lx, yB - 0.52), 0.34, 0.2, fc=WASTE, ec=lc, hatch="///", lw=1.2, zorder=3))
        ax.text(lx + 0.45, yB - 0.42, ltxt, fontsize=9.5, color=M, va="center")

    # C. borrowing: evict coldest KV, tenant in, KV loaded back
    ax.text(0.65, 5.08, "Borrowing", fontfamily=S.DISPLAY, fontsize=19, fontweight="bold", color=W)
    groups = [("An image description request comes in", HOT, [
                  ("image: coldest KV out", "out", m["encoder"]), ("vision in", "in", m["encoder"]),
                  ("idle: KV back from RAM", "back", m["encoder"])]),
              ("A request for a new behaviour (LoRA adapter)", LORA, [
                  ("adapter: coldest KV out", "out", m["lora_slot"]), ("adapter in", "in", m["lora_slot"]),
                  ("idle: KV back from RAM", "back", m["lora_slot"])])]
    h, dy, gh = 0.24, 0.31, 0.34
    ram_x, ram_w = 13.35, 1.95
    cur = 4.7; n = 0; rows_y = []
    for title, col, rows in groups:
        ax.text(0.65, cur, title, fontsize=12, fontweight="bold", color=col, va="center")
        cur -= gh
        for txt, kind, t in rows:
            n += 1
            y = cur - h / 2; yc = cur; rows_y.append(yc)
            ax.text(0.65, yc, f"{n}", ha="left", va="center", fontfamily=S.DISPLAY, fontsize=14, fontweight="bold", color=col)
            ax.text(0.95, yc, txt, ha="left", va="center", fontsize=10.5, color=W)
            x = seg(x0, y, m["weights"], WT, h=h)
            x = seg(x, y, m["graphs"] + floor, GR, h=h)
            x = seg(x, y, kv_after - t, V, h=h)
            xt = x
            if kind == "out":
                x = seg(x, y, t, "none", h=h, ec=V2, ls="--", z=4)
                ax.add_patch(FancyArrowPatch((xt + t * k, yc), (ram_x - 0.03, yc), arrowstyle="-|>", mutation_scale=12, lw=1.6, color=V2, zorder=6))
            elif kind == "in":
                x = seg(x, y, t, col, h=h, z=4)
                ax.add_patch(FancyArrowPatch((ram_x - 0.03, yc), (xt + t * k + 0.03, yc), arrowstyle="-|>", mutation_scale=12, lw=1.6, color=col, zorder=6))
            else:
                x = seg(x, y, t, V, h=h, z=4, ec=V2)
                ax.add_patch(FancyArrowPatch((ram_x - 0.03, yc), (xt + t * k + 0.03, yc), arrowstyle="-|>", mutation_scale=12, lw=1.6, color=V2, ls="--", zorder=6))
            x = seg(x, y, pbuf, WASTE, "", h=h, hatch="///", ec=PRE)                 # the same reserves as rows A and B
            x = seg(x, y, peaks - pbuf, WASTE, "", h=h, hatch="///", ec="#7D77A8")
            assert abs((x - x0) / k - m["total"]) < 1e-6, (n, (x - x0) / k)         # every row is the same card: one total
            cur -= dy
    ram_y = rows_y[-1] - h / 2 - 0.05; ram_h = rows_y[0] + h / 2 + 0.05 - ram_y
    # RAM stick (DIMM, standing on its edge): green PCB, gold edge fingers on the right with a key notch, DRAM chips,
    # a label sticker with the tier name
    ax.add_patch(FancyBboxPatch((ram_x, ram_y), ram_w, ram_h, boxstyle="round,pad=0,rounding_size=0.06", fc="#1F3A2E",
                                ec="#3E6B55", lw=1.6, zorder=3))
    npins = 30
    for i in range(npins):
        if i == 11: continue                                                  # key notch
        py = ram_y + 0.08 + i * (ram_h - 0.16) / npins
        ax.add_patch(Rectangle((ram_x + ram_w, py), 0.13, (ram_h - 0.16) / npins * 0.6, fc="#C9A94A", ec="none", zorder=3))
    ax.add_patch(Rectangle((ram_x + ram_w - 0.01, ram_y + 0.08 + 11 * (ram_h - 0.16) / npins), 0.15,
                           (ram_h - 0.16) / npins, fc=S.BG_BOTTOM, ec="none", zorder=4))
    from matplotlib.patches import Wedge
    for yy, th in ((ram_y + ram_h, (180, 360)), (ram_y, (0, 180))):           # retention notches on both short ends
        ax.add_patch(Wedge((ram_x + ram_w - 0.42, yy), 0.13, th[0], th[1], fc=S.BG_BOTTOM, ec="#3E6B55", lw=1.0, zorder=4))
    for r in range(4):                                                        # DRAM packages, two columns
        for c in range(2):
            ax.add_patch(Rectangle((ram_x + 0.18 + c * 0.82, ram_y + 0.14 + r * 0.42), 0.62, 0.32, fc="#121212",
                                   ec="#3E6B55", lw=0.8, zorder=4))
    sy = ram_y + ram_h - 1.1                                                  # label sticker
    ax.add_patch(FancyBboxPatch((ram_x + 0.12, sy), ram_w - 0.24, 0.98, boxstyle="round,pad=0,rounding_size=0.05",
                                fc="#2A1F5E", ec=V, lw=1.2, zorder=5))
    ax.text(ram_x + ram_w / 2, sy + 0.72, "RAM tier", ha="center", va="center", fontsize=13, color=W, zorder=6)
    ax.text(ram_x + ram_w / 2, sy + 0.44, "evicted KV copies", ha="center", va="center", fontsize=9.5, color=V2, zorder=6)
    ax.text(ram_x + ram_w / 2, sy + 0.18, "vision, adapters", ha="center", va="center", fontsize=9.5, color=HOT, zorder=6)

    # payoff
    bx, by, bw, bh = 0.65, 0.3, 14.7, 0.8
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0,rounding_size=0.2", fc="#2A1F5E", ec=V, lw=2, zorder=3))
    cells = [("+46k", "KV tokens on an RTX 5090\n(436,585 to 482,846, +10.6 %)"),
             ("+49k", "KV tokens on an Arc Pro B70\n(497,302 to 546,454 in production)"),
             ("0", "GiB held back for vision + LoRA\nwhile they are idle")]
    for i, (big, small) in enumerate(cells):
        cx = bx + 0.35 + i * 4.95
        ax.text(cx, by + bh / 2, big, fontfamily=S.DISPLAY, fontsize=30, fontweight="bold", color=W, va="center", zorder=4)
        ax.text(cx + (1.6 if i < 2 else 0.6), by + bh / 2, small, fontsize=11, color=M, va="center", zorder=4)
    ax.text(0.65, 0.1, "RTX 5090 budget to scale (31.4 GiB), vision + 1 LoRA adapter kept loaded vs borrowed. "
            "The +46k is measured on the RTX 5090 (same config otherwise). In production on both GPUs (vision + LoRA on the Arc, vision on the RTX 5090).",
            fontsize=9, color=M)
    return S.save(fig, out, "on-demand")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(HERE, "out")); ap.add_argument("--only", default="")
    a = ap.parse_args(); P, O = load()
    only = set(a.only.split(",")) - {""}
    for name, fn in sorted(globals().items()):
        if name.startswith("chart_") and (not only or name[6:] in only):
            print("wrote", fn(P, O, a.out))
