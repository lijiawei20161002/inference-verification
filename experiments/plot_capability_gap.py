"""Dose-response figure: cheap-proxy detection AUC vs. how far the served model is
downgraded from the claimed Qwen2.5-7B.

Answers Roy's "how different do the models have to be?" with a curve instead of two
extreme points. All AUCs are computed live from the rich per-token features in
difr_data/feats_*.pt (the same 0.5B+1.5B panel detector as detect.py), so the figure
tracks the actual experiment data rather than transcribed numbers.

Left  : panel AUC vs served-model size (log x). The dose-response / capability gap.
Right : panel AUC vs verified tokens, one curve per served tier.
"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from detect import panel_auc, cv_auc, C05, C15, SERVED, P_CLAIMED, KS

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "figures",
                   "fig_difr_capability_gap.png")

# colour per tier (colour-blind-safe), ordered small -> large served model
COLOR = {"sub_0.5b": "#1f77b4", "sub_1.5b": "#2ca02c",
         "sub_3b": "#ff7f0e", "quant_4bit": "#7f7f7f"}
TIERS = ["sub_0.5b", "sub_1.5b", "sub_3b", "quant_4bit"]

# cross-family control (Qwen proxies vs served SmolLM2-360M) as an extra reference point
SMOL_PARAMS = 0.362e9
def smol_auc(k):
    return cv_auc([C05["honest"], C15["honest"]], [C05["smol360"], C15["smol360"]], k)

print("computing panel AUCs from feats data ...")
auc128 = {t: panel_auc(t, 128) for t in TIERS}
auc32  = {t: panel_auc(t, 32)  for t in TIERS}
smol128, smol32 = smol_auc(128), smol_auc(32)
for t in TIERS:
    print(f"  {SERVED[t]['label']:16s} ratio {SERVED[t]['ratio']:5.1f}x  "
          f"AUC@32={auc32[t]:.3f}  AUC@128={auc128[t]:.3f}")
print(f"  SmolLM2-360B(x-fam) ratio {P_CLAIMED/SMOL_PARAMS:5.1f}x  "
      f"AUC@32={smol32:.3f}  AUC@128={smol128:.3f}")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

# ---- Left: dose-response, AUC vs capability gap (× smaller than claimed 7B) -------
# x = the "dose" (attacker's compute saving); bigger gap should be easier to detect.
# same-family substitution tiers (0.5B/1.5B/3B) as a connected curve
sub = ["sub_0.5b", "sub_1.5b", "sub_3b"]
xs = [SERVED[t]["ratio"] for t in sub]
for k, auc, style in [(128, auc128, dict(marker="o", ls="-", lw=2.5)),
                      (32,  auc32,  dict(marker="s", ls="--", lw=1.8, alpha=0.7))]:
    axL.plot(xs, [auc[t] for t in sub], color="#d62728",
             label=f"same-family substitution (@{k} tok)", **style)
# quant (7B nf4, ratio 1x) and cross-family (SmolLM2-360M) as standalone reference markers
axL.scatter([SERVED["quant_4bit"]["ratio"]], [auc128["quant_4bit"]],
            marker="X", s=140, color="#7f7f7f", zorder=5,
            label="4-bit quant of 7B (subtle, @128)")
axL.scatter([P_CLAIMED / SMOL_PARAMS], [smol128], marker="D", s=95, color="#9467bd", zorder=5,
            label="cross-family → SmolLM2-360M (@128)")
axL.axhline(0.5, color="gray", ls=":", alpha=0.7)
axL.text(1.05, 0.508, "chance", fontsize=8, color="gray")
# label each same-family / quant point with its served model
for t in sub + ["quant_4bit"]:
    dy = 10 if t != "sub_3b" else -16
    axL.annotate(SERVED[t]["label"].replace("Qwen2.5-", ""),
                 (SERVED[t]["ratio"], auc128[t]),
                 textcoords="offset points", xytext=(0, dy), fontsize=8.5,
                 ha="center", color="#555")
axL.set_xscale("log")
axL.set_xticks([1, 2, 5, 10, 20])
axL.set_xticklabels(["1×\n(nf4)", "2×", "5×", "10×", "20×"])
axL.set_ylim(0.42, 1.02)
axL.set_xlabel("capability gap  =  claimed-7B ÷ served params   (→ bigger downgrade / attacker savings)")
axL.set_ylabel("detection AUC (0.5B+1.5B panel, 5-fold CV)")
axL.set_title("Dose-response: detectability tracks the capability gap")
axL.grid(alpha=0.3)
axL.legend(fontsize=8.5, loc="center left")

# ---- Right: AUC vs verified tokens, one curve per tier ---------------------------
for t in TIERS:
    axR.plot(KS, [panel_auc(t, k) for k in KS], "o-", lw=2,
             color=COLOR[t], label=SERVED[t]["label"])
axR.plot(KS, [smol_auc(k) for k in KS], "D--", lw=1.8, color="#9467bd",
         label="SmolLM2-360M (x-fam)")
axR.axhline(0.5, color="gray", ls=":", alpha=0.7)
axR.set_xscale("log", base=2)
axR.set_xticks(KS)
axR.set_xticklabels(KS)
axR.set_ylim(0.42, 1.02)
axR.set_xlabel("verified tokens")
axR.set_ylabel("detection AUC (0.5B+1.5B panel, 5-fold CV)")
axR.set_title("Convergence vs. tokens audited, per served tier")
axR.grid(alpha=0.3)
axR.legend(fontsize=8.5, loc="lower right")

fig.suptitle("Cheap-proxy inference verification: detection collapses as the served model "
             "approaches the claimed 7B", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(FIG, dpi=130)
print("saved", os.path.normpath(FIG))
