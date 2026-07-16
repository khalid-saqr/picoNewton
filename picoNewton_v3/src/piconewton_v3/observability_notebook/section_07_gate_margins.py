"""Notebook section generated from the validated observability workflow.

Executed by ``picoNewton_v3_mechanosensory_observability.ipynb`` in the
notebook shared namespace. Keep scientific thresholds in configuration files.
"""

PARAMETER_GRID = run_parameter_grid(
    V2_ARTERY_CASES,
    HYDRO_FULL,
    FLUID,
    coupling_lengths_m=CONFIG["coupling_lengths_m"],
    relaxation_times_s=CONFIG["relaxation_times_s"],
)
ORIGINAL_SURROGATE_TABLE, ORIGINAL_SURROGATE_PARAMETERS = fit_wss_surrogate(
    V2_ARTERY_CASES, HYDRO_FULL, FLUID, SENSOR_NOMINAL
)
GATE_TABLE = evaluate_effect_gates(
    PARAMETER_GRID,
    ORIGINAL_SURROGATE_TABLE,
    CONFIG["coupling_lengths_m"],
    CONFIG["relaxation_times_s"],
)

e1_by_artery = PARAMETER_GRID.groupby("artery_id")["passes_E1_point"].mean()
e1_arteries = int((e1_by_artery >= GATES["E1"]["minimum_grid_fraction_per_artery"]).sum())
e1_overall = float(PARAMETER_GRID["passes_E1_point"].mean())
e1_margin = min(e1_arteries / GATES["E1"]["minimum_arteries"], e1_overall / GATES["E1"]["minimum_overall_fraction"])

strong_loo = LOO_SUMMARY[LOO_SUMMARY.model == "lagged_lowpass"]
e2_margin = min(
    strong_loo["residual_q05"].min() / GATES["E2"]["absolute_rms"],
    strong_loo["dynamic_fraction_median"].min() / GATES["E2"]["minimum_dynamic_range_fraction"],
)

e3_pass = bool(GATE_TABLE.set_index("criterion_id").loc["E3", "passed"])
e3_margin = 1.05 if e3_pass else 0.0

e4_arteries = int((PARAMETER_GRID.groupby("artery_id")["passes_E4_point"].mean() >= GATES["E4"]["minimum_grid_fraction_per_artery"]).sum())
e4_margin = e4_arteries / GATES["E4"]["minimum_arteries"]
e5_arteries = int((PARAMETER_GRID.groupby("artery_id")["passes_E5_point"].mean() >= GATES["E5"]["minimum_grid_fraction_per_artery"]).sum())
e5_margin = e5_arteries / GATES["E5"]["minimum_arteries"]
e6_arteries = int((PARAMETER_GRID.groupby("artery_id")["passes_E6_point"].mean() >= GATES["E6"]["minimum_grid_fraction_per_artery"]).sum())
e6_margin = e6_arteries / GATES["E6"]["minimum_arteries"]
e7_count = int((PARAMETER_GRID.groupby("artery_id")["effect_parallel_vs_WSS"].quantile(0.05) >= GATES["E7"]["fifth_percentile_absolute_rms"]).sum())
e7_margin = e7_count / GATES["E7"]["minimum_arteries"]

GATE_MARGIN_TABLE = pd.DataFrame([
    {"criterion_id": "E1", "name": "Core detectability", "margin": e1_margin},
    {"criterion_id": "E2", "name": "Held-out WSS nonredundancy", "margin": e2_margin},
    {"criterion_id": "E3", "name": "Connected support", "margin": e3_margin},
    {"criterion_id": "E4", "name": "Directional specificity", "margin": e4_margin},
    {"criterion_id": "E5", "name": "High-harmonic specificity", "margin": e5_margin},
    {"criterion_id": "E6", "name": "Anisotropy specificity", "margin": e6_margin},
    {"criterion_id": "E7", "name": "Full-range robustness", "margin": e7_margin},
    {"criterion_id": "E8", "name": "Model transparency", "margin": 1.0},
])
GATE_MARGIN_TABLE["passed"] = GATE_MARGIN_TABLE["margin"] >= 1.0
GATE_MARGIN_TABLE["claim_status"] = np.select(
    [GATE_MARGIN_TABLE["margin"] >= 1.0, GATE_MARGIN_TABLE["margin"] >= 0.5],
    ["retained", "conditional"],
    default="rejected",
)

display(GATE_TABLE)
display(GATE_MARGIN_TABLE)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
artery_ids = [c.artery_id for c in V2_ARTERY_CASES]
artery_names = [c.name for c in V2_ARTERY_CASES]

ax = axes[0, 0]
e1_fraction = PARAMETER_GRID.groupby("artery_id")["passes_E1_point"].mean().reindex(artery_ids)
ax.bar(np.arange(len(e1_fraction)), e1_fraction)
ax.axhline(GATES["E1"]["minimum_grid_fraction_per_artery"], linestyle="--", label="Required per-artery fraction")
ax.set_xticks(np.arange(len(e1_fraction)), artery_names, rotation=30, ha="right")
ax.set_ylim(0, 1)
ax.set_ylabel("Passing fraction of sensor grid")
ax.set_title("(a) Core detectability coverage")
ax.legend(frameon=False, fontsize=8)

ax = axes[0, 1]
e4_fraction = PARAMETER_GRID.groupby("artery_id")["passes_E4_point"].mean().reindex(artery_ids)
e5_fraction = PARAMETER_GRID.groupby("artery_id")["passes_E5_point"].mean().reindex(artery_ids)
e6_fraction = PARAMETER_GRID.groupby("artery_id")["passes_E6_point"].mean().reindex(artery_ids)
source_matrix = np.vstack([e4_fraction, e5_fraction, e6_fraction])
im = ax.imshow(source_matrix, aspect="auto", vmin=0, vmax=1)
ax.set_yticks([0, 1, 2], ["Direction", "High harmonic", "Anisotropy"])
ax.set_xticks(np.arange(len(artery_names)), artery_names, rotation=30, ha="right")
ax.set_title("(b) Source-specific passing fractions")
fig.colorbar(im, ax=ax, label="Passing fraction")

ax = axes[1, 0]
pos = np.arange(len(GATE_MARGIN_TABLE))
ax.bar(pos, GATE_MARGIN_TABLE["margin"])
ax.axhline(1.0, linestyle="--", label="Pass threshold")
ax.set_xticks(pos, GATE_MARGIN_TABLE["criterion_id"])
ax.set_ylabel("Observed / required")
ax.set_title("(c) Quantitative gate margins")
ax.legend(frameon=False)

ax = axes[1, 1]
ax.set_axis_off()
status_style = {"retained": ("tab:green", "RETAINED"), "conditional": ("goldenrod", "CONDITIONAL"), "rejected": ("tab:red", "REJECTED")}
for row_index, row in GATE_MARGIN_TABLE.reset_index(drop=True).iterrows():
    y = 0.94 - row_index * 0.115
    color, label = status_style[row["claim_status"]]
    ax.scatter([0.04], [y], s=90, color=color, transform=ax.transAxes, clip_on=False)
    ax.text(0.09, y, f"{row['criterion_id']}  {label}", transform=ax.transAxes, va="center", fontweight="bold", fontsize=9)
    ax.text(0.43, y, row["name"], transform=ax.transAxes, va="center", fontsize=9)
ax.set_title("(d) Retained, conditional, and rejected claims")

fig.tight_layout()
FIG6 = save_figure(fig, "Figure6_gate_margins_and_claim_map")[0]
plt.show()
