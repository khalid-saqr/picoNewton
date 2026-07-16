"""Notebook section generated from the validated observability workflow.

Executed by ``picoNewton_v3_mechanosensory_observability.ipynb`` in the
notebook shared namespace. Keep scientific thresholds in configuration files.
"""

def save_figure(fig, stem):
    png = FIGURE_DIR / f"{stem}.png"
    pdf = FIGURE_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    STORE.register_file(RUN_ID, f"figures/{png.name}", "output")
    STORE.register_file(RUN_ID, f"figures/{pdf.name}", "output")
    return png, pdf


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ax = axes[0, 0]
x = HYDRO_SUMMARY["reproduction_exposure_rms_pN"]
y = HYDRO_SUMMARY["verified_exposure_rms_pN"]
ax.scatter(x, y, s=55)
for _, row in HYDRO_SUMMARY.iterrows():
    ax.annotate(row["artery_name"], (row["reproduction_exposure_rms_pN"], row["verified_exposure_rms_pN"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
positive = np.concatenate([x[x > 0], y[y > 0]])
if len(positive):
    low, high = positive.min() * 0.8, positive.max() * 1.25
    ax.plot([low, high], [low, high], linestyle="--", linewidth=1)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xscale("log")
    ax.set_yscale("log")
ax.set_xlabel("Reproduction-layout exposure RMS, pN")
ax.set_ylabel("Verified exposure RMS, pN")
ax.set_title("(a) Exposure-force continuity audit")

ax = axes[0, 1]
for case in V2_ARTERY_CASES:
    h = HYDRO_FULL[case.artery_id]
    ax.plot(h["time_cycle"], np.asarray(h["force_signed_n"]) * 1e12, label=case.name)
ax.axhline(0.0, linewidth=0.8)
ax.set_xlabel("Cardiac cycle, t/T")
ax.set_ylabel("Signed Lamb force, pN")
ax.set_title("(b) Verified signed force waveforms")
ax.legend(frameon=False, fontsize=8, ncol=2)

ax = axes[1, 0]
pos = np.arange(len(HYDRO_SUMMARY))
width = 0.25
ax.bar(pos - width, HYDRO_SUMMARY["signed_rms_pN"], width, label="Total signed")
ax.bar(pos, HYDRO_SUMMARY["isotropic_rms_pN"], width, label="Isotropic baseline")
ax.bar(pos + width, HYDRO_SUMMARY["anisotropy_excess_rms_pN"], width, label="Anisotropy excess")
ax.set_xticks(pos, HYDRO_SUMMARY["artery_name"], rotation=30, ha="right")
ax.set_ylabel("RMS force, pN")
ax.set_yscale("log")
ax.set_title("(c) Force-source decomposition")
ax.legend(frameon=False, fontsize=8)

ax = axes[1, 1]
spec = HYDRO_SPECTRA[(HYDRO_SPECTRA.signal == "anisotropy_excess") & (HYDRO_SPECTRA.harmonic.between(1, 12))]
matrix = spec.pivot(index="artery_name", columns="harmonic", values="power").reindex([c.name for c in V2_ARTERY_CASES])
matrix = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
im = ax.imshow(matrix.to_numpy(), aspect="auto", origin="upper")
ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns)
ax.set_xlabel("Output harmonic")
ax.set_title("(d) Normalized anisotropy-excess power")
fig.colorbar(im, ax=ax, label="Power fraction")
fig.tight_layout()
FIG1 = save_figure(fig, "Figure1_signal_continuity_and_decomposition")[0]
plt.show()
