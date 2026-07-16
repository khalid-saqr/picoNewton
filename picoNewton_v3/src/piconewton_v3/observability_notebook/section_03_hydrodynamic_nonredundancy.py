"""Notebook section generated from the validated observability workflow.

Executed by ``picoNewton_v3_mechanosensory_observability.ipynb`` in the
notebook shared namespace. Keep scientific thresholds in configuration files.
"""

def centered_rms_normalize(signal):
    x = np.asarray(signal, dtype=float)
    centered = x - np.mean(x)
    return centered / max(rms(centered), 1e-30)


def periodic_lowpass(signal, relaxation_s, frequency_hz):
    x = np.asarray(signal, dtype=float)
    coefficients = np.fft.rfft(x)
    harmonic = np.arange(len(coefficients), dtype=float)
    transfer = 1.0 / (1.0 + 1j * 2.0 * np.pi * harmonic * frequency_hz * relaxation_s)
    return np.fft.irfft(coefficients * transfer, n=len(x))


def affine_match(predictor, target):
    z = np.asarray(predictor, dtype=float)
    y = np.asarray(target, dtype=float)
    zc = z - z.mean()
    yc = y - y.mean()
    gain = float(np.dot(zc, yc) / max(np.dot(zc, zc), 1e-30))
    offset = float(y.mean() - gain * z.mean())
    prediction = gain * z + offset
    return prediction, gain, offset


def fit_single_artery_surrogate(force, shear, model, frequency_hz, profile):
    n = len(force)
    lag_points = 33 if profile == "quick" else 65
    lag_grid = np.unique(np.round(np.linspace(-0.25, 0.25, lag_points) * n).astype(int))
    tau_grid = np.array([0.0]) if model != "lagged_lowpass" else np.logspace(-3, 1, 17 if profile == "quick" else 33)
    if model == "instantaneous":
        lag_grid = np.array([0])
    best = None
    for tau_filter in tau_grid:
        filtered = periodic_lowpass(shear, tau_filter, frequency_hz) if tau_filter > 0 else np.asarray(shear)
        for lag in lag_grid:
            shifted = np.roll(filtered, int(lag))
            prediction, gain, offset = affine_match(shifted, force)
            residual = rms(prediction - force) / max(rms(force - np.mean(force)), 1e-30)
            if best is None or residual < best["residual_fraction"]:
                best = {"prediction": prediction, "lag_samples": int(lag), "lag_cycle": float(lag / n), "filter_tau_s": float(tau_filter), "gain_n_per_pa": gain, "offset_n": offset, "residual_fraction": float(residual)}
    return best


NONREDUNDANCY_ROWS = []
SURROGATE_WAVEFORMS = {}
PHASE_ROWS = []
for case in V2_ARTERY_CASES:
    full = HYDRO_FULL[case.artery_id]
    force = np.asarray(full["force_signed_n"])
    shear = np.asarray(full["wall_shear_pa"])
    for model in ("instantaneous", "lagged", "lagged_lowpass"):
        result = fit_single_artery_surrogate(force, shear, model, FLUID.fundamental_frequency_hz, PROFILE)
        SURROGATE_WAVEFORMS[(case.artery_id, model)] = result["prediction"]
        NONREDUNDANCY_ROWS.append({"artery_id": case.artery_id, "artery_name": case.name, "alpha": full["alpha"], "model": model, **{k: v for k, v in result.items() if k != "prediction"}})

    f_coeff = np.fft.rfft(force - force.mean())
    w_coeff = np.fft.rfft(shear - shear.mean())
    maximum = min(6, len(f_coeff) - 1, len(w_coeff) - 1)
    for h in range(1, maximum + 1):
        phase = np.angle(f_coeff[h]) - np.angle(w_coeff[h])
        PHASE_ROWS.append({"artery_id": case.artery_id, "artery_name": case.name, "harmonic": h, "phase_difference_rad": float(np.angle(np.exp(1j * phase)))})

NONREDUNDANCY_TABLE = pd.DataFrame(NONREDUNDANCY_ROWS)
PHASE_TABLE = pd.DataFrame(PHASE_ROWS)
display(NONREDUNDANCY_TABLE)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, artery_id, label in [(axes[0, 0], "aortic_root", "(a) High-α example: Aortic Root"), (axes[0, 1], "brachial", "(b) Low-α example: Brachial")]:
    h = HYDRO_FULL[artery_id]
    target = centered_rms_normalize(h["force_signed_n"])
    surrogate = centered_rms_normalize(SURROGATE_WAVEFORMS[(artery_id, "lagged_lowpass")])
    wss = centered_rms_normalize(h["wall_shear_pa"])
    ax.plot(h["time_cycle"], target, label="Lamb force")
    ax.plot(h["time_cycle"], surrogate, label="Best WSS surrogate")
    ax.plot(h["time_cycle"], wss, linestyle="--", alpha=0.7, label="Raw WSS")
    ax.set_xlabel("Cardiac cycle, t/T")
    ax.set_ylabel("Centered RMS-normalized signal")
    ax.set_title(label)
    ax.legend(frameon=False, fontsize=8)

ax = axes[1, 0]
phase_matrix = PHASE_TABLE.pivot(index="artery_name", columns="harmonic", values="phase_difference_rad").reindex([c.name for c in V2_ARTERY_CASES])
im = ax.imshow(phase_matrix.to_numpy(), aspect="auto", vmin=-np.pi, vmax=np.pi, cmap="twilight")
ax.set_yticks(np.arange(len(phase_matrix.index)), phase_matrix.index)
ax.set_xticks(np.arange(len(phase_matrix.columns)), phase_matrix.columns)
ax.set_xlabel("Harmonic")
ax.set_title("(c) Lamb–WSS harmonic phase difference")
fig.colorbar(im, ax=ax, label="Phase difference, rad")

ax = axes[1, 1]
models = ["instantaneous", "lagged", "lagged_lowpass"]
positions = np.arange(len(V2_ARTERY_CASES))
width = 0.24
for j, model in enumerate(models):
    subset = NONREDUNDANCY_TABLE[NONREDUNDANCY_TABLE.model == model].set_index("artery_id").reindex([c.artery_id for c in V2_ARTERY_CASES])
    ax.bar(positions + (j - 1) * width, subset["residual_fraction"], width, label=model.replace("_", " "))
ax.set_xticks(positions, [c.name for c in V2_ARTERY_CASES], rotation=30, ha="right")
ax.set_ylabel(r"Residual fraction, $R_\perp$")
ax.set_title("(d) WSS shape-surrogate challenge")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
FIG2 = save_figure(fig, "Figure2_hydrodynamic_nonredundancy")[0]
plt.show()
