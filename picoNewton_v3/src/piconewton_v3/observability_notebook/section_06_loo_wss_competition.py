"""Notebook section generated from the validated observability workflow.

Executed by ``picoNewton_v3_mechanosensory_observability.ipynb`` in the
notebook shared namespace. Keep scientific thresholds in configuration files.
"""

def fit_global_shape_model(training_ids, model):
    n = CONFIG["time_points"]
    lag_points = 25 if PROFILE == "quick" else 49
    lag_grid = np.unique(np.round(np.linspace(-0.25, 0.25, lag_points) * n).astype(int))
    tau_grid = np.array([0.0]) if model != "lagged_lowpass" else np.logspace(-3, 1, 13 if PROFILE == "quick" else 25)
    if model == "instantaneous":
        lag_grid = np.array([0])
    targets = {cid: centered_rms_normalize(HYDRO_FULL[cid]["force_signed_n"]) for cid in training_ids}
    predictors = {cid: centered_rms_normalize(HYDRO_FULL[cid]["wall_shear_pa"]) for cid in training_ids}
    best = None
    for tau_filter in tau_grid:
        filtered = {cid: periodic_lowpass(signal, tau_filter, FLUID.fundamental_frequency_hz) if tau_filter > 0 else signal for cid, signal in predictors.items()}
        for lag in lag_grid:
            z = np.concatenate([np.roll(filtered[cid], int(lag)) for cid in sorted(training_ids)])
            y = np.concatenate([targets[cid] for cid in sorted(training_ids)])
            prediction, gain, offset = affine_match(z, y)
            mse = float(np.mean((prediction - y) ** 2))
            if best is None or mse < best["training_mse"]:
                best = {"model": model, "lag_samples": int(lag), "lag_cycle": float(lag / n), "filter_tau_s": float(tau_filter), "gain": float(gain), "offset": float(offset), "training_mse": mse}
    return best


def apply_heldout_shape_model(held_out_id, parameters):
    shear = centered_rms_normalize(HYDRO_FULL[held_out_id]["wall_shear_pa"])
    if parameters["filter_tau_s"] > 0:
        shear = periodic_lowpass(shear, parameters["filter_tau_s"], FLUID.fundamental_frequency_hz)
    normalized_prediction = parameters["gain"] * np.roll(shear, parameters["lag_samples"]) + parameters["offset"]
    force = np.asarray(HYDRO_FULL[held_out_id]["force_signed_n"])
    centered_force = force - force.mean()
    normalized_prediction = centered_rms_normalize(normalized_prediction)
    return force.mean() + normalized_prediction * rms(centered_force)


LOO_PARAMETER_ROWS = []
LOO_NOMINAL_ROWS = []
LOO_MODELS = ["instantaneous", "lagged", "lagged_lowpass"]
for held_out_case in V2_ARTERY_CASES:
    held_out_id = held_out_case.artery_id
    training_ids = {c.artery_id for c in V2_ARTERY_CASES if c.artery_id != held_out_id}
    target_force = np.asarray(HYDRO_FULL[held_out_id]["force_signed_n"])
    shear = np.asarray(HYDRO_FULL[held_out_id]["wall_shear_pa"])
    psi_w = wss_work(shear, 1e-22, SENSOR_NOMINAL.temperature_k)
    for model in LOO_MODELS:
        parameters = fit_global_shape_model(training_ids, model)
        predicted_force = apply_heldout_shape_model(held_out_id, parameters)
        for coupling_length in CONFIG["coupling_lengths_m"]:
            for relaxation_time in CONFIG["relaxation_times_s"]:
                sensor = SensorConfig(basal_probability=SENSOR_NOMINAL.basal_probability, relaxation_time_s=float(relaxation_time), transition_fraction=SENSOR_NOMINAL.transition_fraction, temperature_k=SENSOR_NOMINAL.temperature_k)
                target_work = psi_w + lamb_work(target_force, coupling_length, sensor.temperature_k)
                prediction_work = psi_w + lamb_work(predicted_force, coupling_length, sensor.temperature_k)
                target_p, _ = periodic_sensor_solution(target_work, FLUID.fundamental_frequency_hz, sensor)
                prediction_p, _ = periodic_sensor_solution(prediction_work, FLUID.fundamental_frequency_hz, sensor)
                residual = rms_difference(target_p, prediction_p)
                target_range = max(np.ptp(target_p), 1e-12)
                LOO_PARAMETER_ROWS.append({
                    "held_out_artery_id": held_out_id,
                    "held_out_artery_name": held_out_case.name,
                    "model": model,
                    "coupling_length_m": coupling_length,
                    "relaxation_time_s": relaxation_time,
                    "rms_residual": residual,
                    "fraction_target_dynamic_range": residual / target_range,
                    "passes_E2_point": bool(residual >= GATES["E2"]["absolute_rms"] and residual / target_range >= GATES["E2"]["minimum_dynamic_range_fraction"] and residual >= GATES["E2"]["minimum_error_multiple"] * GATES["numerical_sensor_uncertainty"]),
                    **parameters,
                })
        nominal_d = 1e-9
        nominal_tau = 0.1
        sensor = SensorConfig(relaxation_time_s=nominal_tau)
        target_work = psi_w + lamb_work(target_force, nominal_d, sensor.temperature_k)
        prediction_work = psi_w + lamb_work(predicted_force, nominal_d, sensor.temperature_k)
        target_p, _ = periodic_sensor_solution(target_work, FLUID.fundamental_frequency_hz, sensor)
        prediction_p, _ = periodic_sensor_solution(prediction_work, FLUID.fundamental_frequency_hz, sensor)
        for i, cycle in enumerate(HYDRO_FULL[held_out_id]["time_cycle"]):
            LOO_NOMINAL_ROWS.append({"held_out_artery_id": held_out_id, "held_out_artery_name": held_out_case.name, "model": model, "time_cycle": cycle, "target_probability": target_p[i], "predicted_probability": prediction_p[i]})

LOO_PARAMETER_TABLE = pd.DataFrame(LOO_PARAMETER_ROWS)
LOO_NOMINAL_TABLE = pd.DataFrame(LOO_NOMINAL_ROWS)
LOO_SUMMARY = LOO_PARAMETER_TABLE.groupby(["held_out_artery_id", "held_out_artery_name", "model"]).agg(
    residual_q05=("rms_residual", lambda x: x.quantile(0.05)),
    residual_median=("rms_residual", "median"),
    residual_q95=("rms_residual", lambda x: x.quantile(0.95)),
    dynamic_fraction_median=("fraction_target_dynamic_range", "median"),
    pass_fraction=("passes_E2_point", "mean"),
).reset_index()
display(LOO_SUMMARY)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ax = axes[0, 0]
example = LOO_NOMINAL_TABLE[(LOO_NOMINAL_TABLE.held_out_artery_id == "aortic_root") & (LOO_NOMINAL_TABLE.model == "lagged_lowpass")]
ax.plot(example["time_cycle"], example["target_probability"], label="Parallel target")
ax.plot(example["time_cycle"], example["predicted_probability"], label="Held-out WSS surrogate")
ax.set_xlabel("Cardiac cycle, t/T")
ax.set_ylabel("Active-state probability")
ax.set_title("(a) Held-out Aortic Root, nominal sensor")
ax.legend(frameon=False)

ax = axes[0, 1]
strong = LOO_SUMMARY[LOO_SUMMARY.model == "lagged_lowpass"].set_index("held_out_artery_id").reindex([c.artery_id for c in V2_ARTERY_CASES])
pos = np.arange(len(strong))
y = np.maximum(strong["residual_median"].to_numpy(), 1e-6)
q05 = np.maximum(strong["residual_q05"].to_numpy(), 1e-6)
q95 = np.maximum(strong["residual_q95"].to_numpy(), y)
yerr = np.vstack([y - q05, q95 - y])
ax.errorbar(pos, y, yerr=yerr, fmt="o", capsize=4)
ax.axhline(GATES["E2"]["absolute_rms"], linestyle="--", label="E2 threshold")
ax.set_yscale("log")
ax.set_xticks(pos, strong["held_out_artery_name"], rotation=30, ha="right")
ax.set_ylabel("Held-out sensor RMS residual")
ax.set_title("(b) Strongest WSS challenge across sensor grid")
ax.legend(frameon=False, fontsize=8)

ax = axes[1, 0]
for j, model in enumerate(LOO_MODELS):
    subset = LOO_SUMMARY[LOO_SUMMARY.model == model].set_index("held_out_artery_id").reindex([c.artery_id for c in V2_ARTERY_CASES])
    ax.bar(pos + (j - 1) * 0.24, subset["dynamic_fraction_median"], 0.24, label=model.replace("_", " "))
ax.axhline(GATES["E2"]["minimum_dynamic_range_fraction"], linestyle="--")
ax.set_xticks(pos, [c.name for c in V2_ARTERY_CASES], rotation=30, ha="right")
ax.set_ylabel("Residual / target dynamic range")
ax.set_title("(c) Dynamic-range-normalized residual")
ax.legend(frameon=False, fontsize=8)

ax = axes[1, 1]
for j, model in enumerate(LOO_MODELS):
    subset = LOO_SUMMARY[LOO_SUMMARY.model == model].set_index("held_out_artery_id").reindex([c.artery_id for c in V2_ARTERY_CASES])
    ax.bar(pos + (j - 1) * 0.24, subset["pass_fraction"], 0.24, label=model.replace("_", " "))
ax.set_xticks(pos, [c.name for c in V2_ARTERY_CASES], rotation=30, ha="right")
ax.set_ylim(0, 1)
ax.set_ylabel("Fraction of sensor grid passing E2")
ax.set_title("(d) Robustness of nonredundancy")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
FIG5 = save_figure(fig, "Figure5_leave_one_artery_out_WSS_competition")[0]
plt.show()
