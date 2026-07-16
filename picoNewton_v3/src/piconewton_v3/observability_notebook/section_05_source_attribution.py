"""Notebook section generated from the validated observability workflow.

Executed by ``picoNewton_v3_mechanosensory_observability.ipynb`` in the
notebook shared namespace. Keep scientific thresholds in configuration files.
"""

def sensor_from_work(work, relaxation_time_s):
    sensor = SensorConfig(
        basal_probability=SENSOR_NOMINAL.basal_probability,
        relaxation_time_s=float(relaxation_time_s),
        transition_fraction=SENSOR_NOMINAL.transition_fraction,
        temperature_k=SENSOR_NOMINAL.temperature_k,
    )
    return periodic_sensor_solution(work, FLUID.fundamental_frequency_hz, sensor)[0]


ATTRIBUTION_ROWS = []
OBSERVABILITY_ROWS = []
for case in V2_ARTERY_CASES:
    full_force = np.asarray(HYDRO_FULL[case.artery_id]["force_signed_n"])
    iso_force = np.asarray(HYDRO_ISO[case.artery_id]["force_signed_n"])
    low_force = np.asarray(HYDRO_LOW[case.artery_id]["force_signed_n"])
    shear = np.asarray(HYDRO_FULL[case.artery_id]["wall_shear_pa"])

    for coupling_length in CONFIG["coupling_lengths_m"]:
        for relaxation_time in CONFIG["relaxation_times_s"]:
            psi_full = lamb_work(full_force, coupling_length, SENSOR_NOMINAL.temperature_k)
            psi_iso = lamb_work(iso_force, coupling_length, SENSOR_NOMINAL.temperature_k)
            psi_low = lamb_work(low_force, coupling_length, SENSOR_NOMINAL.temperature_k)
            psi_wss = wss_work(shear, 1e-22, SENSOR_NOMINAL.temperature_k)
            p_zero = sensor_from_work(np.zeros_like(psi_full), relaxation_time)
            p_full = sensor_from_work(psi_full, relaxation_time)
            p_iso = sensor_from_work(psi_iso, relaxation_time)
            p_low = sensor_from_work(psi_low, relaxation_time)
            p_wss = sensor_from_work(psi_wss, relaxation_time)
            p_parallel = sensor_from_work(psi_wss + psi_full, relaxation_time)
            p_reverse = sensor_from_work(-psi_full, relaxation_time)
            full_response = rms_difference(p_full, p_zero)
            ATTRIBUTION_ROWS.append({
                "artery_id": case.artery_id,
                "artery_name": case.name,
                "coupling_length_m": coupling_length,
                "relaxation_time_s": relaxation_time,
                "Lambda_RMS": rms(psi_full),
                "Omega": FLUID.angular_frequency_rad_s * relaxation_time,
                "parallel_vs_wss": rms_difference(p_parallel, p_wss),
                "full_response": full_response,
                "anisotropy_effect": rms_difference(p_full, p_iso),
                "high_harmonic_effect": rms_difference(p_full, p_low),
                "direction_effect": rms_difference(p_full, p_reverse),
                "anisotropy_fraction": rms_difference(p_full, p_iso) / max(full_response, 1e-12),
                "high_harmonic_fraction": rms_difference(p_full, p_low) / max(full_response, 1e-12),
            })

    nominal_d = 1e-9
    nominal_tau = 0.1
    omega = FLUID.angular_frequency_rad_s * nominal_tau
    psi_aniso = lamb_work(full_force - iso_force, nominal_d, SENSOR_NOMINAL.temperature_k)
    psi_total = lamb_work(full_force, nominal_d, SENSOR_NOMINAL.temperature_k)
    for signal_name, psi in {"total": psi_total, "anisotropy_excess": psi_aniso}.items():
        coeff = np.fft.rfft(psi) / len(psi)
        for harmonic in range(1, min(12, len(coeff) - 1) + 1):
            work_amplitude = 2.0 * abs(coeff[harmonic])
            observable = linear_response_amplitude(work_amplitude, harmonic, omega)
            OBSERVABILITY_ROWS.append({
                "artery_id": case.artery_id,
                "artery_name": case.name,
                "signal": signal_name,
                "harmonic": harmonic,
                "work_amplitude": work_amplitude,
                "sensor_weighted_amplitude": observable,
            })

ATTRIBUTION_TABLE = pd.DataFrame(ATTRIBUTION_ROWS)
OBSERVABILITY_SPECTRUM = pd.DataFrame(OBSERVABILITY_ROWS)
display(ATTRIBUTION_TABLE.groupby("artery_name")[["anisotropy_effect", "high_harmonic_effect", "parallel_vs_wss"]].median())

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
artery_order = [c.name for c in V2_ARTERY_CASES]
ax = axes[0, 0]
data = [ATTRIBUTION_TABLE.loc[ATTRIBUTION_TABLE.artery_name == name, "anisotropy_effect"].to_numpy() for name in artery_order]
ax.boxplot(data, tick_labels=artery_order, showfliers=False)
ax.set_yscale("log")
ax.set_xticklabels(artery_order, rotation=30, ha="right")
ax.set_ylabel("RMS state difference")
ax.set_title("(a) Anisotropy-specific sensor effect")

ax = axes[0, 1]
data = [ATTRIBUTION_TABLE.loc[ATTRIBUTION_TABLE.artery_name == name, "high_harmonic_effect"].to_numpy() for name in artery_order]
ax.boxplot(data, tick_labels=artery_order, showfliers=False)
ax.set_yscale("log")
ax.set_xticklabels(artery_order, rotation=30, ha="right")
ax.set_ylabel("RMS state difference")
ax.set_title(r"(b) Effect of removing $h\geq3$")

ax = axes[1, 0]
obs = OBSERVABILITY_SPECTRUM[OBSERVABILITY_SPECTRUM.signal == "anisotropy_excess"]
obs_matrix = obs.pivot(index="artery_name", columns="harmonic", values="sensor_weighted_amplitude").reindex(artery_order)
im = ax.imshow(np.log10(np.maximum(obs_matrix.to_numpy(), 1e-14)), aspect="auto", origin="upper")
ax.set_yticks(np.arange(len(artery_order)), artery_order)
ax.set_xticks(np.arange(len(obs_matrix.columns)), obs_matrix.columns)
ax.set_xlabel("Harmonic")
ax.set_title("(c) Sensor-weighted anisotropy spectrum")
fig.colorbar(im, ax=ax, label=r"$\log_{10}$ linear response amplitude")

ax = axes[1, 1]
summary = ATTRIBUTION_TABLE.groupby("artery_name")[["anisotropy_fraction", "high_harmonic_fraction"]].median().reindex(artery_order)
pos = np.arange(len(summary))
ax.bar(pos - 0.18, summary["anisotropy_fraction"], 0.36, label="Anisotropy ablation")
ax.bar(pos + 0.18, summary["high_harmonic_fraction"], 0.36, label="High-harmonic ablation")
ax.set_xticks(pos, artery_order, rotation=30, ha="right")
ax.set_ylabel("Median operational effect fraction")
ax.set_title("(d) Fraction of full response removed")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
FIG4 = save_figure(fig, "Figure4_source_attribution")[0]
plt.show()
