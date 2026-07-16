"""Notebook section generated from the validated observability workflow.

Executed by ``picoNewton_v3_mechanosensory_observability.ipynb`` in the
notebook shared namespace. Keep scientific thresholds in configuration files.
"""

def linear_response_amplitude(work_amplitude, harmonic, omega_dimensionless, p0=0.01):
    return p0 * (1.0 - p0) * work_amplitude / np.sqrt(1.0 + (harmonic * omega_dimensionless) ** 2)


def exact_sinusoidal_response(lambda_work, omega_dimensionless, time_points):
    tau0 = omega_dimensionless / FLUID.angular_frequency_rad_s
    sensor = SensorConfig(
        basal_probability=SENSOR_NOMINAL.basal_probability,
        relaxation_time_s=max(tau0, 1e-8),
        transition_fraction=SENSOR_NOMINAL.transition_fraction,
        temperature_k=SENSOR_NOMINAL.temperature_k,
    )
    t = np.arange(time_points) / time_points
    work = lambda_work * np.sin(2.0 * np.pi * t)
    probability, residual = periodic_sensor_solution(work, FLUID.fundamental_frequency_hz, sensor)
    return rms(probability - probability.mean()), residual


TRANSFER_HARMONICS = np.arange(1, 13)
TRANSFER_OMEGAS = np.array([0.01, 0.1, 1.0, 10.0])
LAMBDA_GRID = np.logspace(-3, 1, 21 if PROFILE == "quick" else 51)
OMEGA_GRID = np.logspace(-3, 2, 21 if PROFILE == "quick" else 51)
TRANSFER_TIME_POINTS = CONFIG["time_points"]

TRANSFER_ROWS = []
for omega in OMEGA_GRID:
    for lambda_work_value in LAMBDA_GRID:
        exact_rms, closure = exact_sinusoidal_response(lambda_work_value, omega, TRANSFER_TIME_POINTS)
        linear_rms = linear_response_amplitude(lambda_work_value, 1, omega) / np.sqrt(2.0)
        TRANSFER_ROWS.append({
            "Lambda": lambda_work_value,
            "Omega": omega,
            "exact_rms": exact_rms,
            "linear_rms": linear_rms,
            "relative_nonlinearity": abs(exact_rms - linear_rms) / max(exact_rms, 1e-12),
            "periodic_residual": closure,
        })
TRANSFER_TABLE = pd.DataFrame(TRANSFER_ROWS)
if TRANSFER_TABLE["periodic_residual"].max() > 1e-10:
    raise RuntimeError("Synthetic transfer map failed periodic closure")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ax = axes[0, 0]
for omega in TRANSFER_OMEGAS:
    gain = SENSOR_NOMINAL.basal_probability * (1 - SENSOR_NOMINAL.basal_probability) / np.sqrt(1 + (TRANSFER_HARMONICS * omega) ** 2)
    ax.plot(TRANSFER_HARMONICS, gain, marker="o", label=fr"$\Omega={omega:g}$")
ax.set_yscale("log")
ax.set_xlabel("Harmonic, h")
ax.set_ylabel("Linear probability gain per unit work")
ax.set_title("(a) Kinetic low-pass transfer")
ax.legend(frameon=False, fontsize=8)

ax = axes[0, 1]
for omega in TRANSFER_OMEGAS:
    subset = TRANSFER_TABLE[np.isclose(TRANSFER_TABLE.Omega, OMEGA_GRID[np.argmin(abs(OMEGA_GRID - omega))])]
    ax.loglog(subset["Lambda"], subset["exact_rms"], label=fr"Exact, $\Omega\approx{omega:g}$")
    ax.loglog(subset["Lambda"], subset["linear_rms"], linestyle="--", alpha=0.7)
ax.set_xlabel(r"Work amplitude, $\Lambda$")
ax.set_ylabel("State-response RMS")
ax.set_title("(b) Linear theory versus nonlinear periodic state")
ax.legend(frameon=False, fontsize=7, ncol=2)

ax = axes[1, 0]
response_matrix = TRANSFER_TABLE.pivot(index="Omega", columns="Lambda", values="exact_rms").sort_index().sort_index(axis=1)
im = ax.imshow(np.log10(np.maximum(response_matrix.to_numpy(), 1e-12)), aspect="auto", origin="lower", extent=[np.log10(response_matrix.columns.min()), np.log10(response_matrix.columns.max()), np.log10(response_matrix.index.min()), np.log10(response_matrix.index.max())])
ax.set_xlabel(r"$\log_{10}\Lambda$")
ax.set_ylabel(r"$\log_{10}\Omega$")
ax.set_title("(c) Exact mechanosensory observability map")
fig.colorbar(im, ax=ax, label=r"$\log_{10}$ response RMS")

ax = axes[1, 1]
nonlinear_matrix = TRANSFER_TABLE.pivot(index="Omega", columns="Lambda", values="relative_nonlinearity").sort_index().sort_index(axis=1)
im = ax.imshow(np.minimum(nonlinear_matrix.to_numpy(), 2.0), aspect="auto", origin="lower", extent=[np.log10(nonlinear_matrix.columns.min()), np.log10(nonlinear_matrix.columns.max()), np.log10(nonlinear_matrix.index.min()), np.log10(nonlinear_matrix.index.max())], vmin=0, vmax=1)
ax.set_xlabel(r"$\log_{10}\Lambda$")
ax.set_ylabel(r"$\log_{10}\Omega$")
ax.set_title("(d) Departure from weak-input theory")
fig.colorbar(im, ax=ax, label="Relative deviation")
fig.tight_layout()
FIG3 = save_figure(fig, "Figure3_mechanosensory_transfer")[0]
plt.show()
