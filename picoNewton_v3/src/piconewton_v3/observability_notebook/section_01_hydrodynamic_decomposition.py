"""Notebook section generated from the validated observability workflow.

Executed by ``picoNewton_v3_mechanosensory_observability.ipynb`` in the
notebook shared namespace. Keep scientific thresholds in configuration files.
"""

FLUID = FluidProperties()
SENSOR_NOMINAL = SensorConfig()
HYDRO_CONFIG = HydrodynamicConfig(
    radial_order=CONFIG["radial_order"],
    time_points=CONFIG["time_points"],
    quadrature_nodes=CONFIG["quadrature_nodes"],
    beta=CONFIG["beta"],
    gamma=CONFIG["gamma"],
    delta=CONFIG["delta"],
    mode="verified",
)
ISO_CONFIG = HydrodynamicConfig(
    radial_order=CONFIG["radial_order"],
    time_points=CONFIG["time_points"],
    quadrature_nodes=CONFIG["quadrature_nodes"],
    beta=0.0,
    gamma=0.0,
    delta=1.0,
    mode="verified",
)
REPRO_CONFIG = HydrodynamicConfig(
    radial_order=CONFIG["radial_order"],
    time_points=CONFIG["time_points"],
    quadrature_nodes=CONFIG["quadrature_nodes"],
    beta=CONFIG["beta"],
    gamma=CONFIG["gamma"],
    delta=CONFIG["delta"],
    mode="reproduction",
)

ISOTROPIC_VALIDATION = pd.DataFrame(
    isotropic_validation(radial_order=HYDRO_CONFIG.radial_order)
)
if not ISOTROPIC_VALIDATION["passed"].all():
    STORE.set_status(RUN_ID, "failed")
    raise RuntimeError("Analytical isotropic validation failed")

def rms(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x * x)))


def harmonic_table(signal, artery_id, artery_name, signal_name, maximum_harmonic=12):
    x = np.asarray(signal, dtype=float)
    coeff = np.fft.rfft(x) / len(x)
    rows = []
    for h in range(min(maximum_harmonic, len(coeff) - 1) + 1):
        amplitude = abs(coeff[h]) if h == 0 else 2.0 * abs(coeff[h])
        rows.append({
            "artery_id": artery_id,
            "artery_name": artery_name,
            "signal": signal_name,
            "harmonic": h,
            "amplitude": float(amplitude),
            "phase_rad": float(np.angle(coeff[h])),
            "power": float(abs(coeff[h]) ** 2),
        })
    return rows


HYDRO_FULL = {}
HYDRO_ISO = {}
HYDRO_LOW = {}
HYDRO_REPRO = {}
SUMMARY_ROWS = []
SPECTRAL_ROWS = []
CLOSURE_ROWS = []

for case in V2_ARTERY_CASES:
    full = compute_hydrodynamics(
        case, HYDRO_CONFIG, FLUID, include_near_wall_fields=True
    )
    iso = compute_hydrodynamics(case, ISO_CONFIG, FLUID)
    low = compute_hydrodynamics(case, HYDRO_CONFIG, FLUID, harmonics_retained=2)
    repro = compute_hydrodynamics(case, REPRO_CONFIG, FLUID)
    HYDRO_FULL[case.artery_id] = full
    HYDRO_ISO[case.artery_id] = iso
    HYDRO_LOW[case.artery_id] = low
    HYDRO_REPRO[case.artery_id] = repro

    force = np.asarray(full["force_signed_n"])
    exposure = np.asarray(full["force_exposure_n"])
    force_iso = np.asarray(iso["force_signed_n"])
    force_aniso = force - force_iso

    r_m = np.asarray(full["near_wall_r_star"]) * case.radius_m
    uz = np.asarray(full["u_z_m_s"])
    ut = np.asarray(full["u_theta_m_s"])
    lamb = np.asarray(full["lamb_r_m_s2"])
    kinetic = 0.5 * (uz**2 + ut**2)
    grad_kinetic = np.gradient(kinetic, r_m, axis=0, edge_order=2)
    convective_r = -(ut**2) / r_m[:, None]
    closure = grad_kinetic - lamb - convective_r
    interior = slice(2, -2) if len(r_m) > 8 else slice(1, -1)
    numerator = np.linalg.norm(closure[interior])
    denominator = (
        np.linalg.norm(grad_kinetic[interior])
        + np.linalg.norm(lamb[interior])
        + np.linalg.norm(convective_r[interior])
    )
    closure_residual = float(numerator / max(denominator, 1e-30))

    SUMMARY_ROWS.append({
        "artery_id": case.artery_id,
        "artery_name": case.name,
        "alpha": full["alpha"],
        "verified_exposure_rms_pN": rms(exposure) * 1e12,
        "reproduction_exposure_rms_pN": rms(repro["force_exposure_n"]) * 1e12,
        "signed_rms_pN": rms(force) * 1e12,
        "isotropic_rms_pN": rms(force_iso) * 1e12,
        "anisotropy_excess_rms_pN": rms(force_aniso) * 1e12,
        "wss_rms_pa": rms(full["wall_shear_pa"]),
        "gl_closure_residual": closure_residual,
        "linear_backward_residual": full["max_normalized_backward_residual"],
    })
    CLOSURE_ROWS.append({
        "artery_id": case.artery_id,
        "artery_name": case.name,
        "gl_closure_residual": closure_residual,
    })
    for name, signal in {
        "signed_total": force,
        "exposure": exposure,
        "isotropic": force_iso,
        "anisotropy_excess": force_aniso,
        "wall_shear": full["wall_shear_pa"],
    }.items():
        SPECTRAL_ROWS.extend(
            harmonic_table(signal, case.artery_id, case.name, name)
        )

HYDRO_SUMMARY = pd.DataFrame(SUMMARY_ROWS)
HYDRO_SPECTRA = pd.DataFrame(SPECTRAL_ROWS)
CLOSURE_TABLE = pd.DataFrame(CLOSURE_ROWS)

if CLOSURE_TABLE["gl_closure_residual"].max() > 1e-4:
    STORE.set_status(RUN_ID, "failed")
    raise RuntimeError("Gromeka–Lamb closure exceeded the predeclared notebook tolerance")

display(ISOTROPIC_VALIDATION)
display(HYDRO_SUMMARY)
