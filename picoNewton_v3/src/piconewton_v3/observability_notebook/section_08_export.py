"""Notebook section generated from the validated observability workflow.

Executed by ``picoNewton_v3_mechanosensory_observability.ipynb`` in the
notebook shared namespace. Keep scientific thresholds in configuration files.
"""

TABLES = {
    "summaries/hydrodynamic_decomposition.csv": HYDRO_SUMMARY,
    "summaries/gromeka_lamb_closure.csv": CLOSURE_TABLE,
    "spectra/hydrodynamic_spectra.csv": HYDRO_SPECTRA,
    "summaries/hydrodynamic_nonredundancy.csv": NONREDUNDANCY_TABLE,
    "spectra/lamb_wss_phase_difference.csv": PHASE_TABLE,
    "summaries/synthetic_transfer_map.csv": TRANSFER_TABLE,
    "summaries/source_attribution.csv": ATTRIBUTION_TABLE,
    "spectra/sensor_weighted_observability.csv": OBSERVABILITY_SPECTRUM,
    "summaries/loo_wss_parameter_results.csv": LOO_PARAMETER_TABLE,
    "summaries/loo_wss_summary.csv": LOO_SUMMARY,
    "summaries/loo_nominal_waveforms.csv": LOO_NOMINAL_TABLE,
    "summaries/parameter_grid.csv": PARAMETER_GRID,
    "summaries/original_gate_table.csv": GATE_TABLE,
    "summaries/gate_margins.csv": GATE_MARGIN_TABLE,
}
for relative, table in TABLES.items():
    STORE.write_csv(f"runs/{RUN_ID}/{relative}", table)
    STORE.register_file(RUN_ID, relative, "output")

FIGURE_MANIFEST = pd.DataFrame([
    {"figure": 1, "stem": "Figure1_signal_continuity_and_decomposition", "scientific_role": "Continuity and force-source decomposition"},
    {"figure": 2, "stem": "Figure2_hydrodynamic_nonredundancy", "scientific_role": "Sensor-independent Lamb–WSS distinction"},
    {"figure": 3, "stem": "Figure3_mechanosensory_transfer", "scientific_role": "Analytical and nonlinear kinetic transfer"},
    {"figure": 4, "stem": "Figure4_source_attribution", "scientific_role": "Anisotropy and harmonic attribution"},
    {"figure": 5, "stem": "Figure5_leave_one_artery_out_WSS_competition", "scientific_role": "Held-out WSS challenge"},
    {"figure": 6, "stem": "Figure6_gate_margins_and_claim_map", "scientific_role": "Robustness and claim selection"},
])
STORE.write_csv(f"runs/{RUN_ID}/summaries/figure_manifest.csv", FIGURE_MANIFEST)
STORE.register_file(RUN_ID, "summaries/figure_manifest.csv", "output")
write_json(RUN_ROOT / "summaries/original_wss_surrogate_parameters.json", ORIGINAL_SURROGATE_PARAMETERS)
STORE.register_file(RUN_ID, "summaries/original_wss_surrogate_parameters.json", "output")

bundle = RUN_ROOT / "publication_bundle"
bundle.mkdir(exist_ok=True)
for source in [
    PROJECT_ROOT / "data/data_dictionary.csv",
    PROJECT_ROOT / "data/source_manifest.csv",
    PROJECT_ROOT / "data/v2_harmonic_inputs.csv",
    PROJECT_ROOT / "configs/effect_gates.json",
    PROJECT_ROOT / "configs/numerical_thresholds.json",
    CONFIG_PATH,
]:
    shutil.copy2(source, bundle / source.name)

CHECKSUM_PATH = STORE.write_checksums(f"runs/{RUN_ID}", "provenance/checksums.sha256")
STORE.set_status(RUN_ID, "complete")
print({"run_root": str(RUN_ROOT), "checksums": str(CHECKSUM_PATH)})
