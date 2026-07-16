#!/usr/bin/env python3
"""Run the picoNewton v3 workflow outside the notebook."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from piconewton_v3.model import (  # noqa: E402
    FluidProperties,
    HydrodynamicConfig,
    SensorConfig,
    V2_ARTERY_CASES,
    V2_EXPECTED_BLOB_SHA,
    isotropic_validation,
)
from piconewton_v3.provenance import (  # noqa: E402
    environment_snapshot,
    execute_stripped_v2,
    git_commit_or_unknown,
    strip_notebook_outputs,
    validate_v2_blob,
    write_json,
)
from piconewton_v3.study_io import StudyStore, resolve_study_root  # noqa: E402
from piconewton_v3.workflow import (  # noqa: E402
    evaluate_effect_gates,
    export_publication_dataset,
    fit_wss_surrogate,
    generate_physiological_design,
    generate_publication_figures,
    generate_sobol_design,
    parameter_dominance,
    run_nominal_controls,
    run_parameter_grid,
    run_physiological_coverage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["quick", "publication"], default="quick")
    parser.add_argument("--storage", choices=["auto", "drive", "local"], default=None)
    parser.add_argument("--root", type=Path, default=None, help="Override output root")
    parser.add_argument(
        "--skip-v2-hash",
        action="store_true",
        help="Development-only: allow a quick run outside the parent repository",
    )
    return parser.parse_args()


def load_config(profile: str) -> dict:
    path = PROJECT_ROOT / "configs" / f"{profile}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def copy_publication_metadata(run_root: Path, config_path: Path) -> None:
    bundle = run_root / "publication_bundle"
    bundle.mkdir(exist_ok=True)
    for source in [
        PROJECT_ROOT / "data" / "data_dictionary.csv",
        PROJECT_ROOT / "data" / "source_manifest.csv",
        PROJECT_ROOT / "data" / "v2_harmonic_inputs.csv",
        PROJECT_ROOT / "data" / "publication_dataset_schema.json",
        PROJECT_ROOT / "configs" / "effect_gates.json",
        PROJECT_ROOT / "configs" / "numerical_thresholds.json",
        PROJECT_ROOT / "docs" / "DATA_AVAILABILITY.md",
        PROJECT_ROOT / "docs" / "CODE_AVAILABILITY.md",
        config_path,
    ]:
        shutil.copy2(source, bundle / source.name)


def main() -> int:
    args = parse_args()
    config = load_config(args.profile)
    if args.storage is not None:
        config["storage"]["mode"] = args.storage
    if args.root is not None:
        config["storage"]["mode"] = "local"
        config["storage"]["local_root"] = str(args.root)

    output_root, storage_mode = resolve_study_root(
        mode=config["storage"]["mode"],
        drive_subdir=config["storage"]["drive_subdir"],
        local_root=config["storage"]["local_root"],
    )
    store = StudyStore(output_root)
    store.initialize_layout()

    repo_root = PROJECT_ROOT.parent
    code_commit = git_commit_or_unknown(repo_root)
    v2_path = repo_root / "picoNewton_v2.ipynb"
    v2_record: dict
    if v2_path.exists():
        v2_record = validate_v2_blob(v2_path)
    elif args.profile == "publication" or not args.skip_v2_hash:
        raise FileNotFoundError(
            f"{v2_path} not found. Run from the full picoNewton repository or use "
            "--skip-v2-hash only for an isolated quick development test."
        )
    else:
        v2_record = {
            "path": str(v2_path),
            "expected_git_blob_sha": V2_EXPECTED_BLOB_SHA,
            "observed_git_blob_sha": "not-available-in-isolated-development-tree",
            "passed": False,
            "development_skip": True,
        }

    hydro_config = HydrodynamicConfig(
        radial_order=int(config["radial_order"]),
        time_points=int(config["time_points"]),
        quadrature_nodes=int(config["quadrature_nodes"]),
        beta=float(config["beta"]),
        gamma=float(config["gamma"]),
        delta=float(config["delta"]),
        mode=config["solver_mode"],
    )
    fluid = FluidProperties()
    sensor = SensorConfig()

    run_id, run_root = store.create_run(
        config=config,
        code_commit=code_commit,
        v2_blob_sha=V2_EXPECTED_BLOB_SHA,
        solver_mode=hydro_config.mode,
        random_seed=int(config["random_seed"]),
    )
    store.set_status(run_id, "running")
    try:
        write_json(run_root / "provenance" / "environment.json", environment_snapshot())
        write_json(run_root / "provenance" / "v2_hash_guard.json", v2_record)

        stripped_record = None
        if v2_path.exists():
            stripped_path = run_root / "provenance" / "picoNewton_v2_stripped.ipynb"
            stripped_record = strip_notebook_outputs(v2_path, stripped_path)
            write_json(run_root / "provenance" / "v2_stripping.json", stripped_record)
        if config.get("run_v2_cold_regression", False):
            if stripped_record is None:
                raise RuntimeError("publication v2 cold regression requires the parent repository")
            execute_stripped_v2(
                stripped_path,
                run_root / "provenance" / "picoNewton_v2_executed.ipynb",
                timeout_s=int(config.get("v2_execution_timeout_s", 1800)),
            )

        validation = pd.DataFrame(
            isotropic_validation(radial_order=hydro_config.radial_order)
        )
        if not validation["passed"].all():
            raise RuntimeError("isotropic Womersley validation failed")
        store.write_csv(f"runs/{run_id}/summaries/isotropic_validation.csv", validation)
        store.register_file(run_id, "summaries/isotropic_validation.csv", "output")

        control_table, waveform_table, hydrodynamics = run_nominal_controls(
            V2_ARTERY_CASES,
            hydro_config,
            sensor,
            fluid,
            coupling_length_m=1e-9,
            wss_activation_volume_m3=1e-22,
            seed=int(config["random_seed"]),
        )
        parameter_grid = run_parameter_grid(
            V2_ARTERY_CASES,
            hydrodynamics,
            fluid,
            coupling_lengths_m=config["coupling_lengths_m"],
            relaxation_times_s=config["relaxation_times_s"],
        )
        surrogate, surrogate_parameters = fit_wss_surrogate(
            V2_ARTERY_CASES,
            hydrodynamics,
            fluid,
            sensor,
        )
        dominance = parameter_dominance(parameter_grid)
        sobol = generate_sobol_design(
            int(config["sobol_samples"]), seed=int(config["random_seed"])
        )
        store.write_csv(f"runs/{run_id}/summaries/sobol_sensor_design.csv", sobol)
        store.register_file(run_id, "summaries/sobol_sensor_design.csv", "output")

        artery_ranges = pd.read_csv(PROJECT_ROOT / "data" / "physiological_artery_ranges.csv")
        physiological_design = generate_physiological_design(
            artery_ranges,
            V2_ARTERY_CASES,
            int(config["sobol_samples"]),
            seed=int(config["random_seed"]),
        )
        store.write_csv(
            f"runs/{run_id}/summaries/physiological_design.csv", physiological_design
        )
        store.register_file(run_id, "summaries/physiological_design.csv", "output")
        physiological_summary, physiological_spectra = run_physiological_coverage(
            physiological_design,
            V2_ARTERY_CASES,
            hydro_config,
            sensor,
            checkpoint_dir=run_root / "checkpoints" / "physiological_coverage",
        )
        store.write_csv(
            f"runs/{run_id}/summaries/physiological_coverage.csv", physiological_summary
        )
        store.register_file(run_id, "summaries/physiological_coverage.csv", "output")
        store.write_csv(
            f"runs/{run_id}/spectra/physiological_force_spectra.csv", physiological_spectra
        )
        store.register_file(run_id, "spectra/physiological_force_spectra.csv", "output")
        gates = evaluate_effect_gates(
            parameter_grid,
            surrogate,
            config["coupling_lengths_m"],
            config["relaxation_times_s"],
            physiological_coverage=physiological_summary,
        )
        write_json(
            run_root / "summaries" / "wss_surrogate_parameters.json",
            surrogate_parameters,
        )
        store.register_file(run_id, "summaries/wss_surrogate_parameters.json", "output")

        export_publication_dataset(
            store,
            run_id,
            waveform_table=waveform_table,
            control_table=control_table,
            parameter_grid=parameter_grid,
            surrogate_table=surrogate,
            gate_table=gates,
            dominance_table=dominance,
            hydrodynamics=hydrodynamics,
        )
        if config.get("generate_figures", True):
            figures = generate_publication_figures(
                run_root / "figures",
                waveform_table,
                control_table,
                parameter_grid,
                surrogate,
                gates,
                dominance,
            )
            for figure in figures:
                store.register_file(run_id, f"figures/{figure.name}", "output")

        copy_publication_metadata(
            run_root, PROJECT_ROOT / "configs" / f"{args.profile}.json"
        )
        store.write_checksums(f"runs/{run_id}", "provenance/checksums.sha256")
        store.set_status(run_id, "complete")
    except Exception:
        store.set_status(run_id, "failed")
        raise

    print(json.dumps({"run_id": run_id, "run_root": str(run_root), "storage": storage_mode}, indent=2))
    print(gates[["criterion_id", "passed", "observed"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
