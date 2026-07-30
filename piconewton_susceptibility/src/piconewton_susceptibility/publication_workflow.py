# ruff: noqa: E501
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import zipfile
from typing import Any

from .publication_core import (
    Step10Config,
    environment_manifest,
    file_records,
    git_provenance,
    load_json,
    sha256,
    validate_manifest_root,
    validate_step2,
    validate_step9,
    write_json,
)
from .publication_figures import build_main_figures, build_supplementary_figures
from .publication_tables import build_publication_tables


STEP_DIRECTORY_MAP = {
    2: "bootstrap/step2",
    3: "step3_parent_continuity",
    4: "step4_perturbation",
    5: "step5_harmonic_kernel",
    6: "step6_susceptibility",
    7: "step7_waveform_experiments",
    8: "step8_reduced_law",
    9: "step9_robustness_claim_lock",
}


def _validate_prior_steps(workflow_root: Path) -> tuple[dict[int, Path], dict[int, dict[str, Any]]]:
    roots: dict[int, Path] = {}
    records: dict[int, dict[str, Any]] = {}
    for step, directory in STEP_DIRECTORY_MAP.items():
        root = workflow_root / directory
        expected = step + 1 if step < 9 else 10
        if step == 2:
            records[step] = validate_step2(root)
        elif step == 9:
            records[step] = validate_step9(root)
        else:
            records[step] = validate_manifest_root(root, step, expected)
        roots[step] = root
    return roots, records


def _copy_locked_science(step_roots: dict[int, Path], output_root: Path) -> list[Path]:
    locked = output_root / "locked_science"
    locked.mkdir(parents=True, exist_ok=True)
    mapping = {
        step_roots[8] / "reduced_law.json": locked / "reduced_law.json",
        step_roots[8] / "step8_reduced_law.npz": locked / "step8_reduced_law.npz",
        step_roots[9] / "claim_lock.json": locked / "claim_lock.json",
        step_roots[9] / "step8_law_continuity.csv": locked / "step8_law_continuity.csv",
    }
    outputs = []
    for source, target in mapping.items():
        if not source.is_file():
            raise RuntimeError(f"locked scientific artifact is missing: {source}")
        shutil.copy2(source, target)
        outputs.append(target)
    return outputs


def _workflow_inventory(records: dict[int, dict[str, Any]], output_root: Path) -> Path:
    path = output_root / "provenance" / "workflow_inventory.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "profile", "status", "allowed_next_step", "scientific_scope", "file_count"])
        writer.writeheader()
        for step, item in sorted(records.items()):
            manifest = item["manifest"]
            writer.writerow({
                "step": step,
                "profile": manifest.get("profile"),
                "status": manifest.get("status"),
                "allowed_next_step": manifest.get("allowed_next_step"),
                "scientific_scope": manifest.get("scientific_scope"),
                "file_count": len(manifest.get("files", {})),
            })
    return path


def _archive(workflow_root: Path, output_root: Path, config: Step10Config, generated: list[Path]) -> Path:
    archive_path = output_root / "publication_archive.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for directory in config.include_step_directories:
            root = workflow_root / directory
            if not root.exists():
                continue
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, Path("steps") / directory / path.relative_to(root))
        for path in sorted(set(generated)):
            if path.is_file() and path != archive_path:
                archive.write(path, Path("publication") / path.relative_to(output_root))
    return archive_path


def run_publication_archive(
    output_root: str | Path,
    workflow_root: str | Path,
    repo_root: str | Path | None = None,
    config: Step10Config | None = None,
) -> dict[str, Any]:
    config = config or Step10Config()
    config.validate()
    output_root = Path(output_root).resolve()
    workflow_root = Path(workflow_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    step_roots, prior_records = _validate_prior_steps(workflow_root)
    claim = prior_records[9]["claim_lock"]
    law = load_json(step_roots[8] / "reduced_law.json")

    generated: list[Path] = []
    figure_records = build_main_figures(step_roots, output_root, config.figure_formats, config.figure_dpi)
    supplementary_records = build_supplementary_figures(
        step_roots, output_root, config.figure_formats, config.figure_dpi
    )
    generated.extend(figure_records["figures"])
    generated.extend(figure_records["sources"])
    generated.extend(supplementary_records["figures"])
    generated.extend(supplementary_records["sources"])
    generated.extend(build_publication_tables(step_roots, output_root))
    generated.extend(_copy_locked_science(step_roots, output_root))

    provenance_root = output_root / "provenance"
    provenance_root.mkdir(parents=True, exist_ok=True)
    environment_path = provenance_root / "environment_manifest.json"
    git_path = provenance_root / "git_provenance.json"
    inventory_path = _workflow_inventory(prior_records, output_root)
    write_json(environment_path, environment_manifest())
    write_json(git_path, git_provenance(repo_root))
    generated.extend([environment_path, git_path, inventory_path])

    inventory_json = output_root / "publication_inventory.json"
    write_json(
        inventory_json,
        {
            "scientific_steps": sorted(prior_records),
            "locked_claim": claim,
            "reduced_law": law,
            "generated_files": sorted(path.relative_to(output_root).as_posix() for path in generated),
        },
    )
    generated.append(inventory_json)
    checksums_path = output_root / "publication_checksums.sha256"
    checksums_path.write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(output_root).as_posix()}\n"
            for path in sorted(generated)
            if path.is_file()
        ),
        encoding="utf-8",
    )
    generated.append(checksums_path)

    claim_exact = load_json(output_root / "locked_science" / "claim_lock.json") == claim
    law_exact = load_json(output_root / "locked_science" / "reduced_law.json") == law
    six_figures = len([p for p in figure_records["figures"] if p.suffix == f".{config.figure_formats[0]}"]) == 6
    all_formats = len(figure_records["figures"]) == 6 * len(config.figure_formats)
    figure_sources_complete = len(figure_records["sources"]) >= 12
    main_tables_complete = len(list((output_root / "tables" / "main").glob("*.csv"))) >= 5
    supplementary_tables_complete = len(list((output_root / "tables" / "supplementary").glob("*.csv"))) >= 6

    prearchive_gate = {
        "step": 10,
        "profile": config.profile,
        "steps_2_to_9_checksum_validated": len(prior_records) == 8,
        "step9_claim_lock_consumed": claim.get("status") == "locked",
        "claim_lock_preserved_exactly": claim_exact,
        "reduced_law_preserved_exactly": law_exact,
        "six_main_figures_complete": six_figures,
        "all_requested_figure_formats_complete": all_formats,
        "figure_source_csv_complete": figure_sources_complete,
        "supplementary_figures_complete": len(supplementary_records["figures"]) == 3 * len(config.figure_formats),
        "main_tables_complete": main_tables_complete,
        "supplementary_tables_complete": supplementary_tables_complete,
        "environment_manifest_complete": environment_path.is_file(),
        "git_provenance_recorded": git_path.is_file(),
        "scientific_claim_modified": False,
        "new_scientific_fit_run": False,
    }
    required = [name for name in prearchive_gate if name not in {"step", "profile", "scientific_claim_modified", "new_scientific_fit_run"}]
    prearchive_gate["passed"] = all(bool(prearchive_gate[name]) for name in required)
    gate_path = output_root / "step10_gate.json"
    write_json(gate_path, prearchive_gate)
    generated.append(gate_path)

    archive_path = _archive(workflow_root, output_root, config, generated)
    archive_checksum_path = output_root / "publication_archive.sha256"
    archive_checksum_path.write_text(f"{sha256(archive_path)}  {archive_path.name}\n", encoding="utf-8")
    generated.extend([archive_path, archive_checksum_path])

    manifest = {
        "step": 10,
        "status": "complete" if prearchive_gate["passed"] else "failed",
        "profile": config.profile,
        "scientific_scope": "final_publication_archive_and_manuscript_facing_outputs",
        "locked_claim_status": claim.get("status"),
        "selected_rank": law.get("selected_rank"),
        "allowed_next_step": None,
        "workflow_complete": bool(prearchive_gate["passed"]),
        "archive_sha256": sha256(archive_path),
        "gates": prearchive_gate,
        "files": file_records(output_root, generated),
    }
    manifest_path = output_root / "step10_manifest.json"
    write_json(manifest_path, manifest)
    return {
        "manifest": manifest,
        "gate": prearchive_gate,
        "archive": archive_path,
        "figures": figure_records["figures"],
        "figure_sources": figure_records["sources"] + supplementary_records["sources"],
    }
