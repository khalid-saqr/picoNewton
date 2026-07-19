"""Calibration-file loading and claim-readiness audit for picoNewton_v4."""
from __future__ import annotations
from pathlib import Path
import json
from .vector_interface import DirectionalSLS, VectorInterfaceParameters
from .endpoints import EndpointParameters

REQUIRED_SOURCE_GROUPS = (
    "tangential_mechanics", "normal_mechanics", "signed_force_localization",
    "exposure_localization", "spatial_channel_distribution", "channel_density",
    "single_channel_conductance", "calcium_conversion", "current_detection_limit",
    "calcium_detection_limit",
)

def reference_parameterization() -> tuple[VectorInterfaceParameters, EndpointParameters, dict[str, object]]:
    interface=VectorInterfaceParameters(); endpoint=EndpointParameters()
    audit={"calibration_status":"uncalibrated_engineering_reference","complete":False,
           "missing_source_groups":list(REQUIRED_SOURCE_GROUPS),"source_file":None}
    return interface,endpoint,audit

def load_parameterization(path: Path, *, require_calibrated: bool=False) -> tuple[VectorInterfaceParameters, EndpointParameters, dict[str, object]]:
    raw=json.loads(path.read_text(encoding="utf-8"))
    values=raw.get("values",{})
    tang=values.get("tangential",{}); normal=values.get("normal",{})
    interface=VectorInterfaceParameters(
        tangential=DirectionalSLS(
            float(tang["instantaneous_modulus_pa"]),float(tang["relaxed_modulus_pa"]),
            float(tang["fast_time_s"]),float(tang["slow_time_s"]),float(tang["fast_fraction"])),
        normal=DirectionalSLS(
            float(normal["instantaneous_modulus_pa"]),float(normal["relaxed_modulus_pa"]),
            float(normal["fast_time_s"]),float(normal["slow_time_s"]),float(normal["fast_fraction"])),
        signed_force_area_m2=float(values["signed_force_area_m2"]),
        exposure_area_m2=float(values["exposure_area_m2"]),
        wss_transfer_fraction=float(values["wss_transfer_fraction"]),
        signed_force_transfer_fraction=float(values["signed_force_transfer_fraction"]),
        exposure_transfer_fraction=float(values["exposure_transfer_fraction"]),
        areal_modulus_n_m=float(values["areal_modulus_n_m"]),
        baseline_apical_tension_n_m=float(values["baseline_apical_tension_n_m"]),
        baseline_junctional_tension_n_m=float(values["baseline_junctional_tension_n_m"]),
        apical_curvature_radius_m=float(values["apical_curvature_radius_m"]),
        junctional_curvature_radius_m=float(values["junctional_curvature_radius_m"]),
        tangential_to_apical_cross_fraction=float(values.get("tangential_to_apical_cross_fraction",0.0)),
        exposure_to_junctional_cross_fraction=float(values.get("exposure_to_junctional_cross_fraction",0.0)),
        maximum_pressure_mmhg=float(values.get("maximum_pressure_mmhg",70.0)),
        gradmu_mv=float(values["gradmu_mv"]),
        apical_channel_fraction=float(values["apical_channel_fraction"]),
    )
    status=str(raw.get("calibration_status","incomplete"))
    endpoint_values=values.get("endpoint",{})
    endpoint=EndpointParameters(
        channel_count=float(endpoint_values["channel_count"]),
        single_channel_conductance_ps=float(endpoint_values["single_channel_conductance_ps"]),
        membrane_voltage_mv=float(endpoint_values["membrane_voltage_mv"]),
        reversal_potential_mv=float(endpoint_values["reversal_potential_mv"]),
        calcium_current_fraction=float(endpoint_values["calcium_current_fraction"]),
        cell_volume_l=float(endpoint_values["cell_volume_l"]),
        calcium_clearance_time_s=float(endpoint_values["calcium_clearance_time_s"]),
        current_detection_limit_pa=float(endpoint_values["current_detection_limit_pa"]),
        calcium_detection_limit_nm=float(endpoint_values["calcium_detection_limit_nm"]),
        calibration_status="experimentally_calibrated" if status=="experimentally_calibrated" else status,
    )
    sources=raw.get("sources",{})
    missing=[name for name in REQUIRED_SOURCE_GROUPS if not str(sources.get(name,"")).strip()]
    complete=status=="experimentally_calibrated" and not missing
    audit={"calibration_status":status,"complete":complete,"missing_source_groups":missing,
           "source_file":str(path),"sources":sources}
    interface.validate(); endpoint.validate()
    if require_calibrated and not complete:
        raise ValueError(f"calibration is incomplete; missing source groups: {missing}")
    return interface,endpoint,audit
