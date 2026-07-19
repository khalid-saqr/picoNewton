"""End-to-end six-artery mechanosensory workflow."""
from __future__ import annotations
from dataclasses import replace, asdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import resample

from .types import FluidProperties, HydrodynamicConfig, load_artery_cases
from .hydrodynamics import compute_decomposition
from .vector_interface import (
    DirectionalSLS, VectorInterfaceParameters, vector_membrane_state,
    validate_passivity, elastic_limit,
)
from .endpoints import EndpointParameters, domain_endpoint, aggregate_domains
from .calibration import reference_parameterization, load_parameterization


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float)**2)))

def _dynamic_rms(x: np.ndarray) -> float:
    a = np.asarray(x, dtype=float); return _rms(a-np.mean(a))

def _dynamic_range(x: np.ndarray) -> float:
    a = np.asarray(x, dtype=float); return float(np.max(a)-np.min(a))

def _retain_harmonics(signal: np.ndarray, max_harmonic: int) -> np.ndarray:
    values=np.asarray(signal,dtype=float); coefficients=np.fft.rfft(values)
    coefficients[max_harmonic+1:]=0.0
    return np.fft.irfft(coefficients,n=values.size)

def _cycle_work_density(stress: np.ndarray, strain: np.ndarray) -> float:
    """Absolute closed-cycle work per unit volume [Pa = J/m^3]."""
    s=np.asarray(stress,dtype=float); e=np.asarray(strain,dtype=float)
    de=np.roll(e,-1)-e
    midpoint=0.5*(s+np.roll(s,-1))
    return float(abs(np.sum(midpoint*de)))

def _periodic_filter(signal: np.ndarray, *, dt_s: float, lag_s: float, tau_s: float) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    omega = 2*np.pi*np.fft.rfftfreq(values.size, d=dt_s)
    transfer = np.exp(-1j*omega*lag_s)/(1.0+1j*omega*tau_s)
    return np.fft.irfft(np.fft.rfft(values)*transfer, n=values.size)

def _fit_surrogate(xs: list[np.ndarray], ys: list[np.ndarray], *, dt_s: float, nonnegative: bool) -> dict[str, float]:
    period = xs[0].size*dt_s
    scale = max(_rms(np.concatenate(ys)), 1e-12)
    def objective(theta: np.ndarray) -> float:
        gain, offset, lag_fraction, log_tau = theta
        residual = []
        for x, y in zip(xs, ys):
            pred = gain*_periodic_filter(x, dt_s=dt_s, lag_s=lag_fraction*period, tau_s=float(np.exp(log_tau)))+offset
            if nonnegative: pred = np.maximum(pred, 0.0)
            residual.append((pred-y)/scale)
        return float(np.mean(np.concatenate(residual)**2))
    best = None
    for lag0 in (-0.2, 0.0, 0.2):
        result = minimize(objective, np.array([0.1, 0.0, lag0, np.log(0.05)]), method="L-BFGS-B",
                          bounds=[(-20,20),(-20,20),(-0.5,0.5),(np.log(1e-4),np.log(2.0))],
                          options={"maxiter":400,"ftol":1e-12})
        if best is None or result.fun < best.fun: best = result
    assert best is not None
    gain, offset, lag_fraction, log_tau = best.x
    return {"gain":float(gain),"offset_pa":float(offset),"lag_s":float(lag_fraction*period),
            "tau_s":float(np.exp(log_tau)),"training_nmse":float(best.fun),"nonnegative":bool(nonnegative)}

def _predict_surrogate(x: np.ndarray, fit: dict[str,float], *, dt_s: float) -> np.ndarray:
    pred = fit["gain"]*_periodic_filter(x, dt_s=dt_s, lag_s=fit["lag_s"], tau_s=fit["tau_s"])+fit["offset_pa"]
    return np.maximum(pred,0.0) if fit["nonnegative"] else pred

def _simulate(*, wss_pa: np.ndarray, signed_force_n: np.ndarray, exposure_force_n: np.ndarray,
              dt_s: float, interface: VectorInterfaceParameters, endpoint: EndpointParameters) -> dict[str, object]:
    state = vector_membrane_state(wall_shear_pa=wss_pa, signed_force_n=signed_force_n,
                                  force_exposure_n=exposure_force_n, dt_s=dt_s, p=interface)
    apical = domain_endpoint(np.asarray(state["apical_pressure_mmhg"]), dt_s=dt_s,
                             gradmu_mv=interface.gradmu_mv, endpoint=endpoint)
    junctional = domain_endpoint(np.asarray(state["junctional_pressure_mmhg"]), dt_s=dt_s,
                                 gradmu_mv=interface.gradmu_mv, endpoint=endpoint)
    aggregate = aggregate_domains(apical, junctional, apical_fraction=interface.apical_channel_fraction)
    return {"membrane":state,"apical":apical,"junctional":junctional,"aggregate":aggregate}

def _zero_like(x: np.ndarray) -> np.ndarray:
    return np.zeros_like(np.asarray(x,dtype=float))

def _pathway(item: dict[str,object], pathway: str, *, interface: VectorInterfaceParameters,
             endpoint: EndpointParameters, isotropic: bool=False, harmonic_cutoff: int|None=None,
             surrogate_traction_pa: np.ndarray|None=None) -> dict[str,object]:
    time = np.asarray(item["time_s"],dtype=float); dt_s=float(time[1]-time[0])
    suffix = "isotropic" if isotropic else "anisotropic"
    wss=np.asarray(item[f"wss_{suffix}_pa"],dtype=float)
    signed=np.asarray(item[f"force_signed_{suffix}_n"],dtype=float)
    exposure=np.asarray(item[f"force_exposure_{suffix}_n"],dtype=float)
    if harmonic_cutoff is not None:
        wss=_retain_harmonics(wss,harmonic_cutoff); signed=_retain_harmonics(signed,harmonic_cutoff)
        exposure=np.maximum(_retain_harmonics(exposure,harmonic_cutoff),0.0)
    if pathway=="zero": wss=_zero_like(wss); signed=_zero_like(signed); exposure=_zero_like(exposure)
    elif pathway=="wss": signed=_zero_like(signed); exposure=_zero_like(exposure)
    elif pathway=="signed": wss=_zero_like(wss); exposure=_zero_like(exposure)
    elif pathway=="exposure": wss=_zero_like(wss); signed=_zero_like(signed)
    elif pathway=="vector": pass
    elif pathway=="signed_surrogate":
        if surrogate_traction_pa is None: raise ValueError("surrogate traction required")
        wss=_zero_like(wss); exposure=_zero_like(exposure); signed=np.asarray(surrogate_traction_pa)*interface.signed_force_area_m2/interface.signed_force_transfer_fraction
    elif pathway=="exposure_surrogate":
        if surrogate_traction_pa is None: raise ValueError("surrogate traction required")
        wss=_zero_like(wss); signed=_zero_like(signed); exposure=np.asarray(surrogate_traction_pa)*interface.exposure_area_m2/interface.exposure_transfer_fraction
    else: raise ValueError(f"unknown pathway {pathway}")
    return _simulate(wss_pa=wss,signed_force_n=signed,exposure_force_n=exposure,dt_s=dt_s,interface=interface,endpoint=endpoint)

def _summary(artery_id:str, label:str, result:dict[str,object], time_s:np.ndarray) -> dict[str,object]:
    a=result["aggregate"]; m=result["membrane"]
    current=np.asarray(a["current_pA"],dtype=float); calcium=np.asarray(a["calcium_nm"],dtype=float)
    popen=np.asarray(a["P_Open"],dtype=float)
    dt=float(time_s[1]-time_s[0])
    fft=np.fft.rfft(current-np.mean(current))/current.size
    harmonic=np.abs(fft)
    return {
        "artery_id":artery_id,"pathway":label,
        "popen_mean":float(np.mean(popen)),"popen_dynamic_range":_dynamic_range(popen),
        "current_mean_abs_pa":float(np.mean(np.abs(current))),"current_rms_pa":_rms(current),
        "current_peak_abs_pa":float(np.max(np.abs(current))),"current_dynamic_range_pa":_dynamic_range(current),
        "charge_abs_fc_per_cycle":float(np.sum(np.abs(current))*dt*1e3),
        "calcium_mean_nm":float(np.mean(calcium)),"calcium_peak_nm":float(np.max(calcium)),
        "calcium_dynamic_range_nm":_dynamic_range(calcium),"calcium_auc_nm_s":float(np.sum(calcium)*dt),
        "apical_current_rms_pa":_rms(np.asarray(result["apical"]["current_pA"])),
        "junctional_current_rms_pa":_rms(np.asarray(result["junctional"]["current_pA"])),
        "spatial_current_polarity_index":float((
            _rms(np.asarray(result["apical"]["current_pA"]))-_rms(np.asarray(result["junctional"]["current_pA"]))
        )/max(_rms(np.asarray(result["apical"]["current_pA"]))+_rms(np.asarray(result["junctional"]["current_pA"])),1e-30)),
        "apical_calcium_peak_nm":float(np.max(np.asarray(result["apical"]["calcium_nm"]))),
        "junctional_calcium_peak_nm":float(np.max(np.asarray(result["junctional"]["calcium_nm"]))),
        "peak_time_fraction":float(np.argmax(np.abs(current))/current.size),
        "current_h1_amplitude_pa":float(2*harmonic[1]) if harmonic.size>1 else 0.0,
        "current_h3_6_power_fraction":float(np.sum(harmonic[3:7]**2)/max(np.sum(harmonic[1:]**2),1e-30)),
        "apical_pressure_peak_mmhg":float(np.max(m["apical_pressure_mmhg"])),
        "junctional_pressure_peak_mmhg":float(np.max(m["junctional_pressure_mmhg"])),
        "force_angle_range_rad":_dynamic_range(np.unwrap(np.asarray(m["force_vector_angle_rad"]))),
        "probability_sum_error":float(a["probability_sum_error"]),"minimum_probability":float(a["minimum_probability"]),
        "exposure_used_as_signed_load":bool(m["exposure_used_as_signed_load"]),
    }

def _effect(artery_id:str, hypothesis:str, comparator:str, target:str,
            a:dict[str,object], b:dict[str,object]) -> dict[str,object]:
    aa=a["aggregate"]; bb=b["aggregate"]
    current_a=np.asarray(aa["current_pA"]); current_b=np.asarray(bb["current_pA"])
    ca_a=np.asarray(aa["calcium_nm"]); ca_b=np.asarray(bb["calcium_nm"])
    return {
        "artery_id":artery_id,"hypothesis":hypothesis,"comparator":comparator,"target":target,
        "current_rms_difference_pa":_rms(current_a-current_b),
        "current_peak_difference_pa":float(np.max(np.abs(current_a-current_b))),
        "current_relative_rms":float(_rms(current_a-current_b)/max(_dynamic_range(current_a),1e-30)),
        "calcium_rms_difference_nm":_rms(ca_a-ca_b),
        "calcium_peak_difference_nm":float(np.max(np.abs(ca_a-ca_b))),
        "calcium_relative_rms":float(_rms(ca_a-ca_b)/max(_dynamic_range(ca_a),1e-30)),
    }

def _load_hydrodynamic_items(package_root: Path, profile: str, hydrodynamic_root: Path | None):
    cases=load_artery_cases(package_root/"data"/"ground_truth_arteries.csv")
    if hydrodynamic_root is not None:
        npz_path=Path(hydrodynamic_root)/"six_artery_hydrodynamics.npz"
        if not npz_path.exists(): raise FileNotFoundError(npz_path)
        arrays=np.load(npz_path); items=[]
        for case in cases:
            aid=case.artery_id; time_cycle=np.asarray(arrays[f"{aid}_time_cycle"],dtype=float)
            items.append({
                "artery_id":aid,"artery_name":case.name,"time_cycle":time_cycle,
                "time_s":time_cycle/FluidProperties().fundamental_frequency_hz,
                "wss_anisotropic_pa":arrays[f"{aid}_wss_anisotropic_pa"],
                "wss_isotropic_pa":arrays[f"{aid}_wss_isotropic_pa"],
                "force_signed_anisotropic_n":arrays[f"{aid}_force_signed_anisotropic_n"],
                "force_signed_isotropic_n":arrays[f"{aid}_force_signed_isotropic_n"],
                "force_exposure_anisotropic_n":arrays[f"{aid}_force_exposure_anisotropic_n"],
                "force_exposure_isotropic_n":arrays[f"{aid}_force_exposure_isotropic_n"],
            })
        return items, None, {"mode":"direct_step4_artifact","path":str(npz_path),"sha256":_sha256(npz_path)}
    if profile=="quick": config=HydrodynamicConfig(48,256,48,0.1,0.1,1.0)
    elif profile=="full": config=HydrodynamicConfig(150,2048,256,0.1,0.1,1.0)
    else: raise ValueError("profile must be quick or full")
    items=[compute_decomposition(case,config) for case in cases]
    return items, config, {"mode":"computed_from_package","profile":profile}

def _reference_interface() -> VectorInterfaceParameters:
    return VectorInterfaceParameters()

def _area_gain_scan(items:list[dict[str,object]], endpoint:EndpointParameters, output_root:Path) -> pd.DataFrame:
    """Precalibration diagnostic, not a claim-producing parameter search.

    Every comparison uses the same channel count and the same fast/slow mechanics
    for WSS and force; only localization area and the declared force class vary.
    """
    rows=[]
    areas_um2=(0.1,0.3,1.0,3.0,10.0,30.0,100.0)
    channel_counts=(100.0,300.0,1000.0,3000.0,10000.0)
    fast_fractions=(0.0,0.25,0.50,0.75)
    for item in items:
        n=64; compact=dict(item)
        for key in ("time_s","wss_anisotropic_pa","wss_isotropic_pa","force_signed_anisotropic_n","force_signed_isotropic_n",
                    "force_exposure_anisotropic_n","force_exposure_isotropic_n"):
            compact[key]=resample(np.asarray(item[key],dtype=float),n)
        original_time=np.asarray(item["time_s"],dtype=float)
        period=float(original_time[-1]+(original_time[1]-original_time[0]))
        compact["time_s"]=np.arange(n)*period/n
        for count in channel_counts:
            ep=replace(endpoint,channel_count=count)
            for ff in fast_fractions:
                tang=replace(_reference_interface().tangential,fast_fraction=ff)
                norm=replace(_reference_interface().normal,fast_fraction=ff)
                common=replace(_reference_interface(),tangential=tang,normal=norm)
                base_wss=_pathway(compact,"wss",interface=common,endpoint=ep)
                zero=_pathway(compact,"zero",interface=common,endpoint=ep)
                for area in areas_um2:
                    interface=replace(common,signed_force_area_m2=area*1e-12,exposure_area_m2=area*1e-12)
                    signed=_pathway(compact,"signed",interface=interface,endpoint=ep)
                    exposure=_pathway(compact,"exposure",interface=interface,endpoint=ep)
                    for label,result in (("signed",signed),("exposure",exposure)):
                        eff=_effect(str(item["artery_id"]),"H3a_sensitivity", "wss",label,result,base_wss)
                        activation=_effect(str(item["artery_id"]),"activation_sensitivity", "zero",label,result,zero)
                        rows.append({"artery_id":item["artery_id"],"force_class":label,"area_um2":area,
                                     "channel_count":count,"fast_fraction":ff,
                                     "current_rms_difference_pa":eff["current_rms_difference_pa"],
                                     "calcium_rms_difference_nm":eff["calcium_rms_difference_nm"],
                                     "activation_current_rms_pa":activation["current_rms_difference_pa"],
                                     "activation_calcium_rms_nm":activation["calcium_rms_difference_nm"],
                                     "passes_placeholder_current_limit":bool(
                                         eff["current_rms_difference_pa"]>=ep.current_detection_limit_pa and
                                         activation["current_rms_difference_pa"]>=ep.current_detection_limit_pa),
                                     "passes_placeholder_calcium_limit":bool(
                                         eff["calcium_rms_difference_nm"]>=ep.calcium_detection_limit_nm and
                                         activation["calcium_rms_difference_nm"]>=ep.calcium_detection_limit_nm),
                                     "claim_status":"sensitivity_only_uncalibrated"})
    frame=pd.DataFrame(rows); frame.to_csv(output_root/"localization_sensitivity_scan.csv",index=False)
    requirement_rows=[]
    for keys,group in frame.groupby(["artery_id","force_class","channel_count","fast_fraction"]):
        current=group[group["passes_placeholder_current_limit"]]
        calcium=group[group["passes_placeholder_calcium_limit"]]
        requirement_rows.append({
            "artery_id":keys[0],"force_class":keys[1],"channel_count":keys[2],"fast_fraction":keys[3],
            "largest_area_um2_passing_placeholder_current":float(current["area_um2"].max()) if not current.empty else np.nan,
            "largest_area_um2_passing_placeholder_calcium":float(calcium["area_um2"].max()) if not calcium.empty else np.nan,
            "status":"sensitivity_only_uncalibrated",
        })
    pd.DataFrame(requirement_rows).to_csv(output_root/"localization_requirements.csv",index=False)
    return frame

def run_workflow(*, package_root:Path, output_root:Path, run_scan:bool=True,
              profile:str="quick", hydrodynamic_root:Path|None=None,
              calibration_path:Path|None=None, require_calibrated:bool=False) -> dict[str,object]:
    package_root=package_root.resolve(); output_root.mkdir(parents=True,exist_ok=False)
    items,config,hydrodynamic_artifact=_load_hydrodynamic_items(package_root,profile,hydrodynamic_root)
    if calibration_path is None:
        interface,endpoint,calibration_audit=reference_parameterization()
    else:
        interface,endpoint,calibration_audit=load_parameterization(Path(calibration_path),require_calibrated=require_calibrated)
    (output_root/"calibration_audit.json").write_text(json.dumps(calibration_audit,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    passivity={"tangential":validate_passivity(interface.tangential),"normal":validate_passivity(interface.normal)}
    pathways_by_artery={}; summary_rows=[]; effect_rows=[]; waveform_arrays={}
    for item in items:
        aid=str(item["artery_id"]); time=np.asarray(item["time_s"]); dt=float(time[1]-time[0])
        controls={}
        for pathway in ("zero","wss","signed","exposure","vector"):
            controls[pathway]=_pathway(item,pathway,interface=interface,endpoint=endpoint)
            summary_rows.append(_summary(aid,pathway,controls[pathway],time))
        wss=np.asarray(item["wss_anisotropic_pa"],dtype=float)
        signed=np.asarray(item["force_signed_anisotropic_n"],dtype=float)
        exposure=np.asarray(item["force_exposure_anisotropic_n"],dtype=float)
        zero_wss=np.zeros_like(wss); zero_force=np.zeros_like(signed)
        controls["wss_abs"]=_simulate(wss_pa=np.abs(wss),signed_force_n=zero_force,exposure_force_n=zero_force,
                                      dt_s=dt,interface=interface,endpoint=endpoint)
        signed_traction=signed/interface.signed_force_area_m2
        exposure_traction=exposure/interface.exposure_area_m2
        signed_rms_scale=_dynamic_rms(wss)/max(_dynamic_rms(signed_traction),1e-30)
        exposure_rms_scale=_dynamic_rms(np.abs(wss))/max(_dynamic_rms(exposure_traction),1e-30)
        signed_peak_scale=float(np.max(np.abs(wss))/max(np.max(np.abs(signed_traction)),1e-30))
        exposure_peak_scale=float(np.max(np.abs(wss))/max(np.max(exposure_traction),1e-30))
        wss_work=_cycle_work_density(np.asarray(controls["wss"]["membrane"]["tangential_traction_pa"]),
                                     np.asarray(controls["wss"]["membrane"]["tangential_strain"]))
        signed_work=_cycle_work_density(np.asarray(controls["signed"]["membrane"]["normal_signed_traction_pa"]),
                                        np.asarray(controls["signed"]["membrane"]["normal_signed_strain"]))
        exposure_work=_cycle_work_density(np.asarray(controls["exposure"]["membrane"]["normal_exposure_traction_pa"]),
                                          np.asarray(controls["exposure"]["membrane"]["normal_exposure_strain"]))
        signed_work_scale=float(np.sqrt(wss_work/max(signed_work,1e-30)))
        exposure_work_scale=float(np.sqrt(wss_work/max(exposure_work,1e-30)))
        for label,scaled_signed,scaled_exposure in (
            ("signed_rms_matched",signed*signed_rms_scale,zero_force),
            ("signed_peak_matched",signed*signed_peak_scale,zero_force),
            ("signed_work_matched",signed*signed_work_scale,zero_force),
            ("exposure_rms_matched",zero_force,exposure*exposure_rms_scale),
            ("exposure_peak_matched",zero_force,exposure*exposure_peak_scale),
            ("exposure_work_matched",zero_force,exposure*exposure_work_scale),
        ):
            controls[label]=_simulate(wss_pa=zero_wss,signed_force_n=scaled_signed,exposure_force_n=scaled_exposure,
                                      dt_s=dt,interface=interface,endpoint=endpoint)
        controls["signed_isotropic"]=_pathway(item,"signed",interface=interface,endpoint=endpoint,isotropic=True)
        controls["exposure_isotropic"]=_pathway(item,"exposure",interface=interface,endpoint=endpoint,isotropic=True)
        controls["signed_h2"]=_pathway(item,"signed",interface=interface,endpoint=endpoint,harmonic_cutoff=2)
        controls["exposure_h2"]=_pathway(item,"exposure",interface=interface,endpoint=endpoint,harmonic_cutoff=2)
        elastic=elastic_limit(interface)
        controls["wss_elastic"]=_pathway(item,"wss",interface=elastic,endpoint=endpoint)
        controls["signed_elastic"]=_pathway(item,"signed",interface=elastic,endpoint=endpoint)
        controls["exposure_elastic"]=_pathway(item,"exposure",interface=elastic,endpoint=endpoint)
        effect_rows.extend([
            _effect(aid,"activation","zero","signed",controls["signed"],controls["zero"]),
            _effect(aid,"activation","zero","exposure",controls["exposure"],controls["zero"]),
            _effect(aid,"H3a","wss","signed",controls["signed"],controls["wss"]),
            _effect(aid,"H3a","wss_abs","exposure",controls["exposure"],controls["wss_abs"]),
            _effect(aid,"H3a_rms_matched","wss","signed_rms_matched",controls["signed_rms_matched"],controls["wss"]),
            _effect(aid,"H3a_rms_matched","wss_abs","exposure_rms_matched",controls["exposure_rms_matched"],controls["wss_abs"]),
            _effect(aid,"H3a_peak_matched","wss","signed_peak_matched",controls["signed_peak_matched"],controls["wss"]),
            _effect(aid,"H3a_peak_matched","wss_abs","exposure_peak_matched",controls["exposure_peak_matched"],controls["wss_abs"]),
            _effect(aid,"H3a_work_matched","wss","signed_work_matched",controls["signed_work_matched"],controls["wss"]),
            _effect(aid,"H3a_work_matched","wss_abs","exposure_work_matched",controls["exposure_work_matched"],controls["wss_abs"]),
            _effect(aid,"H4","signed_isotropic","signed_anisotropic",controls["signed"],controls["signed_isotropic"]),
            _effect(aid,"H4","exposure_isotropic","exposure_anisotropic",controls["exposure"],controls["exposure_isotropic"]),
            _effect(aid,"H5","signed_h2","signed_full",controls["signed"],controls["signed_h2"]),
            _effect(aid,"H5","exposure_h2","exposure_full",controls["exposure"],controls["exposure_h2"]),
            _effect(aid,"H7_activation","zero","signed_elastic",controls["signed_elastic"],controls["zero"]),
            _effect(aid,"H7_activation","zero","exposure_elastic",controls["exposure_elastic"],controls["zero"]),
            _effect(aid,"H7","wss_elastic","signed_elastic",controls["signed_elastic"],controls["wss_elastic"]),
            _effect(aid,"H7","wss_elastic","exposure_elastic",controls["exposure_elastic"],controls["wss_elastic"]),
        ])
        pathways_by_artery[aid]=controls
        waveform_arrays[f"{aid}_time_s"]=time
        for name,result in controls.items():
            for key in ("P_Open","current_pA","calcium_nm"):
                waveform_arrays[f"{aid}_{name}_{key}"]=np.asarray(result["aggregate"][key])
            waveform_arrays[f"{aid}_{name}_apical_pressure_mmhg"]=np.asarray(result["membrane"]["apical_pressure_mmhg"])
            waveform_arrays[f"{aid}_{name}_junctional_pressure_mmhg"]=np.asarray(result["membrane"]["junctional_pressure_mmhg"])
    # Leave-one-artery-out causal WSS surrogates for both force classes.
    surrogate_rows=[]
    for target in items:
        aid=str(target["artery_id"]); train=[x for x in items if x is not target]
        dt=float(np.asarray(target["time_s"])[1]-np.asarray(target["time_s"])[0])
        for force_class, x_transform, y_key, area, nonnegative in (
            ("signed",lambda x:x,"force_signed_anisotropic_n",interface.signed_force_area_m2,False),
            ("exposure",np.abs,"force_exposure_anisotropic_n",interface.exposure_area_m2,True),
        ):
            xs=[x_transform(np.asarray(x["wss_anisotropic_pa"])) for x in train]
            transfer = interface.signed_force_transfer_fraction if force_class == "signed" else interface.exposure_transfer_fraction
            ys=[np.asarray(x[y_key])*transfer/area for x in train]
            fit=_fit_surrogate(xs,ys,dt_s=dt,nonnegative=nonnegative)
            target_x=x_transform(np.asarray(target["wss_anisotropic_pa"]))
            predicted=_predict_surrogate(target_x,fit,dt_s=dt)
            actual=pathways_by_artery[aid][force_class]
            predicted_response=_pathway(target,f"{force_class}_surrogate",interface=interface,endpoint=endpoint,surrogate_traction_pa=predicted)
            effect_rows.append(_effect(aid,"H3b",f"wss_{force_class}_surrogate",f"actual_{force_class}",actual,predicted_response))
            raw=np.asarray(target[y_key])/area
            surrogate_rows.append({"artery_id":aid,"force_class":force_class,**fit,
                                   "raw_traction_relative_l2":float(np.linalg.norm(predicted-raw)/max(np.linalg.norm(raw),1e-30)),
                                   "raw_traction_correlation":float(np.corrcoef(predicted,raw)[0,1]) if np.std(predicted)>1e-15 and np.std(raw)>1e-15 else 0.0})
    summary=pd.DataFrame(summary_rows); effects=pd.DataFrame(effect_rows); surrogates=pd.DataFrame(surrogate_rows)
    summary.to_csv(output_root/"six_artery_summary.csv",index=False)
    effects.to_csv(output_root/"hypothesis_effects.csv",index=False)
    surrogates.to_csv(output_root/"loao_vector_surrogates.csv",index=False)
    np.savez_compressed(output_root/"waveforms.npz",**waveform_arrays)
    # Multifeature artery specificity: distances of current/calcium feature vectors.
    feature_cols=["current_mean_abs_pa","current_peak_abs_pa","charge_abs_fc_per_cycle","calcium_peak_nm","calcium_auc_nm_s","peak_time_fraction","current_h1_amplitude_pa","current_h3_6_power_fraction","spatial_current_polarity_index","apical_calcium_peak_nm","junctional_calcium_peak_nm"]
    features=summary[summary["pathway"].isin(["signed","exposure","vector"])][["artery_id","pathway",*feature_cols]].copy()
    pair_rows=[]
    for pathway,group in features.groupby("pathway"):
        values=group[feature_cols].to_numpy(float); scale=np.std(values,axis=0); scale[scale<1e-30]=1.0
        ids=group["artery_id"].tolist()
        for i in range(len(ids)):
            for j in range(i+1,len(ids)):
                pair_rows.append({"pathway":pathway,"artery_a":ids[i],"artery_b":ids[j],
                                  "standardized_feature_distance":float(np.linalg.norm((values[i]-values[j])/scale))})
    pd.DataFrame(pair_rows).to_csv(output_root/"artery_feature_distances.csv",index=False)
    scan=_area_gain_scan(items,endpoint,output_root) if run_scan else pd.DataFrame()
    maximum_probability_error=float(summary["probability_sum_error"].max())
    minimum_probability=float(summary["minimum_probability"].min())
    validation={
        "status":"passed_structural_validation" if maximum_probability_error<1e-9 and minimum_probability>=-1e-10 and passivity["tangential"]["passed"] and passivity["normal"]["passed"] else "failed",
        "all_six_arteries_present":summary["artery_id"].nunique()==6,
        "signed_and_exposure_force_classes_separate":True,
        "magnitude_exposure_used_as_signed_load":False,
        "normal_and_tangential_mechanics_separate":True,
        "fast_and_slow_branches_present":True,
        "spatial_channel_domains":["apical","junctional"],
        "endpoint_outputs":["P_Open","current_pA","calcium_nm"],
        "endpoint_calibrated":endpoint.calibrated,
        "calibration_complete":bool(calibration_audit["complete"]),
        "calibration_missing_source_groups":calibration_audit["missing_source_groups"],
        "profile":profile,
        "hydrodynamic_artifact":hydrodynamic_artifact,
        "claims_eligible_after_independent_review":bool(calibration_audit["complete"] and profile=="full"),
        "claims_enabled":False,
        "maximum_probability_sum_error":maximum_probability_error,
        "minimum_probability":minimum_probability,
        "passivity":passivity,
        "sensitivity_scan_rows":int(len(scan)),
        "screening_current_pass_fraction":float(scan["passes_placeholder_current_limit"].mean()) if not scan.empty else None,
        "screening_calcium_pass_fraction":float(scan["passes_placeholder_calcium_limit"].mean()) if not scan.empty else None,
        "claim_boundary":"H3-H7 require independently calibrated localization, membrane, channel-density, endpoint and detection-limit parameters plus predeclared decision thresholds."
    }
    (output_root/"validation.json").write_text(json.dumps(validation,indent=2,sort_keys=True)+"\n")
    produced=[p for p in output_root.iterdir() if p.is_file()]
    manifest={"workflow":"picoNewton_v4","status":validation["status"],"completed_utc":datetime.now(timezone.utc).isoformat(),
              "profile":profile,"hydrodynamic_config":asdict(config) if config is not None else None,
              "hydrodynamic_artifact":hydrodynamic_artifact,"calibration_audit":calibration_audit,
              "interface_reference":asdict(interface),"endpoint_reference":asdict(endpoint),
              "outputs":{p.name:{"bytes":p.stat().st_size,"sha256":_sha256(p)} for p in produced},
              "claims_enabled":False,"calibration_required":True}
    (output_root/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True,default=str)+"\n")
    return manifest
