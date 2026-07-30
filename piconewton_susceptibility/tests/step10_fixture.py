# ruff: noqa
from pathlib import Path
import hashlib, json, shutil
import numpy as np, pandas as pd


def build_step10_fixture(root):
    root=Path(root)
    if root.exists(): shutil.rmtree(root)
    root.mkdir()
    arteries=[
    ('aortic_root','Aortic Root',22.03,0.0006667),
    ('thoracic_aorta','Thoracic Aorta',17.62,0.0008333),
    ('femoral','Femoral',5.87,0.0025),
    ('carotid','Carotid',5.14,0.002857),
    ('iliac','Iliac',6.61,0.002222),
    ('brachial','Brachial',2.94,0.005),
    ]
    def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def write_step2(root):
        d=root/'bootstrap'/'step2'; d.mkdir(parents=True)
        source={'passed':True,'files':[]}
        runtime={'passed':True,'storage_mode':'local'}
        manifest={'step':2,'status':'complete','claim_bearing':True,'allowed_next_step':3}
        gate={'step':2,'passed':True,'allowed_next_step':3}
        for name,payload in [('source_validation.json',source),('runtime_validation.json',runtime),('bootstrap_manifest.json',manifest),('completion_gate.json',gate)]:
            (d/name).write_text(json.dumps(payload,indent=2))
        lines=[]
        for name in ['source_validation.json','runtime_validation.json','bootstrap_manifest.json']:
            lines.append(f"{sha(d/name)}  {name}\n")
        (d/'checksums.sha256').write_text(''.join(lines))

    def close_step(step, d, files, next_step):
        gate={'step':step,'profile':'publication','passed':True}
        (d/f'step{step}_gate.json').write_text(json.dumps(gate,indent=2))
        files=list(files)+[f'step{step}_gate.json']
        manifest={'step':step,'status':'complete','profile':'publication','scientific_scope':f'fixture_step_{step}','allowed_next_step':next_step,'files':{}}
        for name in files:
            p=d/name; manifest['files'][name]={'sha256':sha(p),'bytes':p.stat().st_size}
        (d/f'step{step}_manifest.json').write_text(json.dumps(manifest,indent=2))

    write_step2(root)
    s3=root/'step3_parent_continuity'; s3.mkdir()
    pd.DataFrame([{'artery_id':a,'passed':True} for a,*_ in arteries]).to_csv(s3/'parent_continuity.csv',index=False)
    close_step(3,s3,['parent_continuity.csv'],4)

    s4=root/'step4_perturbation'; s4.mkdir()
    pd.DataFrame([{'artery_id':a,'artery_name':n,'epsilon_max_used':0.02,'ut_order':1+0.0001*i,'uz_correction_order':2+0.0001*i,'signed_force_excess_order':2+0.0002*i} for i,(a,n,*_) in enumerate(arteries)]).to_csv(s4/'order_slopes.csv',index=False)
    pd.DataFrame([{'artery_id':a,'artery_name':n,'force_valid_epsilon_max_1pct':0.1 if a in {'femoral','brachial'} else 0.08,'ut_valid_epsilon_max_1pct':0.1,'uz_valid_epsilon_max_1pct':0.1,'minimum_required_epsilon':0.05} for a,n,*_ in arteries]).to_csv(s4/'validity_domains.csv',index=False)
    rows=[]
    for a,n,*_ in arteries:
      for e in [0.01,0.02,0.04,0.06,0.08,0.1]:
        err=e**2
        rows.append({'artery_id':a,'artery_name':n,'epsilon':e,'signed_excess_waveform_relative_l2':err,'signed_excess_rms_relative_error':err*0.99,'signed_excess_peak_relative_error':err*0.98})
    pd.DataFrame(rows).to_csv(s4/'epsilon_sweep.csv',index=False)
    pd.DataFrame([{'artery_id':a,'artery_name':n,'isotropic_waveform_relative_l2':1e-15,'epsilon_0p1_waveform_relative_l2':2e-15} for a,n,*_ in arteries]).to_csv(s4/'step3_waveform_continuity.csv',index=False)
    close_step(4,s4,['order_slopes.csv','validity_domains.csv','epsilon_sweep.csv','step3_waveform_continuity.csv'],5)

    s5=root/'step5_harmonic_kernel'; s5.mkdir()
    closure=[]
    for a,n,*_ in arteries:
      for k in ['second_order','exact_excess','second_order_synthetic_phase','exact_excess_synthetic_phase']:
        closure.append({'artery_id':a,'artery_name':n,'kernel_type':k,'waveform_relative_l2':1e-12,'spectrum_relative_l2':1e-12,'max_normalized_response_residual':1e-15})
    pd.DataFrame(closure).to_csv(s5/'kernel_closure.csv',index=False)
    dom=[]
    for a,n,*_ in arteries:
      for q,pair,share in [(0,(-1,1),0.8),(1,(-1,2),0.75),(2,(1,1),0.7)]:
        dom.append({'artery_id':a,'artery_name':n,'kernel_type':'second_order','q':q,'rank':1,'m':pair[0],'n':pair[1],'combined_contribution_abs':share,'fraction_of_pairwise_absolute_sum':share})
    pd.DataFrame(dom).to_csv(s5/'dominant_pairs.csv',index=False)
    close_step(5,s5,['kernel_closure.csv','dominant_pairs.csv'],6)

    s6=root/'step6_susceptibility'; s6.mkdir()
    native=[]
    for i,(a,n,alpha,eta) in enumerate(arteries):
      phi=1e-6*(7-i)
      native.append({'artery_id':a,'artery_name':n,'alpha':alpha,'eta':eta,'phi_rms':phi,'phi_peak_abs':phi*1.8,'outward_duty':0.35+0.03*i,'inward_duty':0.65-0.03*i,'predicted_rms_at_epsilon_0p1_n':phi*1e-2*1e-9})
    pd.DataFrame(native).to_csv(s6/'native_susceptibility.csv',index=False)
    crit=[]
    for a,n,*_ in arteries:
      for metric in ['rms','peak_abs']:
        for b in [1.0,10.0]:
          crit.append({'artery_id':a,'artery_name':n,'metric':metric,'primary_metric':metric=='rms','benchmark_pn':b,'benchmark_n':b*1e-12,'coefficient_n_per_epsilon2':1e-11,'perturbative_epsilon_critical':0.2+0.03*len(crit)%6,'validated_domain_max':0.08,'perturbative_estimate_in_domain':False,'formal_estimate_constitutively_admissible':b==1.0,'exact_metric_at_domain_max_n':1e-13,'exact_metric_at_domain_max_pn':0.1,'full_model_crossing':np.nan,'relative_prediction_error':np.nan,'status':'unreachable_and_perturbative_estimate_out_of_domain'})
    pd.DataFrame(crit).to_csv(s6/'critical_anisotropy.csv',index=False)
    close_step(6,s6,['native_susceptibility.csv','critical_anisotropy.csv'],7)

    s7=root/'step7_waveform_experiments'; s7.mkdir()
    cross=[]
    for mt in ['hydrodynamic','physiological']:
     for i,(va,vn,*_) in enumerate(arteries):
      for j,(wa,wn,*_) in enumerate(arteries):
       cross.append({'matrix_type':mt,'vessel_id':va,'vessel_name':vn,'waveform_id':wa,'waveform_name':wn,'native_diagonal':va==wa,'eta':0.002,'phi_rms':(7-i)*(7-j)*1e-7,'phi_peak_abs':(7-i)*(7-j)*2e-7,'outward_duty':0.4,'inward_duty':0.6,'high_harmonic_fraction':0.2})
    pd.DataFrame(cross).to_csv(s7/'crossed_susceptibility.csv',index=False)
    controls=[]
    for a,n,*_ in arteries:
     for fam,vals in [('native',[1]),('sign',[1.01]),('phase',[0.8,1.1]),('harmonic_removal',[0.3,0.6]),('harmonic_removal_rms_matched',[0.5,0.7])]:
      for k,v in enumerate(vals): controls.append({'vessel_id':a,'waveform_source':a,'control':f'{fam}_{k}','family':fam,'input_rms':1,'relative_to_native_rms':v,'fractional_change_from_native':v-1,'phi_rms':v*1e-6})
    pd.DataFrame(controls).to_csv(s7/'native_waveform_controls.csv',index=False)
    fams=[]
    for a,*_ in arteries:
     for fam in ['single_tone','two_tone','sparse_three_tone','spectral_slope']:
      for k in range(3): fams.append({'vessel_id':a,'control':f'{fam}_{k}','family':fam,'input_rms':1,'phi_rms':(4-k)*1e-6})
    pd.DataFrame(fams).to_csv(s7/'causal_waveform_families.csv',index=False)
    pd.DataFrame([{'matrix_type':'hydrodynamic','scale':'raw','vessel_fraction':0.93,'waveform_fraction':0.03,'interaction_fraction':0.04},{'matrix_type':'physiological','scale':'raw','vessel_fraction':0.95,'waveform_fraction':0.01,'interaction_fraction':0.04}]).to_csv(s7/'crossed_variance_decomposition.csv',index=False)
    close_step(7,s7,['crossed_susceptibility.csv','native_waveform_controls.csv','causal_waveform_families.csv','crossed_variance_decomposition.csv'],8)

    s8=root/'step8_reduced_law'; s8.mkdir()
    pd.DataFrame([{'candidate':'rank_1_universal_kernel','rank':1,'median_relative_error':0.022,'p90_relative_error':0.104,'maximum_relative_error':0.159,'passed':True},{'candidate':'rank_2_universal_kernel','rank':2,'median_relative_error':0.024,'p90_relative_error':0.105,'maximum_relative_error':0.159,'passed':True},{'candidate':'inverse_harmonic_scalar_moment','rank':0,'median_relative_error':0.069,'p90_relative_error':0.22,'maximum_relative_error':0.49,'passed':False}]).to_csv(s8/'model_selection.csv',index=False)
    pd.DataFrame([{'rank':1,'family':f,'median_relative_error':0.02,'maximum_relative_error':0.15} for f in ['native','phase_challenge','single_tone','two_tone']]).to_csv(s8/'compact_law_family_summary.csv',index=False)
    law={'law':'Phi_hat = C * alpha^p_alpha * eta^p_eta * Psi_R(g)','selected_rank':1,'prefactor':1.74054784,'alpha_exponent':-2.01167284,'eta_exponent':1.95232848,'retained_kernel_energy':0.99998604,'scalar_moment_selected':False}
    (s8/'reduced_law.json').write_text(json.dumps(law,indent=2))
    np.savez_compressed(s8/'step8_reduced_law.npz',selected_kernel=np.eye(12),scale_parameters=np.array([1,-2,2]))
    close_step(8,s8,['model_selection.csv','compact_law_family_summary.csv','reduced_law.json','step8_reduced_law.npz'],9)

    s9=root/'step9_robustness_claim_lock'; s9.mkdir()
    paths=['beta075_gamma125','beta125_gamma075','beta_low','delta_high','delta_low','gamma_low','gamma_only','reciprocal']
    pd.DataFrame([{'path':name,'shape_median':0.006,'shape_p90':0.03,'shape_maximum':0.11 if name=='gamma_only' else 0.08,'frozen_amplitude_median':0.1+i*0.05,'frozen_amplitude_p90':0.12+i*0.08,'frozen_amplitude_maximum':0.15+i*0.2} for i,name in enumerate(paths)]).to_csv(s9/'constitutive_path_metrics.csv',index=False)
    pd.DataFrame([{'path':name,'beta_ratio':1.0,'gamma_ratio':1.0,'delta':1.0,'prefactor_diagnostic':1.7,'alpha_exponent_diagnostic':-2.01,'eta_exponent_diagnostic':1.95,'alpha_exponent_drift':0.0,'eta_exponent_drift':0.0,'rank_one_energy':0.99998} for name in ['reciprocal',*paths[:-1]]]).to_csv(s9/'constitutive_path_summary.csv',index=False)
    pd.DataFrame([{'path':'reciprocal','vessel_id':a,'matrix_type':'hydrodynamic','scale':1e-8,'scale_ratio_to_reciprocal':1.0,'normalised_kernel_error_to_reciprocal':0.0} for a,*_ in arteries]).to_csv(s9/'constitutive_scale_ratios.csv',index=False)
    pd.DataFrame([{'path':'reciprocal','vessel_id':a,'matrix_type':'hydrodynamic','waveform_id':'native','family':'native','exact_phi_rms':1e-6,'shape_prediction':0.99e-6,'shape_relative_error':0.01,'frozen_reciprocal_prediction':1.01e-6,'frozen_amplitude_relative_error':0.01} for a,*_ in arteries]).to_csv(s9/'constitutive_shape_predictions.csv',index=False)
    pd.DataFrame([{'vessel_id':a,'eta_multiplier':m,'waveform_id':'native','family':'native','relative_error':abs(m-1)*0.1} for a,*_ in arteries for m in [0.8,0.9,1.0,1.1,1.2]]).to_csv(s9/'eta_robustness.csv',index=False)
    pd.DataFrame([{'path':'reciprocal','vessel_id':a,'matrix_type':'hydrodynamic','epsilon':0.08,'kernel_relative_error':0.006,'hierarchy_scale':1e-8,'exact_scale':1.006e-8} for a,*_ in arteries]).to_csv(s9/'finite_epsilon_closure.csv',index=False)
    pd.DataFrame([{'radial_order':r,'quadrature_nodes':q,'vessel_id':a,'matrix_type':'hydrodynamic','waveform_id':'native','relative_change':1e-6 if r==120 else 1e-7,'kernel_relative_change':8e-7} for r,q in [(120,192),(180,384)] for a,*_ in arteries]).to_csv(s9/'resolution_robustness.csv',index=False)
    pd.DataFrame([{'prefactor_relative_error':1e-14,'alpha_exponent_absolute_error':1e-14,'eta_exponent_absolute_error':1e-15,'selected_kernel_relative_l2':1e-14,'reconstructed_rank_one_energy':0.999986}]).to_csv(s9/'step8_law_continuity.csv',index=False)
    claim={'status':'locked','selected_law':{'rank':1,'prefactor':1.74054784,'alpha_exponent':-2.01167284,'eta_exponent':1.95232848},'permitted_primary_claim':'locked claim','permitted_secondary_claims':[],'required_qualifier':'amplitude restricted','prohibited_claims':['disease prediction'],'allowed_next_step':10}
    (s9/'claim_lock.json').write_text(json.dumps(claim,indent=2))
    np.savez_compressed(s9/'step9_archive.npz',dummy=np.array([1]))
    files=[p.name for p in s9.glob('*.csv')]+['claim_lock.json','step9_archive.npz']
    close_step(9,s9,files,10)
    return root
