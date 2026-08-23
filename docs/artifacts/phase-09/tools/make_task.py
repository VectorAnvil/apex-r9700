#!/usr/bin/env python3
"""Freeze Phase 9 gate+up fusion feasibility evidence before inspection."""
import hashlib,json
from datetime import UTC,datetime
from pathlib import Path
R=Path(__file__).resolve().parents[1];A=R.parent;P7=A/'results_phase07_pp_mmq_attribution_20260808';P8=A/'results_phase08_pp_mmq_y64_20260808';P2=A/'results_phase02_qwen36_p2p_nographs_20260808'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
 out=R/'task.json'
 if out.exists():raise RuntimeError('task is immutable')
 evidence={'phase2_raw_pp_trace':P2/'pp/raw/cornelius/2490453_kernel_trace.csv','phase7_task':P7/'task-v1.json','phase7_baseline':P7/'baseline-v1-result.json','phase7_attribution':P7/'attribution-summary-v1.json','phase8_static_task':P8/'static-task.json','phase8_static_gate':P8/'static-gate-result.json','phase8_static_proof':P8/'static-infeasibility.json','p2p_patch':P8/'evidence/registered-p2p-baseline.patch','feasibility_markdown':R/'FEASIBILITY.md','feasibility_json':R/'feasibility.json','source_hashes':R/'evidence/source-hashes.txt','pair_proof':R/'pair-proof.json'}
 if any(not p.is_file() for p in evidence.values()):raise RuntimeError('required provenance is unavailable')
 required_findings=['graph_adjacency','shared_input_layouts','tensor_parallel_ownership','numerical_swiglu_contract','fusion_api_fit','implementation_boundary','rollback_default_off_selector','resource_plan','correctness_plan','dispatch_plan','timing_plan','whole_pp_plan']
 fields={'source_commit':'259f2e2a531af9ed3efa7f66adaa5eb5b53da95f','provenance':{name:sha(path) for name,path in evidence.items()},'scope':{'phase':'PP','actual_n':512,'type':'Q6_K','operations':['ffn_gate','ffn_up'],'combined_pp_dispatch_duration':{'numerator_ns':955179820+959724245,'denominator_ns':4999245044,'value':(955179820+959724245)/4999245044},'excluded':['source_edit','candidate_build','GPU_workload','reference_kernel_integration','candidate_registration','TG_changes','ffn_down_port']},'phase9_semantics':{'source_inspection_only':True,'no_code_changes':True,'no_build':True,'no_gpu_execution':True,'no_candidate_claim':True},'required_findings':required_findings,'finding_contract':{'each':'object with status supported|unsupported and evidence list','feasible_only_if':'every required finding is supported','no_candidate_build_in_phase9':True},'candidate_if_feasible':{'must_be_new_task':True,'default_off_selector':True,'rollback_required':True,'dual_gpu_correctness_dispatch_timing_whole_pp_required':True},'write_boundary':{'registered_source_mutation':False,'registered_build_mutation':False,'database_mutation':False,'candidate_build':False,'gpu_workloads':False,'result_root':str(R)},'tool_hashes':{n:sha(R/'tools'/n) for n in ('make_pair_proof.py','make_findings.py','make_task.py','finalize_feasibility.py')}}
 task={'schema_version':'apex.phase9.gate_up_fusion_feasibility_task.v1','immutable':True,'created_at':datetime.now(UTC).isoformat(),'fingerprint_fields':fields,'target_fingerprint':digest(fields)}
 out.write_text(json.dumps(task,indent=2,sort_keys=True)+'\n');print(json.dumps({'task':str(out),'fingerprint':task['target_fingerprint'],'sha256':sha(out)}))
if __name__=='__main__':main()
