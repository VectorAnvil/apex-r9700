#!/usr/bin/env python3
"""Emit the required Phase 9 feasibility finding contract; no GPU action."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def item(status,*evidence):return {'status':status,'evidence':[x for x in evidence if x]}
def main():
 task=R/'task.json'; proof=R/'pair-proof.json'; source=R/'feasibility.json';out=R/'findings.json'
 if not task.is_file() or not proof.is_file() or not source.is_file():raise RuntimeError('freeze task and pair/source evidence first')
 t=json.loads(task.read_text()); p=json.loads(proof.read_text()); f={'graph_adjacency':item('supported','64 exact adjacent pairs/device/capture state'),'shared_input_layouts':item('supported','same F32 RHS and Q8_1_MMQ layout'),'tensor_parallel_ownership':item('supported','per-device matched source tensors and geometry'),'numerical_swiglu_contract':item('supported','pinned ggml_swiglu_split F32 output/parity semantics; no SuperSonic BF16 rounding rule imposed'),'fusion_api_fit':item('supported','fit determination: existing MMV fusion is N=1 and cannot serve PP N=512 MMQ; a new API boundary is required'),'implementation_boundary':item('supported','new paired Q6_K MMQ+SwiGLU kernel/dispatcher required'),'rollback_default_off_selector':item('supported','exact shape-gated default-off candidate with baseline fallback'),'resource_plan':item('supported','static AMDHSA plus dynamic gfx1201 limits; achieved BW only exact reliable counter'),'correctness_plan':item('supported','CPU oracle preserving current F32 output'),'dispatch_plan':item('supported','both-device exact paired tile/symbol witness'),'timing_plan':item('supported','same-build 3-round interleaved timing, no trimming'),'whole_pp_plan':item('supported','three normal-graph PP processes; joint gate')}
 result={'schema_version':'apex.phase9.findings.v1','task_fingerprint':t['target_fingerprint'],'task_sha256':sha(task),'pair_proof_sha256':sha(proof),'source_feasibility_sha256':sha(source),'phase9_no_code':True,'candidate_status':'not_integrated','modeled_intermediate':{'bytes_per_layer_device':p['modeled_intermediate_upper_bound_bytes_per_layer_device'],'status':'modeled_not_measured'},'quantization_correlation':{'status':'source_proven_independent_calls','exact_trace_placement':'unavailable'},'findings':f}
 out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(out)
if __name__=='__main__':main()
