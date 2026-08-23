#!/usr/bin/env python3
"""Host-only proof of the Phase 7 PP gate/up pairing contract."""
import hashlib,json
from collections import defaultdict
from pathlib import Path
R=Path(__file__).resolve().parents[1]; P=R.parent/'results_phase07_pp_mmq_attribution_20260808'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def name(x):return x['src0']['name'] if isinstance(x['src0'],dict) else str(x['src0'])
def main():
 out=R/'pair-proof.json'
 if out.exists():raise RuntimeError('pair proof immutable')
 rec=defaultdict(lambda:defaultdict(list))
 for line in (P/'attribution-run-v1/mmq-attribution.jsonl').read_text().splitlines():
  x=json.loads(line);rec[str(x['device'])][x['capture_state']].append(x)
 pairs={}
 for d in ('0','1'):
  pairs[d]={}
  for state in ('none','active'):
   a=sorted(rec[d][state],key=lambda x:x['seq']); found=[]
   for i,x in enumerate(a[:-1]):
    if '.ffn_gate.weight' not in name(x):continue
    y=a[i+1]
    if name(y)!=name(x).replace('.ffn_gate.weight','.ffn_up.weight'):raise RuntimeError(f'nonadjacent pair d{d} {state} {i}')
    fields=('logical_m','logical_n','logical_k','rhs_quantized_layout','grid','block','need_check')
    if any(x[k]!=y[k] for k in fields) or x['src1']!=y['src1']:raise RuntimeError(f'incompatible pair d{d} {state} {i}')
    found.append({'layer':name(x).split('.')[1],'seq_gate':x['seq'],'seq_up':y['seq'],'shape':[x[k] for k in ('logical_m','logical_n','logical_k')],'layout':x['rhs_quantized_layout'],'grid':x['grid'],'block':x['block'],'rhs':x['src1']})
   if len(found)!=64:raise RuntimeError(f'expected 64 pairs d{d} {state}, got {len(found)}')
   pairs[d][state]=found
 # Phase7 sequence proof already established trace repeats; validate its formal facts.
 s=json.loads((P/'attribution-summary-v1.json').read_text())
 if s['repeat_count_per_device']!=6 or s['family_counts']!={'need_check_false':4800,'need_check_true':1152}:raise RuntimeError('Phase7 repeat/family proof changed')
 ops={x['operation_class']:x for x in s['operation_groups_dual_device']}
 gate,up=ops['ffn_gate'],ops['ffn_up']
 if gate['calls']!=768 or up['calls']!=768:raise RuntimeError('expected 384 trace calls/device for gate and up')
 result={'schema_version':'apex.phase9.gate_up_pair_proof.v1','phase7_jsonl_sha256':sha(P/'attribution-run-v1/mmq-attribution.jsonl'),'phase7_summary_sha256':sha(P/'attribution-summary-v1.json'),'pairs':pairs,'trace_repeat_count_per_device':6,'trace_calls_per_device_each_operation':384,'combined_duration_ns':gate['duration_ns']+up['duration_ns'],'combined_pp_share':(gate['duration_ns']+up['duration_ns'])/s['pp_duration_total_ns'],'modeled_intermediate_upper_bound_bytes_per_layer_device':71303168,'modeled_note':'two F32 projection stores plus two later reads; modeled from exact 8704x512x4 metadata, not measured bandwidth'}
 out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(out)
if __name__=='__main__':main()
