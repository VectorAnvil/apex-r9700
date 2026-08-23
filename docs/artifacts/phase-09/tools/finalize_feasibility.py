#!/usr/bin/env python3
"""Finalize Phase 9 feasibility only; never create or run a candidate."""
import argparse,hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--findings',type=Path,required=True);p.add_argument('--output',type=Path,default=R/'feasibility-result.json');a=p.parse_args()
 if a.output.exists():raise RuntimeError('feasibility result is immutable')
 task=json.loads((R/'task.json').read_text());findings=json.loads(a.findings.read_text())
 if findings.get('task_fingerprint')!=task['target_fingerprint']:raise RuntimeError('findings task fingerprint mismatch')
 if not task['fingerprint_fields']['phase9_semantics']['no_code_changes'] or findings.get('phase9_no_code') is not True:raise RuntimeError('Phase 9 must remain source-inspection only')
 missing=[];unsupported=[]
 for name in task['fingerprint_fields']['required_findings']:
  item=findings.get('findings',{}).get(name)
  if not isinstance(item,dict) or not item.get('evidence'):missing.append(name)
  elif item.get('status')!='supported':unsupported.append(name)
 decision='feasible_candidate_contract' if not missing and not unsupported else 'reject_feasibility'
 result={'schema_version':'apex.phase9.feasibility_result.v1','task_fingerprint':task['target_fingerprint'],'decision':decision,'missing_findings':missing,'unsupported_findings':unsupported,'candidate_build':False,'gpu_workloads':False,'integration':False,'failure_writeup_required':decision=='reject_feasibility','findings_sha256':sha(a.findings)}
 a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
