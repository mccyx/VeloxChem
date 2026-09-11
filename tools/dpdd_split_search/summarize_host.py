#!/usr/bin/env python3
import csv,re,sys
from pathlib import Path
from statistics import mean,stdev
V=("rs5","old_k5","rs_k4","old_k4")
T=re.compile(r"=== DPDD (?P<v>\S+) exchange resplit timing ===\s+old original = (?P<or>[0-9.]+) ms\s+RS original  = (?P<sr>[0-9.]+) ms\s+old MP       = (?P<om>[0-9.]+) ms\s+RS MP        = (?P<sm>[0-9.]+) ms")
E=re.compile(r"=== DPDD (?P<l>RS original vs old original|RS MP vs RS original) ===\s+max \|dJ\|\s+= (?P<a>[0-9.eE+-]+).*?rel error\s+= (?P<r>[0-9.eE+-]+)",re.S)
def main():
 p=Path(sys.argv[1]);rows=[]
 for v in V:
  ts=[];es={"RS original vs old original":[],"RS MP vs RS original":[]}
  for f in sorted(p.glob(f"{v}_run*_ablation.log")):
   x=f.read_text();ms=[m for m in T.finditer(x) if m.group("v")==v]
   if len(ms)!=4:raise ValueError((f,len(ms)))
   ts += [{k:float(m.group(k)) for k in ("or","sr","om","sm")} for m in ms]
   allm=list(E.finditer(x))
   for l in es:
    q=[m for m in allm if m.group("l")==l]
    if len(q)!=4:raise ValueError((f,l,len(q)))
    es[l] += [(float(m.group("a")),float(m.group("r"))) for m in q]
  orr=mean(x["or"] for x in ts);sr=mean(x["sr"] for x in ts);om=mean(x["om"] for x in ts);sms=[x["sm"] for x in ts];sm=mean(sms)
  rows.append((v,sr,orr/sr,sm,om/sm,stdev(sms),max(x[0] for x in es["RS original vs old original"]),max(x[0] for x in es["RS MP vs RS original"]),max(x[1] for x in es["RS MP vs RS original"])))
 with (p/"summary.csv").open("w",newline="") as f:csv.writer(f).writerows([("variant","original_ms","original_speedup","mp_ms","mp_speedup","mp_std","original_max_abs","mp_max_abs","mp_max_rel")]+rows)
 lines=["# DPDD Split Host Benchmark","","Three runs and four SCF interactions per run. Timer includes selected launches, GPU execution, and final synchronization; excludes allocation, zeroing, cuts, copies, and validation.","","| variant | original (ms) | original speedup | MP (ms) | MP speedup | MP std | original max abs | MP max abs | MP max rel |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]+[f"| {v} | {sr:.3f} | {rs:.4f}x | {sm:.3f} | {ms:.4f}x | {sd:.3f} | {oa:.6e} | {ma:.6e} | {mr:.6e} |" for v,sr,rs,sm,ms,sd,oa,ma,mr in rows]
 (p/"README.md").write_text("\n".join(lines)+"\n");print(p/"README.md")
if __name__=="__main__":main()
