#!/usr/bin/env python3
import csv,re,sys
from pathlib import Path
VARIANTS={"rs6":("RS",6),"rs_k5":("K5_RS",5),"old_k5":("K5_OLD",5)}
RX=re.compile(r"computeExchangeFockDDDP(?P<i>\d+)(?P<s>_[A-Z0-9_]+)?\(")
def load(path):
 d={}
 with path.open(newline="") as f:
  for r in csv.DictReader(f):
   m=RX.search(r["Name"])
   if m:d[(int(m.group("i")),m.group("s") or "")]=(int(r["Total Time (ns)"]),int(r["Instances"]))
 return d
def total(d,n,s,instances):
 x=[d[(i,s)] for i in range(n)]
 if {v[1] for v in x}!={instances}:raise ValueError((s,{v[1] for v in x}))
 return sum(v[0]/v[1] for v in x)/1e6
def main():
 p=Path(sys.argv[1]); rows=[]
 for variant,(tag,n) in VARIANTS.items():
  d=load(next(p.glob(f"{variant}_cuda_gpu_kern_sum*.csv"))); oldr=total(d,7,"",12); oldm=total(d,7,"_FP64",8)+total(d,7,"_FP32",8); sr=total(d,n,"_"+tag,4); sm=total(d,n,"_"+tag+"_FP64",4)+total(d,n,"_"+tag+"_FP32",4); rows.append((variant,sr,oldr/sr,sm,oldm/sm))
 lines=["# DDDP K5 Nsys Kernel Timing","","Times are CUDA execution only, normalized to one SCF interaction.","","| variant | original (ms) | original speedup | MP (ms) | MP speedup |","|---|---:|---:|---:|---:|"]+[f"| {v} | {r:.3f} | {rs:.4f}x | {m:.3f} | {ms:.4f}x |" for v,r,rs,m,ms in rows]
 (p/"README.md").write_text("\n".join(lines)+"\n");print(p/"README.md")
if __name__=="__main__":main()
