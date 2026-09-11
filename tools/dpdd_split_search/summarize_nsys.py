#!/usr/bin/env python3
import csv,re,sys
from pathlib import Path
V={"rs5":("RS",5),"old_k5":("K5_OLD",5),"rs_k4":("K4_RS",4),"old_k4":("K4_OLD",4)};R=re.compile(r"computeExchangeFockDPDD(?P<i>\d+)(?P<s>_[A-Z0-9_]+)?\(")
def load(p):
 d={}
 with p.open(newline="") as f:
  for x in csv.DictReader(f):
   m=R.search(x["Name"])
   if m:d[(int(m.group("i")),m.group("s") or "")]=(int(x["Total Time (ns)"]),int(x["Instances"]))
 return d
def total(d,n,s,k):
 a=[d[(i,s)] for i in range(n)]
 if {x[1] for x in a}!={k}:raise ValueError((s,{x[1] for x in a}))
 return sum(x[0]/x[1] for x in a)/1e6
def main():
 p=Path(sys.argv[1]);rows=[]
 for v,(tag,n) in V.items():
  d=load(next(p.glob(f"{v}_cuda_gpu_kern_sum*.csv")));o=total(d,7,"",12);om=total(d,7,"_FP64",8)+total(d,7,"_FP32",8);r=total(d,n,"_"+tag,4);m=total(d,n,"_"+tag+"_FP64",4)+total(d,n,"_"+tag+"_FP32",4);rows.append((v,r,o/r,m,om/m))
 lines=["# DPDD Nsys Kernel Timing","","Times are CUDA execution only per SCF interaction.","","| variant | original (ms) | original speedup | MP (ms) | MP speedup |","|---|---:|---:|---:|---:|"]+[f"| {v} | {r:.3f} | {rs:.4f}x | {m:.3f} | {ms:.4f}x |" for v,r,rs,m,ms in rows];(p/"README.md").write_text("\n".join(lines)+"\n");print(p/"README.md")
if __name__=="__main__":main()
