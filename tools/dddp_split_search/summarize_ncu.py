#!/usr/bin/env python3
import csv,sys
from pathlib import Path
METRICS=("smsp__inst_executed.sum","smsp__inst_issued.sum","smsp__sass_thread_inst_executed_op_fp32_pred_on.sum","smsp__sass_thread_inst_executed_op_fadd_pred_on.sum","smsp__sass_thread_inst_executed_op_fmul_pred_on.sum","smsp__sass_thread_inst_executed_op_ffma_pred_on.sum")
GROUPS=((0,),(1,2),(3,),(4,5),(6,))
def load(p):
 with p.open(newline="") as f:return {r["Metric Name"]:int(r["Metric Value"].replace(",","")) for r in csv.DictReader(f)}
def main():
 p=Path(sys.argv[1]);old=[load(p/f"guanine-8-hf.dddp{i}_fp32.instructions.csv") for i in range(7)];new=[load(p/f"guanine-8-hf.dddp{i}_k5_old_fp32.instructions.csv") for i in range(5)]
 lines=["# DDDP Old7 vs Old-derived K5 NCU","","| K5 | old source(s) | old executed | K5 executed | old/K5 |","|---:|---:|---:|---:|---:|"]
 for i,g in enumerate(GROUPS):
  a=sum(old[j][METRICS[0]] for j in g);b=new[i][METRICS[0]];lines.append(f"| {i} | {'+'.join(map(str,g))} | {a} | {b} | {a/b:.4f}x |")
 lines += ["","| metric | old7 | K5 | old/K5 | reduction |","|---|---:|---:|---:|---:|"]
 for m in METRICS:
  a=sum(x[m] for x in old);b=sum(x[m] for x in new);lines.append(f"| `{m}` | {a} | {b} | {a/b:.4f}x | {100*(a-b)/a:.2f}% |")
 (p/"README.md").write_text("\n".join(lines)+"\n");print(p/"README.md")
if __name__=="__main__":main()
