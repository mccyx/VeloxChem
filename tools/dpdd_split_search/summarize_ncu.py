#!/usr/bin/env python3
import csv,sys
from pathlib import Path
M=("smsp__inst_executed.sum","smsp__inst_issued.sum","smsp__sass_thread_inst_executed_op_fp32_pred_on.sum","smsp__sass_thread_inst_executed_op_fadd_pred_on.sum","smsp__sass_thread_inst_executed_op_fmul_pred_on.sum","smsp__sass_thread_inst_executed_op_ffma_pred_on.sum")
def load(p):
 with p.open(newline="") as f:return {r["Metric Name"]:int(r["Metric Value"].replace(",","")) for r in csv.DictReader(f)}
def main():
 p=Path(sys.argv[1]);old=[load(p/f"guanine-8-hf.dpdd{i}_fp32.instructions.csv") for i in range(7)];new=[load(p/f"guanine-8-hf.dpdd{i}_k4_rs_fp32.instructions.csv") for i in range(4)];lines=["# DPDD Old7 vs RS-derived K4 NCU","","The layouts use different expression groupings, so only complete FP32 bundles are compared.","","| metric | old7 | RS K4 | old/K4 | reduction |","|---|---:|---:|---:|---:|"]
 for m in M:
  a=sum(x[m] for x in old);b=sum(x[m] for x in new);lines.append(f"| `{m}` | {a} | {b} | {a/b:.4f}x | {100*(a-b)/a:.2f}% |")
 (p/"README.md").write_text("\n".join(lines)+"\n");print(p/"README.md")
if __name__=="__main__":main()
