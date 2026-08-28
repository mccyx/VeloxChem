# PET 电子密度初猜与 VeloxChem：阶段学习笔记

更新日期：2026-08-28  
开发分支：`ml-initial-density-pet`  
目标：判断预训练 PET 电子密度模型能否作为 VeloxChem 的 SCF 初始猜测，减少
PBE/def2-SVP 计算的端到端时间，并找到有意义的 AI-infrastructure 优化问题。

## 1. 到目前为止做了什么

当前完成了以下工作：

1. 建立独立开发分支，并把计划写入 `ML_INITIAL_DENSITY_PLAN.md`。
2. 将大文件、Python 环境、checkpoint 和计算结果放到 scratch，避免占满 home。
3. 下载 Atomistic Cookbook 的 PET density recipe、参考 RI coefficients 和预训练
   checkpoint，并记录 SHA-256。
4. 在 Dardel 的 ARM/NVIDIA 环境中建立独立 Python 3.11 + CUDA AI 环境。
5. 在真实 NVIDIA GH200 120GB 计算节点上加载模型并完成 CUDA inference。
6. 复现上游的 PySCF PBE/def2-SVP 结果：
   - SAD 初猜：12 个 SCF cycles；
   - reference RI 初猜：8 个 cycles；
   - PET 初猜：9 个 cycles。
7. 验证三种初猜最终收敛到同一个能量，最大差约 `2e-11 Hartree`。
8. 将 SAD、reference RI、PET 三个初始 density matrices 和 overlap matrix 导出为
   NPZ，以便下一阶段分析 PySCF/VeloxChem AO convention。
9. 分离 PET 的冷启动和稳态 inference：第一次 forward 约 10.94 s，预热后的
   median forward 约 15.73 ms。
10. 识别出当前最重要的 AI-infra 问题是首次 TorchScript/NVRTC 编译和模型生命周期，
    而不是 16 ms 的稳态模型 kernel。

主要结果目录：

```text
/cfs/klemming/scratch/y/yuch4126/dev/ml_initial_density/
├── checkpoints/
│   └── pet-density.pt
├── datasets/
│   ├── ml-density.zip
│   └── ml-density/
├── env/
│   └── pet-density-gh200-py311/
└── benchmark_results/
    ├── pet_pyscf_reproduction_gh200_24007798/
    └── pet_gh200_benchmark_24007953.json
```

## 2. 要解决的科学与性能问题

Hartree–Fock 或 Kohn–Sham DFT 都要自洽求解电子结构。给定 density matrix
`D`，程序构造 Fock/Kohn–Sham matrix：

```text
HF:   F[D] = h + J[D] - K[D]
PBE:  F[D] = h + J[D] + V_xc[D]
```

这里：

- `h` 是动能与核吸引的一电子部分；
- `J` 是 Coulomb 部分；
- `K` 是 HF exchange；
- `V_xc` 是 DFT exchange-correlation potential。

将 `F` 对角化得到新的 orbitals，再由 occupied orbitals 生成新的 `D`。重复此过程，
直到能量、density change 和 orbital gradient 达到阈值。

初始 `D_0` 越接近最终自洽 density，通常所需的 Fock builds 和对角化次数越少。
PET 在这里不替代 ERI、J、K 或 XC kernel；它只试图提供更好的 `D_0`。

因此正确的性能指标不是“少了几个 iteration”，而是：

```text
T_SAD = T_SAD_guess + T_SCF_from_SAD

T_PET = T_model_load/cold_start
      + T_graph_and_PET_forward
      + T_RI_to_DM
      + T_transfer_and_mapping
      + T_SCF_from_PET
```

只有 `T_PET < T_SAD` 才是真正的端到端加速。

## 3. SAD 是什么

SAD 是 Superposition of Atomic Densities，即“原子密度叠加”。它是常见的 SCF
初始猜测方法。

直观过程是：

1. 分别为每一种孤立原子准备 atomic density；
2. 将各原子的 density 放到分子几何中的相应原子中心；
3. 将它们组合成分子初始 density；
4. 这个初猜在最初阶段基本不知道化学键和分子中的电荷重排。

所以 SAD 通常稳定、便宜、适用范围广，但它没有学到 bonding、polarization 和具体分子
环境。在本例中，SAD 需要 12 次 SCF，而 PET 需要 9 次。

PySCF 的 projected SAD 并不严格保持精确电子数。本例有 38 个电子，但：

```text
Tr(D_SAD S) = 37.9629685
```

这不是计算错误，上游 recipe 也得到约 37.963。它只是初猜，后续占据轨道构造会恢复
正确电子数。

VeloxChem 的 SAD 实现还有一个重要细节：它先在 `AO-START-GUESS` minimal basis 中
生成原子初猜，再通过 mixed-basis overlap 投影到实际 AO basis。这个 minimal basis
不是最终 def2-SVP 轨道 basis，也不是 PET 的 auxiliary basis。

## 4. PBE 是什么，以及为什么当前必须用 PBE

PBE 是 Perdew–Burke–Ernzerhof generalized-gradient approximation（GGA）DFT
functional。它是 pure DFT functional，不含 HF exact exchange。

PET checkpoint 的训练数据来自 SCFBench 中的：

```text
method:           PBE
orbital basis:    def2-SVP
auxiliary basis:  def2-universal-jfit
```

因此第一阶段必须保持同样的 functional 和 basis，避免把“模型错误”和“方法/basis
不匹配”混在一起。

你之前的 ERI/exchange mixed-precision 工作多数使用 HF/def2-SVP。这里虽然也用了
def2-SVP，但物理方法不同：

- HF 有 `J-K`，exchange 需要四中心 ERI；
- pure PBE 有 `J+V_xc`，没有 HF exact-exchange `K`；
- PET checkpoint 学的是 PBE ground-state density，不是 HF density。

所以当前 PET 工作与 ERI kernel 优化不是同一条直接调用链。未来若 PET initial guess
接入 VeloxChem，它可能减少 SCF 的 Fock-build 次数，从而间接减少 ERI/J/XC 总工作；
但它没有取代任何 ERI kernel。

## 5. 三种 basis 的区别

这是整个项目里最容易混淆的部分。

### 5.1 def2-SVP：orbital/AO basis

def2-SVP 是实际 Kohn–Sham orbitals 使用的 Gaussian orbital basis。分子轨道写作：

```text
psi_i(r) = sum_mu C_mu,i phi_mu(r)
```

其中 `phi_mu` 是 def2-SVP AO basis functions。density matrix `D_mu,nu` 也定义在这个
AO basis 中：

```text
rho(r) = sum_mu,nu D_mu,nu phi_mu(r) phi_nu(r)
```

你当前 VeloxChem ERI 实验也常用 def2-SVP。因此“basis set 名字”相同，但 PET 当前
目标是 PBE density 初猜，而 ERI 实验通常关注 HF exchange kernel。

即使两边都写 `def2-SVP`，PySCF 和 VeloxChem 仍可能在以下方面不同：

- AO 排列顺序；
- shell/contracted function 排列；
- spherical harmonic 的 `m` 顺序；
- phase/sign convention；
- normalization；
- Cartesian 与 spherical representation。

所以不能直接把 PySCF 的 NumPy density matrix 丢进 VeloxChem，必须先验证 overlap
matrix 并建立 AO transformation。

### 5.2 def2-universal-jfit：auxiliary density basis

PET 不直接输出 def2-SVP 的方阵 `D_mu,nu`。它输出 auxiliary basis 中的 density
expansion coefficients：

```text
rho_tilde(r) = sum_P c_P chi_P(r)
```

`chi_P` 是 `def2-universal-jfit` auxiliary Gaussian functions，`c_P` 是 PET 预测值。

这个 basis 的用途是表达电子密度，而不是表达 molecular orbitals。它与 def2-SVP 的
区别是：

- def2-SVP 的 `phi_mu` 用于 orbitals 和 AO density matrix；
- def2-universal-jfit 的 `chi_P` 用于紧凑地拟合/表达 density；
- auxiliary coefficient vector 长度约为 `N_aux`；
- AO density matrix 大小是 `N_AO x N_AO`。

因此 PET 预测 coefficient vector 通常比直接预测整个 density matrix 更紧凑，也更适合
按原子局域环境学习。

名字中有 `jfit`，但这个 recipe 使用的是 overlap-metric/S-fit density representation。
这里的 auxiliary representation 是模型 target；不要把它和 VeloxChem 中用于 ERI/J
加速的某个具体 density-fitting implementation 自动视为同一件事。

### 5.3 AO-START-GUESS minimal basis

这是 VeloxChem SAD 使用的起始猜测 basis。它用于生成廉价 atomic densities，再投影到
最终 AO basis。

三者关系可以概括为：

```text
AO-START-GUESS minimal basis
    -> VeloxChem SAD 初猜的临时 basis

def2-SVP
    -> 最终 orbitals、Fock matrix、AO density matrix 的 basis

def2-universal-jfit
    -> PET 输出的辅助 density expansion basis
```

## 6. RI 在这里到底做什么

RI 是 Resolution of Identity。这里更准确地说，是把真实 density 用 auxiliary basis
展开。

模型并不预测：

```text
D_mu,nu
```

而是预测：

```text
c_P
```

优点是：

- target 更紧凑；
- auxiliary functions 是 atom-centered，便于局域分解；
- coefficient 可以按 atom、angular momentum、radial channel 和 magnetic component
  组织；
- 模型可以在不同大小的分子间迁移。

checkpoint 输出的是 metatensor `TensorMap`，按 `(lambda, sigma, Z)` 等 key 分 block。
当前 7-atom 测试分子的输出有：

```text
13 blocks
267 values
```

## 7. PET 到底做了什么

PET 是 Point Edge Transformer，是一种 atomistic graph neural network/transformer。

输入主要是：

- atomic numbers/species；
- atomic positions；
- 由 cutoff 和 neighbor-list 建立的局域 atomic graph。

模型通过 point/edge message passing 或 attention，让每个原子的表示包含邻域几何和
化学环境信息。最终输出每个原子、每个 angular/radial channel 对应的 auxiliary density
coefficients。

当前 checkpoint 已在约 45k 个 SCFBench 小有机分子上训练完成。我们没有训练模型，
只做 pretrained inference。

PET 不做以下事情：

- 不直接运行 SCF；
- 不直接输出 converged molecular orbitals；
- 不替代 J/K/XC；
- 不计算四中心 ERI；
- 不保证任意 functional 或 basis 都适用；
- 不直接输出 VeloxChem convention 的 density matrix。

它做的是：

```text
atoms + geometry
    -> graph/neighbor list
    -> PET forward
    -> auxiliary RI coefficients c_P
```

## 8. RI coefficients 如何变成可用于 SCF 的 density matrix

这是一个非常重要的中间步骤。PET coefficients 不是 SCF driver 直接需要的 AO density
matrix。

上游 `dm_from_ri_coefficients` 做了以下工作：

1. 将 metatensor blocks 按 PySCF auxiliary AO ordering 排列；
2. 注意 PySCF p orbitals 使用 `(z, x, y)` ordering；
3. 在 DFT quadrature grid 上计算：

   ```text
   rho_tilde(r) = sum_P c_P chi_P(r)
   ```

4. 由预测 density 计算 `V_xc[rho_tilde]`；
5. 用三中心 integrals 和 coefficients 计算 Coulomb `J[rho_tilde]`；
6. 构造一次非自洽 Kohn–Sham matrix：

   ```text
   F_tilde = h + J[rho_tilde] + V_xc[rho_tilde]
   ```

7. 解 generalized eigenproblem：

   ```text
   F C = S C epsilon
   ```

8. 占据最低的 orbitals，生成一个 idempotent AO density matrix：

   ```text
   D_0 = 2 C_occ C_occ^T
   ```

这个 `D_0` 才传给 PySCF SCF。换句话说，PET 并不是简单地把 RI density reshape 成
方阵；中间还包含 J、XC、一次 diagonalization 和 orbital occupation。

本次测试的电子数：

```text
Tr(D_RI  S) = 38.0000000000
Tr(D_PET S) = 38.0000000000
```

## 9. PySCF 与 VeloxChem density convention

PySCF restricted density 通常是 spin-summed：

```text
Tr(D_PySCF S) = N_electrons
```

本例为 38。

VeloxChem 现有 restricted tests 和 driver 中常使用 per-spin density：

```text
Tr(D_VLX S) = N_electrons / 2
```

所以闭壳层系统很可能需要：

```text
D_VLX = 0.5 * U-related-transform(D_PySCF)
```

但是这个 `0.5` 不能凭经验直接硬编码。必须同时解决 AO ordering transformation `U`：

```text
S_VLX approximately equals U^T S_PySCF U
```

只有确认 `U` 是 permutation/phase matrix，或推导出一般 basis transformation 后，才能
写正确的 density transformation。转换后必须用 electron-count assertion 验证。

## 10. 科学复现结果

Job `24007798`：

| Initial guess | SCF cycles | Converged energy / Ha | `Tr(DS)` |
|---|---:|---:|---:|
| SAD | 12 | -302.5314352046449 | 37.9629685289 |
| Reference RI | 8 | -302.5314352046353 | 38.0000000000 |
| PET | 9 | -302.5314352046283 | 38.0000000000 |

结论：

- PET 的 density 在科学上有效；
- PET 将 iteration 数从 12 降到 9，即减少 25%；
- PET 接近当前 auxiliary basis 可达到的 reference RI 上限（8 cycles）；
- 三个计算收敛到同一个 electronic state/energy；
- 这证明值得继续研究接口与端到端成本，但不等于已经加速。

## 11. 当前 timing 与正确解释

Job `24007798` 的单次 timing：

```text
SAD guess                 0.277 s
SAD-started SCF           1.703 s

reference RI load         0.049 s
reference RI -> DM        1.110 s
reference-RI SCF          1.197 s

model load                0.299 s
PET cold forward          7.800 s
PET -> DM                 0.441 s
PET-started SCF           8.249 s
```

这一轮 PET-started SCF 的 8.249 s 明显与 9 cycles 不相称，也和 reference RI 的
1.197 s 不一致。这可能包含：

- 单次测量噪声；
- 运行顺序效应；
- CPU thread/OpenBLAS 状态变化；
- CUDA/PyTorch 之后的同步或资源状态；
- PySCF grid/cache warmness；
- 72 CPU threads 对小体系产生的不稳定 oversubscription/调度成本。

因此不能用这一次 8.249 s 判断 PET-started SCF 本质更慢。需要随机/交错顺序、预热、
多次重复和 median。

## 12. PET cold start 与 steady state

Job `24007953` 在同一个 process、同一个 model/calculator 上完成：

```text
model/calculator load     1.522 s
first CUDA forward       10.941 s
warm steady median        0.01573 s
warm steady mean          0.01694 s
warm steady range         0.01524--0.02171 s
```

cold forward 比 steady median 慢约 695 倍。

这说明：

- PET 神经网络稳态 inference 对这个 7-atom 分子并不昂贵；
- 当前最大开销是首次 CUDA/TorchScript/NVRTC 初始化和编译；
- 对单次独立分子计算，cold start 会完全吃掉 iteration reduction；
- 对 geometry optimization、MD trajectory 或批量分子，persistent model 可以摊销
  cold start；
- AI-infra 的第一目标应是 model persistence、kernel cache、AOT/prewarm、batching；
- 在此之前优化 16 ms steady kernels 的收益有限。

随后 jobs `24008018` 和 `24008057` 用固定 density、轮换 SCF 执行顺序和 5 次重复，
验证了稳态端到端时间。中位数如下：

```text
                         8 threads                 1 thread
SAD guess                0.048 s                   0.018 s
SAD-started SCF          2.929 s / 12 cycles       4.821 s / 12 cycles
RI -> DM                 0.352 s                   0.579 s
RI-started SCF           2.110 s / 8 cycles        3.490 s / 8 cycles
PET steady forward       0.017 s                   0.019 s
PET -> DM                0.361 s                   0.585 s
PET-started SCF          2.321 s / 9 cycles        3.813 s / 9 cycles
```

因此，不含一次性 model load/compile 的 warm end-to-end 中位数为：

```text
8 threads: SAD 2.977 s, PET 2.700 s，PET 快约 9.3%
1 thread:  SAD 4.839 s, PET 4.417 s，PET 快约 8.7%
```

原先单次测到的 PET-started SCF `8.249 s` 没有复现；它是初始化、线程或运行顺序造成
的异常值，不能代表 PET density 的 SCF 性能。两种线程设置都稳定复现 12/8/9 cycles，
说明 PET 的小幅稳态收益来自减少三个 SCF iterations，而不是偶然的 cache 效应。

还有一个重要细节：第一次 forward 为 7.4--8.9 s，紧接着第二次仍为 2.5--2.7 s，
第三次起才稳定为 17--19 ms。这表明 CUDA/TorchScript/NVRTC 存在多阶段 lazy
initialization/compilation；可靠 benchmark 至少需要两次不计时的 warm-up。按 8-thread
结果粗略估算，相对每步约 0.277 s 的稳态收益，单次约 10 s 的加载和两阶段预热需要约
36 个同类 geometry steps 才能摊销。这个 break-even 只适用于当前 7-atom 测试分子。

## 13. Dardel 环境与遇到的问题

### 13.1 默认 login 与 GH 节点架构不同

默认环境最初只有：

```text
Python 3.6.15
x86_64 login environment
```

GH 节点使用 ARM/aarch64。x86 virtual environment 和 binary wheels 不能直接拿到
aarch64 节点运行。

解决：

- 通过 `ssh logingh` 进入 ARM 环境；
- 加载 `cray-python/3.11.7`；
- 在 scratch 单独创建 `pet-density-gh200-py311`；
- VeloxChem 原环境与 AI 环境完全分离。

### 13.2 PyTorch wheel 一开始只有不合适的 CUDA architecture

最初安装 `torch 2.10.0 + cu126`。它可以 import，但警告只包含较老的预编译 GPU
architectures，不能作为可靠的当前 GPU 环境。

随后测试 `torch 2.11 + cu130`，但 `metatomic-torch 0.1.11` 要求 `torch<2.11`。

最终兼容组合为：

```text
Python             3.11.7
torch              2.10.0+cu130
metatrain           2026.2
metatensor-torch    0.8.5
metatomic-torch     0.1.11
PySCF               2.14.0
ASE                 3.29.0
```

在计算节点上实际识别到：

```text
NVIDIA GH200 120GB
CUDA available: true
```

并通过 tensor CUDA execution probe 和真实 PET forward。

### 13.3 `py3Dmol` 缺失

我们只想使用 `rho_utils.py` 中的 RI-to-DM 函数，但上游文件在顶层无条件 import
`py3Dmol`。第一次完整 reproduction 因此在 import 阶段失败。

解决：将轻量 `py3Dmol` 加入 CPU/GH requirements，并先验证：

```text
import rho_utils: OK
```

这不是模型或科学问题，只是上游 helper 把可视化依赖和数值函数放在同一模块。

### 13.4 SAD electron-count assertion 过严

第二次 reproduction 的三组 SCF 都完成，但后处理要求所有初猜都满足：

```text
abs(Tr(DS) - N) < 1e-8
```

PySCF projected SAD 本来就是 37.963 而不是 38，因此断言错误。

解决：

- SAD 容差设为 0.1 electron；
- RI/PET 仍要求 `1e-8`；
- 最终能量仍严格比较。

### 13.5 第二次 forward 缺少 NVRTC builtins library path

单次 PET forward 成功，但重复 inference 的第二次 forward 触发 TorchScript fused
kernel/NVRTC compilation，报错：

```text
failed to open libnvrtc-builtins.so.13.0
```

库文件实际存在于：

```text
.../site-packages/nvidia/cu13/lib/libnvrtc-builtins.so.13.0
```

只是没有进入动态链接器搜索路径。

解决：所有 GH job 加入：

```bash
export LD_LIBRARY_PATH=".../site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
```

修复后 2 次 warm-up 和 20 次 timing 全部成功。

### 13.6 Slurm controller、login 和 reservation

Codex 执行环境一度无法连接 Slurm controller；`logingh` 看到的是普通 Dardel Slurm
配置，不能直接解析 `gpugh`。同时 Codex 会话没有用户 SSH credential forwarding，
无法替用户 SSH 到 `login1` 提交。

因此 job 由用户从正常 `login1` 提交，结果由共享 scratch 读取。

Summer-school reservation `sum28-gpugh` 只在 2026-08-28 08:00--12:00 有效。
到期后 job 脚本改为：

```text
partition: gpugh
account:   pdc-software-test-gh
```

CPU 小任务使用：

```text
partition: shared
account:   pdc-software-test
```

## 14. 重要 job 记录

```text
24001957  GH compute-node PET smoke：成功
24004558  缺少 py3Dmol：import 失败
24006888  三组 SCF 成功；SAD electron assertion 过严
24007798  完整科学复现成功并保存 JSON/NPZ
24007909  第二次 forward 缺少 NVRTC builtins path
24007953  cold + warm 20-repeat inference benchmark 成功
24008018  8-thread、5-repeat 交错 end-to-end benchmark 成功
24008057  1-thread、5-repeat 对照 benchmark 成功
```

## 15. 当前代码和数据

仓库工具：

```text
tools/ml_initial_density/
├── README.md
├── bootstrap_cpu_env.sh
├── bootstrap_gh200_env.sh
├── requirements-reproduce.txt
├── requirements-gh200.txt
├── reproduce_pet_pyscf.py
├── benchmark_initial_density_end_to_end.py
├── smoke_pet_inference.py
├── submit_reproduce_pet_cpu.sbatch
├── submit_pet_gh200_smoke.sbatch
├── submit_reproduce_pet_gh200.sbatch
├── submit_pet_gh200_inference_benchmark.sbatch
└── submit_pet_repeated_timing_gh200.sbatch
```

Artifact hashes：

```text
ml-density.zip
09e1e2c0f13f6bd1d63d4377f375a60101500c9d67faa883d2fe57f74f3bbe40

pet-density.pt
f27bf90619797e15749107fd98fe6b9f62415ec10c63d326d0db22f4c13a1246

SCFBench reference RI TensorMap
db55bd755a5da702296715e485f3ed1b70bc92cf02ef0a47bb9a1012deca03a8
```

## 16. 接下来应该先做什么

笔记和重复 end-to-end timing 现在都已完成。下一阶段是验证并实现 VeloxChem density
handoff，而不是继续只在 PySCF 中增加 timing 样本。

推荐顺序：

1. 检查 VeloxChem 当前 SCF driver 接收 initial density 的内部入口；
2. 比较 PySCF/VeloxChem overlap matrices、AO ordering、phase 和 spin convention；
3. 实现或暴露 Python-level `initial_density` API；
4. 完成当前小分子的 PET density handoff 和能量/cycle correctness test；
5. 将 density conversion 与模型生命周期封装成可复用接口；
6. 再做 compile cache、persistent service、batching 等 AI-infra 优化。

为什么不立即接 VeloxChem：

- 科学有效性已经证明；
- 稳态端到端收益已经可靠测得，当前小体系约为 9%；
- PySCF/VeloxChem density convention 和 AO mapping 是独立 correctness 风险；
- 先有稳定 benchmark，后面接口与优化的收益才可量化。

## 17. 当前阶段结论

PET 不是“用 AI 替代昂贵 ERI kernel”，而是用 AI 提供更接近自洽解的 density 初猜，
以减少昂贵 kernel 被调用的次数。

当前已经确认：

- pretrained PET 可以直接使用，不需要自己训练；
- 在匹配的 PBE/def2-SVP domain 中，PET 将 SCF cycles 从 12 降到 9；
- 最终能量与 SAD/reference RI 一致；
- 模型稳态 forward 约 16--19 ms；
- 最大 AI-infra 问题是约 7--11 s 的 cold compilation，以及约 2.5--2.7 s 的第二阶段
  warm-up；
- 重复 timing 已确认 warm PET 在当前小体系上相对 SAD 快约 9%；
- 接入 VeloxChem 前必须解决 AO mapping 和 restricted-density factor；
- 最有希望的实际场景不是单个极小分子的独立计算，而是连续 geometry steps、批量
  molecules 或更大体系。

## 18. VeloxChem density handoff 验证

在当前分支完成 release build 后，两套程序对相同几何和 def2-SVP 得到的 AO 数都是
80。源码和 overlap matrix 共同确认：

- PySCF 按 atom/shell/component 排列 AO；
- VeloxChem 按 angular momentum/component/atom/radial shell 排列；
- VeloxChem p 分量顺序为 `(py, pz, px)`；
- d 分量顺序为 `(dxy, dyz, dz2, dxz, dx2-y2)`，与 PySCF 标签顺序一致；
- 此体系不需要额外 phase sign correction。

完整 permutation 作用后：

```text
max |S_VLX - P S_PySCF P^T| = 3.476e-11
RMS overlap difference       = 5.043e-12
```

VeloxChem restricted density 存单自旋块，而 PySCF RKS density 是 spin-summed，因此：

```text
D_VLX = 0.5 * P D_PySCF P^T
```

变换后的电子数为：

```text
PET          Tr(D_VLX S_VLX) = 18.999999999921
reference RI                 = 18.999999999921
PySCF SAD                    = 18.981484264437
```

`ScfDriver.compute()` 已增加 `initial_density` 参数，当前刻意只支持 restricted DIIS，
并验证矩阵 shape、finite values、symmetry 和 contiguous layout。首次 native VeloxChem
PBE/def2-SVP correctness run 得到：

```text
SAD            15 iterations   -302.53142842941986 Eh
PET            13 iterations   -302.53142842942225 Eh
reference RI   11 iterations   -302.53142842940260 Eh
energy spread                   1.97e-11 Eh
```

这证明 PET density 已经真正进入 VeloxChem SCF，并减少 2 次 iterations。该轮 SAD/PET/RI
按固定顺序运行，时间分别为 4.37/0.73/0.63 s；SAD 包含进程首次 GPU、grid 和 library
初始化，因此这些时间不能用于声称 speedup。下一实验应在同一进程交错重复三种固定
density 的 VeloxChem SCF timing。

交错重复实验随后完成。每种初猜先 warm 一次，再轮换执行 5 次，结果为：

```text
SAD            median 0.8272 s   15 iterations
PET            median 0.7089 s   13 iterations
reference RI   median 0.6199 s   11 iterations
```

5 次 iteration 数和能量逐次完全稳定。只计算 native VeloxChem SCF 部分，PET 相对 SAD
快约 14.3%。但 PET density 不是免费得到的。使用当前已测的 steady PET forward 和
PySCF RI-to-DM bridge：

```text
current PET end-to-end ~= 0.017 + 0.361 + 0.709 = 1.087 s
native VeloxChem SAD   ~= 0.827 s
```

因此当前 7-atom 小体系中，warm PET end-to-end 仍比 VeloxChem SAD 慢约 31%，更不用说
cold compilation。PySCF 中观察到的约 9% warm speedup 不能直接外推到更快的 GPU
VeloxChem SCF。

这个结果重新定义了 AI-infra 优化目标：17 ms 模型 forward 已不是主要问题，约 361 ms
的 RI-to-DM bridge 才是。如果能在 VeloxChem 内直接用 RI density 构造一次 Fock/DM、
把 grid/J/Vxc 工作放到 GPU，或者避免 PySCF object/grid 重建，PET 才更可能在小体系上
达到 break-even。另一个自然方向是测试更大体系，因为此时减少 Fock iterations 的收益
增长可能快于 bridge overhead。
