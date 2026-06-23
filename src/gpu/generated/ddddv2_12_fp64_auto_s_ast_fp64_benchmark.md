# DDDD Variant Benchmark Snippets

Baseline for reconstruction: `h_mat_J2_v2_2kernels`

### DDDDv2_12_FP64_auto_s_ast_fp64

Device buffers:
```cpp
double* d_mat_J3_ddddv2_12_fp64_old = nullptr;
double* d_mat_J3_ddddv2_12_fp64_auto_s_ast_fp64 = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_12_fp64_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_12_fp64_auto_s_ast_fp64, dd_prim_pair_count_local * sizeof(double)));
```

Host buffers:
```cpp
std::vector<double> h_mat_J3_ddddv2_12_fp64_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_12_fp64_auto_s_ast_fp64(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_12_fp64_auto_s_ast_fp64_mix(dd_prim_pair_count_local, 0.0);
```

Old-kernel timing block:
```cpp
gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_12_fp64_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    const auto start = std::chrono::steady_clock::now();
    gpu::computeCoulombFockDDDDv2_12_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_12_fp64_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    gpuSafe(gpuStreamSynchronize(stream));
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_12_FP64 baseline",
        std::chrono::duration<double, std::milli>(end - start).count());
}
```

New-kernel timing block:
```cpp
gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_12_fp64_auto_s_ast_fp64, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    const auto start = std::chrono::steady_clock::now();
    gpu::computeCoulombFockDDDDv2_12_FP64_auto_s_ast_fp64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_12_fp64_auto_s_ast_fp64, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    gpuSafe(gpuStreamSynchronize(stream));
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_12_FP64_auto_s_ast_fp64",
        std::chrono::duration<double, std::milli>(end - start).count());
}
```

Memcpy + reconstruction:
```cpp
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_12_fp64_old.data(), d_mat_J3_ddddv2_12_fp64_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_12_fp64_auto_s_ast_fp64.data(), d_mat_J3_ddddv2_12_fp64_auto_s_ast_fp64,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_12_fp64_auto_s_ast_fp64_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_12_fp64_auto_s_ast_fp64_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_12_fp64_old[idx] + h_mat_J3_ddddv2_12_fp64_auto_s_ast_fp64[idx];
}
```

Checks:
```cpp
check_J_against_ref("computeCoulombFockDDDDv2_12_FP64_auto_s_ast_fp64 contribution", h_mat_J3_ddddv2_12_fp64_auto_s_ast_fp64, h_mat_J3_ddddv2_12_fp64_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_12_FP64_auto_s_ast_fp64 mixed result (vs ref)", h_mat_J3_ddddv2_12_fp64_auto_s_ast_fp64_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_12_FP64_auto_s_ast_fp64 mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_12_fp64_auto_s_ast_fp64_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);
```

