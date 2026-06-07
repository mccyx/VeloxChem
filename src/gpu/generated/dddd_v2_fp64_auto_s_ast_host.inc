// DDDDv2_0_FP64_auto_s_ast_fp64
double* d_mat_J3_ddddv2_0_fp64_old = nullptr;
double* d_mat_J3_ddddv2_0_fp64_auto_s_ast_fp64 = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_0_fp64_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_0_fp64_auto_s_ast_fp64, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_0_fp64_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_0_fp64_auto_s_ast_fp64(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_0_fp64_auto_s_ast_fp64_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_0_fp64_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_0_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_0_fp64_old, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_0_FP64 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_0_fp64_auto_s_ast_fp64, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_0_FP64_auto_s_ast_fp64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_0_fp64_auto_s_ast_fp64, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_0_FP64_auto_s_ast_fp64",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_0_fp64_old.data(), d_mat_J3_ddddv2_0_fp64_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_0_fp64_auto_s_ast_fp64.data(), d_mat_J3_ddddv2_0_fp64_auto_s_ast_fp64,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_0_fp64_auto_s_ast_fp64_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_0_fp64_auto_s_ast_fp64_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_0_fp64_old[idx] + h_mat_J3_ddddv2_0_fp64_auto_s_ast_fp64[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_0_FP64_auto_s_ast_fp64 contribution", h_mat_J3_ddddv2_0_fp64_auto_s_ast_fp64, h_mat_J3_ddddv2_0_fp64_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_0_FP64_auto_s_ast_fp64 mixed result (vs ref)", h_mat_J3_ddddv2_0_fp64_auto_s_ast_fp64_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_0_FP64_auto_s_ast_fp64 mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_0_fp64_auto_s_ast_fp64_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_0_fp64_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_0_fp64_auto_s_ast_fp64));

// DDDDv2_1_FP64_auto_s_ast_fp64
double* d_mat_J3_ddddv2_1_fp64_old = nullptr;
double* d_mat_J3_ddddv2_1_fp64_auto_s_ast_fp64 = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_1_fp64_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_1_fp64_auto_s_ast_fp64, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_1_fp64_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_1_fp64_auto_s_ast_fp64(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_1_fp64_auto_s_ast_fp64_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_1_fp64_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_1_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_1_fp64_old, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_1_FP64 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_1_fp64_auto_s_ast_fp64, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_1_FP64_auto_s_ast_fp64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_1_fp64_auto_s_ast_fp64, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_1_FP64_auto_s_ast_fp64",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_1_fp64_old.data(), d_mat_J3_ddddv2_1_fp64_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_1_fp64_auto_s_ast_fp64.data(), d_mat_J3_ddddv2_1_fp64_auto_s_ast_fp64,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_1_fp64_auto_s_ast_fp64_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_1_fp64_auto_s_ast_fp64_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_1_fp64_old[idx] + h_mat_J3_ddddv2_1_fp64_auto_s_ast_fp64[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_1_FP64_auto_s_ast_fp64 contribution", h_mat_J3_ddddv2_1_fp64_auto_s_ast_fp64, h_mat_J3_ddddv2_1_fp64_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_1_FP64_auto_s_ast_fp64 mixed result (vs ref)", h_mat_J3_ddddv2_1_fp64_auto_s_ast_fp64_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_1_FP64_auto_s_ast_fp64 mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_1_fp64_auto_s_ast_fp64_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_1_fp64_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_1_fp64_auto_s_ast_fp64));

// DDDDv2_2_FP64_auto_s_ast_fp64
double* d_mat_J3_ddddv2_2_fp64_old = nullptr;
double* d_mat_J3_ddddv2_2_fp64_auto_s_ast_fp64 = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_2_fp64_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_2_fp64_auto_s_ast_fp64, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_2_fp64_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_2_fp64_auto_s_ast_fp64(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_2_fp64_auto_s_ast_fp64_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_2_fp64_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_2_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_2_fp64_old, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_2_FP64 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_2_fp64_auto_s_ast_fp64, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_2_FP64_auto_s_ast_fp64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_2_fp64_auto_s_ast_fp64, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_2_FP64_auto_s_ast_fp64",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_2_fp64_old.data(), d_mat_J3_ddddv2_2_fp64_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_2_fp64_auto_s_ast_fp64.data(), d_mat_J3_ddddv2_2_fp64_auto_s_ast_fp64,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_2_fp64_auto_s_ast_fp64_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_2_fp64_auto_s_ast_fp64_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_2_fp64_old[idx] + h_mat_J3_ddddv2_2_fp64_auto_s_ast_fp64[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_2_FP64_auto_s_ast_fp64 contribution", h_mat_J3_ddddv2_2_fp64_auto_s_ast_fp64, h_mat_J3_ddddv2_2_fp64_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_2_FP64_auto_s_ast_fp64 mixed result (vs ref)", h_mat_J3_ddddv2_2_fp64_auto_s_ast_fp64_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_2_FP64_auto_s_ast_fp64 mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_2_fp64_auto_s_ast_fp64_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_2_fp64_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_2_fp64_auto_s_ast_fp64));

// DDDDv2_3_FP64_auto_s_ast_fp64
double* d_mat_J3_ddddv2_3_fp64_old = nullptr;
double* d_mat_J3_ddddv2_3_fp64_auto_s_ast_fp64 = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_3_fp64_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_3_fp64_auto_s_ast_fp64, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_3_fp64_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_3_fp64_auto_s_ast_fp64(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_3_fp64_auto_s_ast_fp64_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_3_fp64_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_3_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_3_fp64_old, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_3_FP64 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_3_fp64_auto_s_ast_fp64, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_3_FP64_auto_s_ast_fp64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_3_fp64_auto_s_ast_fp64, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_3_FP64_auto_s_ast_fp64",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_3_fp64_old.data(), d_mat_J3_ddddv2_3_fp64_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_3_fp64_auto_s_ast_fp64.data(), d_mat_J3_ddddv2_3_fp64_auto_s_ast_fp64,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_3_fp64_auto_s_ast_fp64_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_3_fp64_auto_s_ast_fp64_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_3_fp64_old[idx] + h_mat_J3_ddddv2_3_fp64_auto_s_ast_fp64[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_3_FP64_auto_s_ast_fp64 contribution", h_mat_J3_ddddv2_3_fp64_auto_s_ast_fp64, h_mat_J3_ddddv2_3_fp64_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_3_FP64_auto_s_ast_fp64 mixed result (vs ref)", h_mat_J3_ddddv2_3_fp64_auto_s_ast_fp64_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_3_FP64_auto_s_ast_fp64 mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_3_fp64_auto_s_ast_fp64_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_3_fp64_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_3_fp64_auto_s_ast_fp64));

// DDDDv2_4_FP64_auto_s_ast_fp64
double* d_mat_J3_ddddv2_4_fp64_old = nullptr;
double* d_mat_J3_ddddv2_4_fp64_auto_s_ast_fp64 = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_4_fp64_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_4_fp64_auto_s_ast_fp64, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_4_fp64_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_4_fp64_auto_s_ast_fp64(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_4_fp64_auto_s_ast_fp64_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_4_fp64_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_4_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_4_fp64_old, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_4_FP64 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_4_fp64_auto_s_ast_fp64, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_4_FP64_auto_s_ast_fp64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_4_fp64_auto_s_ast_fp64, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_4_FP64_auto_s_ast_fp64",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_4_fp64_old.data(), d_mat_J3_ddddv2_4_fp64_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_4_fp64_auto_s_ast_fp64.data(), d_mat_J3_ddddv2_4_fp64_auto_s_ast_fp64,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_4_fp64_auto_s_ast_fp64_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_4_fp64_auto_s_ast_fp64_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_4_fp64_old[idx] + h_mat_J3_ddddv2_4_fp64_auto_s_ast_fp64[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_4_FP64_auto_s_ast_fp64 contribution", h_mat_J3_ddddv2_4_fp64_auto_s_ast_fp64, h_mat_J3_ddddv2_4_fp64_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_4_FP64_auto_s_ast_fp64 mixed result (vs ref)", h_mat_J3_ddddv2_4_fp64_auto_s_ast_fp64_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_4_FP64_auto_s_ast_fp64 mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_4_fp64_auto_s_ast_fp64_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_4_fp64_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_4_fp64_auto_s_ast_fp64));

// DDDDv2_5_FP64_auto_s_ast_fp64
double* d_mat_J3_ddddv2_5_fp64_old = nullptr;
double* d_mat_J3_ddddv2_5_fp64_auto_s_ast_fp64 = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_5_fp64_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_5_fp64_auto_s_ast_fp64, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_5_fp64_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_5_fp64_auto_s_ast_fp64(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_5_fp64_auto_s_ast_fp64_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_5_fp64_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_5_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_5_fp64_old, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_5_FP64 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_5_fp64_auto_s_ast_fp64, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_5_FP64_auto_s_ast_fp64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_5_fp64_auto_s_ast_fp64, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_5_FP64_auto_s_ast_fp64",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_5_fp64_old.data(), d_mat_J3_ddddv2_5_fp64_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_5_fp64_auto_s_ast_fp64.data(), d_mat_J3_ddddv2_5_fp64_auto_s_ast_fp64,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_5_fp64_auto_s_ast_fp64_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_5_fp64_auto_s_ast_fp64_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_5_fp64_old[idx] + h_mat_J3_ddddv2_5_fp64_auto_s_ast_fp64[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_5_FP64_auto_s_ast_fp64 contribution", h_mat_J3_ddddv2_5_fp64_auto_s_ast_fp64, h_mat_J3_ddddv2_5_fp64_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_5_FP64_auto_s_ast_fp64 mixed result (vs ref)", h_mat_J3_ddddv2_5_fp64_auto_s_ast_fp64_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_5_FP64_auto_s_ast_fp64 mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_5_fp64_auto_s_ast_fp64_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_5_fp64_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_5_fp64_auto_s_ast_fp64));

// DDDDv2_6_FP64_auto_s_ast_fp64
double* d_mat_J3_ddddv2_6_fp64_old = nullptr;
double* d_mat_J3_ddddv2_6_fp64_auto_s_ast_fp64 = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_6_fp64_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_6_fp64_auto_s_ast_fp64, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_6_fp64_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_6_fp64_auto_s_ast_fp64(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_6_fp64_auto_s_ast_fp64_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_6_fp64_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_6_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_6_fp64_old, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_6_FP64 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_6_fp64_auto_s_ast_fp64, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_6_FP64_auto_s_ast_fp64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_6_fp64_auto_s_ast_fp64, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_6_FP64_auto_s_ast_fp64",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_6_fp64_old.data(), d_mat_J3_ddddv2_6_fp64_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_6_fp64_auto_s_ast_fp64.data(), d_mat_J3_ddddv2_6_fp64_auto_s_ast_fp64,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_6_fp64_auto_s_ast_fp64_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_6_fp64_auto_s_ast_fp64_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_6_fp64_old[idx] + h_mat_J3_ddddv2_6_fp64_auto_s_ast_fp64[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_6_FP64_auto_s_ast_fp64 contribution", h_mat_J3_ddddv2_6_fp64_auto_s_ast_fp64, h_mat_J3_ddddv2_6_fp64_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_6_FP64_auto_s_ast_fp64 mixed result (vs ref)", h_mat_J3_ddddv2_6_fp64_auto_s_ast_fp64_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_6_FP64_auto_s_ast_fp64 mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_6_fp64_auto_s_ast_fp64_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_6_fp64_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_6_fp64_auto_s_ast_fp64));

// DDDDv2_7_FP64_auto_s_ast_fp64
double* d_mat_J3_ddddv2_7_fp64_old = nullptr;
double* d_mat_J3_ddddv2_7_fp64_auto_s_ast_fp64 = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_7_fp64_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_7_fp64_auto_s_ast_fp64, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_7_fp64_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_7_fp64_auto_s_ast_fp64(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_7_fp64_auto_s_ast_fp64_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_7_fp64_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_7_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_7_fp64_old, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_7_FP64 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_7_fp64_auto_s_ast_fp64, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_7_FP64_auto_s_ast_fp64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_7_fp64_auto_s_ast_fp64, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_7_FP64_auto_s_ast_fp64",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_7_fp64_old.data(), d_mat_J3_ddddv2_7_fp64_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_7_fp64_auto_s_ast_fp64.data(), d_mat_J3_ddddv2_7_fp64_auto_s_ast_fp64,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_7_fp64_auto_s_ast_fp64_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_7_fp64_auto_s_ast_fp64_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_7_fp64_old[idx] + h_mat_J3_ddddv2_7_fp64_auto_s_ast_fp64[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_7_FP64_auto_s_ast_fp64 contribution", h_mat_J3_ddddv2_7_fp64_auto_s_ast_fp64, h_mat_J3_ddddv2_7_fp64_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_7_FP64_auto_s_ast_fp64 mixed result (vs ref)", h_mat_J3_ddddv2_7_fp64_auto_s_ast_fp64_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_7_FP64_auto_s_ast_fp64 mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_7_fp64_auto_s_ast_fp64_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_7_fp64_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_7_fp64_auto_s_ast_fp64));

// DDDDv2_8_FP64_auto_s_ast_fp64
double* d_mat_J3_ddddv2_8_fp64_old = nullptr;
double* d_mat_J3_ddddv2_8_fp64_auto_s_ast_fp64 = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_8_fp64_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_8_fp64_auto_s_ast_fp64, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_8_fp64_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_8_fp64_auto_s_ast_fp64(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_8_fp64_auto_s_ast_fp64_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_8_fp64_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_8_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_8_fp64_old, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_8_FP64 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_8_fp64_auto_s_ast_fp64, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_8_FP64_auto_s_ast_fp64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_8_fp64_auto_s_ast_fp64, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_8_FP64_auto_s_ast_fp64",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_8_fp64_old.data(), d_mat_J3_ddddv2_8_fp64_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_8_fp64_auto_s_ast_fp64.data(), d_mat_J3_ddddv2_8_fp64_auto_s_ast_fp64,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_8_fp64_auto_s_ast_fp64_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_8_fp64_auto_s_ast_fp64_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_8_fp64_old[idx] + h_mat_J3_ddddv2_8_fp64_auto_s_ast_fp64[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_8_FP64_auto_s_ast_fp64 contribution", h_mat_J3_ddddv2_8_fp64_auto_s_ast_fp64, h_mat_J3_ddddv2_8_fp64_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_8_FP64_auto_s_ast_fp64 mixed result (vs ref)", h_mat_J3_ddddv2_8_fp64_auto_s_ast_fp64_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_8_FP64_auto_s_ast_fp64 mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_8_fp64_auto_s_ast_fp64_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_8_fp64_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_8_fp64_auto_s_ast_fp64));

// DDDDv2_9_FP64_auto_s_ast_fp64
double* d_mat_J3_ddddv2_9_fp64_old = nullptr;
double* d_mat_J3_ddddv2_9_fp64_auto_s_ast_fp64 = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_9_fp64_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_9_fp64_auto_s_ast_fp64, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_9_fp64_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_9_fp64_auto_s_ast_fp64(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_9_fp64_auto_s_ast_fp64_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_9_fp64_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_9_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_9_fp64_old, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_9_FP64 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_9_fp64_auto_s_ast_fp64, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_9_FP64_auto_s_ast_fp64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_9_fp64_auto_s_ast_fp64, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_9_FP64_auto_s_ast_fp64",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_9_fp64_old.data(), d_mat_J3_ddddv2_9_fp64_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_9_fp64_auto_s_ast_fp64.data(), d_mat_J3_ddddv2_9_fp64_auto_s_ast_fp64,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_9_fp64_auto_s_ast_fp64_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_9_fp64_auto_s_ast_fp64_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_9_fp64_old[idx] + h_mat_J3_ddddv2_9_fp64_auto_s_ast_fp64[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_9_FP64_auto_s_ast_fp64 contribution", h_mat_J3_ddddv2_9_fp64_auto_s_ast_fp64, h_mat_J3_ddddv2_9_fp64_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_9_FP64_auto_s_ast_fp64 mixed result (vs ref)", h_mat_J3_ddddv2_9_fp64_auto_s_ast_fp64_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_9_FP64_auto_s_ast_fp64 mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_9_fp64_auto_s_ast_fp64_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_9_fp64_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_9_fp64_auto_s_ast_fp64));

// DDDDv2_10_FP64_auto_s_ast_fp64
double* d_mat_J3_ddddv2_10_fp64_old = nullptr;
double* d_mat_J3_ddddv2_10_fp64_auto_s_ast_fp64 = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_10_fp64_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_10_fp64_auto_s_ast_fp64, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_10_fp64_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_10_fp64_auto_s_ast_fp64(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_10_fp64_auto_s_ast_fp64_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_10_fp64_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_10_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_10_fp64_old, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_10_FP64 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_10_fp64_auto_s_ast_fp64, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_10_FP64_auto_s_ast_fp64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_10_fp64_auto_s_ast_fp64, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_10_FP64_auto_s_ast_fp64",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_10_fp64_old.data(), d_mat_J3_ddddv2_10_fp64_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_10_fp64_auto_s_ast_fp64.data(), d_mat_J3_ddddv2_10_fp64_auto_s_ast_fp64,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_10_fp64_auto_s_ast_fp64_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_10_fp64_auto_s_ast_fp64_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_10_fp64_old[idx] + h_mat_J3_ddddv2_10_fp64_auto_s_ast_fp64[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_10_FP64_auto_s_ast_fp64 contribution", h_mat_J3_ddddv2_10_fp64_auto_s_ast_fp64, h_mat_J3_ddddv2_10_fp64_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_10_FP64_auto_s_ast_fp64 mixed result (vs ref)", h_mat_J3_ddddv2_10_fp64_auto_s_ast_fp64_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_10_FP64_auto_s_ast_fp64 mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_10_fp64_auto_s_ast_fp64_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_10_fp64_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_10_fp64_auto_s_ast_fp64));

// DDDDv2_11_FP64_auto_s_ast_fp64
double* d_mat_J3_ddddv2_11_fp64_old = nullptr;
double* d_mat_J3_ddddv2_11_fp64_auto_s_ast_fp64 = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_11_fp64_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_11_fp64_auto_s_ast_fp64, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_11_fp64_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_11_fp64_auto_s_ast_fp64(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_11_fp64_auto_s_ast_fp64_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_11_fp64_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_11_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_11_fp64_old, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_11_FP64 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_11_fp64_auto_s_ast_fp64, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_11_FP64_auto_s_ast_fp64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_11_fp64_auto_s_ast_fp64, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_11_FP64_auto_s_ast_fp64",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_11_fp64_old.data(), d_mat_J3_ddddv2_11_fp64_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_11_fp64_auto_s_ast_fp64.data(), d_mat_J3_ddddv2_11_fp64_auto_s_ast_fp64,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_11_fp64_auto_s_ast_fp64_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_11_fp64_auto_s_ast_fp64_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_11_fp64_old[idx] + h_mat_J3_ddddv2_11_fp64_auto_s_ast_fp64[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_11_FP64_auto_s_ast_fp64 contribution", h_mat_J3_ddddv2_11_fp64_auto_s_ast_fp64, h_mat_J3_ddddv2_11_fp64_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_11_FP64_auto_s_ast_fp64 mixed result (vs ref)", h_mat_J3_ddddv2_11_fp64_auto_s_ast_fp64_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_11_FP64_auto_s_ast_fp64 mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_11_fp64_auto_s_ast_fp64_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_11_fp64_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_11_fp64_auto_s_ast_fp64));

// DDDDv2_12_FP64_auto_s_ast_fp64
double* d_mat_J3_ddddv2_12_fp64_old = nullptr;
double* d_mat_J3_ddddv2_12_fp64_auto_s_ast_fp64 = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_12_fp64_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_12_fp64_auto_s_ast_fp64, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_12_fp64_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_12_fp64_auto_s_ast_fp64(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_12_fp64_auto_s_ast_fp64_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_12_fp64_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_12_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_12_fp64_old, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_12_FP64 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_12_fp64_auto_s_ast_fp64, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_12_FP64_auto_s_ast_fp64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_12_fp64_auto_s_ast_fp64, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_12_FP64_auto_s_ast_fp64",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_12_fp64_old.data(), d_mat_J3_ddddv2_12_fp64_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_12_fp64_auto_s_ast_fp64.data(), d_mat_J3_ddddv2_12_fp64_auto_s_ast_fp64,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_12_fp64_auto_s_ast_fp64_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_12_fp64_auto_s_ast_fp64_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_12_fp64_old[idx] + h_mat_J3_ddddv2_12_fp64_auto_s_ast_fp64[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_12_FP64_auto_s_ast_fp64 contribution", h_mat_J3_ddddv2_12_fp64_auto_s_ast_fp64, h_mat_J3_ddddv2_12_fp64_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_12_FP64_auto_s_ast_fp64 mixed result (vs ref)", h_mat_J3_ddddv2_12_fp64_auto_s_ast_fp64_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_12_FP64_auto_s_ast_fp64 mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_12_fp64_auto_s_ast_fp64_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_12_fp64_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_12_fp64_auto_s_ast_fp64));

// DDDDv2_13_FP64_auto_s_ast_fp64
double* d_mat_J3_ddddv2_13_fp64_old = nullptr;
double* d_mat_J3_ddddv2_13_fp64_auto_s_ast_fp64 = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_13_fp64_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_13_fp64_auto_s_ast_fp64, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_13_fp64_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_13_fp64_auto_s_ast_fp64(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_13_fp64_auto_s_ast_fp64_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_13_fp64_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_13_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_13_fp64_old, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_13_FP64 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_13_fp64_auto_s_ast_fp64, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_13_FP64_auto_s_ast_fp64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_13_fp64_auto_s_ast_fp64, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_13_FP64_auto_s_ast_fp64",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_13_fp64_old.data(), d_mat_J3_ddddv2_13_fp64_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_13_fp64_auto_s_ast_fp64.data(), d_mat_J3_ddddv2_13_fp64_auto_s_ast_fp64,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_13_fp64_auto_s_ast_fp64_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_13_fp64_auto_s_ast_fp64_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_13_fp64_old[idx] + h_mat_J3_ddddv2_13_fp64_auto_s_ast_fp64[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_13_FP64_auto_s_ast_fp64 contribution", h_mat_J3_ddddv2_13_fp64_auto_s_ast_fp64, h_mat_J3_ddddv2_13_fp64_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_13_FP64_auto_s_ast_fp64 mixed result (vs ref)", h_mat_J3_ddddv2_13_fp64_auto_s_ast_fp64_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_13_FP64_auto_s_ast_fp64 mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_13_fp64_auto_s_ast_fp64_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_13_fp64_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_13_fp64_auto_s_ast_fp64));

// DDDDv2_14_FP64_auto_s_ast_fp64
double* d_mat_J3_ddddv2_14_fp64_old = nullptr;
double* d_mat_J3_ddddv2_14_fp64_auto_s_ast_fp64 = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_14_fp64_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_14_fp64_auto_s_ast_fp64, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_14_fp64_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_14_fp64_auto_s_ast_fp64(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_14_fp64_auto_s_ast_fp64_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_14_fp64_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_14_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_14_fp64_old, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_14_FP64 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_14_fp64_auto_s_ast_fp64, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_14_FP64_auto_s_ast_fp64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_14_fp64_auto_s_ast_fp64, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_14_FP64_auto_s_ast_fp64",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_14_fp64_old.data(), d_mat_J3_ddddv2_14_fp64_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_14_fp64_auto_s_ast_fp64.data(), d_mat_J3_ddddv2_14_fp64_auto_s_ast_fp64,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_14_fp64_auto_s_ast_fp64_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_14_fp64_auto_s_ast_fp64_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_14_fp64_old[idx] + h_mat_J3_ddddv2_14_fp64_auto_s_ast_fp64[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_14_FP64_auto_s_ast_fp64 contribution", h_mat_J3_ddddv2_14_fp64_auto_s_ast_fp64, h_mat_J3_ddddv2_14_fp64_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_14_FP64_auto_s_ast_fp64 mixed result (vs ref)", h_mat_J3_ddddv2_14_fp64_auto_s_ast_fp64_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_14_FP64_auto_s_ast_fp64 mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_14_fp64_auto_s_ast_fp64_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_14_fp64_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_14_fp64_auto_s_ast_fp64));

// DDDDv2_15_FP64_auto_s_ast_fp64
double* d_mat_J3_ddddv2_15_fp64_old = nullptr;
double* d_mat_J3_ddddv2_15_fp64_auto_s_ast_fp64 = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_15_fp64_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_15_fp64_auto_s_ast_fp64, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_15_fp64_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_15_fp64_auto_s_ast_fp64(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_15_fp64_auto_s_ast_fp64_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_15_fp64_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_15_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_15_fp64_old, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_15_FP64 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_15_fp64_auto_s_ast_fp64, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_15_FP64_auto_s_ast_fp64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_15_fp64_auto_s_ast_fp64, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_15_FP64_auto_s_ast_fp64",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_15_fp64_old.data(), d_mat_J3_ddddv2_15_fp64_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_15_fp64_auto_s_ast_fp64.data(), d_mat_J3_ddddv2_15_fp64_auto_s_ast_fp64,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_15_fp64_auto_s_ast_fp64_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_15_fp64_auto_s_ast_fp64_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_15_fp64_old[idx] + h_mat_J3_ddddv2_15_fp64_auto_s_ast_fp64[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_15_FP64_auto_s_ast_fp64 contribution", h_mat_J3_ddddv2_15_fp64_auto_s_ast_fp64, h_mat_J3_ddddv2_15_fp64_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_15_FP64_auto_s_ast_fp64 mixed result (vs ref)", h_mat_J3_ddddv2_15_fp64_auto_s_ast_fp64_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_15_FP64_auto_s_ast_fp64 mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_15_fp64_auto_s_ast_fp64_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_15_fp64_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_15_fp64_auto_s_ast_fp64));

// DDDDv2_16_FP64_auto_s_ast_fp64
double* d_mat_J3_ddddv2_16_fp64_old = nullptr;
double* d_mat_J3_ddddv2_16_fp64_auto_s_ast_fp64 = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_16_fp64_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_16_fp64_auto_s_ast_fp64, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_16_fp64_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_16_fp64_auto_s_ast_fp64(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_16_fp64_auto_s_ast_fp64_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_16_fp64_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_16_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_16_fp64_old, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_16_FP64 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_16_fp64_auto_s_ast_fp64, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_16_FP64_auto_s_ast_fp64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_16_fp64_auto_s_ast_fp64, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_16_FP64_auto_s_ast_fp64",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_16_fp64_old.data(), d_mat_J3_ddddv2_16_fp64_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_16_fp64_auto_s_ast_fp64.data(), d_mat_J3_ddddv2_16_fp64_auto_s_ast_fp64,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_16_fp64_auto_s_ast_fp64_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_16_fp64_auto_s_ast_fp64_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_16_fp64_old[idx] + h_mat_J3_ddddv2_16_fp64_auto_s_ast_fp64[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_16_FP64_auto_s_ast_fp64 contribution", h_mat_J3_ddddv2_16_fp64_auto_s_ast_fp64, h_mat_J3_ddddv2_16_fp64_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_16_FP64_auto_s_ast_fp64 mixed result (vs ref)", h_mat_J3_ddddv2_16_fp64_auto_s_ast_fp64_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_16_FP64_auto_s_ast_fp64 mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_16_fp64_auto_s_ast_fp64_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_16_fp64_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_16_fp64_auto_s_ast_fp64));

// DDDDv2_17_FP64_auto_s_ast_fp64
double* d_mat_J3_ddddv2_17_fp64_old = nullptr;
double* d_mat_J3_ddddv2_17_fp64_auto_s_ast_fp64 = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_17_fp64_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_17_fp64_auto_s_ast_fp64, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_17_fp64_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_17_fp64_auto_s_ast_fp64(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_17_fp64_auto_s_ast_fp64_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_17_fp64_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_17_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_17_fp64_old, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_17_FP64 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_17_fp64_auto_s_ast_fp64, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_17_FP64_auto_s_ast_fp64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_17_fp64_auto_s_ast_fp64, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_17_FP64_auto_s_ast_fp64",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_17_fp64_old.data(), d_mat_J3_ddddv2_17_fp64_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_17_fp64_auto_s_ast_fp64.data(), d_mat_J3_ddddv2_17_fp64_auto_s_ast_fp64,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_17_fp64_auto_s_ast_fp64_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_17_fp64_auto_s_ast_fp64_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_17_fp64_old[idx] + h_mat_J3_ddddv2_17_fp64_auto_s_ast_fp64[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_17_FP64_auto_s_ast_fp64 contribution", h_mat_J3_ddddv2_17_fp64_auto_s_ast_fp64, h_mat_J3_ddddv2_17_fp64_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_17_FP64_auto_s_ast_fp64 mixed result (vs ref)", h_mat_J3_ddddv2_17_fp64_auto_s_ast_fp64_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_17_FP64_auto_s_ast_fp64 mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_17_fp64_auto_s_ast_fp64_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_17_fp64_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_17_fp64_auto_s_ast_fp64));

// DDDDv2_18_FP64_auto_s_ast_fp64
double* d_mat_J3_ddddv2_18_fp64_old = nullptr;
double* d_mat_J3_ddddv2_18_fp64_auto_s_ast_fp64 = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_18_fp64_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_18_fp64_auto_s_ast_fp64, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_18_fp64_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_18_fp64_auto_s_ast_fp64(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_18_fp64_auto_s_ast_fp64_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_18_fp64_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_18_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_18_fp64_old, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_18_FP64 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_18_fp64_auto_s_ast_fp64, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_18_FP64_auto_s_ast_fp64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_18_fp64_auto_s_ast_fp64, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_18_FP64_auto_s_ast_fp64",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_18_fp64_old.data(), d_mat_J3_ddddv2_18_fp64_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_18_fp64_auto_s_ast_fp64.data(), d_mat_J3_ddddv2_18_fp64_auto_s_ast_fp64,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_18_fp64_auto_s_ast_fp64_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_18_fp64_auto_s_ast_fp64_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_18_fp64_old[idx] + h_mat_J3_ddddv2_18_fp64_auto_s_ast_fp64[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_18_FP64_auto_s_ast_fp64 contribution", h_mat_J3_ddddv2_18_fp64_auto_s_ast_fp64, h_mat_J3_ddddv2_18_fp64_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_18_FP64_auto_s_ast_fp64 mixed result (vs ref)", h_mat_J3_ddddv2_18_fp64_auto_s_ast_fp64_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_18_FP64_auto_s_ast_fp64 mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_18_fp64_auto_s_ast_fp64_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_18_fp64_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_18_fp64_auto_s_ast_fp64));

// DDDDv2_19_FP64_auto_s_ast_fp64
double* d_mat_J3_ddddv2_19_fp64_old = nullptr;
double* d_mat_J3_ddddv2_19_fp64_auto_s_ast_fp64 = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_19_fp64_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_19_fp64_auto_s_ast_fp64, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_19_fp64_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_19_fp64_auto_s_ast_fp64(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_19_fp64_auto_s_ast_fp64_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_19_fp64_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_19_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_19_fp64_old, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_19_FP64 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_19_fp64_auto_s_ast_fp64, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_19_FP64_auto_s_ast_fp64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_19_fp64_auto_s_ast_fp64, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_19_FP64_auto_s_ast_fp64",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_19_fp64_old.data(), d_mat_J3_ddddv2_19_fp64_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_19_fp64_auto_s_ast_fp64.data(), d_mat_J3_ddddv2_19_fp64_auto_s_ast_fp64,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_19_fp64_auto_s_ast_fp64_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_19_fp64_auto_s_ast_fp64_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_19_fp64_old[idx] + h_mat_J3_ddddv2_19_fp64_auto_s_ast_fp64[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_19_FP64_auto_s_ast_fp64 contribution", h_mat_J3_ddddv2_19_fp64_auto_s_ast_fp64, h_mat_J3_ddddv2_19_fp64_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_19_FP64_auto_s_ast_fp64 mixed result (vs ref)", h_mat_J3_ddddv2_19_fp64_auto_s_ast_fp64_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_19_FP64_auto_s_ast_fp64 mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_19_fp64_auto_s_ast_fp64_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_19_fp64_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_19_fp64_auto_s_ast_fp64));

// DDDDv2_20_FP64_auto_s_ast_fp64
double* d_mat_J3_ddddv2_20_fp64_old = nullptr;
double* d_mat_J3_ddddv2_20_fp64_auto_s_ast_fp64 = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_20_fp64_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_20_fp64_auto_s_ast_fp64, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_20_fp64_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_20_fp64_auto_s_ast_fp64(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_20_fp64_auto_s_ast_fp64_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_20_fp64_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_20_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_20_fp64_old, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_20_FP64 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_20_fp64_auto_s_ast_fp64, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_20_FP64_auto_s_ast_fp64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_20_fp64_auto_s_ast_fp64, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_20_FP64_auto_s_ast_fp64",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_20_fp64_old.data(), d_mat_J3_ddddv2_20_fp64_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_20_fp64_auto_s_ast_fp64.data(), d_mat_J3_ddddv2_20_fp64_auto_s_ast_fp64,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_20_fp64_auto_s_ast_fp64_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_20_fp64_auto_s_ast_fp64_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_20_fp64_old[idx] + h_mat_J3_ddddv2_20_fp64_auto_s_ast_fp64[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_20_FP64_auto_s_ast_fp64 contribution", h_mat_J3_ddddv2_20_fp64_auto_s_ast_fp64, h_mat_J3_ddddv2_20_fp64_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_20_FP64_auto_s_ast_fp64 mixed result (vs ref)", h_mat_J3_ddddv2_20_fp64_auto_s_ast_fp64_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_20_FP64_auto_s_ast_fp64 mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_20_fp64_auto_s_ast_fp64_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_20_fp64_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_20_fp64_auto_s_ast_fp64));

// DDDDv2_21_FP64_auto_s_ast_fp64
double* d_mat_J3_ddddv2_21_fp64_old = nullptr;
double* d_mat_J3_ddddv2_21_fp64_auto_s_ast_fp64 = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_21_fp64_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_21_fp64_auto_s_ast_fp64, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_21_fp64_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_21_fp64_auto_s_ast_fp64(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_21_fp64_auto_s_ast_fp64_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_21_fp64_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_21_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_21_fp64_old, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_21_FP64 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_21_fp64_auto_s_ast_fp64, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_21_FP64_auto_s_ast_fp64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_21_fp64_auto_s_ast_fp64, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_21_FP64_auto_s_ast_fp64",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_21_fp64_old.data(), d_mat_J3_ddddv2_21_fp64_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_21_fp64_auto_s_ast_fp64.data(), d_mat_J3_ddddv2_21_fp64_auto_s_ast_fp64,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_21_fp64_auto_s_ast_fp64_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_21_fp64_auto_s_ast_fp64_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_21_fp64_old[idx] + h_mat_J3_ddddv2_21_fp64_auto_s_ast_fp64[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_21_FP64_auto_s_ast_fp64 contribution", h_mat_J3_ddddv2_21_fp64_auto_s_ast_fp64, h_mat_J3_ddddv2_21_fp64_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_21_FP64_auto_s_ast_fp64 mixed result (vs ref)", h_mat_J3_ddddv2_21_fp64_auto_s_ast_fp64_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_21_FP64_auto_s_ast_fp64 mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_21_fp64_auto_s_ast_fp64_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_21_fp64_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_21_fp64_auto_s_ast_fp64));

// DDDDv2_22_FP64_auto_s_ast_fp64
double* d_mat_J3_ddddv2_22_fp64_old = nullptr;
double* d_mat_J3_ddddv2_22_fp64_auto_s_ast_fp64 = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_22_fp64_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_22_fp64_auto_s_ast_fp64, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_22_fp64_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_22_fp64_auto_s_ast_fp64(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_22_fp64_auto_s_ast_fp64_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_22_fp64_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_22_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_22_fp64_old, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_22_FP64 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_22_fp64_auto_s_ast_fp64, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_22_FP64_auto_s_ast_fp64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_22_fp64_auto_s_ast_fp64, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_22_FP64_auto_s_ast_fp64",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_22_fp64_old.data(), d_mat_J3_ddddv2_22_fp64_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_22_fp64_auto_s_ast_fp64.data(), d_mat_J3_ddddv2_22_fp64_auto_s_ast_fp64,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_22_fp64_auto_s_ast_fp64_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_22_fp64_auto_s_ast_fp64_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_22_fp64_old[idx] + h_mat_J3_ddddv2_22_fp64_auto_s_ast_fp64[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_22_FP64_auto_s_ast_fp64 contribution", h_mat_J3_ddddv2_22_fp64_auto_s_ast_fp64, h_mat_J3_ddddv2_22_fp64_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_22_FP64_auto_s_ast_fp64 mixed result (vs ref)", h_mat_J3_ddddv2_22_fp64_auto_s_ast_fp64_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_22_FP64_auto_s_ast_fp64 mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_22_fp64_auto_s_ast_fp64_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_22_fp64_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_22_fp64_auto_s_ast_fp64));

// DDDDv2_23_FP64_auto_s_ast_fp64
double* d_mat_J3_ddddv2_23_fp64_old = nullptr;
double* d_mat_J3_ddddv2_23_fp64_auto_s_ast_fp64 = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_23_fp64_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_23_fp64_auto_s_ast_fp64, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_23_fp64_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_23_fp64_auto_s_ast_fp64(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_23_fp64_auto_s_ast_fp64_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_23_fp64_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_23_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_23_fp64_old, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_23_FP64 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_23_fp64_auto_s_ast_fp64, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_23_FP64_auto_s_ast_fp64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_23_fp64_auto_s_ast_fp64, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_23_FP64_auto_s_ast_fp64",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_23_fp64_old.data(), d_mat_J3_ddddv2_23_fp64_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_23_fp64_auto_s_ast_fp64.data(), d_mat_J3_ddddv2_23_fp64_auto_s_ast_fp64,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_23_fp64_auto_s_ast_fp64_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_23_fp64_auto_s_ast_fp64_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_23_fp64_old[idx] + h_mat_J3_ddddv2_23_fp64_auto_s_ast_fp64[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_23_FP64_auto_s_ast_fp64 contribution", h_mat_J3_ddddv2_23_fp64_auto_s_ast_fp64, h_mat_J3_ddddv2_23_fp64_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_23_FP64_auto_s_ast_fp64 mixed result (vs ref)", h_mat_J3_ddddv2_23_fp64_auto_s_ast_fp64_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_23_FP64_auto_s_ast_fp64 mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_23_fp64_auto_s_ast_fp64_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_23_fp64_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_23_fp64_auto_s_ast_fp64));

// DDDDv2_24_FP64_auto_s_ast_fp64
double* d_mat_J3_ddddv2_24_fp64_old = nullptr;
double* d_mat_J3_ddddv2_24_fp64_auto_s_ast_fp64 = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_24_fp64_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_24_fp64_auto_s_ast_fp64, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_24_fp64_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_24_fp64_auto_s_ast_fp64(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_24_fp64_auto_s_ast_fp64_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_24_fp64_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_24_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_24_fp64_old, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_24_FP64 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_24_fp64_auto_s_ast_fp64, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_24_FP64_auto_s_ast_fp64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_24_fp64_auto_s_ast_fp64, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_24_FP64_auto_s_ast_fp64",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_24_fp64_old.data(), d_mat_J3_ddddv2_24_fp64_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_24_fp64_auto_s_ast_fp64.data(), d_mat_J3_ddddv2_24_fp64_auto_s_ast_fp64,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_24_fp64_auto_s_ast_fp64_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_24_fp64_auto_s_ast_fp64_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_24_fp64_old[idx] + h_mat_J3_ddddv2_24_fp64_auto_s_ast_fp64[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_24_FP64_auto_s_ast_fp64 contribution", h_mat_J3_ddddv2_24_fp64_auto_s_ast_fp64, h_mat_J3_ddddv2_24_fp64_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_24_FP64_auto_s_ast_fp64 mixed result (vs ref)", h_mat_J3_ddddv2_24_fp64_auto_s_ast_fp64_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_24_FP64_auto_s_ast_fp64 mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_24_fp64_auto_s_ast_fp64_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_24_fp64_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_24_fp64_auto_s_ast_fp64));

// DDDDv2_25_FP64_auto_s_ast_fp64
double* d_mat_J3_ddddv2_25_fp64_old = nullptr;
double* d_mat_J3_ddddv2_25_fp64_auto_s_ast_fp64 = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_25_fp64_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_25_fp64_auto_s_ast_fp64, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_25_fp64_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_25_fp64_auto_s_ast_fp64(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_25_fp64_auto_s_ast_fp64_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_25_fp64_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_25_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_25_fp64_old, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_25_FP64 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_25_fp64_auto_s_ast_fp64, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_25_FP64_auto_s_ast_fp64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_25_fp64_auto_s_ast_fp64, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_25_FP64_auto_s_ast_fp64",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_25_fp64_old.data(), d_mat_J3_ddddv2_25_fp64_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_25_fp64_auto_s_ast_fp64.data(), d_mat_J3_ddddv2_25_fp64_auto_s_ast_fp64,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_25_fp64_auto_s_ast_fp64_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_25_fp64_auto_s_ast_fp64_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_25_fp64_old[idx] + h_mat_J3_ddddv2_25_fp64_auto_s_ast_fp64[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_25_FP64_auto_s_ast_fp64 contribution", h_mat_J3_ddddv2_25_fp64_auto_s_ast_fp64, h_mat_J3_ddddv2_25_fp64_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_25_FP64_auto_s_ast_fp64 mixed result (vs ref)", h_mat_J3_ddddv2_25_fp64_auto_s_ast_fp64_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_25_FP64_auto_s_ast_fp64 mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_25_fp64_auto_s_ast_fp64_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_25_fp64_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_25_fp64_auto_s_ast_fp64));
