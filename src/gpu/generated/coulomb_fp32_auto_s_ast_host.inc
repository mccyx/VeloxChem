// PPDD_FP32_auto_s_ast
double* d_mat_J3_ppdd_fp32_old = nullptr;
double* d_mat_J3_ppdd_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ppdd_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ppdd_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ppdd_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ppdd_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ppdd_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ppdd_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockPPDD_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ppdd_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockPPDD_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ppdd_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockPPDD_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ppdd_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockPPDD_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ppdd_fp32_old.data(), d_mat_J3_ppdd_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ppdd_fp32_auto_s_ast.data(), d_mat_J3_ppdd_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ppdd_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ppdd_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ppdd_fp32_old[idx] + h_mat_J3_ppdd_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockPPDD_FP32_auto_s_ast contribution", h_mat_J3_ppdd_fp32_auto_s_ast, h_mat_J3_ppdd_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockPPDD_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ppdd_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockPPDD_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ppdd_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ppdd_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ppdd_fp32_auto_s_ast));

// DDSD_FP32_auto_s_ast
double* d_mat_J3_ddsd_fp32_old = nullptr;
double* d_mat_J3_ddsd_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddsd_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddsd_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddsd_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddsd_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddsd_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddsd_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDSD_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddsd_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDSD_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddsd_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDSD_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddsd_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDSD_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddsd_fp32_old.data(), d_mat_J3_ddsd_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddsd_fp32_auto_s_ast.data(), d_mat_J3_ddsd_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddsd_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddsd_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddsd_fp32_old[idx] + h_mat_J3_ddsd_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDSD_FP32_auto_s_ast contribution", h_mat_J3_ddsd_fp32_auto_s_ast, h_mat_J3_ddsd_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDSD_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddsd_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDSD_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddsd_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddsd_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddsd_fp32_auto_s_ast));

// DDPP_FP32_auto_s_ast
double* d_mat_J3_ddpp_fp32_old = nullptr;
double* d_mat_J3_ddpp_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddpp_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddpp_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddpp_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddpp_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddpp_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddpp_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDPP_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddpp_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDPP_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddpp_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDPP_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddpp_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDPP_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddpp_fp32_old.data(), d_mat_J3_ddpp_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddpp_fp32_auto_s_ast.data(), d_mat_J3_ddpp_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddpp_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddpp_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddpp_fp32_old[idx] + h_mat_J3_ddpp_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDPP_FP32_auto_s_ast contribution", h_mat_J3_ddpp_fp32_auto_s_ast, h_mat_J3_ddpp_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDPP_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddpp_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDPP_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddpp_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddpp_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddpp_fp32_auto_s_ast));

// DDPD0_FP32_auto_s_ast
double* d_mat_J3_ddpd0_fp32_old = nullptr;
double* d_mat_J3_ddpd0_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddpd0_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddpd0_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddpd0_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddpd0_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddpd0_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddpd0_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDPD0_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddpd0_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDPD0_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddpd0_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDPD0_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddpd0_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDPD0_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddpd0_fp32_old.data(), d_mat_J3_ddpd0_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddpd0_fp32_auto_s_ast.data(), d_mat_J3_ddpd0_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddpd0_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddpd0_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddpd0_fp32_old[idx] + h_mat_J3_ddpd0_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDPD0_FP32_auto_s_ast contribution", h_mat_J3_ddpd0_fp32_auto_s_ast, h_mat_J3_ddpd0_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDPD0_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddpd0_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDPD0_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddpd0_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddpd0_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddpd0_fp32_auto_s_ast));

// DDPD1_FP32_auto_s_ast
double* d_mat_J3_ddpd1_fp32_old = nullptr;
double* d_mat_J3_ddpd1_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddpd1_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddpd1_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddpd1_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddpd1_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddpd1_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddpd1_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDPD1_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddpd1_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDPD1_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddpd1_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDPD1_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddpd1_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDPD1_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddpd1_fp32_old.data(), d_mat_J3_ddpd1_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddpd1_fp32_auto_s_ast.data(), d_mat_J3_ddpd1_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddpd1_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddpd1_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddpd1_fp32_old[idx] + h_mat_J3_ddpd1_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDPD1_FP32_auto_s_ast contribution", h_mat_J3_ddpd1_fp32_auto_s_ast, h_mat_J3_ddpd1_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDPD1_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddpd1_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDPD1_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddpd1_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddpd1_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddpd1_fp32_auto_s_ast));

// DDPD2_FP32_auto_s_ast
double* d_mat_J3_ddpd2_fp32_old = nullptr;
double* d_mat_J3_ddpd2_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddpd2_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddpd2_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddpd2_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddpd2_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddpd2_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddpd2_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDPD2_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddpd2_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDPD2_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddpd2_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDPD2_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddpd2_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDPD2_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddpd2_fp32_old.data(), d_mat_J3_ddpd2_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddpd2_fp32_auto_s_ast.data(), d_mat_J3_ddpd2_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddpd2_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddpd2_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddpd2_fp32_old[idx] + h_mat_J3_ddpd2_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDPD2_FP32_auto_s_ast contribution", h_mat_J3_ddpd2_fp32_auto_s_ast, h_mat_J3_ddpd2_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDPD2_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddpd2_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDPD2_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddpd2_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddpd2_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddpd2_fp32_auto_s_ast));

// DDPD3_FP32_auto_s_ast
double* d_mat_J3_ddpd3_fp32_old = nullptr;
double* d_mat_J3_ddpd3_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddpd3_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddpd3_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddpd3_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddpd3_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddpd3_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddpd3_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDPD3_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddpd3_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDPD3_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddpd3_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDPD3_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddpd3_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDPD3_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddpd3_fp32_old.data(), d_mat_J3_ddpd3_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddpd3_fp32_auto_s_ast.data(), d_mat_J3_ddpd3_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddpd3_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddpd3_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddpd3_fp32_old[idx] + h_mat_J3_ddpd3_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDPD3_FP32_auto_s_ast contribution", h_mat_J3_ddpd3_fp32_auto_s_ast, h_mat_J3_ddpd3_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDPD3_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddpd3_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDPD3_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddpd3_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddpd3_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddpd3_fp32_auto_s_ast));

// DDPD4_FP32_auto_s_ast
double* d_mat_J3_ddpd4_fp32_old = nullptr;
double* d_mat_J3_ddpd4_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddpd4_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddpd4_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddpd4_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddpd4_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddpd4_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddpd4_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDPD4_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddpd4_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDPD4_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddpd4_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDPD4_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddpd4_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDPD4_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddpd4_fp32_old.data(), d_mat_J3_ddpd4_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddpd4_fp32_auto_s_ast.data(), d_mat_J3_ddpd4_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddpd4_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddpd4_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddpd4_fp32_old[idx] + h_mat_J3_ddpd4_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDPD4_FP32_auto_s_ast contribution", h_mat_J3_ddpd4_fp32_auto_s_ast, h_mat_J3_ddpd4_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDPD4_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddpd4_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDPD4_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddpd4_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddpd4_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddpd4_fp32_auto_s_ast));

// DDPD5_FP32_auto_s_ast
double* d_mat_J3_ddpd5_fp32_old = nullptr;
double* d_mat_J3_ddpd5_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddpd5_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddpd5_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddpd5_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddpd5_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddpd5_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddpd5_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDPD5_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddpd5_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDPD5_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddpd5_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDPD5_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddpd5_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDPD5_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddpd5_fp32_old.data(), d_mat_J3_ddpd5_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddpd5_fp32_auto_s_ast.data(), d_mat_J3_ddpd5_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddpd5_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddpd5_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddpd5_fp32_old[idx] + h_mat_J3_ddpd5_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDPD5_FP32_auto_s_ast contribution", h_mat_J3_ddpd5_fp32_auto_s_ast, h_mat_J3_ddpd5_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDPD5_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddpd5_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDPD5_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddpd5_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddpd5_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddpd5_fp32_auto_s_ast));

// DDPD6_FP32_auto_s_ast
double* d_mat_J3_ddpd6_fp32_old = nullptr;
double* d_mat_J3_ddpd6_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddpd6_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddpd6_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddpd6_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddpd6_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddpd6_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddpd6_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDPD6_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddpd6_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDPD6_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddpd6_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDPD6_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddpd6_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDPD6_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddpd6_fp32_old.data(), d_mat_J3_ddpd6_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddpd6_fp32_auto_s_ast.data(), d_mat_J3_ddpd6_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddpd6_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddpd6_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddpd6_fp32_old[idx] + h_mat_J3_ddpd6_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDPD6_FP32_auto_s_ast contribution", h_mat_J3_ddpd6_fp32_auto_s_ast, h_mat_J3_ddpd6_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDPD6_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddpd6_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDPD6_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddpd6_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddpd6_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddpd6_fp32_auto_s_ast));

// DDPD7_FP32_auto_s_ast
double* d_mat_J3_ddpd7_fp32_old = nullptr;
double* d_mat_J3_ddpd7_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddpd7_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddpd7_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddpd7_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddpd7_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddpd7_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddpd7_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDPD7_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddpd7_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDPD7_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddpd7_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDPD7_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddpd7_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDPD7_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddpd7_fp32_old.data(), d_mat_J3_ddpd7_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddpd7_fp32_auto_s_ast.data(), d_mat_J3_ddpd7_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddpd7_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddpd7_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddpd7_fp32_old[idx] + h_mat_J3_ddpd7_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDPD7_FP32_auto_s_ast contribution", h_mat_J3_ddpd7_fp32_auto_s_ast, h_mat_J3_ddpd7_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDPD7_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddpd7_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDPD7_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddpd7_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddpd7_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddpd7_fp32_auto_s_ast));

// DDPD8_FP32_auto_s_ast
double* d_mat_J3_ddpd8_fp32_old = nullptr;
double* d_mat_J3_ddpd8_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddpd8_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddpd8_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddpd8_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddpd8_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddpd8_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddpd8_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDPD8_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddpd8_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDPD8_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddpd8_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDPD8_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddpd8_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDPD8_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddpd8_fp32_old.data(), d_mat_J3_ddpd8_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddpd8_fp32_auto_s_ast.data(), d_mat_J3_ddpd8_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddpd8_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddpd8_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddpd8_fp32_old[idx] + h_mat_J3_ddpd8_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDPD8_FP32_auto_s_ast contribution", h_mat_J3_ddpd8_fp32_auto_s_ast, h_mat_J3_ddpd8_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDPD8_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddpd8_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDPD8_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddpd8_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddpd8_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddpd8_fp32_auto_s_ast));

// DDPD9_FP32_auto_s_ast
double* d_mat_J3_ddpd9_fp32_old = nullptr;
double* d_mat_J3_ddpd9_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddpd9_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddpd9_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddpd9_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddpd9_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddpd9_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddpd9_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDPD9_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddpd9_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDPD9_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddpd9_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDPD9_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddpd9_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDPD9_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddpd9_fp32_old.data(), d_mat_J3_ddpd9_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddpd9_fp32_auto_s_ast.data(), d_mat_J3_ddpd9_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddpd9_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddpd9_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddpd9_fp32_old[idx] + h_mat_J3_ddpd9_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDPD9_FP32_auto_s_ast contribution", h_mat_J3_ddpd9_fp32_auto_s_ast, h_mat_J3_ddpd9_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDPD9_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddpd9_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDPD9_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddpd9_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddpd9_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddpd9_fp32_auto_s_ast));
