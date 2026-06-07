// DDDDv2_0_FP32_auto_s_ast
double* d_mat_J3_ddddv2_0_fp32_old = nullptr;
double* d_mat_J3_ddddv2_0_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_0_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_0_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_0_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_0_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_0_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_0_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_0_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_0_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_0_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_0_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_0_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_0_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_0_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_0_fp32_old.data(), d_mat_J3_ddddv2_0_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_0_fp32_auto_s_ast.data(), d_mat_J3_ddddv2_0_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_0_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_0_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_0_fp32_old[idx] + h_mat_J3_ddddv2_0_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_0_FP32_auto_s_ast contribution", h_mat_J3_ddddv2_0_fp32_auto_s_ast, h_mat_J3_ddddv2_0_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_0_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddddv2_0_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_0_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_0_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_0_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_0_fp32_auto_s_ast));

// DDDDv2_1_FP32_auto_s_ast
double* d_mat_J3_ddddv2_1_fp32_old = nullptr;
double* d_mat_J3_ddddv2_1_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_1_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_1_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_1_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_1_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_1_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_1_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_1_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_1_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_1_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_1_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_1_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_1_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_1_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_1_fp32_old.data(), d_mat_J3_ddddv2_1_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_1_fp32_auto_s_ast.data(), d_mat_J3_ddddv2_1_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_1_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_1_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_1_fp32_old[idx] + h_mat_J3_ddddv2_1_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_1_FP32_auto_s_ast contribution", h_mat_J3_ddddv2_1_fp32_auto_s_ast, h_mat_J3_ddddv2_1_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_1_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddddv2_1_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_1_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_1_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_1_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_1_fp32_auto_s_ast));

// DDDDv2_2_FP32_auto_s_ast
double* d_mat_J3_ddddv2_2_fp32_old = nullptr;
double* d_mat_J3_ddddv2_2_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_2_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_2_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_2_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_2_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_2_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_2_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_2_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_2_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_2_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_2_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_2_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_2_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_2_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_2_fp32_old.data(), d_mat_J3_ddddv2_2_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_2_fp32_auto_s_ast.data(), d_mat_J3_ddddv2_2_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_2_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_2_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_2_fp32_old[idx] + h_mat_J3_ddddv2_2_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_2_FP32_auto_s_ast contribution", h_mat_J3_ddddv2_2_fp32_auto_s_ast, h_mat_J3_ddddv2_2_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_2_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddddv2_2_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_2_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_2_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_2_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_2_fp32_auto_s_ast));

// DDDDv2_3_FP32_auto_s_ast
double* d_mat_J3_ddddv2_3_fp32_old = nullptr;
double* d_mat_J3_ddddv2_3_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_3_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_3_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_3_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_3_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_3_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_3_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_3_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_3_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_3_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_3_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_3_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_3_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_3_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_3_fp32_old.data(), d_mat_J3_ddddv2_3_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_3_fp32_auto_s_ast.data(), d_mat_J3_ddddv2_3_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_3_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_3_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_3_fp32_old[idx] + h_mat_J3_ddddv2_3_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_3_FP32_auto_s_ast contribution", h_mat_J3_ddddv2_3_fp32_auto_s_ast, h_mat_J3_ddddv2_3_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_3_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddddv2_3_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_3_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_3_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_3_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_3_fp32_auto_s_ast));

// DDDDv2_4_FP32_auto_s_ast
double* d_mat_J3_ddddv2_4_fp32_old = nullptr;
double* d_mat_J3_ddddv2_4_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_4_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_4_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_4_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_4_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_4_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_4_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_4_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_4_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_4_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_4_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_4_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_4_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_4_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_4_fp32_old.data(), d_mat_J3_ddddv2_4_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_4_fp32_auto_s_ast.data(), d_mat_J3_ddddv2_4_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_4_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_4_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_4_fp32_old[idx] + h_mat_J3_ddddv2_4_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_4_FP32_auto_s_ast contribution", h_mat_J3_ddddv2_4_fp32_auto_s_ast, h_mat_J3_ddddv2_4_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_4_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddddv2_4_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_4_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_4_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_4_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_4_fp32_auto_s_ast));

// DDDDv2_5_FP32_auto_s_ast
double* d_mat_J3_ddddv2_5_fp32_old = nullptr;
double* d_mat_J3_ddddv2_5_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_5_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_5_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_5_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_5_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_5_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_5_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_5_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_5_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_5_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_5_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_5_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_5_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_5_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_5_fp32_old.data(), d_mat_J3_ddddv2_5_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_5_fp32_auto_s_ast.data(), d_mat_J3_ddddv2_5_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_5_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_5_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_5_fp32_old[idx] + h_mat_J3_ddddv2_5_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_5_FP32_auto_s_ast contribution", h_mat_J3_ddddv2_5_fp32_auto_s_ast, h_mat_J3_ddddv2_5_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_5_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddddv2_5_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_5_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_5_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_5_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_5_fp32_auto_s_ast));

// DDDDv2_6_FP32_auto_s_ast
double* d_mat_J3_ddddv2_6_fp32_old = nullptr;
double* d_mat_J3_ddddv2_6_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_6_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_6_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_6_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_6_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_6_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_6_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_6_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_6_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_6_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_6_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_6_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_6_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_6_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_6_fp32_old.data(), d_mat_J3_ddddv2_6_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_6_fp32_auto_s_ast.data(), d_mat_J3_ddddv2_6_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_6_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_6_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_6_fp32_old[idx] + h_mat_J3_ddddv2_6_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_6_FP32_auto_s_ast contribution", h_mat_J3_ddddv2_6_fp32_auto_s_ast, h_mat_J3_ddddv2_6_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_6_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddddv2_6_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_6_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_6_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_6_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_6_fp32_auto_s_ast));

// DDDDv2_7_FP32_auto_s_ast
double* d_mat_J3_ddddv2_7_fp32_old = nullptr;
double* d_mat_J3_ddddv2_7_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_7_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_7_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_7_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_7_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_7_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_7_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_7_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_7_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_7_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_7_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_7_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_7_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_7_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_7_fp32_old.data(), d_mat_J3_ddddv2_7_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_7_fp32_auto_s_ast.data(), d_mat_J3_ddddv2_7_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_7_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_7_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_7_fp32_old[idx] + h_mat_J3_ddddv2_7_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_7_FP32_auto_s_ast contribution", h_mat_J3_ddddv2_7_fp32_auto_s_ast, h_mat_J3_ddddv2_7_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_7_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddddv2_7_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_7_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_7_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_7_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_7_fp32_auto_s_ast));

// DDDDv2_8_FP32_auto_s_ast
double* d_mat_J3_ddddv2_8_fp32_old = nullptr;
double* d_mat_J3_ddddv2_8_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_8_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_8_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_8_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_8_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_8_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_8_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_8_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_8_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_8_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_8_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_8_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_8_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_8_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_8_fp32_old.data(), d_mat_J3_ddddv2_8_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_8_fp32_auto_s_ast.data(), d_mat_J3_ddddv2_8_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_8_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_8_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_8_fp32_old[idx] + h_mat_J3_ddddv2_8_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_8_FP32_auto_s_ast contribution", h_mat_J3_ddddv2_8_fp32_auto_s_ast, h_mat_J3_ddddv2_8_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_8_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddddv2_8_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_8_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_8_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_8_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_8_fp32_auto_s_ast));

// DDDDv2_9_FP32_auto_s_ast
double* d_mat_J3_ddddv2_9_fp32_old = nullptr;
double* d_mat_J3_ddddv2_9_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_9_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_9_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_9_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_9_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_9_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_9_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_9_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_9_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_9_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_9_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_9_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_9_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_9_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_9_fp32_old.data(), d_mat_J3_ddddv2_9_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_9_fp32_auto_s_ast.data(), d_mat_J3_ddddv2_9_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_9_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_9_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_9_fp32_old[idx] + h_mat_J3_ddddv2_9_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_9_FP32_auto_s_ast contribution", h_mat_J3_ddddv2_9_fp32_auto_s_ast, h_mat_J3_ddddv2_9_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_9_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddddv2_9_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_9_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_9_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_9_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_9_fp32_auto_s_ast));

// DDDDv2_10_FP32_auto_s_ast
double* d_mat_J3_ddddv2_10_fp32_old = nullptr;
double* d_mat_J3_ddddv2_10_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_10_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_10_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_10_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_10_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_10_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_10_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_10_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_10_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_10_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_10_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_10_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_10_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_10_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_10_fp32_old.data(), d_mat_J3_ddddv2_10_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_10_fp32_auto_s_ast.data(), d_mat_J3_ddddv2_10_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_10_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_10_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_10_fp32_old[idx] + h_mat_J3_ddddv2_10_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_10_FP32_auto_s_ast contribution", h_mat_J3_ddddv2_10_fp32_auto_s_ast, h_mat_J3_ddddv2_10_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_10_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddddv2_10_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_10_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_10_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_10_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_10_fp32_auto_s_ast));

// DDDDv2_11_FP32_auto_s_ast
double* d_mat_J3_ddddv2_11_fp32_old = nullptr;
double* d_mat_J3_ddddv2_11_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_11_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_11_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_11_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_11_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_11_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_11_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_11_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_11_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_11_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_11_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_11_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_11_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_11_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_11_fp32_old.data(), d_mat_J3_ddddv2_11_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_11_fp32_auto_s_ast.data(), d_mat_J3_ddddv2_11_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_11_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_11_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_11_fp32_old[idx] + h_mat_J3_ddddv2_11_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_11_FP32_auto_s_ast contribution", h_mat_J3_ddddv2_11_fp32_auto_s_ast, h_mat_J3_ddddv2_11_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_11_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddddv2_11_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_11_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_11_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_11_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_11_fp32_auto_s_ast));

// DDDDv2_12_FP32_auto_s_ast
double* d_mat_J3_ddddv2_12_fp32_old = nullptr;
double* d_mat_J3_ddddv2_12_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_12_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_12_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_12_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_12_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_12_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_12_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_12_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_12_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_12_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_12_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_12_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_12_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_12_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_12_fp32_old.data(), d_mat_J3_ddddv2_12_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_12_fp32_auto_s_ast.data(), d_mat_J3_ddddv2_12_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_12_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_12_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_12_fp32_old[idx] + h_mat_J3_ddddv2_12_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_12_FP32_auto_s_ast contribution", h_mat_J3_ddddv2_12_fp32_auto_s_ast, h_mat_J3_ddddv2_12_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_12_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddddv2_12_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_12_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_12_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_12_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_12_fp32_auto_s_ast));

// DDDDv2_13_FP32_auto_s_ast
double* d_mat_J3_ddddv2_13_fp32_old = nullptr;
double* d_mat_J3_ddddv2_13_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_13_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_13_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_13_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_13_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_13_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_13_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_13_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_13_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_13_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_13_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_13_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_13_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_13_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_13_fp32_old.data(), d_mat_J3_ddddv2_13_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_13_fp32_auto_s_ast.data(), d_mat_J3_ddddv2_13_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_13_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_13_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_13_fp32_old[idx] + h_mat_J3_ddddv2_13_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_13_FP32_auto_s_ast contribution", h_mat_J3_ddddv2_13_fp32_auto_s_ast, h_mat_J3_ddddv2_13_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_13_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddddv2_13_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_13_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_13_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_13_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_13_fp32_auto_s_ast));

// DDDDv2_14_FP32_auto_s_ast
double* d_mat_J3_ddddv2_14_fp32_old = nullptr;
double* d_mat_J3_ddddv2_14_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_14_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_14_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_14_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_14_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_14_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_14_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_14_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_14_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_14_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_14_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_14_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_14_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_14_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_14_fp32_old.data(), d_mat_J3_ddddv2_14_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_14_fp32_auto_s_ast.data(), d_mat_J3_ddddv2_14_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_14_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_14_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_14_fp32_old[idx] + h_mat_J3_ddddv2_14_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_14_FP32_auto_s_ast contribution", h_mat_J3_ddddv2_14_fp32_auto_s_ast, h_mat_J3_ddddv2_14_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_14_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddddv2_14_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_14_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_14_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_14_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_14_fp32_auto_s_ast));

// DDDDv2_15_FP32_auto_s_ast
double* d_mat_J3_ddddv2_15_fp32_old = nullptr;
double* d_mat_J3_ddddv2_15_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_15_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_15_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_15_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_15_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_15_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_15_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_15_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_15_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_15_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_15_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_15_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_15_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_15_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_15_fp32_old.data(), d_mat_J3_ddddv2_15_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_15_fp32_auto_s_ast.data(), d_mat_J3_ddddv2_15_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_15_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_15_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_15_fp32_old[idx] + h_mat_J3_ddddv2_15_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_15_FP32_auto_s_ast contribution", h_mat_J3_ddddv2_15_fp32_auto_s_ast, h_mat_J3_ddddv2_15_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_15_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddddv2_15_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_15_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_15_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_15_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_15_fp32_auto_s_ast));

// DDDDv2_16_FP32_auto_s_ast
double* d_mat_J3_ddddv2_16_fp32_old = nullptr;
double* d_mat_J3_ddddv2_16_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_16_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_16_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_16_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_16_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_16_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_16_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_16_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_16_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_16_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_16_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_16_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_16_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_16_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_16_fp32_old.data(), d_mat_J3_ddddv2_16_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_16_fp32_auto_s_ast.data(), d_mat_J3_ddddv2_16_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_16_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_16_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_16_fp32_old[idx] + h_mat_J3_ddddv2_16_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_16_FP32_auto_s_ast contribution", h_mat_J3_ddddv2_16_fp32_auto_s_ast, h_mat_J3_ddddv2_16_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_16_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddddv2_16_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_16_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_16_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_16_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_16_fp32_auto_s_ast));

// DDDDv2_17_FP32_auto_s_ast
double* d_mat_J3_ddddv2_17_fp32_old = nullptr;
double* d_mat_J3_ddddv2_17_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_17_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_17_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_17_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_17_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_17_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_17_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_17_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_17_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_17_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_17_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_17_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_17_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_17_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_17_fp32_old.data(), d_mat_J3_ddddv2_17_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_17_fp32_auto_s_ast.data(), d_mat_J3_ddddv2_17_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_17_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_17_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_17_fp32_old[idx] + h_mat_J3_ddddv2_17_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_17_FP32_auto_s_ast contribution", h_mat_J3_ddddv2_17_fp32_auto_s_ast, h_mat_J3_ddddv2_17_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_17_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddddv2_17_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_17_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_17_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_17_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_17_fp32_auto_s_ast));

// DDDDv2_18_FP32_auto_s_ast
double* d_mat_J3_ddddv2_18_fp32_old = nullptr;
double* d_mat_J3_ddddv2_18_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_18_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_18_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_18_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_18_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_18_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_18_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_18_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_18_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_18_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_18_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_18_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_18_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_18_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_18_fp32_old.data(), d_mat_J3_ddddv2_18_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_18_fp32_auto_s_ast.data(), d_mat_J3_ddddv2_18_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_18_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_18_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_18_fp32_old[idx] + h_mat_J3_ddddv2_18_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_18_FP32_auto_s_ast contribution", h_mat_J3_ddddv2_18_fp32_auto_s_ast, h_mat_J3_ddddv2_18_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_18_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddddv2_18_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_18_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_18_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_18_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_18_fp32_auto_s_ast));

// DDDDv2_19_FP32_auto_s_ast
double* d_mat_J3_ddddv2_19_fp32_old = nullptr;
double* d_mat_J3_ddddv2_19_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_19_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_19_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_19_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_19_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_19_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_19_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_19_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_19_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_19_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_19_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_19_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_19_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_19_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_19_fp32_old.data(), d_mat_J3_ddddv2_19_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_19_fp32_auto_s_ast.data(), d_mat_J3_ddddv2_19_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_19_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_19_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_19_fp32_old[idx] + h_mat_J3_ddddv2_19_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_19_FP32_auto_s_ast contribution", h_mat_J3_ddddv2_19_fp32_auto_s_ast, h_mat_J3_ddddv2_19_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_19_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddddv2_19_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_19_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_19_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_19_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_19_fp32_auto_s_ast));

// DDDDv2_20_FP32_auto_s_ast
double* d_mat_J3_ddddv2_20_fp32_old = nullptr;
double* d_mat_J3_ddddv2_20_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_20_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_20_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_20_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_20_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_20_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_20_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_20_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_20_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_20_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_20_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_20_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_20_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_20_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_20_fp32_old.data(), d_mat_J3_ddddv2_20_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_20_fp32_auto_s_ast.data(), d_mat_J3_ddddv2_20_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_20_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_20_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_20_fp32_old[idx] + h_mat_J3_ddddv2_20_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_20_FP32_auto_s_ast contribution", h_mat_J3_ddddv2_20_fp32_auto_s_ast, h_mat_J3_ddddv2_20_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_20_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddddv2_20_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_20_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_20_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_20_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_20_fp32_auto_s_ast));

// DDDDv2_21_FP32_auto_s_ast
double* d_mat_J3_ddddv2_21_fp32_old = nullptr;
double* d_mat_J3_ddddv2_21_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_21_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_21_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_21_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_21_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_21_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_21_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_21_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_21_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_21_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_21_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_21_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_21_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_21_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_21_fp32_old.data(), d_mat_J3_ddddv2_21_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_21_fp32_auto_s_ast.data(), d_mat_J3_ddddv2_21_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_21_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_21_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_21_fp32_old[idx] + h_mat_J3_ddddv2_21_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_21_FP32_auto_s_ast contribution", h_mat_J3_ddddv2_21_fp32_auto_s_ast, h_mat_J3_ddddv2_21_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_21_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddddv2_21_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_21_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_21_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_21_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_21_fp32_auto_s_ast));

// DDDDv2_22_FP32_auto_s_ast
double* d_mat_J3_ddddv2_22_fp32_old = nullptr;
double* d_mat_J3_ddddv2_22_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_22_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_22_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_22_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_22_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_22_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_22_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_22_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_22_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_22_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_22_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_22_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_22_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_22_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_22_fp32_old.data(), d_mat_J3_ddddv2_22_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_22_fp32_auto_s_ast.data(), d_mat_J3_ddddv2_22_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_22_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_22_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_22_fp32_old[idx] + h_mat_J3_ddddv2_22_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_22_FP32_auto_s_ast contribution", h_mat_J3_ddddv2_22_fp32_auto_s_ast, h_mat_J3_ddddv2_22_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_22_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddddv2_22_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_22_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_22_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_22_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_22_fp32_auto_s_ast));

// DDDDv2_23_FP32_auto_s_ast
double* d_mat_J3_ddddv2_23_fp32_old = nullptr;
double* d_mat_J3_ddddv2_23_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_23_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_23_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_23_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_23_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_23_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_23_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_23_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_23_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_23_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_23_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_23_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_23_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_23_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_23_fp32_old.data(), d_mat_J3_ddddv2_23_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_23_fp32_auto_s_ast.data(), d_mat_J3_ddddv2_23_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_23_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_23_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_23_fp32_old[idx] + h_mat_J3_ddddv2_23_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_23_FP32_auto_s_ast contribution", h_mat_J3_ddddv2_23_fp32_auto_s_ast, h_mat_J3_ddddv2_23_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_23_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddddv2_23_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_23_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_23_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_23_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_23_fp32_auto_s_ast));

// DDDDv2_24_FP32_auto_s_ast
double* d_mat_J3_ddddv2_24_fp32_old = nullptr;
double* d_mat_J3_ddddv2_24_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_24_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_24_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_24_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_24_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_24_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_24_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_24_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_24_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_24_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_24_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_24_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_24_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_24_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_24_fp32_old.data(), d_mat_J3_ddddv2_24_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_24_fp32_auto_s_ast.data(), d_mat_J3_ddddv2_24_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_24_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_24_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_24_fp32_old[idx] + h_mat_J3_ddddv2_24_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_24_FP32_auto_s_ast contribution", h_mat_J3_ddddv2_24_fp32_auto_s_ast, h_mat_J3_ddddv2_24_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_24_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddddv2_24_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_24_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_24_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_24_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_24_fp32_auto_s_ast));

// DDDDv2_25_FP32_auto_s_ast
double* d_mat_J3_ddddv2_25_fp32_old = nullptr;
double* d_mat_J3_ddddv2_25_fp32_auto_s_ast = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_25_fp32_old, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_ddddv2_25_fp32_auto_s_ast, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_ddddv2_25_fp32_old(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_25_fp32_auto_s_ast(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_ddddv2_25_fp32_auto_s_ast_mix(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_25_fp32_old, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_25_FP32<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_25_fp32_old, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_25_FP32 baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_ddddv2_25_fp32_auto_s_ast, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::computeCoulombFockDDDDv2_25_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_ddddv2_25_fp32_auto_s_ast, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("computeCoulombFockDDDDv2_25_FP32_auto_s_ast",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}

gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_25_fp32_old.data(), d_mat_J3_ddddv2_25_fp32_old,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_ddddv2_25_fp32_auto_s_ast.data(), d_mat_J3_ddddv2_25_fp32_auto_s_ast,
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_ddddv2_25_fp32_auto_s_ast_mix.size(); ++idx)
{
    h_mat_J3_ddddv2_25_fp32_auto_s_ast_mix[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_ddddv2_25_fp32_old[idx] + h_mat_J3_ddddv2_25_fp32_auto_s_ast[idx];
}

check_J_against_ref("computeCoulombFockDDDDv2_25_FP32_auto_s_ast contribution", h_mat_J3_ddddv2_25_fp32_auto_s_ast, h_mat_J3_ddddv2_25_fp32_old,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_25_FP32_auto_s_ast mixed result (vs ref)", h_mat_J3_ddddv2_25_fp32_auto_s_ast_mix, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("computeCoulombFockDDDDv2_25_FP32_auto_s_ast mixed result (vs split26 v2 baseline)", h_mat_J3_ddddv2_25_fp32_auto_s_ast_mix, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_ddddv2_25_fp32_old));
gpuSafe(gpuFree(d_mat_J3_ddddv2_25_fp32_auto_s_ast));
