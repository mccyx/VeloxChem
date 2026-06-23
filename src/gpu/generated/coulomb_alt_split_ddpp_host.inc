gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
                d_mat_J2_2kernels, static_cast<uint32_t>(dd_prim_pair_count_local));
            gpuSafe(gpuStreamSynchronize(stream));
            gpu::computeCoulombFockDDPP_altA0_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               d_mat_J2_2kernels, d_p_prim_info, static_cast<uint32_t>(p_prim_count), d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_pp_first_inds, d_pp_second_inds, d_pp_pair_data, static_cast<uint32_t>(pp_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
            gpu::computeCoulombFockDDPP_altA1_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               d_mat_J2_2kernels, d_p_prim_info, static_cast<uint32_t>(p_prim_count), d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_pp_first_inds, d_pp_second_inds, d_pp_pair_data, static_cast<uint32_t>(pp_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
            {
                GpuEventHandle start_event;
                GpuEventHandle stop_event;
                createGpuEvent(&start_event);
                createGpuEvent(&stop_event);
                const auto start = std::chrono::steady_clock::now();
                recordGpuEvent(start_event, stream);
            gpu::computeCoulombFockDDPP_altA0_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               d_mat_J2_2kernels, d_p_prim_info_f, static_cast<uint32_t>(p_prim_count), d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_pp_mat_D_f,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_pp_first_inds, d_pp_second_inds, d_pp_pair_data_f, static_cast<uint32_t>(pp_prim_pair_count),
                               d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
            gpu::computeCoulombFockDDPP_altA1_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               d_mat_J2_2kernels, d_p_prim_info_f, static_cast<uint32_t>(p_prim_count), d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_pp_mat_D_f,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_pp_first_inds, d_pp_second_inds, d_pp_pair_data_f, static_cast<uint32_t>(pp_prim_pair_count),
                               d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
                recordGpuEvent(stop_event, stream);
                synchronizeGpuEvent(stop_event);
                const auto end = std::chrono::steady_clock::now();
                append_kernel_timing("computeCoulombFockDDPP_altA_split_FP32_auto_s_ast compare",
                    std::chrono::duration<double, std::milli>(end - start).count(),
                    elapsedGpuEventMs(start_event, stop_event));
                destroyGpuEvent(start_event);
                destroyGpuEvent(stop_event);
            }
            gpuSafe(gpuMemcpyAsync(h_mat_J2_resplit_auto_s_ast.data(), d_mat_J2_2kernels,
                dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
            gpuSafe(gpuStreamSynchronize(stream));
            check_J_against_ref("computeCoulombFockDDPP_altA_split FP32_auto_s_ast mixed result (vs ref)", h_mat_J2_resplit_auto_s_ast, h_mat_J2_ref,
                (uint32_t)dd_prim_pair_count_local);


            gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
                d_mat_J2_2kernels, static_cast<uint32_t>(dd_prim_pair_count_local));
            gpuSafe(gpuStreamSynchronize(stream));
            gpu::computeCoulombFockDDPP_altB0_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               d_mat_J2_2kernels, d_p_prim_info, static_cast<uint32_t>(p_prim_count), d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_pp_first_inds, d_pp_second_inds, d_pp_pair_data, static_cast<uint32_t>(pp_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
            gpu::computeCoulombFockDDPP_altB1_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               d_mat_J2_2kernels, d_p_prim_info, static_cast<uint32_t>(p_prim_count), d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_pp_first_inds, d_pp_second_inds, d_pp_pair_data, static_cast<uint32_t>(pp_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
            gpu::computeCoulombFockDDPP_altB2_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               d_mat_J2_2kernels, d_p_prim_info, static_cast<uint32_t>(p_prim_count), d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_pp_first_inds, d_pp_second_inds, d_pp_pair_data, static_cast<uint32_t>(pp_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
            {
                GpuEventHandle start_event;
                GpuEventHandle stop_event;
                createGpuEvent(&start_event);
                createGpuEvent(&stop_event);
                const auto start = std::chrono::steady_clock::now();
                recordGpuEvent(start_event, stream);
            gpu::computeCoulombFockDDPP_altB0_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               d_mat_J2_2kernels, d_p_prim_info_f, static_cast<uint32_t>(p_prim_count), d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_pp_mat_D_f,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_pp_first_inds, d_pp_second_inds, d_pp_pair_data_f, static_cast<uint32_t>(pp_prim_pair_count),
                               d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
            gpu::computeCoulombFockDDPP_altB1_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               d_mat_J2_2kernels, d_p_prim_info_f, static_cast<uint32_t>(p_prim_count), d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_pp_mat_D_f,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_pp_first_inds, d_pp_second_inds, d_pp_pair_data_f, static_cast<uint32_t>(pp_prim_pair_count),
                               d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
            gpu::computeCoulombFockDDPP_altB2_FP32_auto_s_ast<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               d_mat_J2_2kernels, d_p_prim_info_f, static_cast<uint32_t>(p_prim_count), d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_pp_mat_D_f,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_pp_first_inds, d_pp_second_inds, d_pp_pair_data_f, static_cast<uint32_t>(pp_prim_pair_count),
                               d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
                recordGpuEvent(stop_event, stream);
                synchronizeGpuEvent(stop_event);
                const auto end = std::chrono::steady_clock::now();
                append_kernel_timing("computeCoulombFockDDPP_altB_split_FP32_auto_s_ast compare",
                    std::chrono::duration<double, std::milli>(end - start).count(),
                    elapsedGpuEventMs(start_event, stop_event));
                destroyGpuEvent(start_event);
                destroyGpuEvent(stop_event);
            }
            gpuSafe(gpuMemcpyAsync(h_mat_J2_resplit_auto_s_ast.data(), d_mat_J2_2kernels,
                dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
            gpuSafe(gpuStreamSynchronize(stream));
            check_J_against_ref("computeCoulombFockDDPP_altB_split FP32_auto_s_ast mixed result (vs ref)", h_mat_J2_resplit_auto_s_ast, h_mat_J2_ref,
                (uint32_t)dd_prim_pair_count_local);
