gpu::computeCoulombFockDDDDv2_0_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               target, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
gpu::computeCoulombFockDDDDv2_1_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               target, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
gpu::computeCoulombFockDDDDv2_2_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               target, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
gpu::computeCoulombFockDDDDv2_3_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               target, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
gpu::computeCoulombFockDDDDv2_4_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               target, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
gpu::computeCoulombFockDDDDv2_5_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               target, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
gpu::computeCoulombFockDDDDv2_6_FP64_auto_s_ast_fp64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               target, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
gpu::computeCoulombFockDDDDv2_7_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               target, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
gpu::computeCoulombFockDDDDv2_8_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               target, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
gpu::computeCoulombFockDDDDv2_9_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               target, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
gpu::computeCoulombFockDDDDv2_10_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               target, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
gpu::computeCoulombFockDDDDv2_11_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               target, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
gpu::computeCoulombFockDDDDv2_12_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               target, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
gpu::computeCoulombFockDDDDv2_13_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               target, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
gpu::computeCoulombFockDDDDv2_14_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               target, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
gpu::computeCoulombFockDDDDv2_15_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               target, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
gpu::computeCoulombFockDDDDv2_16_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               target, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
gpu::computeCoulombFockDDDDv2_17_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               target, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
gpu::computeCoulombFockDDDDv2_18_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               target, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
gpu::computeCoulombFockDDDDv2_19_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               target, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
gpu::computeCoulombFockDDDDv2_20_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               target, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
gpu::computeCoulombFockDDDDv2_21_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               target, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
gpu::computeCoulombFockDDDDv2_22_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               target, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
gpu::computeCoulombFockDDDDv2_23_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               target, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
gpu::computeCoulombFockDDDDv2_24_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               target, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
gpu::computeCoulombFockDDDDv2_25_FP64<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
                               target, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile);
