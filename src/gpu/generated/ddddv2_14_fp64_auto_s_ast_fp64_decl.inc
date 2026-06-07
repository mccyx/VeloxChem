__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDDv2_14_FP64_auto_s_ast_fp64(double*         mat_J,
                       const double*   d_prim_info,
                       const uint32_t  d_prim_count,
                       const double*   dd_mat_D,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const double*   dd_pair_data_local,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* dd_first_inds,
                       const uint32_t* dd_second_inds,
                       const double*   dd_pair_data,
                       const uint32_t  dd_prim_pair_count,
                       const double*   boys_func_table,
                       const double*   boys_func_ft,
                       const uint32_t* prec_cut_ij_tile)
;
