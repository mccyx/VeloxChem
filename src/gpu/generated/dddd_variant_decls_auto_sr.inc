__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDDv2_2_FP32_auto_sr(double*         mat_J,
                       const float*    d_prim_info,
                       const uint32_t  d_prim_count,
                       const float*    dd_mat_D,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const float*    dd_pair_data_local,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* dd_first_inds,
                       const uint32_t* dd_second_inds,
                       const float*    dd_pair_data,
                       const uint32_t  dd_prim_pair_count,
                       const float*    boys_func_table,
                       const float*    boys_func_ft,
                       const uint32_t* prec_cut_ij_tile,
                       const uint32_t* screen_cut_ij_tile)
;

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDDv2_4_FP32_auto_sr(double*         mat_J,
                       const float*    d_prim_info,
                       const uint32_t  d_prim_count,
                       const float*    dd_mat_D,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const float*    dd_pair_data_local,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* dd_first_inds,
                       const uint32_t* dd_second_inds,
                       const float*    dd_pair_data,
                       const uint32_t  dd_prim_pair_count,
                       const float*    boys_func_table,
                       const float*    boys_func_ft,
                       const uint32_t* prec_cut_ij_tile,
                       const uint32_t* screen_cut_ij_tile)
;

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDDv2_20_FP32_auto_sr(double*         mat_J,
                       const float*    d_prim_info,
                       const uint32_t  d_prim_count,
                       const float*    dd_mat_D,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const float*    dd_pair_data_local,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* dd_first_inds,
                       const uint32_t* dd_second_inds,
                       const float*    dd_pair_data,
                       const uint32_t  dd_prim_pair_count,
                       const float*    boys_func_table,
                       const float*    boys_func_ft,
                       const uint32_t* prec_cut_ij_tile,
                       const uint32_t* screen_cut_ij_tile)
;

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDDv2_12_FP32_auto_sr(double*         mat_J,
                       const float*    d_prim_info,
                       const uint32_t  d_prim_count,
                       const float*    dd_mat_D,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const float*    dd_pair_data_local,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* dd_first_inds,
                       const uint32_t* dd_second_inds,
                       const float*    dd_pair_data,
                       const uint32_t  dd_prim_pair_count,
                       const float*    boys_func_table,
                       const float*    boys_func_ft,
                       const uint32_t* prec_cut_ij_tile,
                       const uint32_t* screen_cut_ij_tile)
;

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDDv2_16_FP32_auto_sr(double*         mat_J,
                       const float*    d_prim_info,
                       const uint32_t  d_prim_count,
                       const float*    dd_mat_D,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const float*    dd_pair_data_local,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* dd_first_inds,
                       const uint32_t* dd_second_inds,
                       const float*    dd_pair_data,
                       const uint32_t  dd_prim_pair_count,
                       const float*    boys_func_table,
                       const float*    boys_func_ft,
                       const uint32_t* prec_cut_ij_tile,
                       const uint32_t* screen_cut_ij_tile)
;

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDDv2_25_FP32_auto_sr(double*         mat_J,
                       const float*    d_prim_info,
                       const uint32_t  d_prim_count,
                       const float*    dd_mat_D,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const float*    dd_pair_data_local,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* dd_first_inds,
                       const uint32_t* dd_second_inds,
                       const float*    dd_pair_data,
                       const uint32_t  dd_prim_pair_count,
                       const float*    boys_func_table,
                       const float*    boys_func_ft,
                       const uint32_t* prec_cut_ij_tile,
                       const uint32_t* screen_cut_ij_tile)
;
