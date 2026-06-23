__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockPPDD_FP32_auto_s_ast(double*         mat_J,
                       const float*    p_prim_info_f,
                       const uint32_t  p_prim_count,
                       const float*    d_prim_info_f,
                       const uint32_t  d_prim_count,
                       const float*    dd_mat_D_f,
                       const uint32_t* pp_first_inds_local,
                       const uint32_t* pp_second_inds_local,
                       const float*    pp_pair_data_local_f,
                       const uint32_t  pp_prim_pair_count_local,
                       const uint32_t* dd_first_inds,
                       const uint32_t* dd_second_inds,
                       const float*    dd_pair_data_f,
                       const uint32_t  dd_prim_pair_count,
                       const float*    boys_func_table_f,
                       const float*    boys_func_ft_f,
                       const uint32_t* prec_cut_ij_tile,
                       const uint32_t* screen_cut_ij_tile)
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM][TILE_DIM + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ float   delta_f[3][3];

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];
    const uint32_t screen_cut = screen_cut_ij_tile[ij_tile];

    float a_i_f, a_j_f, r_i_f[3], r_j_f[3], S_ij_00_f, S1_f, inv_S1_f;
    float PA_0_f, PB_0_f;
    uint32_t i, j, a0, b0;

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {
        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta_f[0][0] = 1.0f; delta_f[0][1] = 0.0f; delta_f[0][2] = 0.0f;
        delta_f[1][0] = 0.0f; delta_f[1][1] = 1.0f; delta_f[1][2] = 0.0f;
        delta_f[2][0] = 0.0f; delta_f[2][1] = 0.0f; delta_f[2][2] = 1.0f;

    }

    ERIs[threadIdx.y][threadIdx.x] = 0.0;

    __syncthreads();

    if (ij < pp_prim_pair_count_local)
    {
        i = pp_first_inds_local[ij];
        j = pp_second_inds_local[ij];

        a_i_f = p_prim_info_f[i / 3 + p_prim_count * 0];

        r_i_f[0] = p_prim_info_f[i / 3 + p_prim_count * 2];
        r_i_f[1] = p_prim_info_f[i / 3 + p_prim_count * 3];
        r_i_f[2] = p_prim_info_f[i / 3 + p_prim_count * 4];

        a_j_f = p_prim_info_f[j / 3 + p_prim_count * 0];

        r_j_f[0] = p_prim_info_f[j / 3 + p_prim_count * 2];
        r_j_f[1] = p_prim_info_f[j / 3 + p_prim_count * 3];
        r_j_f[2] = p_prim_info_f[j / 3 + p_prim_count * 4];

        S1_f = a_i_f + a_j_f;
        inv_S1_f = (float) (1.0 / (double)S1_f);

        S_ij_00_f = pp_pair_data_local_f[ij];

        a0 = i % 3;
        b0 = j % 3;

        PA_0_f = (a_j_f  * inv_S1_f) * (r_j_f[a0] - r_i_f[a0]);
        PB_0_f = (-a_i_f * inv_S1_f) * (r_j_f[b0] - r_i_f[b0]);

    }

    for (uint32_t m = prec_cut; m < screen_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM + threadIdx.y;

        if ((ij >= pp_prim_pair_count_local) || (kl >= dd_prim_pair_count))
        {
            break;
        }

        const auto k = dd_first_inds[kl];
        const auto l = dd_second_inds[kl];

        const auto a_k_f = d_prim_info_f[k / 6 + d_prim_count * 0];

        const float r_k0_f = d_prim_info_f[k / 6 + d_prim_count * 2];
        const float r_k1_f = d_prim_info_f[k / 6 + d_prim_count * 3];
        const float r_k2_f = d_prim_info_f[k / 6 + d_prim_count * 4];

        const auto a_l_f = d_prim_info_f[l / 6 + d_prim_count * 0];

        const float r_l0_f = d_prim_info_f[l / 6 + d_prim_count * 2];
        const float r_l1_f = d_prim_info_f[l / 6 + d_prim_count * 3];
        const float r_l2_f = d_prim_info_f[l / 6 + d_prim_count * 4];

        const auto S_kl_00_f = dd_pair_data_f[kl];

        const auto c0 = d_cart_inds[k % 6][0];
        const auto c1 = d_cart_inds[k % 6][1];
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        const float r_l_c0_f = (c0 == 0 ? r_l0_f : (c0 == 1 ? r_l1_f : r_l2_f));
        const float r_l_c1_f = (c1 == 0 ? r_l0_f : (c1 == 1 ? r_l1_f : r_l2_f));
        const float r_l_d0_f = (d0 == 0 ? r_l0_f : (d0 == 1 ? r_l1_f : r_l2_f));
        const float r_l_d1_f = (d1 == 0 ? r_l0_f : (d1 == 1 ? r_l1_f : r_l2_f));

        const float r_k_c0_f = (c0 == 0 ? r_k0_f : (c0 == 1 ? r_k1_f : r_k2_f));
        const float r_k_c1_f = (c1 == 0 ? r_k0_f : (c1 == 1 ? r_k1_f : r_k2_f));
        const float r_k_d0_f = (d0 == 0 ? r_k0_f : (d0 == 1 ? r_k1_f : r_k2_f));
        const float r_k_d1_f = (d1 == 0 ? r_k0_f : (d1 == 1 ? r_k1_f : r_k2_f));

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2_f = a_k_f + a_l_f;

        const auto inv_S2_f = 1.0f / S2_f;
        const auto inv_S4_f = 1.0f / (S1_f + S2_f);

        const float PQ0_f = (a_k_f * r_k0_f + a_l_f * r_l0_f) * inv_S2_f - (a_i_f * r_i_f[0] + a_j_f * r_j_f[0]) * inv_S1_f;
        const float PQ1_f = (a_k_f * r_k1_f + a_l_f * r_l1_f) * inv_S2_f - (a_i_f * r_i_f[1] + a_j_f * r_j_f[1]) * inv_S1_f;
        const float PQ2_f = (a_k_f * r_k2_f + a_l_f * r_l2_f) * inv_S2_f - (a_i_f * r_i_f[2] + a_j_f * r_j_f[2]) * inv_S1_f;

        const float PQ_a0_f = (a0 == 0 ? PQ0_f : (a0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_b0_f = (b0 == 0 ? PQ0_f : (b0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_c0_f = (c0 == 0 ? PQ0_f : (c0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_c1_f = (c1 == 0 ? PQ0_f : (c1 == 1 ? PQ1_f : PQ2_f));
        const float PQ_d0_f = (d0 == 0 ? PQ0_f : (d0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_d1_f = (d1 == 0 ? PQ0_f : (d1 == 1 ? PQ1_f : PQ2_f));

        const auto r2_PQ_f = PQ0_f * PQ0_f + PQ1_f * PQ1_f + PQ2_f * PQ2_f;

        const auto Lambda_f = sqrtf(4.0f * S1_f * S2_f * MATH_CONST_INV_PI_F * inv_S4_f);

        float F6_t_f[7];

        gpu::computeBoysFunction_f(F6_t_f, S1_f * S2_f * inv_S4_f * r2_PQ_f, 6, boys_func_table_f, boys_func_ft_f);

        const auto QC_0_f = (a_l_f * inv_S2_f) * (r_l_c0_f - r_k_c0_f);
        const auto QC_1_f = (a_l_f * inv_S2_f) * (r_l_c1_f - r_k_c1_f);
        const auto QD_0_f = (-a_k_f * inv_S2_f) * (r_l_d0_f - r_k_d0_f);
        const auto QD_1_f = (-a_k_f * inv_S2_f) * (r_l_d1_f - r_k_d1_f);

        const float eri_ijkl_f = Lambda_f * S_ij_00_f * S_kl_00_f * (

                + F6_t_f[0] * (

                    0.25f * inv_S1_f * inv_S2_f * (
                        +QC_0_f*QD_0_f*delta_f[a0][b0]*delta_f[c1][d1]
                        +delta_f[a0][b0]*(QC_1_f*(QC_0_f*delta_f[d0][d1] + QD_0_f*delta_f[c0][d1] + QD_1_f*delta_f[c0][d0]) + QD_1_f*(QC_0_f*delta_f[c1][d0] + QD_0_f*delta_f[c0][c1]))
                    )

                    + 0.25f * inv_S2_f * inv_S2_f * (
                        +PA_0_f*PB_0_f*(delta_f[c0][c1]*delta_f[d0][d1] + delta_f[c0][d0]*delta_f[c1][d1] + delta_f[c0][d1]*delta_f[c1][d0])
                    )

                    + 0.5f * inv_S1_f * (
                        +QC_0_f*QC_1_f*QD_0_f*QD_1_f*delta_f[a0][b0]
                    )

                    + 0.5f * inv_S2_f * (
                        +PA_0_f*PB_0_f*QC_0_f*QC_1_f*delta_f[d0][d1]
                        +PA_0_f*PB_0_f*(QD_0_f*(QC_0_f*delta_f[c1][d1] + QC_1_f*delta_f[c0][d1] + QD_1_f*delta_f[c0][c1]) + QD_1_f*(QC_0_f*delta_f[c1][d0] + QC_1_f*delta_f[c0][d0]))
                    )

                    + (
                        +PA_0_f*PB_0_f*QC_0_f*QC_1_f*QD_0_f*QD_1_f
                    )

                    + 0.125f * inv_S1_f * inv_S2_f * inv_S2_f * (
                        +delta_f[a0][b0]*(delta_f[c0][c1]*delta_f[d0][d1] + delta_f[c0][d0]*delta_f[c1][d1] + delta_f[c0][d1]*delta_f[c1][d0])
                    )

                )

                + F6_t_f[1] * (

                    (-0.125f) * inv_S1_f * inv_S2_f * inv_S4_f * (
                        +delta_f[a0][b0]*(delta_f[c0][c1]*delta_f[d0][d1] + delta_f[c0][d0]*delta_f[c1][d1] + delta_f[c0][d1]*delta_f[c1][d0])
                    )

                    + (-0.25f) * inv_S2_f * inv_S2_f * inv_S4_f * (
                        +delta_f[a0][b0]*(delta_f[c0][c1]*delta_f[d0][d1] + delta_f[c0][d0]*delta_f[c1][d1] + delta_f[c0][d1]*delta_f[c1][d0])
                    )

                    + (-0.5f) * S1_f * inv_S2_f * inv_S2_f * inv_S4_f * (
                        +PA_0_f*PB_0_f*(delta_f[c0][c1]*delta_f[d0][d1] + delta_f[c0][d0]*delta_f[c1][d1] + delta_f[c0][d1]*delta_f[c1][d0])
                    )

                    + (-0.25f) * inv_S1_f * inv_S4_f * (
                        +QC_0_f*QC_1_f*delta_f[a0][b0]*delta_f[d0][d1]
                        +delta_f[a0][b0]*(QD_0_f*(QC_0_f*delta_f[c1][d1] + QC_1_f*delta_f[c0][d1] + QD_1_f*delta_f[c0][c1]) + QD_1_f*(QC_0_f*delta_f[c1][d0] + QC_1_f*delta_f[c0][d0]))
                    )

                    + 0.25f * inv_S2_f * inv_S4_f * (
                        +PA_0_f*QD_0_f*(delta_f[b0][c0]*delta_f[c1][d1] + delta_f[b0][c1]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][c1])
                        +PA_0_f*(QC_0_f*(delta_f[b0][c1]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c1][d1] + delta_f[b0][d1]*delta_f[c1][d0]) + QC_1_f*(delta_f[b0][c0]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][d0]) + QD_1_f*(delta_f[b0][c0]*delta_f[c1][d0] + delta_f[b0][c1]*delta_f[c0][d0] + delta_f[b0][d0]*delta_f[c0][c1])) + PB_0_f*(QC_0_f*(delta_f[a0][c1]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[c1][d1] + delta_f[a0][d1]*delta_f[c1][d0]) + QC_1_f*(delta_f[a0][c0]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[c0][d0]) + QD_0_f*(delta_f[a0][c0]*delta_f[c1][d1] + delta_f[a0][c1]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[c0][c1]) + QD_1_f*(delta_f[a0][c0]*delta_f[c1][d0] + delta_f[a0][c1]*delta_f[c0][d0] + delta_f[a0][d0]*delta_f[c0][c1]))
                        +(PA_0_f*PQ_b0_f + PB_0_f*PQ_a0_f)*(delta_f[c0][c1]*delta_f[d0][d1] + delta_f[c0][d0]*delta_f[c1][d1] + delta_f[c0][d1]*delta_f[c1][d0])
                        +delta_f[a0][b0]*(-QC_0_f*(PQ_c1_f*delta_f[d0][d1] + PQ_d0_f*delta_f[c1][d1] + PQ_d1_f*delta_f[c1][d0]) - QC_1_f*(PQ_d0_f*delta_f[c0][d1] + PQ_d1_f*delta_f[c0][d0] + delta_f[d0][d1]*(PQ_c0_f + QC_0_f)) - QD_0_f*(PQ_d1_f*delta_f[c0][c1] + delta_f[c0][d1]*(PQ_c1_f + QC_1_f) + delta_f[c1][d1]*(PQ_c0_f + QC_0_f)) - QD_1_f*(delta_f[c0][c1]*(PQ_d0_f + QD_0_f) + delta_f[c0][d0]*(PQ_c1_f + QC_1_f) + delta_f[c1][d0]*(PQ_c0_f + QC_0_f)))
                    )

                    + (-0.5f) * S1_f * inv_S2_f * inv_S4_f * (
                        +PA_0_f*PB_0_f*delta_f[d0][d1]*(PQ_c1_f*QC_0_f + QC_1_f*(PQ_c0_f + QC_0_f))
                        +PA_0_f*PB_0_f*(QC_0_f*(PQ_d0_f*delta_f[c1][d1] + PQ_d1_f*delta_f[c1][d0]) + QC_1_f*(PQ_d0_f*delta_f[c0][d1] + PQ_d1_f*delta_f[c0][d0]) + QD_0_f*(PQ_d1_f*delta_f[c0][c1] + delta_f[c0][d1]*(PQ_c1_f + QC_1_f) + delta_f[c1][d1]*(PQ_c0_f + QC_0_f)) + QD_1_f*(delta_f[c0][c1]*(PQ_d0_f + QD_0_f) + delta_f[c0][d0]*(PQ_c1_f + QC_1_f) + delta_f[c1][d0]*(PQ_c0_f + QC_0_f)))
                    )

                    + (-0.5f) * S2_f * inv_S1_f * inv_S4_f * (
                        +QC_0_f*QC_1_f*QD_0_f*QD_1_f*delta_f[a0][b0]
                    )

                    + 0.5f * inv_S4_f * (
                        +QC_0_f*QD_0_f*delta_f[c1][d1]*(PA_0_f*PQ_b0_f + PB_0_f*PQ_a0_f)
                        +(PA_0_f*PQ_b0_f + PB_0_f*PQ_a0_f)*(QC_1_f*(QC_0_f*delta_f[d0][d1] + QD_0_f*delta_f[c0][d1] + QD_1_f*delta_f[c0][d0]) + QD_1_f*(QC_0_f*delta_f[c1][d0] + QD_0_f*delta_f[c0][c1]))
                        +QC_0_f*QC_1_f*(PA_0_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0]) + PB_0_f*(QD_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][d0])) + QD_0_f*QD_1_f*(PA_0_f*(QC_0_f*delta_f[b0][c1] + QC_1_f*delta_f[b0][c0]) + PB_0_f*(QC_0_f*delta_f[a0][c1] + QC_1_f*delta_f[a0][c0]))
                        -delta_f[a0][b0]*(QC_0_f*QD_1_f*(PQ_c1_f*QD_0_f + PQ_d0_f*QC_1_f) + QC_1_f*QD_0_f*(PQ_c0_f*QD_1_f + PQ_d1_f*QC_0_f))
                    )

                    + S1_f * inv_S4_f * (
                        -PA_0_f*PB_0_f*(QC_0_f*QD_1_f*(PQ_c1_f*QD_0_f + PQ_d0_f*QC_1_f) + QC_1_f*QD_0_f*(PQ_c0_f*QD_1_f + PQ_d1_f*QC_0_f))
                    )

                    + S2_f * inv_S4_f * (
                        +QC_0_f*QC_1_f*QD_0_f*QD_1_f*(PA_0_f*PQ_b0_f + PB_0_f*PQ_a0_f)
                    )

                )

                + F6_t_f[2] * (

                    0.25f * S1_f * S1_f * inv_S2_f * inv_S2_f * inv_S4_f * inv_S4_f * (
                        +PA_0_f*PB_0_f*(delta_f[c0][c1]*delta_f[d0][d1] + delta_f[c0][d0]*delta_f[c1][d1] + delta_f[c0][d1]*delta_f[c1][d0])
                    )

                    + 0.25f * S1_f * inv_S2_f * inv_S4_f * inv_S4_f * (
                        +(-2*PA_0_f*PQ_b0_f - 2*PB_0_f*PQ_a0_f)*(delta_f[c0][c1]*delta_f[d0][d1] + delta_f[c0][d0]*delta_f[c1][d1] + delta_f[c0][d1]*delta_f[c1][d0]) + (-PA_0_f*PQ_c0_f - PA_0_f*QC_0_f)*(delta_f[b0][c1]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c1][d1] + delta_f[b0][d1]*delta_f[c1][d0]) + (-PA_0_f*PQ_c1_f - PA_0_f*QC_1_f)*(delta_f[b0][c0]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][d0]) + (-PA_0_f*PQ_d0_f - PA_0_f*QD_0_f)*(delta_f[b0][c0]*delta_f[c1][d1] + delta_f[b0][c1]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][c1]) + (-PA_0_f*PQ_d1_f - PA_0_f*QD_1_f)*(delta_f[b0][c0]*delta_f[c1][d0] + delta_f[b0][c1]*delta_f[c0][d0] + delta_f[b0][d0]*delta_f[c0][c1]) + (-PB_0_f*PQ_c0_f - PB_0_f*QC_0_f)*(delta_f[a0][c1]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[c1][d1] + delta_f[a0][d1]*delta_f[c1][d0]) + (-PB_0_f*PQ_c1_f - PB_0_f*QC_1_f)*(delta_f[a0][c0]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[c0][d0]) + (-PB_0_f*PQ_d0_f - PB_0_f*QD_0_f)*(delta_f[a0][c0]*delta_f[c1][d1] + delta_f[a0][c1]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[c0][c1]) + (-PB_0_f*PQ_d1_f - PB_0_f*QD_1_f)*(delta_f[a0][c0]*delta_f[c1][d0] + delta_f[a0][c1]*delta_f[c0][d0] + delta_f[a0][d0]*delta_f[c0][c1])
                        +delta_f[a0][b0]*(PQ_c0_f*(delta_f[c1][d0]*(PQ_d1_f + QD_1_f) + delta_f[c1][d1]*(PQ_d0_f + QD_0_f) + delta_f[d0][d1]*(PQ_c1_f + QC_1_f)) + PQ_c1_f*(QC_0_f*delta_f[d0][d1] + delta_f[c0][d0]*(PQ_d1_f + QD_1_f) + delta_f[c0][d1]*(PQ_d0_f + QD_0_f)) + PQ_d0_f*(QC_0_f*delta_f[c1][d1] + QC_1_f*delta_f[c0][d1] + delta_f[c0][c1]*(PQ_d1_f + QD_1_f)) + PQ_d1_f*(QC_0_f*delta_f[c1][d0] + QC_1_f*delta_f[c0][d0] + QD_0_f*delta_f[c0][c1]))
                    )

                    + 0.25f * inv_S4_f * inv_S4_f * (
                        +PQ_a0_f*QD_0_f*(delta_f[b0][c0]*delta_f[c1][d1] + delta_f[b0][c1]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][c1])
                        +PQ_a0_f*(QC_0_f*(delta_f[b0][c1]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c1][d1] + delta_f[b0][d1]*delta_f[c1][d0]) + QC_1_f*(delta_f[b0][c0]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][d0]) + QD_1_f*(delta_f[b0][c0]*delta_f[c1][d0] + delta_f[b0][c1]*delta_f[c0][d0] + delta_f[b0][d0]*delta_f[c0][c1])) + PQ_b0_f*(QC_0_f*(delta_f[a0][c1]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[c1][d1] + delta_f[a0][d1]*delta_f[c1][d0]) + QC_1_f*(delta_f[a0][c0]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[c0][d0]) + QD_0_f*(delta_f[a0][c0]*delta_f[c1][d1] + delta_f[a0][c1]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[c0][c1]) + QD_1_f*(delta_f[a0][c0]*delta_f[c1][d0] + delta_f[a0][c1]*delta_f[c0][d0] + delta_f[a0][d0]*delta_f[c0][c1]))
                        +delta_f[a0][b0]*(QC_0_f*(PQ_c1_f*delta_f[d0][d1] + PQ_d0_f*delta_f[c1][d1] + PQ_d1_f*delta_f[c1][d0]) + QC_1_f*(PQ_d0_f*delta_f[c0][d1] + PQ_d1_f*delta_f[c0][d0] + delta_f[d0][d1]*(PQ_c0_f + QC_0_f)) + QD_0_f*(PQ_d1_f*delta_f[c0][c1] + delta_f[c0][d1]*(PQ_c1_f + QC_1_f) + delta_f[c1][d1]*(PQ_c0_f + QC_0_f)) + QD_1_f*(delta_f[c0][c1]*(PQ_d0_f + QD_0_f) + delta_f[c0][d0]*(PQ_c1_f + QC_1_f) + delta_f[c1][d0]*(PQ_c0_f + QC_0_f)))
                        +QC_0_f*(QC_1_f*(delta_f[a0][d0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][d0]) + QD_0_f*(delta_f[a0][c1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][c1]) + QD_1_f*(delta_f[a0][c1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][c1])) + QC_1_f*(QD_0_f*(delta_f[a0][c0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][c0]) + QD_1_f*(delta_f[a0][c0]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][c0])) + QD_0_f*QD_1_f*(delta_f[a0][c0]*delta_f[b0][c1] + delta_f[a0][c1]*delta_f[b0][c0])
                        +PQ_a0_f*PQ_b0_f*(delta_f[c0][c1]*delta_f[d0][d1] + delta_f[c0][d0]*delta_f[c1][d1] + delta_f[c0][d1]*delta_f[c1][d0])
                    )

                    + 0.5f * S1_f * S1_f * inv_S2_f * inv_S4_f * inv_S4_f * (
                        +PA_0_f*PB_0_f*delta_f[d0][d1]*(PQ_c0_f*(PQ_c1_f + QC_1_f) + PQ_c1_f*QC_0_f)
                        +PA_0_f*PB_0_f*(PQ_d1_f*(QC_0_f*delta_f[c1][d0] + QC_1_f*delta_f[c0][d0] + QD_0_f*delta_f[c0][c1]) + delta_f[c0][d1]*(PQ_c1_f*(PQ_d0_f + QD_0_f) + PQ_d0_f*QC_1_f) + delta_f[c1][d1]*(PQ_c0_f*(PQ_d0_f + QD_0_f) + PQ_d0_f*QC_0_f) + (PQ_d1_f + QD_1_f)*(PQ_c0_f*delta_f[c1][d0] + PQ_c1_f*delta_f[c0][d0] + PQ_d0_f*delta_f[c0][c1]))
                    )

                    + 0.5f * S1_f * inv_S4_f * inv_S4_f * (
                        -delta_f[d0][d1]*(PA_0_f*PQ_b0_f + PB_0_f*PQ_a0_f)*(PQ_c1_f*QC_0_f + QC_1_f*(PQ_c0_f + QC_0_f))
                        -(PA_0_f*PQ_b0_f + PB_0_f*PQ_a0_f)*(PQ_d0_f*(QC_0_f*delta_f[c1][d1] + QC_1_f*delta_f[c0][d1]) + PQ_d1_f*(QC_0_f*delta_f[c1][d0] + QC_1_f*delta_f[c0][d0] + QD_0_f*delta_f[c0][c1]) + QD_0_f*(delta_f[c0][d1]*(PQ_c1_f + QC_1_f) + delta_f[c1][d1]*(PQ_c0_f + QC_0_f)) + QD_1_f*(delta_f[c0][c1]*(PQ_d0_f + QD_0_f) + delta_f[c0][d0]*(PQ_c1_f + QC_1_f) + delta_f[c1][d0]*(PQ_c0_f + QC_0_f)))
                        -PA_0_f*(QC_0_f*(PQ_c1_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0]) + PQ_d0_f*(QC_1_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][c1]) + PQ_d1_f*(QC_1_f*delta_f[b0][d0] + QD_0_f*delta_f[b0][c1])) + QC_1_f*(PQ_c0_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0]) + delta_f[b0][c0]*(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f))) - PB_0_f*(QC_0_f*(PQ_c1_f*(QD_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][d0]) + PQ_d0_f*(QC_1_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][c1]) + PQ_d1_f*(QC_1_f*delta_f[a0][d0] + QD_0_f*delta_f[a0][c1])) + QC_1_f*(PQ_c0_f*(QD_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][d0]) + delta_f[a0][c0]*(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f))) - QD_0_f*QD_1_f*(PA_0_f*(PQ_c0_f*delta_f[b0][c1] + PQ_c1_f*delta_f[b0][c0]) + PB_0_f*(PQ_c0_f*delta_f[a0][c1] + PQ_c1_f*delta_f[a0][c0]))
                        +delta_f[a0][b0]*(PQ_c0_f*QD_0_f*(PQ_c1_f*QD_1_f + PQ_d1_f*QC_1_f) + PQ_c1_f*QC_0_f*(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f) + PQ_d0_f*QC_1_f*(PQ_c0_f*QD_1_f + PQ_d1_f*QC_0_f))
                    )

                    + 0.5f * S2_f * inv_S4_f * inv_S4_f * (
                        +QC_0_f*QD_0_f*(PQ_a0_f*(PQ_b0_f*delta_f[c1][d1] + QC_1_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][c1]) + PQ_b0_f*(QC_1_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][c1])) + QC_1_f*QD_1_f*(PQ_a0_f*(QC_0_f*delta_f[b0][d0] + QD_0_f*delta_f[b0][c0]) + PQ_b0_f*(QC_0_f*delta_f[a0][d0] + QD_0_f*delta_f[a0][c0]))
                        +PQ_a0_f*PQ_b0_f*(QC_1_f*(QC_0_f*delta_f[d0][d1] + QD_0_f*delta_f[c0][d1] + QD_1_f*delta_f[c0][d0]) + QD_1_f*(QC_0_f*delta_f[c1][d0] + QD_0_f*delta_f[c0][c1]))
                        +delta_f[a0][b0]*(QC_0_f*QD_1_f*(PQ_c1_f*QD_0_f + PQ_d0_f*QC_1_f) + QC_1_f*QD_0_f*(PQ_c0_f*QD_1_f + PQ_d1_f*QC_0_f))
                    )

                    + S1_f * S1_f * inv_S4_f * inv_S4_f * (
                        +PA_0_f*PB_0_f*(PQ_c0_f*QD_0_f*(PQ_c1_f*QD_1_f + PQ_d1_f*QC_1_f) + PQ_c1_f*QC_0_f*(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f) + PQ_d0_f*QC_1_f*(PQ_c0_f*QD_1_f + PQ_d1_f*QC_0_f))
                    )

                    + S1_f * S2_f * inv_S4_f * inv_S4_f * (
                        -(PA_0_f*PQ_b0_f + PB_0_f*PQ_a0_f)*(QC_0_f*QD_1_f*(PQ_c1_f*QD_0_f + PQ_d0_f*QC_1_f) + QC_1_f*QD_0_f*(PQ_c0_f*QD_1_f + PQ_d1_f*QC_0_f))
                    )

                    + S2_f * S2_f * inv_S4_f * inv_S4_f * (
                        +PQ_a0_f*PQ_b0_f*QC_0_f*QC_1_f*QD_0_f*QD_1_f
                    )

                    + 0.125f * S1_f * inv_S2_f * inv_S2_f * inv_S4_f * inv_S4_f * (
                        +delta_f[a0][b0]*(delta_f[c0][c1]*delta_f[d0][d1] + delta_f[c0][d0]*delta_f[c1][d1] + delta_f[c0][d1]*delta_f[c1][d0])
                    )

                    + 0.125f * inv_S2_f * inv_S4_f * inv_S4_f * (
                        +2.0f*delta_f[a0][b0]*(delta_f[c0][c1]*delta_f[d0][d1] + delta_f[c0][d0]*delta_f[c1][d1] + delta_f[c0][d1]*delta_f[c1][d0])
                        +delta_f[a0][c0]*(delta_f[b0][c1]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c1][d1] + delta_f[b0][d1]*delta_f[c1][d0]) + delta_f[a0][c1]*(delta_f[b0][c0]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][d0]) + delta_f[a0][d0]*(delta_f[b0][c0]*delta_f[c1][d1] + delta_f[b0][c1]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][c1]) + delta_f[a0][d1]*(delta_f[b0][c0]*delta_f[c1][d0] + delta_f[b0][c1]*delta_f[c0][d0] + delta_f[b0][d0]*delta_f[c0][c1])
                    )

                )

                + F6_t_f[3] * (

                    (-0.125f) * S1_f * inv_S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +delta_f[a0][b0]*(delta_f[c0][c1]*delta_f[d0][d1] + delta_f[c0][d0]*delta_f[c1][d1] + delta_f[c0][d1]*delta_f[c1][d0]) + delta_f[a0][c0]*(delta_f[b0][c1]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c1][d1] + delta_f[b0][d1]*delta_f[c1][d0]) + delta_f[a0][c1]*(delta_f[b0][c0]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][d0]) + delta_f[a0][d0]*(delta_f[b0][c0]*delta_f[c1][d1] + delta_f[b0][c1]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][c1]) + delta_f[a0][d1]*(delta_f[b0][c0]*delta_f[c1][d0] + delta_f[b0][c1]*delta_f[c0][d0] + delta_f[b0][d0]*delta_f[c0][c1])
                    )

                    + 0.25f * S1_f * S1_f * inv_S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +(PA_0_f*PQ_b0_f + PB_0_f*PQ_a0_f)*(delta_f[c0][c1]*delta_f[d0][d1] + delta_f[c0][d0]*delta_f[c1][d1] + delta_f[c0][d1]*delta_f[c1][d0])
                        +PA_0_f*(PQ_c0_f*(delta_f[b0][c1]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c1][d1] + delta_f[b0][d1]*delta_f[c1][d0]) + PQ_c1_f*(delta_f[b0][c0]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][d0]) + PQ_d0_f*(delta_f[b0][c0]*delta_f[c1][d1] + delta_f[b0][c1]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][c1]) + PQ_d1_f*(delta_f[b0][c0]*delta_f[c1][d0] + delta_f[b0][c1]*delta_f[c0][d0] + delta_f[b0][d0]*delta_f[c0][c1])) + PB_0_f*(PQ_c0_f*(delta_f[a0][c1]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[c1][d1] + delta_f[a0][d1]*delta_f[c1][d0]) + PQ_c1_f*(delta_f[a0][c0]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[c0][d0]) + PQ_d0_f*(delta_f[a0][c0]*delta_f[c1][d1] + delta_f[a0][c1]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[c0][c1]) + PQ_d1_f*(delta_f[a0][c0]*delta_f[c1][d0] + delta_f[a0][c1]*delta_f[c0][d0] + delta_f[a0][d0]*delta_f[c0][c1]))
                        +delta_f[a0][b0]*(-PQ_c0_f*(PQ_c1_f*delta_f[d0][d1] + PQ_d0_f*delta_f[c1][d1] + PQ_d1_f*delta_f[c1][d0]) - PQ_c1_f*(PQ_d0_f*delta_f[c0][d1] + PQ_d1_f*delta_f[c0][d0]) - PQ_d0_f*PQ_d1_f*delta_f[c0][c1])
                    )

                    + 0.25f * S1_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        -2*PQ_a0_f*PQ_b0_f*(delta_f[c0][c1]*delta_f[d0][d1] + delta_f[c0][d0]*delta_f[c1][d1] + delta_f[c0][d1]*delta_f[c1][d0])
                        -PQ_a0_f*(delta_f[b0][c0]*(delta_f[c1][d0]*(PQ_d1_f + QD_1_f) + delta_f[c1][d1]*(PQ_d0_f + QD_0_f) + delta_f[d0][d1]*(PQ_c1_f + QC_1_f)) + delta_f[b0][c1]*(delta_f[c0][d0]*(PQ_d1_f + QD_1_f) + delta_f[c0][d1]*(PQ_d0_f + QD_0_f) + delta_f[d0][d1]*(PQ_c0_f + QC_0_f)) + delta_f[b0][d0]*(delta_f[c0][c1]*(PQ_d1_f + QD_1_f) + delta_f[c0][d1]*(PQ_c1_f + QC_1_f) + delta_f[c1][d1]*(PQ_c0_f + QC_0_f)) + delta_f[b0][d1]*(delta_f[c0][c1]*(PQ_d0_f + QD_0_f) + delta_f[c0][d0]*(PQ_c1_f + QC_1_f) + delta_f[c1][d0]*(PQ_c0_f + QC_0_f))) - PQ_b0_f*(delta_f[a0][c0]*(delta_f[c1][d0]*(PQ_d1_f + QD_1_f) + delta_f[c1][d1]*(PQ_d0_f + QD_0_f) + delta_f[d0][d1]*(PQ_c1_f + QC_1_f)) + delta_f[a0][c1]*(delta_f[c0][d0]*(PQ_d1_f + QD_1_f) + delta_f[c0][d1]*(PQ_d0_f + QD_0_f) + delta_f[d0][d1]*(PQ_c0_f + QC_0_f)) + delta_f[a0][d0]*(delta_f[c0][c1]*(PQ_d1_f + QD_1_f) + delta_f[c0][d1]*(PQ_c1_f + QC_1_f) + delta_f[c1][d1]*(PQ_c0_f + QC_0_f)) + delta_f[a0][d1]*(delta_f[c0][c1]*(PQ_d0_f + QD_0_f) + delta_f[c0][d0]*(PQ_c1_f + QC_1_f) + delta_f[c1][d0]*(PQ_c0_f + QC_0_f)))
                        +delta_f[a0][b0]*(-PQ_c0_f*(delta_f[c1][d0]*(PQ_d1_f + QD_1_f) + delta_f[c1][d1]*(PQ_d0_f + QD_0_f) + delta_f[d0][d1]*(PQ_c1_f + QC_1_f)) - PQ_c1_f*(QC_0_f*delta_f[d0][d1] + delta_f[c0][d0]*(PQ_d1_f + QD_1_f) + delta_f[c0][d1]*(PQ_d0_f + QD_0_f)) - PQ_d0_f*(QC_0_f*delta_f[c1][d1] + QC_1_f*delta_f[c0][d1] + delta_f[c0][c1]*(PQ_d1_f + QD_1_f)) - PQ_d1_f*(QC_0_f*delta_f[c1][d0] + QC_1_f*delta_f[c0][d0] + QD_0_f*delta_f[c0][c1]))
                        +(-PQ_c0_f*QC_1_f - PQ_c1_f*QC_0_f)*(delta_f[a0][d0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][d0]) + (-PQ_c0_f*QD_0_f - PQ_d0_f*QC_0_f)*(delta_f[a0][c1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][c1]) + (-PQ_c0_f*QD_1_f - PQ_d1_f*QC_0_f)*(delta_f[a0][c1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][c1]) + (-PQ_c1_f*QD_0_f - PQ_d0_f*QC_1_f)*(delta_f[a0][c0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][c0]) + (-PQ_c1_f*QD_1_f - PQ_d1_f*QC_1_f)*(delta_f[a0][c0]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][c0]) + (-PQ_d0_f*QD_1_f - PQ_d1_f*QD_0_f)*(delta_f[a0][c0]*delta_f[b0][c1] + delta_f[a0][c1]*delta_f[b0][c0])
                    )

                    + (-0.5f) * S1_f * S1_f * S1_f * inv_S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PA_0_f*PB_0_f*PQ_c0_f*PQ_c1_f*delta_f[d0][d1]
                        +PA_0_f*PB_0_f*(PQ_d0_f*(PQ_c0_f*delta_f[c1][d1] + PQ_c1_f*delta_f[c0][d1] + PQ_d1_f*delta_f[c0][c1]) + PQ_d1_f*(PQ_c0_f*delta_f[c1][d0] + PQ_c1_f*delta_f[c0][d0]))
                    )

                    + 0.5f * S1_f * S1_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +delta_f[c1][d1]*(PA_0_f*PQ_b0_f + PB_0_f*PQ_a0_f)*(PQ_c0_f*(PQ_d0_f + QD_0_f) + PQ_d0_f*QC_0_f)
                        +(PA_0_f*PQ_b0_f + PB_0_f*PQ_a0_f)*(PQ_d1_f*(QC_0_f*delta_f[c1][d0] + QC_1_f*delta_f[c0][d0] + QD_0_f*delta_f[c0][c1]) + delta_f[c0][d1]*(PQ_c1_f*(PQ_d0_f + QD_0_f) + PQ_d0_f*QC_1_f) + delta_f[d0][d1]*(PQ_c0_f*(PQ_c1_f + QC_1_f) + PQ_c1_f*QC_0_f) + (PQ_d1_f + QD_1_f)*(PQ_c0_f*delta_f[c1][d0] + PQ_c1_f*delta_f[c0][d0] + PQ_d0_f*delta_f[c0][c1]))
                        +PA_0_f*(PQ_c0_f*(PQ_c1_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0]) + PQ_d0_f*(QC_1_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][c1]) + PQ_d1_f*(QC_1_f*delta_f[b0][d0] + QD_0_f*delta_f[b0][c1])) + PQ_c1_f*(PQ_d0_f*(QC_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][c0]) + PQ_d1_f*(QC_0_f*delta_f[b0][d0] + QD_0_f*delta_f[b0][c0]))) + PB_0_f*(PQ_c0_f*(PQ_c1_f*(QD_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][d0]) + PQ_d0_f*(QC_1_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][c1]) + PQ_d1_f*(QC_1_f*delta_f[a0][d0] + QD_0_f*delta_f[a0][c1])) + PQ_c1_f*(PQ_d0_f*(QC_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][c0]) + PQ_d1_f*(QC_0_f*delta_f[a0][d0] + QD_0_f*delta_f[a0][c0]))) + PQ_d0_f*PQ_d1_f*(PA_0_f*(QC_0_f*delta_f[b0][c1] + QC_1_f*delta_f[b0][c0]) + PB_0_f*(QC_0_f*delta_f[a0][c1] + QC_1_f*delta_f[a0][c0]))
                        -delta_f[a0][b0]*(PQ_c0_f*PQ_c1_f*(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f) + PQ_d0_f*PQ_d1_f*(PQ_c0_f*QC_1_f + PQ_c1_f*QC_0_f))
                    )

                    + (-0.5f) * S1_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_a0_f*PQ_b0_f*delta_f[d0][d1]*(PQ_c1_f*QC_0_f + QC_1_f*(PQ_c0_f + QC_0_f))
                        +PQ_a0_f*PQ_b0_f*(QC_0_f*(PQ_d0_f*delta_f[c1][d1] + PQ_d1_f*delta_f[c1][d0]) + QC_1_f*(PQ_d0_f*delta_f[c0][d1] + PQ_d1_f*delta_f[c0][d0]) + QD_0_f*(PQ_d1_f*delta_f[c0][c1] + delta_f[c0][d1]*(PQ_c1_f + QC_1_f) + delta_f[c1][d1]*(PQ_c0_f + QC_0_f)) + QD_1_f*(delta_f[c0][c1]*(PQ_d0_f + QD_0_f) + delta_f[c0][d0]*(PQ_c1_f + QC_1_f) + delta_f[c1][d0]*(PQ_c0_f + QC_0_f)))
                        +PQ_a0_f*(QC_0_f*(PQ_c1_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0]) + PQ_d0_f*(QC_1_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][c1]) + PQ_d1_f*(QC_1_f*delta_f[b0][d0] + QD_0_f*delta_f[b0][c1])) + QC_1_f*(PQ_c0_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0]) + delta_f[b0][c0]*(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f))) + PQ_b0_f*(QC_0_f*(PQ_c1_f*(QD_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][d0]) + PQ_d0_f*(QC_1_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][c1]) + PQ_d1_f*(QC_1_f*delta_f[a0][d0] + QD_0_f*delta_f[a0][c1])) + QC_1_f*(PQ_c0_f*(QD_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][d0]) + delta_f[a0][c0]*(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f))) + QD_0_f*QD_1_f*(PQ_a0_f*(PQ_c0_f*delta_f[b0][c1] + PQ_c1_f*delta_f[b0][c0]) + PQ_b0_f*(PQ_c0_f*delta_f[a0][c1] + PQ_c1_f*delta_f[a0][c0]))
                        +delta_f[a0][b0]*(PQ_c0_f*QD_0_f*(PQ_c1_f*QD_1_f + PQ_d1_f*QC_1_f) + PQ_c1_f*QC_0_f*(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f) + PQ_d0_f*QC_1_f*(PQ_c0_f*QD_1_f + PQ_d1_f*QC_0_f))
                    )

                    + S1_f * S1_f * S1_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        -PA_0_f*PB_0_f*(PQ_c0_f*PQ_c1_f*(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f) + PQ_d0_f*PQ_d1_f*(PQ_c0_f*QC_1_f + PQ_c1_f*QC_0_f))
                    )

                    + S1_f * S1_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +(PA_0_f*PQ_b0_f + PB_0_f*PQ_a0_f)*(PQ_c0_f*QD_0_f*(PQ_c1_f*QD_1_f + PQ_d1_f*QC_1_f) + PQ_c1_f*QC_0_f*(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f) + PQ_d0_f*QC_1_f*(PQ_c0_f*QD_1_f + PQ_d1_f*QC_0_f))
                    )

                    + S1_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        -PQ_a0_f*PQ_b0_f*(QC_0_f*QD_1_f*(PQ_c1_f*QD_0_f + PQ_d0_f*QC_1_f) + QC_1_f*QD_0_f*(PQ_c0_f*QD_1_f + PQ_d1_f*QC_0_f))
                    )

                )

                + F6_t_f[4] * (

                    0.5f * S1_f * S1_f * S1_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_c0_f*PQ_c1_f*delta_f[d0][d1]*(-PA_0_f*PQ_b0_f - PB_0_f*PQ_a0_f)
                        -(PA_0_f*PQ_b0_f + PB_0_f*PQ_a0_f)*(PQ_d0_f*(PQ_c0_f*delta_f[c1][d1] + PQ_c1_f*delta_f[c0][d1] + PQ_d1_f*delta_f[c0][c1]) + PQ_d1_f*(PQ_c0_f*delta_f[c1][d0] + PQ_c1_f*delta_f[c0][d0]))
                        -PQ_c0_f*PQ_c1_f*(PA_0_f*(PQ_d0_f*delta_f[b0][d1] + PQ_d1_f*delta_f[b0][d0]) + PB_0_f*(PQ_d0_f*delta_f[a0][d1] + PQ_d1_f*delta_f[a0][d0])) - PQ_d0_f*PQ_d1_f*(PA_0_f*(PQ_c0_f*delta_f[b0][c1] + PQ_c1_f*delta_f[b0][c0]) + PB_0_f*(PQ_c0_f*delta_f[a0][c1] + PQ_c1_f*delta_f[a0][c0]))
                        +PQ_c0_f*PQ_c1_f*PQ_d0_f*PQ_d1_f*delta_f[a0][b0]
                    )

                    + 0.5f * S1_f * S1_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_c0_f*(PQ_a0_f*(PQ_b0_f*delta_f[c1][d1]*(PQ_d0_f + QD_0_f) + PQ_c1_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0]) + PQ_d0_f*(QC_1_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][c1]) + PQ_d1_f*(QC_1_f*delta_f[b0][d0] + QD_0_f*delta_f[b0][c1])) + PQ_b0_f*(PQ_c1_f*(QD_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][d0]) + PQ_d0_f*(QC_1_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][c1]) + PQ_d1_f*(QC_1_f*delta_f[a0][d0] + QD_0_f*delta_f[a0][c1]))) + PQ_c1_f*(PQ_a0_f*delta_f[b0][c0]*(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f) + PQ_b0_f*(PQ_d0_f*(QC_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][c0]) + PQ_d1_f*QD_0_f*delta_f[a0][c0]) + PQ_d1_f*QC_0_f*(PQ_a0_f*delta_f[b0][d0] + PQ_b0_f*delta_f[a0][d0])) + PQ_d0_f*(PQ_d1_f*QC_1_f*(PQ_a0_f*delta_f[b0][c0] + PQ_b0_f*delta_f[a0][c0]) + QC_0_f*(PQ_a0_f*(PQ_b0_f*delta_f[c1][d1] + PQ_c1_f*delta_f[b0][d1] + PQ_d1_f*delta_f[b0][c1]) + PQ_b0_f*PQ_d1_f*delta_f[a0][c1]))
                        +PQ_a0_f*PQ_b0_f*(PQ_d1_f*(QC_0_f*delta_f[c1][d0] + QC_1_f*delta_f[c0][d0] + QD_0_f*delta_f[c0][c1]) + delta_f[c0][d1]*(PQ_c1_f*(PQ_d0_f + QD_0_f) + PQ_d0_f*QC_1_f) + delta_f[d0][d1]*(PQ_c0_f*(PQ_c1_f + QC_1_f) + PQ_c1_f*QC_0_f) + (PQ_d1_f + QD_1_f)*(PQ_c0_f*delta_f[c1][d0] + PQ_c1_f*delta_f[c0][d0] + PQ_d0_f*delta_f[c0][c1]))
                        +delta_f[a0][b0]*(PQ_c0_f*PQ_c1_f*(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f) + PQ_d0_f*PQ_d1_f*(PQ_c0_f*QC_1_f + PQ_c1_f*QC_0_f))
                    )

                    + S1_f * S1_f * S1_f * S1_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PA_0_f*PB_0_f*PQ_c0_f*PQ_c1_f*PQ_d0_f*PQ_d1_f
                    )

                    + S1_f * S1_f * S1_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        -(PA_0_f*PQ_b0_f + PB_0_f*PQ_a0_f)*(PQ_c0_f*PQ_c1_f*(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f) + PQ_d0_f*PQ_d1_f*(PQ_c0_f*QC_1_f + PQ_c1_f*QC_0_f))
                    )

                    + S1_f * S1_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_a0_f*PQ_b0_f*(PQ_c0_f*QD_0_f*(PQ_c1_f*QD_1_f + PQ_d1_f*QC_1_f) + PQ_c1_f*QC_0_f*(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f) + PQ_d0_f*QC_1_f*(PQ_c0_f*QD_1_f + PQ_d1_f*QC_0_f))
                    )

                    + 0.25f * S1_f * S1_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_a0_f*PQ_b0_f*(delta_f[c0][c1]*delta_f[d0][d1] + delta_f[c0][d0]*delta_f[c1][d1] + delta_f[c0][d1]*delta_f[c1][d0])
                        +PQ_c0_f*(PQ_a0_f*(delta_f[b0][c1]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c1][d1] + delta_f[b0][d1]*delta_f[c1][d0]) + PQ_b0_f*(delta_f[a0][c1]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[c1][d1] + delta_f[a0][d1]*delta_f[c1][d0]) + PQ_c1_f*(delta_f[a0][b0]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][d0]) + PQ_d0_f*(delta_f[a0][b0]*delta_f[c1][d1] + delta_f[a0][c1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][c1]) + PQ_d1_f*(delta_f[a0][b0]*delta_f[c1][d0] + delta_f[a0][c1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][c1])) + PQ_c1_f*(PQ_a0_f*(delta_f[b0][c0]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][d0]) + PQ_b0_f*(delta_f[a0][c0]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[c0][d0]) + PQ_d0_f*(delta_f[a0][b0]*delta_f[c0][d1] + delta_f[a0][c0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][c0]) + PQ_d1_f*(delta_f[a0][b0]*delta_f[c0][d0] + delta_f[a0][c0]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][c0])) + PQ_d0_f*(PQ_a0_f*(delta_f[b0][c0]*delta_f[c1][d1] + delta_f[b0][c1]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][c1]) + PQ_b0_f*(delta_f[a0][c0]*delta_f[c1][d1] + delta_f[a0][c1]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[c0][c1]) + PQ_d1_f*(delta_f[a0][b0]*delta_f[c0][c1] + delta_f[a0][c0]*delta_f[b0][c1] + delta_f[a0][c1]*delta_f[b0][c0])) + PQ_d1_f*(PQ_a0_f*(delta_f[b0][c0]*delta_f[c1][d0] + delta_f[b0][c1]*delta_f[c0][d0] + delta_f[b0][d0]*delta_f[c0][c1]) + PQ_b0_f*(delta_f[a0][c0]*delta_f[c1][d0] + delta_f[a0][c1]*delta_f[c0][d0] + delta_f[a0][d0]*delta_f[c0][c1]))
                    )

                )

                + F6_t_f[5] * (

                    (-0.5f) * S1_f * S1_f * S1_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_a0_f*PQ_b0_f*PQ_c0_f*PQ_c1_f*delta_f[d0][d1]
                        +PQ_d0_f*(PQ_a0_f*(PQ_b0_f*(PQ_c0_f*delta_f[c1][d1] + PQ_c1_f*delta_f[c0][d1] + PQ_d1_f*delta_f[c0][c1]) + PQ_c1_f*PQ_d1_f*delta_f[b0][c0]) + PQ_c0_f*PQ_c1_f*(PQ_a0_f*delta_f[b0][d1] + PQ_b0_f*delta_f[a0][d1] + PQ_d1_f*delta_f[a0][b0])) + PQ_d1_f*(PQ_b0_f*PQ_c1_f*(PQ_a0_f*delta_f[c0][d0] + PQ_c0_f*delta_f[a0][d0] + PQ_d0_f*delta_f[a0][c0]) + PQ_c0_f*(PQ_a0_f*(PQ_b0_f*delta_f[c1][d0] + PQ_c1_f*delta_f[b0][d0] + PQ_d0_f*delta_f[b0][c1]) + PQ_b0_f*PQ_d0_f*delta_f[a0][c1]))
                    )

                    + S1_f * S1_f * S1_f * S1_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_c0_f*PQ_c1_f*PQ_d0_f*PQ_d1_f*(PA_0_f*PQ_b0_f + PB_0_f*PQ_a0_f)
                    )

                    + S1_f * S1_f * S1_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        -PQ_a0_f*PQ_b0_f*(PQ_c0_f*PQ_c1_f*(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f) + PQ_d0_f*PQ_d1_f*(PQ_c0_f*QC_1_f + PQ_c1_f*QC_0_f))
                    )

                )

                + F6_t_f[6] * (

                    S1_f * S1_f * S1_f * S1_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_a0_f*PQ_b0_f*PQ_c0_f*PQ_c1_f*PQ_d0_f*PQ_d1_f
                    )

                )

                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0f) == (k == l ? 1.0f : 2.0f)
        const float D_f = dd_mat_D_f[kl];
        const float sym_f = (k != l) ? 2.0f : 1.0f;
        const float contrib_f = eri_ijkl_f * D_f * sym_f;
        ERIs[threadIdx.y][threadIdx.x] += (double)contrib_f;

    }

    __syncthreads();

    if ((threadIdx.y == 0) && (ij < pp_prim_pair_count_local))
    {
        double J_ij = 0.0;

        for (uint32_t n = 0; n < TILE_DIM; n++)
        {
            J_ij += ERIs[n][threadIdx.x];
        }

        mat_J[ij] += J_ij;
    }
}

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDSD_FP32_auto_s_ast(double*         mat_J,
                       const float*    s_prim_info_f,
                       const uint32_t  s_prim_count,
                       const float*    d_prim_info_f,
                       const uint32_t  d_prim_count,
                       const float*    sd_mat_D_f,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const float*    dd_pair_data_local_f,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* sd_first_inds,
                       const uint32_t* sd_second_inds,
                       const float*    sd_pair_data_f,
                       const uint32_t  sd_prim_pair_count,
                       const float*    boys_func_table_f,
                       const float*    boys_func_ft_f,
                       const uint32_t* prec_cut_ij_tile,
                       const uint32_t* screen_cut_ij_tile)
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ float   delta_f[3][3];

    __shared__ float a_i_f, a_j_f, r_i_f[3], r_j_f[3], S_ij_00_f, S1_f, inv_S1_f;
    __shared__ float PA_0_f, PA_1_f, PB_0_f, PB_1_f;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];
    const uint32_t screen_cut = screen_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta_f[0][0] = 1.0f; delta_f[0][1] = 0.0f; delta_f[0][2] = 0.0f;
        delta_f[1][0] = 0.0f; delta_f[1][1] = 1.0f; delta_f[1][2] = 0.0f;
        delta_f[2][0] = 0.0f; delta_f[2][1] = 0.0f; delta_f[2][2] = 1.0f;

        if (ij < dd_prim_pair_count_local)
        {
            i = dd_first_inds_local[ij];
            j = dd_second_inds_local[ij];

            a_i_f = d_prim_info_f[i / 6 + d_prim_count * 0];

            r_i_f[0] = d_prim_info_f[i / 6 + d_prim_count * 2];
            r_i_f[1] = d_prim_info_f[i / 6 + d_prim_count * 3];
            r_i_f[2] = d_prim_info_f[i / 6 + d_prim_count * 4];

            a_j_f = d_prim_info_f[j / 6 + d_prim_count * 0];

            r_j_f[0] = d_prim_info_f[j / 6 + d_prim_count * 2];
            r_j_f[1] = d_prim_info_f[j / 6 + d_prim_count * 3];
            r_j_f[2] = d_prim_info_f[j / 6 + d_prim_count * 4];

            S1_f = a_i_f + a_j_f;
            inv_S1_f = (float) (1.0 / (double)S1_f);

            S_ij_00_f = dd_pair_data_local_f[ij];

            a0 = d_cart_inds[i % 6][0];
            a1 = d_cart_inds[i % 6][1];
            b0 = d_cart_inds[j % 6][0];
            b1 = d_cart_inds[j % 6][1];

            PA_0_f = (a_j_f  * inv_S1_f) * (r_j_f[a0] - r_i_f[a0]);
            PA_1_f = (a_j_f  * inv_S1_f) * (r_j_f[a1] - r_i_f[a1]);
            PB_0_f = (-a_i_f * inv_S1_f) * (r_j_f[b0] - r_i_f[b0]);
            PB_1_f = (-a_i_f * inv_S1_f) * (r_j_f[b1] - r_i_f[b1]);

        }

    }

    ERIs[threadIdx.y] = 0.0;

    __syncthreads();

    for (uint32_t m = prec_cut; m < screen_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= sd_prim_pair_count))
        {
            break;
        }

        const auto k = sd_first_inds[kl];
        const auto l = sd_second_inds[kl];

        const auto a_k_f = s_prim_info_f[k + s_prim_count * 0];

        const float r_k0_f = s_prim_info_f[k + s_prim_count * 2];
        const float r_k1_f = s_prim_info_f[k + s_prim_count * 3];
        const float r_k2_f = s_prim_info_f[k + s_prim_count * 4];

        const auto a_l_f = d_prim_info_f[l / 6 + d_prim_count * 0];

        const float r_l0_f = d_prim_info_f[l / 6 + d_prim_count * 2];
        const float r_l1_f = d_prim_info_f[l / 6 + d_prim_count * 3];
        const float r_l2_f = d_prim_info_f[l / 6 + d_prim_count * 4];

        const auto S_kl_00_f = sd_pair_data_f[kl];

        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        const float r_l_d0_f = (d0 == 0 ? r_l0_f : (d0 == 1 ? r_l1_f : r_l2_f));
        const float r_l_d1_f = (d1 == 0 ? r_l0_f : (d1 == 1 ? r_l1_f : r_l2_f));

        const float r_k_d0_f = (d0 == 0 ? r_k0_f : (d0 == 1 ? r_k1_f : r_k2_f));
        const float r_k_d1_f = (d1 == 0 ? r_k0_f : (d1 == 1 ? r_k1_f : r_k2_f));

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2_f = a_k_f + a_l_f;

        const auto inv_S2_f = 1.0f / S2_f;
        const auto inv_S4_f = 1.0f / (S1_f + S2_f);

        const float PQ0_f = (a_k_f * r_k0_f + a_l_f * r_l0_f) * inv_S2_f - (a_i_f * r_i_f[0] + a_j_f * r_j_f[0]) * inv_S1_f;
        const float PQ1_f = (a_k_f * r_k1_f + a_l_f * r_l1_f) * inv_S2_f - (a_i_f * r_i_f[1] + a_j_f * r_j_f[1]) * inv_S1_f;
        const float PQ2_f = (a_k_f * r_k2_f + a_l_f * r_l2_f) * inv_S2_f - (a_i_f * r_i_f[2] + a_j_f * r_j_f[2]) * inv_S1_f;

        const float PQ_a0_f = (a0 == 0 ? PQ0_f : (a0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_a1_f = (a1 == 0 ? PQ0_f : (a1 == 1 ? PQ1_f : PQ2_f));
        const float PQ_b0_f = (b0 == 0 ? PQ0_f : (b0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_b1_f = (b1 == 0 ? PQ0_f : (b1 == 1 ? PQ1_f : PQ2_f));
        const float PQ_d0_f = (d0 == 0 ? PQ0_f : (d0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_d1_f = (d1 == 0 ? PQ0_f : (d1 == 1 ? PQ1_f : PQ2_f));

        const auto r2_PQ_f = PQ0_f * PQ0_f + PQ1_f * PQ1_f + PQ2_f * PQ2_f;

        const auto Lambda_f = sqrtf(4.0f * S1_f * S2_f * MATH_CONST_INV_PI_F * inv_S4_f);

        float F6_t_f[7];

        gpu::computeBoysFunction_f(F6_t_f, S1_f * S2_f * inv_S4_f * r2_PQ_f, 6, boys_func_table_f, boys_func_ft_f);

        const auto QD_0_f = (-a_k_f * inv_S2_f) * (r_l_d0_f - r_k_d0_f);
        const auto QD_1_f = (-a_k_f * inv_S2_f) * (r_l_d1_f - r_k_d1_f);

        const float eri_ijkl_f = Lambda_f * S_ij_00_f * S_kl_00_f * (

                + F6_t_f[0] * (

                    0.25f * inv_S1_f * inv_S1_f * (
                        +QD_0_f*QD_1_f*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.25f * inv_S1_f * inv_S2_f * (
                        +PA_0_f*PB_0_f*delta_f[a1][b1]*delta_f[d0][d1]
                        +delta_f[d0][d1]*(PA_1_f*(PA_0_f*delta_f[b0][b1] + PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PB_1_f*(PA_0_f*delta_f[a1][b0] + PB_0_f*delta_f[a0][a1]))
                    )

                    + 0.5f * inv_S1_f * (
                        +PA_0_f*PA_1_f*QD_0_f*QD_1_f*delta_f[b0][b1]
                        +QD_0_f*QD_1_f*(PB_0_f*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) + PB_1_f*(PA_0_f*delta_f[a1][b0] + PA_1_f*delta_f[a0][b0]))
                    )

                    + 0.5f * inv_S2_f * (
                        +PA_0_f*PA_1_f*PB_0_f*PB_1_f*delta_f[d0][d1]
                    )

                    + (
                        +PA_0_f*PA_1_f*PB_0_f*PB_1_f*QD_0_f*QD_1_f
                    )

                    + 0.125f * inv_S1_f * inv_S1_f * inv_S2_f * (
                        +delta_f[d0][d1]*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                )

                + F6_t_f[1] * (

                    (-0.25f) * inv_S1_f * inv_S1_f * inv_S4_f * (
                        +delta_f[d0][d1]*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + (-0.125f) * inv_S1_f * inv_S2_f * inv_S4_f * (
                        +delta_f[d0][d1]*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + (-0.5f) * S2_f * inv_S1_f * inv_S1_f * inv_S4_f * (
                        +QD_0_f*QD_1_f*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.25f * inv_S1_f * inv_S4_f * (
                        +delta_f[b0][b1]*delta_f[d0][d1]*(PA_0_f*(-PA_1_f + PQ_a1_f) + PA_1_f*PQ_a0_f)
                        +delta_f[d0][d1]*(PB_0_f*(-PA_0_f*delta_f[a1][b1] - PA_1_f*delta_f[a0][b1] - PB_1_f*delta_f[a0][a1] + PQ_a0_f*delta_f[a1][b1] + PQ_a1_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][a1]) + PQ_b0_f*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) + delta_f[a0][b0]*(PA_1_f*(-PB_1_f + PQ_b1_f) + PB_1_f*PQ_a1_f) + delta_f[a1][b0]*(PA_0_f*(-PB_1_f + PQ_b1_f) + PB_1_f*PQ_a0_f))
                        +QD_0_f*(PA_0_f*(delta_f[a1][b0]*delta_f[b1][d1] + delta_f[a1][b1]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][b1]) + PA_1_f*(delta_f[a0][b0]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][b1]) + PB_0_f*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + PB_1_f*(delta_f[a0][a1]*delta_f[b0][d1] + delta_f[a0][b0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b0])) + QD_1_f*(PA_0_f*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + PA_1_f*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + PB_0_f*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + PB_1_f*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0]))
                        -(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f)*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + (-0.25f) * inv_S2_f * inv_S4_f * (
                        +PA_0_f*PA_1_f*delta_f[b0][b1]*delta_f[d0][d1]
                        +delta_f[d0][d1]*(PB_0_f*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) + PB_1_f*(PA_0_f*delta_f[a1][b0] + PA_1_f*delta_f[a0][b0]))
                    )

                    + (-0.5f) * S1_f * inv_S2_f * inv_S4_f * (
                        +PA_0_f*PA_1_f*PB_0_f*PB_1_f*delta_f[d0][d1]
                    )

                    + 0.5f * S2_f * inv_S1_f * inv_S4_f * (
                        +QD_0_f*QD_1_f*delta_f[b0][b1]*(PA_0_f*(-PA_1_f + PQ_a1_f) + PA_1_f*PQ_a0_f)
                        +QD_0_f*QD_1_f*(PB_0_f*(-PA_0_f*delta_f[a1][b1] - PA_1_f*delta_f[a0][b1] - PB_1_f*delta_f[a0][a1] + PQ_a0_f*delta_f[a1][b1] + PQ_a1_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][a1]) + PQ_b0_f*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) + delta_f[a0][b0]*(PA_1_f*(-PB_1_f + PQ_b1_f) + PB_1_f*PQ_a1_f) + delta_f[a1][b0]*(PA_0_f*(-PB_1_f + PQ_b1_f) + PB_1_f*PQ_a0_f))
                    )

                    + 0.5f * inv_S4_f * (
                        +delta_f[d0][d1]*(PA_0_f*PA_1_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PB_1_f*(PA_0_f*PQ_a1_f + PA_1_f*PQ_a0_f))
                        +PA_0_f*PA_1_f*(PB_0_f*(QD_0_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][d0]) + PB_1_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0])) + PB_0_f*PB_1_f*(PA_0_f*(QD_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][d0]) + PA_1_f*(QD_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][d0]))
                        -(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f)*(PA_0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) + PA_1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PB_0_f*PB_1_f*delta_f[a0][a1])
                    )

                    + S1_f * inv_S4_f * (
                        +PA_0_f*PA_1_f*PB_0_f*PB_1_f*(-PQ_d0_f*QD_1_f - PQ_d1_f*QD_0_f)
                    )

                    + S2_f * inv_S4_f * (
                        +QD_0_f*QD_1_f*(PA_0_f*PA_1_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PB_1_f*(PA_0_f*PQ_a1_f + PA_1_f*PQ_a0_f))
                    )

                )

                + F6_t_f[2] * (

                    0.25f * S2_f * S2_f * inv_S1_f * inv_S1_f * inv_S4_f * inv_S4_f * (
                        +QD_0_f*QD_1_f*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.25f * S2_f * inv_S1_f * inv_S4_f * inv_S4_f * (
                        +delta_f[b0][b1]*delta_f[d0][d1]*(-PA_0_f*PQ_a1_f + PQ_a0_f*(-PA_1_f + PQ_a1_f))
                        +delta_f[d0][d1]*(PQ_a0_f*delta_f[a1][b0]*(-PB_1_f + PQ_b1_f) + PQ_a0_f*delta_f[a1][b1]*(-PB_0_f + PQ_b0_f) + PQ_a1_f*delta_f[a0][b0]*(-PB_1_f + PQ_b1_f) + PQ_a1_f*delta_f[a0][b1]*(-PB_0_f + PQ_b0_f) + PQ_b0_f*PQ_b1_f*delta_f[a0][a1] - PQ_b0_f*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) - PQ_b1_f*(PA_0_f*delta_f[a1][b0] + PA_1_f*delta_f[a0][b0] + PB_0_f*delta_f[a0][a1]))
                        +(-PA_0_f*QD_0_f + PQ_a0_f*QD_0_f)*(delta_f[a1][b0]*delta_f[b1][d1] + delta_f[a1][b1]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][b1]) + (-PA_0_f*QD_1_f + PQ_a0_f*QD_1_f)*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + (-PA_1_f*QD_0_f + PQ_a1_f*QD_0_f)*(delta_f[a0][b0]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][b1]) + (-PA_1_f*QD_1_f + PQ_a1_f*QD_1_f)*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + (-PB_0_f*QD_0_f + PQ_b0_f*QD_0_f)*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + (-PB_0_f*QD_1_f + PQ_b0_f*QD_1_f)*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + (-PB_1_f*QD_0_f + PQ_b1_f*QD_0_f)*(delta_f[a0][a1]*delta_f[b0][d1] + delta_f[a0][b0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b0]) + (-PB_1_f*QD_1_f + PQ_b1_f*QD_1_f)*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0])
                        +2.0f*(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f)*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.25f * inv_S4_f * inv_S4_f * (
                        +delta_f[a1][b1]*delta_f[d0][d1]*(PA_0_f*(PB_0_f - PQ_b0_f) - PB_0_f*PQ_a0_f)
                        +PA_0_f*(PA_1_f*(delta_f[b0][d0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][d0]) + PB_0_f*(delta_f[a1][d0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b1][d0]) + PB_1_f*(delta_f[a1][d0]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][d0])) + PA_1_f*(PB_0_f*(delta_f[a0][d0]*delta_f[b1][d1] + delta_f[a0][d1]*delta_f[b1][d0]) + PB_1_f*(delta_f[a0][d0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][d0])) + PB_0_f*PB_1_f*(delta_f[a0][d0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][d0])
                        +delta_f[d0][d1]*(PA_1_f*(PB_1_f*delta_f[a0][b0] + delta_f[a0][b1]*(PB_0_f - PQ_b0_f) + delta_f[b0][b1]*(PA_0_f - PQ_a0_f)) + PB_1_f*delta_f[a0][a1]*(PB_0_f - PQ_b0_f) + PB_1_f*delta_f[a1][b0]*(PA_0_f - PQ_a0_f) - PQ_a1_f*(PA_0_f*delta_f[b0][b1] + PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) - PQ_b1_f*(PA_0_f*delta_f[a1][b0] + PA_1_f*delta_f[a0][b0] + PB_0_f*delta_f[a0][a1]))
                        -PQ_d0_f*(PA_0_f*(delta_f[a1][b0]*delta_f[b1][d1] + delta_f[a1][b1]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][b1]) + PA_1_f*(delta_f[a0][b0]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][b1]) + PB_0_f*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + PB_1_f*(delta_f[a0][a1]*delta_f[b0][d1] + delta_f[a0][b0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b0])) - PQ_d1_f*(PA_0_f*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + PA_1_f*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + PB_0_f*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + PB_1_f*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0]))
                        +PQ_d0_f*PQ_d1_f*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.5f * S1_f * inv_S4_f * inv_S4_f * (
                        -delta_f[d0][d1]*(PA_0_f*PA_1_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PB_1_f*(PA_0_f*PQ_a1_f + PA_1_f*PQ_a0_f))
                        -PA_0_f*PA_1_f*(PB_0_f*(PQ_d0_f*delta_f[b1][d1] + PQ_d1_f*delta_f[b1][d0]) + PB_1_f*(PQ_d0_f*delta_f[b0][d1] + PQ_d1_f*delta_f[b0][d0])) - PB_0_f*PB_1_f*(PA_0_f*(PQ_d0_f*delta_f[a1][d1] + PQ_d1_f*delta_f[a1][d0]) + PA_1_f*(PQ_d0_f*delta_f[a0][d1] + PQ_d1_f*delta_f[a0][d0]))
                        +PQ_d0_f*PQ_d1_f*(PA_0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) + PA_1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PB_0_f*PB_1_f*delta_f[a0][a1])
                    )

                    + 0.5f * S2_f * S2_f * inv_S1_f * inv_S4_f * inv_S4_f * (
                        +QD_0_f*QD_1_f*delta_f[b0][b1]*(-PA_0_f*PQ_a1_f + PQ_a0_f*(-PA_1_f + PQ_a1_f))
                        +QD_0_f*QD_1_f*(PQ_a0_f*delta_f[a1][b0]*(-PB_1_f + PQ_b1_f) + PQ_a0_f*delta_f[a1][b1]*(-PB_0_f + PQ_b0_f) + PQ_a1_f*delta_f[a0][b0]*(-PB_1_f + PQ_b1_f) + PQ_a1_f*delta_f[a0][b1]*(-PB_0_f + PQ_b0_f) + PQ_b0_f*PQ_b1_f*delta_f[a0][a1] - PQ_b0_f*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) - PQ_b1_f*(PA_0_f*delta_f[a1][b0] + PA_1_f*delta_f[a0][b0] + PB_0_f*delta_f[a0][a1]))
                    )

                    + 0.5f * S2_f * inv_S4_f * inv_S4_f * (
                        +QD_0_f*delta_f[b1][d1]*(PA_0_f*(PA_1_f*PQ_b0_f + PB_0_f*PQ_a1_f) + PA_1_f*PB_0_f*PQ_a0_f)
                        +PB_1_f*(PA_0_f*(PQ_a1_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0]) + PQ_b0_f*(QD_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][d0])) + PQ_a0_f*(PA_1_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0]) + PB_0_f*(QD_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][d0]))) + QD_0_f*(PA_0_f*PQ_b1_f*(PA_1_f*delta_f[b0][d1] + PB_0_f*delta_f[a1][d1]) + delta_f[a0][d1]*(PA_1_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PB_1_f*PQ_a1_f)) + QD_1_f*(PA_0_f*(PA_1_f*(PQ_b0_f*delta_f[b1][d0] + PQ_b1_f*delta_f[b0][d0]) + PB_0_f*(PQ_a1_f*delta_f[b1][d0] + PQ_b1_f*delta_f[a1][d0])) + PA_1_f*PB_0_f*(PQ_a0_f*delta_f[b1][d0] + PQ_b1_f*delta_f[a0][d0]) + PB_1_f*delta_f[a0][d0]*(PA_1_f*PQ_b0_f + PB_0_f*PQ_a1_f))
                        +(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f)*(PA_0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) - PA_0_f*(PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) + PA_1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) - PA_1_f*(PQ_a0_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][b0]) + PB_0_f*PB_1_f*delta_f[a0][a1] - PB_0_f*(PQ_a0_f*delta_f[a1][b1] + PQ_a1_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][a1]) - PB_1_f*(PQ_a0_f*delta_f[a1][b0] + PQ_a1_f*delta_f[a0][b0] + PQ_b0_f*delta_f[a0][a1]))
                        +delta_f[d0][d1]*(PA_0_f*PQ_b0_f*(PA_1_f*PQ_b1_f + PB_1_f*PQ_a1_f) + PA_1_f*PQ_a0_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PQ_a1_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))
                    )

                    + S1_f * S1_f * inv_S4_f * inv_S4_f * (
                        +PA_0_f*PA_1_f*PB_0_f*PB_1_f*PQ_d0_f*PQ_d1_f
                    )

                    + S1_f * S2_f * inv_S4_f * inv_S4_f * (
                        -(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f)*(PA_0_f*PA_1_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PB_1_f*(PA_0_f*PQ_a1_f + PA_1_f*PQ_a0_f))
                    )

                    + S2_f * S2_f * inv_S4_f * inv_S4_f * (
                        +QD_0_f*QD_1_f*(PA_0_f*PQ_b0_f*(PA_1_f*PQ_b1_f + PB_1_f*PQ_a1_f) + PA_1_f*PQ_a0_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PQ_a1_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))
                    )

                    + 0.125f * S2_f * inv_S1_f * inv_S1_f * inv_S4_f * inv_S4_f * (
                        +delta_f[d0][d1]*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.125f * inv_S1_f * inv_S4_f * inv_S4_f * (
                        +2.0f*delta_f[d0][d1]*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                        +delta_f[b0][b1]*(delta_f[a0][d0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][d0]) + delta_f[b0][d0]*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + delta_f[b0][d1]*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + delta_f[b1][d0]*(delta_f[a0][b0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b0]) + delta_f[b1][d1]*(delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0])
                    )

                )

                + F6_t_f[3] * (

                    (-0.125f) * S2_f * inv_S1_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +delta_f[a0][a1]*(delta_f[b0][b1]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][d0]) + delta_f[a0][b0]*(delta_f[a1][b1]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b1][d0]) + delta_f[a0][b1]*(delta_f[a1][b0]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][d0]) + delta_f[a0][d0]*(delta_f[a1][b0]*delta_f[b1][d1] + delta_f[a1][b1]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][b1]) + delta_f[a0][d1]*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1])
                    )

                    + (-0.25f) * S2_f * S2_f * inv_S1_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_a0_f*PQ_a1_f*delta_f[b0][b1]*delta_f[d0][d1]
                        +delta_f[d0][d1]*(PQ_b0_f*(PQ_a0_f*delta_f[a1][b1] + PQ_a1_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][a1]) + PQ_b1_f*(PQ_a0_f*delta_f[a1][b0] + PQ_a1_f*delta_f[a0][b0]))
                        +QD_0_f*(PQ_a0_f*(delta_f[a1][b0]*delta_f[b1][d1] + delta_f[a1][b1]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][b1]) + PQ_a1_f*(delta_f[a0][b0]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][b1]) + PQ_b0_f*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + PQ_b1_f*(delta_f[a0][a1]*delta_f[b0][d1] + delta_f[a0][b0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b0])) + QD_1_f*(PQ_a0_f*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + PQ_a1_f*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + PQ_b0_f*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + PQ_b1_f*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0]))
                        +(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f)*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.25f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +delta_f[a1][b1]*delta_f[d0][d1]*(PB_0_f*PQ_a0_f + PQ_b0_f*(PA_0_f - PQ_a0_f))
                        +(PA_0_f*PQ_a1_f + PA_1_f*PQ_a0_f)*(delta_f[b0][d0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][d0]) + (PA_0_f*PQ_b0_f + PB_0_f*PQ_a0_f)*(delta_f[a1][d0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b1][d0]) + (PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f)*(delta_f[a1][d0]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][d0]) + (PA_1_f*PQ_b0_f + PB_0_f*PQ_a1_f)*(delta_f[a0][d0]*delta_f[b1][d1] + delta_f[a0][d1]*delta_f[b1][d0]) + (PA_1_f*PQ_b1_f + PB_1_f*PQ_a1_f)*(delta_f[a0][d0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][d0]) + (PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f)*(delta_f[a0][d0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][d0])
                        +delta_f[d0][d1]*(PA_1_f*(PQ_a0_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][b0]) + PB_1_f*(PQ_a0_f*delta_f[a1][b0] + PQ_b0_f*delta_f[a0][a1]) + PQ_a1_f*(PA_0_f*delta_f[b0][b1] + PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0] - PQ_a0_f*delta_f[b0][b1] - PQ_b0_f*delta_f[a0][b1] - PQ_b1_f*delta_f[a0][b0]) + PQ_b1_f*(delta_f[a0][a1]*(PB_0_f - PQ_b0_f) + delta_f[a1][b0]*(PA_0_f - PQ_a0_f)))
                        +(PA_0_f*PQ_d0_f - PQ_a0_f*PQ_d0_f)*(delta_f[a1][b0]*delta_f[b1][d1] + delta_f[a1][b1]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][b1]) + (PA_0_f*PQ_d1_f - PQ_a0_f*PQ_d1_f)*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + (PA_1_f*PQ_d0_f - PQ_a1_f*PQ_d0_f)*(delta_f[a0][b0]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][b1]) + (PA_1_f*PQ_d1_f - PQ_a1_f*PQ_d1_f)*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + (PB_0_f*PQ_d0_f - PQ_b0_f*PQ_d0_f)*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + (PB_0_f*PQ_d1_f - PQ_b0_f*PQ_d1_f)*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + (PB_1_f*PQ_d0_f - PQ_b1_f*PQ_d0_f)*(delta_f[a0][a1]*delta_f[b0][d1] + delta_f[a0][b0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b0]) + (PB_1_f*PQ_d1_f - PQ_b1_f*PQ_d1_f)*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0])
                        -2*PQ_d0_f*PQ_d1_f*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.5f * S1_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        -delta_f[d0][d1]*(PA_0_f*PQ_b0_f*(PA_1_f*PQ_b1_f + PB_1_f*PQ_a1_f) + PA_1_f*PQ_a0_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PQ_a1_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))
                        -PA_0_f*(PA_1_f*(PQ_b0_f*(PQ_d0_f*delta_f[b1][d1] + PQ_d1_f*delta_f[b1][d0]) + PQ_b1_f*(PQ_d0_f*delta_f[b0][d1] + PQ_d1_f*delta_f[b0][d0])) + PB_0_f*(PQ_a1_f*(PQ_d0_f*delta_f[b1][d1] + PQ_d1_f*delta_f[b1][d0]) + PQ_b1_f*(PQ_d0_f*delta_f[a1][d1] + PQ_d1_f*delta_f[a1][d0])) + PB_1_f*(PQ_a1_f*(PQ_d0_f*delta_f[b0][d1] + PQ_d1_f*delta_f[b0][d0]) + PQ_b0_f*(PQ_d0_f*delta_f[a1][d1] + PQ_d1_f*delta_f[a1][d0]))) - PA_1_f*(PB_0_f*(PQ_a0_f*(PQ_d0_f*delta_f[b1][d1] + PQ_d1_f*delta_f[b1][d0]) + PQ_b1_f*(PQ_d0_f*delta_f[a0][d1] + PQ_d1_f*delta_f[a0][d0])) + PB_1_f*(PQ_a0_f*(PQ_d0_f*delta_f[b0][d1] + PQ_d1_f*delta_f[b0][d0]) + PQ_b0_f*(PQ_d0_f*delta_f[a0][d1] + PQ_d1_f*delta_f[a0][d0]))) - PB_0_f*PB_1_f*(PQ_a0_f*(PQ_d0_f*delta_f[a1][d1] + PQ_d1_f*delta_f[a1][d0]) + PQ_a1_f*(PQ_d0_f*delta_f[a0][d1] + PQ_d1_f*delta_f[a0][d0]))
                        +PQ_d0_f*PQ_d1_f*(PA_0_f*(-PA_1_f*delta_f[b0][b1] - PB_0_f*delta_f[a1][b1] - PB_1_f*delta_f[a1][b0] + PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) + PA_1_f*(PQ_a0_f*delta_f[b0][b1] + delta_f[a0][b0]*(-PB_1_f + PQ_b1_f) + delta_f[a0][b1]*(-PB_0_f + PQ_b0_f)) + PB_0_f*(PQ_a0_f*delta_f[a1][b1] + PQ_a1_f*delta_f[a0][b1] + delta_f[a0][a1]*(-PB_1_f + PQ_b1_f)) + PB_1_f*(PQ_a0_f*delta_f[a1][b0] + PQ_a1_f*delta_f[a0][b0] + PQ_b0_f*delta_f[a0][a1]))
                    )

                    + (-0.5f) * S2_f * S2_f * S2_f * inv_S1_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_a0_f*PQ_a1_f*QD_0_f*QD_1_f*delta_f[b0][b1]
                        +QD_0_f*QD_1_f*(PQ_b0_f*(PQ_a0_f*delta_f[a1][b1] + PQ_a1_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][a1]) + PQ_b1_f*(PQ_a0_f*delta_f[a1][b0] + PQ_a1_f*delta_f[a0][b0]))
                    )

                    + 0.5f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +QD_0_f*delta_f[b1][d1]*(PA_1_f*PQ_a0_f*PQ_b0_f + PQ_a1_f*(PA_0_f*PQ_b0_f + PB_0_f*PQ_a0_f))
                        +PB_1_f*PQ_a0_f*(PQ_a1_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0]) + PQ_b0_f*(QD_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][d0])) + QD_0_f*(PB_1_f*PQ_a1_f*PQ_b0_f*delta_f[a0][d1] + PQ_b1_f*(PA_0_f*(PQ_a1_f*delta_f[b0][d1] + PQ_b0_f*delta_f[a1][d1]) + PA_1_f*(PQ_a0_f*delta_f[b0][d1] + PQ_b0_f*delta_f[a0][d1]) + PB_0_f*(PQ_a0_f*delta_f[a1][d1] + PQ_a1_f*delta_f[a0][d1]))) + QD_1_f*(PQ_a0_f*(PA_1_f*(PQ_b0_f*delta_f[b1][d0] + PQ_b1_f*delta_f[b0][d0]) + PB_0_f*(PQ_a1_f*delta_f[b1][d0] + PQ_b1_f*delta_f[a1][d0])) + PQ_a1_f*(PA_0_f*(PQ_b0_f*delta_f[b1][d0] + PQ_b1_f*delta_f[b0][d0]) + delta_f[a0][d0]*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f)) + PQ_b0_f*PQ_b1_f*(PA_0_f*delta_f[a1][d0] + PA_1_f*delta_f[a0][d0]))
                        +(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f)*(PA_0_f*(PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) + PQ_a0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0] - PQ_a1_f*delta_f[b0][b1] - PQ_b0_f*delta_f[a1][b1] - PQ_b1_f*delta_f[a1][b0]) + PQ_a1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PQ_b0_f*(PB_1_f*delta_f[a0][a1] + delta_f[a0][b1]*(PA_1_f - PQ_a1_f)) + PQ_b1_f*(delta_f[a0][a1]*(PB_0_f - PQ_b0_f) + delta_f[a0][b0]*(PA_1_f - PQ_a1_f)))
                        +delta_f[d0][d1]*(PQ_a0_f*PQ_b1_f*(PA_1_f*PQ_b0_f + PB_0_f*PQ_a1_f) + PQ_a1_f*PQ_b0_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))
                    )

                    + S1_f * S1_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_d0_f*PQ_d1_f*(PA_0_f*PA_1_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PB_1_f*(PA_0_f*PQ_a1_f + PA_1_f*PQ_a0_f))
                    )

                    + S1_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        -(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f)*(PA_0_f*PQ_b0_f*(PA_1_f*PQ_b1_f + PB_1_f*PQ_a1_f) + PA_1_f*PQ_a0_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PQ_a1_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))
                    )

                    + S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +QD_0_f*QD_1_f*(PQ_a0_f*PQ_b1_f*(PA_1_f*PQ_b0_f + PB_0_f*PQ_a1_f) + PQ_a1_f*PQ_b0_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))
                    )

                )

                + F6_t_f[4] * (

                    0.5f * S1_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        -delta_f[d0][d1]*(PQ_a0_f*PQ_b1_f*(PA_1_f*PQ_b0_f + PB_0_f*PQ_a1_f) + PQ_a1_f*PQ_b0_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))
                        -PQ_a0_f*(PA_1_f*(PQ_b0_f*(PQ_d0_f*delta_f[b1][d1] + PQ_d1_f*delta_f[b1][d0]) + PQ_b1_f*(PQ_d0_f*delta_f[b0][d1] + PQ_d1_f*delta_f[b0][d0])) + PB_0_f*(PQ_a1_f*(PQ_d0_f*delta_f[b1][d1] + PQ_d1_f*delta_f[b1][d0]) + PQ_b1_f*(PQ_d0_f*delta_f[a1][d1] + PQ_d1_f*delta_f[a1][d0])) + PB_1_f*(PQ_a1_f*(PQ_d0_f*delta_f[b0][d1] + PQ_d1_f*delta_f[b0][d0]) + PQ_b0_f*(PQ_d0_f*delta_f[a1][d1] + PQ_d1_f*delta_f[a1][d0]))) - PQ_a1_f*(PA_0_f*(PQ_b0_f*(PQ_d0_f*delta_f[b1][d1] + PQ_d1_f*delta_f[b1][d0]) + PQ_b1_f*(PQ_d0_f*delta_f[b0][d1] + PQ_d1_f*delta_f[b0][d0])) + (PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f)*(PQ_d0_f*delta_f[a0][d1] + PQ_d1_f*delta_f[a0][d0])) - PQ_b0_f*PQ_b1_f*(PA_0_f*(PQ_d0_f*delta_f[a1][d1] + PQ_d1_f*delta_f[a1][d0]) + PA_1_f*(PQ_d0_f*delta_f[a0][d1] + PQ_d1_f*delta_f[a0][d0]))
                        +PQ_d0_f*PQ_d1_f*(-PA_0_f*(PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) - PQ_a0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) + PQ_a0_f*(PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) - PQ_a1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PQ_a1_f*(PQ_b0_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][b0]) + PQ_b0_f*PQ_b1_f*delta_f[a0][a1] - PQ_b0_f*(PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) - PQ_b1_f*(PA_1_f*delta_f[a0][b0] + PB_0_f*delta_f[a0][a1]))
                    )

                    + 0.5f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_a0_f*PQ_a1_f*PQ_b0_f*QD_0_f*delta_f[b1][d1]
                        +PQ_a0_f*PQ_a1_f*QD_1_f*(PQ_b0_f*delta_f[b1][d0] + PQ_b1_f*delta_f[b0][d0]) + PQ_b1_f*(PQ_a0_f*QD_0_f*(PQ_a1_f*delta_f[b0][d1] + PQ_b0_f*delta_f[a1][d1]) + PQ_b0_f*(PQ_a1_f*QD_0_f*delta_f[a0][d1] + QD_1_f*(PQ_a0_f*delta_f[a1][d0] + PQ_a1_f*delta_f[a0][d0])))
                        +(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f)*(PQ_a0_f*(PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) + PQ_a1_f*(PQ_b0_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][b0]) + PQ_b0_f*PQ_b1_f*delta_f[a0][a1])
                        +PQ_a0_f*PQ_a1_f*PQ_b0_f*PQ_b1_f*delta_f[d0][d1]
                    )

                    + S1_f * S1_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_d0_f*PQ_d1_f*(PA_0_f*PQ_b0_f*(PA_1_f*PQ_b1_f + PB_1_f*PQ_a1_f) + PA_1_f*PQ_a0_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PQ_a1_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))
                    )

                    + S1_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        -(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f)*(PQ_a0_f*PQ_b1_f*(PA_1_f*PQ_b0_f + PB_0_f*PQ_a1_f) + PQ_a1_f*PQ_b0_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))
                    )

                    + S2_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_a0_f*PQ_a1_f*PQ_b0_f*PQ_b1_f*QD_0_f*QD_1_f
                    )

                    + 0.25f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_a0_f*PQ_a1_f*(delta_f[b0][b1]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][d0])
                        +PQ_b0_f*(PQ_a0_f*(delta_f[a1][b1]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b1][d0]) + PQ_a1_f*(delta_f[a0][b1]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[b1][d1] + delta_f[a0][d1]*delta_f[b1][d0]) + PQ_b1_f*(delta_f[a0][a1]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][d0]) + PQ_d0_f*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + PQ_d1_f*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1])) + PQ_b1_f*(PQ_a0_f*(delta_f[a1][b0]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][d0]) + PQ_a1_f*(delta_f[a0][b0]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][d0]) + PQ_d0_f*(delta_f[a0][a1]*delta_f[b0][d1] + delta_f[a0][b0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b0]) + PQ_d1_f*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0])) + PQ_d0_f*(PQ_a0_f*(delta_f[a1][b0]*delta_f[b1][d1] + delta_f[a1][b1]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][b1]) + PQ_a1_f*(delta_f[a0][b0]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][b1]) + PQ_d1_f*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])) + PQ_d1_f*(PQ_a0_f*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + PQ_a1_f*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]))
                    )

                )

                + F6_t_f[5] * (

                    (-0.5f) * S1_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_a0_f*PQ_a1_f*PQ_b0_f*PQ_b1_f*delta_f[d0][d1]
                        +PQ_d0_f*(PQ_a0_f*(PQ_a1_f*(PQ_b0_f*delta_f[b1][d1] + PQ_b1_f*delta_f[b0][d1] + PQ_d1_f*delta_f[b0][b1]) + PQ_b1_f*PQ_d1_f*delta_f[a1][b0]) + PQ_b0_f*PQ_b1_f*(PQ_a0_f*delta_f[a1][d1] + PQ_a1_f*delta_f[a0][d1] + PQ_d1_f*delta_f[a0][a1])) + PQ_d1_f*(PQ_a1_f*PQ_b1_f*(PQ_a0_f*delta_f[b0][d0] + PQ_b0_f*delta_f[a0][d0] + PQ_d0_f*delta_f[a0][b0]) + PQ_b0_f*(PQ_a0_f*(PQ_a1_f*delta_f[b1][d0] + PQ_b1_f*delta_f[a1][d0] + PQ_d0_f*delta_f[a1][b1]) + PQ_a1_f*PQ_d0_f*delta_f[a0][b1]))
                    )

                    + S1_f * S1_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_d0_f*PQ_d1_f*(PQ_a0_f*PQ_b1_f*(PA_1_f*PQ_b0_f + PB_0_f*PQ_a1_f) + PQ_a1_f*PQ_b0_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))
                    )

                    + S1_f * S2_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_a0_f*PQ_a1_f*PQ_b0_f*PQ_b1_f*(-PQ_d0_f*QD_1_f - PQ_d1_f*QD_0_f)
                    )

                )

                + F6_t_f[6] * (

                    S1_f * S1_f * S2_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_a0_f*PQ_a1_f*PQ_b0_f*PQ_b1_f*PQ_d0_f*PQ_d1_f
                    )

                )

                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        const float D_f = sd_mat_D_f[kl];
        const float contrib_f = eri_ijkl_f * D_f * 2.0f;
        ERIs[threadIdx.y] += (double)contrib_f;
    }

    __syncthreads();

    if ((threadIdx.y == 0) && (ij < dd_prim_pair_count_local))
    {
        double J_ij = 0.0f;

        for (uint32_t n = 0; n < TILE_DIM_LARGE; n++)
        {
            J_ij += ERIs[n];
        }

        mat_J[ij] += J_ij;
    }
}

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDPP_FP32_auto_s_ast(double*         mat_J,
                       const float*    p_prim_info_f,
                       const uint32_t  p_prim_count,
                       const float*    d_prim_info_f,
                       const uint32_t  d_prim_count,
                       const float*    pp_mat_D_f,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const float*    dd_pair_data_local_f,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* pp_first_inds,
                       const uint32_t* pp_second_inds,
                       const float*    pp_pair_data_f,
                       const uint32_t  pp_prim_pair_count,
                       const float*    boys_func_table_f,
                       const float*    boys_func_ft_f,
                       const uint32_t* prec_cut_ij_tile,
                       const uint32_t* screen_cut_ij_tile)
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ float   delta_f[3][3];

    __shared__ float a_i_f, a_j_f, r_i_f[3], r_j_f[3], S_ij_00_f, S1_f, inv_S1_f;
    __shared__ float PA_0_f, PA_1_f, PB_0_f, PB_1_f;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];
    const uint32_t screen_cut = screen_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta_f[0][0] = 1.0f; delta_f[0][1] = 0.0f; delta_f[0][2] = 0.0f;
        delta_f[1][0] = 0.0f; delta_f[1][1] = 1.0f; delta_f[1][2] = 0.0f;
        delta_f[2][0] = 0.0f; delta_f[2][1] = 0.0f; delta_f[2][2] = 1.0f;

        if (ij < dd_prim_pair_count_local)
        {
            i = dd_first_inds_local[ij];
            j = dd_second_inds_local[ij];

            a_i_f = d_prim_info_f[i / 6 + d_prim_count * 0];

            r_i_f[0] = d_prim_info_f[i / 6 + d_prim_count * 2];
            r_i_f[1] = d_prim_info_f[i / 6 + d_prim_count * 3];
            r_i_f[2] = d_prim_info_f[i / 6 + d_prim_count * 4];

            a_j_f = d_prim_info_f[j / 6 + d_prim_count * 0];

            r_j_f[0] = d_prim_info_f[j / 6 + d_prim_count * 2];
            r_j_f[1] = d_prim_info_f[j / 6 + d_prim_count * 3];
            r_j_f[2] = d_prim_info_f[j / 6 + d_prim_count * 4];

            S1_f = a_i_f + a_j_f;
            inv_S1_f = (float) (1.0 / (double)S1_f);

            S_ij_00_f = dd_pair_data_local_f[ij];

            a0 = d_cart_inds[i % 6][0];
            a1 = d_cart_inds[i % 6][1];
            b0 = d_cart_inds[j % 6][0];
            b1 = d_cart_inds[j % 6][1];

            PA_0_f = (a_j_f  * inv_S1_f) * (r_j_f[a0] - r_i_f[a0]);
            PA_1_f = (a_j_f  * inv_S1_f) * (r_j_f[a1] - r_i_f[a1]);
            PB_0_f = (-a_i_f * inv_S1_f) * (r_j_f[b0] - r_i_f[b0]);
            PB_1_f = (-a_i_f * inv_S1_f) * (r_j_f[b1] - r_i_f[b1]);

        }

    }

    ERIs[threadIdx.y] = 0.0;

    __syncthreads();

    for (uint32_t m = prec_cut; m < screen_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= pp_prim_pair_count))
        {
            break;
        }

        const auto k = pp_first_inds[kl];
        const auto l = pp_second_inds[kl];

        const auto a_k_f = p_prim_info_f[k / 3 + p_prim_count * 0];

        const float r_k0_f = p_prim_info_f[k / 3 + p_prim_count * 2];
        const float r_k1_f = p_prim_info_f[k / 3 + p_prim_count * 3];
        const float r_k2_f = p_prim_info_f[k / 3 + p_prim_count * 4];

        const auto a_l_f = p_prim_info_f[l / 3 + p_prim_count * 0];

        const float r_l0_f = p_prim_info_f[l / 3 + p_prim_count * 2];
        const float r_l1_f = p_prim_info_f[l / 3 + p_prim_count * 3];
        const float r_l2_f = p_prim_info_f[l / 3 + p_prim_count * 4];

        const auto S_kl_00_f = pp_pair_data_f[kl];

        const auto c0 = k % 3;
        const auto d0 = l % 3;

        const float r_l_c0_f = (c0 == 0 ? r_l0_f : (c0 == 1 ? r_l1_f : r_l2_f));
        const float r_l_d0_f = (d0 == 0 ? r_l0_f : (d0 == 1 ? r_l1_f : r_l2_f));

        const float r_k_c0_f = (c0 == 0 ? r_k0_f : (c0 == 1 ? r_k1_f : r_k2_f));
        const float r_k_d0_f = (d0 == 0 ? r_k0_f : (d0 == 1 ? r_k1_f : r_k2_f));

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2_f = a_k_f + a_l_f;

        const auto inv_S2_f = 1.0f / S2_f;
        const auto inv_S4_f = 1.0f / (S1_f + S2_f);

        const float PQ0_f = (a_k_f * r_k0_f + a_l_f * r_l0_f) * inv_S2_f - (a_i_f * r_i_f[0] + a_j_f * r_j_f[0]) * inv_S1_f;
        const float PQ1_f = (a_k_f * r_k1_f + a_l_f * r_l1_f) * inv_S2_f - (a_i_f * r_i_f[1] + a_j_f * r_j_f[1]) * inv_S1_f;
        const float PQ2_f = (a_k_f * r_k2_f + a_l_f * r_l2_f) * inv_S2_f - (a_i_f * r_i_f[2] + a_j_f * r_j_f[2]) * inv_S1_f;

        const float PQ_a0_f = (a0 == 0 ? PQ0_f : (a0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_a1_f = (a1 == 0 ? PQ0_f : (a1 == 1 ? PQ1_f : PQ2_f));
        const float PQ_b0_f = (b0 == 0 ? PQ0_f : (b0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_b1_f = (b1 == 0 ? PQ0_f : (b1 == 1 ? PQ1_f : PQ2_f));
        const float PQ_c0_f = (c0 == 0 ? PQ0_f : (c0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_d0_f = (d0 == 0 ? PQ0_f : (d0 == 1 ? PQ1_f : PQ2_f));

        const auto r2_PQ_f = PQ0_f * PQ0_f + PQ1_f * PQ1_f + PQ2_f * PQ2_f;

        const auto Lambda_f = sqrtf(4.0f * S1_f * S2_f * MATH_CONST_INV_PI_F * inv_S4_f);

        float F6_t_f[7];

        gpu::computeBoysFunction_f(F6_t_f, S1_f * S2_f * inv_S4_f * r2_PQ_f, 6, boys_func_table_f, boys_func_ft_f);

        const auto QC_0_f = (a_l_f * inv_S2_f) * (r_l_c0_f - r_k_c0_f);
        const auto QD_0_f = (-a_k_f * inv_S2_f) * (r_l_d0_f - r_k_d0_f);

        const float eri_ijkl_f = Lambda_f * S_ij_00_f * S_kl_00_f * (

                + F6_t_f[0] * (

                    0.25f * inv_S1_f * inv_S1_f * (
                        +QC_0_f*QD_0_f*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.25f * inv_S1_f * inv_S2_f * (
                        +PA_0_f*PB_0_f*delta_f[a1][b1]*delta_f[c0][d0]
                        +delta_f[c0][d0]*(PA_1_f*(PA_0_f*delta_f[b0][b1] + PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PB_1_f*(PA_0_f*delta_f[a1][b0] + PB_0_f*delta_f[a0][a1]))
                    )

                    + 0.5f * inv_S1_f * (
                        +PA_0_f*PA_1_f*QC_0_f*QD_0_f*delta_f[b0][b1]
                        +QC_0_f*QD_0_f*(PB_0_f*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) + PB_1_f*(PA_0_f*delta_f[a1][b0] + PA_1_f*delta_f[a0][b0]))
                    )

                    + 0.5f * inv_S2_f * (
                        +PA_0_f*PA_1_f*PB_0_f*PB_1_f*delta_f[c0][d0]
                    )

                    + (
                        +PA_0_f*PA_1_f*PB_0_f*PB_1_f*QC_0_f*QD_0_f
                    )

                    + 0.125f * inv_S1_f * inv_S1_f * inv_S2_f * (
                        +delta_f[c0][d0]*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                )

                + F6_t_f[1] * (

                    (-0.25f) * inv_S1_f * inv_S1_f * inv_S4_f * (
                        +delta_f[c0][d0]*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + (-0.125f) * inv_S1_f * inv_S2_f * inv_S4_f * (
                        +delta_f[c0][d0]*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + (-0.5f) * S2_f * inv_S1_f * inv_S1_f * inv_S4_f * (
                        +QC_0_f*QD_0_f*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.25f * inv_S1_f * inv_S4_f * (
                        +delta_f[b0][b1]*delta_f[c0][d0]*(PA_0_f*(-PA_1_f + PQ_a1_f) + PA_1_f*PQ_a0_f)
                        +delta_f[c0][d0]*(PB_0_f*(-PA_0_f*delta_f[a1][b1] - PA_1_f*delta_f[a0][b1] - PB_1_f*delta_f[a0][a1] + PQ_a0_f*delta_f[a1][b1] + PQ_a1_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][a1]) + PQ_b0_f*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) + delta_f[a0][b0]*(PA_1_f*(-PB_1_f + PQ_b1_f) + PB_1_f*PQ_a1_f) + delta_f[a1][b0]*(PA_0_f*(-PB_1_f + PQ_b1_f) + PB_1_f*PQ_a0_f))
                        +QC_0_f*(PA_0_f*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + PA_1_f*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + PB_0_f*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + PB_1_f*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0])) + QD_0_f*(PA_0_f*(delta_f[a1][b0]*delta_f[b1][c0] + delta_f[a1][b1]*delta_f[b0][c0] + delta_f[a1][c0]*delta_f[b0][b1]) + PA_1_f*(delta_f[a0][b0]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[b0][c0] + delta_f[a0][c0]*delta_f[b0][b1]) + PB_0_f*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1]) + PB_1_f*(delta_f[a0][a1]*delta_f[b0][c0] + delta_f[a0][b0]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b0]))
                        -(PQ_c0_f*QD_0_f + PQ_d0_f*QC_0_f)*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + (-0.25f) * inv_S2_f * inv_S4_f * (
                        +PA_0_f*PA_1_f*delta_f[b0][b1]*delta_f[c0][d0]
                        +delta_f[c0][d0]*(PB_0_f*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) + PB_1_f*(PA_0_f*delta_f[a1][b0] + PA_1_f*delta_f[a0][b0]))
                    )

                    + (-0.5f) * S1_f * inv_S2_f * inv_S4_f * (
                        +PA_0_f*PA_1_f*PB_0_f*PB_1_f*delta_f[c0][d0]
                    )

                    + 0.5f * S2_f * inv_S1_f * inv_S4_f * (
                        +QC_0_f*QD_0_f*delta_f[b0][b1]*(PA_0_f*(-PA_1_f + PQ_a1_f) + PA_1_f*PQ_a0_f)
                        +QC_0_f*QD_0_f*(PB_0_f*(-PA_0_f*delta_f[a1][b1] - PA_1_f*delta_f[a0][b1] - PB_1_f*delta_f[a0][a1] + PQ_a0_f*delta_f[a1][b1] + PQ_a1_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][a1]) + PQ_b0_f*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) + delta_f[a0][b0]*(PA_1_f*(-PB_1_f + PQ_b1_f) + PB_1_f*PQ_a1_f) + delta_f[a1][b0]*(PA_0_f*(-PB_1_f + PQ_b1_f) + PB_1_f*PQ_a0_f))
                    )

                    + 0.5f * inv_S4_f * (
                        +delta_f[c0][d0]*(PA_0_f*PA_1_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PB_1_f*(PA_0_f*PQ_a1_f + PA_1_f*PQ_a0_f))
                        +PA_0_f*PA_1_f*(PB_0_f*(QC_0_f*delta_f[b1][d0] + QD_0_f*delta_f[b1][c0]) + PB_1_f*(QC_0_f*delta_f[b0][d0] + QD_0_f*delta_f[b0][c0])) + PB_0_f*PB_1_f*(PA_0_f*(QC_0_f*delta_f[a1][d0] + QD_0_f*delta_f[a1][c0]) + PA_1_f*(QC_0_f*delta_f[a0][d0] + QD_0_f*delta_f[a0][c0]))
                        -(PQ_c0_f*QD_0_f + PQ_d0_f*QC_0_f)*(PA_0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) + PA_1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PB_0_f*PB_1_f*delta_f[a0][a1])
                    )

                    + S1_f * inv_S4_f * (
                        +PA_0_f*PA_1_f*PB_0_f*PB_1_f*(-PQ_c0_f*QD_0_f - PQ_d0_f*QC_0_f)
                    )

                    + S2_f * inv_S4_f * (
                        +QC_0_f*QD_0_f*(PA_0_f*PA_1_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PB_1_f*(PA_0_f*PQ_a1_f + PA_1_f*PQ_a0_f))
                    )

                )

                + F6_t_f[2] * (

                    0.25f * S2_f * S2_f * inv_S1_f * inv_S1_f * inv_S4_f * inv_S4_f * (
                        +QC_0_f*QD_0_f*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.25f * S2_f * inv_S1_f * inv_S4_f * inv_S4_f * (
                        +delta_f[b0][b1]*delta_f[c0][d0]*(-PA_0_f*PQ_a1_f + PQ_a0_f*(-PA_1_f + PQ_a1_f))
                        +delta_f[c0][d0]*(PQ_a0_f*delta_f[a1][b0]*(-PB_1_f + PQ_b1_f) + PQ_a0_f*delta_f[a1][b1]*(-PB_0_f + PQ_b0_f) + PQ_a1_f*delta_f[a0][b0]*(-PB_1_f + PQ_b1_f) + PQ_a1_f*delta_f[a0][b1]*(-PB_0_f + PQ_b0_f) + PQ_b0_f*PQ_b1_f*delta_f[a0][a1] - PQ_b0_f*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) - PQ_b1_f*(PA_0_f*delta_f[a1][b0] + PA_1_f*delta_f[a0][b0] + PB_0_f*delta_f[a0][a1]))
                        +(-PA_0_f*QC_0_f + PQ_a0_f*QC_0_f)*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + (-PA_0_f*QD_0_f + PQ_a0_f*QD_0_f)*(delta_f[a1][b0]*delta_f[b1][c0] + delta_f[a1][b1]*delta_f[b0][c0] + delta_f[a1][c0]*delta_f[b0][b1]) + (-PA_1_f*QC_0_f + PQ_a1_f*QC_0_f)*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + (-PA_1_f*QD_0_f + PQ_a1_f*QD_0_f)*(delta_f[a0][b0]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[b0][c0] + delta_f[a0][c0]*delta_f[b0][b1]) + (-PB_0_f*QC_0_f + PQ_b0_f*QC_0_f)*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + (-PB_0_f*QD_0_f + PQ_b0_f*QD_0_f)*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1]) + (-PB_1_f*QC_0_f + PQ_b1_f*QC_0_f)*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0]) + (-PB_1_f*QD_0_f + PQ_b1_f*QD_0_f)*(delta_f[a0][a1]*delta_f[b0][c0] + delta_f[a0][b0]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b0])
                        +2.0f*(PQ_c0_f*QD_0_f + PQ_d0_f*QC_0_f)*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.25f * inv_S4_f * inv_S4_f * (
                        +delta_f[a1][b1]*delta_f[c0][d0]*(PA_0_f*(PB_0_f - PQ_b0_f) - PB_0_f*PQ_a0_f)
                        +PA_0_f*(PA_1_f*(delta_f[b0][c0]*delta_f[b1][d0] + delta_f[b0][d0]*delta_f[b1][c0]) + PB_0_f*(delta_f[a1][c0]*delta_f[b1][d0] + delta_f[a1][d0]*delta_f[b1][c0]) + PB_1_f*(delta_f[a1][c0]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][c0])) + PA_1_f*(PB_0_f*(delta_f[a0][c0]*delta_f[b1][d0] + delta_f[a0][d0]*delta_f[b1][c0]) + PB_1_f*(delta_f[a0][c0]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][c0])) + PB_0_f*PB_1_f*(delta_f[a0][c0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][c0])
                        +delta_f[c0][d0]*(PA_1_f*(PB_1_f*delta_f[a0][b0] + delta_f[a0][b1]*(PB_0_f - PQ_b0_f) + delta_f[b0][b1]*(PA_0_f - PQ_a0_f)) + PB_1_f*delta_f[a0][a1]*(PB_0_f - PQ_b0_f) + PB_1_f*delta_f[a1][b0]*(PA_0_f - PQ_a0_f) - PQ_a1_f*(PA_0_f*delta_f[b0][b1] + PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) - PQ_b1_f*(PA_0_f*delta_f[a1][b0] + PA_1_f*delta_f[a0][b0] + PB_0_f*delta_f[a0][a1]))
                        -PQ_c0_f*(PA_0_f*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + PA_1_f*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + PB_0_f*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + PB_1_f*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0])) - PQ_d0_f*(PA_0_f*(delta_f[a1][b0]*delta_f[b1][c0] + delta_f[a1][b1]*delta_f[b0][c0] + delta_f[a1][c0]*delta_f[b0][b1]) + PA_1_f*(delta_f[a0][b0]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[b0][c0] + delta_f[a0][c0]*delta_f[b0][b1]) + PB_0_f*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1]) + PB_1_f*(delta_f[a0][a1]*delta_f[b0][c0] + delta_f[a0][b0]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b0]))
                        +PQ_c0_f*PQ_d0_f*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.5f * S1_f * inv_S4_f * inv_S4_f * (
                        -delta_f[c0][d0]*(PA_0_f*PA_1_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PB_1_f*(PA_0_f*PQ_a1_f + PA_1_f*PQ_a0_f))
                        -PA_0_f*PA_1_f*(PB_0_f*(PQ_c0_f*delta_f[b1][d0] + PQ_d0_f*delta_f[b1][c0]) + PB_1_f*(PQ_c0_f*delta_f[b0][d0] + PQ_d0_f*delta_f[b0][c0])) - PB_0_f*PB_1_f*(PA_0_f*(PQ_c0_f*delta_f[a1][d0] + PQ_d0_f*delta_f[a1][c0]) + PA_1_f*(PQ_c0_f*delta_f[a0][d0] + PQ_d0_f*delta_f[a0][c0]))
                        +PQ_c0_f*PQ_d0_f*(PA_0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) + PA_1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PB_0_f*PB_1_f*delta_f[a0][a1])
                    )

                    + 0.5f * S2_f * S2_f * inv_S1_f * inv_S4_f * inv_S4_f * (
                        +QC_0_f*QD_0_f*delta_f[b0][b1]*(-PA_0_f*PQ_a1_f + PQ_a0_f*(-PA_1_f + PQ_a1_f))
                        +QC_0_f*QD_0_f*(PQ_a0_f*delta_f[a1][b0]*(-PB_1_f + PQ_b1_f) + PQ_a0_f*delta_f[a1][b1]*(-PB_0_f + PQ_b0_f) + PQ_a1_f*delta_f[a0][b0]*(-PB_1_f + PQ_b1_f) + PQ_a1_f*delta_f[a0][b1]*(-PB_0_f + PQ_b0_f) + PQ_b0_f*PQ_b1_f*delta_f[a0][a1] - PQ_b0_f*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) - PQ_b1_f*(PA_0_f*delta_f[a1][b0] + PA_1_f*delta_f[a0][b0] + PB_0_f*delta_f[a0][a1]))
                    )

                    + 0.5f * S2_f * inv_S4_f * inv_S4_f * (
                        +QD_0_f*delta_f[b1][c0]*(PA_0_f*(PA_1_f*PQ_b0_f + PB_0_f*PQ_a1_f) + PA_1_f*PB_0_f*PQ_a0_f)
                        +PB_1_f*(PA_0_f*(PQ_a1_f*(QC_0_f*delta_f[b0][d0] + QD_0_f*delta_f[b0][c0]) + PQ_b0_f*(QC_0_f*delta_f[a1][d0] + QD_0_f*delta_f[a1][c0])) + PQ_a0_f*(PA_1_f*(QC_0_f*delta_f[b0][d0] + QD_0_f*delta_f[b0][c0]) + PB_0_f*(QC_0_f*delta_f[a1][d0] + QD_0_f*delta_f[a1][c0]))) + QC_0_f*(PA_0_f*(PA_1_f*(PQ_b0_f*delta_f[b1][d0] + PQ_b1_f*delta_f[b0][d0]) + PB_0_f*(PQ_a1_f*delta_f[b1][d0] + PQ_b1_f*delta_f[a1][d0])) + PA_1_f*PB_0_f*(PQ_a0_f*delta_f[b1][d0] + PQ_b1_f*delta_f[a0][d0]) + PB_1_f*delta_f[a0][d0]*(PA_1_f*PQ_b0_f + PB_0_f*PQ_a1_f)) + QD_0_f*(PA_0_f*PQ_b1_f*(PA_1_f*delta_f[b0][c0] + PB_0_f*delta_f[a1][c0]) + delta_f[a0][c0]*(PA_1_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PB_1_f*PQ_a1_f))
                        +(PQ_c0_f*QD_0_f + PQ_d0_f*QC_0_f)*(PA_0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) - PA_0_f*(PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) + PA_1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) - PA_1_f*(PQ_a0_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][b0]) + PB_0_f*PB_1_f*delta_f[a0][a1] - PB_0_f*(PQ_a0_f*delta_f[a1][b1] + PQ_a1_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][a1]) - PB_1_f*(PQ_a0_f*delta_f[a1][b0] + PQ_a1_f*delta_f[a0][b0] + PQ_b0_f*delta_f[a0][a1]))
                        +delta_f[c0][d0]*(PA_0_f*PQ_b0_f*(PA_1_f*PQ_b1_f + PB_1_f*PQ_a1_f) + PA_1_f*PQ_a0_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PQ_a1_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))
                    )

                    + S1_f * S1_f * inv_S4_f * inv_S4_f * (
                        +PA_0_f*PA_1_f*PB_0_f*PB_1_f*PQ_c0_f*PQ_d0_f
                    )

                    + S1_f * S2_f * inv_S4_f * inv_S4_f * (
                        -(PQ_c0_f*QD_0_f + PQ_d0_f*QC_0_f)*(PA_0_f*PA_1_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PB_1_f*(PA_0_f*PQ_a1_f + PA_1_f*PQ_a0_f))
                    )

                    + S2_f * S2_f * inv_S4_f * inv_S4_f * (
                        +QC_0_f*QD_0_f*(PA_0_f*PQ_b0_f*(PA_1_f*PQ_b1_f + PB_1_f*PQ_a1_f) + PA_1_f*PQ_a0_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PQ_a1_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))
                    )

                    + 0.125f * S2_f * inv_S1_f * inv_S1_f * inv_S4_f * inv_S4_f * (
                        +delta_f[c0][d0]*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.125f * inv_S1_f * inv_S4_f * inv_S4_f * (
                        +2.0f*delta_f[c0][d0]*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                        +delta_f[b0][b1]*(delta_f[a0][c0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][c0]) + delta_f[b0][c0]*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + delta_f[b0][d0]*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1]) + delta_f[b1][c0]*(delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0]) + delta_f[b1][d0]*(delta_f[a0][b0]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b0])
                    )

                )

                + F6_t_f[3] * (

                    (-0.125f) * S2_f * inv_S1_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +delta_f[a0][a1]*(delta_f[b0][b1]*delta_f[c0][d0] + delta_f[b0][c0]*delta_f[b1][d0] + delta_f[b0][d0]*delta_f[b1][c0]) + delta_f[a0][b0]*(delta_f[a1][b1]*delta_f[c0][d0] + delta_f[a1][c0]*delta_f[b1][d0] + delta_f[a1][d0]*delta_f[b1][c0]) + delta_f[a0][b1]*(delta_f[a1][b0]*delta_f[c0][d0] + delta_f[a1][c0]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][c0]) + delta_f[a0][c0]*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + delta_f[a0][d0]*(delta_f[a1][b0]*delta_f[b1][c0] + delta_f[a1][b1]*delta_f[b0][c0] + delta_f[a1][c0]*delta_f[b0][b1])
                    )

                    + (-0.25f) * S2_f * S2_f * inv_S1_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_a0_f*PQ_a1_f*delta_f[b0][b1]*delta_f[c0][d0]
                        +delta_f[c0][d0]*(PQ_b0_f*(PQ_a0_f*delta_f[a1][b1] + PQ_a1_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][a1]) + PQ_b1_f*(PQ_a0_f*delta_f[a1][b0] + PQ_a1_f*delta_f[a0][b0]))
                        +QC_0_f*(PQ_a0_f*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + PQ_a1_f*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + PQ_b0_f*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + PQ_b1_f*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0])) + QD_0_f*(PQ_a0_f*(delta_f[a1][b0]*delta_f[b1][c0] + delta_f[a1][b1]*delta_f[b0][c0] + delta_f[a1][c0]*delta_f[b0][b1]) + PQ_a1_f*(delta_f[a0][b0]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[b0][c0] + delta_f[a0][c0]*delta_f[b0][b1]) + PQ_b0_f*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1]) + PQ_b1_f*(delta_f[a0][a1]*delta_f[b0][c0] + delta_f[a0][b0]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b0]))
                        +(PQ_c0_f*QD_0_f + PQ_d0_f*QC_0_f)*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.25f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +delta_f[a1][b1]*delta_f[c0][d0]*(PB_0_f*PQ_a0_f + PQ_b0_f*(PA_0_f - PQ_a0_f))
                        +(PA_0_f*PQ_a1_f + PA_1_f*PQ_a0_f)*(delta_f[b0][c0]*delta_f[b1][d0] + delta_f[b0][d0]*delta_f[b1][c0]) + (PA_0_f*PQ_b0_f + PB_0_f*PQ_a0_f)*(delta_f[a1][c0]*delta_f[b1][d0] + delta_f[a1][d0]*delta_f[b1][c0]) + (PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f)*(delta_f[a1][c0]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][c0]) + (PA_1_f*PQ_b0_f + PB_0_f*PQ_a1_f)*(delta_f[a0][c0]*delta_f[b1][d0] + delta_f[a0][d0]*delta_f[b1][c0]) + (PA_1_f*PQ_b1_f + PB_1_f*PQ_a1_f)*(delta_f[a0][c0]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][c0]) + (PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f)*(delta_f[a0][c0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][c0])
                        +delta_f[c0][d0]*(PA_1_f*(PQ_a0_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][b0]) + PB_1_f*(PQ_a0_f*delta_f[a1][b0] + PQ_b0_f*delta_f[a0][a1]) + PQ_a1_f*(PA_0_f*delta_f[b0][b1] + PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0] - PQ_a0_f*delta_f[b0][b1] - PQ_b0_f*delta_f[a0][b1] - PQ_b1_f*delta_f[a0][b0]) + PQ_b1_f*(delta_f[a0][a1]*(PB_0_f - PQ_b0_f) + delta_f[a1][b0]*(PA_0_f - PQ_a0_f)))
                        +(PA_0_f*PQ_c0_f - PQ_a0_f*PQ_c0_f)*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + (PA_0_f*PQ_d0_f - PQ_a0_f*PQ_d0_f)*(delta_f[a1][b0]*delta_f[b1][c0] + delta_f[a1][b1]*delta_f[b0][c0] + delta_f[a1][c0]*delta_f[b0][b1]) + (PA_1_f*PQ_c0_f - PQ_a1_f*PQ_c0_f)*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + (PA_1_f*PQ_d0_f - PQ_a1_f*PQ_d0_f)*(delta_f[a0][b0]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[b0][c0] + delta_f[a0][c0]*delta_f[b0][b1]) + (PB_0_f*PQ_c0_f - PQ_b0_f*PQ_c0_f)*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + (PB_0_f*PQ_d0_f - PQ_b0_f*PQ_d0_f)*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1]) + (PB_1_f*PQ_c0_f - PQ_b1_f*PQ_c0_f)*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0]) + (PB_1_f*PQ_d0_f - PQ_b1_f*PQ_d0_f)*(delta_f[a0][a1]*delta_f[b0][c0] + delta_f[a0][b0]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b0])
                        -2*PQ_c0_f*PQ_d0_f*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.5f * S1_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        -delta_f[c0][d0]*(PA_0_f*PQ_b0_f*(PA_1_f*PQ_b1_f + PB_1_f*PQ_a1_f) + PA_1_f*PQ_a0_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PQ_a1_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))
                        -PA_0_f*(PA_1_f*(PQ_b0_f*(PQ_c0_f*delta_f[b1][d0] + PQ_d0_f*delta_f[b1][c0]) + PQ_b1_f*(PQ_c0_f*delta_f[b0][d0] + PQ_d0_f*delta_f[b0][c0])) + PB_0_f*(PQ_a1_f*(PQ_c0_f*delta_f[b1][d0] + PQ_d0_f*delta_f[b1][c0]) + PQ_b1_f*(PQ_c0_f*delta_f[a1][d0] + PQ_d0_f*delta_f[a1][c0])) + PB_1_f*(PQ_a1_f*(PQ_c0_f*delta_f[b0][d0] + PQ_d0_f*delta_f[b0][c0]) + PQ_b0_f*(PQ_c0_f*delta_f[a1][d0] + PQ_d0_f*delta_f[a1][c0]))) - PA_1_f*(PB_0_f*(PQ_a0_f*(PQ_c0_f*delta_f[b1][d0] + PQ_d0_f*delta_f[b1][c0]) + PQ_b1_f*(PQ_c0_f*delta_f[a0][d0] + PQ_d0_f*delta_f[a0][c0])) + PB_1_f*(PQ_a0_f*(PQ_c0_f*delta_f[b0][d0] + PQ_d0_f*delta_f[b0][c0]) + PQ_b0_f*(PQ_c0_f*delta_f[a0][d0] + PQ_d0_f*delta_f[a0][c0]))) - PB_0_f*PB_1_f*(PQ_a0_f*(PQ_c0_f*delta_f[a1][d0] + PQ_d0_f*delta_f[a1][c0]) + PQ_a1_f*(PQ_c0_f*delta_f[a0][d0] + PQ_d0_f*delta_f[a0][c0]))
                        +PQ_c0_f*PQ_d0_f*(PA_0_f*(-PA_1_f*delta_f[b0][b1] - PB_0_f*delta_f[a1][b1] - PB_1_f*delta_f[a1][b0] + PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) + PA_1_f*(PQ_a0_f*delta_f[b0][b1] + delta_f[a0][b0]*(-PB_1_f + PQ_b1_f) + delta_f[a0][b1]*(-PB_0_f + PQ_b0_f)) + PB_0_f*(PQ_a0_f*delta_f[a1][b1] + PQ_a1_f*delta_f[a0][b1] + delta_f[a0][a1]*(-PB_1_f + PQ_b1_f)) + PB_1_f*(PQ_a0_f*delta_f[a1][b0] + PQ_a1_f*delta_f[a0][b0] + PQ_b0_f*delta_f[a0][a1]))
                    )

                    + (-0.5f) * S2_f * S2_f * S2_f * inv_S1_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_a0_f*PQ_a1_f*QC_0_f*QD_0_f*delta_f[b0][b1]
                        +QC_0_f*QD_0_f*(PQ_b0_f*(PQ_a0_f*delta_f[a1][b1] + PQ_a1_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][a1]) + PQ_b1_f*(PQ_a0_f*delta_f[a1][b0] + PQ_a1_f*delta_f[a0][b0]))
                    )

                    + 0.5f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +QD_0_f*delta_f[b1][c0]*(PA_1_f*PQ_a0_f*PQ_b0_f + PQ_a1_f*(PA_0_f*PQ_b0_f + PB_0_f*PQ_a0_f))
                        +PB_1_f*PQ_a0_f*(PQ_a1_f*(QC_0_f*delta_f[b0][d0] + QD_0_f*delta_f[b0][c0]) + PQ_b0_f*(QC_0_f*delta_f[a1][d0] + QD_0_f*delta_f[a1][c0])) + QC_0_f*(PQ_a0_f*(PA_1_f*(PQ_b0_f*delta_f[b1][d0] + PQ_b1_f*delta_f[b0][d0]) + PB_0_f*(PQ_a1_f*delta_f[b1][d0] + PQ_b1_f*delta_f[a1][d0])) + PQ_a1_f*(PA_0_f*(PQ_b0_f*delta_f[b1][d0] + PQ_b1_f*delta_f[b0][d0]) + delta_f[a0][d0]*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f)) + PQ_b0_f*PQ_b1_f*(PA_0_f*delta_f[a1][d0] + PA_1_f*delta_f[a0][d0])) + QD_0_f*(PB_1_f*PQ_a1_f*PQ_b0_f*delta_f[a0][c0] + PQ_b1_f*(PA_0_f*(PQ_a1_f*delta_f[b0][c0] + PQ_b0_f*delta_f[a1][c0]) + PA_1_f*(PQ_a0_f*delta_f[b0][c0] + PQ_b0_f*delta_f[a0][c0]) + PB_0_f*(PQ_a0_f*delta_f[a1][c0] + PQ_a1_f*delta_f[a0][c0])))
                        +(PQ_c0_f*QD_0_f + PQ_d0_f*QC_0_f)*(PA_0_f*(PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) + PQ_a0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0] - PQ_a1_f*delta_f[b0][b1] - PQ_b0_f*delta_f[a1][b1] - PQ_b1_f*delta_f[a1][b0]) + PQ_a1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PQ_b0_f*(PB_1_f*delta_f[a0][a1] + delta_f[a0][b1]*(PA_1_f - PQ_a1_f)) + PQ_b1_f*(delta_f[a0][a1]*(PB_0_f - PQ_b0_f) + delta_f[a0][b0]*(PA_1_f - PQ_a1_f)))
                        +delta_f[c0][d0]*(PQ_a0_f*PQ_b1_f*(PA_1_f*PQ_b0_f + PB_0_f*PQ_a1_f) + PQ_a1_f*PQ_b0_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))
                    )

                    + S1_f * S1_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_c0_f*PQ_d0_f*(PA_0_f*PA_1_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PB_1_f*(PA_0_f*PQ_a1_f + PA_1_f*PQ_a0_f))
                    )

                    + S1_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        -(PQ_c0_f*QD_0_f + PQ_d0_f*QC_0_f)*(PA_0_f*PQ_b0_f*(PA_1_f*PQ_b1_f + PB_1_f*PQ_a1_f) + PA_1_f*PQ_a0_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PQ_a1_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))
                    )

                    + S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +QC_0_f*QD_0_f*(PQ_a0_f*PQ_b1_f*(PA_1_f*PQ_b0_f + PB_0_f*PQ_a1_f) + PQ_a1_f*PQ_b0_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))
                    )

                )

                + F6_t_f[4] * (

                    0.5f * S1_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        -delta_f[c0][d0]*(PQ_a0_f*PQ_b1_f*(PA_1_f*PQ_b0_f + PB_0_f*PQ_a1_f) + PQ_a1_f*PQ_b0_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))
                        -PQ_a0_f*(PA_1_f*(PQ_b0_f*(PQ_c0_f*delta_f[b1][d0] + PQ_d0_f*delta_f[b1][c0]) + PQ_b1_f*(PQ_c0_f*delta_f[b0][d0] + PQ_d0_f*delta_f[b0][c0])) + PB_0_f*(PQ_a1_f*(PQ_c0_f*delta_f[b1][d0] + PQ_d0_f*delta_f[b1][c0]) + PQ_b1_f*(PQ_c0_f*delta_f[a1][d0] + PQ_d0_f*delta_f[a1][c0])) + PB_1_f*(PQ_a1_f*(PQ_c0_f*delta_f[b0][d0] + PQ_d0_f*delta_f[b0][c0]) + PQ_b0_f*(PQ_c0_f*delta_f[a1][d0] + PQ_d0_f*delta_f[a1][c0]))) - PQ_a1_f*(PA_0_f*(PQ_b0_f*(PQ_c0_f*delta_f[b1][d0] + PQ_d0_f*delta_f[b1][c0]) + PQ_b1_f*(PQ_c0_f*delta_f[b0][d0] + PQ_d0_f*delta_f[b0][c0])) + (PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f)*(PQ_c0_f*delta_f[a0][d0] + PQ_d0_f*delta_f[a0][c0])) - PQ_b0_f*PQ_b1_f*(PA_0_f*(PQ_c0_f*delta_f[a1][d0] + PQ_d0_f*delta_f[a1][c0]) + PA_1_f*(PQ_c0_f*delta_f[a0][d0] + PQ_d0_f*delta_f[a0][c0]))
                        +PQ_c0_f*PQ_d0_f*(-PA_0_f*(PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) - PQ_a0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) + PQ_a0_f*(PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) - PQ_a1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PQ_a1_f*(PQ_b0_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][b0]) + PQ_b0_f*PQ_b1_f*delta_f[a0][a1] - PQ_b0_f*(PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) - PQ_b1_f*(PA_1_f*delta_f[a0][b0] + PB_0_f*delta_f[a0][a1]))
                    )

                    + 0.5f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_a0_f*PQ_a1_f*PQ_b0_f*QD_0_f*delta_f[b1][c0]
                        +PQ_a0_f*PQ_a1_f*QC_0_f*(PQ_b0_f*delta_f[b1][d0] + PQ_b1_f*delta_f[b0][d0]) + PQ_b1_f*(PQ_a0_f*QD_0_f*(PQ_a1_f*delta_f[b0][c0] + PQ_b0_f*delta_f[a1][c0]) + PQ_b0_f*(PQ_a1_f*QD_0_f*delta_f[a0][c0] + QC_0_f*(PQ_a0_f*delta_f[a1][d0] + PQ_a1_f*delta_f[a0][d0])))
                        +(PQ_c0_f*QD_0_f + PQ_d0_f*QC_0_f)*(PQ_a0_f*(PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) + PQ_a1_f*(PQ_b0_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][b0]) + PQ_b0_f*PQ_b1_f*delta_f[a0][a1])
                        +PQ_a0_f*PQ_a1_f*PQ_b0_f*PQ_b1_f*delta_f[c0][d0]
                    )

                    + S1_f * S1_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_c0_f*PQ_d0_f*(PA_0_f*PQ_b0_f*(PA_1_f*PQ_b1_f + PB_1_f*PQ_a1_f) + PA_1_f*PQ_a0_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PQ_a1_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))
                    )

                    + S1_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        -(PQ_c0_f*QD_0_f + PQ_d0_f*QC_0_f)*(PQ_a0_f*PQ_b1_f*(PA_1_f*PQ_b0_f + PB_0_f*PQ_a1_f) + PQ_a1_f*PQ_b0_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))
                    )

                    + S2_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_a0_f*PQ_a1_f*PQ_b0_f*PQ_b1_f*QC_0_f*QD_0_f
                    )

                    + 0.25f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_a0_f*PQ_a1_f*(delta_f[b0][b1]*delta_f[c0][d0] + delta_f[b0][c0]*delta_f[b1][d0] + delta_f[b0][d0]*delta_f[b1][c0])
                        +PQ_b0_f*(PQ_a0_f*(delta_f[a1][b1]*delta_f[c0][d0] + delta_f[a1][c0]*delta_f[b1][d0] + delta_f[a1][d0]*delta_f[b1][c0]) + PQ_a1_f*(delta_f[a0][b1]*delta_f[c0][d0] + delta_f[a0][c0]*delta_f[b1][d0] + delta_f[a0][d0]*delta_f[b1][c0]) + PQ_b1_f*(delta_f[a0][a1]*delta_f[c0][d0] + delta_f[a0][c0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][c0]) + PQ_c0_f*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + PQ_d0_f*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1])) + PQ_b1_f*(PQ_a0_f*(delta_f[a1][b0]*delta_f[c0][d0] + delta_f[a1][c0]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][c0]) + PQ_a1_f*(delta_f[a0][b0]*delta_f[c0][d0] + delta_f[a0][c0]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][c0]) + PQ_c0_f*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0]) + PQ_d0_f*(delta_f[a0][a1]*delta_f[b0][c0] + delta_f[a0][b0]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b0])) + PQ_c0_f*(PQ_a0_f*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + PQ_a1_f*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + PQ_d0_f*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])) + PQ_d0_f*(PQ_a0_f*(delta_f[a1][b0]*delta_f[b1][c0] + delta_f[a1][b1]*delta_f[b0][c0] + delta_f[a1][c0]*delta_f[b0][b1]) + PQ_a1_f*(delta_f[a0][b0]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[b0][c0] + delta_f[a0][c0]*delta_f[b0][b1]))
                    )

                )

                + F6_t_f[5] * (

                    (-0.5f) * S1_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_a0_f*PQ_a1_f*PQ_b0_f*PQ_b1_f*delta_f[c0][d0]
                        +PQ_c0_f*(PQ_a0_f*(PQ_a1_f*(PQ_b0_f*delta_f[b1][d0] + PQ_b1_f*delta_f[b0][d0] + PQ_d0_f*delta_f[b0][b1]) + PQ_b1_f*PQ_d0_f*delta_f[a1][b0]) + PQ_b0_f*PQ_b1_f*(PQ_a0_f*delta_f[a1][d0] + PQ_a1_f*delta_f[a0][d0] + PQ_d0_f*delta_f[a0][a1])) + PQ_d0_f*(PQ_a1_f*PQ_b1_f*(PQ_a0_f*delta_f[b0][c0] + PQ_b0_f*delta_f[a0][c0] + PQ_c0_f*delta_f[a0][b0]) + PQ_b0_f*(PQ_a0_f*(PQ_a1_f*delta_f[b1][c0] + PQ_b1_f*delta_f[a1][c0] + PQ_c0_f*delta_f[a1][b1]) + PQ_a1_f*PQ_c0_f*delta_f[a0][b1]))
                    )

                    + S1_f * S1_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_c0_f*PQ_d0_f*(PQ_a0_f*PQ_b1_f*(PA_1_f*PQ_b0_f + PB_0_f*PQ_a1_f) + PQ_a1_f*PQ_b0_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))
                    )

                    + S1_f * S2_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_a0_f*PQ_a1_f*PQ_b0_f*PQ_b1_f*(-PQ_c0_f*QD_0_f - PQ_d0_f*QC_0_f)
                    )

                )

                + F6_t_f[6] * (

                    S1_f * S1_f * S2_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_a0_f*PQ_a1_f*PQ_b0_f*PQ_b1_f*PQ_c0_f*PQ_d0_f
                    )

                )

                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0f) == (k == l ? 1.0f : 2.0f)
        const float D_f = pp_mat_D_f[kl];
        const float sym_f = (k != l) ? 2.0f : 1.0f;
        const float contrib_f = eri_ijkl_f * D_f * sym_f;
        ERIs[threadIdx.y] += (double)contrib_f;
    }

    __syncthreads();

    if ((threadIdx.y == 0) && (ij < dd_prim_pair_count_local))
    {
        double J_ij = 0.0f;

        for (uint32_t n = 0; n < TILE_DIM_LARGE; n++)
        {
            J_ij += ERIs[n];
        }

        mat_J[ij] += J_ij;
    }
}

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDPD0_FP32_auto_s_ast(double*         mat_J,
                       const float*    p_prim_info_f,
                       const uint32_t  p_prim_count,
                       const float*    d_prim_info_f,
                       const uint32_t  d_prim_count,
                       const float*    pd_mat_D_f,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const float*    dd_pair_data_local_f,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* pd_first_inds,
                       const uint32_t* pd_second_inds,
                       const float*    pd_pair_data_f,
                       const uint32_t  pd_prim_pair_count,
                       const float*    boys_func_table_f,
                       const float*    boys_func_ft_f,
                       const uint32_t* prec_cut_ij_tile,
                       const uint32_t* screen_cut_ij_tile)
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ float   delta_f[3][3];

    __shared__ float a_i_f, a_j_f, r_i_f[3], r_j_f[3], S_ij_00_f, S1_f, inv_S1_f;
    __shared__ float PA_0_f, PA_1_f, PB_0_f, PB_1_f;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];
    const uint32_t screen_cut = screen_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta_f[0][0] = 1.0f; delta_f[0][1] = 0.0f; delta_f[0][2] = 0.0f;
        delta_f[1][0] = 0.0f; delta_f[1][1] = 1.0f; delta_f[1][2] = 0.0f;
        delta_f[2][0] = 0.0f; delta_f[2][1] = 0.0f; delta_f[2][2] = 1.0f;

        if (ij < dd_prim_pair_count_local)
        {
            i = dd_first_inds_local[ij];
            j = dd_second_inds_local[ij];

            a_i_f = d_prim_info_f[i / 6 + d_prim_count * 0];

            r_i_f[0] = d_prim_info_f[i / 6 + d_prim_count * 2];
            r_i_f[1] = d_prim_info_f[i / 6 + d_prim_count * 3];
            r_i_f[2] = d_prim_info_f[i / 6 + d_prim_count * 4];

            a_j_f = d_prim_info_f[j / 6 + d_prim_count * 0];

            r_j_f[0] = d_prim_info_f[j / 6 + d_prim_count * 2];
            r_j_f[1] = d_prim_info_f[j / 6 + d_prim_count * 3];
            r_j_f[2] = d_prim_info_f[j / 6 + d_prim_count * 4];

            S1_f = a_i_f + a_j_f;
            inv_S1_f = (float) (1.0 / (double)S1_f);

            S_ij_00_f = dd_pair_data_local_f[ij];

            a0 = d_cart_inds[i % 6][0];
            a1 = d_cart_inds[i % 6][1];
            b0 = d_cart_inds[j % 6][0];
            b1 = d_cart_inds[j % 6][1];

            PA_0_f = (a_j_f  * inv_S1_f) * (r_j_f[a0] - r_i_f[a0]);
            PA_1_f = (a_j_f  * inv_S1_f) * (r_j_f[a1] - r_i_f[a1]);
            PB_0_f = (-a_i_f * inv_S1_f) * (r_j_f[b0] - r_i_f[b0]);
            PB_1_f = (-a_i_f * inv_S1_f) * (r_j_f[b1] - r_i_f[b1]);

        }

    }

    ERIs[threadIdx.y] = 0.0;

    __syncthreads();

    for (uint32_t m = prec_cut; m < screen_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= pd_prim_pair_count))
        {
            break;
        }

        const auto k = pd_first_inds[kl];
        const auto l = pd_second_inds[kl];

        const auto a_k_f = p_prim_info_f[k / 3 + p_prim_count * 0];

        const float r_k0_f = p_prim_info_f[k / 3 + p_prim_count * 2];
        const float r_k1_f = p_prim_info_f[k / 3 + p_prim_count * 3];
        const float r_k2_f = p_prim_info_f[k / 3 + p_prim_count * 4];

        const auto a_l_f = d_prim_info_f[l / 6 + d_prim_count * 0];

        const float r_l0_f = d_prim_info_f[l / 6 + d_prim_count * 2];
        const float r_l1_f = d_prim_info_f[l / 6 + d_prim_count * 3];
        const float r_l2_f = d_prim_info_f[l / 6 + d_prim_count * 4];

        const auto S_kl_00_f = pd_pair_data_f[kl];

        const auto c0 = k % 3;
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        const float r_l_c0_f = (c0 == 0 ? r_l0_f : (c0 == 1 ? r_l1_f : r_l2_f));
        const float r_l_d0_f = (d0 == 0 ? r_l0_f : (d0 == 1 ? r_l1_f : r_l2_f));
        const float r_l_d1_f = (d1 == 0 ? r_l0_f : (d1 == 1 ? r_l1_f : r_l2_f));

        const float r_k_c0_f = (c0 == 0 ? r_k0_f : (c0 == 1 ? r_k1_f : r_k2_f));
        const float r_k_d0_f = (d0 == 0 ? r_k0_f : (d0 == 1 ? r_k1_f : r_k2_f));
        const float r_k_d1_f = (d1 == 0 ? r_k0_f : (d1 == 1 ? r_k1_f : r_k2_f));

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2_f = a_k_f + a_l_f;

        const auto inv_S2_f = 1.0f / S2_f;
        const auto inv_S4_f = 1.0f / (S1_f + S2_f);

        const float PQ0_f = (a_k_f * r_k0_f + a_l_f * r_l0_f) * inv_S2_f - (a_i_f * r_i_f[0] + a_j_f * r_j_f[0]) * inv_S1_f;
        const float PQ1_f = (a_k_f * r_k1_f + a_l_f * r_l1_f) * inv_S2_f - (a_i_f * r_i_f[1] + a_j_f * r_j_f[1]) * inv_S1_f;
        const float PQ2_f = (a_k_f * r_k2_f + a_l_f * r_l2_f) * inv_S2_f - (a_i_f * r_i_f[2] + a_j_f * r_j_f[2]) * inv_S1_f;

        const float PQ_c0_f = (c0 == 0 ? PQ0_f : (c0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_d0_f = (d0 == 0 ? PQ0_f : (d0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_d1_f = (d1 == 0 ? PQ0_f : (d1 == 1 ? PQ1_f : PQ2_f));

        const auto r2_PQ_f = PQ0_f * PQ0_f + PQ1_f * PQ1_f + PQ2_f * PQ2_f;

        const auto Lambda_f = sqrtf(4.0f * S1_f * S2_f * MATH_CONST_INV_PI_F * inv_S4_f);

        float F7_t_f[2];

        gpu::computeBoysFunction_f(F7_t_f, S1_f * S2_f * inv_S4_f * r2_PQ_f, 1, boys_func_table_f, boys_func_ft_f);

        const auto QC_0_f = (a_l_f * inv_S2_f) * (r_l_c0_f - r_k_c0_f);
        const auto QD_0_f = (-a_k_f * inv_S2_f) * (r_l_d0_f - r_k_d0_f);
        const auto QD_1_f = (-a_k_f * inv_S2_f) * (r_l_d1_f - r_k_d1_f);

        const float eri_ijkl_f = Lambda_f * S_ij_00_f * S_kl_00_f * (

                    + F7_t_f[0] * 0.125f * inv_S1_f * inv_S1_f * inv_S2_f * (

                        +(QC_0_f*delta_f[d0][d1] + QD_0_f*delta_f[c0][d1] + QD_1_f*delta_f[c0][d0])*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])

                    )

                    + F7_t_f[0] * 0.25f * inv_S1_f * inv_S1_f * (

                        +QC_0_f*QD_0_f*QD_1_f*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])

                    )

                    + F7_t_f[0] * 0.25f * inv_S1_f * inv_S2_f * (

                        +(PA_0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) + PA_1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PB_0_f*PB_1_f*delta_f[a0][a1])*(QC_0_f*delta_f[d0][d1] + QD_0_f*delta_f[c0][d1] + QD_1_f*delta_f[c0][d0])

                    )

                    + F7_t_f[0] * 0.5f * inv_S1_f * (

                        +QC_0_f*QD_0_f*QD_1_f*(PA_0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) + PA_1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PB_0_f*PB_1_f*delta_f[a0][a1])

                    )

                    + F7_t_f[0] * 0.5f * inv_S2_f * (

                        +PA_0_f*PA_1_f*PB_0_f*PB_1_f*(QC_0_f*delta_f[d0][d1] + QD_0_f*delta_f[c0][d1] + QD_1_f*delta_f[c0][d0])

                    )

                    + F7_t_f[0] * (

                        +PA_0_f*PA_1_f*PB_0_f*PB_1_f*QC_0_f*QD_0_f*QD_1_f

                    )

                    + F7_t_f[1] * (-0.25f) * inv_S1_f * inv_S1_f * inv_S4_f * (

                        +(QC_0_f*delta_f[d0][d1] + QD_0_f*delta_f[c0][d1] + QD_1_f*delta_f[c0][d0])*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])

                    )

                    + F7_t_f[1] * 0.125f * inv_S1_f * inv_S2_f * inv_S4_f * (

                        +PA_0_f*(delta_f[a1][b0]*(delta_f[b1][c0]*delta_f[d0][d1] + delta_f[b1][d0]*delta_f[c0][d1] + delta_f[b1][d1]*delta_f[c0][d0]) + delta_f[a1][b1]*(delta_f[b0][c0]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][d0]) + delta_f[b0][b1]*(delta_f[a1][c0]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[c0][d1] + delta_f[a1][d1]*delta_f[c0][d0])) + PA_1_f*(delta_f[a0][b0]*(delta_f[b1][c0]*delta_f[d0][d1] + delta_f[b1][d0]*delta_f[c0][d1] + delta_f[b1][d1]*delta_f[c0][d0]) + delta_f[a0][b1]*(delta_f[b0][c0]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][d0]) + delta_f[b0][b1]*(delta_f[a0][c0]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[c0][d0])) + PB_0_f*(delta_f[a0][a1]*(delta_f[b1][c0]*delta_f[d0][d1] + delta_f[b1][d0]*delta_f[c0][d1] + delta_f[b1][d1]*delta_f[c0][d0]) + delta_f[a0][b1]*(delta_f[a1][c0]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[c0][d1] + delta_f[a1][d1]*delta_f[c0][d0]) + delta_f[a1][b1]*(delta_f[a0][c0]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[c0][d0])) + PB_1_f*(delta_f[a0][a1]*(delta_f[b0][c0]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][d0]) + delta_f[a0][b0]*(delta_f[a1][c0]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[c0][d1] + delta_f[a1][d1]*delta_f[c0][d0]) + delta_f[a1][b0]*(delta_f[a0][c0]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[c0][d0]))

                        -(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])*(delta_f[c0][d0]*(PQ_d1_f + QD_1_f) + delta_f[c0][d1]*(PQ_d0_f + QD_0_f) + delta_f[d0][d1]*(PQ_c0_f + QC_0_f))

                    )

                    + F7_t_f[1] * (-0.5f) * S2_f * inv_S1_f * inv_S1_f * inv_S4_f * (

                        +QC_0_f*QD_0_f*QD_1_f*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        const float D_f = pd_mat_D_f[kl];
        const float contrib_f = eri_ijkl_f * D_f * 2.0f;
        ERIs[threadIdx.y] += (double)contrib_f;
    }

    __syncthreads();

    if ((threadIdx.y == 0) && (ij < dd_prim_pair_count_local))
    {
        double J_ij = 0.0f;

        for (uint32_t n = 0; n < TILE_DIM_LARGE; n++)
        {
            J_ij += ERIs[n];
        }

        mat_J[ij] += J_ij;
    }
}

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDPD1_FP32_auto_s_ast(double*         mat_J,
                       const float*    p_prim_info_f,
                       const uint32_t  p_prim_count,
                       const float*    d_prim_info_f,
                       const uint32_t  d_prim_count,
                       const float*    pd_mat_D_f,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const float*    dd_pair_data_local_f,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* pd_first_inds,
                       const uint32_t* pd_second_inds,
                       const float*    pd_pair_data_f,
                       const uint32_t  pd_prim_pair_count,
                       const float*    boys_func_table_f,
                       const float*    boys_func_ft_f,
                       const uint32_t* prec_cut_ij_tile,
                       const uint32_t* screen_cut_ij_tile)
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ float   delta_f[3][3];

    __shared__ float a_i_f, a_j_f, r_i_f[3], r_j_f[3], S_ij_00_f, S1_f, inv_S1_f;
    __shared__ float PA_0_f, PA_1_f, PB_0_f, PB_1_f;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];
    const uint32_t screen_cut = screen_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta_f[0][0] = 1.0f; delta_f[0][1] = 0.0f; delta_f[0][2] = 0.0f;
        delta_f[1][0] = 0.0f; delta_f[1][1] = 1.0f; delta_f[1][2] = 0.0f;
        delta_f[2][0] = 0.0f; delta_f[2][1] = 0.0f; delta_f[2][2] = 1.0f;

        if (ij < dd_prim_pair_count_local)
        {
            i = dd_first_inds_local[ij];
            j = dd_second_inds_local[ij];

            a_i_f = d_prim_info_f[i / 6 + d_prim_count * 0];

            r_i_f[0] = d_prim_info_f[i / 6 + d_prim_count * 2];
            r_i_f[1] = d_prim_info_f[i / 6 + d_prim_count * 3];
            r_i_f[2] = d_prim_info_f[i / 6 + d_prim_count * 4];

            a_j_f = d_prim_info_f[j / 6 + d_prim_count * 0];

            r_j_f[0] = d_prim_info_f[j / 6 + d_prim_count * 2];
            r_j_f[1] = d_prim_info_f[j / 6 + d_prim_count * 3];
            r_j_f[2] = d_prim_info_f[j / 6 + d_prim_count * 4];

            S1_f = a_i_f + a_j_f;
            inv_S1_f = (float) (1.0 / (double)S1_f);

            S_ij_00_f = dd_pair_data_local_f[ij];

            a0 = d_cart_inds[i % 6][0];
            a1 = d_cart_inds[i % 6][1];
            b0 = d_cart_inds[j % 6][0];
            b1 = d_cart_inds[j % 6][1];

            PA_0_f = (a_j_f  * inv_S1_f) * (r_j_f[a0] - r_i_f[a0]);
            PA_1_f = (a_j_f  * inv_S1_f) * (r_j_f[a1] - r_i_f[a1]);
            PB_0_f = (-a_i_f * inv_S1_f) * (r_j_f[b0] - r_i_f[b0]);
            PB_1_f = (-a_i_f * inv_S1_f) * (r_j_f[b1] - r_i_f[b1]);

        }

    }

    ERIs[threadIdx.y] = 0.0;

    __syncthreads();

    for (uint32_t m = prec_cut; m < screen_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= pd_prim_pair_count))
        {
            break;
        }

        const auto k = pd_first_inds[kl];
        const auto l = pd_second_inds[kl];

        const auto a_k_f = p_prim_info_f[k / 3 + p_prim_count * 0];

        const float r_k0_f = p_prim_info_f[k / 3 + p_prim_count * 2];
        const float r_k1_f = p_prim_info_f[k / 3 + p_prim_count * 3];
        const float r_k2_f = p_prim_info_f[k / 3 + p_prim_count * 4];

        const auto a_l_f = d_prim_info_f[l / 6 + d_prim_count * 0];

        const float r_l0_f = d_prim_info_f[l / 6 + d_prim_count * 2];
        const float r_l1_f = d_prim_info_f[l / 6 + d_prim_count * 3];
        const float r_l2_f = d_prim_info_f[l / 6 + d_prim_count * 4];

        const auto S_kl_00_f = pd_pair_data_f[kl];

        const auto c0 = k % 3;
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        const float r_l_c0_f = (c0 == 0 ? r_l0_f : (c0 == 1 ? r_l1_f : r_l2_f));
        const float r_l_d0_f = (d0 == 0 ? r_l0_f : (d0 == 1 ? r_l1_f : r_l2_f));
        const float r_l_d1_f = (d1 == 0 ? r_l0_f : (d1 == 1 ? r_l1_f : r_l2_f));

        const float r_k_c0_f = (c0 == 0 ? r_k0_f : (c0 == 1 ? r_k1_f : r_k2_f));
        const float r_k_d0_f = (d0 == 0 ? r_k0_f : (d0 == 1 ? r_k1_f : r_k2_f));
        const float r_k_d1_f = (d1 == 0 ? r_k0_f : (d1 == 1 ? r_k1_f : r_k2_f));

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2_f = a_k_f + a_l_f;

        const auto inv_S2_f = 1.0f / S2_f;
        const auto inv_S4_f = 1.0f / (S1_f + S2_f);

        const float PQ0_f = (a_k_f * r_k0_f + a_l_f * r_l0_f) * inv_S2_f - (a_i_f * r_i_f[0] + a_j_f * r_j_f[0]) * inv_S1_f;
        const float PQ1_f = (a_k_f * r_k1_f + a_l_f * r_l1_f) * inv_S2_f - (a_i_f * r_i_f[1] + a_j_f * r_j_f[1]) * inv_S1_f;
        const float PQ2_f = (a_k_f * r_k2_f + a_l_f * r_l2_f) * inv_S2_f - (a_i_f * r_i_f[2] + a_j_f * r_j_f[2]) * inv_S1_f;

        const float PQ_a0_f = (a0 == 0 ? PQ0_f : (a0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_a1_f = (a1 == 0 ? PQ0_f : (a1 == 1 ? PQ1_f : PQ2_f));
        const float PQ_b0_f = (b0 == 0 ? PQ0_f : (b0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_b1_f = (b1 == 0 ? PQ0_f : (b1 == 1 ? PQ1_f : PQ2_f));
        const float PQ_c0_f = (c0 == 0 ? PQ0_f : (c0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_d0_f = (d0 == 0 ? PQ0_f : (d0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_d1_f = (d1 == 0 ? PQ0_f : (d1 == 1 ? PQ1_f : PQ2_f));

        const auto r2_PQ_f = PQ0_f * PQ0_f + PQ1_f * PQ1_f + PQ2_f * PQ2_f;

        const auto Lambda_f = sqrtf(4.0f * S1_f * S2_f * MATH_CONST_INV_PI_F * inv_S4_f);

        float F7_t_f[3];

        gpu::computeBoysFunction_f(F7_t_f, S1_f * S2_f * inv_S4_f * r2_PQ_f, 2, boys_func_table_f, boys_func_ft_f);

        const auto QC_0_f = (a_l_f * inv_S2_f) * (r_l_c0_f - r_k_c0_f);
        const auto QD_0_f = (-a_k_f * inv_S2_f) * (r_l_d0_f - r_k_d0_f);
        const auto QD_1_f = (-a_k_f * inv_S2_f) * (r_l_d1_f - r_k_d1_f);

        const float eri_ijkl_f = Lambda_f * S_ij_00_f * S_kl_00_f * (

                    + F7_t_f[1] * 0.25f * inv_S1_f * inv_S4_f * (

                        +(QC_0_f*delta_f[d0][d1] + QD_0_f*delta_f[c0][d1] + QD_1_f*delta_f[c0][d0])*(PA_0_f*(-PA_1_f*delta_f[b0][b1] - PB_0_f*delta_f[a1][b1] - PB_1_f*delta_f[a1][b0] + PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) + PA_1_f*(PQ_a0_f*delta_f[b0][b1] + delta_f[a0][b0]*(-PB_1_f + PQ_b1_f) + delta_f[a0][b1]*(-PB_0_f + PQ_b0_f)) + PB_0_f*(PQ_a0_f*delta_f[a1][b1] + PQ_a1_f*delta_f[a0][b1] + delta_f[a0][a1]*(-PB_1_f + PQ_b1_f)) + PB_1_f*(PQ_a0_f*delta_f[a1][b0] + PQ_a1_f*delta_f[a0][b0] + PQ_b0_f*delta_f[a0][a1]))

                        +QC_0_f*(QD_0_f*(PA_0_f*(delta_f[a1][b0]*delta_f[b1][d1] + delta_f[a1][b1]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][b1]) + PA_1_f*(delta_f[a0][b0]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][b1]) + PB_0_f*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + PB_1_f*(delta_f[a0][a1]*delta_f[b0][d1] + delta_f[a0][b0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b0])) + QD_1_f*(PA_0_f*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + PA_1_f*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + PB_0_f*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + PB_1_f*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0]))) + QD_0_f*QD_1_f*(PA_0_f*(delta_f[a1][b0]*delta_f[b1][c0] + delta_f[a1][b1]*delta_f[b0][c0] + delta_f[a1][c0]*delta_f[b0][b1]) + PA_1_f*(delta_f[a0][b0]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[b0][c0] + delta_f[a0][c0]*delta_f[b0][b1]) + PB_0_f*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1]) + PB_1_f*(delta_f[a0][a1]*delta_f[b0][c0] + delta_f[a0][b0]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b0]))

                        -(PQ_d0_f*QC_0_f*QD_1_f + QD_0_f*(PQ_c0_f*QD_1_f + PQ_d1_f*QC_0_f))*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])

                    )

                    + F7_t_f[1] * 0.25f * inv_S2_f * inv_S4_f * (

                        +PA_0_f*PA_1_f*(PB_0_f*(delta_f[b1][c0]*delta_f[d0][d1] + delta_f[b1][d0]*delta_f[c0][d1] + delta_f[b1][d1]*delta_f[c0][d0]) + PB_1_f*(delta_f[b0][c0]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][d0])) + PB_0_f*PB_1_f*(PA_0_f*(delta_f[a1][c0]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[c0][d1] + delta_f[a1][d1]*delta_f[c0][d0]) + PA_1_f*(delta_f[a0][c0]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[c0][d0]))

                        -(PA_0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) + PA_1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PB_0_f*PB_1_f*delta_f[a0][a1])*(delta_f[c0][d0]*(PQ_d1_f + QD_1_f) + delta_f[c0][d1]*(PQ_d0_f + QD_0_f) + delta_f[d0][d1]*(PQ_c0_f + QC_0_f))

                    )

                    + F7_t_f[1] * (-0.5f) * S1_f * inv_S2_f * inv_S4_f * (

                        +PA_0_f*PA_1_f*PB_0_f*PB_1_f*(delta_f[c0][d0]*(PQ_d1_f + QD_1_f) + delta_f[c0][d1]*(PQ_d0_f + QD_0_f) + delta_f[d0][d1]*(PQ_c0_f + QC_0_f))

                    )

                    + F7_t_f[1] * 0.5f * S2_f * inv_S1_f * inv_S4_f * (

                        +QC_0_f*QD_0_f*QD_1_f*(PA_0_f*(-PA_1_f*delta_f[b0][b1] - PB_0_f*delta_f[a1][b1] - PB_1_f*delta_f[a1][b0] + PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) + PA_1_f*(PQ_a0_f*delta_f[b0][b1] + delta_f[a0][b0]*(-PB_1_f + PQ_b1_f) + delta_f[a0][b1]*(-PB_0_f + PQ_b0_f)) + PB_0_f*(PQ_a0_f*delta_f[a1][b1] + PQ_a1_f*delta_f[a0][b1] + delta_f[a0][a1]*(-PB_1_f + PQ_b1_f)) + PB_1_f*(PQ_a0_f*delta_f[a1][b0] + PQ_a1_f*delta_f[a0][b0] + PQ_b0_f*delta_f[a0][a1]))

                    )

                    + F7_t_f[1] * 0.5f * inv_S4_f * (

                        +(PA_0_f*PA_1_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PB_1_f*(PA_0_f*PQ_a1_f + PA_1_f*PQ_a0_f))*(QC_0_f*delta_f[d0][d1] + QD_0_f*delta_f[c0][d1] + QD_1_f*delta_f[c0][d0])

                        +PA_0_f*PA_1_f*(QC_0_f*(PB_0_f*(QD_0_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][d0]) + PB_1_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0])) + QD_0_f*QD_1_f*(PB_0_f*delta_f[b1][c0] + PB_1_f*delta_f[b0][c0])) + PB_0_f*PB_1_f*(QC_0_f*(PA_0_f*(QD_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][d0]) + PA_1_f*(QD_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][d0])) + QD_0_f*QD_1_f*(PA_0_f*delta_f[a1][c0] + PA_1_f*delta_f[a0][c0]))

                        -(PQ_d0_f*QC_0_f*QD_1_f + QD_0_f*(PQ_c0_f*QD_1_f + PQ_d1_f*QC_0_f))*(PA_0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) + PA_1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PB_0_f*PB_1_f*delta_f[a0][a1])

                    )

                    + F7_t_f[1] * S1_f * inv_S4_f * (

                        +PA_0_f*PA_1_f*PB_0_f*PB_1_f*(-PQ_d0_f*QC_0_f*QD_1_f - QD_0_f*(PQ_c0_f*QD_1_f + PQ_d1_f*QC_0_f))

                    )

                    + F7_t_f[1] * S2_f * inv_S4_f * (

                        +QC_0_f*QD_0_f*QD_1_f*(PA_0_f*PA_1_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PB_1_f*(PA_0_f*PQ_a1_f + PA_1_f*PQ_a0_f))

                    )

                    + F7_t_f[2] * 0.125f * S2_f * inv_S1_f * inv_S1_f * inv_S4_f * inv_S4_f * (

                        +(QC_0_f*delta_f[d0][d1] + QD_0_f*delta_f[c0][d1] + QD_1_f*delta_f[c0][d0])*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])

                    )

                    + F7_t_f[2] * 0.125f * inv_S1_f * inv_S4_f * inv_S4_f * (

                        +(-PA_0_f + PQ_a0_f)*(delta_f[a1][b0]*delta_f[b1][c0]*delta_f[d0][d1] + delta_f[a1][b0]*delta_f[b1][d0]*delta_f[c0][d1] + delta_f[a1][b0]*delta_f[b1][d1]*delta_f[c0][d0] + delta_f[a1][b1]*delta_f[b0][c0]*delta_f[d0][d1] + delta_f[a1][b1]*delta_f[b0][d0]*delta_f[c0][d1] + delta_f[a1][b1]*delta_f[b0][d1]*delta_f[c0][d0] + delta_f[a1][c0]*delta_f[b0][b1]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[b0][b1]*delta_f[c0][d1] + delta_f[a1][d1]*delta_f[b0][b1]*delta_f[c0][d0]) + (-PA_1_f + PQ_a1_f)*(delta_f[a0][b0]*delta_f[b1][c0]*delta_f[d0][d1] + delta_f[a0][b0]*delta_f[b1][d0]*delta_f[c0][d1] + delta_f[a0][b0]*delta_f[b1][d1]*delta_f[c0][d0] + delta_f[a0][b1]*delta_f[b0][c0]*delta_f[d0][d1] + delta_f[a0][b1]*delta_f[b0][d0]*delta_f[c0][d1] + delta_f[a0][b1]*delta_f[b0][d1]*delta_f[c0][d0] + delta_f[a0][c0]*delta_f[b0][b1]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[b0][b1]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[b0][b1]*delta_f[c0][d0]) + (-PB_0_f + PQ_b0_f)*(delta_f[a0][a1]*delta_f[b1][c0]*delta_f[d0][d1] + delta_f[a0][a1]*delta_f[b1][d0]*delta_f[c0][d1] + delta_f[a0][a1]*delta_f[b1][d1]*delta_f[c0][d0] + delta_f[a0][b1]*delta_f[a1][c0]*delta_f[d0][d1] + delta_f[a0][b1]*delta_f[a1][d0]*delta_f[c0][d1] + delta_f[a0][b1]*delta_f[a1][d1]*delta_f[c0][d0] + delta_f[a0][c0]*delta_f[a1][b1]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[a1][b1]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[a1][b1]*delta_f[c0][d0]) + (-PB_1_f + PQ_b1_f)*(delta_f[a0][a1]*delta_f[b0][c0]*delta_f[d0][d1] + delta_f[a0][a1]*delta_f[b0][d0]*delta_f[c0][d1] + delta_f[a0][a1]*delta_f[b0][d1]*delta_f[c0][d0] + delta_f[a0][b0]*delta_f[a1][c0]*delta_f[d0][d1] + delta_f[a0][b0]*delta_f[a1][d0]*delta_f[c0][d1] + delta_f[a0][b0]*delta_f[a1][d1]*delta_f[c0][d0] + delta_f[a0][c0]*delta_f[a1][b0]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[a1][b0]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[a1][b0]*delta_f[c0][d0])

                        +2.0f*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])*(delta_f[c0][d0]*(PQ_d1_f + QD_1_f) + delta_f[c0][d1]*(PQ_d0_f + QD_0_f) + delta_f[d0][d1]*(PQ_c0_f + QC_0_f))

                        +QC_0_f*(delta_f[b0][b1]*(delta_f[a0][d0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][d0]) + delta_f[b0][d0]*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + delta_f[b0][d1]*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + delta_f[b1][d0]*(delta_f[a0][b0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b0]) + delta_f[b1][d1]*(delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0])) + QD_0_f*(delta_f[b0][b1]*(delta_f[a0][c0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][c0]) + delta_f[b0][c0]*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + delta_f[b0][d1]*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1]) + delta_f[b1][c0]*(delta_f[a0][b0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b0]) + delta_f[b1][d1]*(delta_f[a0][b0]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b0])) + QD_1_f*(delta_f[b0][b1]*(delta_f[a0][c0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][c0]) + delta_f[b0][c0]*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + delta_f[b0][d0]*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1]) + delta_f[b1][c0]*(delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0]) + delta_f[b1][d0]*(delta_f[a0][b0]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b0]))

                    )

                    + F7_t_f[2] * 0.125f * inv_S2_f * inv_S4_f * inv_S4_f * (

                        -PA_0_f*(delta_f[a1][b0]*(delta_f[b1][c0]*delta_f[d0][d1] + delta_f[b1][d0]*delta_f[c0][d1] + delta_f[b1][d1]*delta_f[c0][d0]) + delta_f[a1][b1]*(delta_f[b0][c0]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][d0]) + delta_f[b0][b1]*(delta_f[a1][c0]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[c0][d1] + delta_f[a1][d1]*delta_f[c0][d0])) - PA_1_f*(delta_f[a0][b0]*(delta_f[b1][c0]*delta_f[d0][d1] + delta_f[b1][d0]*delta_f[c0][d1] + delta_f[b1][d1]*delta_f[c0][d0]) + delta_f[a0][b1]*(delta_f[b0][c0]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][d0]) + delta_f[b0][b1]*(delta_f[a0][c0]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[c0][d0])) - PB_0_f*(delta_f[a0][a1]*(delta_f[b1][c0]*delta_f[d0][d1] + delta_f[b1][d0]*delta_f[c0][d1] + delta_f[b1][d1]*delta_f[c0][d0]) + delta_f[a0][b1]*(delta_f[a1][c0]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[c0][d1] + delta_f[a1][d1]*delta_f[c0][d0]) + delta_f[a1][b1]*(delta_f[a0][c0]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[c0][d0])) - PB_1_f*(delta_f[a0][a1]*(delta_f[b0][c0]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][d0]) + delta_f[a0][b0]*(delta_f[a1][c0]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[c0][d1] + delta_f[a1][d1]*delta_f[c0][d0]) + delta_f[a1][b0]*(delta_f[a0][c0]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[c0][d0]))

                        +(PQ_c0_f*delta_f[d0][d1] + PQ_d0_f*delta_f[c0][d1] + PQ_d1_f*delta_f[c0][d0])*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])

                    )

                    + F7_t_f[2] * 0.25f * S1_f * inv_S2_f * inv_S4_f * inv_S4_f * (

                        -PA_0_f*PA_1_f*(PB_0_f*(delta_f[b1][c0]*delta_f[d0][d1] + delta_f[b1][d0]*delta_f[c0][d1] + delta_f[b1][d1]*delta_f[c0][d0]) + PB_1_f*(delta_f[b0][c0]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][d0])) - PB_0_f*PB_1_f*(PA_0_f*(delta_f[a1][c0]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[c0][d1] + delta_f[a1][d1]*delta_f[c0][d0]) + PA_1_f*(delta_f[a0][c0]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[c0][d0]))

                        +(PA_0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) + PA_1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PB_0_f*PB_1_f*delta_f[a0][a1])*(PQ_c0_f*delta_f[d0][d1] + PQ_d0_f*delta_f[c0][d1] + PQ_d1_f*delta_f[c0][d0])

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        const float D_f = pd_mat_D_f[kl];
        const float contrib_f = eri_ijkl_f * D_f * 2.0f;
        ERIs[threadIdx.y] += (double)contrib_f;
    }

    __syncthreads();

    if ((threadIdx.y == 0) && (ij < dd_prim_pair_count_local))
    {
        double J_ij = 0.0f;

        for (uint32_t n = 0; n < TILE_DIM_LARGE; n++)
        {
            J_ij += ERIs[n];
        }

        mat_J[ij] += J_ij;
    }
}

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDPD2_FP32_auto_s_ast(double*         mat_J,
                       const float*    p_prim_info_f,
                       const uint32_t  p_prim_count,
                       const float*    d_prim_info_f,
                       const uint32_t  d_prim_count,
                       const float*    pd_mat_D_f,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const float*    dd_pair_data_local_f,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* pd_first_inds,
                       const uint32_t* pd_second_inds,
                       const float*    pd_pair_data_f,
                       const uint32_t  pd_prim_pair_count,
                       const float*    boys_func_table_f,
                       const float*    boys_func_ft_f,
                       const uint32_t* prec_cut_ij_tile,
                       const uint32_t* screen_cut_ij_tile)
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ float   delta_f[3][3];

    __shared__ float a_i_f, a_j_f, r_i_f[3], r_j_f[3], S_ij_00_f, S1_f, inv_S1_f;
    //__shared__ float PA_0_f, PA_1_f, PB_0_f, PB_1_f;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];
    const uint32_t screen_cut = screen_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta_f[0][0] = 1.0f; delta_f[0][1] = 0.0f; delta_f[0][2] = 0.0f;
        delta_f[1][0] = 0.0f; delta_f[1][1] = 1.0f; delta_f[1][2] = 0.0f;
        delta_f[2][0] = 0.0f; delta_f[2][1] = 0.0f; delta_f[2][2] = 1.0f;

        if (ij < dd_prim_pair_count_local)
        {
            i = dd_first_inds_local[ij];
            j = dd_second_inds_local[ij];

            a_i_f = d_prim_info_f[i / 6 + d_prim_count * 0];

            r_i_f[0] = d_prim_info_f[i / 6 + d_prim_count * 2];
            r_i_f[1] = d_prim_info_f[i / 6 + d_prim_count * 3];
            r_i_f[2] = d_prim_info_f[i / 6 + d_prim_count * 4];

            a_j_f = d_prim_info_f[j / 6 + d_prim_count * 0];

            r_j_f[0] = d_prim_info_f[j / 6 + d_prim_count * 2];
            r_j_f[1] = d_prim_info_f[j / 6 + d_prim_count * 3];
            r_j_f[2] = d_prim_info_f[j / 6 + d_prim_count * 4];

            S1_f = a_i_f + a_j_f;
            inv_S1_f = (float) (1.0 / (double)S1_f);

            S_ij_00_f = dd_pair_data_local_f[ij];

            a0 = d_cart_inds[i % 6][0];
            a1 = d_cart_inds[i % 6][1];
            b0 = d_cart_inds[j % 6][0];
            b1 = d_cart_inds[j % 6][1];

            //PA_0_f = (a_j_f  * inv_S1_f) * (r_j_f[a0] - r_i_f[a0]);
            //PA_1_f = (a_j_f  * inv_S1_f) * (r_j_f[a1] - r_i_f[a1]);
            //PB_0_f = (-a_i_f * inv_S1_f) * (r_j_f[b0] - r_i_f[b0]);
            //PB_1_f = (-a_i_f * inv_S1_f) * (r_j_f[b1] - r_i_f[b1]);

        }

    }

    ERIs[threadIdx.y] = 0.0;

    __syncthreads();

    for (uint32_t m = prec_cut; m < screen_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= pd_prim_pair_count))
        {
            break;
        }

        const auto k = pd_first_inds[kl];
        const auto l = pd_second_inds[kl];

        const auto a_k_f = p_prim_info_f[k / 3 + p_prim_count * 0];

        const float r_k0_f = p_prim_info_f[k / 3 + p_prim_count * 2];
        const float r_k1_f = p_prim_info_f[k / 3 + p_prim_count * 3];
        const float r_k2_f = p_prim_info_f[k / 3 + p_prim_count * 4];

        const auto a_l_f = d_prim_info_f[l / 6 + d_prim_count * 0];

        const float r_l0_f = d_prim_info_f[l / 6 + d_prim_count * 2];
        const float r_l1_f = d_prim_info_f[l / 6 + d_prim_count * 3];
        const float r_l2_f = d_prim_info_f[l / 6 + d_prim_count * 4];

        const auto S_kl_00_f = pd_pair_data_f[kl];

        const auto c0 = k % 3;
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        const float r_l_c0_f = (c0 == 0 ? r_l0_f : (c0 == 1 ? r_l1_f : r_l2_f));
        const float r_l_d0_f = (d0 == 0 ? r_l0_f : (d0 == 1 ? r_l1_f : r_l2_f));
        const float r_l_d1_f = (d1 == 0 ? r_l0_f : (d1 == 1 ? r_l1_f : r_l2_f));

        const float r_k_c0_f = (c0 == 0 ? r_k0_f : (c0 == 1 ? r_k1_f : r_k2_f));
        const float r_k_d0_f = (d0 == 0 ? r_k0_f : (d0 == 1 ? r_k1_f : r_k2_f));
        const float r_k_d1_f = (d1 == 0 ? r_k0_f : (d1 == 1 ? r_k1_f : r_k2_f));

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2_f = a_k_f + a_l_f;

        const auto inv_S2_f = 1.0f / S2_f;
        const auto inv_S4_f = 1.0f / (S1_f + S2_f);

        const float PQ0_f = (a_k_f * r_k0_f + a_l_f * r_l0_f) * inv_S2_f - (a_i_f * r_i_f[0] + a_j_f * r_j_f[0]) * inv_S1_f;
        const float PQ1_f = (a_k_f * r_k1_f + a_l_f * r_l1_f) * inv_S2_f - (a_i_f * r_i_f[1] + a_j_f * r_j_f[1]) * inv_S1_f;
        const float PQ2_f = (a_k_f * r_k2_f + a_l_f * r_l2_f) * inv_S2_f - (a_i_f * r_i_f[2] + a_j_f * r_j_f[2]) * inv_S1_f;

        const auto r2_PQ_f = PQ0_f * PQ0_f + PQ1_f * PQ1_f + PQ2_f * PQ2_f;

        const auto Lambda_f = sqrtf(4.0f * S1_f * S2_f * MATH_CONST_INV_PI_F * inv_S4_f);

        float F7_t_f[3];

        gpu::computeBoysFunction_f(F7_t_f, S1_f * S2_f * inv_S4_f * r2_PQ_f, 2, boys_func_table_f, boys_func_ft_f);

        const auto QC_0_f = (a_l_f * inv_S2_f) * (r_l_c0_f - r_k_c0_f);
        const auto QD_0_f = (-a_k_f * inv_S2_f) * (r_l_d0_f - r_k_d0_f);
        const auto QD_1_f = (-a_k_f * inv_S2_f) * (r_l_d1_f - r_k_d1_f);

        const float eri_ijkl_f = Lambda_f * S_ij_00_f * S_kl_00_f * (

                    + F7_t_f[2] * 0.25f * S2_f * S2_f * inv_S1_f * inv_S1_f * inv_S4_f * inv_S4_f * (

                        +QC_0_f*QD_0_f*QD_1_f*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        const float D_f = pd_mat_D_f[kl];
        const float contrib_f = eri_ijkl_f * D_f * 2.0f;
        ERIs[threadIdx.y] += (double)contrib_f;
    }

    __syncthreads();

    if ((threadIdx.y == 0) && (ij < dd_prim_pair_count_local))
    {
        double J_ij = 0.0f;

        for (uint32_t n = 0; n < TILE_DIM_LARGE; n++)
        {
            J_ij += ERIs[n];
        }

        mat_J[ij] += J_ij;
    }
}

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDPD3_FP32_auto_s_ast(double*         mat_J,
                       const float*    p_prim_info_f,
                       const uint32_t  p_prim_count,
                       const float*    d_prim_info_f,
                       const uint32_t  d_prim_count,
                       const float*    pd_mat_D_f,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const float*    dd_pair_data_local_f,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* pd_first_inds,
                       const uint32_t* pd_second_inds,
                       const float*    pd_pair_data_f,
                       const uint32_t  pd_prim_pair_count,
                       const float*    boys_func_table_f,
                       const float*    boys_func_ft_f,
                       const uint32_t* prec_cut_ij_tile,
                       const uint32_t* screen_cut_ij_tile)
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ float   delta_f[3][3];

    __shared__ float a_i_f, a_j_f, r_i_f[3], r_j_f[3], S_ij_00_f, S1_f, inv_S1_f;
    __shared__ float PA_0_f, PA_1_f, PB_0_f, PB_1_f;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];
    const uint32_t screen_cut = screen_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta_f[0][0] = 1.0f; delta_f[0][1] = 0.0f; delta_f[0][2] = 0.0f;
        delta_f[1][0] = 0.0f; delta_f[1][1] = 1.0f; delta_f[1][2] = 0.0f;
        delta_f[2][0] = 0.0f; delta_f[2][1] = 0.0f; delta_f[2][2] = 1.0f;

        if (ij < dd_prim_pair_count_local)
        {
            i = dd_first_inds_local[ij];
            j = dd_second_inds_local[ij];

            a_i_f = d_prim_info_f[i / 6 + d_prim_count * 0];

            r_i_f[0] = d_prim_info_f[i / 6 + d_prim_count * 2];
            r_i_f[1] = d_prim_info_f[i / 6 + d_prim_count * 3];
            r_i_f[2] = d_prim_info_f[i / 6 + d_prim_count * 4];

            a_j_f = d_prim_info_f[j / 6 + d_prim_count * 0];

            r_j_f[0] = d_prim_info_f[j / 6 + d_prim_count * 2];
            r_j_f[1] = d_prim_info_f[j / 6 + d_prim_count * 3];
            r_j_f[2] = d_prim_info_f[j / 6 + d_prim_count * 4];

            S1_f = a_i_f + a_j_f;
            inv_S1_f = (float) (1.0 / (double)S1_f);

            S_ij_00_f = dd_pair_data_local_f[ij];

            a0 = d_cart_inds[i % 6][0];
            a1 = d_cart_inds[i % 6][1];
            b0 = d_cart_inds[j % 6][0];
            b1 = d_cart_inds[j % 6][1];

            PA_0_f = (a_j_f  * inv_S1_f) * (r_j_f[a0] - r_i_f[a0]);
            PA_1_f = (a_j_f  * inv_S1_f) * (r_j_f[a1] - r_i_f[a1]);
            PB_0_f = (-a_i_f * inv_S1_f) * (r_j_f[b0] - r_i_f[b0]);
            PB_1_f = (-a_i_f * inv_S1_f) * (r_j_f[b1] - r_i_f[b1]);

        }

    }

    ERIs[threadIdx.y] = 0.0;

    __syncthreads();

    for (uint32_t m = prec_cut; m < screen_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= pd_prim_pair_count))
        {
            break;
        }

        const auto k = pd_first_inds[kl];
        const auto l = pd_second_inds[kl];

        const auto a_k_f = p_prim_info_f[k / 3 + p_prim_count * 0];

        const float r_k0_f = p_prim_info_f[k / 3 + p_prim_count * 2];
        const float r_k1_f = p_prim_info_f[k / 3 + p_prim_count * 3];
        const float r_k2_f = p_prim_info_f[k / 3 + p_prim_count * 4];

        const auto a_l_f = d_prim_info_f[l / 6 + d_prim_count * 0];

        const float r_l0_f = d_prim_info_f[l / 6 + d_prim_count * 2];
        const float r_l1_f = d_prim_info_f[l / 6 + d_prim_count * 3];
        const float r_l2_f = d_prim_info_f[l / 6 + d_prim_count * 4];

        const auto S_kl_00_f = pd_pair_data_f[kl];

        const auto c0 = k % 3;
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        const float r_l_c0_f = (c0 == 0 ? r_l0_f : (c0 == 1 ? r_l1_f : r_l2_f));
        const float r_l_d0_f = (d0 == 0 ? r_l0_f : (d0 == 1 ? r_l1_f : r_l2_f));
        const float r_l_d1_f = (d1 == 0 ? r_l0_f : (d1 == 1 ? r_l1_f : r_l2_f));

        const float r_k_c0_f = (c0 == 0 ? r_k0_f : (c0 == 1 ? r_k1_f : r_k2_f));
        const float r_k_d0_f = (d0 == 0 ? r_k0_f : (d0 == 1 ? r_k1_f : r_k2_f));
        const float r_k_d1_f = (d1 == 0 ? r_k0_f : (d1 == 1 ? r_k1_f : r_k2_f));

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2_f = a_k_f + a_l_f;

        const auto inv_S2_f = 1.0f / S2_f;
        const auto inv_S4_f = 1.0f / (S1_f + S2_f);

        const float PQ0_f = (a_k_f * r_k0_f + a_l_f * r_l0_f) * inv_S2_f - (a_i_f * r_i_f[0] + a_j_f * r_j_f[0]) * inv_S1_f;
        const float PQ1_f = (a_k_f * r_k1_f + a_l_f * r_l1_f) * inv_S2_f - (a_i_f * r_i_f[1] + a_j_f * r_j_f[1]) * inv_S1_f;
        const float PQ2_f = (a_k_f * r_k2_f + a_l_f * r_l2_f) * inv_S2_f - (a_i_f * r_i_f[2] + a_j_f * r_j_f[2]) * inv_S1_f;

        const float PQ_a0_f = (a0 == 0 ? PQ0_f : (a0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_a1_f = (a1 == 0 ? PQ0_f : (a1 == 1 ? PQ1_f : PQ2_f));
        const float PQ_b0_f = (b0 == 0 ? PQ0_f : (b0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_b1_f = (b1 == 0 ? PQ0_f : (b1 == 1 ? PQ1_f : PQ2_f));
        const float PQ_c0_f = (c0 == 0 ? PQ0_f : (c0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_d0_f = (d0 == 0 ? PQ0_f : (d0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_d1_f = (d1 == 0 ? PQ0_f : (d1 == 1 ? PQ1_f : PQ2_f));

        const auto r2_PQ_f = PQ0_f * PQ0_f + PQ1_f * PQ1_f + PQ2_f * PQ2_f;

        const auto Lambda_f = sqrtf(4.0f * S1_f * S2_f * MATH_CONST_INV_PI_F * inv_S4_f);

        float F7_t_f[3];

        gpu::computeBoysFunction_f(F7_t_f, S1_f * S2_f * inv_S4_f * r2_PQ_f, 2, boys_func_table_f, boys_func_ft_f);

        const auto QC_0_f = (a_l_f * inv_S2_f) * (r_l_c0_f - r_k_c0_f);
        const auto QD_0_f = (-a_k_f * inv_S2_f) * (r_l_d0_f - r_k_d0_f);
        const auto QD_1_f = (-a_k_f * inv_S2_f) * (r_l_d1_f - r_k_d1_f);

        const float eri_ijkl_f = Lambda_f * S_ij_00_f * S_kl_00_f * (

                    + F7_t_f[2] * 0.25f * S2_f * inv_S1_f * inv_S4_f * inv_S4_f * (

                        +(QC_0_f*delta_f[d0][d1] + QD_0_f*delta_f[c0][d1] + QD_1_f*delta_f[c0][d0])*(-PA_0_f*(PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) - PA_1_f*(PQ_a0_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][b0]) - PB_0_f*(PQ_a0_f*delta_f[a1][b1] + PQ_a1_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][a1]) - PB_1_f*(PQ_a0_f*delta_f[a1][b0] + PQ_a1_f*delta_f[a0][b0] + PQ_b0_f*delta_f[a0][a1]) + PQ_a0_f*PQ_a1_f*delta_f[b0][b1] + PQ_a0_f*PQ_b0_f*delta_f[a1][b1] + PQ_a0_f*PQ_b1_f*delta_f[a1][b0] + PQ_a1_f*PQ_b0_f*delta_f[a0][b1] + PQ_a1_f*PQ_b1_f*delta_f[a0][b0] + PQ_b0_f*PQ_b1_f*delta_f[a0][a1])

                        +(-PA_0_f*QC_0_f*QD_0_f + PQ_a0_f*QC_0_f*QD_0_f)*(delta_f[a1][b0]*delta_f[b1][d1] + delta_f[a1][b1]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][b1]) + (-PA_0_f*QC_0_f*QD_1_f + PQ_a0_f*QC_0_f*QD_1_f)*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + (-PA_0_f*QD_0_f*QD_1_f + PQ_a0_f*QD_0_f*QD_1_f)*(delta_f[a1][b0]*delta_f[b1][c0] + delta_f[a1][b1]*delta_f[b0][c0] + delta_f[a1][c0]*delta_f[b0][b1]) + (-PA_1_f*QC_0_f*QD_0_f + PQ_a1_f*QC_0_f*QD_0_f)*(delta_f[a0][b0]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][b1]) + (-PA_1_f*QC_0_f*QD_1_f + PQ_a1_f*QC_0_f*QD_1_f)*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + (-PA_1_f*QD_0_f*QD_1_f + PQ_a1_f*QD_0_f*QD_1_f)*(delta_f[a0][b0]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[b0][c0] + delta_f[a0][c0]*delta_f[b0][b1]) + (-PB_0_f*QC_0_f*QD_0_f + PQ_b0_f*QC_0_f*QD_0_f)*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + (-PB_0_f*QC_0_f*QD_1_f + PQ_b0_f*QC_0_f*QD_1_f)*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + (-PB_0_f*QD_0_f*QD_1_f + PQ_b0_f*QD_0_f*QD_1_f)*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1]) + (-PB_1_f*QC_0_f*QD_0_f + PQ_b1_f*QC_0_f*QD_0_f)*(delta_f[a0][a1]*delta_f[b0][d1] + delta_f[a0][b0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b0]) + (-PB_1_f*QC_0_f*QD_1_f + PQ_b1_f*QC_0_f*QD_1_f)*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0]) + (-PB_1_f*QD_0_f*QD_1_f + PQ_b1_f*QD_0_f*QD_1_f)*(delta_f[a0][a1]*delta_f[b0][c0] + delta_f[a0][b0]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b0])

                        +2.0f*(PQ_d0_f*QC_0_f*QD_1_f + QD_0_f*(PQ_c0_f*QD_1_f + PQ_d1_f*QC_0_f))*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])

                    )

                    + F7_t_f[2] * 0.25f * inv_S4_f * inv_S4_f * (

                        +(delta_f[c0][d0]*(PQ_d1_f + QD_1_f) + delta_f[c0][d1]*(PQ_d0_f + QD_0_f) + delta_f[d0][d1]*(PQ_c0_f + QC_0_f))*(PA_0_f*PA_1_f*delta_f[b0][b1] + PA_0_f*PB_0_f*delta_f[a1][b1] + PA_0_f*PB_1_f*delta_f[a1][b0] - PA_0_f*(PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) + PA_1_f*PB_0_f*delta_f[a0][b1] + PA_1_f*PB_1_f*delta_f[a0][b0] - PA_1_f*(PQ_a0_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][b0]) + PB_0_f*PB_1_f*delta_f[a0][a1] - PB_0_f*(PQ_a0_f*delta_f[a1][b1] + PQ_a1_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][a1]) - PB_1_f*(PQ_a0_f*delta_f[a1][b0] + PQ_a1_f*delta_f[a0][b0] + PQ_b0_f*delta_f[a0][a1]))

                        +PA_0_f*(PA_1_f*(QC_0_f*(delta_f[b0][d0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][d0]) + QD_0_f*(delta_f[b0][c0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][c0]) + QD_1_f*(delta_f[b0][c0]*delta_f[b1][d0] + delta_f[b0][d0]*delta_f[b1][c0])) + PB_0_f*(QC_0_f*(delta_f[a1][d0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b1][d0]) + QD_0_f*(delta_f[a1][c0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b1][c0]) + QD_1_f*(delta_f[a1][c0]*delta_f[b1][d0] + delta_f[a1][d0]*delta_f[b1][c0])) + PB_1_f*(QC_0_f*(delta_f[a1][d0]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][d0]) + QD_0_f*(delta_f[a1][c0]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][c0]) + QD_1_f*(delta_f[a1][c0]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][c0]))) + PA_1_f*(PB_0_f*(QC_0_f*(delta_f[a0][d0]*delta_f[b1][d1] + delta_f[a0][d1]*delta_f[b1][d0]) + QD_0_f*(delta_f[a0][c0]*delta_f[b1][d1] + delta_f[a0][d1]*delta_f[b1][c0]) + QD_1_f*(delta_f[a0][c0]*delta_f[b1][d0] + delta_f[a0][d0]*delta_f[b1][c0])) + PB_1_f*(QC_0_f*(delta_f[a0][d0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][d0]) + QD_0_f*(delta_f[a0][c0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][c0]) + QD_1_f*(delta_f[a0][c0]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][c0]))) + PB_0_f*PB_1_f*(QC_0_f*(delta_f[a0][d0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][d0]) + QD_0_f*(delta_f[a0][c0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][c0]) + QD_1_f*(delta_f[a0][c0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][c0]))

                        +(delta_f[a0][c0]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[c0][d0])*(PA_1_f*PB_0_f*PQ_b1_f + PA_1_f*PB_1_f*PQ_b0_f + PB_0_f*PB_1_f*PQ_a1_f) + (delta_f[a1][c0]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[c0][d1] + delta_f[a1][d1]*delta_f[c0][d0])*(PA_0_f*PB_0_f*PQ_b1_f + PA_0_f*PB_1_f*PQ_b0_f + PB_0_f*PB_1_f*PQ_a0_f) + (delta_f[b0][c0]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][d0])*(PA_0_f*PA_1_f*PQ_b1_f + PA_0_f*PB_1_f*PQ_a1_f + PA_1_f*PB_1_f*PQ_a0_f) + (delta_f[b1][c0]*delta_f[d0][d1] + delta_f[b1][d0]*delta_f[c0][d1] + delta_f[b1][d1]*delta_f[c0][d0])*(PA_0_f*PA_1_f*PQ_b0_f + PA_0_f*PB_0_f*PQ_a1_f + PA_1_f*PB_0_f*PQ_a0_f)

                        +(-PA_0_f*PQ_c0_f*QD_0_f - PA_0_f*PQ_d0_f*QC_0_f)*(delta_f[a1][b0]*delta_f[b1][d1] + delta_f[a1][b1]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][b1]) + (-PA_0_f*PQ_c0_f*QD_1_f - PA_0_f*PQ_d1_f*QC_0_f)*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + (-PA_0_f*PQ_d0_f*QD_1_f - PA_0_f*PQ_d1_f*QD_0_f)*(delta_f[a1][b0]*delta_f[b1][c0] + delta_f[a1][b1]*delta_f[b0][c0] + delta_f[a1][c0]*delta_f[b0][b1]) + (-PA_1_f*PQ_c0_f*QD_0_f - PA_1_f*PQ_d0_f*QC_0_f)*(delta_f[a0][b0]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][b1]) + (-PA_1_f*PQ_c0_f*QD_1_f - PA_1_f*PQ_d1_f*QC_0_f)*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + (-PA_1_f*PQ_d0_f*QD_1_f - PA_1_f*PQ_d1_f*QD_0_f)*(delta_f[a0][b0]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[b0][c0] + delta_f[a0][c0]*delta_f[b0][b1]) + (-PB_0_f*PQ_c0_f*QD_0_f - PB_0_f*PQ_d0_f*QC_0_f)*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + (-PB_0_f*PQ_c0_f*QD_1_f - PB_0_f*PQ_d1_f*QC_0_f)*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + (-PB_0_f*PQ_d0_f*QD_1_f - PB_0_f*PQ_d1_f*QD_0_f)*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1]) + (-PB_1_f*PQ_c0_f*QD_0_f - PB_1_f*PQ_d0_f*QC_0_f)*(delta_f[a0][a1]*delta_f[b0][d1] + delta_f[a0][b0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b0]) + (-PB_1_f*PQ_c0_f*QD_1_f - PB_1_f*PQ_d1_f*QC_0_f)*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0]) + (-PB_1_f*PQ_d0_f*QD_1_f - PB_1_f*PQ_d1_f*QD_0_f)*(delta_f[a0][a1]*delta_f[b0][c0] + delta_f[a0][b0]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b0])

                        +(PQ_c0_f*(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f) + PQ_d0_f*PQ_d1_f*QC_0_f)*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])

                    )

                    + F7_t_f[2] * 0.5f * S1_f * S1_f * inv_S2_f * inv_S4_f * inv_S4_f * (

                        +PA_0_f*PA_1_f*PB_0_f*PB_1_f*(PQ_c0_f*delta_f[d0][d1] + PQ_d0_f*delta_f[c0][d1] + PQ_d1_f*delta_f[c0][d0])

                    )

                    + F7_t_f[2] * 0.5f * S1_f * inv_S4_f * inv_S4_f * (

                        -(PA_0_f*PA_1_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PB_1_f*(PA_0_f*PQ_a1_f + PA_1_f*PQ_a0_f))*(delta_f[c0][d0]*(PQ_d1_f + QD_1_f) + delta_f[c0][d1]*(PQ_d0_f + QD_0_f) + delta_f[d0][d1]*(PQ_c0_f + QC_0_f))

                        -PA_0_f*PA_1_f*(PB_0_f*(PQ_c0_f*(QD_0_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][d0]) + PQ_d0_f*(QC_0_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][c0]) + PQ_d1_f*(QC_0_f*delta_f[b1][d0] + QD_0_f*delta_f[b1][c0])) + PB_1_f*(PQ_c0_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0]) + PQ_d0_f*(QC_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][c0]) + PQ_d1_f*(QC_0_f*delta_f[b0][d0] + QD_0_f*delta_f[b0][c0]))) - PB_0_f*PB_1_f*(PA_0_f*(PQ_c0_f*(QD_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][d0]) + PQ_d0_f*(QC_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][c0]) + PQ_d1_f*(QC_0_f*delta_f[a1][d0] + QD_0_f*delta_f[a1][c0])) + PA_1_f*(PQ_c0_f*(QD_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][d0]) + PQ_d0_f*(QC_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][c0]) + PQ_d1_f*(QC_0_f*delta_f[a0][d0] + QD_0_f*delta_f[a0][c0])))

                        +(PQ_c0_f*(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f) + PQ_d0_f*PQ_d1_f*QC_0_f)*(PA_0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) + PA_1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PB_0_f*PB_1_f*delta_f[a0][a1])

                    )

                    + F7_t_f[2] * 0.5f * S2_f * S2_f * inv_S1_f * inv_S4_f * inv_S4_f * (

                        +QC_0_f*QD_0_f*QD_1_f*(-PA_0_f*(PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) - PQ_a0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) + PQ_a0_f*(PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) - PQ_a1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PQ_a1_f*(PQ_b0_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][b0]) + PQ_b0_f*PQ_b1_f*delta_f[a0][a1] - PQ_b0_f*(PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) - PQ_b1_f*(PA_1_f*delta_f[a0][b0] + PB_0_f*delta_f[a0][a1]))

                    )

                    + F7_t_f[2] * 0.5f * S2_f * inv_S4_f * inv_S4_f * (

                        +(QC_0_f*delta_f[d0][d1] + QD_0_f*delta_f[c0][d1] + QD_1_f*delta_f[c0][d0])*(PA_0_f*PQ_b0_f*(PA_1_f*PQ_b1_f + PB_1_f*PQ_a1_f) + PA_1_f*PQ_a0_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PQ_a1_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))

                        +QC_0_f*(PA_0_f*(PA_1_f*(PQ_b0_f*(QD_0_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][d0]) + PQ_b1_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0])) + PB_0_f*(PQ_a1_f*(QD_0_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][d0]) + PQ_b1_f*(QD_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][d0])) + PB_1_f*(PQ_a1_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0]) + PQ_b0_f*(QD_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][d0]))) + PA_1_f*(PB_0_f*(PQ_a0_f*(QD_0_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][d0]) + PQ_b1_f*(QD_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][d0])) + PB_1_f*(PQ_a0_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0]) + PQ_b0_f*(QD_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][d0]))) + PB_0_f*PB_1_f*(PQ_a0_f*(QD_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][d0]) + PQ_a1_f*(QD_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][d0]))) + QD_0_f*QD_1_f*(PA_0_f*(PA_1_f*(PQ_b0_f*delta_f[b1][c0] + PQ_b1_f*delta_f[b0][c0]) + PB_0_f*(PQ_a1_f*delta_f[b1][c0] + PQ_b1_f*delta_f[a1][c0]) + PB_1_f*(PQ_a1_f*delta_f[b0][c0] + PQ_b0_f*delta_f[a1][c0])) + PA_1_f*(PB_0_f*(PQ_a0_f*delta_f[b1][c0] + PQ_b1_f*delta_f[a0][c0]) + PB_1_f*(PQ_a0_f*delta_f[b0][c0] + PQ_b0_f*delta_f[a0][c0])) + PB_0_f*PB_1_f*(PQ_a0_f*delta_f[a1][c0] + PQ_a1_f*delta_f[a0][c0]))

                        +(PQ_d0_f*QC_0_f*QD_1_f + QD_0_f*(PQ_c0_f*QD_1_f + PQ_d1_f*QC_0_f))*(PA_0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) - PA_0_f*(PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) + PA_1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) - PA_1_f*(PQ_a0_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][b0]) + PB_0_f*PB_1_f*delta_f[a0][a1] - PB_0_f*(PQ_a0_f*delta_f[a1][b1] + PQ_a1_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][a1]) - PB_1_f*(PQ_a0_f*delta_f[a1][b0] + PQ_a1_f*delta_f[a0][b0] + PQ_b0_f*delta_f[a0][a1]))

                    )

                    + F7_t_f[2] * S1_f * S1_f * inv_S4_f * inv_S4_f * (

                        +PA_0_f*PA_1_f*PB_0_f*PB_1_f*(PQ_c0_f*(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f) + PQ_d0_f*PQ_d1_f*QC_0_f)

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        const float D_f = pd_mat_D_f[kl];
        const float contrib_f = eri_ijkl_f * D_f * 2.0f;
        ERIs[threadIdx.y] += (double)contrib_f;
    }

    __syncthreads();

    if ((threadIdx.y == 0) && (ij < dd_prim_pair_count_local))
    {
        double J_ij = 0.0f;

        for (uint32_t n = 0; n < TILE_DIM_LARGE; n++)
        {
            J_ij += ERIs[n];
        }

        mat_J[ij] += J_ij;
    }
}

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDPD4_FP32_auto_s_ast(double*         mat_J,
                       const float*    p_prim_info_f,
                       const uint32_t  p_prim_count,
                       const float*    d_prim_info_f,
                       const uint32_t  d_prim_count,
                       const float*    pd_mat_D_f,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const float*    dd_pair_data_local_f,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* pd_first_inds,
                       const uint32_t* pd_second_inds,
                       const float*    pd_pair_data_f,
                       const uint32_t  pd_prim_pair_count,
                       const float*    boys_func_table_f,
                       const float*    boys_func_ft_f,
                       const uint32_t* prec_cut_ij_tile,
                       const uint32_t* screen_cut_ij_tile)
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ float   delta_f[3][3];

    __shared__ float a_i_f, a_j_f, r_i_f[3], r_j_f[3], S_ij_00_f, S1_f, inv_S1_f;
    __shared__ float PA_0_f, PA_1_f, PB_0_f, PB_1_f;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];
    const uint32_t screen_cut = screen_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta_f[0][0] = 1.0f; delta_f[0][1] = 0.0f; delta_f[0][2] = 0.0f;
        delta_f[1][0] = 0.0f; delta_f[1][1] = 1.0f; delta_f[1][2] = 0.0f;
        delta_f[2][0] = 0.0f; delta_f[2][1] = 0.0f; delta_f[2][2] = 1.0f;

        if (ij < dd_prim_pair_count_local)
        {
            i = dd_first_inds_local[ij];
            j = dd_second_inds_local[ij];

            a_i_f = d_prim_info_f[i / 6 + d_prim_count * 0];

            r_i_f[0] = d_prim_info_f[i / 6 + d_prim_count * 2];
            r_i_f[1] = d_prim_info_f[i / 6 + d_prim_count * 3];
            r_i_f[2] = d_prim_info_f[i / 6 + d_prim_count * 4];

            a_j_f = d_prim_info_f[j / 6 + d_prim_count * 0];

            r_j_f[0] = d_prim_info_f[j / 6 + d_prim_count * 2];
            r_j_f[1] = d_prim_info_f[j / 6 + d_prim_count * 3];
            r_j_f[2] = d_prim_info_f[j / 6 + d_prim_count * 4];

            S1_f = a_i_f + a_j_f;
            inv_S1_f = (float) (1.0 / (double)S1_f);

            S_ij_00_f = dd_pair_data_local_f[ij];

            a0 = d_cart_inds[i % 6][0];
            a1 = d_cart_inds[i % 6][1];
            b0 = d_cart_inds[j % 6][0];
            b1 = d_cart_inds[j % 6][1];

            PA_0_f = (a_j_f  * inv_S1_f) * (r_j_f[a0] - r_i_f[a0]);
            PA_1_f = (a_j_f  * inv_S1_f) * (r_j_f[a1] - r_i_f[a1]);
            PB_0_f = (-a_i_f * inv_S1_f) * (r_j_f[b0] - r_i_f[b0]);
            PB_1_f = (-a_i_f * inv_S1_f) * (r_j_f[b1] - r_i_f[b1]);

        }

    }

    ERIs[threadIdx.y] = 0.0;

    __syncthreads();

    for (uint32_t m = prec_cut; m < screen_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= pd_prim_pair_count))
        {
            break;
        }

        const auto k = pd_first_inds[kl];
        const auto l = pd_second_inds[kl];

        const auto a_k_f = p_prim_info_f[k / 3 + p_prim_count * 0];

        const float r_k0_f = p_prim_info_f[k / 3 + p_prim_count * 2];
        const float r_k1_f = p_prim_info_f[k / 3 + p_prim_count * 3];
        const float r_k2_f = p_prim_info_f[k / 3 + p_prim_count * 4];

        const auto a_l_f = d_prim_info_f[l / 6 + d_prim_count * 0];

        const float r_l0_f = d_prim_info_f[l / 6 + d_prim_count * 2];
        const float r_l1_f = d_prim_info_f[l / 6 + d_prim_count * 3];
        const float r_l2_f = d_prim_info_f[l / 6 + d_prim_count * 4];

        const auto S_kl_00_f = pd_pair_data_f[kl];

        const auto c0 = k % 3;
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        const float r_l_c0_f = (c0 == 0 ? r_l0_f : (c0 == 1 ? r_l1_f : r_l2_f));
        const float r_l_d0_f = (d0 == 0 ? r_l0_f : (d0 == 1 ? r_l1_f : r_l2_f));
        const float r_l_d1_f = (d1 == 0 ? r_l0_f : (d1 == 1 ? r_l1_f : r_l2_f));

        const float r_k_c0_f = (c0 == 0 ? r_k0_f : (c0 == 1 ? r_k1_f : r_k2_f));
        const float r_k_d0_f = (d0 == 0 ? r_k0_f : (d0 == 1 ? r_k1_f : r_k2_f));
        const float r_k_d1_f = (d1 == 0 ? r_k0_f : (d1 == 1 ? r_k1_f : r_k2_f));

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2_f = a_k_f + a_l_f;

        const auto inv_S2_f = 1.0f / S2_f;
        const auto inv_S4_f = 1.0f / (S1_f + S2_f);

        const float PQ0_f = (a_k_f * r_k0_f + a_l_f * r_l0_f) * inv_S2_f - (a_i_f * r_i_f[0] + a_j_f * r_j_f[0]) * inv_S1_f;
        const float PQ1_f = (a_k_f * r_k1_f + a_l_f * r_l1_f) * inv_S2_f - (a_i_f * r_i_f[1] + a_j_f * r_j_f[1]) * inv_S1_f;
        const float PQ2_f = (a_k_f * r_k2_f + a_l_f * r_l2_f) * inv_S2_f - (a_i_f * r_i_f[2] + a_j_f * r_j_f[2]) * inv_S1_f;

        const float PQ_a0_f = (a0 == 0 ? PQ0_f : (a0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_a1_f = (a1 == 0 ? PQ0_f : (a1 == 1 ? PQ1_f : PQ2_f));
        const float PQ_b0_f = (b0 == 0 ? PQ0_f : (b0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_b1_f = (b1 == 0 ? PQ0_f : (b1 == 1 ? PQ1_f : PQ2_f));
        const float PQ_c0_f = (c0 == 0 ? PQ0_f : (c0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_d0_f = (d0 == 0 ? PQ0_f : (d0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_d1_f = (d1 == 0 ? PQ0_f : (d1 == 1 ? PQ1_f : PQ2_f));

        const auto r2_PQ_f = PQ0_f * PQ0_f + PQ1_f * PQ1_f + PQ2_f * PQ2_f;

        const auto Lambda_f = sqrtf(4.0f * S1_f * S2_f * MATH_CONST_INV_PI_F * inv_S4_f);

        float F7_t_f[4];

        gpu::computeBoysFunction_f(F7_t_f, S1_f * S2_f * inv_S4_f * r2_PQ_f, 3, boys_func_table_f, boys_func_ft_f);

        const auto QC_0_f = (a_l_f * inv_S2_f) * (r_l_c0_f - r_k_c0_f);
        const auto QD_0_f = (-a_k_f * inv_S2_f) * (r_l_d0_f - r_k_d0_f);
        const auto QD_1_f = (-a_k_f * inv_S2_f) * (r_l_d1_f - r_k_d1_f);

        const float eri_ijkl_f = Lambda_f * S_ij_00_f * S_kl_00_f * (

                    + F7_t_f[2] * S1_f * S2_f * inv_S4_f * inv_S4_f * (

                        -(PA_0_f*PA_1_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PB_1_f*(PA_0_f*PQ_a1_f + PA_1_f*PQ_a0_f))*(PQ_d0_f*QC_0_f*QD_1_f + QD_0_f*(PQ_c0_f*QD_1_f + PQ_d1_f*QC_0_f))

                    )

                    + F7_t_f[2] * S2_f * S2_f * inv_S4_f * inv_S4_f * (

                        +QC_0_f*QD_0_f*QD_1_f*(PA_0_f*PQ_b0_f*(PA_1_f*PQ_b1_f + PB_1_f*PQ_a1_f) + PA_1_f*PQ_a0_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PQ_a1_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))

                    )

                    + F7_t_f[3] * (-0.125f) * S2_f * inv_S1_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        +PQ_a0_f*(delta_f[a1][b0]*(delta_f[b1][c0]*delta_f[d0][d1] + delta_f[b1][d0]*delta_f[c0][d1] + delta_f[b1][d1]*delta_f[c0][d0]) + delta_f[a1][b1]*(delta_f[b0][c0]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][d0]) + delta_f[b0][b1]*(delta_f[a1][c0]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[c0][d1] + delta_f[a1][d1]*delta_f[c0][d0])) + PQ_a1_f*(delta_f[a0][b0]*(delta_f[b1][c0]*delta_f[d0][d1] + delta_f[b1][d0]*delta_f[c0][d1] + delta_f[b1][d1]*delta_f[c0][d0]) + delta_f[a0][b1]*(delta_f[b0][c0]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][d0]) + delta_f[b0][b1]*(delta_f[a0][c0]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[c0][d0])) + PQ_b0_f*(delta_f[a0][a1]*(delta_f[b1][c0]*delta_f[d0][d1] + delta_f[b1][d0]*delta_f[c0][d1] + delta_f[b1][d1]*delta_f[c0][d0]) + delta_f[a0][b1]*(delta_f[a1][c0]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[c0][d1] + delta_f[a1][d1]*delta_f[c0][d0]) + delta_f[a1][b1]*(delta_f[a0][c0]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[c0][d0])) + PQ_b1_f*(delta_f[a0][a1]*(delta_f[b0][c0]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][d0]) + delta_f[a0][b0]*(delta_f[a1][c0]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[c0][d1] + delta_f[a1][d1]*delta_f[c0][d0]) + delta_f[a1][b0]*(delta_f[a0][c0]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[c0][d0]))

                        +(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])*(delta_f[c0][d0]*(PQ_d1_f + QD_1_f) + delta_f[c0][d1]*(PQ_d0_f + QD_0_f) + delta_f[d0][d1]*(PQ_c0_f + QC_0_f))

                        +QC_0_f*(delta_f[b0][b1]*(delta_f[a0][d0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][d0]) + delta_f[b0][d0]*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + delta_f[b0][d1]*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + delta_f[b1][d0]*(delta_f[a0][b0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b0]) + delta_f[b1][d1]*(delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0])) + QD_0_f*(delta_f[b0][b1]*(delta_f[a0][c0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][c0]) + delta_f[b0][c0]*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + delta_f[b0][d1]*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1]) + delta_f[b1][c0]*(delta_f[a0][b0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b0]) + delta_f[b1][d1]*(delta_f[a0][b0]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b0])) + QD_1_f*(delta_f[b0][b1]*(delta_f[a0][c0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][c0]) + delta_f[b0][c0]*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + delta_f[b0][d0]*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1]) + delta_f[b1][c0]*(delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0]) + delta_f[b1][d0]*(delta_f[a0][b0]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b0]))

                    )

                    + F7_t_f[3] * 0.125f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        +(PA_0_f - PQ_a0_f)*(delta_f[a1][b0]*delta_f[b1][c0]*delta_f[d0][d1] + delta_f[a1][b0]*delta_f[b1][d0]*delta_f[c0][d1] + delta_f[a1][b0]*delta_f[b1][d1]*delta_f[c0][d0] + delta_f[a1][b1]*delta_f[b0][c0]*delta_f[d0][d1] + delta_f[a1][b1]*delta_f[b0][d0]*delta_f[c0][d1] + delta_f[a1][b1]*delta_f[b0][d1]*delta_f[c0][d0] + delta_f[a1][c0]*delta_f[b0][b1]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[b0][b1]*delta_f[c0][d1] + delta_f[a1][d1]*delta_f[b0][b1]*delta_f[c0][d0]) + (PA_1_f - PQ_a1_f)*(delta_f[a0][b0]*delta_f[b1][c0]*delta_f[d0][d1] + delta_f[a0][b0]*delta_f[b1][d0]*delta_f[c0][d1] + delta_f[a0][b0]*delta_f[b1][d1]*delta_f[c0][d0] + delta_f[a0][b1]*delta_f[b0][c0]*delta_f[d0][d1] + delta_f[a0][b1]*delta_f[b0][d0]*delta_f[c0][d1] + delta_f[a0][b1]*delta_f[b0][d1]*delta_f[c0][d0] + delta_f[a0][c0]*delta_f[b0][b1]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[b0][b1]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[b0][b1]*delta_f[c0][d0]) + (PB_0_f - PQ_b0_f)*(delta_f[a0][a1]*delta_f[b1][c0]*delta_f[d0][d1] + delta_f[a0][a1]*delta_f[b1][d0]*delta_f[c0][d1] + delta_f[a0][a1]*delta_f[b1][d1]*delta_f[c0][d0] + delta_f[a0][b1]*delta_f[a1][c0]*delta_f[d0][d1] + delta_f[a0][b1]*delta_f[a1][d0]*delta_f[c0][d1] + delta_f[a0][b1]*delta_f[a1][d1]*delta_f[c0][d0] + delta_f[a0][c0]*delta_f[a1][b1]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[a1][b1]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[a1][b1]*delta_f[c0][d0]) + (PB_1_f - PQ_b1_f)*(delta_f[a0][a1]*delta_f[b0][c0]*delta_f[d0][d1] + delta_f[a0][a1]*delta_f[b0][d0]*delta_f[c0][d1] + delta_f[a0][a1]*delta_f[b0][d1]*delta_f[c0][d0] + delta_f[a0][b0]*delta_f[a1][c0]*delta_f[d0][d1] + delta_f[a0][b0]*delta_f[a1][d0]*delta_f[c0][d1] + delta_f[a0][b0]*delta_f[a1][d1]*delta_f[c0][d0] + delta_f[a0][c0]*delta_f[a1][b0]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[a1][b0]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[a1][b0]*delta_f[c0][d0])

                        +PA_0_f*(delta_f[a1][c0]*(delta_f[b0][d0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][d0]) + delta_f[a1][d0]*(delta_f[b0][c0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][c0]) + delta_f[a1][d1]*(delta_f[b0][c0]*delta_f[b1][d0] + delta_f[b0][d0]*delta_f[b1][c0])) + PA_1_f*(delta_f[a0][c0]*(delta_f[b0][d0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][d0]) + delta_f[a0][d0]*(delta_f[b0][c0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][c0]) + delta_f[a0][d1]*(delta_f[b0][c0]*delta_f[b1][d0] + delta_f[b0][d0]*delta_f[b1][c0])) + PB_0_f*(delta_f[a0][c0]*(delta_f[a1][d0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b1][d0]) + delta_f[a0][d0]*(delta_f[a1][c0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b1][c0]) + delta_f[a0][d1]*(delta_f[a1][c0]*delta_f[b1][d0] + delta_f[a1][d0]*delta_f[b1][c0])) + PB_1_f*(delta_f[a0][c0]*(delta_f[a1][d0]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][d0]) + delta_f[a0][d0]*(delta_f[a1][c0]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][c0]) + delta_f[a0][d1]*(delta_f[a1][c0]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][c0]))

                        -2*(PQ_c0_f*delta_f[d0][d1] + PQ_d0_f*delta_f[c0][d1] + PQ_d1_f*delta_f[c0][d0])*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])

                        -PQ_c0_f*(delta_f[a0][b0]*(delta_f[a1][d0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b1][d0]) + delta_f[a0][d0]*(delta_f[a1][b0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b0][b1]) + delta_f[a0][d1]*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + delta_f[b0][d0]*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + delta_f[b0][d1]*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1])) - PQ_d0_f*(delta_f[a0][b0]*(delta_f[a1][c0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b1][c0]) + delta_f[a0][c0]*(delta_f[a1][b0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b0][b1]) + delta_f[a0][d1]*(delta_f[a1][b0]*delta_f[b1][c0] + delta_f[a1][c0]*delta_f[b0][b1]) + delta_f[b0][c0]*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + delta_f[b0][d1]*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1])) - PQ_d1_f*(delta_f[a0][b0]*(delta_f[a1][c0]*delta_f[b1][d0] + delta_f[a1][d0]*delta_f[b1][c0]) + delta_f[a0][c0]*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + delta_f[a0][d0]*(delta_f[a1][b0]*delta_f[b1][c0] + delta_f[a1][c0]*delta_f[b0][b1]) + delta_f[b0][c0]*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + delta_f[b0][d0]*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1]))

                    )

                    + F7_t_f[3] * 0.25f * S1_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        +(delta_f[a0][c0]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[c0][d0])*(-PA_1_f*PB_0_f*PQ_b1_f - PA_1_f*PB_1_f*PQ_b0_f - PB_0_f*PB_1_f*PQ_a1_f) + (delta_f[a1][c0]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[c0][d1] + delta_f[a1][d1]*delta_f[c0][d0])*(-PA_0_f*PB_0_f*PQ_b1_f - PA_0_f*PB_1_f*PQ_b0_f - PB_0_f*PB_1_f*PQ_a0_f) + (delta_f[b0][c0]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][d0])*(-PA_0_f*PA_1_f*PQ_b1_f - PA_0_f*PB_1_f*PQ_a1_f - PA_1_f*PB_1_f*PQ_a0_f) + (delta_f[b1][c0]*delta_f[d0][d1] + delta_f[b1][d0]*delta_f[c0][d1] + delta_f[b1][d1]*delta_f[c0][d0])*(-PA_0_f*PA_1_f*PQ_b0_f - PA_0_f*PB_0_f*PQ_a1_f - PA_1_f*PB_0_f*PQ_a0_f)

                        +(PQ_c0_f*delta_f[d0][d1] + PQ_d0_f*delta_f[c0][d1] + PQ_d1_f*delta_f[c0][d0])*(PA_0_f*(-PA_1_f*delta_f[b0][b1] - PB_0_f*delta_f[a1][b1] - PB_1_f*delta_f[a1][b0] + PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) + PA_1_f*(PQ_a0_f*delta_f[b0][b1] + delta_f[a0][b0]*(-PB_1_f + PQ_b1_f) + delta_f[a0][b1]*(-PB_0_f + PQ_b0_f)) + PB_0_f*(PQ_a0_f*delta_f[a1][b1] + PQ_a1_f*delta_f[a0][b1] + delta_f[a0][a1]*(-PB_1_f + PQ_b1_f)) + PB_1_f*(PQ_a0_f*delta_f[a1][b0] + PQ_a1_f*delta_f[a0][b0] + PQ_b0_f*delta_f[a0][a1]))

                        -PA_0_f*(PA_1_f*(PQ_c0_f*(delta_f[b0][d0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][d0]) + PQ_d0_f*(delta_f[b0][c0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][c0]) + PQ_d1_f*(delta_f[b0][c0]*delta_f[b1][d0] + delta_f[b0][d0]*delta_f[b1][c0])) + PB_0_f*(PQ_c0_f*(delta_f[a1][d0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b1][d0]) + PQ_d0_f*(delta_f[a1][c0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b1][c0]) + PQ_d1_f*(delta_f[a1][c0]*delta_f[b1][d0] + delta_f[a1][d0]*delta_f[b1][c0])) + PB_1_f*(PQ_c0_f*(delta_f[a1][d0]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][d0]) + PQ_d0_f*(delta_f[a1][c0]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][c0]) + PQ_d1_f*(delta_f[a1][c0]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][c0]))) - PA_1_f*(PB_0_f*(PQ_c0_f*(delta_f[a0][d0]*delta_f[b1][d1] + delta_f[a0][d1]*delta_f[b1][d0]) + PQ_d0_f*(delta_f[a0][c0]*delta_f[b1][d1] + delta_f[a0][d1]*delta_f[b1][c0]) + PQ_d1_f*(delta_f[a0][c0]*delta_f[b1][d0] + delta_f[a0][d0]*delta_f[b1][c0])) + PB_1_f*(PQ_c0_f*(delta_f[a0][d0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][d0]) + PQ_d0_f*(delta_f[a0][c0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][c0]) + PQ_d1_f*(delta_f[a0][c0]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][c0]))) - PB_0_f*PB_1_f*(PQ_c0_f*(delta_f[a0][d0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][d0]) + PQ_d0_f*(delta_f[a0][c0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][c0]) + PQ_d1_f*(delta_f[a0][c0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][c0]))

                        +PQ_c0_f*(PQ_d0_f*(PA_0_f*(delta_f[a1][b0]*delta_f[b1][d1] + delta_f[a1][b1]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][b1]) + PA_1_f*(delta_f[a0][b0]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][b1]) + PB_0_f*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + PB_1_f*(delta_f[a0][a1]*delta_f[b0][d1] + delta_f[a0][b0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b0])) + PQ_d1_f*(PA_0_f*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + PA_1_f*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + PB_0_f*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + PB_1_f*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0]))) + PQ_d0_f*PQ_d1_f*(PA_0_f*(delta_f[a1][b0]*delta_f[b1][c0] + delta_f[a1][b1]*delta_f[b0][c0] + delta_f[a1][c0]*delta_f[b0][b1]) + PA_1_f*(delta_f[a0][b0]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[b0][c0] + delta_f[a0][c0]*delta_f[b0][b1]) + PB_0_f*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1]) + PB_1_f*(delta_f[a0][a1]*delta_f[b0][c0] + delta_f[a0][b0]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b0]))

                        +PQ_c0_f*PQ_d0_f*PQ_d1_f*(-delta_f[a0][a1]*delta_f[b0][b1] - delta_f[a0][b0]*delta_f[a1][b1] - delta_f[a0][b1]*delta_f[a1][b0])

                    )

                    + F7_t_f[3] * (-0.25f) * S2_f * S2_f * inv_S1_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        +(PQ_a0_f*(PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) + PQ_a1_f*(PQ_b0_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][b0]) + PQ_b0_f*PQ_b1_f*delta_f[a0][a1])*(QC_0_f*delta_f[d0][d1] + QD_0_f*delta_f[c0][d1] + QD_1_f*delta_f[c0][d0])

                        +QC_0_f*(QD_0_f*(PQ_a0_f*(delta_f[a1][b0]*delta_f[b1][d1] + delta_f[a1][b1]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][b1]) + PQ_a1_f*(delta_f[a0][b0]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][b1]) + PQ_b0_f*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + PQ_b1_f*(delta_f[a0][a1]*delta_f[b0][d1] + delta_f[a0][b0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b0])) + QD_1_f*(PQ_a0_f*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + PQ_a1_f*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + PQ_b0_f*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + PQ_b1_f*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0]))) + QD_0_f*QD_1_f*(PQ_a0_f*(delta_f[a1][b0]*delta_f[b1][c0] + delta_f[a1][b1]*delta_f[b0][c0] + delta_f[a1][c0]*delta_f[b0][b1]) + PQ_a1_f*(delta_f[a0][b0]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[b0][c0] + delta_f[a0][c0]*delta_f[b0][b1]) + PQ_b0_f*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1]) + PQ_b1_f*(delta_f[a0][a1]*delta_f[b0][c0] + delta_f[a0][b0]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b0]))

                        +(PQ_d0_f*QC_0_f*QD_1_f + QD_0_f*(PQ_c0_f*QD_1_f + PQ_d1_f*QC_0_f))*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])

                    )

                    + F7_t_f[3] * 0.25f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        +(delta_f[c0][d0]*(PQ_d1_f + QD_1_f) + delta_f[c0][d1]*(PQ_d0_f + QD_0_f) + delta_f[d0][d1]*(PQ_c0_f + QC_0_f))*(PA_0_f*(PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) + PQ_a0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0] - PQ_a1_f*delta_f[b0][b1] - PQ_b0_f*delta_f[a1][b1] - PQ_b1_f*delta_f[a1][b0]) + PQ_a1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PQ_b0_f*(PB_1_f*delta_f[a0][a1] + delta_f[a0][b1]*(PA_1_f - PQ_a1_f)) + PQ_b1_f*(delta_f[a0][a1]*(PB_0_f - PQ_b0_f) + delta_f[a0][b0]*(PA_1_f - PQ_a1_f)))

                        +(delta_f[a0][c0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][c0])*(PB_0_f*PQ_b1_f*QD_1_f + PB_1_f*PQ_b0_f*QD_1_f) + (delta_f[a0][c0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][c0])*(PB_0_f*PQ_b1_f*QD_0_f + PB_1_f*PQ_b0_f*QD_0_f) + (delta_f[a0][c0]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][c0])*(PA_1_f*PQ_b1_f*QD_1_f + PB_1_f*PQ_a1_f*QD_1_f) + (delta_f[a0][c0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][c0])*(PA_1_f*PQ_b1_f*QD_0_f + PB_1_f*PQ_a1_f*QD_0_f) + (delta_f[a0][c0]*delta_f[b1][d0] + delta_f[a0][d0]*delta_f[b1][c0])*(PA_1_f*PQ_b0_f*QD_1_f + PB_0_f*PQ_a1_f*QD_1_f) + (delta_f[a0][c0]*delta_f[b1][d1] + delta_f[a0][d1]*delta_f[b1][c0])*(PA_1_f*PQ_b0_f*QD_0_f + PB_0_f*PQ_a1_f*QD_0_f) + (delta_f[a0][d0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][d0])*(PB_0_f*PQ_b1_f*QC_0_f + PB_1_f*PQ_b0_f*QC_0_f) + (delta_f[a0][d0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][d0])*(PA_1_f*PQ_b1_f*QC_0_f + PB_1_f*PQ_a1_f*QC_0_f) + (delta_f[a0][d0]*delta_f[b1][d1] + delta_f[a0][d1]*delta_f[b1][d0])*(PA_1_f*PQ_b0_f*QC_0_f + PB_0_f*PQ_a1_f*QC_0_f) + (delta_f[a1][c0]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][c0])*(PA_0_f*PQ_b1_f*QD_1_f + PB_1_f*PQ_a0_f*QD_1_f) + (delta_f[a1][c0]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][c0])*(PA_0_f*PQ_b1_f*QD_0_f + PB_1_f*PQ_a0_f*QD_0_f) + (delta_f[a1][c0]*delta_f[b1][d0] + delta_f[a1][d0]*delta_f[b1][c0])*(PA_0_f*PQ_b0_f*QD_1_f + PB_0_f*PQ_a0_f*QD_1_f) + (delta_f[a1][c0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b1][c0])*(PA_0_f*PQ_b0_f*QD_0_f + PB_0_f*PQ_a0_f*QD_0_f) + (delta_f[a1][d0]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][d0])*(PA_0_f*PQ_b1_f*QC_0_f + PB_1_f*PQ_a0_f*QC_0_f) + (delta_f[a1][d0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b1][d0])*(PA_0_f*PQ_b0_f*QC_0_f + PB_0_f*PQ_a0_f*QC_0_f) + (delta_f[b0][c0]*delta_f[b1][d0] + delta_f[b0][d0]*delta_f[b1][c0])*(PA_0_f*PQ_a1_f*QD_1_f + PA_1_f*PQ_a0_f*QD_1_f) + (delta_f[b0][c0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][c0])*(PA_0_f*PQ_a1_f*QD_0_f + PA_1_f*PQ_a0_f*QD_0_f) + (delta_f[b0][d0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][d0])*(PA_0_f*PQ_a1_f*QC_0_f + PA_1_f*PQ_a0_f*QC_0_f)

                        +(delta_f[a0][a1]*delta_f[b0][c0] + delta_f[a0][b0]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b0])*(PB_1_f*PQ_d0_f*QD_1_f + PB_1_f*PQ_d1_f*QD_0_f - PQ_b1_f*PQ_d0_f*QD_1_f - PQ_b1_f*PQ_d1_f*QD_0_f) + (delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0])*(PB_1_f*PQ_c0_f*QD_1_f + PB_1_f*PQ_d1_f*QC_0_f - PQ_b1_f*PQ_c0_f*QD_1_f - PQ_b1_f*PQ_d1_f*QC_0_f) + (delta_f[a0][a1]*delta_f[b0][d1] + delta_f[a0][b0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b0])*(PB_1_f*PQ_c0_f*QD_0_f + PB_1_f*PQ_d0_f*QC_0_f - PQ_b1_f*PQ_c0_f*QD_0_f - PQ_b1_f*PQ_d0_f*QC_0_f) + (delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1])*(PB_0_f*PQ_d0_f*QD_1_f + PB_0_f*PQ_d1_f*QD_0_f - PQ_b0_f*PQ_d0_f*QD_1_f - PQ_b0_f*PQ_d1_f*QD_0_f) + (delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1])*(PB_0_f*PQ_c0_f*QD_1_f + PB_0_f*PQ_d1_f*QC_0_f - PQ_b0_f*PQ_c0_f*QD_1_f - PQ_b0_f*PQ_d1_f*QC_0_f) + (delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1])*(PB_0_f*PQ_c0_f*QD_0_f + PB_0_f*PQ_d0_f*QC_0_f - PQ_b0_f*PQ_c0_f*QD_0_f - PQ_b0_f*PQ_d0_f*QC_0_f) + (delta_f[a0][b0]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[b0][c0] + delta_f[a0][c0]*delta_f[b0][b1])*(PA_1_f*PQ_d0_f*QD_1_f + PA_1_f*PQ_d1_f*QD_0_f - PQ_a1_f*PQ_d0_f*QD_1_f - PQ_a1_f*PQ_d1_f*QD_0_f) + (delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1])*(PA_1_f*PQ_c0_f*QD_1_f + PA_1_f*PQ_d1_f*QC_0_f - PQ_a1_f*PQ_c0_f*QD_1_f - PQ_a1_f*PQ_d1_f*QC_0_f) + (delta_f[a0][b0]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][b1])*(PA_1_f*PQ_c0_f*QD_0_f + PA_1_f*PQ_d0_f*QC_0_f - PQ_a1_f*PQ_c0_f*QD_0_f - PQ_a1_f*PQ_d0_f*QC_0_f) + (delta_f[a1][b0]*delta_f[b1][c0] + delta_f[a1][b1]*delta_f[b0][c0] + delta_f[a1][c0]*delta_f[b0][b1])*(PA_0_f*PQ_d0_f*QD_1_f + PA_0_f*PQ_d1_f*QD_0_f - PQ_a0_f*PQ_d0_f*QD_1_f - PQ_a0_f*PQ_d1_f*QD_0_f) + (delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1])*(PA_0_f*PQ_c0_f*QD_1_f + PA_0_f*PQ_d1_f*QC_0_f - PQ_a0_f*PQ_c0_f*QD_1_f - PQ_a0_f*PQ_d1_f*QC_0_f) + (delta_f[a1][b0]*delta_f[b1][d1] + delta_f[a1][b1]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][b1])*(PA_0_f*PQ_c0_f*QD_0_f + PA_0_f*PQ_d0_f*QC_0_f - PQ_a0_f*PQ_c0_f*QD_0_f - PQ_a0_f*PQ_d0_f*QC_0_f)

                        +(delta_f[a0][c0]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[c0][d0])*(PA_1_f*PQ_b0_f*PQ_b1_f + PB_0_f*PQ_a1_f*PQ_b1_f + PB_1_f*PQ_a1_f*PQ_b0_f) + (delta_f[a1][c0]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[c0][d1] + delta_f[a1][d1]*delta_f[c0][d0])*(PA_0_f*PQ_b0_f*PQ_b1_f + PB_0_f*PQ_a0_f*PQ_b1_f + PB_1_f*PQ_a0_f*PQ_b0_f) + (delta_f[b0][c0]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][d0])*(PA_0_f*PQ_a1_f*PQ_b1_f + PA_1_f*PQ_a0_f*PQ_b1_f + PB_1_f*PQ_a0_f*PQ_a1_f) + (delta_f[b1][c0]*delta_f[d0][d1] + delta_f[b1][d0]*delta_f[c0][d1] + delta_f[b1][d1]*delta_f[c0][d0])*(PA_0_f*PQ_a1_f*PQ_b0_f + PA_1_f*PQ_a0_f*PQ_b0_f + PB_0_f*PQ_a0_f*PQ_a1_f)

                        -2*(PQ_c0_f*(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f) + PQ_d0_f*PQ_d1_f*QC_0_f)*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])

                    )

                    + F7_t_f[3] * 0.5f * S1_f * S1_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        +(PA_0_f*PA_1_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PB_1_f*(PA_0_f*PQ_a1_f + PA_1_f*PQ_a0_f))*(PQ_c0_f*delta_f[d0][d1] + PQ_d0_f*delta_f[c0][d1] + PQ_d1_f*delta_f[c0][d0])

                        +PA_0_f*PA_1_f*(PQ_c0_f*(PB_0_f*(PQ_d0_f*delta_f[b1][d1] + PQ_d1_f*delta_f[b1][d0]) + PB_1_f*(PQ_d0_f*delta_f[b0][d1] + PQ_d1_f*delta_f[b0][d0])) + PQ_d0_f*PQ_d1_f*(PB_0_f*delta_f[b1][c0] + PB_1_f*delta_f[b0][c0])) + PB_0_f*PB_1_f*(PQ_c0_f*(PA_0_f*(PQ_d0_f*delta_f[a1][d1] + PQ_d1_f*delta_f[a1][d0]) + PA_1_f*(PQ_d0_f*delta_f[a0][d1] + PQ_d1_f*delta_f[a0][d0])) + PQ_d0_f*PQ_d1_f*(PA_0_f*delta_f[a1][c0] + PA_1_f*delta_f[a0][c0]))

                        +PQ_c0_f*PQ_d0_f*PQ_d1_f*(-PA_0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) - PA_1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) - PB_0_f*PB_1_f*delta_f[a0][a1])

                    )

                    + F7_t_f[3] * 0.5f * S1_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        -(delta_f[c0][d0]*(PQ_d1_f + QD_1_f) + delta_f[c0][d1]*(PQ_d0_f + QD_0_f) + delta_f[d0][d1]*(PQ_c0_f + QC_0_f))*(PA_0_f*PQ_b0_f*(PA_1_f*PQ_b1_f + PB_1_f*PQ_a1_f) + PA_1_f*PQ_a0_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PQ_a1_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))

                        -PA_0_f*(PA_1_f*(PQ_b0_f*(PQ_c0_f*(QD_0_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][d0]) + PQ_d0_f*(QC_0_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][c0]) + PQ_d1_f*(QC_0_f*delta_f[b1][d0] + QD_0_f*delta_f[b1][c0])) + PQ_b1_f*(PQ_c0_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0]) + PQ_d0_f*(QC_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][c0]) + PQ_d1_f*(QC_0_f*delta_f[b0][d0] + QD_0_f*delta_f[b0][c0]))) + PB_0_f*(PQ_a1_f*(PQ_c0_f*(QD_0_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][d0]) + PQ_d0_f*(QC_0_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][c0]) + PQ_d1_f*(QC_0_f*delta_f[b1][d0] + QD_0_f*delta_f[b1][c0])) + PQ_b1_f*(PQ_c0_f*(QD_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][d0]) + PQ_d0_f*(QC_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][c0]) + PQ_d1_f*(QC_0_f*delta_f[a1][d0] + QD_0_f*delta_f[a1][c0]))) + PB_1_f*(PQ_a1_f*(PQ_c0_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0]) + PQ_d0_f*(QC_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][c0]) + PQ_d1_f*(QC_0_f*delta_f[b0][d0] + QD_0_f*delta_f[b0][c0])) + PQ_b0_f*(PQ_c0_f*(QD_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][d0]) + PQ_d0_f*(QC_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][c0]) + PQ_d1_f*(QC_0_f*delta_f[a1][d0] + QD_0_f*delta_f[a1][c0])))) - PA_1_f*(PB_0_f*(PQ_a0_f*(PQ_c0_f*(QD_0_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][d0]) + PQ_d0_f*(QC_0_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][c0]) + PQ_d1_f*(QC_0_f*delta_f[b1][d0] + QD_0_f*delta_f[b1][c0])) + PQ_b1_f*(PQ_c0_f*(QD_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][d0]) + PQ_d0_f*(QC_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][c0]) + PQ_d1_f*(QC_0_f*delta_f[a0][d0] + QD_0_f*delta_f[a0][c0]))) + PB_1_f*(PQ_a0_f*(PQ_c0_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0]) + PQ_d0_f*(QC_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][c0]) + PQ_d1_f*(QC_0_f*delta_f[b0][d0] + QD_0_f*delta_f[b0][c0])) + PQ_b0_f*(PQ_c0_f*(QD_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][d0]) + PQ_d0_f*(QC_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][c0]) + PQ_d1_f*(QC_0_f*delta_f[a0][d0] + QD_0_f*delta_f[a0][c0])))) - PB_0_f*PB_1_f*(PQ_a0_f*(PQ_c0_f*(QD_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][d0]) + PQ_d0_f*(QC_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][c0]) + PQ_d1_f*(QC_0_f*delta_f[a1][d0] + QD_0_f*delta_f[a1][c0])) + PQ_a1_f*(PQ_c0_f*(QD_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][d0]) + PQ_d0_f*(QC_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][c0]) + PQ_d1_f*(QC_0_f*delta_f[a0][d0] + QD_0_f*delta_f[a0][c0])))

                        +(PQ_c0_f*(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f) + PQ_d0_f*PQ_d1_f*QC_0_f)*(PA_0_f*(-PA_1_f*delta_f[b0][b1] - PB_0_f*delta_f[a1][b1] - PB_1_f*delta_f[a1][b0] + PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) + PA_1_f*(PQ_a0_f*delta_f[b0][b1] + delta_f[a0][b0]*(-PB_1_f + PQ_b1_f) + delta_f[a0][b1]*(-PB_0_f + PQ_b0_f)) + PB_0_f*(PQ_a0_f*delta_f[a1][b1] + PQ_a1_f*delta_f[a0][b1] + delta_f[a0][a1]*(-PB_1_f + PQ_b1_f)) + PB_1_f*(PQ_a0_f*delta_f[a1][b0] + PQ_a1_f*delta_f[a0][b0] + PQ_b0_f*delta_f[a0][a1]))

                    )

                    + F7_t_f[3] * (-0.5f) * S2_f * S2_f * S2_f * inv_S1_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        +QC_0_f*QD_0_f*QD_1_f*(PQ_a0_f*(PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) + PQ_a1_f*(PQ_b0_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][b0]) + PQ_b0_f*PQ_b1_f*delta_f[a0][a1])

                    )

                    + F7_t_f[3] * 0.5f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        +(PQ_a0_f*PQ_b1_f*(PA_1_f*PQ_b0_f + PB_0_f*PQ_a1_f) + PQ_a1_f*PQ_b0_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))*(QC_0_f*delta_f[d0][d1] + QD_0_f*delta_f[c0][d1] + QD_1_f*delta_f[c0][d0])

                        +QC_0_f*(PQ_a0_f*(PA_1_f*(PQ_b0_f*(QD_0_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][d0]) + PQ_b1_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0])) + PB_0_f*(PQ_a1_f*(QD_0_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][d0]) + PQ_b1_f*(QD_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][d0])) + PB_1_f*(PQ_a1_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0]) + PQ_b0_f*(QD_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][d0]))) + PQ_a1_f*(PA_0_f*(PQ_b0_f*(QD_0_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][d0]) + PQ_b1_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0])) + (PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f)*(QD_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][d0])) + PQ_b0_f*PQ_b1_f*(PA_0_f*(QD_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][d0]) + PA_1_f*(QD_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][d0]))) + QD_0_f*QD_1_f*(PQ_a0_f*(PA_1_f*(PQ_b0_f*delta_f[b1][c0] + PQ_b1_f*delta_f[b0][c0]) + PB_0_f*(PQ_a1_f*delta_f[b1][c0] + PQ_b1_f*delta_f[a1][c0]) + PB_1_f*(PQ_a1_f*delta_f[b0][c0] + PQ_b0_f*delta_f[a1][c0])) + PQ_a1_f*(PA_0_f*(PQ_b0_f*delta_f[b1][c0] + PQ_b1_f*delta_f[b0][c0]) + delta_f[a0][c0]*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f)) + PQ_b0_f*PQ_b1_f*(PA_0_f*delta_f[a1][c0] + PA_1_f*delta_f[a0][c0]))

                        +(PQ_d0_f*QC_0_f*QD_1_f + QD_0_f*(PQ_c0_f*QD_1_f + PQ_d1_f*QC_0_f))*(PA_0_f*(PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) + PQ_a0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0] - PQ_a1_f*delta_f[b0][b1] - PQ_b0_f*delta_f[a1][b1] - PQ_b1_f*delta_f[a1][b0]) + PQ_a1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PQ_b0_f*(PB_1_f*delta_f[a0][a1] + delta_f[a0][b1]*(PA_1_f - PQ_a1_f)) + PQ_b1_f*(delta_f[a0][a1]*(PB_0_f - PQ_b0_f) + delta_f[a0][b0]*(PA_1_f - PQ_a1_f)))

                    )

                    + F7_t_f[3] * S1_f * S1_f * S1_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        -PA_0_f*PA_1_f*PB_0_f*PB_1_f*PQ_c0_f*PQ_d0_f*PQ_d1_f

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        const float D_f = pd_mat_D_f[kl];
        const float contrib_f = eri_ijkl_f * D_f * 2.0f;
        ERIs[threadIdx.y] += (double)contrib_f;
    }

    __syncthreads();

    if ((threadIdx.y == 0) && (ij < dd_prim_pair_count_local))
    {
        double J_ij = 0.0f;

        for (uint32_t n = 0; n < TILE_DIM_LARGE; n++)
        {
            J_ij += ERIs[n];
        }

        mat_J[ij] += J_ij;
    }
}

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDPD5_FP32_auto_s_ast(double*         mat_J,
                       const float*    p_prim_info_f,
                       const uint32_t  p_prim_count,
                       const float*    d_prim_info_f,
                       const uint32_t  d_prim_count,
                       const float*    pd_mat_D_f,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const float*    dd_pair_data_local_f,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* pd_first_inds,
                       const uint32_t* pd_second_inds,
                       const float*    pd_pair_data_f,
                       const uint32_t  pd_prim_pair_count,
                       const float*    boys_func_table_f,
                       const float*    boys_func_ft_f,
                       const uint32_t* prec_cut_ij_tile,
                       const uint32_t* screen_cut_ij_tile)
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ float   delta_f[3][3];

    __shared__ float a_i_f, a_j_f, r_i_f[3], r_j_f[3], S_ij_00_f, S1_f, inv_S1_f;
    __shared__ float PA_0_f, PA_1_f, PB_0_f, PB_1_f;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];
    const uint32_t screen_cut = screen_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta_f[0][0] = 1.0f; delta_f[0][1] = 0.0f; delta_f[0][2] = 0.0f;
        delta_f[1][0] = 0.0f; delta_f[1][1] = 1.0f; delta_f[1][2] = 0.0f;
        delta_f[2][0] = 0.0f; delta_f[2][1] = 0.0f; delta_f[2][2] = 1.0f;

        if (ij < dd_prim_pair_count_local)
        {
            i = dd_first_inds_local[ij];
            j = dd_second_inds_local[ij];

            a_i_f = d_prim_info_f[i / 6 + d_prim_count * 0];

            r_i_f[0] = d_prim_info_f[i / 6 + d_prim_count * 2];
            r_i_f[1] = d_prim_info_f[i / 6 + d_prim_count * 3];
            r_i_f[2] = d_prim_info_f[i / 6 + d_prim_count * 4];

            a_j_f = d_prim_info_f[j / 6 + d_prim_count * 0];

            r_j_f[0] = d_prim_info_f[j / 6 + d_prim_count * 2];
            r_j_f[1] = d_prim_info_f[j / 6 + d_prim_count * 3];
            r_j_f[2] = d_prim_info_f[j / 6 + d_prim_count * 4];

            S1_f = a_i_f + a_j_f;
            inv_S1_f = (float) (1.0 / (double)S1_f);

            S_ij_00_f = dd_pair_data_local_f[ij];

            a0 = d_cart_inds[i % 6][0];
            a1 = d_cart_inds[i % 6][1];
            b0 = d_cart_inds[j % 6][0];
            b1 = d_cart_inds[j % 6][1];

            PA_0_f = (a_j_f  * inv_S1_f) * (r_j_f[a0] - r_i_f[a0]);
            PA_1_f = (a_j_f  * inv_S1_f) * (r_j_f[a1] - r_i_f[a1]);
            PB_0_f = (-a_i_f * inv_S1_f) * (r_j_f[b0] - r_i_f[b0]);
            PB_1_f = (-a_i_f * inv_S1_f) * (r_j_f[b1] - r_i_f[b1]);

        }

    }

    ERIs[threadIdx.y] = 0.0;

    __syncthreads();

    for (uint32_t m = prec_cut; m < screen_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= pd_prim_pair_count))
        {
            break;
        }

        const auto k = pd_first_inds[kl];
        const auto l = pd_second_inds[kl];

        const auto a_k_f = p_prim_info_f[k / 3 + p_prim_count * 0];

        const float r_k0_f = p_prim_info_f[k / 3 + p_prim_count * 2];
        const float r_k1_f = p_prim_info_f[k / 3 + p_prim_count * 3];
        const float r_k2_f = p_prim_info_f[k / 3 + p_prim_count * 4];

        const auto a_l_f = d_prim_info_f[l / 6 + d_prim_count * 0];

        const float r_l0_f = d_prim_info_f[l / 6 + d_prim_count * 2];
        const float r_l1_f = d_prim_info_f[l / 6 + d_prim_count * 3];
        const float r_l2_f = d_prim_info_f[l / 6 + d_prim_count * 4];

        const auto S_kl_00_f = pd_pair_data_f[kl];

        const auto c0 = k % 3;
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        const float r_l_c0_f = (c0 == 0 ? r_l0_f : (c0 == 1 ? r_l1_f : r_l2_f));
        const float r_l_d0_f = (d0 == 0 ? r_l0_f : (d0 == 1 ? r_l1_f : r_l2_f));
        const float r_l_d1_f = (d1 == 0 ? r_l0_f : (d1 == 1 ? r_l1_f : r_l2_f));

        const float r_k_c0_f = (c0 == 0 ? r_k0_f : (c0 == 1 ? r_k1_f : r_k2_f));
        const float r_k_d0_f = (d0 == 0 ? r_k0_f : (d0 == 1 ? r_k1_f : r_k2_f));
        const float r_k_d1_f = (d1 == 0 ? r_k0_f : (d1 == 1 ? r_k1_f : r_k2_f));

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2_f = a_k_f + a_l_f;

        const auto inv_S2_f = 1.0f / S2_f;
        const auto inv_S4_f = 1.0f / (S1_f + S2_f);

        const float PQ0_f = (a_k_f * r_k0_f + a_l_f * r_l0_f) * inv_S2_f - (a_i_f * r_i_f[0] + a_j_f * r_j_f[0]) * inv_S1_f;
        const float PQ1_f = (a_k_f * r_k1_f + a_l_f * r_l1_f) * inv_S2_f - (a_i_f * r_i_f[1] + a_j_f * r_j_f[1]) * inv_S1_f;
        const float PQ2_f = (a_k_f * r_k2_f + a_l_f * r_l2_f) * inv_S2_f - (a_i_f * r_i_f[2] + a_j_f * r_j_f[2]) * inv_S1_f;

        const float PQ_a0_f = (a0 == 0 ? PQ0_f : (a0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_a1_f = (a1 == 0 ? PQ0_f : (a1 == 1 ? PQ1_f : PQ2_f));
        const float PQ_b0_f = (b0 == 0 ? PQ0_f : (b0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_b1_f = (b1 == 0 ? PQ0_f : (b1 == 1 ? PQ1_f : PQ2_f));
        const float PQ_c0_f = (c0 == 0 ? PQ0_f : (c0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_d0_f = (d0 == 0 ? PQ0_f : (d0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_d1_f = (d1 == 0 ? PQ0_f : (d1 == 1 ? PQ1_f : PQ2_f));

        const auto r2_PQ_f = PQ0_f * PQ0_f + PQ1_f * PQ1_f + PQ2_f * PQ2_f;

        const auto Lambda_f = sqrtf(4.0f * S1_f * S2_f * MATH_CONST_INV_PI_F * inv_S4_f);

        float F7_t_f[5];

        gpu::computeBoysFunction_f(F7_t_f, S1_f * S2_f * inv_S4_f * r2_PQ_f, 4, boys_func_table_f, boys_func_ft_f);

        const auto QC_0_f = (a_l_f * inv_S2_f) * (r_l_c0_f - r_k_c0_f);
        const auto QD_0_f = (-a_k_f * inv_S2_f) * (r_l_d0_f - r_k_d0_f);
        const auto QD_1_f = (-a_k_f * inv_S2_f) * (r_l_d1_f - r_k_d1_f);

        const float eri_ijkl_f = Lambda_f * S_ij_00_f * S_kl_00_f * (

                    + F7_t_f[3] * S1_f * S1_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        +(PQ_c0_f*(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f) + PQ_d0_f*PQ_d1_f*QC_0_f)*(PA_0_f*PA_1_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PB_1_f*(PA_0_f*PQ_a1_f + PA_1_f*PQ_a0_f))

                    )

                    + F7_t_f[3] * S1_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        -(PQ_d0_f*QC_0_f*QD_1_f + QD_0_f*(PQ_c0_f*QD_1_f + PQ_d1_f*QC_0_f))*(PA_0_f*PQ_b0_f*(PA_1_f*PQ_b1_f + PB_1_f*PQ_a1_f) + PA_1_f*PQ_a0_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PQ_a1_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))

                    )

                    + F7_t_f[3] * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        +QC_0_f*QD_0_f*QD_1_f*(PQ_a0_f*PQ_b1_f*(PA_1_f*PQ_b0_f + PB_0_f*PQ_a1_f) + PQ_a1_f*PQ_b0_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))

                    )

                    + F7_t_f[4] * 0.25f * S1_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        +(delta_f[a0][c0]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[c0][d0])*(-PA_1_f*PQ_b0_f*PQ_b1_f - PB_0_f*PQ_a1_f*PQ_b1_f - PB_1_f*PQ_a1_f*PQ_b0_f) + (delta_f[a1][c0]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[c0][d1] + delta_f[a1][d1]*delta_f[c0][d0])*(-PA_0_f*PQ_b0_f*PQ_b1_f - PB_0_f*PQ_a0_f*PQ_b1_f - PB_1_f*PQ_a0_f*PQ_b0_f) + (delta_f[b0][c0]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][d0])*(-PA_0_f*PQ_a1_f*PQ_b1_f - PA_1_f*PQ_a0_f*PQ_b1_f - PB_1_f*PQ_a0_f*PQ_a1_f) + (delta_f[b1][c0]*delta_f[d0][d1] + delta_f[b1][d0]*delta_f[c0][d1] + delta_f[b1][d1]*delta_f[c0][d0])*(-PA_0_f*PQ_a1_f*PQ_b0_f - PA_1_f*PQ_a0_f*PQ_b0_f - PB_0_f*PQ_a0_f*PQ_a1_f)

                        +(PQ_c0_f*delta_f[d0][d1] + PQ_d0_f*delta_f[c0][d1] + PQ_d1_f*delta_f[c0][d0])*(-PA_0_f*(PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) - PA_1_f*(PQ_a0_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][b0]) - PB_0_f*(PQ_a0_f*delta_f[a1][b1] + PQ_a1_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][a1]) - PB_1_f*(PQ_a0_f*delta_f[a1][b0] + PQ_a1_f*delta_f[a0][b0] + PQ_b0_f*delta_f[a0][a1]) + PQ_a0_f*PQ_a1_f*delta_f[b0][b1] + PQ_a0_f*PQ_b0_f*delta_f[a1][b1] + PQ_a0_f*PQ_b1_f*delta_f[a1][b0] + PQ_a1_f*PQ_b0_f*delta_f[a0][b1] + PQ_a1_f*PQ_b1_f*delta_f[a0][b0] + PQ_b0_f*PQ_b1_f*delta_f[a0][a1])

                        -PA_0_f*(PQ_a1_f*(PQ_c0_f*(delta_f[b0][d0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][d0]) + PQ_d0_f*(delta_f[b0][c0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][c0]) + PQ_d1_f*(delta_f[b0][c0]*delta_f[b1][d0] + delta_f[b0][d0]*delta_f[b1][c0])) + PQ_b0_f*(PQ_c0_f*(delta_f[a1][d0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b1][d0]) + PQ_d0_f*(delta_f[a1][c0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b1][c0]) + PQ_d1_f*(delta_f[a1][c0]*delta_f[b1][d0] + delta_f[a1][d0]*delta_f[b1][c0])) + PQ_b1_f*(PQ_c0_f*(delta_f[a1][d0]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][d0]) + PQ_d0_f*(delta_f[a1][c0]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][c0]) + PQ_d1_f*(delta_f[a1][c0]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][c0]))) - PA_1_f*(PQ_a0_f*(PQ_c0_f*(delta_f[b0][d0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][d0]) + PQ_d0_f*(delta_f[b0][c0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][c0]) + PQ_d1_f*(delta_f[b0][c0]*delta_f[b1][d0] + delta_f[b0][d0]*delta_f[b1][c0])) + PQ_b0_f*(PQ_c0_f*(delta_f[a0][d0]*delta_f[b1][d1] + delta_f[a0][d1]*delta_f[b1][d0]) + PQ_d0_f*(delta_f[a0][c0]*delta_f[b1][d1] + delta_f[a0][d1]*delta_f[b1][c0]) + PQ_d1_f*(delta_f[a0][c0]*delta_f[b1][d0] + delta_f[a0][d0]*delta_f[b1][c0])) + PQ_b1_f*(PQ_c0_f*(delta_f[a0][d0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][d0]) + PQ_d0_f*(delta_f[a0][c0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][c0]) + PQ_d1_f*(delta_f[a0][c0]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][c0]))) - PB_0_f*(PQ_a0_f*(PQ_c0_f*(delta_f[a1][d0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b1][d0]) + PQ_d0_f*(delta_f[a1][c0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b1][c0]) + PQ_d1_f*(delta_f[a1][c0]*delta_f[b1][d0] + delta_f[a1][d0]*delta_f[b1][c0])) + PQ_a1_f*(PQ_c0_f*(delta_f[a0][d0]*delta_f[b1][d1] + delta_f[a0][d1]*delta_f[b1][d0]) + PQ_d0_f*(delta_f[a0][c0]*delta_f[b1][d1] + delta_f[a0][d1]*delta_f[b1][c0]) + PQ_d1_f*(delta_f[a0][c0]*delta_f[b1][d0] + delta_f[a0][d0]*delta_f[b1][c0])) + PQ_b1_f*(PQ_c0_f*(delta_f[a0][d0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][d0]) + PQ_d0_f*(delta_f[a0][c0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][c0]) + PQ_d1_f*(delta_f[a0][c0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][c0]))) - PB_1_f*(PQ_a0_f*(PQ_c0_f*(delta_f[a1][d0]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][d0]) + PQ_d0_f*(delta_f[a1][c0]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][c0]) + PQ_d1_f*(delta_f[a1][c0]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][c0])) + PQ_a1_f*(PQ_c0_f*(delta_f[a0][d0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][d0]) + PQ_d0_f*(delta_f[a0][c0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][c0]) + PQ_d1_f*(delta_f[a0][c0]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][c0])) + PQ_b0_f*(PQ_c0_f*(delta_f[a0][d0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][d0]) + PQ_d0_f*(delta_f[a0][c0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][c0]) + PQ_d1_f*(delta_f[a0][c0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][c0])))

                        +(-PA_0_f*PQ_c0_f*PQ_d0_f + PQ_a0_f*PQ_c0_f*PQ_d0_f)*(delta_f[a1][b0]*delta_f[b1][d1] + delta_f[a1][b1]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][b1]) + (-PA_0_f*PQ_c0_f*PQ_d1_f + PQ_a0_f*PQ_c0_f*PQ_d1_f)*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + (-PA_0_f*PQ_d0_f*PQ_d1_f + PQ_a0_f*PQ_d0_f*PQ_d1_f)*(delta_f[a1][b0]*delta_f[b1][c0] + delta_f[a1][b1]*delta_f[b0][c0] + delta_f[a1][c0]*delta_f[b0][b1]) + (-PA_1_f*PQ_c0_f*PQ_d0_f + PQ_a1_f*PQ_c0_f*PQ_d0_f)*(delta_f[a0][b0]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][b1]) + (-PA_1_f*PQ_c0_f*PQ_d1_f + PQ_a1_f*PQ_c0_f*PQ_d1_f)*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + (-PA_1_f*PQ_d0_f*PQ_d1_f + PQ_a1_f*PQ_d0_f*PQ_d1_f)*(delta_f[a0][b0]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[b0][c0] + delta_f[a0][c0]*delta_f[b0][b1]) + (-PB_0_f*PQ_c0_f*PQ_d0_f + PQ_b0_f*PQ_c0_f*PQ_d0_f)*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + (-PB_0_f*PQ_c0_f*PQ_d1_f + PQ_b0_f*PQ_c0_f*PQ_d1_f)*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + (-PB_0_f*PQ_d0_f*PQ_d1_f + PQ_b0_f*PQ_d0_f*PQ_d1_f)*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1]) + (-PB_1_f*PQ_c0_f*PQ_d0_f + PQ_b1_f*PQ_c0_f*PQ_d0_f)*(delta_f[a0][a1]*delta_f[b0][d1] + delta_f[a0][b0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b0]) + (-PB_1_f*PQ_c0_f*PQ_d1_f + PQ_b1_f*PQ_c0_f*PQ_d1_f)*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0]) + (-PB_1_f*PQ_d0_f*PQ_d1_f + PQ_b1_f*PQ_d0_f*PQ_d1_f)*(delta_f[a0][a1]*delta_f[b0][c0] + delta_f[a0][b0]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b0])

                        +2.0f*PQ_c0_f*PQ_d0_f*PQ_d1_f*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])

                    )

                    + F7_t_f[4] * 0.25f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        +(PQ_a0_f*(PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) + PQ_a1_f*(PQ_b0_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][b0]) + PQ_b0_f*PQ_b1_f*delta_f[a0][a1])*(delta_f[c0][d0]*(PQ_d1_f + QD_1_f) + delta_f[c0][d1]*(PQ_d0_f + QD_0_f) + delta_f[d0][d1]*(PQ_c0_f + QC_0_f))

                        +PQ_a0_f*(PQ_a1_f*(QC_0_f*(delta_f[b0][d0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][d0]) + QD_0_f*(delta_f[b0][c0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][c0]) + QD_1_f*(delta_f[b0][c0]*delta_f[b1][d0] + delta_f[b0][d0]*delta_f[b1][c0])) + PQ_b0_f*(QC_0_f*(delta_f[a1][d0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b1][d0]) + QD_0_f*(delta_f[a1][c0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b1][c0]) + QD_1_f*(delta_f[a1][c0]*delta_f[b1][d0] + delta_f[a1][d0]*delta_f[b1][c0])) + PQ_b1_f*(QC_0_f*(delta_f[a1][d0]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][d0]) + QD_0_f*(delta_f[a1][c0]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][c0]) + QD_1_f*(delta_f[a1][c0]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][c0]))) + PQ_a1_f*(PQ_b0_f*(QC_0_f*(delta_f[a0][d0]*delta_f[b1][d1] + delta_f[a0][d1]*delta_f[b1][d0]) + QD_0_f*(delta_f[a0][c0]*delta_f[b1][d1] + delta_f[a0][d1]*delta_f[b1][c0]) + QD_1_f*(delta_f[a0][c0]*delta_f[b1][d0] + delta_f[a0][d0]*delta_f[b1][c0])) + PQ_b1_f*(QC_0_f*(delta_f[a0][d0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][d0]) + QD_0_f*(delta_f[a0][c0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][c0]) + QD_1_f*(delta_f[a0][c0]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][c0]))) + PQ_b0_f*PQ_b1_f*(QC_0_f*(delta_f[a0][d0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][d0]) + QD_0_f*(delta_f[a0][c0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][c0]) + QD_1_f*(delta_f[a0][c0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][c0]))

                        +(PQ_a0_f*PQ_c0_f*QD_0_f + PQ_a0_f*PQ_d0_f*QC_0_f)*(delta_f[a1][b0]*delta_f[b1][d1] + delta_f[a1][b1]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][b1]) + (PQ_a0_f*PQ_c0_f*QD_1_f + PQ_a0_f*PQ_d1_f*QC_0_f)*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + (PQ_a0_f*PQ_d0_f*QD_1_f + PQ_a0_f*PQ_d1_f*QD_0_f)*(delta_f[a1][b0]*delta_f[b1][c0] + delta_f[a1][b1]*delta_f[b0][c0] + delta_f[a1][c0]*delta_f[b0][b1]) + (PQ_a1_f*PQ_c0_f*QD_0_f + PQ_a1_f*PQ_d0_f*QC_0_f)*(delta_f[a0][b0]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][b1]) + (PQ_a1_f*PQ_c0_f*QD_1_f + PQ_a1_f*PQ_d1_f*QC_0_f)*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + (PQ_a1_f*PQ_d0_f*QD_1_f + PQ_a1_f*PQ_d1_f*QD_0_f)*(delta_f[a0][b0]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[b0][c0] + delta_f[a0][c0]*delta_f[b0][b1]) + (PQ_b0_f*PQ_c0_f*QD_0_f + PQ_b0_f*PQ_d0_f*QC_0_f)*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + (PQ_b0_f*PQ_c0_f*QD_1_f + PQ_b0_f*PQ_d1_f*QC_0_f)*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + (PQ_b0_f*PQ_d0_f*QD_1_f + PQ_b0_f*PQ_d1_f*QD_0_f)*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1]) + (PQ_b1_f*PQ_c0_f*QD_0_f + PQ_b1_f*PQ_d0_f*QC_0_f)*(delta_f[a0][a1]*delta_f[b0][d1] + delta_f[a0][b0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b0]) + (PQ_b1_f*PQ_c0_f*QD_1_f + PQ_b1_f*PQ_d1_f*QC_0_f)*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0]) + (PQ_b1_f*PQ_d0_f*QD_1_f + PQ_b1_f*PQ_d1_f*QD_0_f)*(delta_f[a0][a1]*delta_f[b0][c0] + delta_f[a0][b0]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b0])

                        +(PQ_c0_f*(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f) + PQ_d0_f*PQ_d1_f*QC_0_f)*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])

                        +PQ_a0_f*PQ_a1_f*(PQ_b0_f*(delta_f[b1][c0]*delta_f[d0][d1] + delta_f[b1][d0]*delta_f[c0][d1] + delta_f[b1][d1]*delta_f[c0][d0]) + PQ_b1_f*(delta_f[b0][c0]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][d0])) + PQ_b0_f*PQ_b1_f*(PQ_a0_f*(delta_f[a1][c0]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[c0][d1] + delta_f[a1][d1]*delta_f[c0][d0]) + PQ_a1_f*(delta_f[a0][c0]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[c0][d0]))

                    )

                    + F7_t_f[4] * 0.5f * S1_f * S1_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        +(PQ_c0_f*delta_f[d0][d1] + PQ_d0_f*delta_f[c0][d1] + PQ_d1_f*delta_f[c0][d0])*(PA_0_f*PQ_b0_f*(PA_1_f*PQ_b1_f + PB_1_f*PQ_a1_f) + PA_1_f*PQ_a0_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PQ_a1_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))

                        +PQ_c0_f*(PA_0_f*(PA_1_f*(PQ_b0_f*(PQ_d0_f*delta_f[b1][d1] + PQ_d1_f*delta_f[b1][d0]) + PQ_b1_f*(PQ_d0_f*delta_f[b0][d1] + PQ_d1_f*delta_f[b0][d0])) + PB_0_f*(PQ_a1_f*(PQ_d0_f*delta_f[b1][d1] + PQ_d1_f*delta_f[b1][d0]) + PQ_b1_f*(PQ_d0_f*delta_f[a1][d1] + PQ_d1_f*delta_f[a1][d0])) + PB_1_f*(PQ_a1_f*(PQ_d0_f*delta_f[b0][d1] + PQ_d1_f*delta_f[b0][d0]) + PQ_b0_f*(PQ_d0_f*delta_f[a1][d1] + PQ_d1_f*delta_f[a1][d0]))) + PA_1_f*(PB_0_f*(PQ_a0_f*(PQ_d0_f*delta_f[b1][d1] + PQ_d1_f*delta_f[b1][d0]) + PQ_b1_f*(PQ_d0_f*delta_f[a0][d1] + PQ_d1_f*delta_f[a0][d0])) + PB_1_f*(PQ_a0_f*(PQ_d0_f*delta_f[b0][d1] + PQ_d1_f*delta_f[b0][d0]) + PQ_b0_f*(PQ_d0_f*delta_f[a0][d1] + PQ_d1_f*delta_f[a0][d0]))) + PB_0_f*PB_1_f*(PQ_a0_f*(PQ_d0_f*delta_f[a1][d1] + PQ_d1_f*delta_f[a1][d0]) + PQ_a1_f*(PQ_d0_f*delta_f[a0][d1] + PQ_d1_f*delta_f[a0][d0]))) + PQ_d0_f*PQ_d1_f*(PA_0_f*(PA_1_f*(PQ_b0_f*delta_f[b1][c0] + PQ_b1_f*delta_f[b0][c0]) + PB_0_f*(PQ_a1_f*delta_f[b1][c0] + PQ_b1_f*delta_f[a1][c0]) + PB_1_f*(PQ_a1_f*delta_f[b0][c0] + PQ_b0_f*delta_f[a1][c0])) + PA_1_f*(PB_0_f*(PQ_a0_f*delta_f[b1][c0] + PQ_b1_f*delta_f[a0][c0]) + PB_1_f*(PQ_a0_f*delta_f[b0][c0] + PQ_b0_f*delta_f[a0][c0])) + PB_0_f*PB_1_f*(PQ_a0_f*delta_f[a1][c0] + PQ_a1_f*delta_f[a0][c0]))

                        +PQ_c0_f*PQ_d0_f*PQ_d1_f*(PA_0_f*PA_1_f*delta_f[b0][b1] - PA_0_f*(PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) - PA_1_f*(PQ_a0_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][b0]) + PB_0_f*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1] - PQ_a0_f*delta_f[a1][b1] - PQ_a1_f*delta_f[a0][b1] - PQ_b1_f*delta_f[a0][a1]) + PB_1_f*(PA_0_f*delta_f[a1][b0] + PA_1_f*delta_f[a0][b0] - PQ_a0_f*delta_f[a1][b0] - PQ_a1_f*delta_f[a0][b0] - PQ_b0_f*delta_f[a0][a1]))

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        const float D_f = pd_mat_D_f[kl];
        const float contrib_f = eri_ijkl_f * D_f * 2.0f;
        ERIs[threadIdx.y] += (double)contrib_f;
    }

    __syncthreads();

    if ((threadIdx.y == 0) && (ij < dd_prim_pair_count_local))
    {
        double J_ij = 0.0f;

        for (uint32_t n = 0; n < TILE_DIM_LARGE; n++)
        {
            J_ij += ERIs[n];
        }

        mat_J[ij] += J_ij;
    }
}

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDPD6_FP32_auto_s_ast(double*         mat_J,
                       const float*    p_prim_info_f,
                       const uint32_t  p_prim_count,
                       const float*    d_prim_info_f,
                       const uint32_t  d_prim_count,
                       const float*    pd_mat_D_f,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const float*    dd_pair_data_local_f,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* pd_first_inds,
                       const uint32_t* pd_second_inds,
                       const float*    pd_pair_data_f,
                       const uint32_t  pd_prim_pair_count,
                       const float*    boys_func_table_f,
                       const float*    boys_func_ft_f,
                       const uint32_t* prec_cut_ij_tile,
                       const uint32_t* screen_cut_ij_tile)
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ float   delta_f[3][3];

    __shared__ float a_i_f, a_j_f, r_i_f[3], r_j_f[3], S_ij_00_f, S1_f, inv_S1_f;
    __shared__ float PA_0_f, PA_1_f, PB_0_f, PB_1_f;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];
    const uint32_t screen_cut = screen_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta_f[0][0] = 1.0f; delta_f[0][1] = 0.0f; delta_f[0][2] = 0.0f;
        delta_f[1][0] = 0.0f; delta_f[1][1] = 1.0f; delta_f[1][2] = 0.0f;
        delta_f[2][0] = 0.0f; delta_f[2][1] = 0.0f; delta_f[2][2] = 1.0f;

        if (ij < dd_prim_pair_count_local)
        {
            i = dd_first_inds_local[ij];
            j = dd_second_inds_local[ij];

            a_i_f = d_prim_info_f[i / 6 + d_prim_count * 0];

            r_i_f[0] = d_prim_info_f[i / 6 + d_prim_count * 2];
            r_i_f[1] = d_prim_info_f[i / 6 + d_prim_count * 3];
            r_i_f[2] = d_prim_info_f[i / 6 + d_prim_count * 4];

            a_j_f = d_prim_info_f[j / 6 + d_prim_count * 0];

            r_j_f[0] = d_prim_info_f[j / 6 + d_prim_count * 2];
            r_j_f[1] = d_prim_info_f[j / 6 + d_prim_count * 3];
            r_j_f[2] = d_prim_info_f[j / 6 + d_prim_count * 4];

            S1_f = a_i_f + a_j_f;
            inv_S1_f = (float) (1.0 / (double)S1_f);

            S_ij_00_f = dd_pair_data_local_f[ij];

            a0 = d_cart_inds[i % 6][0];
            a1 = d_cart_inds[i % 6][1];
            b0 = d_cart_inds[j % 6][0];
            b1 = d_cart_inds[j % 6][1];

            PA_0_f = (a_j_f  * inv_S1_f) * (r_j_f[a0] - r_i_f[a0]);
            PA_1_f = (a_j_f  * inv_S1_f) * (r_j_f[a1] - r_i_f[a1]);
            PB_0_f = (-a_i_f * inv_S1_f) * (r_j_f[b0] - r_i_f[b0]);
            PB_1_f = (-a_i_f * inv_S1_f) * (r_j_f[b1] - r_i_f[b1]);

        }

    }

    ERIs[threadIdx.y] = 0.0;

    __syncthreads();

    for (uint32_t m = prec_cut; m < screen_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= pd_prim_pair_count))
        {
            break;
        }

        const auto k = pd_first_inds[kl];
        const auto l = pd_second_inds[kl];

        const auto a_k_f = p_prim_info_f[k / 3 + p_prim_count * 0];

        const float r_k0_f = p_prim_info_f[k / 3 + p_prim_count * 2];
        const float r_k1_f = p_prim_info_f[k / 3 + p_prim_count * 3];
        const float r_k2_f = p_prim_info_f[k / 3 + p_prim_count * 4];

        const auto a_l_f = d_prim_info_f[l / 6 + d_prim_count * 0];

        const float r_l0_f = d_prim_info_f[l / 6 + d_prim_count * 2];
        const float r_l1_f = d_prim_info_f[l / 6 + d_prim_count * 3];
        const float r_l2_f = d_prim_info_f[l / 6 + d_prim_count * 4];

        const auto S_kl_00_f = pd_pair_data_f[kl];

        const auto c0 = k % 3;
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        const float r_l_c0_f = (c0 == 0 ? r_l0_f : (c0 == 1 ? r_l1_f : r_l2_f));
        const float r_l_d0_f = (d0 == 0 ? r_l0_f : (d0 == 1 ? r_l1_f : r_l2_f));
        const float r_l_d1_f = (d1 == 0 ? r_l0_f : (d1 == 1 ? r_l1_f : r_l2_f));

        const float r_k_c0_f = (c0 == 0 ? r_k0_f : (c0 == 1 ? r_k1_f : r_k2_f));
        const float r_k_d0_f = (d0 == 0 ? r_k0_f : (d0 == 1 ? r_k1_f : r_k2_f));
        const float r_k_d1_f = (d1 == 0 ? r_k0_f : (d1 == 1 ? r_k1_f : r_k2_f));

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2_f = a_k_f + a_l_f;

        const auto inv_S2_f = 1.0f / S2_f;
        const auto inv_S4_f = 1.0f / (S1_f + S2_f);

        const float PQ0_f = (a_k_f * r_k0_f + a_l_f * r_l0_f) * inv_S2_f - (a_i_f * r_i_f[0] + a_j_f * r_j_f[0]) * inv_S1_f;
        const float PQ1_f = (a_k_f * r_k1_f + a_l_f * r_l1_f) * inv_S2_f - (a_i_f * r_i_f[1] + a_j_f * r_j_f[1]) * inv_S1_f;
        const float PQ2_f = (a_k_f * r_k2_f + a_l_f * r_l2_f) * inv_S2_f - (a_i_f * r_i_f[2] + a_j_f * r_j_f[2]) * inv_S1_f;

        const float PQ_a0_f = (a0 == 0 ? PQ0_f : (a0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_a1_f = (a1 == 0 ? PQ0_f : (a1 == 1 ? PQ1_f : PQ2_f));
        const float PQ_b0_f = (b0 == 0 ? PQ0_f : (b0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_b1_f = (b1 == 0 ? PQ0_f : (b1 == 1 ? PQ1_f : PQ2_f));
        const float PQ_c0_f = (c0 == 0 ? PQ0_f : (c0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_d0_f = (d0 == 0 ? PQ0_f : (d0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_d1_f = (d1 == 0 ? PQ0_f : (d1 == 1 ? PQ1_f : PQ2_f));

        const auto r2_PQ_f = PQ0_f * PQ0_f + PQ1_f * PQ1_f + PQ2_f * PQ2_f;

        const auto Lambda_f = sqrtf(4.0f * S1_f * S2_f * MATH_CONST_INV_PI_F * inv_S4_f);

        float F7_t_f[5];

        gpu::computeBoysFunction_f(F7_t_f, S1_f * S2_f * inv_S4_f * r2_PQ_f, 4, boys_func_table_f, boys_func_ft_f);

        const auto QC_0_f = (a_l_f * inv_S2_f) * (r_l_c0_f - r_k_c0_f);
        const auto QD_0_f = (-a_k_f * inv_S2_f) * (r_l_d0_f - r_k_d0_f);
        const auto QD_1_f = (-a_k_f * inv_S2_f) * (r_l_d1_f - r_k_d1_f);

        const float eri_ijkl_f = Lambda_f * S_ij_00_f * S_kl_00_f * (

                    + F7_t_f[4] * 0.5f * S1_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        -(PQ_a0_f*PQ_b1_f*(PA_1_f*PQ_b0_f + PB_0_f*PQ_a1_f) + PQ_a1_f*PQ_b0_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))*(delta_f[c0][d0]*(PQ_d1_f + QD_1_f) + delta_f[c0][d1]*(PQ_d0_f + QD_0_f) + delta_f[d0][d1]*(PQ_c0_f + QC_0_f))

                        -PQ_a0_f*(PA_1_f*(PQ_b0_f*(PQ_c0_f*(QD_0_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][d0]) + PQ_d0_f*(QC_0_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][c0]) + PQ_d1_f*(QC_0_f*delta_f[b1][d0] + QD_0_f*delta_f[b1][c0])) + PQ_b1_f*(PQ_c0_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0]) + PQ_d0_f*(QC_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][c0]) + PQ_d1_f*(QC_0_f*delta_f[b0][d0] + QD_0_f*delta_f[b0][c0]))) + PB_0_f*(PQ_a1_f*(PQ_c0_f*(QD_0_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][d0]) + PQ_d0_f*(QC_0_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][c0]) + PQ_d1_f*(QC_0_f*delta_f[b1][d0] + QD_0_f*delta_f[b1][c0])) + PQ_b1_f*(PQ_c0_f*(QD_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][d0]) + PQ_d0_f*(QC_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][c0]) + PQ_d1_f*(QC_0_f*delta_f[a1][d0] + QD_0_f*delta_f[a1][c0]))) + PB_1_f*(PQ_a1_f*(PQ_c0_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0]) + PQ_d0_f*(QC_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][c0]) + PQ_d1_f*(QC_0_f*delta_f[b0][d0] + QD_0_f*delta_f[b0][c0])) + PQ_b0_f*(PQ_c0_f*(QD_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][d0]) + PQ_d0_f*(QC_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][c0]) + PQ_d1_f*(QC_0_f*delta_f[a1][d0] + QD_0_f*delta_f[a1][c0])))) - PQ_a1_f*(PA_0_f*(PQ_b0_f*(PQ_c0_f*(QD_0_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][d0]) + PQ_d0_f*(QC_0_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][c0]) + PQ_d1_f*(QC_0_f*delta_f[b1][d0] + QD_0_f*delta_f[b1][c0])) + PQ_b1_f*(PQ_c0_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0]) + PQ_d0_f*(QC_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][c0]) + PQ_d1_f*(QC_0_f*delta_f[b0][d0] + QD_0_f*delta_f[b0][c0]))) + (PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f)*(PQ_c0_f*(QD_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][d0]) + PQ_d0_f*(QC_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][c0]) + PQ_d1_f*(QC_0_f*delta_f[a0][d0] + QD_0_f*delta_f[a0][c0]))) - PQ_b0_f*PQ_b1_f*(PA_0_f*(PQ_c0_f*(QD_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][d0]) + PQ_d0_f*(QC_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][c0]) + PQ_d1_f*(QC_0_f*delta_f[a1][d0] + QD_0_f*delta_f[a1][c0])) + PA_1_f*(PQ_c0_f*(QD_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][d0]) + PQ_d0_f*(QC_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][c0]) + PQ_d1_f*(QC_0_f*delta_f[a0][d0] + QD_0_f*delta_f[a0][c0])))

                        +(PQ_c0_f*(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f) + PQ_d0_f*PQ_d1_f*QC_0_f)*(-PA_0_f*(PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) - PA_1_f*(PQ_a0_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][b0]) - PB_0_f*(PQ_a0_f*delta_f[a1][b1] + PQ_a1_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][a1]) - PB_1_f*(PQ_a0_f*delta_f[a1][b0] + PQ_a1_f*delta_f[a0][b0] + PQ_b0_f*delta_f[a0][a1]) + PQ_a0_f*(PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) + PQ_a1_f*(PQ_b0_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][b0]) + PQ_b0_f*PQ_b1_f*delta_f[a0][a1])

                    )

                    + F7_t_f[4] * 0.5f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        +PQ_a0_f*PQ_a1_f*PQ_b0_f*PQ_b1_f*(QC_0_f*delta_f[d0][d1] + QD_0_f*delta_f[c0][d1] + QD_1_f*delta_f[c0][d0])

                        +PQ_a0_f*PQ_a1_f*(QC_0_f*(PQ_b0_f*(QD_0_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][d0]) + PQ_b1_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0])) + QD_0_f*QD_1_f*(PQ_b0_f*delta_f[b1][c0] + PQ_b1_f*delta_f[b0][c0])) + PQ_b0_f*PQ_b1_f*(QC_0_f*(PQ_a0_f*(QD_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][d0]) + PQ_a1_f*(QD_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][d0])) + QD_0_f*QD_1_f*(PQ_a0_f*delta_f[a1][c0] + PQ_a1_f*delta_f[a0][c0]))

                        +(PQ_d0_f*QC_0_f*QD_1_f + QD_0_f*(PQ_c0_f*QD_1_f + PQ_d1_f*QC_0_f))*(PQ_a0_f*(PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) + PQ_a1_f*(PQ_b0_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][b0]) + PQ_b0_f*PQ_b1_f*delta_f[a0][a1])

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        const float D_f = pd_mat_D_f[kl];
        const float contrib_f = eri_ijkl_f * D_f * 2.0f;
        ERIs[threadIdx.y] += (double)contrib_f;
    }

    __syncthreads();

    if ((threadIdx.y == 0) && (ij < dd_prim_pair_count_local))
    {
        double J_ij = 0.0f;

        for (uint32_t n = 0; n < TILE_DIM_LARGE; n++)
        {
            J_ij += ERIs[n];
        }

        mat_J[ij] += J_ij;
    }
}

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDPD7_FP32_auto_s_ast(double*         mat_J,
                       const float*    p_prim_info_f,
                       const uint32_t  p_prim_count,
                       const float*    d_prim_info_f,
                       const uint32_t  d_prim_count,
                       const float*    pd_mat_D_f,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const float*    dd_pair_data_local_f,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* pd_first_inds,
                       const uint32_t* pd_second_inds,
                       const float*    pd_pair_data_f,
                       const uint32_t  pd_prim_pair_count,
                       const float*    boys_func_table_f,
                       const float*    boys_func_ft_f,
                       const uint32_t* prec_cut_ij_tile,
                       const uint32_t* screen_cut_ij_tile)
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    //__shared__ float   delta_f[3][3];

    __shared__ float a_i_f, a_j_f, r_i_f[3], r_j_f[3], S_ij_00_f, S1_f, inv_S1_f;
    __shared__ float PA_0_f, PA_1_f, PB_0_f, PB_1_f;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];
    const uint32_t screen_cut = screen_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        //delta_f[0][0] = 1.0f; delta_f[0][1] = 0.0f; delta_f[0][2] = 0.0f;
        //delta_f[1][0] = 0.0f; delta_f[1][1] = 1.0f; delta_f[1][2] = 0.0f;
        //delta_f[2][0] = 0.0f; delta_f[2][1] = 0.0f; delta_f[2][2] = 1.0f;

        if (ij < dd_prim_pair_count_local)
        {
            i = dd_first_inds_local[ij];
            j = dd_second_inds_local[ij];

            a_i_f = d_prim_info_f[i / 6 + d_prim_count * 0];

            r_i_f[0] = d_prim_info_f[i / 6 + d_prim_count * 2];
            r_i_f[1] = d_prim_info_f[i / 6 + d_prim_count * 3];
            r_i_f[2] = d_prim_info_f[i / 6 + d_prim_count * 4];

            a_j_f = d_prim_info_f[j / 6 + d_prim_count * 0];

            r_j_f[0] = d_prim_info_f[j / 6 + d_prim_count * 2];
            r_j_f[1] = d_prim_info_f[j / 6 + d_prim_count * 3];
            r_j_f[2] = d_prim_info_f[j / 6 + d_prim_count * 4];

            S1_f = a_i_f + a_j_f;
            inv_S1_f = (float) (1.0 / (double)S1_f);

            S_ij_00_f = dd_pair_data_local_f[ij];

            a0 = d_cart_inds[i % 6][0];
            a1 = d_cart_inds[i % 6][1];
            b0 = d_cart_inds[j % 6][0];
            b1 = d_cart_inds[j % 6][1];

            PA_0_f = (a_j_f  * inv_S1_f) * (r_j_f[a0] - r_i_f[a0]);
            PA_1_f = (a_j_f  * inv_S1_f) * (r_j_f[a1] - r_i_f[a1]);
            PB_0_f = (-a_i_f * inv_S1_f) * (r_j_f[b0] - r_i_f[b0]);
            PB_1_f = (-a_i_f * inv_S1_f) * (r_j_f[b1] - r_i_f[b1]);

        }

    }

    ERIs[threadIdx.y] = 0.0;

    __syncthreads();

    for (uint32_t m = prec_cut; m < screen_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= pd_prim_pair_count))
        {
            break;
        }

        const auto k = pd_first_inds[kl];
        const auto l = pd_second_inds[kl];

        const auto a_k_f = p_prim_info_f[k / 3 + p_prim_count * 0];

        const float r_k0_f = p_prim_info_f[k / 3 + p_prim_count * 2];
        const float r_k1_f = p_prim_info_f[k / 3 + p_prim_count * 3];
        const float r_k2_f = p_prim_info_f[k / 3 + p_prim_count * 4];

        const auto a_l_f = d_prim_info_f[l / 6 + d_prim_count * 0];

        const float r_l0_f = d_prim_info_f[l / 6 + d_prim_count * 2];
        const float r_l1_f = d_prim_info_f[l / 6 + d_prim_count * 3];
        const float r_l2_f = d_prim_info_f[l / 6 + d_prim_count * 4];

        const auto S_kl_00_f = pd_pair_data_f[kl];

        const auto c0 = k % 3;
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        const float r_l_c0_f = (c0 == 0 ? r_l0_f : (c0 == 1 ? r_l1_f : r_l2_f));
        const float r_l_d0_f = (d0 == 0 ? r_l0_f : (d0 == 1 ? r_l1_f : r_l2_f));
        const float r_l_d1_f = (d1 == 0 ? r_l0_f : (d1 == 1 ? r_l1_f : r_l2_f));

        const float r_k_c0_f = (c0 == 0 ? r_k0_f : (c0 == 1 ? r_k1_f : r_k2_f));
        const float r_k_d0_f = (d0 == 0 ? r_k0_f : (d0 == 1 ? r_k1_f : r_k2_f));
        const float r_k_d1_f = (d1 == 0 ? r_k0_f : (d1 == 1 ? r_k1_f : r_k2_f));

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2_f = a_k_f + a_l_f;

        const auto inv_S2_f = 1.0f / S2_f;
        const auto inv_S4_f = 1.0f / (S1_f + S2_f);

        const float PQ0_f = (a_k_f * r_k0_f + a_l_f * r_l0_f) * inv_S2_f - (a_i_f * r_i_f[0] + a_j_f * r_j_f[0]) * inv_S1_f;
        const float PQ1_f = (a_k_f * r_k1_f + a_l_f * r_l1_f) * inv_S2_f - (a_i_f * r_i_f[1] + a_j_f * r_j_f[1]) * inv_S1_f;
        const float PQ2_f = (a_k_f * r_k2_f + a_l_f * r_l2_f) * inv_S2_f - (a_i_f * r_i_f[2] + a_j_f * r_j_f[2]) * inv_S1_f;

        const float PQ_a0_f = (a0 == 0 ? PQ0_f : (a0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_a1_f = (a1 == 0 ? PQ0_f : (a1 == 1 ? PQ1_f : PQ2_f));
        const float PQ_b0_f = (b0 == 0 ? PQ0_f : (b0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_b1_f = (b1 == 0 ? PQ0_f : (b1 == 1 ? PQ1_f : PQ2_f));
        const float PQ_c0_f = (c0 == 0 ? PQ0_f : (c0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_d0_f = (d0 == 0 ? PQ0_f : (d0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_d1_f = (d1 == 0 ? PQ0_f : (d1 == 1 ? PQ1_f : PQ2_f));

        const auto r2_PQ_f = PQ0_f * PQ0_f + PQ1_f * PQ1_f + PQ2_f * PQ2_f;

        const auto Lambda_f = sqrtf(4.0f * S1_f * S2_f * MATH_CONST_INV_PI_F * inv_S4_f);

        float F7_t_f[5];

        gpu::computeBoysFunction_f(F7_t_f, S1_f * S2_f * inv_S4_f * r2_PQ_f, 4, boys_func_table_f, boys_func_ft_f);

        const auto QC_0_f = (a_l_f * inv_S2_f) * (r_l_c0_f - r_k_c0_f);
        const auto QD_0_f = (-a_k_f * inv_S2_f) * (r_l_d0_f - r_k_d0_f);
        const auto QD_1_f = (-a_k_f * inv_S2_f) * (r_l_d1_f - r_k_d1_f);

        const float eri_ijkl_f = Lambda_f * S_ij_00_f * S_kl_00_f * (

                    + F7_t_f[4] * S1_f * S1_f * S1_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        -PQ_c0_f*PQ_d0_f*PQ_d1_f*(PA_0_f*PA_1_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PB_1_f*(PA_0_f*PQ_a1_f + PA_1_f*PQ_a0_f))

                    )

                    + F7_t_f[4] * S1_f * S1_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        +(PQ_c0_f*(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f) + PQ_d0_f*PQ_d1_f*QC_0_f)*(PA_0_f*PQ_b0_f*(PA_1_f*PQ_b1_f + PB_1_f*PQ_a1_f) + PA_1_f*PQ_a0_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PQ_a1_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))

                    )

                    + F7_t_f[4] * S1_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        -(PQ_a0_f*PQ_b1_f*(PA_1_f*PQ_b0_f + PB_0_f*PQ_a1_f) + PQ_a1_f*PQ_b0_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))*(PQ_d0_f*QC_0_f*QD_1_f + QD_0_f*(PQ_c0_f*QD_1_f + PQ_d1_f*QC_0_f))

                    )

                    + F7_t_f[4] * S2_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        +PQ_a0_f*PQ_a1_f*PQ_b0_f*PQ_b1_f*QC_0_f*QD_0_f*QD_1_f

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        const float D_f = pd_mat_D_f[kl];
        const float contrib_f = eri_ijkl_f * D_f * 2.0f;
        ERIs[threadIdx.y] += (double)contrib_f;
    }

    __syncthreads();

    if ((threadIdx.y == 0) && (ij < dd_prim_pair_count_local))
    {
        double J_ij = 0.0f;

        for (uint32_t n = 0; n < TILE_DIM_LARGE; n++)
        {
            J_ij += ERIs[n];
        }

        mat_J[ij] += J_ij;
    }
}

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDPD8_FP32_auto_s_ast(double*         mat_J,
                       const float*    p_prim_info_f,
                       const uint32_t  p_prim_count,
                       const float*    d_prim_info_f,
                       const uint32_t  d_prim_count,
                       const float*    pd_mat_D_f,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const float*    dd_pair_data_local_f,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* pd_first_inds,
                       const uint32_t* pd_second_inds,
                       const float*    pd_pair_data_f,
                       const uint32_t  pd_prim_pair_count,
                       const float*    boys_func_table_f,
                       const float*    boys_func_ft_f,
                       const uint32_t* prec_cut_ij_tile,
                       const uint32_t* screen_cut_ij_tile)
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ float   delta_f[3][3];

    __shared__ float a_i_f, a_j_f, r_i_f[3], r_j_f[3], S_ij_00_f, S1_f, inv_S1_f;
    //__shared__ float PA_0_f, PA_1_f, PB_0_f, PB_1_f;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];
    const uint32_t screen_cut = screen_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta_f[0][0] = 1.0f; delta_f[0][1] = 0.0f; delta_f[0][2] = 0.0f;
        delta_f[1][0] = 0.0f; delta_f[1][1] = 1.0f; delta_f[1][2] = 0.0f;
        delta_f[2][0] = 0.0f; delta_f[2][1] = 0.0f; delta_f[2][2] = 1.0f;

        if (ij < dd_prim_pair_count_local)
        {
            i = dd_first_inds_local[ij];
            j = dd_second_inds_local[ij];

            a_i_f = d_prim_info_f[i / 6 + d_prim_count * 0];

            r_i_f[0] = d_prim_info_f[i / 6 + d_prim_count * 2];
            r_i_f[1] = d_prim_info_f[i / 6 + d_prim_count * 3];
            r_i_f[2] = d_prim_info_f[i / 6 + d_prim_count * 4];

            a_j_f = d_prim_info_f[j / 6 + d_prim_count * 0];

            r_j_f[0] = d_prim_info_f[j / 6 + d_prim_count * 2];
            r_j_f[1] = d_prim_info_f[j / 6 + d_prim_count * 3];
            r_j_f[2] = d_prim_info_f[j / 6 + d_prim_count * 4];

            S1_f = a_i_f + a_j_f;
            inv_S1_f = (float) (1.0 / (double)S1_f);

            S_ij_00_f = dd_pair_data_local_f[ij];

            a0 = d_cart_inds[i % 6][0];
            a1 = d_cart_inds[i % 6][1];
            b0 = d_cart_inds[j % 6][0];
            b1 = d_cart_inds[j % 6][1];

            //PA_0_f = (a_j_f  * inv_S1_f) * (r_j_f[a0] - r_i_f[a0]);
            //PA_1_f = (a_j_f  * inv_S1_f) * (r_j_f[a1] - r_i_f[a1]);
            //PB_0_f = (-a_i_f * inv_S1_f) * (r_j_f[b0] - r_i_f[b0]);
            //PB_1_f = (-a_i_f * inv_S1_f) * (r_j_f[b1] - r_i_f[b1]);

        }

    }

    ERIs[threadIdx.y] = 0.0;

    __syncthreads();

    for (uint32_t m = prec_cut; m < screen_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= pd_prim_pair_count))
        {
            break;
        }

        const auto k = pd_first_inds[kl];
        const auto l = pd_second_inds[kl];

        const auto a_k_f = p_prim_info_f[k / 3 + p_prim_count * 0];

        const float r_k0_f = p_prim_info_f[k / 3 + p_prim_count * 2];
        const float r_k1_f = p_prim_info_f[k / 3 + p_prim_count * 3];
        const float r_k2_f = p_prim_info_f[k / 3 + p_prim_count * 4];
        const float r_k_f[3] = {r_k0_f, r_k1_f, r_k2_f};

        const auto a_l_f = d_prim_info_f[l / 6 + d_prim_count * 0];

        const float r_l0_f = d_prim_info_f[l / 6 + d_prim_count * 2];
        const float r_l1_f = d_prim_info_f[l / 6 + d_prim_count * 3];
        const float r_l2_f = d_prim_info_f[l / 6 + d_prim_count * 4];
        const float r_l_f[3] = {r_l0_f, r_l1_f, r_l2_f};

        const auto S_kl_00_f = pd_pair_data_f[kl];

        const auto c0 = k % 3;
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2_f = a_k_f + a_l_f;

        const auto inv_S2_f = 1.0f / S2_f;
        const auto inv_S4_f = 1.0f / (S1_f + S2_f);

        const float PQ0_f = (a_k_f * r_k0_f + a_l_f * r_l0_f) * inv_S2_f - (a_i_f * r_i_f[0] + a_j_f * r_j_f[0]) * inv_S1_f;
        const float PQ1_f = (a_k_f * r_k1_f + a_l_f * r_l1_f) * inv_S2_f - (a_i_f * r_i_f[1] + a_j_f * r_j_f[1]) * inv_S1_f;
        const float PQ2_f = (a_k_f * r_k2_f + a_l_f * r_l2_f) * inv_S2_f - (a_i_f * r_i_f[2] + a_j_f * r_j_f[2]) * inv_S1_f;

        const float PQ_a0_f = (a0 == 0 ? PQ0_f : (a0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_a1_f = (a1 == 0 ? PQ0_f : (a1 == 1 ? PQ1_f : PQ2_f));
        const float PQ_b0_f = (b0 == 0 ? PQ0_f : (b0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_b1_f = (b1 == 0 ? PQ0_f : (b1 == 1 ? PQ1_f : PQ2_f));
        const float PQ_c0_f = (c0 == 0 ? PQ0_f : (c0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_d0_f = (d0 == 0 ? PQ0_f : (d0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_d1_f = (d1 == 0 ? PQ0_f : (d1 == 1 ? PQ1_f : PQ2_f));

        const auto r2_PQ_f = PQ0_f * PQ0_f + PQ1_f * PQ1_f + PQ2_f * PQ2_f;

        const auto Lambda_f = sqrtf(4.0f * S1_f * S2_f * MATH_CONST_INV_PI_F * inv_S4_f);

        float F7_t_f[5];

        gpu::computeBoysFunction_f(F7_t_f, S1_f * S2_f * inv_S4_f * r2_PQ_f, 4, boys_func_table_f, boys_func_ft_f);

        //const auto QC_0_f = (a_l_f * inv_S2_f) * (r_l_f[c0] - r_k_f[c0]);
        //const auto QD_0_f = (-a_k_f * inv_S2_f) * (r_l_f[d0] - r_k_f[d0]);
        //const auto QD_1_f = (-a_k_f * inv_S2_f) * (r_l_f[d1] - r_k_f[d1]);

        const float eri_ijkl_f = Lambda_f * S_ij_00_f * S_kl_00_f * (

                    + F7_t_f[4] * 0.125f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        +PQ_a0_f*(delta_f[a1][b0]*(delta_f[b1][c0]*delta_f[d0][d1] + delta_f[b1][d0]*delta_f[c0][d1] + delta_f[b1][d1]*delta_f[c0][d0]) + delta_f[a1][b1]*(delta_f[b0][c0]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][d0]) + delta_f[a1][c0]*(delta_f[b0][b1]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][d0]) + delta_f[a1][d0]*(delta_f[b0][b1]*delta_f[c0][d1] + delta_f[b0][c0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][c0]) + delta_f[a1][d1]*(delta_f[b0][b1]*delta_f[c0][d0] + delta_f[b0][c0]*delta_f[b1][d0] + delta_f[b0][d0]*delta_f[b1][c0])) + PQ_a1_f*(delta_f[a0][b0]*(delta_f[b1][c0]*delta_f[d0][d1] + delta_f[b1][d0]*delta_f[c0][d1] + delta_f[b1][d1]*delta_f[c0][d0]) + delta_f[a0][b1]*(delta_f[b0][c0]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][d0]) + delta_f[a0][c0]*(delta_f[b0][b1]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][d0]) + delta_f[a0][d0]*(delta_f[b0][b1]*delta_f[c0][d1] + delta_f[b0][c0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][c0]) + delta_f[a0][d1]*(delta_f[b0][b1]*delta_f[c0][d0] + delta_f[b0][c0]*delta_f[b1][d0] + delta_f[b0][d0]*delta_f[b1][c0])) + PQ_b0_f*(delta_f[a0][a1]*(delta_f[b1][c0]*delta_f[d0][d1] + delta_f[b1][d0]*delta_f[c0][d1] + delta_f[b1][d1]*delta_f[c0][d0]) + delta_f[a0][b1]*(delta_f[a1][c0]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[c0][d1] + delta_f[a1][d1]*delta_f[c0][d0]) + delta_f[a0][c0]*(delta_f[a1][b1]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b1][d0]) + delta_f[a0][d0]*(delta_f[a1][b1]*delta_f[c0][d1] + delta_f[a1][c0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b1][c0]) + delta_f[a0][d1]*(delta_f[a1][b1]*delta_f[c0][d0] + delta_f[a1][c0]*delta_f[b1][d0] + delta_f[a1][d0]*delta_f[b1][c0])) + PQ_b1_f*(delta_f[a0][a1]*(delta_f[b0][c0]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][d0]) + delta_f[a0][b0]*(delta_f[a1][c0]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[c0][d1] + delta_f[a1][d1]*delta_f[c0][d0]) + delta_f[a0][c0]*(delta_f[a1][b0]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][d0]) + delta_f[a0][d0]*(delta_f[a1][b0]*delta_f[c0][d1] + delta_f[a1][c0]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][c0]) + delta_f[a0][d1]*(delta_f[a1][b0]*delta_f[c0][d0] + delta_f[a1][c0]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][c0])) + PQ_c0_f*(delta_f[a0][a1]*(delta_f[b0][b1]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][d0]) + delta_f[a0][b0]*(delta_f[a1][b1]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b1][d0]) + delta_f[a0][b1]*(delta_f[a1][b0]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][d0]) + delta_f[a0][d0]*(delta_f[a1][b0]*delta_f[b1][d1] + delta_f[a1][b1]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][b1]) + delta_f[a0][d1]*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1])) + PQ_d0_f*(delta_f[a0][a1]*(delta_f[b0][b1]*delta_f[c0][d1] + delta_f[b0][c0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][c0]) + delta_f[a0][b0]*(delta_f[a1][b1]*delta_f[c0][d1] + delta_f[a1][c0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b1][c0]) + delta_f[a0][b1]*(delta_f[a1][b0]*delta_f[c0][d1] + delta_f[a1][c0]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][c0]) + delta_f[a0][c0]*(delta_f[a1][b0]*delta_f[b1][d1] + delta_f[a1][b1]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][b1]) + delta_f[a0][d1]*(delta_f[a1][b0]*delta_f[b1][c0] + delta_f[a1][b1]*delta_f[b0][c0] + delta_f[a1][c0]*delta_f[b0][b1])) + PQ_d1_f*(delta_f[a0][a1]*(delta_f[b0][b1]*delta_f[c0][d0] + delta_f[b0][c0]*delta_f[b1][d0] + delta_f[b0][d0]*delta_f[b1][c0]) + delta_f[a0][b0]*(delta_f[a1][b1]*delta_f[c0][d0] + delta_f[a1][c0]*delta_f[b1][d0] + delta_f[a1][d0]*delta_f[b1][c0]) + delta_f[a0][b1]*(delta_f[a1][b0]*delta_f[c0][d0] + delta_f[a1][c0]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][c0]) + delta_f[a0][c0]*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + delta_f[a0][d0]*(delta_f[a1][b0]*delta_f[b1][c0] + delta_f[a1][b1]*delta_f[b0][c0] + delta_f[a1][c0]*delta_f[b0][b1]))

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        const float D_f = pd_mat_D_f[kl];
        const float contrib_f = eri_ijkl_f * D_f * 2.0f;
        ERIs[threadIdx.y] += (double)contrib_f;
    }

    __syncthreads();

    if ((threadIdx.y == 0) && (ij < dd_prim_pair_count_local))
    {
        double J_ij = 0.0f;

        for (uint32_t n = 0; n < TILE_DIM_LARGE; n++)
        {
            J_ij += ERIs[n];
        }

        mat_J[ij] += J_ij;
    }
}

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDPD9_FP32_auto_s_ast(double*         mat_J,
                       const float*    p_prim_info_f,
                       const uint32_t  p_prim_count,
                       const float*    d_prim_info_f,
                       const uint32_t  d_prim_count,
                       const float*    pd_mat_D_f,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const float*    dd_pair_data_local_f,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* pd_first_inds,
                       const uint32_t* pd_second_inds,
                       const float*    pd_pair_data_f,
                       const uint32_t  pd_prim_pair_count,
                       const float*    boys_func_table_f,
                       const float*    boys_func_ft_f,
                       const uint32_t* prec_cut_ij_tile,
                       const uint32_t* screen_cut_ij_tile)
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ float   delta_f[3][3];

    __shared__ float a_i_f, a_j_f, r_i_f[3], r_j_f[3], S_ij_00_f, S1_f, inv_S1_f;
    __shared__ float PA_0_f, PA_1_f, PB_0_f, PB_1_f;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];
    const uint32_t screen_cut = screen_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta_f[0][0] = 1.0f; delta_f[0][1] = 0.0f; delta_f[0][2] = 0.0f;
        delta_f[1][0] = 0.0f; delta_f[1][1] = 1.0f; delta_f[1][2] = 0.0f;
        delta_f[2][0] = 0.0f; delta_f[2][1] = 0.0f; delta_f[2][2] = 1.0f;

        if (ij < dd_prim_pair_count_local)
        {
            i = dd_first_inds_local[ij];
            j = dd_second_inds_local[ij];

            a_i_f = d_prim_info_f[i / 6 + d_prim_count * 0];

            r_i_f[0] = d_prim_info_f[i / 6 + d_prim_count * 2];
            r_i_f[1] = d_prim_info_f[i / 6 + d_prim_count * 3];
            r_i_f[2] = d_prim_info_f[i / 6 + d_prim_count * 4];

            a_j_f = d_prim_info_f[j / 6 + d_prim_count * 0];

            r_j_f[0] = d_prim_info_f[j / 6 + d_prim_count * 2];
            r_j_f[1] = d_prim_info_f[j / 6 + d_prim_count * 3];
            r_j_f[2] = d_prim_info_f[j / 6 + d_prim_count * 4];

            S1_f = a_i_f + a_j_f;
            inv_S1_f = (float) (1.0 / (double)S1_f);

            S_ij_00_f = dd_pair_data_local_f[ij];

            a0 = d_cart_inds[i % 6][0];
            a1 = d_cart_inds[i % 6][1];
            b0 = d_cart_inds[j % 6][0];
            b1 = d_cart_inds[j % 6][1];

            PA_0_f = (a_j_f  * inv_S1_f) * (r_j_f[a0] - r_i_f[a0]);
            PA_1_f = (a_j_f  * inv_S1_f) * (r_j_f[a1] - r_i_f[a1]);
            PB_0_f = (-a_i_f * inv_S1_f) * (r_j_f[b0] - r_i_f[b0]);
            PB_1_f = (-a_i_f * inv_S1_f) * (r_j_f[b1] - r_i_f[b1]);

        }

    }

    ERIs[threadIdx.y] = 0.0;

    __syncthreads();

    for (uint32_t m = prec_cut; m < screen_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= pd_prim_pair_count))
        {
            break;
        }

        const auto k = pd_first_inds[kl];
        const auto l = pd_second_inds[kl];

        const auto a_k_f = p_prim_info_f[k / 3 + p_prim_count * 0];

        const float r_k0_f = p_prim_info_f[k / 3 + p_prim_count * 2];
        const float r_k1_f = p_prim_info_f[k / 3 + p_prim_count * 3];
        const float r_k2_f = p_prim_info_f[k / 3 + p_prim_count * 4];

        const auto a_l_f = d_prim_info_f[l / 6 + d_prim_count * 0];

        const float r_l0_f = d_prim_info_f[l / 6 + d_prim_count * 2];
        const float r_l1_f = d_prim_info_f[l / 6 + d_prim_count * 3];
        const float r_l2_f = d_prim_info_f[l / 6 + d_prim_count * 4];

        const auto S_kl_00_f = pd_pair_data_f[kl];

        const auto c0 = k % 3;
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        const float r_l_c0_f = (c0 == 0 ? r_l0_f : (c0 == 1 ? r_l1_f : r_l2_f));
        const float r_l_d0_f = (d0 == 0 ? r_l0_f : (d0 == 1 ? r_l1_f : r_l2_f));
        const float r_l_d1_f = (d1 == 0 ? r_l0_f : (d1 == 1 ? r_l1_f : r_l2_f));

        const float r_k_c0_f = (c0 == 0 ? r_k0_f : (c0 == 1 ? r_k1_f : r_k2_f));
        const float r_k_d0_f = (d0 == 0 ? r_k0_f : (d0 == 1 ? r_k1_f : r_k2_f));
        const float r_k_d1_f = (d1 == 0 ? r_k0_f : (d1 == 1 ? r_k1_f : r_k2_f));

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2_f = a_k_f + a_l_f;

        const auto inv_S2_f = 1.0f / S2_f;
        const auto inv_S4_f = 1.0f / (S1_f + S2_f);

        const float PQ0_f = (a_k_f * r_k0_f + a_l_f * r_l0_f) * inv_S2_f - (a_i_f * r_i_f[0] + a_j_f * r_j_f[0]) * inv_S1_f;
        const float PQ1_f = (a_k_f * r_k1_f + a_l_f * r_l1_f) * inv_S2_f - (a_i_f * r_i_f[1] + a_j_f * r_j_f[1]) * inv_S1_f;
        const float PQ2_f = (a_k_f * r_k2_f + a_l_f * r_l2_f) * inv_S2_f - (a_i_f * r_i_f[2] + a_j_f * r_j_f[2]) * inv_S1_f;

        const float PQ_a0_f = (a0 == 0 ? PQ0_f : (a0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_a1_f = (a1 == 0 ? PQ0_f : (a1 == 1 ? PQ1_f : PQ2_f));
        const float PQ_b0_f = (b0 == 0 ? PQ0_f : (b0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_b1_f = (b1 == 0 ? PQ0_f : (b1 == 1 ? PQ1_f : PQ2_f));
        const float PQ_c0_f = (c0 == 0 ? PQ0_f : (c0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_d0_f = (d0 == 0 ? PQ0_f : (d0 == 1 ? PQ1_f : PQ2_f));
        const float PQ_d1_f = (d1 == 0 ? PQ0_f : (d1 == 1 ? PQ1_f : PQ2_f));

        const auto r2_PQ_f = PQ0_f * PQ0_f + PQ1_f * PQ1_f + PQ2_f * PQ2_f;

        const auto Lambda_f = sqrtf(4.0f * S1_f * S2_f * MATH_CONST_INV_PI_F * inv_S4_f);

        float F7_t_f[8];

        gpu::computeBoysFunction_f(F7_t_f, S1_f * S2_f * inv_S4_f * r2_PQ_f, 7, boys_func_table_f, boys_func_ft_f);

        const auto QC_0_f = (a_l_f * inv_S2_f) * (r_l_c0_f - r_k_c0_f);
        const auto QD_0_f = (-a_k_f * inv_S2_f) * (r_l_d0_f - r_k_d0_f);
        const auto QD_1_f = (-a_k_f * inv_S2_f) * (r_l_d1_f - r_k_d1_f);

        const float eri_ijkl_f = Lambda_f * S_ij_00_f * S_kl_00_f * (

                    + F7_t_f[5] * (-0.25f) * S1_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        +PQ_a0_f*(PQ_a1_f*(PQ_b0_f*(delta_f[b1][c0]*delta_f[d0][d1] + delta_f[b1][d0]*delta_f[c0][d1] + delta_f[b1][d1]*delta_f[c0][d0]) + PQ_b1_f*(delta_f[b0][c0]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][d0]) + PQ_c0_f*(delta_f[b0][b1]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][d0]) + PQ_d0_f*(delta_f[b0][b1]*delta_f[c0][d1] + delta_f[b0][c0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][c0]) + PQ_d1_f*(delta_f[b0][b1]*delta_f[c0][d0] + delta_f[b0][c0]*delta_f[b1][d0] + delta_f[b0][d0]*delta_f[b1][c0])) + PQ_d1_f*(PQ_b0_f*(delta_f[a1][b1]*delta_f[c0][d0] + delta_f[a1][c0]*delta_f[b1][d0] + delta_f[a1][d0]*delta_f[b1][c0]) + PQ_b1_f*(delta_f[a1][b0]*delta_f[c0][d0] + delta_f[a1][c0]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][c0]) + PQ_c0_f*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + PQ_d0_f*(delta_f[a1][b0]*delta_f[b1][c0] + delta_f[a1][b1]*delta_f[b0][c0] + delta_f[a1][c0]*delta_f[b0][b1]))) + PQ_a1_f*PQ_d1_f*(PQ_b0_f*(delta_f[a0][b1]*delta_f[c0][d0] + delta_f[a0][c0]*delta_f[b1][d0] + delta_f[a0][d0]*delta_f[b1][c0]) + PQ_b1_f*(delta_f[a0][b0]*delta_f[c0][d0] + delta_f[a0][c0]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][c0]) + PQ_c0_f*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + PQ_d0_f*(delta_f[a0][b0]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[b0][c0] + delta_f[a0][c0]*delta_f[b0][b1])) + PQ_b0_f*(PQ_b1_f*(PQ_a0_f*(delta_f[a1][c0]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[c0][d1] + delta_f[a1][d1]*delta_f[c0][d0]) + PQ_a1_f*(delta_f[a0][c0]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[c0][d0]) + PQ_c0_f*(delta_f[a0][a1]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][d0]) + PQ_d0_f*(delta_f[a0][a1]*delta_f[c0][d1] + delta_f[a0][c0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][c0]) + PQ_d1_f*(delta_f[a0][a1]*delta_f[c0][d0] + delta_f[a0][c0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][c0])) + PQ_c0_f*(PQ_a0_f*(delta_f[a1][b1]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b1][d0]) + PQ_a1_f*(delta_f[a0][b1]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[b1][d1] + delta_f[a0][d1]*delta_f[b1][d0]) + PQ_d1_f*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1])) + PQ_d0_f*(PQ_a0_f*(delta_f[a1][b1]*delta_f[c0][d1] + delta_f[a1][c0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b1][c0]) + PQ_a1_f*(delta_f[a0][b1]*delta_f[c0][d1] + delta_f[a0][c0]*delta_f[b1][d1] + delta_f[a0][d1]*delta_f[b1][c0]) + PQ_d1_f*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1]))) + PQ_b1_f*(PQ_c0_f*(PQ_a0_f*(delta_f[a1][b0]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][d0]) + PQ_a1_f*(delta_f[a0][b0]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][d0]) + PQ_d1_f*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0])) + PQ_d0_f*(PQ_a0_f*(delta_f[a1][b0]*delta_f[c0][d1] + delta_f[a1][c0]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][c0]) + PQ_a1_f*(delta_f[a0][b0]*delta_f[c0][d1] + delta_f[a0][c0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][c0]) + PQ_d1_f*(delta_f[a0][a1]*delta_f[b0][c0] + delta_f[a0][b0]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b0]))) + PQ_c0_f*PQ_d0_f*(PQ_a0_f*(delta_f[a1][b0]*delta_f[b1][d1] + delta_f[a1][b1]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][b1]) + PQ_a1_f*(delta_f[a0][b0]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][b1]) + PQ_b0_f*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + PQ_b1_f*(delta_f[a0][a1]*delta_f[b0][d1] + delta_f[a0][b0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b0]) + PQ_d1_f*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0]))

                    )

                    + F7_t_f[5] * 0.5f * S1_f * S1_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        +(PQ_a0_f*PQ_b1_f*(PA_1_f*PQ_b0_f + PB_0_f*PQ_a1_f) + PQ_a1_f*PQ_b0_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))*(PQ_c0_f*delta_f[d0][d1] + PQ_d0_f*delta_f[c0][d1] + PQ_d1_f*delta_f[c0][d0])

                        +PQ_c0_f*(PQ_a0_f*(PA_1_f*(PQ_b0_f*(PQ_d0_f*delta_f[b1][d1] + PQ_d1_f*delta_f[b1][d0]) + PQ_b1_f*(PQ_d0_f*delta_f[b0][d1] + PQ_d1_f*delta_f[b0][d0])) + PB_0_f*(PQ_a1_f*(PQ_d0_f*delta_f[b1][d1] + PQ_d1_f*delta_f[b1][d0]) + PQ_b1_f*(PQ_d0_f*delta_f[a1][d1] + PQ_d1_f*delta_f[a1][d0])) + PB_1_f*(PQ_a1_f*(PQ_d0_f*delta_f[b0][d1] + PQ_d1_f*delta_f[b0][d0]) + PQ_b0_f*(PQ_d0_f*delta_f[a1][d1] + PQ_d1_f*delta_f[a1][d0]))) + PQ_a1_f*(PA_0_f*(PQ_b0_f*(PQ_d0_f*delta_f[b1][d1] + PQ_d1_f*delta_f[b1][d0]) + PQ_b1_f*(PQ_d0_f*delta_f[b0][d1] + PQ_d1_f*delta_f[b0][d0])) + (PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f)*(PQ_d0_f*delta_f[a0][d1] + PQ_d1_f*delta_f[a0][d0])) + PQ_b0_f*PQ_b1_f*(PA_0_f*(PQ_d0_f*delta_f[a1][d1] + PQ_d1_f*delta_f[a1][d0]) + PA_1_f*(PQ_d0_f*delta_f[a0][d1] + PQ_d1_f*delta_f[a0][d0]))) + PQ_d0_f*PQ_d1_f*(PQ_a0_f*(PA_1_f*(PQ_b0_f*delta_f[b1][c0] + PQ_b1_f*delta_f[b0][c0]) + PB_0_f*(PQ_a1_f*delta_f[b1][c0] + PQ_b1_f*delta_f[a1][c0]) + PB_1_f*(PQ_a1_f*delta_f[b0][c0] + PQ_b0_f*delta_f[a1][c0])) + PQ_a1_f*(PA_0_f*(PQ_b0_f*delta_f[b1][c0] + PQ_b1_f*delta_f[b0][c0]) + delta_f[a0][c0]*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f)) + PQ_b0_f*PQ_b1_f*(PA_0_f*delta_f[a1][c0] + PA_1_f*delta_f[a0][c0]))

                        +PQ_c0_f*PQ_d0_f*PQ_d1_f*(PA_0_f*(PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) + PQ_a0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0] - PQ_a1_f*delta_f[b0][b1] - PQ_b0_f*delta_f[a1][b1] - PQ_b1_f*delta_f[a1][b0]) + PQ_a1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PQ_b0_f*(PB_1_f*delta_f[a0][a1] + delta_f[a0][b1]*(PA_1_f - PQ_a1_f)) + PQ_b1_f*(delta_f[a0][a1]*(PB_0_f - PQ_b0_f) + delta_f[a0][b0]*(PA_1_f - PQ_a1_f)))

                    )

                    + F7_t_f[5] * (-0.5f) * S1_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        +PQ_a0_f*PQ_a1_f*PQ_b0_f*PQ_b1_f*(delta_f[c0][d0]*(PQ_d1_f + QD_1_f) + delta_f[c0][d1]*(PQ_d0_f + QD_0_f) + delta_f[d0][d1]*(PQ_c0_f + QC_0_f))

                        +PQ_a0_f*PQ_a1_f*(PQ_b0_f*(PQ_c0_f*(QD_0_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][d0]) + PQ_d0_f*(QC_0_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][c0]) + PQ_d1_f*(QC_0_f*delta_f[b1][d0] + QD_0_f*delta_f[b1][c0])) + PQ_b1_f*(PQ_c0_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0]) + PQ_d0_f*(QC_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][c0]) + PQ_d1_f*(QC_0_f*delta_f[b0][d0] + QD_0_f*delta_f[b0][c0]))) + PQ_b0_f*PQ_b1_f*(PQ_a0_f*(PQ_c0_f*(QD_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][d0]) + PQ_d0_f*(QC_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][c0]) + PQ_d1_f*(QC_0_f*delta_f[a1][d0] + QD_0_f*delta_f[a1][c0])) + PQ_a1_f*(PQ_c0_f*(QD_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][d0]) + PQ_d0_f*(QC_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][c0]) + PQ_d1_f*(QC_0_f*delta_f[a0][d0] + QD_0_f*delta_f[a0][c0])))

                        +(PQ_c0_f*(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f) + PQ_d0_f*PQ_d1_f*QC_0_f)*(PQ_a0_f*(PQ_a1_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a1][b1] + PQ_b1_f*delta_f[a1][b0]) + PQ_a1_f*(PQ_b0_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][b0]) + PQ_b0_f*PQ_b1_f*delta_f[a0][a1])

                    )

                    + F7_t_f[5] * (-1.0f) * S1_f * S1_f * S1_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        +PQ_c0_f*PQ_d0_f*PQ_d1_f*(PA_0_f*PQ_b0_f*(PA_1_f*PQ_b1_f + PB_1_f*PQ_a1_f) + PA_1_f*PQ_a0_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PQ_a1_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))

                    )

                    + F7_t_f[5] * S1_f * S1_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        +(PQ_c0_f*(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f) + PQ_d0_f*PQ_d1_f*QC_0_f)*(PQ_a0_f*PQ_b1_f*(PA_1_f*PQ_b0_f + PB_0_f*PQ_a1_f) + PQ_a1_f*PQ_b0_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))

                    )

                    + F7_t_f[5] * S1_f * S2_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        +PQ_a0_f*PQ_a1_f*PQ_b0_f*PQ_b1_f*(-PQ_d0_f*QC_0_f*QD_1_f - QD_0_f*(PQ_c0_f*QD_1_f + PQ_d1_f*QC_0_f))

                    )

                    + F7_t_f[6] * (-1.0f) * S1_f * S1_f * S1_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        +PQ_c0_f*PQ_d0_f*PQ_d1_f*(PQ_a0_f*PQ_b1_f*(PA_1_f*PQ_b0_f + PB_0_f*PQ_a1_f) + PQ_a1_f*PQ_b0_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))

                    )

                    + F7_t_f[6] * S1_f * S1_f * S2_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        +PQ_a0_f*PQ_a1_f*PQ_b0_f*PQ_b1_f*(PQ_c0_f*(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f) + PQ_d0_f*PQ_d1_f*QC_0_f)

                    )

                    + F7_t_f[6] * 0.5f * S1_f * S1_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        +PQ_a0_f*PQ_d0_f*PQ_d1_f*(PQ_a1_f*(PQ_b0_f*delta_f[b1][c0] + PQ_b1_f*delta_f[b0][c0] + PQ_c0_f*delta_f[b0][b1]) + PQ_b0_f*(PQ_b1_f*delta_f[a1][c0] + PQ_c0_f*delta_f[a1][b1])) + PQ_a1_f*PQ_b0_f*(PQ_a0_f*(PQ_b1_f*(PQ_c0_f*delta_f[d0][d1] + PQ_d0_f*delta_f[c0][d1] + PQ_d1_f*delta_f[c0][d0]) + PQ_c0_f*(PQ_d0_f*delta_f[b1][d1] + PQ_d1_f*delta_f[b1][d0])) + PQ_b1_f*PQ_c0_f*(PQ_d0_f*delta_f[a0][d1] + PQ_d1_f*delta_f[a0][d0]) + PQ_d0_f*PQ_d1_f*(PQ_b1_f*delta_f[a0][c0] + PQ_c0_f*delta_f[a0][b1])) + PQ_b1_f*PQ_c0_f*(PQ_a0_f*(PQ_a1_f*(PQ_d0_f*delta_f[b0][d1] + PQ_d1_f*delta_f[b0][d0]) + PQ_b0_f*(PQ_d0_f*delta_f[a1][d1] + PQ_d1_f*delta_f[a1][d0])) + PQ_d0_f*PQ_d1_f*(PQ_a0_f*delta_f[a1][b0] + PQ_a1_f*delta_f[a0][b0] + PQ_b0_f*delta_f[a0][a1]))

                    )

                    + F7_t_f[7] * (-1.0f) * S1_f * S1_f * S1_f * S2_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                        +PQ_a0_f*PQ_a1_f*PQ_b0_f*PQ_b1_f*PQ_c0_f*PQ_d0_f*PQ_d1_f

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        const float D_f = pd_mat_D_f[kl];
        const float contrib_f = eri_ijkl_f * D_f * 2.0f;
        ERIs[threadIdx.y] += (double)contrib_f;
    }

    __syncthreads();

    if ((threadIdx.y == 0) && (ij < dd_prim_pair_count_local))
    {
        double J_ij = 0.0f;

        for (uint32_t n = 0; n < TILE_DIM_LARGE; n++)
        {
            J_ij += ERIs[n];
        }

        mat_J[ij] += J_ij;
    }
}
