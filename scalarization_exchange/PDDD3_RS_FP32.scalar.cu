computeExchangeFockPDDD3_RS_FP32(double*         mat_K,
                        const uint32_t* pair_inds_i_for_K_pd,
                        const uint32_t* pair_inds_k_for_K_pd,
                        const uint32_t  pair_inds_count_for_K_pd,
                        const float*    p_prim_info_f,
                        const uint32_t* p_prim_aoinds,
                        const uint32_t  p_prim_count,
                        const float*    d_prim_info_f,
                        const uint32_t* d_prim_aoinds,
                        const uint32_t  d_prim_count,
                        const double*   mat_D_full_AO,
                        const uint32_t  naos,
                        const uint32_t* D_inds_K_pd,
                        const uint32_t* D_inds_K_dd,
                        const uint32_t* pair_displs_K_pd,
                        const uint32_t* pair_displs_K_dd,
                        const uint32_t* pair_counts_K_pd,
                        const uint32_t* pair_counts_K_dd,
                        const float*    pair_data_K_pd_f,
                        const float*    pair_data_K_dd_f,
                        const float*    boys_func_table_f,
                        const float*    boys_func_ft_f,
                        const double    omega,
                       const uint32_t* prec_cut_flat,
                       const uint32_t* screen_cut_flat,
                       const uint32_t* displ_cuts)
{
    // each thread block scans over [i?|k?] and sum up to a primitive K matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_Y_K][TILE_DIM_X_K + 1];
    __shared__ uint32_t i, k, count_i, count_k, displ_i, displ_k;
    __shared__ float    a_i_f, r_i_f[3], a_k_f, r_k_f[3];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ float    delta_f[3][3];

    const uint32_t ik = blockIdx.x;

    // we make sure that ik < pair_inds_count_for_K_pd when calling the kernel

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

        i = pair_inds_i_for_K_pd[ik];
        k = pair_inds_k_for_K_pd[ik];

        count_i = pair_counts_K_pd[i];
        count_k = pair_counts_K_dd[k];

        displ_i = pair_displs_K_pd[i];
        displ_k = pair_displs_K_dd[k];

        a_i_f = p_prim_info_f[i / 3 + p_prim_count * 0];

        r_i_f[0] = p_prim_info_f[i / 3 + p_prim_count * 2];
        r_i_f[1] = p_prim_info_f[i / 3 + p_prim_count * 3];
        r_i_f[2] = p_prim_info_f[i / 3 + p_prim_count * 4];

        a_k_f = d_prim_info_f[k / 6 + d_prim_count * 0];

        r_k_f[0] = d_prim_info_f[k / 6 + d_prim_count * 2];
        r_k_f[1] = d_prim_info_f[k / 6 + d_prim_count * 3];
        r_k_f[2] = d_prim_info_f[k / 6 + d_prim_count * 4];
    }

    ERIs[threadIdx.y][threadIdx.x] = 0.0;

    __syncthreads();

    for (uint32_t m = 0; m < (count_i + TILE_DIM_Y_K - 1) / TILE_DIM_Y_K; m++)
    {
        const uint32_t j = m * TILE_DIM_Y_K + threadIdx.y;

        // sync threads before starting a new scan
        __syncthreads();

        float a_j_f, r_j_f[3], S_ij_00_f, S1_f, inv_S1_f;
        uint32_t j_prim, j_cgto;

        if (j < count_i)
        {

            j_prim = D_inds_K_pd[displ_i + j];

            j_cgto = d_prim_aoinds[(j_prim / 6) + d_prim_count * (j_prim % 6)];

            a_j_f = d_prim_info_f[j_prim / 6 + d_prim_count * 0];

            r_j_f[0] = d_prim_info_f[j_prim / 6 + d_prim_count * 2];
            r_j_f[1] = d_prim_info_f[j_prim / 6 + d_prim_count * 3];
            r_j_f[2] = d_prim_info_f[j_prim / 6 + d_prim_count * 4];

            S1_f = a_i_f + a_j_f;
            inv_S1_f = 1.0f / S1_f;

            S_ij_00_f = pair_data_K_pd_f[displ_i + j];
        }

            const uint32_t prec_cut_m   = prec_cut_flat  [displ_cuts[ik] + m];
            const uint32_t screen_cut_m = screen_cut_flat[displ_cuts[ik] + m];

            for (uint32_t n = prec_cut_m; n < screen_cut_m; n++)
        {
            const uint32_t l = n * TILE_DIM_X_K + threadIdx.x;

            if (j >= count_i) break;
            if (l >= count_k) continue;

            const auto l_prim = D_inds_K_dd[displ_k + l];

            const auto l_cgto = d_prim_aoinds[(l_prim / 6) + d_prim_count * (l_prim % 6)];

            const auto a_l_f = d_prim_info_f[l_prim / 6 + d_prim_count * 0];

            const float r_l0_f = d_prim_info_f[l_prim / 6 + d_prim_count * 2];
            const float r_l1_f = d_prim_info_f[l_prim / 6 + d_prim_count * 3];
            const float r_l2_f = d_prim_info_f[l_prim / 6 + d_prim_count * 4];

            const auto S_kl_00_f = pair_data_K_dd_f[displ_k + l];

            const auto a0 = i % 3;
            const auto b0 = d_cart_inds[j_prim % 6][0];
            const auto b1 = d_cart_inds[j_prim % 6][1];
            const auto c0 = d_cart_inds[k % 6][0];
            const auto c1 = d_cart_inds[k % 6][1];
            const auto d0 = d_cart_inds[l_prim % 6][0];
            const auto d1 = d_cart_inds[l_prim % 6][1];

            const float r_l_c0_f = (c0 == 0 ? r_l0_f : (c0 == 1 ? r_l1_f : r_l2_f));
            const float r_l_c1_f = (c1 == 0 ? r_l0_f : (c1 == 1 ? r_l1_f : r_l2_f));
            const float r_l_d0_f = (d0 == 0 ? r_l0_f : (d0 == 1 ? r_l1_f : r_l2_f));
            const float r_l_d1_f = (d1 == 0 ? r_l0_f : (d1 == 1 ? r_l1_f : r_l2_f));

            // J. Chem. Phys. 84, 3963-3974 (1986)

            const auto S2_f = a_k_f + a_l_f;

            const auto inv_S2_f = 1.0f / S2_f;
            const auto inv_S4_f = 1.0f / (S1_f + S2_f);

            const float PQ0_f = (a_k_f * r_k_f[0] + a_l_f * r_l0_f) * inv_S2_f - (a_i_f * r_i_f[0] + a_j_f * r_j_f[0]) * inv_S1_f;
            const float PQ1_f = (a_k_f * r_k_f[1] + a_l_f * r_l1_f) * inv_S2_f - (a_i_f * r_i_f[1] + a_j_f * r_j_f[1]) * inv_S1_f;
            const float PQ2_f = (a_k_f * r_k_f[2] + a_l_f * r_l2_f) * inv_S2_f - (a_i_f * r_i_f[2] + a_j_f * r_j_f[2]) * inv_S1_f;

            const float PQ_a0_f = (a0 == 0 ? PQ0_f : (a0 == 1 ? PQ1_f : PQ2_f));
            const float PQ_b0_f = (b0 == 0 ? PQ0_f : (b0 == 1 ? PQ1_f : PQ2_f));
            const float PQ_b1_f = (b1 == 0 ? PQ0_f : (b1 == 1 ? PQ1_f : PQ2_f));
            const float PQ_c0_f = (c0 == 0 ? PQ0_f : (c0 == 1 ? PQ1_f : PQ2_f));
            const float PQ_c1_f = (c1 == 0 ? PQ0_f : (c1 == 1 ? PQ1_f : PQ2_f));
            const float PQ_d0_f = (d0 == 0 ? PQ0_f : (d0 == 1 ? PQ1_f : PQ2_f));
            const float PQ_d1_f = (d1 == 0 ? PQ0_f : (d1 == 1 ? PQ1_f : PQ2_f));

            const auto r2_PQ_f = PQ0_f * PQ0_f + PQ1_f * PQ1_f + PQ2_f * PQ2_f;

            const auto rho_f = S1_f * S2_f * inv_S4_f;

            float d2_f = 1.0f;

            if (omega != 0.0) d2_f = (float)(omega * omega / ((double)rho_f + omega * omega));

            const auto Lambda_f = sqrtf(4.0f * rho_f * d2_f * MATH_CONST_INV_PI_F);

            float F7_t_f[5];

            gpu::computeBoysFunction_f(F7_t_f, rho_f * d2_f * r2_PQ_f, 4, boys_func_table_f, boys_func_ft_f);

            if (omega != 0.0)
            {
                F7_t_f[1] *= d2_f;
                F7_t_f[2] *= d2_f * d2_f;
                F7_t_f[3] *= d2_f * d2_f * d2_f;
                F7_t_f[4] *= d2_f * d2_f * d2_f * d2_f;
            }

            const auto PA_0_f = (a_j_f  * inv_S1_f) * (r_j_f[a0] - r_i_f[a0]);
            const auto PB_0_f = (-a_i_f * inv_S1_f) * (r_j_f[b0] - r_i_f[b0]);
            const auto PB_1_f = (-a_i_f * inv_S1_f) * (r_j_f[b1] - r_i_f[b1]);
            const auto QC_0_f = (a_l_f * inv_S2_f) * (r_l_c0_f - r_k_f[c0]);
            const auto QC_1_f = (a_l_f * inv_S2_f) * (r_l_c1_f - r_k_f[c1]);
            const auto QD_0_f = (-a_k_f * inv_S2_f) * (r_l_d0_f - r_k_f[d0]);
            const auto QD_1_f = (-a_k_f * inv_S2_f) * (r_l_d1_f - r_k_f[d1]);

            const float eri_ijkl_f = Lambda_f * S_ij_00_f * S_kl_00_f * (
                        + F7_t_f[3] * S1_f * S1_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                            +(PA_0_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PB_1_f*PQ_a0_f)*(PQ_c0_f*QD_0_f*(PQ_c1_f*QD_1_f + PQ_d1_f*QC_1_f) + PQ_c1_f*QC_0_f*(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f) + PQ_d0_f*QC_1_f*(PQ_c0_f*QD_1_f + PQ_d1_f*QC_0_f))

                        )

                        + F7_t_f[3] * S1_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                            -(PB_0_f*PQ_a0_f*PQ_b1_f + PQ_b0_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))*(QC_0_f*QD_1_f*(PQ_c1_f*QD_0_f + PQ_d0_f*QC_1_f) + QC_1_f*QD_0_f*(PQ_c0_f*QD_1_f + PQ_d1_f*QC_0_f))

                        )

                        + F7_t_f[3] * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                            +PQ_a0_f*PQ_b0_f*PQ_b1_f*QC_0_f*QC_1_f*QD_0_f*QD_1_f

                        )

                        + F7_t_f[4] * (-0.125f) * S1_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                            +PQ_a0_f*(delta_f[b0][b1]*(delta_f[c0][c1]*delta_f[d0][d1] + delta_f[c0][d0]*delta_f[c1][d1] + delta_f[c0][d1]*delta_f[c1][d0]) + delta_f[b0][c0]*(delta_f[b1][c1]*delta_f[d0][d1] + delta_f[b1][d0]*delta_f[c1][d1] + delta_f[b1][d1]*delta_f[c1][d0]) + delta_f[b0][c1]*(delta_f[b1][c0]*delta_f[d0][d1] + delta_f[b1][d0]*delta_f[c0][d1] + delta_f[b1][d1]*delta_f[c0][d0]) + delta_f[b0][d0]*(delta_f[b1][c0]*delta_f[c1][d1] + delta_f[b1][c1]*delta_f[c0][d1] + delta_f[b1][d1]*delta_f[c0][c1]) + delta_f[b0][d1]*(delta_f[b1][c0]*delta_f[c1][d0] + delta_f[b1][c1]*delta_f[c0][d0] + delta_f[b1][d0]*delta_f[c0][c1])) + PQ_b0_f*(delta_f[a0][b1]*(delta_f[c0][c1]*delta_f[d0][d1] + delta_f[c0][d0]*delta_f[c1][d1] + delta_f[c0][d1]*delta_f[c1][d0]) + delta_f[a0][c0]*(delta_f[b1][c1]*delta_f[d0][d1] + delta_f[b1][d0]*delta_f[c1][d1] + delta_f[b1][d1]*delta_f[c1][d0]) + delta_f[a0][c1]*(delta_f[b1][c0]*delta_f[d0][d1] + delta_f[b1][d0]*delta_f[c0][d1] + delta_f[b1][d1]*delta_f[c0][d0]) + delta_f[a0][d0]*(delta_f[b1][c0]*delta_f[c1][d1] + delta_f[b1][c1]*delta_f[c0][d1] + delta_f[b1][d1]*delta_f[c0][c1]) + delta_f[a0][d1]*(delta_f[b1][c0]*delta_f[c1][d0] + delta_f[b1][c1]*delta_f[c0][d0] + delta_f[b1][d0]*delta_f[c0][c1])) + PQ_b1_f*(delta_f[a0][b0]*(delta_f[c0][c1]*delta_f[d0][d1] + delta_f[c0][d0]*delta_f[c1][d1] + delta_f[c0][d1]*delta_f[c1][d0]) + delta_f[a0][c0]*(delta_f[b0][c1]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c1][d1] + delta_f[b0][d1]*delta_f[c1][d0]) + delta_f[a0][c1]*(delta_f[b0][c0]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][d0]) + delta_f[a0][d0]*(delta_f[b0][c0]*delta_f[c1][d1] + delta_f[b0][c1]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][c1]) + delta_f[a0][d1]*(delta_f[b0][c0]*delta_f[c1][d0] + delta_f[b0][c1]*delta_f[c0][d0] + delta_f[b0][d0]*delta_f[c0][c1])) + PQ_c0_f*(delta_f[a0][b0]*(delta_f[b1][c1]*delta_f[d0][d1] + delta_f[b1][d0]*delta_f[c1][d1] + delta_f[b1][d1]*delta_f[c1][d0]) + delta_f[a0][b1]*(delta_f[b0][c1]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c1][d1] + delta_f[b0][d1]*delta_f[c1][d0]) + delta_f[a0][c1]*(delta_f[b0][b1]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][d0]) + delta_f[a0][d0]*(delta_f[b0][b1]*delta_f[c1][d1] + delta_f[b0][c1]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][c1]) + delta_f[a0][d1]*(delta_f[b0][b1]*delta_f[c1][d0] + delta_f[b0][c1]*delta_f[b1][d0] + delta_f[b0][d0]*delta_f[b1][c1])) + PQ_c1_f*(delta_f[a0][b0]*(delta_f[b1][c0]*delta_f[d0][d1] + delta_f[b1][d0]*delta_f[c0][d1] + delta_f[b1][d1]*delta_f[c0][d0]) + delta_f[a0][b1]*(delta_f[b0][c0]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][d0]) + delta_f[a0][c0]*(delta_f[b0][b1]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][d0]) + delta_f[a0][d0]*(delta_f[b0][b1]*delta_f[c0][d1] + delta_f[b0][c0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][c0]) + delta_f[a0][d1]*(delta_f[b0][b1]*delta_f[c0][d0] + delta_f[b0][c0]*delta_f[b1][d0] + delta_f[b0][d0]*delta_f[b1][c0])) + PQ_d0_f*(delta_f[a0][b0]*(delta_f[b1][c0]*delta_f[c1][d1] + delta_f[b1][c1]*delta_f[c0][d1] + delta_f[b1][d1]*delta_f[c0][c1]) + delta_f[a0][b1]*(delta_f[b0][c0]*delta_f[c1][d1] + delta_f[b0][c1]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][c1]) + delta_f[a0][c0]*(delta_f[b0][b1]*delta_f[c1][d1] + delta_f[b0][c1]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][c1]) + delta_f[a0][c1]*(delta_f[b0][b1]*delta_f[c0][d1] + delta_f[b0][c0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][c0]) + delta_f[a0][d1]*(delta_f[b0][b1]*delta_f[c0][c1] + delta_f[b0][c0]*delta_f[b1][c1] + delta_f[b0][c1]*delta_f[b1][c0])) + PQ_d1_f*(delta_f[a0][b0]*(delta_f[b1][c0]*delta_f[c1][d0] + delta_f[b1][c1]*delta_f[c0][d0] + delta_f[b1][d0]*delta_f[c0][c1]) + delta_f[a0][b1]*(delta_f[b0][c0]*delta_f[c1][d0] + delta_f[b0][c1]*delta_f[c0][d0] + delta_f[b0][d0]*delta_f[c0][c1]) + delta_f[a0][c0]*(delta_f[b0][b1]*delta_f[c1][d0] + delta_f[b0][c1]*delta_f[b1][d0] + delta_f[b0][d0]*delta_f[b1][c1]) + delta_f[a0][c1]*(delta_f[b0][b1]*delta_f[c0][d0] + delta_f[b0][c0]*delta_f[b1][d0] + delta_f[b0][d0]*delta_f[b1][c0]) + delta_f[a0][d0]*(delta_f[b0][b1]*delta_f[c0][c1] + delta_f[b0][c0]*delta_f[b1][c1] + delta_f[b0][c1]*delta_f[b1][c0]))

                        )

                        + F7_t_f[4] * 0.25f * S1_f * S1_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                            +(PB_0_f*PQ_a0_f*PQ_b1_f + PQ_b0_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))*(delta_f[c0][c1]*delta_f[d0][d1] + delta_f[c0][d0]*delta_f[c1][d1] + delta_f[c0][d1]*delta_f[c1][d0])

                            +(PA_0_f*PQ_b0_f*PQ_c0_f + PB_0_f*PQ_a0_f*PQ_c0_f)*(delta_f[b1][c1]*delta_f[d0][d1] + delta_f[b1][d0]*delta_f[c1][d1] + delta_f[b1][d1]*delta_f[c1][d0]) + (PA_0_f*PQ_b0_f*PQ_c1_f + PB_0_f*PQ_a0_f*PQ_c1_f)*(delta_f[b1][c0]*delta_f[d0][d1] + delta_f[b1][d0]*delta_f[c0][d1] + delta_f[b1][d1]*delta_f[c0][d0]) + (PA_0_f*PQ_b0_f*PQ_d0_f + PB_0_f*PQ_a0_f*PQ_d0_f)*(delta_f[b1][c0]*delta_f[c1][d1] + delta_f[b1][c1]*delta_f[c0][d1] + delta_f[b1][d1]*delta_f[c0][c1]) + (PA_0_f*PQ_b0_f*PQ_d1_f + PB_0_f*PQ_a0_f*PQ_d1_f)*(delta_f[b1][c0]*delta_f[c1][d0] + delta_f[b1][c1]*delta_f[c0][d0] + delta_f[b1][d0]*delta_f[c0][c1]) + (PA_0_f*PQ_b1_f*PQ_c0_f + PB_1_f*PQ_a0_f*PQ_c0_f)*(delta_f[b0][c1]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c1][d1] + delta_f[b0][d1]*delta_f[c1][d0]) + (PA_0_f*PQ_b1_f*PQ_c1_f + PB_1_f*PQ_a0_f*PQ_c1_f)*(delta_f[b0][c0]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][d0]) + (PA_0_f*PQ_b1_f*PQ_d0_f + PB_1_f*PQ_a0_f*PQ_d0_f)*(delta_f[b0][c0]*delta_f[c1][d1] + delta_f[b0][c1]*delta_f[c0][d1] + delta_f[b0][d1]*delta_f[c0][c1]) + (PA_0_f*PQ_b1_f*PQ_d1_f + PB_1_f*PQ_a0_f*PQ_d1_f)*(delta_f[b0][c0]*delta_f[c1][d0] + delta_f[b0][c1]*delta_f[c0][d0] + delta_f[b0][d0]*delta_f[c0][c1]) + (PB_0_f*PQ_b1_f*PQ_c0_f + PB_1_f*PQ_b0_f*PQ_c0_f)*(delta_f[a0][c1]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[c1][d1] + delta_f[a0][d1]*delta_f[c1][d0]) + (PB_0_f*PQ_b1_f*PQ_c1_f + PB_1_f*PQ_b0_f*PQ_c1_f)*(delta_f[a0][c0]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[c0][d0]) + (PB_0_f*PQ_b1_f*PQ_d0_f + PB_1_f*PQ_b0_f*PQ_d0_f)*(delta_f[a0][c0]*delta_f[c1][d1] + delta_f[a0][c1]*delta_f[c0][d1] + delta_f[a0][d1]*delta_f[c0][c1]) + (PB_0_f*PQ_b1_f*PQ_d1_f + PB_1_f*PQ_b0_f*PQ_d1_f)*(delta_f[a0][c0]*delta_f[c1][d0] + delta_f[a0][c1]*delta_f[c0][d0] + delta_f[a0][d0]*delta_f[c0][c1])

                            +(PQ_c0_f*(PQ_c1_f*delta_f[d0][d1] + PQ_d0_f*delta_f[c1][d1] + PQ_d1_f*delta_f[c1][d0]) + PQ_c1_f*(PQ_d0_f*delta_f[c0][d1] + PQ_d1_f*delta_f[c0][d0]) + PQ_d0_f*PQ_d1_f*delta_f[c0][c1])*(PA_0_f*delta_f[b0][b1] + PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0] - PQ_a0_f*delta_f[b0][b1] - PQ_b0_f*delta_f[a0][b1] - PQ_b1_f*delta_f[a0][b0])

                            +PQ_c0_f*(PA_0_f*(PQ_c1_f*(delta_f[b0][d0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][d0]) + PQ_d0_f*(delta_f[b0][c1]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][c1]) + PQ_d1_f*(delta_f[b0][c1]*delta_f[b1][d0] + delta_f[b0][d0]*delta_f[b1][c1])) + PB_0_f*(PQ_c1_f*(delta_f[a0][d0]*delta_f[b1][d1] + delta_f[a0][d1]*delta_f[b1][d0]) + PQ_d0_f*(delta_f[a0][c1]*delta_f[b1][d1] + delta_f[a0][d1]*delta_f[b1][c1]) + PQ_d1_f*(delta_f[a0][c1]*delta_f[b1][d0] + delta_f[a0][d0]*delta_f[b1][c1])) + PB_1_f*(PQ_c1_f*(delta_f[a0][d0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][d0]) + PQ_d0_f*(delta_f[a0][c1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][c1]) + PQ_d1_f*(delta_f[a0][c1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][c1]))) + PQ_c1_f*(PQ_d0_f*(PA_0_f*(delta_f[b0][c0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][c0]) + PB_0_f*(delta_f[a0][c0]*delta_f[b1][d1] + delta_f[a0][d1]*delta_f[b1][c0]) + PB_1_f*(delta_f[a0][c0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][c0])) + PQ_d1_f*(PA_0_f*(delta_f[b0][c0]*delta_f[b1][d0] + delta_f[b0][d0]*delta_f[b1][c0]) + PB_0_f*(delta_f[a0][c0]*delta_f[b1][d0] + delta_f[a0][d0]*delta_f[b1][c0]) + PB_1_f*(delta_f[a0][c0]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][c0]))) + PQ_d0_f*PQ_d1_f*(PA_0_f*(delta_f[b0][c0]*delta_f[b1][c1] + delta_f[b0][c1]*delta_f[b1][c0]) + PB_0_f*(delta_f[a0][c0]*delta_f[b1][c1] + delta_f[a0][c1]*delta_f[b1][c0]) + PB_1_f*(delta_f[a0][c0]*delta_f[b0][c1] + delta_f[a0][c1]*delta_f[b0][c0]))

                            -PQ_c0_f*PQ_c1_f*(PQ_d0_f*(delta_f[a0][b0]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][b1]) + PQ_d1_f*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1])) - PQ_d0_f*PQ_d1_f*(PQ_c0_f*(delta_f[a0][b0]*delta_f[b1][c1] + delta_f[a0][b1]*delta_f[b0][c1] + delta_f[a0][c1]*delta_f[b0][b1]) + PQ_c1_f*(delta_f[a0][b0]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[b0][c0] + delta_f[a0][c0]*delta_f[b0][b1]))

                        )

                        + F7_t_f[4] * 0.25f * S1_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                            -PQ_a0_f*(2*PQ_b0_f*PQ_b1_f*(delta_f[c0][c1]*delta_f[d0][d1] + delta_f[c0][d0]*delta_f[c1][d1] + delta_f[c0][d1]*delta_f[c1][d0]) + PQ_c0_f*(QC_1_f*(delta_f[b0][d0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][d0]) + QD_0_f*(delta_f[b0][c1]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][c1]) + QD_1_f*(delta_f[b0][c1]*delta_f[b1][d0] + delta_f[b0][d0]*delta_f[b1][c1])) + PQ_c1_f*(QC_0_f*(delta_f[b0][d0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][d0]) + QD_0_f*(delta_f[b0][c0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][c0]) + QD_1_f*(delta_f[b0][c0]*delta_f[b1][d0] + delta_f[b0][d0]*delta_f[b1][c0])) + PQ_d0_f*(QC_0_f*(delta_f[b0][c1]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][c1]) + QC_1_f*(delta_f[b0][c0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][c0]) + QD_1_f*(delta_f[b0][c0]*delta_f[b1][c1] + delta_f[b0][c1]*delta_f[b1][c0])) + PQ_d1_f*(QC_0_f*(delta_f[b0][c1]*delta_f[b1][d0] + delta_f[b0][d0]*delta_f[b1][c1]) + QC_1_f*(delta_f[b0][c0]*delta_f[b1][d0] + delta_f[b0][d0]*delta_f[b1][c0]) + QD_0_f*(delta_f[b0][c0]*delta_f[b1][c1] + delta_f[b0][c1]*delta_f[b1][c0]))) - PQ_b0_f*(PQ_c0_f*(QC_1_f*(delta_f[a0][d0]*delta_f[b1][d1] + delta_f[a0][d1]*delta_f[b1][d0]) + QD_0_f*(delta_f[a0][c1]*delta_f[b1][d1] + delta_f[a0][d1]*delta_f[b1][c1]) + QD_1_f*(delta_f[a0][c1]*delta_f[b1][d0] + delta_f[a0][d0]*delta_f[b1][c1])) + PQ_c1_f*(QC_0_f*(delta_f[a0][d0]*delta_f[b1][d1] + delta_f[a0][d1]*delta_f[b1][d0]) + QD_0_f*(delta_f[a0][c0]*delta_f[b1][d1] + delta_f[a0][d1]*delta_f[b1][c0]) + QD_1_f*(delta_f[a0][c0]*delta_f[b1][d0] + delta_f[a0][d0]*delta_f[b1][c0])) + PQ_d0_f*(QC_0_f*(delta_f[a0][c1]*delta_f[b1][d1] + delta_f[a0][d1]*delta_f[b1][c1]) + QC_1_f*(delta_f[a0][c0]*delta_f[b1][d1] + delta_f[a0][d1]*delta_f[b1][c0]) + QD_1_f*(delta_f[a0][c0]*delta_f[b1][c1] + delta_f[a0][c1]*delta_f[b1][c0])) + PQ_d1_f*(QC_0_f*(delta_f[a0][c1]*delta_f[b1][d0] + delta_f[a0][d0]*delta_f[b1][c1]) + QC_1_f*(delta_f[a0][c0]*delta_f[b1][d0] + delta_f[a0][d0]*delta_f[b1][c0]) + QD_0_f*(delta_f[a0][c0]*delta_f[b1][c1] + delta_f[a0][c1]*delta_f[b1][c0]))) - PQ_b1_f*(PQ_c0_f*(QC_1_f*(delta_f[a0][d0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][d0]) + QD_0_f*(delta_f[a0][c1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][c1]) + QD_1_f*(delta_f[a0][c1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][c1])) + PQ_c1_f*(QC_0_f*(delta_f[a0][d0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][d0]) + QD_0_f*(delta_f[a0][c0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][c0]) + QD_1_f*(delta_f[a0][c0]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][c0])) + PQ_d0_f*(QC_0_f*(delta_f[a0][c1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][c1]) + QC_1_f*(delta_f[a0][c0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][c0]) + QD_1_f*(delta_f[a0][c0]*delta_f[b0][c1] + delta_f[a0][c1]*delta_f[b0][c0])) + PQ_d1_f*(QC_0_f*(delta_f[a0][c1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][c1]) + QC_1_f*(delta_f[a0][c0]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][c0]) + QD_0_f*(delta_f[a0][c0]*delta_f[b0][c1] + delta_f[a0][c1]*delta_f[b0][c0])))

                            -PQ_a0_f*(PQ_b0_f*(delta_f[b1][c0]*(delta_f[c1][d0]*(PQ_d1_f + QD_1_f) + delta_f[c1][d1]*(PQ_d0_f + QD_0_f) + delta_f[d0][d1]*(PQ_c1_f + QC_1_f)) + delta_f[b1][c1]*(delta_f[c0][d0]*(PQ_d1_f + QD_1_f) + delta_f[c0][d1]*(PQ_d0_f + QD_0_f) + delta_f[d0][d1]*(PQ_c0_f + QC_0_f)) + delta_f[b1][d0]*(delta_f[c0][c1]*(PQ_d1_f + QD_1_f) + delta_f[c0][d1]*(PQ_c1_f + QC_1_f) + delta_f[c1][d1]*(PQ_c0_f + QC_0_f)) + delta_f[b1][d1]*(delta_f[c0][c1]*(PQ_d0_f + QD_0_f) + delta_f[c0][d0]*(PQ_c1_f + QC_1_f) + delta_f[c1][d0]*(PQ_c0_f + QC_0_f))) + PQ_b1_f*(delta_f[b0][c0]*(delta_f[c1][d0]*(PQ_d1_f + QD_1_f) + delta_f[c1][d1]*(PQ_d0_f + QD_0_f) + delta_f[d0][d1]*(PQ_c1_f + QC_1_f)) + delta_f[b0][c1]*(delta_f[c0][d0]*(PQ_d1_f + QD_1_f) + delta_f[c0][d1]*(PQ_d0_f + QD_0_f) + delta_f[d0][d1]*(PQ_c0_f + QC_0_f)) + delta_f[b0][d0]*(delta_f[c0][c1]*(PQ_d1_f + QD_1_f) + delta_f[c0][d1]*(PQ_c1_f + QC_1_f) + delta_f[c1][d1]*(PQ_c0_f + QC_0_f)) + delta_f[b0][d1]*(delta_f[c0][c1]*(PQ_d0_f + QD_0_f) + delta_f[c0][d0]*(PQ_c1_f + QC_1_f) + delta_f[c1][d0]*(PQ_c0_f + QC_0_f)))) - PQ_b0_f*PQ_b1_f*(delta_f[a0][c0]*(delta_f[c1][d0]*(PQ_d1_f + QD_1_f) + delta_f[c1][d1]*(PQ_d0_f + QD_0_f) + delta_f[d0][d1]*(PQ_c1_f + QC_1_f)) + delta_f[a0][c1]*(delta_f[c0][d0]*(PQ_d1_f + QD_1_f) + delta_f[c0][d1]*(PQ_d0_f + QD_0_f) + delta_f[d0][d1]*(PQ_c0_f + QC_0_f)) + delta_f[a0][d0]*(delta_f[c0][c1]*(PQ_d1_f + QD_1_f) + delta_f[c0][d1]*(PQ_c1_f + QC_1_f) + delta_f[c1][d1]*(PQ_c0_f + QC_0_f)) + delta_f[a0][d1]*(delta_f[c0][c1]*(PQ_d0_f + QD_0_f) + delta_f[c0][d0]*(PQ_c1_f + QC_1_f) + delta_f[c1][d0]*(PQ_c0_f + QC_0_f)))

                            -(PQ_a0_f*delta_f[b0][b1] + PQ_b0_f*delta_f[a0][b1] + PQ_b1_f*delta_f[a0][b0])*(PQ_c0_f*(delta_f[c1][d0]*(PQ_d1_f + QD_1_f) + delta_f[c1][d1]*(PQ_d0_f + QD_0_f) + delta_f[d0][d1]*(PQ_c1_f + QC_1_f)) + PQ_c1_f*(QC_0_f*delta_f[d0][d1] + delta_f[c0][d0]*(PQ_d1_f + QD_1_f) + delta_f[c0][d1]*(PQ_d0_f + QD_0_f)) + PQ_d0_f*(QC_0_f*delta_f[c1][d1] + QC_1_f*delta_f[c0][d1] + delta_f[c0][c1]*(PQ_d1_f + QD_1_f)) + PQ_d1_f*(QC_0_f*delta_f[c1][d0] + QC_1_f*delta_f[c0][d0] + QD_0_f*delta_f[c0][c1]))

                            +(delta_f[a0][b0]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[b0][c0] + delta_f[a0][c0]*delta_f[b0][b1])*(-PQ_c1_f*PQ_d0_f*QD_1_f - PQ_c1_f*PQ_d1_f*QD_0_f - PQ_d0_f*PQ_d1_f*QC_1_f) + (delta_f[a0][b0]*delta_f[b1][c1] + delta_f[a0][b1]*delta_f[b0][c1] + delta_f[a0][c1]*delta_f[b0][b1])*(-PQ_c0_f*PQ_d0_f*QD_1_f - PQ_c0_f*PQ_d1_f*QD_0_f - PQ_d0_f*PQ_d1_f*QC_0_f) + (delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1])*(-PQ_c0_f*PQ_c1_f*QD_1_f - PQ_c0_f*PQ_d1_f*QC_1_f - PQ_c1_f*PQ_d1_f*QC_0_f) + (delta_f[a0][b0]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][b1])*(-PQ_c0_f*PQ_c1_f*QD_0_f - PQ_c0_f*PQ_d0_f*QC_1_f - PQ_c1_f*PQ_d0_f*QC_0_f)

                        )

                        + F7_t_f[4] * 0.5f * S1_f * S1_f * S1_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                            -(PA_0_f*(PB_0_f*PQ_b1_f + PB_1_f*PQ_b0_f) + PB_0_f*PB_1_f*PQ_a0_f)*(PQ_c0_f*(PQ_c1_f*delta_f[d0][d1] + PQ_d0_f*delta_f[c1][d1] + PQ_d1_f*delta_f[c1][d0]) + PQ_c1_f*(PQ_d0_f*delta_f[c0][d1] + PQ_d1_f*delta_f[c0][d0]) + PQ_d0_f*PQ_d1_f*delta_f[c0][c1])

                            -PQ_c0_f*PQ_c1_f*(PA_0_f*(PB_0_f*(PQ_d0_f*delta_f[b1][d1] + PQ_d1_f*delta_f[b1][d0]) + PB_1_f*(PQ_d0_f*delta_f[b0][d1] + PQ_d1_f*delta_f[b0][d0])) + PB_0_f*PB_1_f*(PQ_d0_f*delta_f[a0][d1] + PQ_d1_f*delta_f[a0][d0])) - PQ_d0_f*PQ_d1_f*(PA_0_f*(PB_0_f*(PQ_c0_f*delta_f[b1][c1] + PQ_c1_f*delta_f[b1][c0]) + PB_1_f*(PQ_c0_f*delta_f[b0][c1] + PQ_c1_f*delta_f[b0][c0])) + PB_0_f*PB_1_f*(PQ_c0_f*delta_f[a0][c1] + PQ_c1_f*delta_f[a0][c0]))

                            +PQ_c0_f*PQ_c1_f*PQ_d0_f*PQ_d1_f*(PA_0_f*delta_f[b0][b1] + PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0])

                        )

                        + F7_t_f[4] * 0.5f * S1_f * S1_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (

                            +(PB_0_f*PQ_a0_f*PQ_b1_f + PQ_b0_f*(PA_0_f*PQ_b1_f + PB_1_f*PQ_a0_f))*(PQ_c0_f*(delta_f[c1][d0]*(PQ_d1_f + QD_1_f) + delta_f[c1][d1]*(PQ_d0_f + QD_0_f) + delta_f[d0][d1]*(PQ_c1_f + QC_1_f)) + PQ_c1_f*(QC_0_f*delta_f[d0][d1] + delta_f[c0][d0]*(PQ_d1_f + QD_1_f) + delta_f[c0][d1]*(PQ_d0_f + QD_0_f)) + PQ_d0_f*(QC_0_f*delta_f[c1][d1] + QC_1_f*delta_f[c0][d1] + delta_f[c0][c1]*(PQ_d1_f + QD_1_f)) + PQ_d1_f*(QC_0_f*delta_f[c1][d0] + QC_1_f*delta_f[c0][d0] + QD_0_f*delta_f[c0][c1]))

                            +PA_0_f*(PQ_b0_f*(PQ_c0_f*(PQ_c1_f*(QD_0_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][d0]) + PQ_d0_f*(QC_1_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][c1]) + PQ_d1_f*(QC_1_f*delta_f[b1][d0] + QD_0_f*delta_f[b1][c1])) + PQ_c1_f*(PQ_d0_f*(QC_0_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][c0]) + PQ_d1_f*(QC_0_f*delta_f[b1][d0] + QD_0_f*delta_f[b1][c0]))) + PQ_b1_f*(PQ_c0_f*(PQ_c1_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0]) + PQ_d0_f*(QC_1_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][c1]) + PQ_d1_f*(QC_1_f*delta_f[b0][d0] + QD_0_f*delta_f[b0][c1])) + PQ_c1_f*(PQ_d0_f*(QC_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][c0]) + PQ_d1_f*(QC_0_f*delta_f[b0][d0] + QD_0_f*delta_f[b0][c0])))) + PB_0_f*(PQ_a0_f*(PQ_c0_f*(PQ_c1_f*(QD_0_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][d0]) + PQ_d0_f*(QC_1_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][c1]) + PQ_d1_f*(QC_1_f*delta_f[b1][d0] + QD_0_f*delta_f[b1][c1])) + PQ_c1_f*(PQ_d0_f*(QC_0_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][c0]) + PQ_d1_f*(QC_0_f*delta_f[b1][d0] + QD_0_f*delta_f[b1][c0]))) + PQ_b1_f*(PQ_c0_f*(PQ_c1_f*(QD_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][d0]) + PQ_d0_f*(QC_1_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][c1]) + PQ_d1_f*(QC_1_f*delta_f[a0][d0] + QD_0_f*delta_f[a0][c1])) + PQ_c1_f*(PQ_d0_f*(QC_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][c0]) + PQ_d1_f*(QC_0_f*delta_f[a0][d0] + QD_0_f*delta_f[a0][c0])))) + PB_1_f*(PQ_a0_f*(PQ_c0_f*(PQ_c1_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0]) + PQ_d0_f*(QC_1_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][c1]) + PQ_d1_f*(QC_1_f*delta_f[b0][d0] + QD_0_f*delta_f[b0][c1])) + PQ_c1_f*(PQ_d0_f*(QC_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][c0]) + PQ_d1_f*(QC_0_f*delta_f[b0][d0] + QD_0_f*delta_f[b0][c0]))) + PQ_b0_f*(PQ_c0_f*(PQ_c1_f*(QD_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][d0]) + PQ_d0_f*(QC_1_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][c1]) + PQ_d1_f*(QC_1_f*delta_f[a0][d0] + QD_0_f*delta_f[a0][c1])) + PQ_c1_f*(PQ_d0_f*(QC_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][c0]) + PQ_d1_f*(QC_0_f*delta_f[a0][d0] + QD_0_f*delta_f[a0][c0])))) + PQ_d0_f*PQ_d1_f*(PA_0_f*(PQ_b0_f*(QC_0_f*delta_f[b1][c1] + QC_1_f*delta_f[b1][c0]) + PQ_b1_f*(QC_0_f*delta_f[b0][c1] + QC_1_f*delta_f[b0][c0])) + PB_0_f*(PQ_a0_f*(QC_0_f*delta_f[b1][c1] + QC_1_f*delta_f[b1][c0]) + PQ_b1_f*(QC_0_f*delta_f[a0][c1] + QC_1_f*delta_f[a0][c0])) + PB_1_f*(PQ_a0_f*(QC_0_f*delta_f[b0][c1] + QC_1_f*delta_f[b0][c0]) + PQ_b0_f*(QC_0_f*delta_f[a0][c1] + QC_1_f*delta_f[a0][c0])))

                            +(PQ_c0_f*PQ_c1_f*(PQ_d0_f*QD_1_f + PQ_d1_f*QD_0_f) + PQ_d0_f*PQ_d1_f*(PQ_c0_f*QC_1_f + PQ_c1_f*QC_0_f))*(PA_0_f*delta_f[b0][b1] + PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0] - PQ_a0_f*delta_f[b0][b1] - PQ_b0_f*delta_f[a0][b1] - PQ_b1_f*delta_f[a0][b0])

                        )

                );

            ERIs[threadIdx.y][threadIdx.x] += (double)eri_ijkl_f * mat_D_full_AO[j_cgto * naos + l_cgto];
        }
    }

    __syncthreads();

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {
        double K_ik = 0.0;

        for (uint32_t y = 0; y < TILE_DIM_Y_K; y++)
        {
            for (uint32_t x = 0; x < TILE_DIM_X_K; x++)
            {
                K_ik += ERIs[y][x];
            }
        }

        mat_K[ik] += K_ik;
    }
}