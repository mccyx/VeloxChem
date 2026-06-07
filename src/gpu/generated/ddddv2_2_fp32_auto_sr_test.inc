computeCoulombFockDDDDv2_2_FP32_auto_sr_test(double*         mat_J,
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
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ float    delta[3][3];

    __shared__ float  a_i, a_j, r_i[3], r_j[3], S_ij_00, S1, inv_S1;
    __shared__ float  PA_0, PA_1, PB_0, PB_1;
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

        delta[0][0] = 1.0f; delta[0][1] = 0.0f; delta[0][2] = 0.0f;
        delta[1][0] = 0.0f; delta[1][1] = 1.0f; delta[1][2] = 0.0f;
        delta[2][0] = 0.0f; delta[2][1] = 0.0f; delta[2][2] = 1.0f;

        if (ij < dd_prim_pair_count_local)
        {
            i = dd_first_inds_local[ij];
            j = dd_second_inds_local[ij];

            a_i = d_prim_info[i / 6 + d_prim_count * 0];

            r_i[0] = d_prim_info[i / 6 + d_prim_count * 2];
            r_i[1] = d_prim_info[i / 6 + d_prim_count * 3];
            r_i[2] = d_prim_info[i / 6 + d_prim_count * 4];

            a_j = d_prim_info[j / 6 + d_prim_count * 0];

            r_j[0] = d_prim_info[j / 6 + d_prim_count * 2];
            r_j[1] = d_prim_info[j / 6 + d_prim_count * 3];
            r_j[2] = d_prim_info[j / 6 + d_prim_count * 4];

            S1 = a_i + a_j;
            inv_S1 = (float) (1.0 / (double)S1);

            S_ij_00 = dd_pair_data_local[ij];

            a0 = d_cart_inds[i % 6][0];
            a1 = d_cart_inds[i % 6][1];
            b0 = d_cart_inds[j % 6][0];
            b1 = d_cart_inds[j % 6][1];

            PA_0 = (a_j  * inv_S1) * (r_j[a0] - r_i[a0]);
            PA_1 = (a_j  * inv_S1) * (r_j[a1] - r_i[a1]);
            PB_0 = (-a_i * inv_S1) * (r_j[b0] - r_i[b0]);
            PB_1 = (-a_i * inv_S1) * (r_j[b1] - r_i[b1]);

        }

    }

    ERIs[threadIdx.y] = 0.0;

    __syncthreads();

    for (uint32_t m = prec_cut; m < screen_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= dd_prim_pair_count))
        {
            break;
        }

        const auto k = dd_first_inds[kl];
        const auto l = dd_second_inds[kl];

        const auto a_k = d_prim_info[k / 6 + d_prim_count * 0];

        const float  r_k[3] = {d_prim_info[k / 6 + d_prim_count * 2],
                               d_prim_info[k / 6 + d_prim_count * 3],
                               d_prim_info[k / 6 + d_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const float  r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = dd_pair_data[kl];

        const auto c0 = d_cart_inds[k % 6][0];
        const auto c1 = d_cart_inds[k % 6][1];
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0f / S2;
        const auto inv_S4 = 1.0f / (S1 + S2);

        const float  PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrtf(4.0f * S1 * S2 * MATH_CONST_INV_PI_F * inv_S4);

        float  F8_t[3];

        gpu::computeBoysFunction_f(F8_t, S1 * S2 * inv_S4 * r2_PQ, 2, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QC_1 = (a_l * inv_S2) * (r_l[c1] - r_k[c1]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const float  eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                    + F8_t[2] * 0.25f * S1 * S1 * inv_S2 * inv_S2 * inv_S4 * inv_S4 * (

                        +PA_0*PA_1*PB_0*PB_1*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])

                    )

                    + F8_t[2] * 0.25f * S1 * inv_S2 * inv_S4 * inv_S4 * (

                        -2*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])

                        -PA_0*PA_1*(PB_0*(delta[b1][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[b1][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[b1][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[b1][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))) + PB_1*(delta[b0][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[b0][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[b0][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[b0][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0)))) - PB_0*PB_1*(PA_0*(delta[a1][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[a1][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[a1][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[a1][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))) + PA_1*(delta[a0][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[a0][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[a0][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[a0][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))))

                        +(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1])*(PQ[c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + PQ[c1]*(QC_0*delta[d0][d1] + delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0)) + PQ[d0]*(QC_0*delta[c1][d1] + QC_1*delta[c0][d1] + delta[c0][c1]*(PQ[d1] + QD_1)) + PQ[d1]*(QC_0*delta[c1][d0] + QC_1*delta[c0][d0] + QD_0*delta[c0][c1]))

                    )

                    + F8_t[2] * 0.25f * S2 * S2 * inv_S1 * inv_S1 * inv_S4 * inv_S4 * (

                        +QC_0*QC_1*QD_0*QD_1*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<float>(k != l) + 1.0f) == (k == l ? 1.0f : 2.0f)
        ERIs[threadIdx.y] += (double)(eri_ijkl * dd_mat_D[kl] * (static_cast<float>(k != l) + 1.0f));
    }

    __syncthreads();

    if ((threadIdx.y == 0) && (ij < dd_prim_pair_count_local))
    {
        double J_ij = 0.0;

        for (uint32_t n = 0; n < TILE_DIM_LARGE; n++)
        {
            J_ij += ERIs[n];
        }

        mat_J[ij] += J_ij;
    }
}
