__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDD0_FP64(double*         mat_J,
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
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ double   delta[3][3];

    __shared__ double a_i, a_j, r_i[3], r_j[3], S_ij_00, S1, inv_S1;
    __shared__ double PA_0, PA_1, PB_0, PB_1;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta[0][0] = 1.0; delta[0][1] = 0.0; delta[0][2] = 0.0;
        delta[1][0] = 0.0; delta[1][1] = 1.0; delta[1][2] = 0.0;
        delta[2][0] = 0.0; delta[2][1] = 0.0; delta[2][2] = 1.0;

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
            inv_S1 = 1.0 / S1;

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

    for (uint32_t m = 0; m < prec_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= dd_prim_pair_count))
        {
            break;
        }

        const auto k = dd_first_inds[kl];
        const auto l = dd_second_inds[kl];

        const auto a_k = d_prim_info[k / 6 + d_prim_count * 0];

        const double r_k[3] = {d_prim_info[k / 6 + d_prim_count * 2],
                               d_prim_info[k / 6 + d_prim_count * 3],
                               d_prim_info[k / 6 + d_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = dd_pair_data[kl];

        const auto c0 = d_cart_inds[k % 6][0];
        const auto c1 = d_cart_inds[k % 6][1];
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F8_t[3];

        gpu::computeBoysFunction(F8_t, S1 * S2 * inv_S4 * r2_PQ, 2, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QC_1 = (a_l * inv_S2) * (r_l[c1] - r_k[c1]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                    + F8_t[0] * 0.125 * inv_S1 * inv_S1 * inv_S2 * (

                        +(QC_0*(QC_1*delta[d0][d1] + QD_0*delta[c1][d1] + QD_1*delta[c1][d0]) + QC_1*(QD_0*delta[c0][d1] + QD_1*delta[c0][d0]) + QD_0*QD_1*delta[c0][c1])*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])

                    )

                    + F8_t[0] * 0.125 * inv_S1 * inv_S2 * inv_S2 * (

                        +(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1])*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])

                    )

                    + F8_t[0] * 0.25 * inv_S1 * inv_S1 * (

                        +QC_0*QC_1*QD_0*QD_1*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])

                    )

                    + F8_t[0] * 0.25 * inv_S1 * inv_S2 * (

                        +(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1])*(QC_0*(QC_1*delta[d0][d1] + QD_0*delta[c1][d1] + QD_1*delta[c1][d0]) + QC_1*(QD_0*delta[c0][d1] + QD_1*delta[c0][d0]) + QD_0*QD_1*delta[c0][c1])

                    )

                    + F8_t[0] * 0.25 * inv_S2 * inv_S2 * (

                        +PA_0*PA_1*PB_0*PB_1*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])

                    )

                    + F8_t[0] * 0.5 * inv_S1 * (

                        +QC_0*QC_1*QD_0*QD_1*(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1])

                    )

                    + F8_t[0] * 0.5 * inv_S2 * (

                        +PA_0*PA_1*PB_0*PB_1*(QC_0*(QC_1*delta[d0][d1] + QD_0*delta[c1][d1] + QD_1*delta[c1][d0]) + QC_1*(QD_0*delta[c0][d1] + QD_1*delta[c0][d0]) + QD_0*QD_1*delta[c0][c1])

                    )

                    + F8_t[0] * (

                        +PA_0*PA_1*PB_0*PB_1*QC_0*QC_1*QD_0*QD_1

                    )

                    + F8_t[0] * 0.0625 * inv_S1 * inv_S1 * inv_S2 * inv_S2 * (

                        +(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])

                    )

                    + F8_t[1] * (-0.125) * inv_S1 * inv_S1 * inv_S2 * inv_S4 * (

                        +(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])

                    )

                    + F8_t[1] * (-0.125) * inv_S1 * inv_S2 * inv_S2 * inv_S4 * (

                        +(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])

                    )

                    + F8_t[1] * (-0.25) * inv_S1 * inv_S1 * inv_S4 * (

                        +(QC_0*(QC_1*delta[d0][d1] + QD_0*delta[c1][d1] + QD_1*delta[c1][d0]) + QC_1*(QD_0*delta[c0][d1] + QD_1*delta[c0][d0]) + QD_0*QD_1*delta[c0][c1])*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])

                    )

                    + F8_t[1] * 0.125 * inv_S1 * inv_S2 * inv_S4 * (

                        +(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])*(PA_0*(-PA_1*delta[b0][b1] - PB_0*delta[a1][b1] - PB_1*delta[a1][b0] + PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PA_1*(PQ[a0]*delta[b0][b1] + delta[a0][b0]*(-PB_1 + PQ[b1]) + delta[a0][b1]*(-PB_0 + PQ[b0])) + PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + delta[a0][a1]*(-PB_1 + PQ[b1])) + PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]))

                        +PA_0*(delta[a1][b0]*(QC_0*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + QC_1*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + QD_0*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + QD_1*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) + delta[a1][b1]*(QC_0*(delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0]) + QC_1*(delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0]) + QD_0*(delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1]) + QD_1*(delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1])) + delta[b0][b1]*(QC_0*(delta[a1][c1]*delta[d0][d1] + delta[a1][d0]*delta[c1][d1] + delta[a1][d1]*delta[c1][d0]) + QC_1*(delta[a1][c0]*delta[d0][d1] + delta[a1][d0]*delta[c0][d1] + delta[a1][d1]*delta[c0][d0]) + QD_0*(delta[a1][c0]*delta[c1][d1] + delta[a1][c1]*delta[c0][d1] + delta[a1][d1]*delta[c0][c1]) + QD_1*(delta[a1][c0]*delta[c1][d0] + delta[a1][c1]*delta[c0][d0] + delta[a1][d0]*delta[c0][c1]))) + PA_1*(delta[a0][b0]*(QC_0*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + QC_1*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + QD_0*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + QD_1*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) + delta[a0][b1]*(QC_0*(delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0]) + QC_1*(delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0]) + QD_0*(delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1]) + QD_1*(delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1])) + delta[b0][b1]*(QC_0*(delta[a0][c1]*delta[d0][d1] + delta[a0][d0]*delta[c1][d1] + delta[a0][d1]*delta[c1][d0]) + QC_1*(delta[a0][c0]*delta[d0][d1] + delta[a0][d0]*delta[c0][d1] + delta[a0][d1]*delta[c0][d0]) + QD_0*(delta[a0][c0]*delta[c1][d1] + delta[a0][c1]*delta[c0][d1] + delta[a0][d1]*delta[c0][c1]) + QD_1*(delta[a0][c0]*delta[c1][d0] + delta[a0][c1]*delta[c0][d0] + delta[a0][d0]*delta[c0][c1]))) + PB_0*(delta[a0][a1]*(QC_0*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + QC_1*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + QD_0*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + QD_1*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) + delta[a0][b1]*(QC_0*(delta[a1][c1]*delta[d0][d1] + delta[a1][d0]*delta[c1][d1] + delta[a1][d1]*delta[c1][d0]) + QC_1*(delta[a1][c0]*delta[d0][d1] + delta[a1][d0]*delta[c0][d1] + delta[a1][d1]*delta[c0][d0]) + QD_0*(delta[a1][c0]*delta[c1][d1] + delta[a1][c1]*delta[c0][d1] + delta[a1][d1]*delta[c0][c1]) + QD_1*(delta[a1][c0]*delta[c1][d0] + delta[a1][c1]*delta[c0][d0] + delta[a1][d0]*delta[c0][c1])) + delta[a1][b1]*(QC_0*(delta[a0][c1]*delta[d0][d1] + delta[a0][d0]*delta[c1][d1] + delta[a0][d1]*delta[c1][d0]) + QC_1*(delta[a0][c0]*delta[d0][d1] + delta[a0][d0]*delta[c0][d1] + delta[a0][d1]*delta[c0][d0]) + QD_0*(delta[a0][c0]*delta[c1][d1] + delta[a0][c1]*delta[c0][d1] + delta[a0][d1]*delta[c0][c1]) + QD_1*(delta[a0][c0]*delta[c1][d0] + delta[a0][c1]*delta[c0][d0] + delta[a0][d0]*delta[c0][c1]))) + PB_1*(delta[a0][a1]*(QC_0*(delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0]) + QC_1*(delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0]) + QD_0*(delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1]) + QD_1*(delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1])) + delta[a0][b0]*(QC_0*(delta[a1][c1]*delta[d0][d1] + delta[a1][d0]*delta[c1][d1] + delta[a1][d1]*delta[c1][d0]) + QC_1*(delta[a1][c0]*delta[d0][d1] + delta[a1][d0]*delta[c0][d1] + delta[a1][d1]*delta[c0][d0]) + QD_0*(delta[a1][c0]*delta[c1][d1] + delta[a1][c1]*delta[c0][d1] + delta[a1][d1]*delta[c0][c1]) + QD_1*(delta[a1][c0]*delta[c1][d0] + delta[a1][c1]*delta[c0][d0] + delta[a1][d0]*delta[c0][c1])) + delta[a1][b0]*(QC_0*(delta[a0][c1]*delta[d0][d1] + delta[a0][d0]*delta[c1][d1] + delta[a0][d1]*delta[c1][d0]) + QC_1*(delta[a0][c0]*delta[d0][d1] + delta[a0][d0]*delta[c0][d1] + delta[a0][d1]*delta[c0][d0]) + QD_0*(delta[a0][c0]*delta[c1][d1] + delta[a0][c1]*delta[c0][d1] + delta[a0][d1]*delta[c0][c1]) + QD_1*(delta[a0][c0]*delta[c1][d0] + delta[a0][c1]*delta[c0][d0] + delta[a0][d0]*delta[c0][c1])))

                        -(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])*(QC_0*(PQ[c1]*delta[d0][d1] + PQ[d0]*delta[c1][d1] + PQ[d1]*delta[c1][d0]) + QC_1*(PQ[d0]*delta[c0][d1] + PQ[d1]*delta[c0][d0] + delta[d0][d1]*(PQ[c0] + QC_0)) + QD_0*(PQ[d1]*delta[c0][c1] + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + QD_1*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0)))

                    )

                    + F8_t[1] * (-0.25) * inv_S2 * inv_S2 * inv_S4 * (

                        +(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1])*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])

                    )

                    + F8_t[1] * (-0.5) * S1 * inv_S2 * inv_S2 * inv_S4 * (

                        +PA_0*PA_1*PB_0*PB_1*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])

                    )

                    + F8_t[1] * (-0.5) * S2 * inv_S1 * inv_S1 * inv_S4 * (

                        +QC_0*QC_1*QD_0*QD_1*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])

                    )

                    + F8_t[1] * 0.25 * inv_S1 * inv_S4 * (

                        +(QC_0*(QC_1*delta[d0][d1] + QD_0*delta[c1][d1] + QD_1*delta[c1][d0]) + QC_1*(QD_0*delta[c0][d1] + QD_1*delta[c0][d0]) + QD_0*QD_1*delta[c0][c1])*(PA_0*(-PA_1*delta[b0][b1] - PB_0*delta[a1][b1] - PB_1*delta[a1][b0] + PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PA_1*(PQ[a0]*delta[b0][b1] + delta[a0][b0]*(-PB_1 + PQ[b1]) + delta[a0][b1]*(-PB_0 + PQ[b0])) + PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + delta[a0][a1]*(-PB_1 + PQ[b1])) + PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]))

                        +QC_0*QC_1*(QD_0*(PA_0*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0])) + QD_1*(PA_0*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]))) + QD_0*QD_1*(QC_0*(PA_0*(delta[a1][b0]*delta[b1][c1] + delta[a1][b1]*delta[b0][c1] + delta[a1][c1]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][c1] + delta[a0][b1]*delta[b0][c1] + delta[a0][c1]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][c1] + delta[a0][b1]*delta[a1][c1] + delta[a0][c1]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][c1] + delta[a0][b0]*delta[a1][c1] + delta[a0][c1]*delta[a1][b0])) + QC_1*(PA_0*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0])))

                        -(QC_0*QD_1*(PQ[c1]*QD_0 + PQ[d0]*QC_1) + QC_1*QD_0*(PQ[c0]*QD_1 + PQ[d1]*QC_0))*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])

                    )

                    + F8_t[1] * 0.25 * inv_S2 * inv_S4 * (

                        +(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])

                        +PA_0*PA_1*(PB_0*(QC_0*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + QC_1*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + QD_0*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + QD_1*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) + PB_1*(QC_0*(delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0]) + QC_1*(delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0]) + QD_0*(delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1]) + QD_1*(delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1]))) + PB_0*PB_1*(PA_0*(QC_0*(delta[a1][c1]*delta[d0][d1] + delta[a1][d0]*delta[c1][d1] + delta[a1][d1]*delta[c1][d0]) + QC_1*(delta[a1][c0]*delta[d0][d1] + delta[a1][d0]*delta[c0][d1] + delta[a1][d1]*delta[c0][d0]) + QD_0*(delta[a1][c0]*delta[c1][d1] + delta[a1][c1]*delta[c0][d1] + delta[a1][d1]*delta[c0][c1]) + QD_1*(delta[a1][c0]*delta[c1][d0] + delta[a1][c1]*delta[c0][d0] + delta[a1][d0]*delta[c0][c1])) + PA_1*(QC_0*(delta[a0][c1]*delta[d0][d1] + delta[a0][d0]*delta[c1][d1] + delta[a0][d1]*delta[c1][d0]) + QC_1*(delta[a0][c0]*delta[d0][d1] + delta[a0][d0]*delta[c0][d1] + delta[a0][d1]*delta[c0][d0]) + QD_0*(delta[a0][c0]*delta[c1][d1] + delta[a0][c1]*delta[c0][d1] + delta[a0][d1]*delta[c0][c1]) + QD_1*(delta[a0][c0]*delta[c1][d0] + delta[a0][c1]*delta[c0][d0] + delta[a0][d0]*delta[c0][c1])))

                        -(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1])*(QC_0*(PQ[c1]*delta[d0][d1] + PQ[d0]*delta[c1][d1] + PQ[d1]*delta[c1][d0]) + QC_1*(PQ[d0]*delta[c0][d1] + PQ[d1]*delta[c0][d0] + delta[d0][d1]*(PQ[c0] + QC_0)) + QD_0*(PQ[d1]*delta[c0][c1] + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + QD_1*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0)))

                    )

                    + F8_t[1] * (-0.5) * S1 * inv_S2 * inv_S4 * (

                        +PA_0*PA_1*PB_0*PB_1*(QC_0*(PQ[c1]*delta[d0][d1] + PQ[d0]*delta[c1][d1] + PQ[d1]*delta[c1][d0]) + QC_1*(PQ[d0]*delta[c0][d1] + PQ[d1]*delta[c0][d0] + delta[d0][d1]*(PQ[c0] + QC_0)) + QD_0*(PQ[d1]*delta[c0][c1] + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + QD_1*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0)))

                    )

                    + F8_t[1] * 0.5 * S2 * inv_S1 * inv_S4 * (

                        +QC_0*QC_1*QD_0*QD_1*(PA_0*(-PA_1*delta[b0][b1] - PB_0*delta[a1][b1] - PB_1*delta[a1][b0] + PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PA_1*(PQ[a0]*delta[b0][b1] + delta[a0][b0]*(-PB_1 + PQ[b1]) + delta[a0][b1]*(-PB_0 + PQ[b0])) + PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + delta[a0][a1]*(-PB_1 + PQ[b1])) + PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]))

                    )

                    + F8_t[1] * 0.5 * inv_S4 * (

                        +(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))*(QC_0*(QC_1*delta[d0][d1] + QD_0*delta[c1][d1] + QD_1*delta[c1][d0]) + QC_1*(QD_0*delta[c0][d1] + QD_1*delta[c0][d0]) + QD_0*QD_1*delta[c0][c1])

                        +PA_0*PA_1*(QC_0*QC_1*(PB_0*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + PB_1*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0])) + QD_0*QD_1*(PB_0*(QC_0*delta[b1][c1] + QC_1*delta[b1][c0]) + PB_1*(QC_0*delta[b0][c1] + QC_1*delta[b0][c0]))) + PB_0*PB_1*(QC_0*QC_1*(PA_0*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]) + PA_1*(QD_0*delta[a0][d1] + QD_1*delta[a0][d0])) + QD_0*QD_1*(PA_0*(QC_0*delta[a1][c1] + QC_1*delta[a1][c0]) + PA_1*(QC_0*delta[a0][c1] + QC_1*delta[a0][c0])))

                        -(QC_0*QD_1*(PQ[c1]*QD_0 + PQ[d0]*QC_1) + QC_1*QD_0*(PQ[c0]*QD_1 + PQ[d1]*QC_0))*(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1])

                    )

                    + F8_t[1] * S1 * inv_S4 * (

                        -PA_0*PA_1*PB_0*PB_1*(QC_0*QD_1*(PQ[c1]*QD_0 + PQ[d0]*QC_1) + QC_1*QD_0*(PQ[c0]*QD_1 + PQ[d1]*QC_0))

                    )

                    + F8_t[1] * S2 * inv_S4 * (

                        +QC_0*QC_1*QD_0*QD_1*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))

                    )

                    + F8_t[2] * 0.125 * S1 * inv_S2 * inv_S2 * inv_S4 * inv_S4 * (

                        +(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1])*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * dd_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDD1_FP64(double*         mat_J,
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
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ double   delta[3][3];

    __shared__ double a_i, a_j, r_i[3], r_j[3], S_ij_00, S1, inv_S1;
    __shared__ double PA_0, PA_1, PB_0, PB_1;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta[0][0] = 1.0; delta[0][1] = 0.0; delta[0][2] = 0.0;
        delta[1][0] = 0.0; delta[1][1] = 1.0; delta[1][2] = 0.0;
        delta[2][0] = 0.0; delta[2][1] = 0.0; delta[2][2] = 1.0;

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
            inv_S1 = 1.0 / S1;

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

    for (uint32_t m = 0; m < prec_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= dd_prim_pair_count))
        {
            break;
        }

        const auto k = dd_first_inds[kl];
        const auto l = dd_second_inds[kl];

        const auto a_k = d_prim_info[k / 6 + d_prim_count * 0];

        const double r_k[3] = {d_prim_info[k / 6 + d_prim_count * 2],
                               d_prim_info[k / 6 + d_prim_count * 3],
                               d_prim_info[k / 6 + d_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = dd_pair_data[kl];

        const auto c0 = d_cart_inds[k % 6][0];
        const auto c1 = d_cart_inds[k % 6][1];
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F8_t[3];

        gpu::computeBoysFunction(F8_t, S1 * S2 * inv_S4 * r2_PQ, 2, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QC_1 = (a_l * inv_S2) * (r_l[c1] - r_k[c1]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                    + F8_t[2] * 0.125 * S2 * inv_S1 * inv_S1 * inv_S4 * inv_S4 * (

                        +(QC_0*(QC_1*delta[d0][d1] + QD_0*delta[c1][d1] + QD_1*delta[c1][d0]) + QC_1*(QD_0*delta[c0][d1] + QD_1*delta[c0][d0]) + QD_0*QD_1*delta[c0][c1])*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])

                    )

                    + F8_t[2] * 0.125 * inv_S1 * inv_S4 * inv_S4 * (

                        +(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])*(-PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) - PA_1*(PQ[a0]*delta[b0][b1] + PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) - PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) - PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]) + PQ[a0]*PQ[a1]*delta[b0][b1] + PQ[a0]*PQ[b0]*delta[a1][b1] + PQ[a0]*PQ[b1]*delta[a1][b0] + PQ[a1]*PQ[b0]*delta[a0][b1] + PQ[a1]*PQ[b1]*delta[a0][b0] + PQ[b0]*PQ[b1]*delta[a0][a1])

                        +(-PA_0*QC_0 + PQ[a0]*QC_0)*(delta[a1][b0]*delta[b1][c1]*delta[d0][d1] + delta[a1][b0]*delta[b1][d0]*delta[c1][d1] + delta[a1][b0]*delta[b1][d1]*delta[c1][d0] + delta[a1][b1]*delta[b0][c1]*delta[d0][d1] + delta[a1][b1]*delta[b0][d0]*delta[c1][d1] + delta[a1][b1]*delta[b0][d1]*delta[c1][d0] + delta[a1][c1]*delta[b0][b1]*delta[d0][d1] + delta[a1][d0]*delta[b0][b1]*delta[c1][d1] + delta[a1][d1]*delta[b0][b1]*delta[c1][d0]) + (-PA_0*QC_1 + PQ[a0]*QC_1)*(delta[a1][b0]*delta[b1][c0]*delta[d0][d1] + delta[a1][b0]*delta[b1][d0]*delta[c0][d1] + delta[a1][b0]*delta[b1][d1]*delta[c0][d0] + delta[a1][b1]*delta[b0][c0]*delta[d0][d1] + delta[a1][b1]*delta[b0][d0]*delta[c0][d1] + delta[a1][b1]*delta[b0][d1]*delta[c0][d0] + delta[a1][c0]*delta[b0][b1]*delta[d0][d1] + delta[a1][d0]*delta[b0][b1]*delta[c0][d1] + delta[a1][d1]*delta[b0][b1]*delta[c0][d0]) + (-PA_0*QD_0 + PQ[a0]*QD_0)*(delta[a1][b0]*delta[b1][c0]*delta[c1][d1] + delta[a1][b0]*delta[b1][c1]*delta[c0][d1] + delta[a1][b0]*delta[b1][d1]*delta[c0][c1] + delta[a1][b1]*delta[b0][c0]*delta[c1][d1] + delta[a1][b1]*delta[b0][c1]*delta[c0][d1] + delta[a1][b1]*delta[b0][d1]*delta[c0][c1] + delta[a1][c0]*delta[b0][b1]*delta[c1][d1] + delta[a1][c1]*delta[b0][b1]*delta[c0][d1] + delta[a1][d1]*delta[b0][b1]*delta[c0][c1]) + (-PA_0*QD_1 + PQ[a0]*QD_1)*(delta[a1][b0]*delta[b1][c0]*delta[c1][d0] + delta[a1][b0]*delta[b1][c1]*delta[c0][d0] + delta[a1][b0]*delta[b1][d0]*delta[c0][c1] + delta[a1][b1]*delta[b0][c0]*delta[c1][d0] + delta[a1][b1]*delta[b0][c1]*delta[c0][d0] + delta[a1][b1]*delta[b0][d0]*delta[c0][c1] + delta[a1][c0]*delta[b0][b1]*delta[c1][d0] + delta[a1][c1]*delta[b0][b1]*delta[c0][d0] + delta[a1][d0]*delta[b0][b1]*delta[c0][c1]) + (-PA_1*QC_0 + PQ[a1]*QC_0)*(delta[a0][b0]*delta[b1][c1]*delta[d0][d1] + delta[a0][b0]*delta[b1][d0]*delta[c1][d1] + delta[a0][b0]*delta[b1][d1]*delta[c1][d0] + delta[a0][b1]*delta[b0][c1]*delta[d0][d1] + delta[a0][b1]*delta[b0][d0]*delta[c1][d1] + delta[a0][b1]*delta[b0][d1]*delta[c1][d0] + delta[a0][c1]*delta[b0][b1]*delta[d0][d1] + delta[a0][d0]*delta[b0][b1]*delta[c1][d1] + delta[a0][d1]*delta[b0][b1]*delta[c1][d0]) + (-PA_1*QC_1 + PQ[a1]*QC_1)*(delta[a0][b0]*delta[b1][c0]*delta[d0][d1] + delta[a0][b0]*delta[b1][d0]*delta[c0][d1] + delta[a0][b0]*delta[b1][d1]*delta[c0][d0] + delta[a0][b1]*delta[b0][c0]*delta[d0][d1] + delta[a0][b1]*delta[b0][d0]*delta[c0][d1] + delta[a0][b1]*delta[b0][d1]*delta[c0][d0] + delta[a0][c0]*delta[b0][b1]*delta[d0][d1] + delta[a0][d0]*delta[b0][b1]*delta[c0][d1] + delta[a0][d1]*delta[b0][b1]*delta[c0][d0]) + (-PA_1*QD_0 + PQ[a1]*QD_0)*(delta[a0][b0]*delta[b1][c0]*delta[c1][d1] + delta[a0][b0]*delta[b1][c1]*delta[c0][d1] + delta[a0][b0]*delta[b1][d1]*delta[c0][c1] + delta[a0][b1]*delta[b0][c0]*delta[c1][d1] + delta[a0][b1]*delta[b0][c1]*delta[c0][d1] + delta[a0][b1]*delta[b0][d1]*delta[c0][c1] + delta[a0][c0]*delta[b0][b1]*delta[c1][d1] + delta[a0][c1]*delta[b0][b1]*delta[c0][d1] + delta[a0][d1]*delta[b0][b1]*delta[c0][c1]) + (-PA_1*QD_1 + PQ[a1]*QD_1)*(delta[a0][b0]*delta[b1][c0]*delta[c1][d0] + delta[a0][b0]*delta[b1][c1]*delta[c0][d0] + delta[a0][b0]*delta[b1][d0]*delta[c0][c1] + delta[a0][b1]*delta[b0][c0]*delta[c1][d0] + delta[a0][b1]*delta[b0][c1]*delta[c0][d0] + delta[a0][b1]*delta[b0][d0]*delta[c0][c1] + delta[a0][c0]*delta[b0][b1]*delta[c1][d0] + delta[a0][c1]*delta[b0][b1]*delta[c0][d0] + delta[a0][d0]*delta[b0][b1]*delta[c0][c1]) + (-PB_0*QC_0 + PQ[b0]*QC_0)*(delta[a0][a1]*delta[b1][c1]*delta[d0][d1] + delta[a0][a1]*delta[b1][d0]*delta[c1][d1] + delta[a0][a1]*delta[b1][d1]*delta[c1][d0] + delta[a0][b1]*delta[a1][c1]*delta[d0][d1] + delta[a0][b1]*delta[a1][d0]*delta[c1][d1] + delta[a0][b1]*delta[a1][d1]*delta[c1][d0] + delta[a0][c1]*delta[a1][b1]*delta[d0][d1] + delta[a0][d0]*delta[a1][b1]*delta[c1][d1] + delta[a0][d1]*delta[a1][b1]*delta[c1][d0]) + (-PB_0*QC_1 + PQ[b0]*QC_1)*(delta[a0][a1]*delta[b1][c0]*delta[d0][d1] + delta[a0][a1]*delta[b1][d0]*delta[c0][d1] + delta[a0][a1]*delta[b1][d1]*delta[c0][d0] + delta[a0][b1]*delta[a1][c0]*delta[d0][d1] + delta[a0][b1]*delta[a1][d0]*delta[c0][d1] + delta[a0][b1]*delta[a1][d1]*delta[c0][d0] + delta[a0][c0]*delta[a1][b1]*delta[d0][d1] + delta[a0][d0]*delta[a1][b1]*delta[c0][d1] + delta[a0][d1]*delta[a1][b1]*delta[c0][d0]) + (-PB_0*QD_0 + PQ[b0]*QD_0)*(delta[a0][a1]*delta[b1][c0]*delta[c1][d1] + delta[a0][a1]*delta[b1][c1]*delta[c0][d1] + delta[a0][a1]*delta[b1][d1]*delta[c0][c1] + delta[a0][b1]*delta[a1][c0]*delta[c1][d1] + delta[a0][b1]*delta[a1][c1]*delta[c0][d1] + delta[a0][b1]*delta[a1][d1]*delta[c0][c1] + delta[a0][c0]*delta[a1][b1]*delta[c1][d1] + delta[a0][c1]*delta[a1][b1]*delta[c0][d1] + delta[a0][d1]*delta[a1][b1]*delta[c0][c1]) + (-PB_0*QD_1 + PQ[b0]*QD_1)*(delta[a0][a1]*delta[b1][c0]*delta[c1][d0] + delta[a0][a1]*delta[b1][c1]*delta[c0][d0] + delta[a0][a1]*delta[b1][d0]*delta[c0][c1] + delta[a0][b1]*delta[a1][c0]*delta[c1][d0] + delta[a0][b1]*delta[a1][c1]*delta[c0][d0] + delta[a0][b1]*delta[a1][d0]*delta[c0][c1] + delta[a0][c0]*delta[a1][b1]*delta[c1][d0] + delta[a0][c1]*delta[a1][b1]*delta[c0][d0] + delta[a0][d0]*delta[a1][b1]*delta[c0][c1]) + (-PB_1*QC_0 + PQ[b1]*QC_0)*(delta[a0][a1]*delta[b0][c1]*delta[d0][d1] + delta[a0][a1]*delta[b0][d0]*delta[c1][d1] + delta[a0][a1]*delta[b0][d1]*delta[c1][d0] + delta[a0][b0]*delta[a1][c1]*delta[d0][d1] + delta[a0][b0]*delta[a1][d0]*delta[c1][d1] + delta[a0][b0]*delta[a1][d1]*delta[c1][d0] + delta[a0][c1]*delta[a1][b0]*delta[d0][d1] + delta[a0][d0]*delta[a1][b0]*delta[c1][d1] + delta[a0][d1]*delta[a1][b0]*delta[c1][d0]) + (-PB_1*QC_1 + PQ[b1]*QC_1)*(delta[a0][a1]*delta[b0][c0]*delta[d0][d1] + delta[a0][a1]*delta[b0][d0]*delta[c0][d1] + delta[a0][a1]*delta[b0][d1]*delta[c0][d0] + delta[a0][b0]*delta[a1][c0]*delta[d0][d1] + delta[a0][b0]*delta[a1][d0]*delta[c0][d1] + delta[a0][b0]*delta[a1][d1]*delta[c0][d0] + delta[a0][c0]*delta[a1][b0]*delta[d0][d1] + delta[a0][d0]*delta[a1][b0]*delta[c0][d1] + delta[a0][d1]*delta[a1][b0]*delta[c0][d0]) + (-PB_1*QD_0 + PQ[b1]*QD_0)*(delta[a0][a1]*delta[b0][c0]*delta[c1][d1] + delta[a0][a1]*delta[b0][c1]*delta[c0][d1] + delta[a0][a1]*delta[b0][d1]*delta[c0][c1] + delta[a0][b0]*delta[a1][c0]*delta[c1][d1] + delta[a0][b0]*delta[a1][c1]*delta[c0][d1] + delta[a0][b0]*delta[a1][d1]*delta[c0][c1] + delta[a0][c0]*delta[a1][b0]*delta[c1][d1] + delta[a0][c1]*delta[a1][b0]*delta[c0][d1] + delta[a0][d1]*delta[a1][b0]*delta[c0][c1]) + (-PB_1*QD_1 + PQ[b1]*QD_1)*(delta[a0][a1]*delta[b0][c0]*delta[c1][d0] + delta[a0][a1]*delta[b0][c1]*delta[c0][d0] + delta[a0][a1]*delta[b0][d0]*delta[c0][c1] + delta[a0][b0]*delta[a1][c0]*delta[c1][d0] + delta[a0][b0]*delta[a1][c1]*delta[c0][d0] + delta[a0][b0]*delta[a1][d0]*delta[c0][c1] + delta[a0][c0]*delta[a1][b0]*delta[c1][d0] + delta[a0][c1]*delta[a1][b0]*delta[c0][d0] + delta[a0][d0]*delta[a1][b0]*delta[c0][c1])

                        +2.0*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])*(QC_0*(PQ[c1]*delta[d0][d1] + PQ[d0]*delta[c1][d1] + PQ[d1]*delta[c1][d0]) + QC_1*(PQ[d0]*delta[c0][d1] + PQ[d1]*delta[c0][d0] + delta[d0][d1]*(PQ[c0] + QC_0)) + QD_0*(PQ[d1]*delta[c0][c1] + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + QD_1*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0)))

                        +QC_0*(QC_1*(delta[a0][b0]*(delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + delta[a0][d0]*(delta[a1][b0]*delta[b1][d1] + delta[a1][d1]*delta[b0][b1]) + delta[a0][d1]*(delta[a1][b0]*delta[b1][d0] + delta[a1][d0]*delta[b0][b1]) + delta[b0][d0]*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + delta[b0][d1]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1])) + QD_0*(delta[a0][b0]*(delta[a1][c1]*delta[b1][d1] + delta[a1][d1]*delta[b1][c1]) + delta[a0][c1]*(delta[a1][b0]*delta[b1][d1] + delta[a1][d1]*delta[b0][b1]) + delta[a0][d1]*(delta[a1][b0]*delta[b1][c1] + delta[a1][c1]*delta[b0][b1]) + delta[b0][c1]*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + delta[b0][d1]*(delta[a0][a1]*delta[b1][c1] + delta[a0][b1]*delta[a1][c1] + delta[a0][c1]*delta[a1][b1])) + QD_1*(delta[a0][b0]*(delta[a1][c1]*delta[b1][d0] + delta[a1][d0]*delta[b1][c1]) + delta[a0][c1]*(delta[a1][b0]*delta[b1][d0] + delta[a1][d0]*delta[b0][b1]) + delta[a0][d0]*(delta[a1][b0]*delta[b1][c1] + delta[a1][c1]*delta[b0][b1]) + delta[b0][c1]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + delta[b0][d0]*(delta[a0][a1]*delta[b1][c1] + delta[a0][b1]*delta[a1][c1] + delta[a0][c1]*delta[a1][b1]))) + QC_1*(QD_0*(delta[a0][b0]*(delta[a1][c0]*delta[b1][d1] + delta[a1][d1]*delta[b1][c0]) + delta[a0][c0]*(delta[a1][b0]*delta[b1][d1] + delta[a1][d1]*delta[b0][b1]) + delta[a0][d1]*(delta[a1][b0]*delta[b1][c0] + delta[a1][c0]*delta[b0][b1]) + delta[b0][c0]*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + delta[b0][d1]*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1])) + QD_1*(delta[a0][b0]*(delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + delta[a0][c0]*(delta[a1][b0]*delta[b1][d0] + delta[a1][d0]*delta[b0][b1]) + delta[a0][d0]*(delta[a1][b0]*delta[b1][c0] + delta[a1][c0]*delta[b0][b1]) + delta[b0][c0]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + delta[b0][d0]*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]))) + QD_0*QD_1*(delta[a0][b0]*(delta[a1][c0]*delta[b1][c1] + delta[a1][c1]*delta[b1][c0]) + delta[a0][c0]*(delta[a1][b0]*delta[b1][c1] + delta[a1][c1]*delta[b0][b1]) + delta[a0][c1]*(delta[a1][b0]*delta[b1][c0] + delta[a1][c0]*delta[b0][b1]) + delta[b0][c0]*(delta[a0][a1]*delta[b1][c1] + delta[a0][b1]*delta[a1][c1] + delta[a0][c1]*delta[a1][b1]) + delta[b0][c1]*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]))

                    )

                    + F8_t[2] * 0.125 * inv_S2 * inv_S4 * inv_S4 * (

                        +(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])*(2.0*PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) - 2*PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + 2.0*PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) - 2*PA_1*(PQ[a0]*delta[b0][b1] + PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + 2.0*PB_0*PB_1*delta[a0][a1] - 2*PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) - 2*PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]))

                        +PA_0*(PA_1*(delta[b0][c0]*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + delta[b0][c1]*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + delta[b0][d0]*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + delta[b0][d1]*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) + PB_0*(delta[a1][c0]*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + delta[a1][c1]*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + delta[a1][d0]*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + delta[a1][d1]*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) + PB_1*(delta[a1][c0]*(delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0]) + delta[a1][c1]*(delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0]) + delta[a1][d0]*(delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1]) + delta[a1][d1]*(delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1]))) + PA_1*(PB_0*(delta[a0][c0]*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + delta[a0][c1]*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + delta[a0][d0]*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + delta[a0][d1]*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) + PB_1*(delta[a0][c0]*(delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0]) + delta[a0][c1]*(delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0]) + delta[a0][d0]*(delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1]) + delta[a0][d1]*(delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1]))) + PB_0*PB_1*(delta[a0][c0]*(delta[a1][c1]*delta[d0][d1] + delta[a1][d0]*delta[c1][d1] + delta[a1][d1]*delta[c1][d0]) + delta[a0][c1]*(delta[a1][c0]*delta[d0][d1] + delta[a1][d0]*delta[c0][d1] + delta[a1][d1]*delta[c0][d0]) + delta[a0][d0]*(delta[a1][c0]*delta[c1][d1] + delta[a1][c1]*delta[c0][d1] + delta[a1][d1]*delta[c0][c1]) + delta[a0][d1]*(delta[a1][c0]*delta[c1][d0] + delta[a1][c1]*delta[c0][d0] + delta[a1][d0]*delta[c0][c1]))

                        -PA_0*(delta[a1][b0]*(delta[b1][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[b1][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[b1][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[b1][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))) + delta[a1][b1]*(delta[b0][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[b0][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[b0][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[b0][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))) + delta[b0][b1]*(delta[a1][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[a1][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[a1][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[a1][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0)))) - PA_1*(delta[a0][b0]*(delta[b1][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[b1][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[b1][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[b1][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))) + delta[a0][b1]*(delta[b0][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[b0][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[b0][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[b0][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))) + delta[b0][b1]*(delta[a0][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[a0][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[a0][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[a0][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0)))) - PB_0*(delta[a0][a1]*(delta[b1][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[b1][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[b1][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[b1][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))) + delta[a0][b1]*(delta[a1][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[a1][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[a1][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[a1][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))) + delta[a1][b1]*(delta[a0][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[a0][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[a0][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[a0][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0)))) - PB_1*(delta[a0][a1]*(delta[b0][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[b0][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[b0][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[b0][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))) + delta[a0][b0]*(delta[a1][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[a1][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[a1][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[a1][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))) + delta[a1][b0]*(delta[a0][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[a0][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[a0][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[a0][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))))

                        +(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])*(PQ[c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + PQ[c1]*(QC_0*delta[d0][d1] + delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0)) + PQ[d0]*(QC_0*delta[c1][d1] + QC_1*delta[c0][d1] + delta[c0][c1]*(PQ[d1] + QD_1)) + PQ[d1]*(QC_0*delta[c1][d0] + QC_1*delta[c0][d0] + QD_0*delta[c0][c1]))

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * dd_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDD2_FP64(double*         mat_J,
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
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ double   delta[3][3];

    __shared__ double a_i, a_j, r_i[3], r_j[3], S_ij_00, S1, inv_S1;
    __shared__ double PA_0, PA_1, PB_0, PB_1;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta[0][0] = 1.0; delta[0][1] = 0.0; delta[0][2] = 0.0;
        delta[1][0] = 0.0; delta[1][1] = 1.0; delta[1][2] = 0.0;
        delta[2][0] = 0.0; delta[2][1] = 0.0; delta[2][2] = 1.0;

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
            inv_S1 = 1.0 / S1;

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

    for (uint32_t m = 0; m < prec_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= dd_prim_pair_count))
        {
            break;
        }

        const auto k = dd_first_inds[kl];
        const auto l = dd_second_inds[kl];

        const auto a_k = d_prim_info[k / 6 + d_prim_count * 0];

        const double r_k[3] = {d_prim_info[k / 6 + d_prim_count * 2],
                               d_prim_info[k / 6 + d_prim_count * 3],
                               d_prim_info[k / 6 + d_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = dd_pair_data[kl];

        const auto c0 = d_cart_inds[k % 6][0];
        const auto c1 = d_cart_inds[k % 6][1];
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F8_t[3];

        gpu::computeBoysFunction(F8_t, S1 * S2 * inv_S4 * r2_PQ, 2, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QC_1 = (a_l * inv_S2) * (r_l[c1] - r_k[c1]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                    + F8_t[2] * 0.25 * S1 * S1 * inv_S2 * inv_S2 * inv_S4 * inv_S4 * (

                        +PA_0*PA_1*PB_0*PB_1*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])

                    )

                    + F8_t[2] * 0.25 * S1 * inv_S2 * inv_S4 * inv_S4 * (

                        -2*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])

                        -PA_0*PA_1*(PB_0*(delta[b1][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[b1][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[b1][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[b1][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))) + PB_1*(delta[b0][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[b0][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[b0][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[b0][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0)))) - PB_0*PB_1*(PA_0*(delta[a1][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[a1][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[a1][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[a1][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))) + PA_1*(delta[a0][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[a0][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[a0][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[a0][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))))

                        +(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1])*(PQ[c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + PQ[c1]*(QC_0*delta[d0][d1] + delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0)) + PQ[d0]*(QC_0*delta[c1][d1] + QC_1*delta[c0][d1] + delta[c0][c1]*(PQ[d1] + QD_1)) + PQ[d1]*(QC_0*delta[c1][d0] + QC_1*delta[c0][d0] + QD_0*delta[c0][c1]))

                    )

                    + F8_t[2] * 0.25 * S2 * S2 * inv_S1 * inv_S1 * inv_S4 * inv_S4 * (

                        +QC_0*QC_1*QD_0*QD_1*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * dd_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDD3_FP64(double*         mat_J,
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
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ double   delta[3][3];

    __shared__ double a_i, a_j, r_i[3], r_j[3], S_ij_00, S1, inv_S1;
    __shared__ double PA_0, PA_1, PB_0, PB_1;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta[0][0] = 1.0; delta[0][1] = 0.0; delta[0][2] = 0.0;
        delta[1][0] = 0.0; delta[1][1] = 1.0; delta[1][2] = 0.0;
        delta[2][0] = 0.0; delta[2][1] = 0.0; delta[2][2] = 1.0;

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
            inv_S1 = 1.0 / S1;

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

    for (uint32_t m = 0; m < prec_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= dd_prim_pair_count))
        {
            break;
        }

        const auto k = dd_first_inds[kl];
        const auto l = dd_second_inds[kl];

        const auto a_k = d_prim_info[k / 6 + d_prim_count * 0];

        const double r_k[3] = {d_prim_info[k / 6 + d_prim_count * 2],
                               d_prim_info[k / 6 + d_prim_count * 3],
                               d_prim_info[k / 6 + d_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = dd_pair_data[kl];

        const auto c0 = d_cart_inds[k % 6][0];
        const auto c1 = d_cart_inds[k % 6][1];
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F8_t[3];

        gpu::computeBoysFunction(F8_t, S1 * S2 * inv_S4 * r2_PQ, 2, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QC_1 = (a_l * inv_S2) * (r_l[c1] - r_k[c1]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                    + F8_t[2] * 0.25 * S2 * inv_S1 * inv_S4 * inv_S4 * (

                        +(QC_0*(QC_1*delta[d0][d1] + QD_0*delta[c1][d1] + QD_1*delta[c1][d0]) + QC_1*(QD_0*delta[c0][d1] + QD_1*delta[c0][d0]) + QD_0*QD_1*delta[c0][c1])*(-PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) - PA_1*(PQ[a0]*delta[b0][b1] + PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) - PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) - PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]) + PQ[a0]*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PQ[a1]*(PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PQ[b0]*PQ[b1]*delta[a0][a1])

                        +(-PA_0*QC_0*QC_1*QD_0 + PQ[a0]*QC_0*QC_1*QD_0)*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + (-PA_0*QC_0*QC_1*QD_1 + PQ[a0]*QC_0*QC_1*QD_1)*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + (-PA_0*QC_0*QD_0*QD_1 + PQ[a0]*QC_0*QD_0*QD_1)*(delta[a1][b0]*delta[b1][c1] + delta[a1][b1]*delta[b0][c1] + delta[a1][c1]*delta[b0][b1]) + (-PA_0*QC_1*QD_0*QD_1 + PQ[a0]*QC_1*QD_0*QD_1)*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1]) + (-PA_1*QC_0*QC_1*QD_0 + PQ[a1]*QC_0*QC_1*QD_0)*(delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1]) + (-PA_1*QC_0*QC_1*QD_1 + PQ[a1]*QC_0*QC_1*QD_1)*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + (-PA_1*QC_0*QD_0*QD_1 + PQ[a1]*QC_0*QD_0*QD_1)*(delta[a0][b0]*delta[b1][c1] + delta[a0][b1]*delta[b0][c1] + delta[a0][c1]*delta[b0][b1]) + (-PA_1*QC_1*QD_0*QD_1 + PQ[a1]*QC_1*QD_0*QD_1)*(delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1]) + (-PB_0*QC_0*QC_1*QD_0 + PQ[b0]*QC_0*QC_1*QD_0)*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + (-PB_0*QC_0*QC_1*QD_1 + PQ[b0]*QC_0*QC_1*QD_1)*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + (-PB_0*QC_0*QD_0*QD_1 + PQ[b0]*QC_0*QD_0*QD_1)*(delta[a0][a1]*delta[b1][c1] + delta[a0][b1]*delta[a1][c1] + delta[a0][c1]*delta[a1][b1]) + (-PB_0*QC_1*QD_0*QD_1 + PQ[b0]*QC_1*QD_0*QD_1)*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]) + (-PB_1*QC_0*QC_1*QD_0 + PQ[b1]*QC_0*QC_1*QD_0)*(delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0]) + (-PB_1*QC_0*QC_1*QD_1 + PQ[b1]*QC_0*QC_1*QD_1)*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]) + (-PB_1*QC_0*QD_0*QD_1 + PQ[b1]*QC_0*QD_0*QD_1)*(delta[a0][a1]*delta[b0][c1] + delta[a0][b0]*delta[a1][c1] + delta[a0][c1]*delta[a1][b0]) + (-PB_1*QC_1*QD_0*QD_1 + PQ[b1]*QC_1*QD_0*QD_1)*(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0])

                        +2.0*(QC_0*QD_1*(PQ[c1]*QD_0 + PQ[d0]*QC_1) + QC_1*QD_0*(PQ[c0]*QD_1 + PQ[d1]*QC_0))*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])

                    )

                    + F8_t[2] * 0.25 * inv_S4 * inv_S4 * (

                        +(delta[a0][c0]*delta[c1][d0] + delta[a0][c1]*delta[c0][d0] + delta[a0][d0]*delta[c0][c1])*(PA_1*PB_0*PQ[b1]*QD_1 + PA_1*PB_1*PQ[b0]*QD_1 + PB_0*PB_1*PQ[a1]*QD_1) + (delta[a0][c0]*delta[c1][d1] + delta[a0][c1]*delta[c0][d1] + delta[a0][d1]*delta[c0][c1])*(PA_1*PB_0*PQ[b1]*QD_0 + PA_1*PB_1*PQ[b0]*QD_0 + PB_0*PB_1*PQ[a1]*QD_0) + (delta[a0][c0]*delta[d0][d1] + delta[a0][d0]*delta[c0][d1] + delta[a0][d1]*delta[c0][d0])*(PA_1*PB_0*PQ[b1]*QC_1 + PA_1*PB_1*PQ[b0]*QC_1 + PB_0*PB_1*PQ[a1]*QC_1) + (delta[a0][c1]*delta[d0][d1] + delta[a0][d0]*delta[c1][d1] + delta[a0][d1]*delta[c1][d0])*(PA_1*PB_0*PQ[b1]*QC_0 + PA_1*PB_1*PQ[b0]*QC_0 + PB_0*PB_1*PQ[a1]*QC_0) + (delta[a1][c0]*delta[c1][d0] + delta[a1][c1]*delta[c0][d0] + delta[a1][d0]*delta[c0][c1])*(PA_0*PB_0*PQ[b1]*QD_1 + PA_0*PB_1*PQ[b0]*QD_1 + PB_0*PB_1*PQ[a0]*QD_1) + (delta[a1][c0]*delta[c1][d1] + delta[a1][c1]*delta[c0][d1] + delta[a1][d1]*delta[c0][c1])*(PA_0*PB_0*PQ[b1]*QD_0 + PA_0*PB_1*PQ[b0]*QD_0 + PB_0*PB_1*PQ[a0]*QD_0) + (delta[a1][c0]*delta[d0][d1] + delta[a1][d0]*delta[c0][d1] + delta[a1][d1]*delta[c0][d0])*(PA_0*PB_0*PQ[b1]*QC_1 + PA_0*PB_1*PQ[b0]*QC_1 + PB_0*PB_1*PQ[a0]*QC_1) + (delta[a1][c1]*delta[d0][d1] + delta[a1][d0]*delta[c1][d1] + delta[a1][d1]*delta[c1][d0])*(PA_0*PB_0*PQ[b1]*QC_0 + PA_0*PB_1*PQ[b0]*QC_0 + PB_0*PB_1*PQ[a0]*QC_0) + (delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1])*(PA_0*PA_1*PQ[b1]*QD_1 + PA_0*PB_1*PQ[a1]*QD_1 + PA_1*PB_1*PQ[a0]*QD_1) + (delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1])*(PA_0*PA_1*PQ[b1]*QD_0 + PA_0*PB_1*PQ[a1]*QD_0 + PA_1*PB_1*PQ[a0]*QD_0) + (delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0])*(PA_0*PA_1*PQ[b1]*QC_1 + PA_0*PB_1*PQ[a1]*QC_1 + PA_1*PB_1*PQ[a0]*QC_1) + (delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0])*(PA_0*PA_1*PQ[b1]*QC_0 + PA_0*PB_1*PQ[a1]*QC_0 + PA_1*PB_1*PQ[a0]*QC_0) + (delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])*(PA_0*PA_1*PQ[b0]*QD_1 + PA_0*PB_0*PQ[a1]*QD_1 + PA_1*PB_0*PQ[a0]*QD_1) + (delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1])*(PA_0*PA_1*PQ[b0]*QD_0 + PA_0*PB_0*PQ[a1]*QD_0 + PA_1*PB_0*PQ[a0]*QD_0) + (delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0])*(PA_0*PA_1*PQ[b0]*QC_1 + PA_0*PB_0*PQ[a1]*QC_1 + PA_1*PB_0*PQ[a0]*QC_1) + (delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0])*(PA_0*PA_1*PQ[b0]*QC_0 + PA_0*PB_0*PQ[a1]*QC_0 + PA_1*PB_0*PQ[a0]*QC_0)

                        +(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0] - PQ[a1]*delta[b0][b1] - PQ[b0]*delta[a1][b1] - PQ[b1]*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0] - PQ[a0]*delta[b0][b1] - PQ[b0]*delta[a0][b1] - PQ[b1]*delta[a0][b0]) + PB_0*(PB_1*delta[a0][a1] - PQ[a0]*delta[a1][b1] - PQ[a1]*delta[a0][b1] - PQ[b1]*delta[a0][a1]) - PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]))*(QC_0*(PQ[c1]*delta[d0][d1] + PQ[d0]*delta[c1][d1] + PQ[d1]*delta[c1][d0]) + QC_1*(PQ[d0]*delta[c0][d1] + PQ[d1]*delta[c0][d0] + delta[d0][d1]*(PQ[c0] + QC_0)) + QD_0*(PQ[d1]*delta[c0][c1] + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + QD_1*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0)))

                        +PA_0*(QC_0*(PA_1*(QC_1*(delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + QD_0*(delta[b0][c1]*delta[b1][d1] + delta[b0][d1]*delta[b1][c1]) + QD_1*(delta[b0][c1]*delta[b1][d0] + delta[b0][d0]*delta[b1][c1])) + PB_0*(QC_1*(delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + QD_0*(delta[a1][c1]*delta[b1][d1] + delta[a1][d1]*delta[b1][c1]) + QD_1*(delta[a1][c1]*delta[b1][d0] + delta[a1][d0]*delta[b1][c1])) + PB_1*(QC_1*(delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + QD_0*(delta[a1][c1]*delta[b0][d1] + delta[a1][d1]*delta[b0][c1]) + QD_1*(delta[a1][c1]*delta[b0][d0] + delta[a1][d0]*delta[b0][c1]))) + QC_1*(QD_0*(PA_1*(delta[b0][c0]*delta[b1][d1] + delta[b0][d1]*delta[b1][c0]) + PB_0*(delta[a1][c0]*delta[b1][d1] + delta[a1][d1]*delta[b1][c0]) + PB_1*(delta[a1][c0]*delta[b0][d1] + delta[a1][d1]*delta[b0][c0])) + QD_1*(PA_1*(delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0]) + PB_0*(delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + PB_1*(delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0])))) + PA_1*(PB_0*(QC_0*(QC_1*(delta[a0][d0]*delta[b1][d1] + delta[a0][d1]*delta[b1][d0]) + QD_0*(delta[a0][c1]*delta[b1][d1] + delta[a0][d1]*delta[b1][c1]) + QD_1*(delta[a0][c1]*delta[b1][d0] + delta[a0][d0]*delta[b1][c1])) + QC_1*(QD_0*(delta[a0][c0]*delta[b1][d1] + delta[a0][d1]*delta[b1][c0]) + QD_1*(delta[a0][c0]*delta[b1][d0] + delta[a0][d0]*delta[b1][c0]))) + PB_1*(QC_0*(QC_1*(delta[a0][d0]*delta[b0][d1] + delta[a0][d1]*delta[b0][d0]) + QD_0*(delta[a0][c1]*delta[b0][d1] + delta[a0][d1]*delta[b0][c1]) + QD_1*(delta[a0][c1]*delta[b0][d0] + delta[a0][d0]*delta[b0][c1])) + QC_1*(QD_0*(delta[a0][c0]*delta[b0][d1] + delta[a0][d1]*delta[b0][c0]) + QD_1*(delta[a0][c0]*delta[b0][d0] + delta[a0][d0]*delta[b0][c0])))) + PB_0*PB_1*(QC_0*(QC_1*(delta[a0][d0]*delta[a1][d1] + delta[a0][d1]*delta[a1][d0]) + QD_0*(delta[a0][c1]*delta[a1][d1] + delta[a0][d1]*delta[a1][c1]) + QD_1*(delta[a0][c1]*delta[a1][d0] + delta[a0][d0]*delta[a1][c1])) + QC_1*(QD_0*(delta[a0][c0]*delta[a1][d1] + delta[a0][d1]*delta[a1][c0]) + QD_1*(delta[a0][c0]*delta[a1][d0] + delta[a0][d0]*delta[a1][c0]))) + QD_0*QD_1*(PA_0*(PA_1*(delta[b0][c0]*delta[b1][c1] + delta[b0][c1]*delta[b1][c0]) + PB_0*(delta[a1][c0]*delta[b1][c1] + delta[a1][c1]*delta[b1][c0]) + PB_1*(delta[a1][c0]*delta[b0][c1] + delta[a1][c1]*delta[b0][c0])) + PA_1*(PB_0*(delta[a0][c0]*delta[b1][c1] + delta[a0][c1]*delta[b1][c0]) + PB_1*(delta[a0][c0]*delta[b0][c1] + delta[a0][c1]*delta[b0][c0])) + PB_0*PB_1*(delta[a0][c0]*delta[a1][c1] + delta[a0][c1]*delta[a1][c0]))

                        +(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])*(PQ[c0]*QD_0*(PQ[c1]*QD_1 + PQ[d1]*QC_1) + PQ[c1]*QC_0*(PQ[d0]*QD_1 + PQ[d1]*QD_0) + PQ[d0]*QC_1*(PQ[c0]*QD_1 + PQ[d1]*QC_0)) + (delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))

                        +(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0])*(-PB_1*PQ[c1]*QD_0*QD_1 - PB_1*PQ[d0]*QC_1*QD_1 - PB_1*PQ[d1]*QC_1*QD_0) + (delta[a0][a1]*delta[b0][c1] + delta[a0][b0]*delta[a1][c1] + delta[a0][c1]*delta[a1][b0])*(-PB_1*PQ[c0]*QD_0*QD_1 - PB_1*PQ[d0]*QC_0*QD_1 - PB_1*PQ[d1]*QC_0*QD_0) + (delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])*(-PB_1*PQ[c0]*QC_1*QD_1 - PB_1*PQ[c1]*QC_0*QD_1 - PB_1*PQ[d1]*QC_0*QC_1) + (delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0])*(-PB_1*PQ[c0]*QC_1*QD_0 - PB_1*PQ[c1]*QC_0*QD_0 - PB_1*PQ[d0]*QC_0*QC_1) + (delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1])*(-PB_0*PQ[c1]*QD_0*QD_1 - PB_0*PQ[d0]*QC_1*QD_1 - PB_0*PQ[d1]*QC_1*QD_0) + (delta[a0][a1]*delta[b1][c1] + delta[a0][b1]*delta[a1][c1] + delta[a0][c1]*delta[a1][b1])*(-PB_0*PQ[c0]*QD_0*QD_1 - PB_0*PQ[d0]*QC_0*QD_1 - PB_0*PQ[d1]*QC_0*QD_0) + (delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1])*(-PB_0*PQ[c0]*QC_1*QD_1 - PB_0*PQ[c1]*QC_0*QD_1 - PB_0*PQ[d1]*QC_0*QC_1) + (delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1])*(-PB_0*PQ[c0]*QC_1*QD_0 - PB_0*PQ[c1]*QC_0*QD_0 - PB_0*PQ[d0]*QC_0*QC_1) + (delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1])*(-PA_1*PQ[c1]*QD_0*QD_1 - PA_1*PQ[d0]*QC_1*QD_1 - PA_1*PQ[d1]*QC_1*QD_0) + (delta[a0][b0]*delta[b1][c1] + delta[a0][b1]*delta[b0][c1] + delta[a0][c1]*delta[b0][b1])*(-PA_1*PQ[c0]*QD_0*QD_1 - PA_1*PQ[d0]*QC_0*QD_1 - PA_1*PQ[d1]*QC_0*QD_0) + (delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1])*(-PA_1*PQ[c0]*QC_1*QD_1 - PA_1*PQ[c1]*QC_0*QD_1 - PA_1*PQ[d1]*QC_0*QC_1) + (delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1])*(-PA_1*PQ[c0]*QC_1*QD_0 - PA_1*PQ[c1]*QC_0*QD_0 - PA_1*PQ[d0]*QC_0*QC_1) + (delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1])*(-PA_0*PQ[c1]*QD_0*QD_1 - PA_0*PQ[d0]*QC_1*QD_1 - PA_0*PQ[d1]*QC_1*QD_0) + (delta[a1][b0]*delta[b1][c1] + delta[a1][b1]*delta[b0][c1] + delta[a1][c1]*delta[b0][b1])*(-PA_0*PQ[c0]*QD_0*QD_1 - PA_0*PQ[d0]*QC_0*QD_1 - PA_0*PQ[d1]*QC_0*QD_0) + (delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1])*(-PA_0*PQ[c0]*QC_1*QD_1 - PA_0*PQ[c1]*QC_0*QD_1 - PA_0*PQ[d1]*QC_0*QC_1) + (delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1])*(-PA_0*PQ[c0]*QC_1*QD_0 - PA_0*PQ[c1]*QC_0*QD_0 - PA_0*PQ[d0]*QC_0*QC_1)

                    )

                    + F8_t[2] * 0.5 * S1 * S1 * inv_S2 * inv_S4 * inv_S4 * (

                        +PA_0*PA_1*PB_0*PB_1*(PQ[c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + PQ[c1]*(QC_0*delta[d0][d1] + delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0)) + PQ[d0]*(QC_0*delta[c1][d1] + QC_1*delta[c0][d1] + delta[c0][c1]*(PQ[d1] + QD_1)) + PQ[d1]*(QC_0*delta[c1][d0] + QC_1*delta[c0][d0] + QD_0*delta[c0][c1]))

                    )

                    + F8_t[2] * 0.5 * S1 * inv_S4 * inv_S4 * (

                        -(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))*(QC_0*(PQ[c1]*delta[d0][d1] + PQ[d0]*delta[c1][d1] + PQ[d1]*delta[c1][d0]) + QC_1*(PQ[d0]*delta[c0][d1] + PQ[d1]*delta[c0][d0] + delta[d0][d1]*(PQ[c0] + QC_0)) + QD_0*(PQ[d1]*delta[c0][c1] + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + QD_1*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0)))

                        -PA_0*PA_1*(PB_0*(QC_0*(PQ[c1]*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + PQ[d0]*(QC_1*delta[b1][d1] + QD_1*delta[b1][c1]) + PQ[d1]*(QC_1*delta[b1][d0] + QD_0*delta[b1][c1])) + QC_1*(PQ[c0]*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + delta[b1][c0]*(PQ[d0]*QD_1 + PQ[d1]*QD_0))) + PB_1*(QC_0*(PQ[c1]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + PQ[d0]*(QC_1*delta[b0][d1] + QD_1*delta[b0][c1]) + PQ[d1]*(QC_1*delta[b0][d0] + QD_0*delta[b0][c1])) + QC_1*(PQ[c0]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + delta[b0][c0]*(PQ[d0]*QD_1 + PQ[d1]*QD_0))) + QD_0*QD_1*(PB_0*(PQ[c0]*delta[b1][c1] + PQ[c1]*delta[b1][c0]) + PB_1*(PQ[c0]*delta[b0][c1] + PQ[c1]*delta[b0][c0]))) - PB_0*PB_1*(PA_0*(QC_0*(PQ[c1]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]) + PQ[d0]*(QC_1*delta[a1][d1] + QD_1*delta[a1][c1]) + PQ[d1]*(QC_1*delta[a1][d0] + QD_0*delta[a1][c1])) + QC_1*(PQ[c0]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]) + delta[a1][c0]*(PQ[d0]*QD_1 + PQ[d1]*QD_0))) + PA_1*(QC_0*(PQ[c1]*(QD_0*delta[a0][d1] + QD_1*delta[a0][d0]) + PQ[d0]*(QC_1*delta[a0][d1] + QD_1*delta[a0][c1]) + PQ[d1]*(QC_1*delta[a0][d0] + QD_0*delta[a0][c1])) + QC_1*(PQ[c0]*(QD_0*delta[a0][d1] + QD_1*delta[a0][d0]) + delta[a0][c0]*(PQ[d0]*QD_1 + PQ[d1]*QD_0))) + QD_0*QD_1*(PA_0*(PQ[c0]*delta[a1][c1] + PQ[c1]*delta[a1][c0]) + PA_1*(PQ[c0]*delta[a0][c1] + PQ[c1]*delta[a0][c0])))

                        +(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1])*(PQ[c0]*QD_0*(PQ[c1]*QD_1 + PQ[d1]*QC_1) + PQ[c1]*QC_0*(PQ[d0]*QD_1 + PQ[d1]*QD_0) + PQ[d0]*QC_1*(PQ[c0]*QD_1 + PQ[d1]*QC_0))

                    )

                    + F8_t[2] * 0.5 * S2 * S2 * inv_S1 * inv_S4 * inv_S4 * (

                        +QC_0*QC_1*QD_0*QD_1*(-PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) - PA_1*(PQ[a0]*delta[b0][b1] + PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) - PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) - PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]) + PQ[a0]*PQ[a1]*delta[b0][b1] + PQ[a0]*PQ[b0]*delta[a1][b1] + PQ[a0]*PQ[b1]*delta[a1][b0] + PQ[a1]*PQ[b0]*delta[a0][b1] + PQ[a1]*PQ[b1]*delta[a0][b0] + PQ[b0]*PQ[b1]*delta[a0][a1])

                    )

                    + F8_t[2] * 0.5 * S2 * inv_S4 * inv_S4 * (

                        +(QC_0*(QC_1*delta[d0][d1] + QD_0*delta[c1][d1] + QD_1*delta[c1][d0]) + QC_1*(QD_0*delta[c0][d1] + QD_1*delta[c0][d0]) + QD_0*QD_1*delta[c0][c1])*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))

                        +QC_0*QC_1*(PA_0*(PA_1*(PQ[b0]*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + PQ[b1]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0])) + PB_0*(PQ[a1]*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + PQ[b1]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0])) + PB_1*(PQ[a1]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + PQ[b0]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]))) + PA_1*(PB_0*(PQ[a0]*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + PQ[b1]*(QD_0*delta[a0][d1] + QD_1*delta[a0][d0])) + PB_1*(PQ[a0]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + PQ[b0]*(QD_0*delta[a0][d1] + QD_1*delta[a0][d0]))) + PB_0*PB_1*(PQ[a0]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]) + PQ[a1]*(QD_0*delta[a0][d1] + QD_1*delta[a0][d0]))) + QD_0*QD_1*(PA_0*(PA_1*(PQ[b0]*(QC_0*delta[b1][c1] + QC_1*delta[b1][c0]) + PQ[b1]*(QC_0*delta[b0][c1] + QC_1*delta[b0][c0])) + PB_0*(PQ[a1]*(QC_0*delta[b1][c1] + QC_1*delta[b1][c0]) + PQ[b1]*(QC_0*delta[a1][c1] + QC_1*delta[a1][c0])) + PB_1*(PQ[a1]*(QC_0*delta[b0][c1] + QC_1*delta[b0][c0]) + PQ[b0]*(QC_0*delta[a1][c1] + QC_1*delta[a1][c0]))) + PA_1*(PB_0*(PQ[a0]*(QC_0*delta[b1][c1] + QC_1*delta[b1][c0]) + PQ[b1]*(QC_0*delta[a0][c1] + QC_1*delta[a0][c0])) + PB_1*(PQ[a0]*(QC_0*delta[b0][c1] + QC_1*delta[b0][c0]) + PQ[b0]*(QC_0*delta[a0][c1] + QC_1*delta[a0][c0]))) + PB_0*PB_1*(PQ[a0]*(QC_0*delta[a1][c1] + QC_1*delta[a1][c0]) + PQ[a1]*(QC_0*delta[a0][c1] + QC_1*delta[a0][c0])))

                        +(QC_0*QD_1*(PQ[c1]*QD_0 + PQ[d0]*QC_1) + QC_1*QD_0*(PQ[c0]*QD_1 + PQ[d1]*QC_0))*(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) - PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) - PA_1*(PQ[a0]*delta[b0][b1] + PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1] - PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) - PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]))

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * dd_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDD4_FP64(double*         mat_J,
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
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ double   delta[3][3];

    __shared__ double a_i, a_j, r_i[3], r_j[3], S_ij_00, S1, inv_S1;
    __shared__ double PA_0, PA_1, PB_0, PB_1;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta[0][0] = 1.0; delta[0][1] = 0.0; delta[0][2] = 0.0;
        delta[1][0] = 0.0; delta[1][1] = 1.0; delta[1][2] = 0.0;
        delta[2][0] = 0.0; delta[2][1] = 0.0; delta[2][2] = 1.0;

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
            inv_S1 = 1.0 / S1;

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

    for (uint32_t m = 0; m < prec_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= dd_prim_pair_count))
        {
            break;
        }

        const auto k = dd_first_inds[kl];
        const auto l = dd_second_inds[kl];

        const auto a_k = d_prim_info[k / 6 + d_prim_count * 0];

        const double r_k[3] = {d_prim_info[k / 6 + d_prim_count * 2],
                               d_prim_info[k / 6 + d_prim_count * 3],
                               d_prim_info[k / 6 + d_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = dd_pair_data[kl];

        const auto c0 = d_cart_inds[k % 6][0];
        const auto c1 = d_cart_inds[k % 6][1];
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F8_t[3];

        gpu::computeBoysFunction(F8_t, S1 * S2 * inv_S4 * r2_PQ, 2, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QC_1 = (a_l * inv_S2) * (r_l[c1] - r_k[c1]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                    + F8_t[2] * S1 * S1 * inv_S4 * inv_S4 * (

                        +PA_0*PA_1*PB_0*PB_1*(PQ[c0]*QD_0*(PQ[c1]*QD_1 + PQ[d1]*QC_1) + PQ[c1]*QC_0*(PQ[d0]*QD_1 + PQ[d1]*QD_0) + PQ[d0]*QC_1*(PQ[c0]*QD_1 + PQ[d1]*QC_0))

                    )

                    + F8_t[2] * S1 * S2 * inv_S4 * inv_S4 * (

                        -(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))*(QC_0*QD_1*(PQ[c1]*QD_0 + PQ[d0]*QC_1) + QC_1*QD_0*(PQ[c0]*QD_1 + PQ[d1]*QC_0))

                    )

                    + F8_t[2] * S2 * S2 * inv_S4 * inv_S4 * (

                        +QC_0*QC_1*QD_0*QD_1*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))

                    )

                    + F8_t[2] * 0.0625 * inv_S1 * inv_S1 * inv_S4 * inv_S4 * (

                        +(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])

                    )

                    + F8_t[2] * 0.0625 * inv_S1 * inv_S2 * inv_S4 * inv_S4 * (

                        +4.0*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])

                        +delta[a0][a1]*(delta[b0][c0]*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + delta[b0][c1]*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + delta[b0][d0]*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + delta[b0][d1]*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) + delta[a0][b0]*(delta[a1][c0]*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + delta[a1][c1]*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + delta[a1][d0]*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + delta[a1][d1]*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) + delta[a0][b1]*(delta[a1][c0]*(delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0]) + delta[a1][c1]*(delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0]) + delta[a1][d0]*(delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1]) + delta[a1][d1]*(delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1])) + delta[a1][b0]*(delta[a0][c0]*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + delta[a0][c1]*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + delta[a0][d0]*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + delta[a0][d1]*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) + delta[a1][b1]*(delta[a0][c0]*(delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0]) + delta[a0][c1]*(delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0]) + delta[a0][d0]*(delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1]) + delta[a0][d1]*(delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1])) + delta[b0][b1]*(delta[a0][c0]*(delta[a1][c1]*delta[d0][d1] + delta[a1][d0]*delta[c1][d1] + delta[a1][d1]*delta[c1][d0]) + delta[a0][c1]*(delta[a1][c0]*delta[d0][d1] + delta[a1][d0]*delta[c0][d1] + delta[a1][d1]*delta[c0][d0]) + delta[a0][d0]*(delta[a1][c0]*delta[c1][d1] + delta[a1][c1]*delta[c0][d1] + delta[a1][d1]*delta[c0][c1]) + delta[a0][d1]*(delta[a1][c0]*delta[c1][d0] + delta[a1][c1]*delta[c0][d0] + delta[a1][d0]*delta[c0][c1]))

                    )

                    + F8_t[2] * 0.0625 * inv_S2 * inv_S2 * inv_S4 * inv_S4 * (

                        +(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * dd_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDD5_FP64(double*         mat_J,
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
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ double   delta[3][3];

    __shared__ double a_i, a_j, r_i[3], r_j[3], S_ij_00, S1, inv_S1;
    __shared__ double PA_0, PA_1, PB_0, PB_1;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta[0][0] = 1.0; delta[0][1] = 0.0; delta[0][2] = 0.0;
        delta[1][0] = 0.0; delta[1][1] = 1.0; delta[1][2] = 0.0;
        delta[2][0] = 0.0; delta[2][1] = 0.0; delta[2][2] = 1.0;

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
            inv_S1 = 1.0 / S1;

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

    for (uint32_t m = 0; m < prec_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= dd_prim_pair_count))
        {
            break;
        }

        const auto k = dd_first_inds[kl];
        const auto l = dd_second_inds[kl];

        const auto a_k = d_prim_info[k / 6 + d_prim_count * 0];

        const double r_k[3] = {d_prim_info[k / 6 + d_prim_count * 2],
                               d_prim_info[k / 6 + d_prim_count * 3],
                               d_prim_info[k / 6 + d_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = dd_pair_data[kl];

        const auto c0 = d_cart_inds[k % 6][0];
        const auto c1 = d_cart_inds[k % 6][1];
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F8_t[4];

        gpu::computeBoysFunction(F8_t, S1 * S2 * inv_S4 * r2_PQ, 3, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QC_1 = (a_l * inv_S2) * (r_l[c1] - r_k[c1]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                    + F8_t[3] * 0.0625 * inv_S1 * inv_S4 * inv_S4 * inv_S4 * (

                        -2*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])

                        -delta[a0][a1]*(delta[b0][c0]*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + delta[b0][c1]*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + delta[b0][d0]*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + delta[b0][d1]*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) - delta[a0][b0]*(delta[a1][c0]*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + delta[a1][c1]*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + delta[a1][d0]*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + delta[a1][d1]*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) - delta[a0][b1]*(delta[a1][c0]*(delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0]) + delta[a1][c1]*(delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0]) + delta[a1][d0]*(delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1]) + delta[a1][d1]*(delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1])) - delta[a1][b0]*(delta[a0][c0]*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + delta[a0][c1]*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + delta[a0][d0]*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + delta[a0][d1]*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) - delta[a1][b1]*(delta[a0][c0]*(delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0]) + delta[a0][c1]*(delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0]) + delta[a0][d0]*(delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1]) + delta[a0][d1]*(delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1])) - delta[b0][b1]*(delta[a0][c0]*(delta[a1][c1]*delta[d0][d1] + delta[a1][d0]*delta[c1][d1] + delta[a1][d1]*delta[c1][d0]) + delta[a0][c1]*(delta[a1][c0]*delta[d0][d1] + delta[a1][d0]*delta[c0][d1] + delta[a1][d1]*delta[c0][d0]) + delta[a0][d0]*(delta[a1][c0]*delta[c1][d1] + delta[a1][c1]*delta[c0][d1] + delta[a1][d1]*delta[c0][c1]) + delta[a0][d1]*(delta[a1][c0]*delta[c1][d0] + delta[a1][c1]*delta[c0][d0] + delta[a1][d0]*delta[c0][c1]))

                    )

                    + F8_t[3] * 0.0625 * inv_S2 * inv_S4 * inv_S4 * inv_S4 * (

                        -2*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])

                        -delta[a0][a1]*(delta[b0][c0]*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + delta[b0][c1]*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + delta[b0][d0]*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + delta[b0][d1]*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) - delta[a0][b0]*(delta[a1][c0]*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + delta[a1][c1]*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + delta[a1][d0]*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + delta[a1][d1]*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) - delta[a0][b1]*(delta[a1][c0]*(delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0]) + delta[a1][c1]*(delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0]) + delta[a1][d0]*(delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1]) + delta[a1][d1]*(delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1])) - delta[a1][b0]*(delta[a0][c0]*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + delta[a0][c1]*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + delta[a0][d0]*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + delta[a0][d1]*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) - delta[a1][b1]*(delta[a0][c0]*(delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0]) + delta[a0][c1]*(delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0]) + delta[a0][d0]*(delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1]) + delta[a0][d1]*(delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1])) - delta[b0][b1]*(delta[a0][c0]*(delta[a1][c1]*delta[d0][d1] + delta[a1][d0]*delta[c1][d1] + delta[a1][d1]*delta[c1][d0]) + delta[a0][c1]*(delta[a1][c0]*delta[d0][d1] + delta[a1][d0]*delta[c0][d1] + delta[a1][d1]*delta[c0][d0]) + delta[a0][d0]*(delta[a1][c0]*delta[c1][d1] + delta[a1][c1]*delta[c0][d1] + delta[a1][d1]*delta[c0][c1]) + delta[a0][d1]*(delta[a1][c0]*delta[c1][d0] + delta[a1][c1]*delta[c0][d0] + delta[a1][d0]*delta[c0][c1]))

                    )

                    + F8_t[3] * 0.125 * S1 * inv_S2 * inv_S4 * inv_S4 * inv_S4 * (

                        +(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])*(PA_0*(-PA_1*delta[b0][b1] - PB_0*delta[a1][b1] - PB_1*delta[a1][b0] + PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PA_1*(PQ[a0]*delta[b0][b1] + delta[a0][b0]*(-PB_1 + PQ[b1]) + delta[a0][b1]*(-PB_0 + PQ[b0])) + PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + delta[a0][a1]*(-PB_1 + PQ[b1])) + PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]))

                        -PA_0*(PA_1*(delta[b0][c0]*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + delta[b0][c1]*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + delta[b0][d0]*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + delta[b0][d1]*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) + PB_0*(delta[a1][c0]*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + delta[a1][c1]*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + delta[a1][d0]*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + delta[a1][d1]*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) + PB_1*(delta[a1][c0]*(delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0]) + delta[a1][c1]*(delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0]) + delta[a1][d0]*(delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1]) + delta[a1][d1]*(delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1]))) - PA_1*(PB_0*(delta[a0][c0]*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + delta[a0][c1]*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + delta[a0][d0]*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + delta[a0][d1]*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) + PB_1*(delta[a0][c0]*(delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0]) + delta[a0][c1]*(delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0]) + delta[a0][d0]*(delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1]) + delta[a0][d1]*(delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1]))) - PB_0*PB_1*(delta[a0][c0]*(delta[a1][c1]*delta[d0][d1] + delta[a1][d0]*delta[c1][d1] + delta[a1][d1]*delta[c1][d0]) + delta[a0][c1]*(delta[a1][c0]*delta[d0][d1] + delta[a1][d0]*delta[c0][d1] + delta[a1][d1]*delta[c0][d0]) + delta[a0][d0]*(delta[a1][c0]*delta[c1][d1] + delta[a1][c1]*delta[c0][d1] + delta[a1][d1]*delta[c0][c1]) + delta[a0][d1]*(delta[a1][c0]*delta[c1][d0] + delta[a1][c1]*delta[c0][d0] + delta[a1][d0]*delta[c0][c1]))

                        +PA_0*(delta[a1][b0]*(PQ[c0]*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + PQ[c1]*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + PQ[d0]*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + PQ[d1]*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) + delta[a1][b1]*(PQ[c0]*(delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0]) + PQ[c1]*(delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0]) + PQ[d0]*(delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1]) + PQ[d1]*(delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1])) + delta[b0][b1]*(PQ[c0]*(delta[a1][c1]*delta[d0][d1] + delta[a1][d0]*delta[c1][d1] + delta[a1][d1]*delta[c1][d0]) + PQ[c1]*(delta[a1][c0]*delta[d0][d1] + delta[a1][d0]*delta[c0][d1] + delta[a1][d1]*delta[c0][d0]) + PQ[d0]*(delta[a1][c0]*delta[c1][d1] + delta[a1][c1]*delta[c0][d1] + delta[a1][d1]*delta[c0][c1]) + PQ[d1]*(delta[a1][c0]*delta[c1][d0] + delta[a1][c1]*delta[c0][d0] + delta[a1][d0]*delta[c0][c1]))) + PA_1*(delta[a0][b0]*(PQ[c0]*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + PQ[c1]*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + PQ[d0]*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + PQ[d1]*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) + delta[a0][b1]*(PQ[c0]*(delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0]) + PQ[c1]*(delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0]) + PQ[d0]*(delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1]) + PQ[d1]*(delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1])) + delta[b0][b1]*(PQ[c0]*(delta[a0][c1]*delta[d0][d1] + delta[a0][d0]*delta[c1][d1] + delta[a0][d1]*delta[c1][d0]) + PQ[c1]*(delta[a0][c0]*delta[d0][d1] + delta[a0][d0]*delta[c0][d1] + delta[a0][d1]*delta[c0][d0]) + PQ[d0]*(delta[a0][c0]*delta[c1][d1] + delta[a0][c1]*delta[c0][d1] + delta[a0][d1]*delta[c0][c1]) + PQ[d1]*(delta[a0][c0]*delta[c1][d0] + delta[a0][c1]*delta[c0][d0] + delta[a0][d0]*delta[c0][c1]))) + PB_0*(delta[a0][a1]*(PQ[c0]*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + PQ[c1]*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + PQ[d0]*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + PQ[d1]*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) + delta[a0][b1]*(PQ[c0]*(delta[a1][c1]*delta[d0][d1] + delta[a1][d0]*delta[c1][d1] + delta[a1][d1]*delta[c1][d0]) + PQ[c1]*(delta[a1][c0]*delta[d0][d1] + delta[a1][d0]*delta[c0][d1] + delta[a1][d1]*delta[c0][d0]) + PQ[d0]*(delta[a1][c0]*delta[c1][d1] + delta[a1][c1]*delta[c0][d1] + delta[a1][d1]*delta[c0][c1]) + PQ[d1]*(delta[a1][c0]*delta[c1][d0] + delta[a1][c1]*delta[c0][d0] + delta[a1][d0]*delta[c0][c1])) + delta[a1][b1]*(PQ[c0]*(delta[a0][c1]*delta[d0][d1] + delta[a0][d0]*delta[c1][d1] + delta[a0][d1]*delta[c1][d0]) + PQ[c1]*(delta[a0][c0]*delta[d0][d1] + delta[a0][d0]*delta[c0][d1] + delta[a0][d1]*delta[c0][d0]) + PQ[d0]*(delta[a0][c0]*delta[c1][d1] + delta[a0][c1]*delta[c0][d1] + delta[a0][d1]*delta[c0][c1]) + PQ[d1]*(delta[a0][c0]*delta[c1][d0] + delta[a0][c1]*delta[c0][d0] + delta[a0][d0]*delta[c0][c1]))) + PB_1*(delta[a0][a1]*(PQ[c0]*(delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0]) + PQ[c1]*(delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0]) + PQ[d0]*(delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1]) + PQ[d1]*(delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1])) + delta[a0][b0]*(PQ[c0]*(delta[a1][c1]*delta[d0][d1] + delta[a1][d0]*delta[c1][d1] + delta[a1][d1]*delta[c1][d0]) + PQ[c1]*(delta[a1][c0]*delta[d0][d1] + delta[a1][d0]*delta[c0][d1] + delta[a1][d1]*delta[c0][d0]) + PQ[d0]*(delta[a1][c0]*delta[c1][d1] + delta[a1][c1]*delta[c0][d1] + delta[a1][d1]*delta[c0][c1]) + PQ[d1]*(delta[a1][c0]*delta[c1][d0] + delta[a1][c1]*delta[c0][d0] + delta[a1][d0]*delta[c0][c1])) + delta[a1][b0]*(PQ[c0]*(delta[a0][c1]*delta[d0][d1] + delta[a0][d0]*delta[c1][d1] + delta[a0][d1]*delta[c1][d0]) + PQ[c1]*(delta[a0][c0]*delta[d0][d1] + delta[a0][d0]*delta[c0][d1] + delta[a0][d1]*delta[c0][d0]) + PQ[d0]*(delta[a0][c0]*delta[c1][d1] + delta[a0][c1]*delta[c0][d1] + delta[a0][d1]*delta[c0][c1]) + PQ[d1]*(delta[a0][c0]*delta[c1][d0] + delta[a0][c1]*delta[c0][d0] + delta[a0][d0]*delta[c0][c1])))

                        -(PQ[c0]*(PQ[c1]*delta[d0][d1] + PQ[d0]*delta[c1][d1] + PQ[d1]*delta[c1][d0]) + PQ[c1]*(PQ[d0]*delta[c0][d1] + PQ[d1]*delta[c0][d0]) + PQ[d0]*PQ[d1]*delta[c0][c1])*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * dd_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDD6_FP64(double*         mat_J,
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
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ double   delta[3][3];

    __shared__ double a_i, a_j, r_i[3], r_j[3], S_ij_00, S1, inv_S1;
    __shared__ double PA_0, PA_1, PB_0, PB_1;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta[0][0] = 1.0; delta[0][1] = 0.0; delta[0][2] = 0.0;
        delta[1][0] = 0.0; delta[1][1] = 1.0; delta[1][2] = 0.0;
        delta[2][0] = 0.0; delta[2][1] = 0.0; delta[2][2] = 1.0;

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
            inv_S1 = 1.0 / S1;

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

    for (uint32_t m = 0; m < prec_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= dd_prim_pair_count))
        {
            break;
        }

        const auto k = dd_first_inds[kl];
        const auto l = dd_second_inds[kl];

        const auto a_k = d_prim_info[k / 6 + d_prim_count * 0];

        const double r_k[3] = {d_prim_info[k / 6 + d_prim_count * 2],
                               d_prim_info[k / 6 + d_prim_count * 3],
                               d_prim_info[k / 6 + d_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = dd_pair_data[kl];

        const auto c0 = d_cart_inds[k % 6][0];
        const auto c1 = d_cart_inds[k % 6][1];
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F8_t[4];

        gpu::computeBoysFunction(F8_t, S1 * S2 * inv_S4 * r2_PQ, 3, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QC_1 = (a_l * inv_S2) * (r_l[c1] - r_k[c1]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                    + F8_t[3] * (-0.125) * S2 * inv_S1 * inv_S4 * inv_S4 * inv_S4 * (

                        +(PQ[a0]*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PQ[a1]*(PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PQ[b0]*PQ[b1]*delta[a0][a1])*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])

                        +PQ[a0]*(delta[a1][b0]*(QC_0*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + QC_1*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + QD_0*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + QD_1*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) + delta[a1][b1]*(QC_0*(delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0]) + QC_1*(delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0]) + QD_0*(delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1]) + QD_1*(delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1])) + delta[b0][b1]*(QC_0*(delta[a1][c1]*delta[d0][d1] + delta[a1][d0]*delta[c1][d1] + delta[a1][d1]*delta[c1][d0]) + QC_1*(delta[a1][c0]*delta[d0][d1] + delta[a1][d0]*delta[c0][d1] + delta[a1][d1]*delta[c0][d0]) + QD_0*(delta[a1][c0]*delta[c1][d1] + delta[a1][c1]*delta[c0][d1] + delta[a1][d1]*delta[c0][c1]) + QD_1*(delta[a1][c0]*delta[c1][d0] + delta[a1][c1]*delta[c0][d0] + delta[a1][d0]*delta[c0][c1]))) + PQ[a1]*(delta[a0][b0]*(QC_0*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + QC_1*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + QD_0*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + QD_1*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) + delta[a0][b1]*(QC_0*(delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0]) + QC_1*(delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0]) + QD_0*(delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1]) + QD_1*(delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1])) + delta[b0][b1]*(QC_0*(delta[a0][c1]*delta[d0][d1] + delta[a0][d0]*delta[c1][d1] + delta[a0][d1]*delta[c1][d0]) + QC_1*(delta[a0][c0]*delta[d0][d1] + delta[a0][d0]*delta[c0][d1] + delta[a0][d1]*delta[c0][d0]) + QD_0*(delta[a0][c0]*delta[c1][d1] + delta[a0][c1]*delta[c0][d1] + delta[a0][d1]*delta[c0][c1]) + QD_1*(delta[a0][c0]*delta[c1][d0] + delta[a0][c1]*delta[c0][d0] + delta[a0][d0]*delta[c0][c1]))) + PQ[b0]*(delta[a0][a1]*(QC_0*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + QC_1*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + QD_0*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + QD_1*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) + delta[a0][b1]*(QC_0*(delta[a1][c1]*delta[d0][d1] + delta[a1][d0]*delta[c1][d1] + delta[a1][d1]*delta[c1][d0]) + QC_1*(delta[a1][c0]*delta[d0][d1] + delta[a1][d0]*delta[c0][d1] + delta[a1][d1]*delta[c0][d0]) + QD_0*(delta[a1][c0]*delta[c1][d1] + delta[a1][c1]*delta[c0][d1] + delta[a1][d1]*delta[c0][c1]) + QD_1*(delta[a1][c0]*delta[c1][d0] + delta[a1][c1]*delta[c0][d0] + delta[a1][d0]*delta[c0][c1])) + delta[a1][b1]*(QC_0*(delta[a0][c1]*delta[d0][d1] + delta[a0][d0]*delta[c1][d1] + delta[a0][d1]*delta[c1][d0]) + QC_1*(delta[a0][c0]*delta[d0][d1] + delta[a0][d0]*delta[c0][d1] + delta[a0][d1]*delta[c0][d0]) + QD_0*(delta[a0][c0]*delta[c1][d1] + delta[a0][c1]*delta[c0][d1] + delta[a0][d1]*delta[c0][c1]) + QD_1*(delta[a0][c0]*delta[c1][d0] + delta[a0][c1]*delta[c0][d0] + delta[a0][d0]*delta[c0][c1]))) + PQ[b1]*(delta[a0][a1]*(QC_0*(delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0]) + QC_1*(delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0]) + QD_0*(delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1]) + QD_1*(delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1])) + delta[a0][b0]*(QC_0*(delta[a1][c1]*delta[d0][d1] + delta[a1][d0]*delta[c1][d1] + delta[a1][d1]*delta[c1][d0]) + QC_1*(delta[a1][c0]*delta[d0][d1] + delta[a1][d0]*delta[c0][d1] + delta[a1][d1]*delta[c0][d0]) + QD_0*(delta[a1][c0]*delta[c1][d1] + delta[a1][c1]*delta[c0][d1] + delta[a1][d1]*delta[c0][c1]) + QD_1*(delta[a1][c0]*delta[c1][d0] + delta[a1][c1]*delta[c0][d0] + delta[a1][d0]*delta[c0][c1])) + delta[a1][b0]*(QC_0*(delta[a0][c1]*delta[d0][d1] + delta[a0][d0]*delta[c1][d1] + delta[a0][d1]*delta[c1][d0]) + QC_1*(delta[a0][c0]*delta[d0][d1] + delta[a0][d0]*delta[c0][d1] + delta[a0][d1]*delta[c0][d0]) + QD_0*(delta[a0][c0]*delta[c1][d1] + delta[a0][c1]*delta[c0][d1] + delta[a0][d1]*delta[c0][c1]) + QD_1*(delta[a0][c0]*delta[c1][d0] + delta[a0][c1]*delta[c0][d0] + delta[a0][d0]*delta[c0][c1])))

                        +(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])*(QC_0*(PQ[c1]*delta[d0][d1] + PQ[d0]*delta[c1][d1] + PQ[d1]*delta[c1][d0]) + QC_1*(PQ[d0]*delta[c0][d1] + PQ[d1]*delta[c0][d0] + delta[d0][d1]*(PQ[c0] + QC_0)) + QD_0*(PQ[d1]*delta[c0][c1] + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + QD_1*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0)))

                        +QC_0*(QC_1*(delta[a0][b0]*(delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + delta[a0][d0]*(delta[a1][b0]*delta[b1][d1] + delta[a1][d1]*delta[b0][b1]) + delta[a0][d1]*(delta[a1][b0]*delta[b1][d0] + delta[a1][d0]*delta[b0][b1]) + delta[b0][d0]*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + delta[b0][d1]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1])) + QD_0*(delta[a0][b0]*(delta[a1][c1]*delta[b1][d1] + delta[a1][d1]*delta[b1][c1]) + delta[a0][c1]*(delta[a1][b0]*delta[b1][d1] + delta[a1][d1]*delta[b0][b1]) + delta[a0][d1]*(delta[a1][b0]*delta[b1][c1] + delta[a1][c1]*delta[b0][b1]) + delta[b0][c1]*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + delta[b0][d1]*(delta[a0][a1]*delta[b1][c1] + delta[a0][b1]*delta[a1][c1] + delta[a0][c1]*delta[a1][b1])) + QD_1*(delta[a0][b0]*(delta[a1][c1]*delta[b1][d0] + delta[a1][d0]*delta[b1][c1]) + delta[a0][c1]*(delta[a1][b0]*delta[b1][d0] + delta[a1][d0]*delta[b0][b1]) + delta[a0][d0]*(delta[a1][b0]*delta[b1][c1] + delta[a1][c1]*delta[b0][b1]) + delta[b0][c1]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + delta[b0][d0]*(delta[a0][a1]*delta[b1][c1] + delta[a0][b1]*delta[a1][c1] + delta[a0][c1]*delta[a1][b1]))) + QC_1*(QD_0*(delta[a0][b0]*(delta[a1][c0]*delta[b1][d1] + delta[a1][d1]*delta[b1][c0]) + delta[a0][c0]*(delta[a1][b0]*delta[b1][d1] + delta[a1][d1]*delta[b0][b1]) + delta[a0][d1]*(delta[a1][b0]*delta[b1][c0] + delta[a1][c0]*delta[b0][b1]) + delta[b0][c0]*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + delta[b0][d1]*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1])) + QD_1*(delta[a0][b0]*(delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + delta[a0][c0]*(delta[a1][b0]*delta[b1][d0] + delta[a1][d0]*delta[b0][b1]) + delta[a0][d0]*(delta[a1][b0]*delta[b1][c0] + delta[a1][c0]*delta[b0][b1]) + delta[b0][c0]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + delta[b0][d0]*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]))) + QD_0*QD_1*(delta[a0][b0]*(delta[a1][c0]*delta[b1][c1] + delta[a1][c1]*delta[b1][c0]) + delta[a0][c0]*(delta[a1][b0]*delta[b1][c1] + delta[a1][c1]*delta[b0][b1]) + delta[a0][c1]*(delta[a1][b0]*delta[b1][c0] + delta[a1][c0]*delta[b0][b1]) + delta[b0][c0]*(delta[a0][a1]*delta[b1][c1] + delta[a0][b1]*delta[a1][c1] + delta[a0][c1]*delta[a1][b1]) + delta[b0][c1]*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]))

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * dd_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDD7_FP64(double*         mat_J,
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
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ double   delta[3][3];

    __shared__ double a_i, a_j, r_i[3], r_j[3], S_ij_00, S1, inv_S1;
    __shared__ double PA_0, PA_1, PB_0, PB_1;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta[0][0] = 1.0; delta[0][1] = 0.0; delta[0][2] = 0.0;
        delta[1][0] = 0.0; delta[1][1] = 1.0; delta[1][2] = 0.0;
        delta[2][0] = 0.0; delta[2][1] = 0.0; delta[2][2] = 1.0;

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
            inv_S1 = 1.0 / S1;

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

    for (uint32_t m = 0; m < prec_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= dd_prim_pair_count))
        {
            break;
        }

        const auto k = dd_first_inds[kl];
        const auto l = dd_second_inds[kl];

        const auto a_k = d_prim_info[k / 6 + d_prim_count * 0];

        const double r_k[3] = {d_prim_info[k / 6 + d_prim_count * 2],
                               d_prim_info[k / 6 + d_prim_count * 3],
                               d_prim_info[k / 6 + d_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = dd_pair_data[kl];

        const auto c0 = d_cart_inds[k % 6][0];
        const auto c1 = d_cart_inds[k % 6][1];
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F8_t[4];

        gpu::computeBoysFunction(F8_t, S1 * S2 * inv_S4 * r2_PQ, 3, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QC_1 = (a_l * inv_S2) * (r_l[c1] - r_k[c1]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                    + F8_t[3] * 0.125 * inv_S4 * inv_S4 * inv_S4 * (

                        +(PA_0*PQ[c0] + PA_0*QC_0 - PQ[a0]*PQ[c0] - PQ[a0]*QC_0)*(delta[a1][b0]*delta[b1][c1]*delta[d0][d1] + delta[a1][b0]*delta[b1][d0]*delta[c1][d1] + delta[a1][b0]*delta[b1][d1]*delta[c1][d0] + delta[a1][b1]*delta[b0][c1]*delta[d0][d1] + delta[a1][b1]*delta[b0][d0]*delta[c1][d1] + delta[a1][b1]*delta[b0][d1]*delta[c1][d0] + delta[a1][c1]*delta[b0][b1]*delta[d0][d1] + delta[a1][d0]*delta[b0][b1]*delta[c1][d1] + delta[a1][d1]*delta[b0][b1]*delta[c1][d0]) + (PA_0*PQ[c1] + PA_0*QC_1 - PQ[a0]*PQ[c1] - PQ[a0]*QC_1)*(delta[a1][b0]*delta[b1][c0]*delta[d0][d1] + delta[a1][b0]*delta[b1][d0]*delta[c0][d1] + delta[a1][b0]*delta[b1][d1]*delta[c0][d0] + delta[a1][b1]*delta[b0][c0]*delta[d0][d1] + delta[a1][b1]*delta[b0][d0]*delta[c0][d1] + delta[a1][b1]*delta[b0][d1]*delta[c0][d0] + delta[a1][c0]*delta[b0][b1]*delta[d0][d1] + delta[a1][d0]*delta[b0][b1]*delta[c0][d1] + delta[a1][d1]*delta[b0][b1]*delta[c0][d0]) + (PA_0*PQ[d0] + PA_0*QD_0 - PQ[a0]*PQ[d0] - PQ[a0]*QD_0)*(delta[a1][b0]*delta[b1][c0]*delta[c1][d1] + delta[a1][b0]*delta[b1][c1]*delta[c0][d1] + delta[a1][b0]*delta[b1][d1]*delta[c0][c1] + delta[a1][b1]*delta[b0][c0]*delta[c1][d1] + delta[a1][b1]*delta[b0][c1]*delta[c0][d1] + delta[a1][b1]*delta[b0][d1]*delta[c0][c1] + delta[a1][c0]*delta[b0][b1]*delta[c1][d1] + delta[a1][c1]*delta[b0][b1]*delta[c0][d1] + delta[a1][d1]*delta[b0][b1]*delta[c0][c1]) + (PA_0*PQ[d1] + PA_0*QD_1 - PQ[a0]*PQ[d1] - PQ[a0]*QD_1)*(delta[a1][b0]*delta[b1][c0]*delta[c1][d0] + delta[a1][b0]*delta[b1][c1]*delta[c0][d0] + delta[a1][b0]*delta[b1][d0]*delta[c0][c1] + delta[a1][b1]*delta[b0][c0]*delta[c1][d0] + delta[a1][b1]*delta[b0][c1]*delta[c0][d0] + delta[a1][b1]*delta[b0][d0]*delta[c0][c1] + delta[a1][c0]*delta[b0][b1]*delta[c1][d0] + delta[a1][c1]*delta[b0][b1]*delta[c0][d0] + delta[a1][d0]*delta[b0][b1]*delta[c0][c1]) + (PA_1*PQ[c0] + PA_1*QC_0 - PQ[a1]*PQ[c0] - PQ[a1]*QC_0)*(delta[a0][b0]*delta[b1][c1]*delta[d0][d1] + delta[a0][b0]*delta[b1][d0]*delta[c1][d1] + delta[a0][b0]*delta[b1][d1]*delta[c1][d0] + delta[a0][b1]*delta[b0][c1]*delta[d0][d1] + delta[a0][b1]*delta[b0][d0]*delta[c1][d1] + delta[a0][b1]*delta[b0][d1]*delta[c1][d0] + delta[a0][c1]*delta[b0][b1]*delta[d0][d1] + delta[a0][d0]*delta[b0][b1]*delta[c1][d1] + delta[a0][d1]*delta[b0][b1]*delta[c1][d0]) + (PA_1*PQ[c1] + PA_1*QC_1 - PQ[a1]*PQ[c1] - PQ[a1]*QC_1)*(delta[a0][b0]*delta[b1][c0]*delta[d0][d1] + delta[a0][b0]*delta[b1][d0]*delta[c0][d1] + delta[a0][b0]*delta[b1][d1]*delta[c0][d0] + delta[a0][b1]*delta[b0][c0]*delta[d0][d1] + delta[a0][b1]*delta[b0][d0]*delta[c0][d1] + delta[a0][b1]*delta[b0][d1]*delta[c0][d0] + delta[a0][c0]*delta[b0][b1]*delta[d0][d1] + delta[a0][d0]*delta[b0][b1]*delta[c0][d1] + delta[a0][d1]*delta[b0][b1]*delta[c0][d0]) + (PA_1*PQ[d0] + PA_1*QD_0 - PQ[a1]*PQ[d0] - PQ[a1]*QD_0)*(delta[a0][b0]*delta[b1][c0]*delta[c1][d1] + delta[a0][b0]*delta[b1][c1]*delta[c0][d1] + delta[a0][b0]*delta[b1][d1]*delta[c0][c1] + delta[a0][b1]*delta[b0][c0]*delta[c1][d1] + delta[a0][b1]*delta[b0][c1]*delta[c0][d1] + delta[a0][b1]*delta[b0][d1]*delta[c0][c1] + delta[a0][c0]*delta[b0][b1]*delta[c1][d1] + delta[a0][c1]*delta[b0][b1]*delta[c0][d1] + delta[a0][d1]*delta[b0][b1]*delta[c0][c1]) + (PA_1*PQ[d1] + PA_1*QD_1 - PQ[a1]*PQ[d1] - PQ[a1]*QD_1)*(delta[a0][b0]*delta[b1][c0]*delta[c1][d0] + delta[a0][b0]*delta[b1][c1]*delta[c0][d0] + delta[a0][b0]*delta[b1][d0]*delta[c0][c1] + delta[a0][b1]*delta[b0][c0]*delta[c1][d0] + delta[a0][b1]*delta[b0][c1]*delta[c0][d0] + delta[a0][b1]*delta[b0][d0]*delta[c0][c1] + delta[a0][c0]*delta[b0][b1]*delta[c1][d0] + delta[a0][c1]*delta[b0][b1]*delta[c0][d0] + delta[a0][d0]*delta[b0][b1]*delta[c0][c1]) + (PB_0*PQ[c0] + PB_0*QC_0 - PQ[b0]*PQ[c0] - PQ[b0]*QC_0)*(delta[a0][a1]*delta[b1][c1]*delta[d0][d1] + delta[a0][a1]*delta[b1][d0]*delta[c1][d1] + delta[a0][a1]*delta[b1][d1]*delta[c1][d0] + delta[a0][b1]*delta[a1][c1]*delta[d0][d1] + delta[a0][b1]*delta[a1][d0]*delta[c1][d1] + delta[a0][b1]*delta[a1][d1]*delta[c1][d0] + delta[a0][c1]*delta[a1][b1]*delta[d0][d1] + delta[a0][d0]*delta[a1][b1]*delta[c1][d1] + delta[a0][d1]*delta[a1][b1]*delta[c1][d0]) + (PB_0*PQ[c1] + PB_0*QC_1 - PQ[b0]*PQ[c1] - PQ[b0]*QC_1)*(delta[a0][a1]*delta[b1][c0]*delta[d0][d1] + delta[a0][a1]*delta[b1][d0]*delta[c0][d1] + delta[a0][a1]*delta[b1][d1]*delta[c0][d0] + delta[a0][b1]*delta[a1][c0]*delta[d0][d1] + delta[a0][b1]*delta[a1][d0]*delta[c0][d1] + delta[a0][b1]*delta[a1][d1]*delta[c0][d0] + delta[a0][c0]*delta[a1][b1]*delta[d0][d1] + delta[a0][d0]*delta[a1][b1]*delta[c0][d1] + delta[a0][d1]*delta[a1][b1]*delta[c0][d0]) + (PB_0*PQ[d0] + PB_0*QD_0 - PQ[b0]*PQ[d0] - PQ[b0]*QD_0)*(delta[a0][a1]*delta[b1][c0]*delta[c1][d1] + delta[a0][a1]*delta[b1][c1]*delta[c0][d1] + delta[a0][a1]*delta[b1][d1]*delta[c0][c1] + delta[a0][b1]*delta[a1][c0]*delta[c1][d1] + delta[a0][b1]*delta[a1][c1]*delta[c0][d1] + delta[a0][b1]*delta[a1][d1]*delta[c0][c1] + delta[a0][c0]*delta[a1][b1]*delta[c1][d1] + delta[a0][c1]*delta[a1][b1]*delta[c0][d1] + delta[a0][d1]*delta[a1][b1]*delta[c0][c1]) + (PB_0*PQ[d1] + PB_0*QD_1 - PQ[b0]*PQ[d1] - PQ[b0]*QD_1)*(delta[a0][a1]*delta[b1][c0]*delta[c1][d0] + delta[a0][a1]*delta[b1][c1]*delta[c0][d0] + delta[a0][a1]*delta[b1][d0]*delta[c0][c1] + delta[a0][b1]*delta[a1][c0]*delta[c1][d0] + delta[a0][b1]*delta[a1][c1]*delta[c0][d0] + delta[a0][b1]*delta[a1][d0]*delta[c0][c1] + delta[a0][c0]*delta[a1][b1]*delta[c1][d0] + delta[a0][c1]*delta[a1][b1]*delta[c0][d0] + delta[a0][d0]*delta[a1][b1]*delta[c0][c1]) + (PB_1*PQ[c0] + PB_1*QC_0 - PQ[b1]*PQ[c0] - PQ[b1]*QC_0)*(delta[a0][a1]*delta[b0][c1]*delta[d0][d1] + delta[a0][a1]*delta[b0][d0]*delta[c1][d1] + delta[a0][a1]*delta[b0][d1]*delta[c1][d0] + delta[a0][b0]*delta[a1][c1]*delta[d0][d1] + delta[a0][b0]*delta[a1][d0]*delta[c1][d1] + delta[a0][b0]*delta[a1][d1]*delta[c1][d0] + delta[a0][c1]*delta[a1][b0]*delta[d0][d1] + delta[a0][d0]*delta[a1][b0]*delta[c1][d1] + delta[a0][d1]*delta[a1][b0]*delta[c1][d0]) + (PB_1*PQ[c1] + PB_1*QC_1 - PQ[b1]*PQ[c1] - PQ[b1]*QC_1)*(delta[a0][a1]*delta[b0][c0]*delta[d0][d1] + delta[a0][a1]*delta[b0][d0]*delta[c0][d1] + delta[a0][a1]*delta[b0][d1]*delta[c0][d0] + delta[a0][b0]*delta[a1][c0]*delta[d0][d1] + delta[a0][b0]*delta[a1][d0]*delta[c0][d1] + delta[a0][b0]*delta[a1][d1]*delta[c0][d0] + delta[a0][c0]*delta[a1][b0]*delta[d0][d1] + delta[a0][d0]*delta[a1][b0]*delta[c0][d1] + delta[a0][d1]*delta[a1][b0]*delta[c0][d0]) + (PB_1*PQ[d0] + PB_1*QD_0 - PQ[b1]*PQ[d0] - PQ[b1]*QD_0)*(delta[a0][a1]*delta[b0][c0]*delta[c1][d1] + delta[a0][a1]*delta[b0][c1]*delta[c0][d1] + delta[a0][a1]*delta[b0][d1]*delta[c0][c1] + delta[a0][b0]*delta[a1][c0]*delta[c1][d1] + delta[a0][b0]*delta[a1][c1]*delta[c0][d1] + delta[a0][b0]*delta[a1][d1]*delta[c0][c1] + delta[a0][c0]*delta[a1][b0]*delta[c1][d1] + delta[a0][c1]*delta[a1][b0]*delta[c0][d1] + delta[a0][d1]*delta[a1][b0]*delta[c0][c1]) + (PB_1*PQ[d1] + PB_1*QD_1 - PQ[b1]*PQ[d1] - PQ[b1]*QD_1)*(delta[a0][a1]*delta[b0][c0]*delta[c1][d0] + delta[a0][a1]*delta[b0][c1]*delta[c0][d0] + delta[a0][a1]*delta[b0][d0]*delta[c0][c1] + delta[a0][b0]*delta[a1][c0]*delta[c1][d0] + delta[a0][b0]*delta[a1][c1]*delta[c0][d0] + delta[a0][b0]*delta[a1][d0]*delta[c0][c1] + delta[a0][c0]*delta[a1][b0]*delta[c1][d0] + delta[a0][c1]*delta[a1][b0]*delta[c0][d0] + delta[a0][d0]*delta[a1][b0]*delta[c0][c1])

                        +PA_0*(QC_0*(delta[a1][c1]*(delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + delta[a1][d0]*(delta[b0][c1]*delta[b1][d1] + delta[b0][d1]*delta[b1][c1]) + delta[a1][d1]*(delta[b0][c1]*delta[b1][d0] + delta[b0][d0]*delta[b1][c1])) + QC_1*(delta[a1][c0]*(delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + delta[a1][d0]*(delta[b0][c0]*delta[b1][d1] + delta[b0][d1]*delta[b1][c0]) + delta[a1][d1]*(delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0])) + QD_0*(delta[a1][c0]*(delta[b0][c1]*delta[b1][d1] + delta[b0][d1]*delta[b1][c1]) + delta[a1][c1]*(delta[b0][c0]*delta[b1][d1] + delta[b0][d1]*delta[b1][c0]) + delta[a1][d1]*(delta[b0][c0]*delta[b1][c1] + delta[b0][c1]*delta[b1][c0])) + QD_1*(delta[a1][c0]*(delta[b0][c1]*delta[b1][d0] + delta[b0][d0]*delta[b1][c1]) + delta[a1][c1]*(delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0]) + delta[a1][d0]*(delta[b0][c0]*delta[b1][c1] + delta[b0][c1]*delta[b1][c0]))) + PA_1*(QC_0*(delta[a0][c1]*(delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + delta[a0][d0]*(delta[b0][c1]*delta[b1][d1] + delta[b0][d1]*delta[b1][c1]) + delta[a0][d1]*(delta[b0][c1]*delta[b1][d0] + delta[b0][d0]*delta[b1][c1])) + QC_1*(delta[a0][c0]*(delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + delta[a0][d0]*(delta[b0][c0]*delta[b1][d1] + delta[b0][d1]*delta[b1][c0]) + delta[a0][d1]*(delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0])) + QD_0*(delta[a0][c0]*(delta[b0][c1]*delta[b1][d1] + delta[b0][d1]*delta[b1][c1]) + delta[a0][c1]*(delta[b0][c0]*delta[b1][d1] + delta[b0][d1]*delta[b1][c0]) + delta[a0][d1]*(delta[b0][c0]*delta[b1][c1] + delta[b0][c1]*delta[b1][c0])) + QD_1*(delta[a0][c0]*(delta[b0][c1]*delta[b1][d0] + delta[b0][d0]*delta[b1][c1]) + delta[a0][c1]*(delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0]) + delta[a0][d0]*(delta[b0][c0]*delta[b1][c1] + delta[b0][c1]*delta[b1][c0]))) + PB_0*(QC_0*(delta[a0][c1]*(delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + delta[a0][d0]*(delta[a1][c1]*delta[b1][d1] + delta[a1][d1]*delta[b1][c1]) + delta[a0][d1]*(delta[a1][c1]*delta[b1][d0] + delta[a1][d0]*delta[b1][c1])) + QC_1*(delta[a0][c0]*(delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + delta[a0][d0]*(delta[a1][c0]*delta[b1][d1] + delta[a1][d1]*delta[b1][c0]) + delta[a0][d1]*(delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0])) + QD_0*(delta[a0][c0]*(delta[a1][c1]*delta[b1][d1] + delta[a1][d1]*delta[b1][c1]) + delta[a0][c1]*(delta[a1][c0]*delta[b1][d1] + delta[a1][d1]*delta[b1][c0]) + delta[a0][d1]*(delta[a1][c0]*delta[b1][c1] + delta[a1][c1]*delta[b1][c0])) + QD_1*(delta[a0][c0]*(delta[a1][c1]*delta[b1][d0] + delta[a1][d0]*delta[b1][c1]) + delta[a0][c1]*(delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + delta[a0][d0]*(delta[a1][c0]*delta[b1][c1] + delta[a1][c1]*delta[b1][c0]))) + PB_1*(QC_0*(delta[a0][c1]*(delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + delta[a0][d0]*(delta[a1][c1]*delta[b0][d1] + delta[a1][d1]*delta[b0][c1]) + delta[a0][d1]*(delta[a1][c1]*delta[b0][d0] + delta[a1][d0]*delta[b0][c1])) + QC_1*(delta[a0][c0]*(delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + delta[a0][d0]*(delta[a1][c0]*delta[b0][d1] + delta[a1][d1]*delta[b0][c0]) + delta[a0][d1]*(delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0])) + QD_0*(delta[a0][c0]*(delta[a1][c1]*delta[b0][d1] + delta[a1][d1]*delta[b0][c1]) + delta[a0][c1]*(delta[a1][c0]*delta[b0][d1] + delta[a1][d1]*delta[b0][c0]) + delta[a0][d1]*(delta[a1][c0]*delta[b0][c1] + delta[a1][c1]*delta[b0][c0])) + QD_1*(delta[a0][c0]*(delta[a1][c1]*delta[b0][d0] + delta[a1][d0]*delta[b0][c1]) + delta[a0][c1]*(delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0]) + delta[a0][d0]*(delta[a1][c0]*delta[b0][c1] + delta[a1][c1]*delta[b0][c0])))

                        +(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])*(2.0*PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + 2.0*PA_1*(PQ[a0]*delta[b0][b1] + PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + 2.0*PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + 2.0*PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]) - 2*PQ[a0]*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) - 2*PQ[a1]*(PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) - 2*PQ[b0]*PQ[b1]*delta[a0][a1])

                        +(PA_0*PQ[a1] + PA_1*PQ[a0])*(delta[b0][c0]*delta[b1][c1]*delta[d0][d1] + delta[b0][c0]*delta[b1][d0]*delta[c1][d1] + delta[b0][c0]*delta[b1][d1]*delta[c1][d0] + delta[b0][c1]*delta[b1][c0]*delta[d0][d1] + delta[b0][c1]*delta[b1][d0]*delta[c0][d1] + delta[b0][c1]*delta[b1][d1]*delta[c0][d0] + delta[b0][d0]*delta[b1][c0]*delta[c1][d1] + delta[b0][d0]*delta[b1][c1]*delta[c0][d1] + delta[b0][d0]*delta[b1][d1]*delta[c0][c1] + delta[b0][d1]*delta[b1][c0]*delta[c1][d0] + delta[b0][d1]*delta[b1][c1]*delta[c0][d0] + delta[b0][d1]*delta[b1][d0]*delta[c0][c1]) + (PA_0*PQ[b0] + PB_0*PQ[a0])*(delta[a1][c0]*delta[b1][c1]*delta[d0][d1] + delta[a1][c0]*delta[b1][d0]*delta[c1][d1] + delta[a1][c0]*delta[b1][d1]*delta[c1][d0] + delta[a1][c1]*delta[b1][c0]*delta[d0][d1] + delta[a1][c1]*delta[b1][d0]*delta[c0][d1] + delta[a1][c1]*delta[b1][d1]*delta[c0][d0] + delta[a1][d0]*delta[b1][c0]*delta[c1][d1] + delta[a1][d0]*delta[b1][c1]*delta[c0][d1] + delta[a1][d0]*delta[b1][d1]*delta[c0][c1] + delta[a1][d1]*delta[b1][c0]*delta[c1][d0] + delta[a1][d1]*delta[b1][c1]*delta[c0][d0] + delta[a1][d1]*delta[b1][d0]*delta[c0][c1]) + (PA_0*PQ[b1] + PB_1*PQ[a0])*(delta[a1][c0]*delta[b0][c1]*delta[d0][d1] + delta[a1][c0]*delta[b0][d0]*delta[c1][d1] + delta[a1][c0]*delta[b0][d1]*delta[c1][d0] + delta[a1][c1]*delta[b0][c0]*delta[d0][d1] + delta[a1][c1]*delta[b0][d0]*delta[c0][d1] + delta[a1][c1]*delta[b0][d1]*delta[c0][d0] + delta[a1][d0]*delta[b0][c0]*delta[c1][d1] + delta[a1][d0]*delta[b0][c1]*delta[c0][d1] + delta[a1][d0]*delta[b0][d1]*delta[c0][c1] + delta[a1][d1]*delta[b0][c0]*delta[c1][d0] + delta[a1][d1]*delta[b0][c1]*delta[c0][d0] + delta[a1][d1]*delta[b0][d0]*delta[c0][c1]) + (PA_1*PQ[b0] + PB_0*PQ[a1])*(delta[a0][c0]*delta[b1][c1]*delta[d0][d1] + delta[a0][c0]*delta[b1][d0]*delta[c1][d1] + delta[a0][c0]*delta[b1][d1]*delta[c1][d0] + delta[a0][c1]*delta[b1][c0]*delta[d0][d1] + delta[a0][c1]*delta[b1][d0]*delta[c0][d1] + delta[a0][c1]*delta[b1][d1]*delta[c0][d0] + delta[a0][d0]*delta[b1][c0]*delta[c1][d1] + delta[a0][d0]*delta[b1][c1]*delta[c0][d1] + delta[a0][d0]*delta[b1][d1]*delta[c0][c1] + delta[a0][d1]*delta[b1][c0]*delta[c1][d0] + delta[a0][d1]*delta[b1][c1]*delta[c0][d0] + delta[a0][d1]*delta[b1][d0]*delta[c0][c1]) + (PA_1*PQ[b1] + PB_1*PQ[a1])*(delta[a0][c0]*delta[b0][c1]*delta[d0][d1] + delta[a0][c0]*delta[b0][d0]*delta[c1][d1] + delta[a0][c0]*delta[b0][d1]*delta[c1][d0] + delta[a0][c1]*delta[b0][c0]*delta[d0][d1] + delta[a0][c1]*delta[b0][d0]*delta[c0][d1] + delta[a0][c1]*delta[b0][d1]*delta[c0][d0] + delta[a0][d0]*delta[b0][c0]*delta[c1][d1] + delta[a0][d0]*delta[b0][c1]*delta[c0][d1] + delta[a0][d0]*delta[b0][d1]*delta[c0][c1] + delta[a0][d1]*delta[b0][c0]*delta[c1][d0] + delta[a0][d1]*delta[b0][c1]*delta[c0][d0] + delta[a0][d1]*delta[b0][d0]*delta[c0][c1]) + (PB_0*PQ[b1] + PB_1*PQ[b0])*(delta[a0][c0]*delta[a1][c1]*delta[d0][d1] + delta[a0][c0]*delta[a1][d0]*delta[c1][d1] + delta[a0][c0]*delta[a1][d1]*delta[c1][d0] + delta[a0][c1]*delta[a1][c0]*delta[d0][d1] + delta[a0][c1]*delta[a1][d0]*delta[c0][d1] + delta[a0][c1]*delta[a1][d1]*delta[c0][d0] + delta[a0][d0]*delta[a1][c0]*delta[c1][d1] + delta[a0][d0]*delta[a1][c1]*delta[c0][d1] + delta[a0][d0]*delta[a1][d1]*delta[c0][c1] + delta[a0][d1]*delta[a1][c0]*delta[c1][d0] + delta[a0][d1]*delta[a1][c1]*delta[c0][d0] + delta[a0][d1]*delta[a1][d0]*delta[c0][c1])

                        -2*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])*(PQ[c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + PQ[c1]*(QC_0*delta[d0][d1] + delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0)) + PQ[d0]*(QC_0*delta[c1][d1] + QC_1*delta[c0][d1] + delta[c0][c1]*(PQ[d1] + QD_1)) + PQ[d1]*(QC_0*delta[c1][d0] + QC_1*delta[c0][d0] + QD_0*delta[c0][c1]))

                        +(-PQ[c0]*QC_1 - PQ[c1]*QC_0)*(delta[a0][a1]*delta[b0][d0]*delta[b1][d1] + delta[a0][a1]*delta[b0][d1]*delta[b1][d0] + delta[a0][b0]*delta[a1][d0]*delta[b1][d1] + delta[a0][b0]*delta[a1][d1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0]*delta[b0][d1] + delta[a0][b1]*delta[a1][d1]*delta[b0][d0] + delta[a0][d0]*delta[a1][b0]*delta[b1][d1] + delta[a0][d0]*delta[a1][b1]*delta[b0][d1] + delta[a0][d0]*delta[a1][d1]*delta[b0][b1] + delta[a0][d1]*delta[a1][b0]*delta[b1][d0] + delta[a0][d1]*delta[a1][b1]*delta[b0][d0] + delta[a0][d1]*delta[a1][d0]*delta[b0][b1]) + (-PQ[c0]*QD_0 - PQ[d0]*QC_0)*(delta[a0][a1]*delta[b0][c1]*delta[b1][d1] + delta[a0][a1]*delta[b0][d1]*delta[b1][c1] + delta[a0][b0]*delta[a1][c1]*delta[b1][d1] + delta[a0][b0]*delta[a1][d1]*delta[b1][c1] + delta[a0][b1]*delta[a1][c1]*delta[b0][d1] + delta[a0][b1]*delta[a1][d1]*delta[b0][c1] + delta[a0][c1]*delta[a1][b0]*delta[b1][d1] + delta[a0][c1]*delta[a1][b1]*delta[b0][d1] + delta[a0][c1]*delta[a1][d1]*delta[b0][b1] + delta[a0][d1]*delta[a1][b0]*delta[b1][c1] + delta[a0][d1]*delta[a1][b1]*delta[b0][c1] + delta[a0][d1]*delta[a1][c1]*delta[b0][b1]) + (-PQ[c0]*QD_1 - PQ[d1]*QC_0)*(delta[a0][a1]*delta[b0][c1]*delta[b1][d0] + delta[a0][a1]*delta[b0][d0]*delta[b1][c1] + delta[a0][b0]*delta[a1][c1]*delta[b1][d0] + delta[a0][b0]*delta[a1][d0]*delta[b1][c1] + delta[a0][b1]*delta[a1][c1]*delta[b0][d0] + delta[a0][b1]*delta[a1][d0]*delta[b0][c1] + delta[a0][c1]*delta[a1][b0]*delta[b1][d0] + delta[a0][c1]*delta[a1][b1]*delta[b0][d0] + delta[a0][c1]*delta[a1][d0]*delta[b0][b1] + delta[a0][d0]*delta[a1][b0]*delta[b1][c1] + delta[a0][d0]*delta[a1][b1]*delta[b0][c1] + delta[a0][d0]*delta[a1][c1]*delta[b0][b1]) + (-PQ[c1]*QD_0 - PQ[d0]*QC_1)*(delta[a0][a1]*delta[b0][c0]*delta[b1][d1] + delta[a0][a1]*delta[b0][d1]*delta[b1][c0] + delta[a0][b0]*delta[a1][c0]*delta[b1][d1] + delta[a0][b0]*delta[a1][d1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0]*delta[b0][d1] + delta[a0][b1]*delta[a1][d1]*delta[b0][c0] + delta[a0][c0]*delta[a1][b0]*delta[b1][d1] + delta[a0][c0]*delta[a1][b1]*delta[b0][d1] + delta[a0][c0]*delta[a1][d1]*delta[b0][b1] + delta[a0][d1]*delta[a1][b0]*delta[b1][c0] + delta[a0][d1]*delta[a1][b1]*delta[b0][c0] + delta[a0][d1]*delta[a1][c0]*delta[b0][b1]) + (-PQ[c1]*QD_1 - PQ[d1]*QC_1)*(delta[a0][a1]*delta[b0][c0]*delta[b1][d0] + delta[a0][a1]*delta[b0][d0]*delta[b1][c0] + delta[a0][b0]*delta[a1][c0]*delta[b1][d0] + delta[a0][b0]*delta[a1][d0]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0]*delta[b0][d0] + delta[a0][b1]*delta[a1][d0]*delta[b0][c0] + delta[a0][c0]*delta[a1][b0]*delta[b1][d0] + delta[a0][c0]*delta[a1][b1]*delta[b0][d0] + delta[a0][c0]*delta[a1][d0]*delta[b0][b1] + delta[a0][d0]*delta[a1][b0]*delta[b1][c0] + delta[a0][d0]*delta[a1][b1]*delta[b0][c0] + delta[a0][d0]*delta[a1][c0]*delta[b0][b1]) + (-PQ[d0]*QD_1 - PQ[d1]*QD_0)*(delta[a0][a1]*delta[b0][c0]*delta[b1][c1] + delta[a0][a1]*delta[b0][c1]*delta[b1][c0] + delta[a0][b0]*delta[a1][c0]*delta[b1][c1] + delta[a0][b0]*delta[a1][c1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0]*delta[b0][c1] + delta[a0][b1]*delta[a1][c1]*delta[b0][c0] + delta[a0][c0]*delta[a1][b0]*delta[b1][c1] + delta[a0][c0]*delta[a1][b1]*delta[b0][c1] + delta[a0][c0]*delta[a1][c1]*delta[b0][b1] + delta[a0][c1]*delta[a1][b0]*delta[b1][c0] + delta[a0][c1]*delta[a1][b1]*delta[b0][c0] + delta[a0][c1]*delta[a1][c0]*delta[b0][b1])

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * dd_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDD8_FP64(double*         mat_J,
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
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ double   delta[3][3];

    __shared__ double a_i, a_j, r_i[3], r_j[3], S_ij_00, S1, inv_S1;
    __shared__ double PA_0, PA_1, PB_0, PB_1;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta[0][0] = 1.0; delta[0][1] = 0.0; delta[0][2] = 0.0;
        delta[1][0] = 0.0; delta[1][1] = 1.0; delta[1][2] = 0.0;
        delta[2][0] = 0.0; delta[2][1] = 0.0; delta[2][2] = 1.0;

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
            inv_S1 = 1.0 / S1;

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

    for (uint32_t m = 0; m < prec_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= dd_prim_pair_count))
        {
            break;
        }

        const auto k = dd_first_inds[kl];
        const auto l = dd_second_inds[kl];

        const auto a_k = d_prim_info[k / 6 + d_prim_count * 0];

        const double r_k[3] = {d_prim_info[k / 6 + d_prim_count * 2],
                               d_prim_info[k / 6 + d_prim_count * 3],
                               d_prim_info[k / 6 + d_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = dd_pair_data[kl];

        const auto c0 = d_cart_inds[k % 6][0];
        const auto c1 = d_cart_inds[k % 6][1];
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F8_t[4];

        gpu::computeBoysFunction(F8_t, S1 * S2 * inv_S4 * r2_PQ, 3, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QC_1 = (a_l * inv_S2) * (r_l[c1] - r_k[c1]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                    + F8_t[3] * 0.25 * S1 * S1 * inv_S2 * inv_S4 * inv_S4 * inv_S4 * (

                        +(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])

                        +PA_0*PA_1*(PB_0*(PQ[c0]*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + PQ[c1]*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + PQ[d0]*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + PQ[d1]*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) + PB_1*(PQ[c0]*(delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0]) + PQ[c1]*(delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0]) + PQ[d0]*(delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1]) + PQ[d1]*(delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1]))) + PB_0*PB_1*(PA_0*(PQ[c0]*(delta[a1][c1]*delta[d0][d1] + delta[a1][d0]*delta[c1][d1] + delta[a1][d1]*delta[c1][d0]) + PQ[c1]*(delta[a1][c0]*delta[d0][d1] + delta[a1][d0]*delta[c0][d1] + delta[a1][d1]*delta[c0][d0]) + PQ[d0]*(delta[a1][c0]*delta[c1][d1] + delta[a1][c1]*delta[c0][d1] + delta[a1][d1]*delta[c0][c1]) + PQ[d1]*(delta[a1][c0]*delta[c1][d0] + delta[a1][c1]*delta[c0][d0] + delta[a1][d0]*delta[c0][c1])) + PA_1*(PQ[c0]*(delta[a0][c1]*delta[d0][d1] + delta[a0][d0]*delta[c1][d1] + delta[a0][d1]*delta[c1][d0]) + PQ[c1]*(delta[a0][c0]*delta[d0][d1] + delta[a0][d0]*delta[c0][d1] + delta[a0][d1]*delta[c0][d0]) + PQ[d0]*(delta[a0][c0]*delta[c1][d1] + delta[a0][c1]*delta[c0][d1] + delta[a0][d1]*delta[c0][c1]) + PQ[d1]*(delta[a0][c0]*delta[c1][d0] + delta[a0][c1]*delta[c0][d0] + delta[a0][d0]*delta[c0][c1])))

                        -(PA_0*PA_1*delta[b0][b1] + PA_0*PB_0*delta[a1][b1] + PA_0*PB_1*delta[a1][b0] + PA_1*PB_0*delta[a0][b1] + PA_1*PB_1*delta[a0][b0] + PB_0*PB_1*delta[a0][a1])*(PQ[c0]*PQ[c1]*delta[d0][d1] + PQ[c0]*PQ[d0]*delta[c1][d1] + PQ[c0]*PQ[d1]*delta[c1][d0] + PQ[c1]*PQ[d0]*delta[c0][d1] + PQ[c1]*PQ[d1]*delta[c0][d0] + PQ[d0]*PQ[d1]*delta[c0][c1])

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * dd_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDD9_FP64(double*         mat_J,
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
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ double   delta[3][3];

    __shared__ double a_i, a_j, r_i[3], r_j[3], S_ij_00, S1, inv_S1;
    __shared__ double PA_0, PA_1, PB_0, PB_1;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta[0][0] = 1.0; delta[0][1] = 0.0; delta[0][2] = 0.0;
        delta[1][0] = 0.0; delta[1][1] = 1.0; delta[1][2] = 0.0;
        delta[2][0] = 0.0; delta[2][1] = 0.0; delta[2][2] = 1.0;

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
            inv_S1 = 1.0 / S1;

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

    for (uint32_t m = 0; m < prec_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= dd_prim_pair_count))
        {
            break;
        }

        const auto k = dd_first_inds[kl];
        const auto l = dd_second_inds[kl];

        const auto a_k = d_prim_info[k / 6 + d_prim_count * 0];

        const double r_k[3] = {d_prim_info[k / 6 + d_prim_count * 2],
                               d_prim_info[k / 6 + d_prim_count * 3],
                               d_prim_info[k / 6 + d_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = dd_pair_data[kl];

        const auto c0 = d_cart_inds[k % 6][0];
        const auto c1 = d_cart_inds[k % 6][1];
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F8_t[4];

        gpu::computeBoysFunction(F8_t, S1 * S2 * inv_S4 * r2_PQ, 3, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QC_1 = (a_l * inv_S2) * (r_l[c1] - r_k[c1]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                    + F8_t[3] * 0.25 * S1 * inv_S4 * inv_S4 * inv_S4 * (

                        -2*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))

                        +(delta[a0][c0]*delta[c1][d0] + delta[a0][c1]*delta[c0][d0] + delta[a0][d0]*delta[c0][c1])*(-PA_1*PB_0*PQ[b1]*PQ[d1] - PA_1*PB_0*PQ[b1]*QD_1 - PA_1*PB_1*PQ[b0]*PQ[d1] - PA_1*PB_1*PQ[b0]*QD_1 - PB_0*PB_1*PQ[a1]*PQ[d1] - PB_0*PB_1*PQ[a1]*QD_1) + (delta[a0][c0]*delta[c1][d1] + delta[a0][c1]*delta[c0][d1] + delta[a0][d1]*delta[c0][c1])*(-PA_1*PB_0*PQ[b1]*PQ[d0] - PA_1*PB_0*PQ[b1]*QD_0 - PA_1*PB_1*PQ[b0]*PQ[d0] - PA_1*PB_1*PQ[b0]*QD_0 - PB_0*PB_1*PQ[a1]*PQ[d0] - PB_0*PB_1*PQ[a1]*QD_0) + (delta[a0][c0]*delta[d0][d1] + delta[a0][d0]*delta[c0][d1] + delta[a0][d1]*delta[c0][d0])*(-PA_1*PB_0*PQ[b1]*PQ[c1] - PA_1*PB_0*PQ[b1]*QC_1 - PA_1*PB_1*PQ[b0]*PQ[c1] - PA_1*PB_1*PQ[b0]*QC_1 - PB_0*PB_1*PQ[a1]*PQ[c1] - PB_0*PB_1*PQ[a1]*QC_1) + (delta[a0][c1]*delta[d0][d1] + delta[a0][d0]*delta[c1][d1] + delta[a0][d1]*delta[c1][d0])*(-PA_1*PB_0*PQ[b1]*PQ[c0] - PA_1*PB_0*PQ[b1]*QC_0 - PA_1*PB_1*PQ[b0]*PQ[c0] - PA_1*PB_1*PQ[b0]*QC_0 - PB_0*PB_1*PQ[a1]*PQ[c0] - PB_0*PB_1*PQ[a1]*QC_0) + (delta[a1][c0]*delta[c1][d0] + delta[a1][c1]*delta[c0][d0] + delta[a1][d0]*delta[c0][c1])*(-PA_0*PB_0*PQ[b1]*PQ[d1] - PA_0*PB_0*PQ[b1]*QD_1 - PA_0*PB_1*PQ[b0]*PQ[d1] - PA_0*PB_1*PQ[b0]*QD_1 - PB_0*PB_1*PQ[a0]*PQ[d1] - PB_0*PB_1*PQ[a0]*QD_1) + (delta[a1][c0]*delta[c1][d1] + delta[a1][c1]*delta[c0][d1] + delta[a1][d1]*delta[c0][c1])*(-PA_0*PB_0*PQ[b1]*PQ[d0] - PA_0*PB_0*PQ[b1]*QD_0 - PA_0*PB_1*PQ[b0]*PQ[d0] - PA_0*PB_1*PQ[b0]*QD_0 - PB_0*PB_1*PQ[a0]*PQ[d0] - PB_0*PB_1*PQ[a0]*QD_0) + (delta[a1][c0]*delta[d0][d1] + delta[a1][d0]*delta[c0][d1] + delta[a1][d1]*delta[c0][d0])*(-PA_0*PB_0*PQ[b1]*PQ[c1] - PA_0*PB_0*PQ[b1]*QC_1 - PA_0*PB_1*PQ[b0]*PQ[c1] - PA_0*PB_1*PQ[b0]*QC_1 - PB_0*PB_1*PQ[a0]*PQ[c1] - PB_0*PB_1*PQ[a0]*QC_1) + (delta[a1][c1]*delta[d0][d1] + delta[a1][d0]*delta[c1][d1] + delta[a1][d1]*delta[c1][d0])*(-PA_0*PB_0*PQ[b1]*PQ[c0] - PA_0*PB_0*PQ[b1]*QC_0 - PA_0*PB_1*PQ[b0]*PQ[c0] - PA_0*PB_1*PQ[b0]*QC_0 - PB_0*PB_1*PQ[a0]*PQ[c0] - PB_0*PB_1*PQ[a0]*QC_0) + (delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1])*(-PA_0*PA_1*PQ[b1]*PQ[d1] - PA_0*PA_1*PQ[b1]*QD_1 - PA_0*PB_1*PQ[a1]*PQ[d1] - PA_0*PB_1*PQ[a1]*QD_1 - PA_1*PB_1*PQ[a0]*PQ[d1] - PA_1*PB_1*PQ[a0]*QD_1) + (delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1])*(-PA_0*PA_1*PQ[b1]*PQ[d0] - PA_0*PA_1*PQ[b1]*QD_0 - PA_0*PB_1*PQ[a1]*PQ[d0] - PA_0*PB_1*PQ[a1]*QD_0 - PA_1*PB_1*PQ[a0]*PQ[d0] - PA_1*PB_1*PQ[a0]*QD_0) + (delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0])*(-PA_0*PA_1*PQ[b1]*PQ[c1] - PA_0*PA_1*PQ[b1]*QC_1 - PA_0*PB_1*PQ[a1]*PQ[c1] - PA_0*PB_1*PQ[a1]*QC_1 - PA_1*PB_1*PQ[a0]*PQ[c1] - PA_1*PB_1*PQ[a0]*QC_1) + (delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0])*(-PA_0*PA_1*PQ[b1]*PQ[c0] - PA_0*PA_1*PQ[b1]*QC_0 - PA_0*PB_1*PQ[a1]*PQ[c0] - PA_0*PB_1*PQ[a1]*QC_0 - PA_1*PB_1*PQ[a0]*PQ[c0] - PA_1*PB_1*PQ[a0]*QC_0) + (delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])*(-PA_0*PA_1*PQ[b0]*PQ[d1] - PA_0*PA_1*PQ[b0]*QD_1 - PA_0*PB_0*PQ[a1]*PQ[d1] - PA_0*PB_0*PQ[a1]*QD_1 - PA_1*PB_0*PQ[a0]*PQ[d1] - PA_1*PB_0*PQ[a0]*QD_1) + (delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1])*(-PA_0*PA_1*PQ[b0]*PQ[d0] - PA_0*PA_1*PQ[b0]*QD_0 - PA_0*PB_0*PQ[a1]*PQ[d0] - PA_0*PB_0*PQ[a1]*QD_0 - PA_1*PB_0*PQ[a0]*PQ[d0] - PA_1*PB_0*PQ[a0]*QD_0) + (delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0])*(-PA_0*PA_1*PQ[b0]*PQ[c1] - PA_0*PA_1*PQ[b0]*QC_1 - PA_0*PB_0*PQ[a1]*PQ[c1] - PA_0*PB_0*PQ[a1]*QC_1 - PA_1*PB_0*PQ[a0]*PQ[c1] - PA_1*PB_0*PQ[a0]*QC_1) + (delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0])*(-PA_0*PA_1*PQ[b0]*PQ[c0] - PA_0*PA_1*PQ[b0]*QC_0 - PA_0*PB_0*PQ[a1]*PQ[c0] - PA_0*PB_0*PQ[a1]*QC_0 - PA_1*PB_0*PQ[a0]*PQ[c0] - PA_1*PB_0*PQ[a0]*QC_0)

                        +(PA_0*(-PA_1*delta[b0][b1] - PB_0*delta[a1][b1] - PB_1*delta[a1][b0] + PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PA_1*(PQ[a0]*delta[b0][b1] + delta[a0][b0]*(-PB_1 + PQ[b1]) + delta[a0][b1]*(-PB_0 + PQ[b0])) + PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + delta[a0][a1]*(-PB_1 + PQ[b1])) + PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]))*(PQ[c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + PQ[c1]*(QC_0*delta[d0][d1] + delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0)) + PQ[d0]*(QC_0*delta[c1][d1] + QC_1*delta[c0][d1] + delta[c0][c1]*(PQ[d1] + QD_1)) + PQ[d1]*(QC_0*delta[c1][d0] + QC_1*delta[c0][d0] + QD_0*delta[c0][c1]))

                        -PA_0*(PA_1*(PQ[c0]*(QC_1*(delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + QD_0*(delta[b0][c1]*delta[b1][d1] + delta[b0][d1]*delta[b1][c1]) + QD_1*(delta[b0][c1]*delta[b1][d0] + delta[b0][d0]*delta[b1][c1])) + PQ[c1]*(QC_0*(delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + QD_0*(delta[b0][c0]*delta[b1][d1] + delta[b0][d1]*delta[b1][c0]) + QD_1*(delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0])) + PQ[d0]*(QC_0*(delta[b0][c1]*delta[b1][d1] + delta[b0][d1]*delta[b1][c1]) + QC_1*(delta[b0][c0]*delta[b1][d1] + delta[b0][d1]*delta[b1][c0]) + QD_1*(delta[b0][c0]*delta[b1][c1] + delta[b0][c1]*delta[b1][c0])) + PQ[d1]*(QC_0*(delta[b0][c1]*delta[b1][d0] + delta[b0][d0]*delta[b1][c1]) + QC_1*(delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0]) + QD_0*(delta[b0][c0]*delta[b1][c1] + delta[b0][c1]*delta[b1][c0]))) + PB_0*(PQ[c0]*(QC_1*(delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + QD_0*(delta[a1][c1]*delta[b1][d1] + delta[a1][d1]*delta[b1][c1]) + QD_1*(delta[a1][c1]*delta[b1][d0] + delta[a1][d0]*delta[b1][c1])) + PQ[c1]*(QC_0*(delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + QD_0*(delta[a1][c0]*delta[b1][d1] + delta[a1][d1]*delta[b1][c0]) + QD_1*(delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0])) + PQ[d0]*(QC_0*(delta[a1][c1]*delta[b1][d1] + delta[a1][d1]*delta[b1][c1]) + QC_1*(delta[a1][c0]*delta[b1][d1] + delta[a1][d1]*delta[b1][c0]) + QD_1*(delta[a1][c0]*delta[b1][c1] + delta[a1][c1]*delta[b1][c0])) + PQ[d1]*(QC_0*(delta[a1][c1]*delta[b1][d0] + delta[a1][d0]*delta[b1][c1]) + QC_1*(delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + QD_0*(delta[a1][c0]*delta[b1][c1] + delta[a1][c1]*delta[b1][c0]))) + PB_1*(PQ[c0]*(QC_1*(delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + QD_0*(delta[a1][c1]*delta[b0][d1] + delta[a1][d1]*delta[b0][c1]) + QD_1*(delta[a1][c1]*delta[b0][d0] + delta[a1][d0]*delta[b0][c1])) + PQ[c1]*(QC_0*(delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + QD_0*(delta[a1][c0]*delta[b0][d1] + delta[a1][d1]*delta[b0][c0]) + QD_1*(delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0])) + PQ[d0]*(QC_0*(delta[a1][c1]*delta[b0][d1] + delta[a1][d1]*delta[b0][c1]) + QC_1*(delta[a1][c0]*delta[b0][d1] + delta[a1][d1]*delta[b0][c0]) + QD_1*(delta[a1][c0]*delta[b0][c1] + delta[a1][c1]*delta[b0][c0])) + PQ[d1]*(QC_0*(delta[a1][c1]*delta[b0][d0] + delta[a1][d0]*delta[b0][c1]) + QC_1*(delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0]) + QD_0*(delta[a1][c0]*delta[b0][c1] + delta[a1][c1]*delta[b0][c0])))) - PA_1*(PB_0*(PQ[c0]*(QC_1*(delta[a0][d0]*delta[b1][d1] + delta[a0][d1]*delta[b1][d0]) + QD_0*(delta[a0][c1]*delta[b1][d1] + delta[a0][d1]*delta[b1][c1]) + QD_1*(delta[a0][c1]*delta[b1][d0] + delta[a0][d0]*delta[b1][c1])) + PQ[c1]*(QC_0*(delta[a0][d0]*delta[b1][d1] + delta[a0][d1]*delta[b1][d0]) + QD_0*(delta[a0][c0]*delta[b1][d1] + delta[a0][d1]*delta[b1][c0]) + QD_1*(delta[a0][c0]*delta[b1][d0] + delta[a0][d0]*delta[b1][c0])) + PQ[d0]*(QC_0*(delta[a0][c1]*delta[b1][d1] + delta[a0][d1]*delta[b1][c1]) + QC_1*(delta[a0][c0]*delta[b1][d1] + delta[a0][d1]*delta[b1][c0]) + QD_1*(delta[a0][c0]*delta[b1][c1] + delta[a0][c1]*delta[b1][c0])) + PQ[d1]*(QC_0*(delta[a0][c1]*delta[b1][d0] + delta[a0][d0]*delta[b1][c1]) + QC_1*(delta[a0][c0]*delta[b1][d0] + delta[a0][d0]*delta[b1][c0]) + QD_0*(delta[a0][c0]*delta[b1][c1] + delta[a0][c1]*delta[b1][c0]))) + PB_1*(PQ[c0]*(QC_1*(delta[a0][d0]*delta[b0][d1] + delta[a0][d1]*delta[b0][d0]) + QD_0*(delta[a0][c1]*delta[b0][d1] + delta[a0][d1]*delta[b0][c1]) + QD_1*(delta[a0][c1]*delta[b0][d0] + delta[a0][d0]*delta[b0][c1])) + PQ[c1]*(QC_0*(delta[a0][d0]*delta[b0][d1] + delta[a0][d1]*delta[b0][d0]) + QD_0*(delta[a0][c0]*delta[b0][d1] + delta[a0][d1]*delta[b0][c0]) + QD_1*(delta[a0][c0]*delta[b0][d0] + delta[a0][d0]*delta[b0][c0])) + PQ[d0]*(QC_0*(delta[a0][c1]*delta[b0][d1] + delta[a0][d1]*delta[b0][c1]) + QC_1*(delta[a0][c0]*delta[b0][d1] + delta[a0][d1]*delta[b0][c0]) + QD_1*(delta[a0][c0]*delta[b0][c1] + delta[a0][c1]*delta[b0][c0])) + PQ[d1]*(QC_0*(delta[a0][c1]*delta[b0][d0] + delta[a0][d0]*delta[b0][c1]) + QC_1*(delta[a0][c0]*delta[b0][d0] + delta[a0][d0]*delta[b0][c0]) + QD_0*(delta[a0][c0]*delta[b0][c1] + delta[a0][c1]*delta[b0][c0])))) - PB_0*PB_1*(PQ[c0]*(QC_1*(delta[a0][d0]*delta[a1][d1] + delta[a0][d1]*delta[a1][d0]) + QD_0*(delta[a0][c1]*delta[a1][d1] + delta[a0][d1]*delta[a1][c1]) + QD_1*(delta[a0][c1]*delta[a1][d0] + delta[a0][d0]*delta[a1][c1])) + PQ[c1]*(QC_0*(delta[a0][d0]*delta[a1][d1] + delta[a0][d1]*delta[a1][d0]) + QD_0*(delta[a0][c0]*delta[a1][d1] + delta[a0][d1]*delta[a1][c0]) + QD_1*(delta[a0][c0]*delta[a1][d0] + delta[a0][d0]*delta[a1][c0])) + PQ[d0]*(QC_0*(delta[a0][c1]*delta[a1][d1] + delta[a0][d1]*delta[a1][c1]) + QC_1*(delta[a0][c0]*delta[a1][d1] + delta[a0][d1]*delta[a1][c0]) + QD_1*(delta[a0][c0]*delta[a1][c1] + delta[a0][c1]*delta[a1][c0])) + PQ[d1]*(QC_0*(delta[a0][c1]*delta[a1][d0] + delta[a0][d0]*delta[a1][c1]) + QC_1*(delta[a0][c0]*delta[a1][d0] + delta[a0][d0]*delta[a1][c0]) + QD_0*(delta[a0][c0]*delta[a1][c1] + delta[a0][c1]*delta[a1][c0])))

                        +(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0])*(PB_1*PQ[c1]*PQ[d0]*QD_1 + PB_1*PQ[c1]*PQ[d1]*QD_0 + PB_1*PQ[d0]*PQ[d1]*QC_1) + (delta[a0][a1]*delta[b0][c1] + delta[a0][b0]*delta[a1][c1] + delta[a0][c1]*delta[a1][b0])*(PB_1*PQ[c0]*PQ[d0]*QD_1 + PB_1*PQ[c0]*PQ[d1]*QD_0 + PB_1*PQ[d0]*PQ[d1]*QC_0) + (delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])*(PB_1*PQ[c0]*PQ[c1]*QD_1 + PB_1*PQ[c0]*PQ[d1]*QC_1 + PB_1*PQ[c1]*PQ[d1]*QC_0) + (delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0])*(PB_1*PQ[c0]*PQ[c1]*QD_0 + PB_1*PQ[c0]*PQ[d0]*QC_1 + PB_1*PQ[c1]*PQ[d0]*QC_0) + (delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1])*(PB_0*PQ[c1]*PQ[d0]*QD_1 + PB_0*PQ[c1]*PQ[d1]*QD_0 + PB_0*PQ[d0]*PQ[d1]*QC_1) + (delta[a0][a1]*delta[b1][c1] + delta[a0][b1]*delta[a1][c1] + delta[a0][c1]*delta[a1][b1])*(PB_0*PQ[c0]*PQ[d0]*QD_1 + PB_0*PQ[c0]*PQ[d1]*QD_0 + PB_0*PQ[d0]*PQ[d1]*QC_0) + (delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1])*(PB_0*PQ[c0]*PQ[c1]*QD_1 + PB_0*PQ[c0]*PQ[d1]*QC_1 + PB_0*PQ[c1]*PQ[d1]*QC_0) + (delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1])*(PB_0*PQ[c0]*PQ[c1]*QD_0 + PB_0*PQ[c0]*PQ[d0]*QC_1 + PB_0*PQ[c1]*PQ[d0]*QC_0) + (delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1])*(PA_1*PQ[c1]*PQ[d0]*QD_1 + PA_1*PQ[c1]*PQ[d1]*QD_0 + PA_1*PQ[d0]*PQ[d1]*QC_1) + (delta[a0][b0]*delta[b1][c1] + delta[a0][b1]*delta[b0][c1] + delta[a0][c1]*delta[b0][b1])*(PA_1*PQ[c0]*PQ[d0]*QD_1 + PA_1*PQ[c0]*PQ[d1]*QD_0 + PA_1*PQ[d0]*PQ[d1]*QC_0) + (delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1])*(PA_1*PQ[c0]*PQ[c1]*QD_1 + PA_1*PQ[c0]*PQ[d1]*QC_1 + PA_1*PQ[c1]*PQ[d1]*QC_0) + (delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1])*(PA_1*PQ[c0]*PQ[c1]*QD_0 + PA_1*PQ[c0]*PQ[d0]*QC_1 + PA_1*PQ[c1]*PQ[d0]*QC_0) + (delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1])*(PA_0*PQ[c1]*PQ[d0]*QD_1 + PA_0*PQ[c1]*PQ[d1]*QD_0 + PA_0*PQ[d0]*PQ[d1]*QC_1) + (delta[a1][b0]*delta[b1][c1] + delta[a1][b1]*delta[b0][c1] + delta[a1][c1]*delta[b0][b1])*(PA_0*PQ[c0]*PQ[d0]*QD_1 + PA_0*PQ[c0]*PQ[d1]*QD_0 + PA_0*PQ[d0]*PQ[d1]*QC_0) + (delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1])*(PA_0*PQ[c0]*PQ[c1]*QD_1 + PA_0*PQ[c0]*PQ[d1]*QC_1 + PA_0*PQ[c1]*PQ[d1]*QC_0) + (delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1])*(PA_0*PQ[c0]*PQ[c1]*QD_0 + PA_0*PQ[c0]*PQ[d0]*QC_1 + PA_0*PQ[c1]*PQ[d0]*QC_0)

                        -(PQ[c0]*PQ[c1]*(PQ[d0]*QD_1 + PQ[d1]*QD_0) + PQ[d0]*PQ[d1]*(PQ[c0]*QC_1 + PQ[c1]*QC_0))*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])

                    )

                    + F8_t[3] * (-0.25) * S2 * S2 * inv_S1 * inv_S4 * inv_S4 * inv_S4 * (

                        +(PQ[a0]*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PQ[a1]*(PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PQ[b0]*PQ[b1]*delta[a0][a1])*(QC_0*(QC_1*delta[d0][d1] + QD_0*delta[c1][d1] + QD_1*delta[c1][d0]) + QC_1*(QD_0*delta[c0][d1] + QD_1*delta[c0][d0]) + QD_0*QD_1*delta[c0][c1])

                        +QC_0*QC_1*(QD_0*(PQ[a0]*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1]) + PQ[b0]*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + PQ[b1]*(delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0])) + QD_1*(PQ[a0]*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + PQ[b0]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + PQ[b1]*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]))) + QD_0*QD_1*(QC_0*(PQ[a0]*(delta[a1][b0]*delta[b1][c1] + delta[a1][b1]*delta[b0][c1] + delta[a1][c1]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][c1] + delta[a0][b1]*delta[b0][c1] + delta[a0][c1]*delta[b0][b1]) + PQ[b0]*(delta[a0][a1]*delta[b1][c1] + delta[a0][b1]*delta[a1][c1] + delta[a0][c1]*delta[a1][b1]) + PQ[b1]*(delta[a0][a1]*delta[b0][c1] + delta[a0][b0]*delta[a1][c1] + delta[a0][c1]*delta[a1][b0])) + QC_1*(PQ[a0]*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1]) + PQ[b0]*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]) + PQ[b1]*(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0])))

                        +(QC_0*QD_1*(PQ[c1]*QD_0 + PQ[d0]*QC_1) + QC_1*QD_0*(PQ[c0]*QD_1 + PQ[d1]*QC_0))*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * dd_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDD10_FP64(double*         mat_J,
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
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ double   delta[3][3];

    __shared__ double a_i, a_j, r_i[3], r_j[3], S_ij_00, S1, inv_S1;
    __shared__ double PA_0, PA_1, PB_0, PB_1;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta[0][0] = 1.0; delta[0][1] = 0.0; delta[0][2] = 0.0;
        delta[1][0] = 0.0; delta[1][1] = 1.0; delta[1][2] = 0.0;
        delta[2][0] = 0.0; delta[2][1] = 0.0; delta[2][2] = 1.0;

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
            inv_S1 = 1.0 / S1;

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

    for (uint32_t m = 0; m < prec_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= dd_prim_pair_count))
        {
            break;
        }

        const auto k = dd_first_inds[kl];
        const auto l = dd_second_inds[kl];

        const auto a_k = d_prim_info[k / 6 + d_prim_count * 0];

        const double r_k[3] = {d_prim_info[k / 6 + d_prim_count * 2],
                               d_prim_info[k / 6 + d_prim_count * 3],
                               d_prim_info[k / 6 + d_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = dd_pair_data[kl];

        const auto c0 = d_cart_inds[k % 6][0];
        const auto c1 = d_cart_inds[k % 6][1];
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F8_t[4];

        gpu::computeBoysFunction(F8_t, S1 * S2 * inv_S4 * r2_PQ, 3, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QC_1 = (a_l * inv_S2) * (r_l[c1] - r_k[c1]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                    + F8_t[3] * 0.25 * S2 * inv_S4 * inv_S4 * inv_S4 * (

                        +(delta[a0][c0]*delta[c1][d0] + delta[a0][c1]*delta[c0][d0] + delta[a0][d0]*delta[c0][c1])*(PA_1*PQ[b0]*PQ[b1]*QD_1 + PB_0*PQ[a1]*PQ[b1]*QD_1 + PB_1*PQ[a1]*PQ[b0]*QD_1) + (delta[a0][c0]*delta[c1][d1] + delta[a0][c1]*delta[c0][d1] + delta[a0][d1]*delta[c0][c1])*(PA_1*PQ[b0]*PQ[b1]*QD_0 + PB_0*PQ[a1]*PQ[b1]*QD_0 + PB_1*PQ[a1]*PQ[b0]*QD_0) + (delta[a0][c0]*delta[d0][d1] + delta[a0][d0]*delta[c0][d1] + delta[a0][d1]*delta[c0][d0])*(PA_1*PQ[b0]*PQ[b1]*QC_1 + PB_0*PQ[a1]*PQ[b1]*QC_1 + PB_1*PQ[a1]*PQ[b0]*QC_1) + (delta[a0][c1]*delta[d0][d1] + delta[a0][d0]*delta[c1][d1] + delta[a0][d1]*delta[c1][d0])*(PA_1*PQ[b0]*PQ[b1]*QC_0 + PB_0*PQ[a1]*PQ[b1]*QC_0 + PB_1*PQ[a1]*PQ[b0]*QC_0) + (delta[a1][c0]*delta[c1][d0] + delta[a1][c1]*delta[c0][d0] + delta[a1][d0]*delta[c0][c1])*(PA_0*PQ[b0]*PQ[b1]*QD_1 + PB_0*PQ[a0]*PQ[b1]*QD_1 + PB_1*PQ[a0]*PQ[b0]*QD_1) + (delta[a1][c0]*delta[c1][d1] + delta[a1][c1]*delta[c0][d1] + delta[a1][d1]*delta[c0][c1])*(PA_0*PQ[b0]*PQ[b1]*QD_0 + PB_0*PQ[a0]*PQ[b1]*QD_0 + PB_1*PQ[a0]*PQ[b0]*QD_0) + (delta[a1][c0]*delta[d0][d1] + delta[a1][d0]*delta[c0][d1] + delta[a1][d1]*delta[c0][d0])*(PA_0*PQ[b0]*PQ[b1]*QC_1 + PB_0*PQ[a0]*PQ[b1]*QC_1 + PB_1*PQ[a0]*PQ[b0]*QC_1) + (delta[a1][c1]*delta[d0][d1] + delta[a1][d0]*delta[c1][d1] + delta[a1][d1]*delta[c1][d0])*(PA_0*PQ[b0]*PQ[b1]*QC_0 + PB_0*PQ[a0]*PQ[b1]*QC_0 + PB_1*PQ[a0]*PQ[b0]*QC_0) + (delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1])*(PA_0*PQ[a1]*PQ[b1]*QD_1 + PA_1*PQ[a0]*PQ[b1]*QD_1 + PB_1*PQ[a0]*PQ[a1]*QD_1) + (delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1])*(PA_0*PQ[a1]*PQ[b1]*QD_0 + PA_1*PQ[a0]*PQ[b1]*QD_0 + PB_1*PQ[a0]*PQ[a1]*QD_0) + (delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0])*(PA_0*PQ[a1]*PQ[b1]*QC_1 + PA_1*PQ[a0]*PQ[b1]*QC_1 + PB_1*PQ[a0]*PQ[a1]*QC_1) + (delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0])*(PA_0*PQ[a1]*PQ[b1]*QC_0 + PA_1*PQ[a0]*PQ[b1]*QC_0 + PB_1*PQ[a0]*PQ[a1]*QC_0) + (delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])*(PA_0*PQ[a1]*PQ[b0]*QD_1 + PA_1*PQ[a0]*PQ[b0]*QD_1 + PB_0*PQ[a0]*PQ[a1]*QD_1) + (delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1])*(PA_0*PQ[a1]*PQ[b0]*QD_0 + PA_1*PQ[a0]*PQ[b0]*QD_0 + PB_0*PQ[a0]*PQ[a1]*QD_0) + (delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0])*(PA_0*PQ[a1]*PQ[b0]*QC_1 + PA_1*PQ[a0]*PQ[b0]*QC_1 + PB_0*PQ[a0]*PQ[a1]*QC_1) + (delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0])*(PA_0*PQ[a1]*PQ[b0]*QC_0 + PA_1*PQ[a0]*PQ[b0]*QC_0 + PB_0*PQ[a0]*PQ[a1]*QC_0)

                        +(QC_0*(PQ[c1]*delta[d0][d1] + PQ[d0]*delta[c1][d1] + PQ[d1]*delta[c1][d0]) + QC_1*(PQ[d0]*delta[c0][d1] + PQ[d1]*delta[c0][d0] + delta[d0][d1]*(PQ[c0] + QC_0)) + QD_0*(PQ[d1]*delta[c0][c1] + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + QD_1*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0)))*(PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PQ[a0]*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0] - PQ[a1]*delta[b0][b1] - PQ[b0]*delta[a1][b1] - PQ[b1]*delta[a1][b0]) + PQ[a1]*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PQ[b0]*(PB_1*delta[a0][a1] + delta[a0][b1]*(PA_1 - PQ[a1])) + PQ[b1]*(delta[a0][a1]*(PB_0 - PQ[b0]) + delta[a0][b0]*(PA_1 - PQ[a1])))

                        +QC_0*(PA_0*(PQ[a1]*(QC_1*(delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + QD_0*(delta[b0][c1]*delta[b1][d1] + delta[b0][d1]*delta[b1][c1]) + QD_1*(delta[b0][c1]*delta[b1][d0] + delta[b0][d0]*delta[b1][c1])) + PQ[b0]*(QC_1*(delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + QD_0*(delta[a1][c1]*delta[b1][d1] + delta[a1][d1]*delta[b1][c1]) + QD_1*(delta[a1][c1]*delta[b1][d0] + delta[a1][d0]*delta[b1][c1])) + PQ[b1]*(QC_1*(delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + QD_0*(delta[a1][c1]*delta[b0][d1] + delta[a1][d1]*delta[b0][c1]) + QD_1*(delta[a1][c1]*delta[b0][d0] + delta[a1][d0]*delta[b0][c1]))) + PA_1*(PQ[a0]*(QC_1*(delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + QD_0*(delta[b0][c1]*delta[b1][d1] + delta[b0][d1]*delta[b1][c1]) + QD_1*(delta[b0][c1]*delta[b1][d0] + delta[b0][d0]*delta[b1][c1])) + PQ[b0]*(QC_1*(delta[a0][d0]*delta[b1][d1] + delta[a0][d1]*delta[b1][d0]) + QD_0*(delta[a0][c1]*delta[b1][d1] + delta[a0][d1]*delta[b1][c1]) + QD_1*(delta[a0][c1]*delta[b1][d0] + delta[a0][d0]*delta[b1][c1])) + PQ[b1]*(QC_1*(delta[a0][d0]*delta[b0][d1] + delta[a0][d1]*delta[b0][d0]) + QD_0*(delta[a0][c1]*delta[b0][d1] + delta[a0][d1]*delta[b0][c1]) + QD_1*(delta[a0][c1]*delta[b0][d0] + delta[a0][d0]*delta[b0][c1]))) + PB_0*(PQ[a0]*(QC_1*(delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + QD_0*(delta[a1][c1]*delta[b1][d1] + delta[a1][d1]*delta[b1][c1]) + QD_1*(delta[a1][c1]*delta[b1][d0] + delta[a1][d0]*delta[b1][c1])) + PQ[a1]*(QC_1*(delta[a0][d0]*delta[b1][d1] + delta[a0][d1]*delta[b1][d0]) + QD_0*(delta[a0][c1]*delta[b1][d1] + delta[a0][d1]*delta[b1][c1]) + QD_1*(delta[a0][c1]*delta[b1][d0] + delta[a0][d0]*delta[b1][c1])) + PQ[b1]*(QC_1*(delta[a0][d0]*delta[a1][d1] + delta[a0][d1]*delta[a1][d0]) + QD_0*(delta[a0][c1]*delta[a1][d1] + delta[a0][d1]*delta[a1][c1]) + QD_1*(delta[a0][c1]*delta[a1][d0] + delta[a0][d0]*delta[a1][c1]))) + PB_1*(PQ[a0]*(QC_1*(delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + QD_0*(delta[a1][c1]*delta[b0][d1] + delta[a1][d1]*delta[b0][c1]) + QD_1*(delta[a1][c1]*delta[b0][d0] + delta[a1][d0]*delta[b0][c1])) + PQ[a1]*(QC_1*(delta[a0][d0]*delta[b0][d1] + delta[a0][d1]*delta[b0][d0]) + QD_0*(delta[a0][c1]*delta[b0][d1] + delta[a0][d1]*delta[b0][c1]) + QD_1*(delta[a0][c1]*delta[b0][d0] + delta[a0][d0]*delta[b0][c1])) + PQ[b0]*(QC_1*(delta[a0][d0]*delta[a1][d1] + delta[a0][d1]*delta[a1][d0]) + QD_0*(delta[a0][c1]*delta[a1][d1] + delta[a0][d1]*delta[a1][c1]) + QD_1*(delta[a0][c1]*delta[a1][d0] + delta[a0][d0]*delta[a1][c1])))) + QC_1*(QD_0*(PA_0*(PQ[a1]*(delta[b0][c0]*delta[b1][d1] + delta[b0][d1]*delta[b1][c0]) + PQ[b0]*(delta[a1][c0]*delta[b1][d1] + delta[a1][d1]*delta[b1][c0]) + PQ[b1]*(delta[a1][c0]*delta[b0][d1] + delta[a1][d1]*delta[b0][c0])) + PA_1*(PQ[a0]*(delta[b0][c0]*delta[b1][d1] + delta[b0][d1]*delta[b1][c0]) + PQ[b0]*(delta[a0][c0]*delta[b1][d1] + delta[a0][d1]*delta[b1][c0]) + PQ[b1]*(delta[a0][c0]*delta[b0][d1] + delta[a0][d1]*delta[b0][c0])) + PB_0*(PQ[a0]*(delta[a1][c0]*delta[b1][d1] + delta[a1][d1]*delta[b1][c0]) + PQ[a1]*(delta[a0][c0]*delta[b1][d1] + delta[a0][d1]*delta[b1][c0]) + PQ[b1]*(delta[a0][c0]*delta[a1][d1] + delta[a0][d1]*delta[a1][c0])) + PB_1*(PQ[a0]*(delta[a1][c0]*delta[b0][d1] + delta[a1][d1]*delta[b0][c0]) + PQ[a1]*(delta[a0][c0]*delta[b0][d1] + delta[a0][d1]*delta[b0][c0]) + PQ[b0]*(delta[a0][c0]*delta[a1][d1] + delta[a0][d1]*delta[a1][c0]))) + QD_1*(PA_0*(PQ[a1]*(delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0]) + PQ[b0]*(delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + PQ[b1]*(delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0])) + PA_1*(PQ[a0]*(delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0]) + PQ[b0]*(delta[a0][c0]*delta[b1][d0] + delta[a0][d0]*delta[b1][c0]) + PQ[b1]*(delta[a0][c0]*delta[b0][d0] + delta[a0][d0]*delta[b0][c0])) + PB_0*(PQ[a0]*(delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + PQ[a1]*(delta[a0][c0]*delta[b1][d0] + delta[a0][d0]*delta[b1][c0]) + PQ[b1]*(delta[a0][c0]*delta[a1][d0] + delta[a0][d0]*delta[a1][c0])) + PB_1*(PQ[a0]*(delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0]) + PQ[a1]*(delta[a0][c0]*delta[b0][d0] + delta[a0][d0]*delta[b0][c0]) + PQ[b0]*(delta[a0][c0]*delta[a1][d0] + delta[a0][d0]*delta[a1][c0])))) + QD_0*QD_1*(PA_0*(PQ[a1]*(delta[b0][c0]*delta[b1][c1] + delta[b0][c1]*delta[b1][c0]) + PQ[b0]*(delta[a1][c0]*delta[b1][c1] + delta[a1][c1]*delta[b1][c0]) + PQ[b1]*(delta[a1][c0]*delta[b0][c1] + delta[a1][c1]*delta[b0][c0])) + PA_1*(PQ[a0]*(delta[b0][c0]*delta[b1][c1] + delta[b0][c1]*delta[b1][c0]) + PQ[b0]*(delta[a0][c0]*delta[b1][c1] + delta[a0][c1]*delta[b1][c0]) + PQ[b1]*(delta[a0][c0]*delta[b0][c1] + delta[a0][c1]*delta[b0][c0])) + PB_0*(PQ[a0]*(delta[a1][c0]*delta[b1][c1] + delta[a1][c1]*delta[b1][c0]) + PQ[a1]*(delta[a0][c0]*delta[b1][c1] + delta[a0][c1]*delta[b1][c0]) + PQ[b1]*(delta[a0][c0]*delta[a1][c1] + delta[a0][c1]*delta[a1][c0])) + PB_1*(PQ[a0]*(delta[a1][c0]*delta[b0][c1] + delta[a1][c1]*delta[b0][c0]) + PQ[a1]*(delta[a0][c0]*delta[b0][c1] + delta[a0][c1]*delta[b0][c0]) + PQ[b0]*(delta[a0][c0]*delta[a1][c1] + delta[a0][c1]*delta[a1][c0])))

                        +(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0])*(PB_1*PQ[c1]*QD_0*QD_1 + PB_1*PQ[d0]*QC_1*QD_1 + PB_1*PQ[d1]*QC_1*QD_0 - PQ[b1]*PQ[c1]*QD_0*QD_1 - PQ[b1]*PQ[d0]*QC_1*QD_1 - PQ[b1]*PQ[d1]*QC_1*QD_0) + (delta[a0][a1]*delta[b0][c1] + delta[a0][b0]*delta[a1][c1] + delta[a0][c1]*delta[a1][b0])*(PB_1*PQ[c0]*QD_0*QD_1 + PB_1*PQ[d0]*QC_0*QD_1 + PB_1*PQ[d1]*QC_0*QD_0 - PQ[b1]*PQ[c0]*QD_0*QD_1 - PQ[b1]*PQ[d0]*QC_0*QD_1 - PQ[b1]*PQ[d1]*QC_0*QD_0) + (delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])*(PB_1*PQ[c0]*QC_1*QD_1 + PB_1*PQ[c1]*QC_0*QD_1 + PB_1*PQ[d1]*QC_0*QC_1 - PQ[b1]*PQ[c0]*QC_1*QD_1 - PQ[b1]*PQ[c1]*QC_0*QD_1 - PQ[b1]*PQ[d1]*QC_0*QC_1) + (delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0])*(PB_1*PQ[c0]*QC_1*QD_0 + PB_1*PQ[c1]*QC_0*QD_0 + PB_1*PQ[d0]*QC_0*QC_1 - PQ[b1]*PQ[c0]*QC_1*QD_0 - PQ[b1]*PQ[c1]*QC_0*QD_0 - PQ[b1]*PQ[d0]*QC_0*QC_1) + (delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1])*(PB_0*PQ[c1]*QD_0*QD_1 + PB_0*PQ[d0]*QC_1*QD_1 + PB_0*PQ[d1]*QC_1*QD_0 - PQ[b0]*PQ[c1]*QD_0*QD_1 - PQ[b0]*PQ[d0]*QC_1*QD_1 - PQ[b0]*PQ[d1]*QC_1*QD_0) + (delta[a0][a1]*delta[b1][c1] + delta[a0][b1]*delta[a1][c1] + delta[a0][c1]*delta[a1][b1])*(PB_0*PQ[c0]*QD_0*QD_1 + PB_0*PQ[d0]*QC_0*QD_1 + PB_0*PQ[d1]*QC_0*QD_0 - PQ[b0]*PQ[c0]*QD_0*QD_1 - PQ[b0]*PQ[d0]*QC_0*QD_1 - PQ[b0]*PQ[d1]*QC_0*QD_0) + (delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1])*(PB_0*PQ[c0]*QC_1*QD_1 + PB_0*PQ[c1]*QC_0*QD_1 + PB_0*PQ[d1]*QC_0*QC_1 - PQ[b0]*PQ[c0]*QC_1*QD_1 - PQ[b0]*PQ[c1]*QC_0*QD_1 - PQ[b0]*PQ[d1]*QC_0*QC_1) + (delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1])*(PB_0*PQ[c0]*QC_1*QD_0 + PB_0*PQ[c1]*QC_0*QD_0 + PB_0*PQ[d0]*QC_0*QC_1 - PQ[b0]*PQ[c0]*QC_1*QD_0 - PQ[b0]*PQ[c1]*QC_0*QD_0 - PQ[b0]*PQ[d0]*QC_0*QC_1) + (delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1])*(PA_1*PQ[c1]*QD_0*QD_1 + PA_1*PQ[d0]*QC_1*QD_1 + PA_1*PQ[d1]*QC_1*QD_0 - PQ[a1]*PQ[c1]*QD_0*QD_1 - PQ[a1]*PQ[d0]*QC_1*QD_1 - PQ[a1]*PQ[d1]*QC_1*QD_0) + (delta[a0][b0]*delta[b1][c1] + delta[a0][b1]*delta[b0][c1] + delta[a0][c1]*delta[b0][b1])*(PA_1*PQ[c0]*QD_0*QD_1 + PA_1*PQ[d0]*QC_0*QD_1 + PA_1*PQ[d1]*QC_0*QD_0 - PQ[a1]*PQ[c0]*QD_0*QD_1 - PQ[a1]*PQ[d0]*QC_0*QD_1 - PQ[a1]*PQ[d1]*QC_0*QD_0) + (delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1])*(PA_1*PQ[c0]*QC_1*QD_1 + PA_1*PQ[c1]*QC_0*QD_1 + PA_1*PQ[d1]*QC_0*QC_1 - PQ[a1]*PQ[c0]*QC_1*QD_1 - PQ[a1]*PQ[c1]*QC_0*QD_1 - PQ[a1]*PQ[d1]*QC_0*QC_1) + (delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1])*(PA_1*PQ[c0]*QC_1*QD_0 + PA_1*PQ[c1]*QC_0*QD_0 + PA_1*PQ[d0]*QC_0*QC_1 - PQ[a1]*PQ[c0]*QC_1*QD_0 - PQ[a1]*PQ[c1]*QC_0*QD_0 - PQ[a1]*PQ[d0]*QC_0*QC_1) + (delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1])*(PA_0*PQ[c1]*QD_0*QD_1 + PA_0*PQ[d0]*QC_1*QD_1 + PA_0*PQ[d1]*QC_1*QD_0 - PQ[a0]*PQ[c1]*QD_0*QD_1 - PQ[a0]*PQ[d0]*QC_1*QD_1 - PQ[a0]*PQ[d1]*QC_1*QD_0) + (delta[a1][b0]*delta[b1][c1] + delta[a1][b1]*delta[b0][c1] + delta[a1][c1]*delta[b0][b1])*(PA_0*PQ[c0]*QD_0*QD_1 + PA_0*PQ[d0]*QC_0*QD_1 + PA_0*PQ[d1]*QC_0*QD_0 - PQ[a0]*PQ[c0]*QD_0*QD_1 - PQ[a0]*PQ[d0]*QC_0*QD_1 - PQ[a0]*PQ[d1]*QC_0*QD_0) + (delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1])*(PA_0*PQ[c0]*QC_1*QD_1 + PA_0*PQ[c1]*QC_0*QD_1 + PA_0*PQ[d1]*QC_0*QC_1 - PQ[a0]*PQ[c0]*QC_1*QD_1 - PQ[a0]*PQ[c1]*QC_0*QD_1 - PQ[a0]*PQ[d1]*QC_0*QC_1) + (delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1])*(PA_0*PQ[c0]*QC_1*QD_0 + PA_0*PQ[c1]*QC_0*QD_0 + PA_0*PQ[d0]*QC_0*QC_1 - PQ[a0]*PQ[c0]*QC_1*QD_0 - PQ[a0]*PQ[c1]*QC_0*QD_0 - PQ[a0]*PQ[d0]*QC_0*QC_1)

                        +(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])

                        -2*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])*(PQ[c0]*QD_0*(PQ[c1]*QD_1 + PQ[d1]*QC_1) + PQ[c1]*QC_0*(PQ[d0]*QD_1 + PQ[d1]*QD_0) + PQ[d0]*QC_1*(PQ[c0]*QD_1 + PQ[d1]*QC_0))

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * dd_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDD11_FP64(double*         mat_J,
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
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ double   delta[3][3];

    __shared__ double a_i, a_j, r_i[3], r_j[3], S_ij_00, S1, inv_S1;
    __shared__ double PA_0, PA_1, PB_0, PB_1;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta[0][0] = 1.0; delta[0][1] = 0.0; delta[0][2] = 0.0;
        delta[1][0] = 0.0; delta[1][1] = 1.0; delta[1][2] = 0.0;
        delta[2][0] = 0.0; delta[2][1] = 0.0; delta[2][2] = 1.0;

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
            inv_S1 = 1.0 / S1;

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

    for (uint32_t m = 0; m < prec_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= dd_prim_pair_count))
        {
            break;
        }

        const auto k = dd_first_inds[kl];
        const auto l = dd_second_inds[kl];

        const auto a_k = d_prim_info[k / 6 + d_prim_count * 0];

        const double r_k[3] = {d_prim_info[k / 6 + d_prim_count * 2],
                               d_prim_info[k / 6 + d_prim_count * 3],
                               d_prim_info[k / 6 + d_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = dd_pair_data[kl];

        const auto c0 = d_cart_inds[k % 6][0];
        const auto c1 = d_cart_inds[k % 6][1];
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F8_t[4];

        gpu::computeBoysFunction(F8_t, S1 * S2 * inv_S4 * r2_PQ, 3, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QC_1 = (a_l * inv_S2) * (r_l[c1] - r_k[c1]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                    + F8_t[3] * (-0.5) * S1 * S1 * S1 * inv_S2 * inv_S4 * inv_S4 * inv_S4 * (

                        +PA_0*PA_1*PB_0*PB_1*(PQ[c0]*(PQ[c1]*delta[d0][d1] + PQ[d0]*delta[c1][d1] + PQ[d1]*delta[c1][d0]) + PQ[c1]*(PQ[d0]*delta[c0][d1] + PQ[d1]*delta[c0][d0]) + PQ[d0]*PQ[d1]*delta[c0][c1])

                    )

                    + F8_t[3] * 0.5 * S1 * S1 * inv_S4 * inv_S4 * inv_S4 * (

                        +(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))*(PQ[c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + PQ[c1]*(QC_0*delta[d0][d1] + delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0)) + PQ[d0]*(QC_0*delta[c1][d1] + QC_1*delta[c0][d1] + delta[c0][c1]*(PQ[d1] + QD_1)) + PQ[d1]*(QC_0*delta[c1][d0] + QC_1*delta[c0][d0] + QD_0*delta[c0][c1]))

                        +PA_0*PA_1*(PB_0*(PQ[c0]*(PQ[c1]*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + PQ[d0]*(QC_1*delta[b1][d1] + QD_1*delta[b1][c1]) + PQ[d1]*(QC_1*delta[b1][d0] + QD_0*delta[b1][c1])) + PQ[c1]*(PQ[d0]*(QC_0*delta[b1][d1] + QD_1*delta[b1][c0]) + PQ[d1]*(QC_0*delta[b1][d0] + QD_0*delta[b1][c0]))) + PB_1*(PQ[c0]*(PQ[c1]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + PQ[d0]*(QC_1*delta[b0][d1] + QD_1*delta[b0][c1]) + PQ[d1]*(QC_1*delta[b0][d0] + QD_0*delta[b0][c1])) + PQ[c1]*(PQ[d0]*(QC_0*delta[b0][d1] + QD_1*delta[b0][c0]) + PQ[d1]*(QC_0*delta[b0][d0] + QD_0*delta[b0][c0]))) + PQ[d0]*PQ[d1]*(PB_0*(QC_0*delta[b1][c1] + QC_1*delta[b1][c0]) + PB_1*(QC_0*delta[b0][c1] + QC_1*delta[b0][c0]))) + PB_0*PB_1*(PA_0*(PQ[c0]*(PQ[c1]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]) + PQ[d0]*(QC_1*delta[a1][d1] + QD_1*delta[a1][c1]) + PQ[d1]*(QC_1*delta[a1][d0] + QD_0*delta[a1][c1])) + PQ[c1]*(PQ[d0]*(QC_0*delta[a1][d1] + QD_1*delta[a1][c0]) + PQ[d1]*(QC_0*delta[a1][d0] + QD_0*delta[a1][c0]))) + PA_1*(PQ[c0]*(PQ[c1]*(QD_0*delta[a0][d1] + QD_1*delta[a0][d0]) + PQ[d0]*(QC_1*delta[a0][d1] + QD_1*delta[a0][c1]) + PQ[d1]*(QC_1*delta[a0][d0] + QD_0*delta[a0][c1])) + PQ[c1]*(PQ[d0]*(QC_0*delta[a0][d1] + QD_1*delta[a0][c0]) + PQ[d1]*(QC_0*delta[a0][d0] + QD_0*delta[a0][c0]))) + PQ[d0]*PQ[d1]*(PA_0*(QC_0*delta[a1][c1] + QC_1*delta[a1][c0]) + PA_1*(QC_0*delta[a0][c1] + QC_1*delta[a0][c0])))

                        -(PQ[c0]*PQ[c1]*(PQ[d0]*QD_1 + PQ[d1]*QD_0) + PQ[d0]*PQ[d1]*(PQ[c0]*QC_1 + PQ[c1]*QC_0))*(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1])

                    )

                    + F8_t[3] * 0.5 * S1 * S2 * inv_S4 * inv_S4 * inv_S4 * (

                        -(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))*(QC_0*(PQ[c1]*delta[d0][d1] + PQ[d0]*delta[c1][d1] + PQ[d1]*delta[c1][d0]) + QC_1*(PQ[d0]*delta[c0][d1] + PQ[d1]*delta[c0][d0] + delta[d0][d1]*(PQ[c0] + QC_0)) + QD_0*(PQ[d1]*delta[c0][c1] + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + QD_1*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0)))

                        -PA_0*(PA_1*(PQ[b0]*(QC_0*(PQ[c1]*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + PQ[d0]*(QC_1*delta[b1][d1] + QD_1*delta[b1][c1]) + PQ[d1]*(QC_1*delta[b1][d0] + QD_0*delta[b1][c1])) + QC_1*(PQ[c0]*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + delta[b1][c0]*(PQ[d0]*QD_1 + PQ[d1]*QD_0))) + PQ[b1]*(QC_0*(PQ[c1]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + PQ[d0]*(QC_1*delta[b0][d1] + QD_1*delta[b0][c1]) + PQ[d1]*(QC_1*delta[b0][d0] + QD_0*delta[b0][c1])) + QC_1*(PQ[c0]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + delta[b0][c0]*(PQ[d0]*QD_1 + PQ[d1]*QD_0)))) + PB_0*(PQ[a1]*(QC_0*(PQ[c1]*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + PQ[d0]*(QC_1*delta[b1][d1] + QD_1*delta[b1][c1]) + PQ[d1]*(QC_1*delta[b1][d0] + QD_0*delta[b1][c1])) + QC_1*(PQ[c0]*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + delta[b1][c0]*(PQ[d0]*QD_1 + PQ[d1]*QD_0))) + PQ[b1]*(QC_0*(PQ[c1]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]) + PQ[d0]*(QC_1*delta[a1][d1] + QD_1*delta[a1][c1]) + PQ[d1]*(QC_1*delta[a1][d0] + QD_0*delta[a1][c1])) + QC_1*(PQ[c0]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]) + delta[a1][c0]*(PQ[d0]*QD_1 + PQ[d1]*QD_0)))) + PB_1*(PQ[a1]*(QC_0*(PQ[c1]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + PQ[d0]*(QC_1*delta[b0][d1] + QD_1*delta[b0][c1]) + PQ[d1]*(QC_1*delta[b0][d0] + QD_0*delta[b0][c1])) + QC_1*(PQ[c0]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + delta[b0][c0]*(PQ[d0]*QD_1 + PQ[d1]*QD_0))) + PQ[b0]*(QC_0*(PQ[c1]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]) + PQ[d0]*(QC_1*delta[a1][d1] + QD_1*delta[a1][c1]) + PQ[d1]*(QC_1*delta[a1][d0] + QD_0*delta[a1][c1])) + QC_1*(PQ[c0]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]) + delta[a1][c0]*(PQ[d0]*QD_1 + PQ[d1]*QD_0))))) - PA_1*(PB_0*(PQ[a0]*(QC_0*(PQ[c1]*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + PQ[d0]*(QC_1*delta[b1][d1] + QD_1*delta[b1][c1]) + PQ[d1]*(QC_1*delta[b1][d0] + QD_0*delta[b1][c1])) + QC_1*(PQ[c0]*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + delta[b1][c0]*(PQ[d0]*QD_1 + PQ[d1]*QD_0))) + PQ[b1]*(QC_0*(PQ[c1]*(QD_0*delta[a0][d1] + QD_1*delta[a0][d0]) + PQ[d0]*(QC_1*delta[a0][d1] + QD_1*delta[a0][c1]) + PQ[d1]*(QC_1*delta[a0][d0] + QD_0*delta[a0][c1])) + QC_1*(PQ[c0]*(QD_0*delta[a0][d1] + QD_1*delta[a0][d0]) + delta[a0][c0]*(PQ[d0]*QD_1 + PQ[d1]*QD_0)))) + PB_1*(PQ[a0]*(QC_0*(PQ[c1]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + PQ[d0]*(QC_1*delta[b0][d1] + QD_1*delta[b0][c1]) + PQ[d1]*(QC_1*delta[b0][d0] + QD_0*delta[b0][c1])) + QC_1*(PQ[c0]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + delta[b0][c0]*(PQ[d0]*QD_1 + PQ[d1]*QD_0))) + PQ[b0]*(QC_0*(PQ[c1]*(QD_0*delta[a0][d1] + QD_1*delta[a0][d0]) + PQ[d0]*(QC_1*delta[a0][d1] + QD_1*delta[a0][c1]) + PQ[d1]*(QC_1*delta[a0][d0] + QD_0*delta[a0][c1])) + QC_1*(PQ[c0]*(QD_0*delta[a0][d1] + QD_1*delta[a0][d0]) + delta[a0][c0]*(PQ[d0]*QD_1 + PQ[d1]*QD_0))))) - PB_0*PB_1*(PQ[a0]*(QC_0*(PQ[c1]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]) + PQ[d0]*(QC_1*delta[a1][d1] + QD_1*delta[a1][c1]) + PQ[d1]*(QC_1*delta[a1][d0] + QD_0*delta[a1][c1])) + QC_1*(PQ[c0]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]) + delta[a1][c0]*(PQ[d0]*QD_1 + PQ[d1]*QD_0))) + PQ[a1]*(QC_0*(PQ[c1]*(QD_0*delta[a0][d1] + QD_1*delta[a0][d0]) + PQ[d0]*(QC_1*delta[a0][d1] + QD_1*delta[a0][c1]) + PQ[d1]*(QC_1*delta[a0][d0] + QD_0*delta[a0][c1])) + QC_1*(PQ[c0]*(QD_0*delta[a0][d1] + QD_1*delta[a0][d0]) + delta[a0][c0]*(PQ[d0]*QD_1 + PQ[d1]*QD_0)))) - QD_0*QD_1*(PA_0*(PA_1*(PQ[b0]*(PQ[c0]*delta[b1][c1] + PQ[c1]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[b0][c1] + PQ[c1]*delta[b0][c0])) + PB_0*(PQ[a1]*(PQ[c0]*delta[b1][c1] + PQ[c1]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[a1][c1] + PQ[c1]*delta[a1][c0])) + PB_1*(PQ[a1]*(PQ[c0]*delta[b0][c1] + PQ[c1]*delta[b0][c0]) + PQ[b0]*(PQ[c0]*delta[a1][c1] + PQ[c1]*delta[a1][c0]))) + PA_1*(PB_0*(PQ[a0]*(PQ[c0]*delta[b1][c1] + PQ[c1]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[a0][c1] + PQ[c1]*delta[a0][c0])) + PB_1*(PQ[a0]*(PQ[c0]*delta[b0][c1] + PQ[c1]*delta[b0][c0]) + PQ[b0]*(PQ[c0]*delta[a0][c1] + PQ[c1]*delta[a0][c0]))) + PB_0*PB_1*(PQ[a0]*(PQ[c0]*delta[a1][c1] + PQ[c1]*delta[a1][c0]) + PQ[a1]*(PQ[c0]*delta[a0][c1] + PQ[c1]*delta[a0][c0])))

                        +(PQ[c0]*QD_0*(PQ[c1]*QD_1 + PQ[d1]*QC_1) + PQ[c1]*QC_0*(PQ[d0]*QD_1 + PQ[d1]*QD_0) + PQ[d0]*QC_1*(PQ[c0]*QD_1 + PQ[d1]*QC_0))*(PA_0*(-PA_1*delta[b0][b1] - PB_0*delta[a1][b1] - PB_1*delta[a1][b0] + PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PA_1*(PQ[a0]*delta[b0][b1] + delta[a0][b0]*(-PB_1 + PQ[b1]) + delta[a0][b1]*(-PB_0 + PQ[b0])) + PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + delta[a0][a1]*(-PB_1 + PQ[b1])) + PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]))

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * dd_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDD12_FP64(double*         mat_J,
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
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ double   delta[3][3];

    __shared__ double a_i, a_j, r_i[3], r_j[3], S_ij_00, S1, inv_S1;
    __shared__ double PA_0, PA_1, PB_0, PB_1;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta[0][0] = 1.0; delta[0][1] = 0.0; delta[0][2] = 0.0;
        delta[1][0] = 0.0; delta[1][1] = 1.0; delta[1][2] = 0.0;
        delta[2][0] = 0.0; delta[2][1] = 0.0; delta[2][2] = 1.0;

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
            inv_S1 = 1.0 / S1;

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

    for (uint32_t m = 0; m < prec_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= dd_prim_pair_count))
        {
            break;
        }

        const auto k = dd_first_inds[kl];
        const auto l = dd_second_inds[kl];

        const auto a_k = d_prim_info[k / 6 + d_prim_count * 0];

        const double r_k[3] = {d_prim_info[k / 6 + d_prim_count * 2],
                               d_prim_info[k / 6 + d_prim_count * 3],
                               d_prim_info[k / 6 + d_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = dd_pair_data[kl];

        const auto c0 = d_cart_inds[k % 6][0];
        const auto c1 = d_cart_inds[k % 6][1];
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F8_t[4];

        gpu::computeBoysFunction(F8_t, S1 * S2 * inv_S4 * r2_PQ, 3, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QC_1 = (a_l * inv_S2) * (r_l[c1] - r_k[c1]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                    + F8_t[3] * (-0.5) * S2 * S2 * S2 * inv_S1 * inv_S4 * inv_S4 * inv_S4 * (

                        +QC_0*QC_1*QD_0*QD_1*(PQ[a0]*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PQ[a1]*(PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PQ[b0]*PQ[b1]*delta[a0][a1])

                    )

                    + F8_t[3] * 0.5 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * (

                        +(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))*(QC_0*(QC_1*delta[d0][d1] + QD_0*delta[c1][d1] + QD_1*delta[c1][d0]) + QC_1*(QD_0*delta[c0][d1] + QD_1*delta[c0][d0]) + QD_0*QD_1*delta[c0][c1])

                        +QC_0*QC_1*(PQ[a0]*(PA_1*(PQ[b0]*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + PQ[b1]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0])) + PB_0*(PQ[a1]*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + PQ[b1]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0])) + PB_1*(PQ[a1]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + PQ[b0]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]))) + PQ[a1]*(PA_0*(PQ[b0]*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + PQ[b1]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0])) + (PB_0*PQ[b1] + PB_1*PQ[b0])*(QD_0*delta[a0][d1] + QD_1*delta[a0][d0])) + PQ[b0]*PQ[b1]*(PA_0*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]) + PA_1*(QD_0*delta[a0][d1] + QD_1*delta[a0][d0]))) + QD_0*QD_1*(PQ[a0]*(PA_1*(PQ[b0]*(QC_0*delta[b1][c1] + QC_1*delta[b1][c0]) + PQ[b1]*(QC_0*delta[b0][c1] + QC_1*delta[b0][c0])) + PB_0*(PQ[a1]*(QC_0*delta[b1][c1] + QC_1*delta[b1][c0]) + PQ[b1]*(QC_0*delta[a1][c1] + QC_1*delta[a1][c0])) + PB_1*(PQ[a1]*(QC_0*delta[b0][c1] + QC_1*delta[b0][c0]) + PQ[b0]*(QC_0*delta[a1][c1] + QC_1*delta[a1][c0]))) + PQ[a1]*(PA_0*(PQ[b0]*(QC_0*delta[b1][c1] + QC_1*delta[b1][c0]) + PQ[b1]*(QC_0*delta[b0][c1] + QC_1*delta[b0][c0])) + (PB_0*PQ[b1] + PB_1*PQ[b0])*(QC_0*delta[a0][c1] + QC_1*delta[a0][c0])) + PQ[b0]*PQ[b1]*(PA_0*(QC_0*delta[a1][c1] + QC_1*delta[a1][c0]) + PA_1*(QC_0*delta[a0][c1] + QC_1*delta[a0][c0])))

                        +(QC_0*QD_1*(PQ[c1]*QD_0 + PQ[d0]*QC_1) + QC_1*QD_0*(PQ[c0]*QD_1 + PQ[d1]*QC_0))*(PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PQ[a0]*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0] - PQ[a1]*delta[b0][b1] - PQ[b0]*delta[a1][b1] - PQ[b1]*delta[a1][b0]) + PQ[a1]*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PQ[b0]*(PB_1*delta[a0][a1] + delta[a0][b1]*(PA_1 - PQ[a1])) + PQ[b1]*(delta[a0][a1]*(PB_0 - PQ[b0]) + delta[a0][b0]*(PA_1 - PQ[a1])))

                    )

                    + F8_t[3] * S1 * S1 * S1 * inv_S4 * inv_S4 * inv_S4 * (

                        -PA_0*PA_1*PB_0*PB_1*(PQ[c0]*PQ[c1]*(PQ[d0]*QD_1 + PQ[d1]*QD_0) + PQ[d0]*PQ[d1]*(PQ[c0]*QC_1 + PQ[c1]*QC_0))

                    )

                    + F8_t[3] * S1 * S1 * S2 * inv_S4 * inv_S4 * inv_S4 * (

                        +(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))*(PQ[c0]*QD_0*(PQ[c1]*QD_1 + PQ[d1]*QC_1) + PQ[c1]*QC_0*(PQ[d0]*QD_1 + PQ[d1]*QD_0) + PQ[d0]*QC_1*(PQ[c0]*QD_1 + PQ[d1]*QC_0))

                    )

                    + F8_t[3] * S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * (

                        -(QC_0*QD_1*(PQ[c1]*QD_0 + PQ[d0]*QC_1) + QC_1*QD_0*(PQ[c0]*QD_1 + PQ[d1]*QC_0))*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))

                    )

                    + F8_t[3] * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * (

                        +QC_0*QC_1*QD_0*QD_1*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * dd_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDD13_FP64(double*         mat_J,
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
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ double   delta[3][3];

    __shared__ double a_i, a_j, r_i[3], r_j[3], S_ij_00, S1, inv_S1;
    __shared__ double PA_0, PA_1, PB_0, PB_1;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta[0][0] = 1.0; delta[0][1] = 0.0; delta[0][2] = 0.0;
        delta[1][0] = 0.0; delta[1][1] = 1.0; delta[1][2] = 0.0;
        delta[2][0] = 0.0; delta[2][1] = 0.0; delta[2][2] = 1.0;

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
            inv_S1 = 1.0 / S1;

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

    for (uint32_t m = 0; m < prec_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= dd_prim_pair_count))
        {
            break;
        }

        const auto k = dd_first_inds[kl];
        const auto l = dd_second_inds[kl];

        const auto a_k = d_prim_info[k / 6 + d_prim_count * 0];

        const double r_k[3] = {d_prim_info[k / 6 + d_prim_count * 2],
                               d_prim_info[k / 6 + d_prim_count * 3],
                               d_prim_info[k / 6 + d_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = dd_pair_data[kl];

        const auto c0 = d_cart_inds[k % 6][0];
        const auto c1 = d_cart_inds[k % 6][1];
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F8_t[5];

        gpu::computeBoysFunction(F8_t, S1 * S2 * inv_S4 * r2_PQ, 4, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QC_1 = (a_l * inv_S2) * (r_l[c1] - r_k[c1]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                    + F8_t[4] * 0.125 * S1 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        +(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])*(-PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) - PA_1*(PQ[a0]*delta[b0][b1] + PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) - PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) - PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]) + PQ[a0]*PQ[a1]*delta[b0][b1] + PQ[a0]*PQ[b0]*delta[a1][b1] + PQ[a0]*PQ[b1]*delta[a1][b0] + PQ[a1]*PQ[b0]*delta[a0][b1] + PQ[a1]*PQ[b1]*delta[a0][b0] + PQ[b0]*PQ[b1]*delta[a0][a1])

                        +(-PA_0*PQ[a1] - PA_1*PQ[a0])*(delta[b0][c0]*delta[b1][c1]*delta[d0][d1] + delta[b0][c0]*delta[b1][d0]*delta[c1][d1] + delta[b0][c0]*delta[b1][d1]*delta[c1][d0] + delta[b0][c1]*delta[b1][c0]*delta[d0][d1] + delta[b0][c1]*delta[b1][d0]*delta[c0][d1] + delta[b0][c1]*delta[b1][d1]*delta[c0][d0] + delta[b0][d0]*delta[b1][c0]*delta[c1][d1] + delta[b0][d0]*delta[b1][c1]*delta[c0][d1] + delta[b0][d0]*delta[b1][d1]*delta[c0][c1] + delta[b0][d1]*delta[b1][c0]*delta[c1][d0] + delta[b0][d1]*delta[b1][c1]*delta[c0][d0] + delta[b0][d1]*delta[b1][d0]*delta[c0][c1]) + (-PA_0*PQ[b0] - PB_0*PQ[a0])*(delta[a1][c0]*delta[b1][c1]*delta[d0][d1] + delta[a1][c0]*delta[b1][d0]*delta[c1][d1] + delta[a1][c0]*delta[b1][d1]*delta[c1][d0] + delta[a1][c1]*delta[b1][c0]*delta[d0][d1] + delta[a1][c1]*delta[b1][d0]*delta[c0][d1] + delta[a1][c1]*delta[b1][d1]*delta[c0][d0] + delta[a1][d0]*delta[b1][c0]*delta[c1][d1] + delta[a1][d0]*delta[b1][c1]*delta[c0][d1] + delta[a1][d0]*delta[b1][d1]*delta[c0][c1] + delta[a1][d1]*delta[b1][c0]*delta[c1][d0] + delta[a1][d1]*delta[b1][c1]*delta[c0][d0] + delta[a1][d1]*delta[b1][d0]*delta[c0][c1]) + (-PA_0*PQ[b1] - PB_1*PQ[a0])*(delta[a1][c0]*delta[b0][c1]*delta[d0][d1] + delta[a1][c0]*delta[b0][d0]*delta[c1][d1] + delta[a1][c0]*delta[b0][d1]*delta[c1][d0] + delta[a1][c1]*delta[b0][c0]*delta[d0][d1] + delta[a1][c1]*delta[b0][d0]*delta[c0][d1] + delta[a1][c1]*delta[b0][d1]*delta[c0][d0] + delta[a1][d0]*delta[b0][c0]*delta[c1][d1] + delta[a1][d0]*delta[b0][c1]*delta[c0][d1] + delta[a1][d0]*delta[b0][d1]*delta[c0][c1] + delta[a1][d1]*delta[b0][c0]*delta[c1][d0] + delta[a1][d1]*delta[b0][c1]*delta[c0][d0] + delta[a1][d1]*delta[b0][d0]*delta[c0][c1]) + (-PA_1*PQ[b0] - PB_0*PQ[a1])*(delta[a0][c0]*delta[b1][c1]*delta[d0][d1] + delta[a0][c0]*delta[b1][d0]*delta[c1][d1] + delta[a0][c0]*delta[b1][d1]*delta[c1][d0] + delta[a0][c1]*delta[b1][c0]*delta[d0][d1] + delta[a0][c1]*delta[b1][d0]*delta[c0][d1] + delta[a0][c1]*delta[b1][d1]*delta[c0][d0] + delta[a0][d0]*delta[b1][c0]*delta[c1][d1] + delta[a0][d0]*delta[b1][c1]*delta[c0][d1] + delta[a0][d0]*delta[b1][d1]*delta[c0][c1] + delta[a0][d1]*delta[b1][c0]*delta[c1][d0] + delta[a0][d1]*delta[b1][c1]*delta[c0][d0] + delta[a0][d1]*delta[b1][d0]*delta[c0][c1]) + (-PA_1*PQ[b1] - PB_1*PQ[a1])*(delta[a0][c0]*delta[b0][c1]*delta[d0][d1] + delta[a0][c0]*delta[b0][d0]*delta[c1][d1] + delta[a0][c0]*delta[b0][d1]*delta[c1][d0] + delta[a0][c1]*delta[b0][c0]*delta[d0][d1] + delta[a0][c1]*delta[b0][d0]*delta[c0][d1] + delta[a0][c1]*delta[b0][d1]*delta[c0][d0] + delta[a0][d0]*delta[b0][c0]*delta[c1][d1] + delta[a0][d0]*delta[b0][c1]*delta[c0][d1] + delta[a0][d0]*delta[b0][d1]*delta[c0][c1] + delta[a0][d1]*delta[b0][c0]*delta[c1][d0] + delta[a0][d1]*delta[b0][c1]*delta[c0][d0] + delta[a0][d1]*delta[b0][d0]*delta[c0][c1]) + (-PB_0*PQ[b1] - PB_1*PQ[b0])*(delta[a0][c0]*delta[a1][c1]*delta[d0][d1] + delta[a0][c0]*delta[a1][d0]*delta[c1][d1] + delta[a0][c0]*delta[a1][d1]*delta[c1][d0] + delta[a0][c1]*delta[a1][c0]*delta[d0][d1] + delta[a0][c1]*delta[a1][d0]*delta[c0][d1] + delta[a0][c1]*delta[a1][d1]*delta[c0][d0] + delta[a0][d0]*delta[a1][c0]*delta[c1][d1] + delta[a0][d0]*delta[a1][c1]*delta[c0][d1] + delta[a0][d0]*delta[a1][d1]*delta[c0][c1] + delta[a0][d1]*delta[a1][c0]*delta[c1][d0] + delta[a0][d1]*delta[a1][c1]*delta[c0][d0] + delta[a0][d1]*delta[a1][d0]*delta[c0][c1])

                        +(-PA_0*PQ[c0] + PQ[a0]*PQ[c0])*(delta[a1][b0]*delta[b1][c1]*delta[d0][d1] + delta[a1][b0]*delta[b1][d0]*delta[c1][d1] + delta[a1][b0]*delta[b1][d1]*delta[c1][d0] + delta[a1][b1]*delta[b0][c1]*delta[d0][d1] + delta[a1][b1]*delta[b0][d0]*delta[c1][d1] + delta[a1][b1]*delta[b0][d1]*delta[c1][d0] + delta[a1][c1]*delta[b0][b1]*delta[d0][d1] + delta[a1][d0]*delta[b0][b1]*delta[c1][d1] + delta[a1][d1]*delta[b0][b1]*delta[c1][d0]) + (-PA_0*PQ[c1] + PQ[a0]*PQ[c1])*(delta[a1][b0]*delta[b1][c0]*delta[d0][d1] + delta[a1][b0]*delta[b1][d0]*delta[c0][d1] + delta[a1][b0]*delta[b1][d1]*delta[c0][d0] + delta[a1][b1]*delta[b0][c0]*delta[d0][d1] + delta[a1][b1]*delta[b0][d0]*delta[c0][d1] + delta[a1][b1]*delta[b0][d1]*delta[c0][d0] + delta[a1][c0]*delta[b0][b1]*delta[d0][d1] + delta[a1][d0]*delta[b0][b1]*delta[c0][d1] + delta[a1][d1]*delta[b0][b1]*delta[c0][d0]) + (-PA_0*PQ[d0] + PQ[a0]*PQ[d0])*(delta[a1][b0]*delta[b1][c0]*delta[c1][d1] + delta[a1][b0]*delta[b1][c1]*delta[c0][d1] + delta[a1][b0]*delta[b1][d1]*delta[c0][c1] + delta[a1][b1]*delta[b0][c0]*delta[c1][d1] + delta[a1][b1]*delta[b0][c1]*delta[c0][d1] + delta[a1][b1]*delta[b0][d1]*delta[c0][c1] + delta[a1][c0]*delta[b0][b1]*delta[c1][d1] + delta[a1][c1]*delta[b0][b1]*delta[c0][d1] + delta[a1][d1]*delta[b0][b1]*delta[c0][c1]) + (-PA_0*PQ[d1] + PQ[a0]*PQ[d1])*(delta[a1][b0]*delta[b1][c0]*delta[c1][d0] + delta[a1][b0]*delta[b1][c1]*delta[c0][d0] + delta[a1][b0]*delta[b1][d0]*delta[c0][c1] + delta[a1][b1]*delta[b0][c0]*delta[c1][d0] + delta[a1][b1]*delta[b0][c1]*delta[c0][d0] + delta[a1][b1]*delta[b0][d0]*delta[c0][c1] + delta[a1][c0]*delta[b0][b1]*delta[c1][d0] + delta[a1][c1]*delta[b0][b1]*delta[c0][d0] + delta[a1][d0]*delta[b0][b1]*delta[c0][c1]) + (-PA_1*PQ[c0] + PQ[a1]*PQ[c0])*(delta[a0][b0]*delta[b1][c1]*delta[d0][d1] + delta[a0][b0]*delta[b1][d0]*delta[c1][d1] + delta[a0][b0]*delta[b1][d1]*delta[c1][d0] + delta[a0][b1]*delta[b0][c1]*delta[d0][d1] + delta[a0][b1]*delta[b0][d0]*delta[c1][d1] + delta[a0][b1]*delta[b0][d1]*delta[c1][d0] + delta[a0][c1]*delta[b0][b1]*delta[d0][d1] + delta[a0][d0]*delta[b0][b1]*delta[c1][d1] + delta[a0][d1]*delta[b0][b1]*delta[c1][d0]) + (-PA_1*PQ[c1] + PQ[a1]*PQ[c1])*(delta[a0][b0]*delta[b1][c0]*delta[d0][d1] + delta[a0][b0]*delta[b1][d0]*delta[c0][d1] + delta[a0][b0]*delta[b1][d1]*delta[c0][d0] + delta[a0][b1]*delta[b0][c0]*delta[d0][d1] + delta[a0][b1]*delta[b0][d0]*delta[c0][d1] + delta[a0][b1]*delta[b0][d1]*delta[c0][d0] + delta[a0][c0]*delta[b0][b1]*delta[d0][d1] + delta[a0][d0]*delta[b0][b1]*delta[c0][d1] + delta[a0][d1]*delta[b0][b1]*delta[c0][d0]) + (-PA_1*PQ[d0] + PQ[a1]*PQ[d0])*(delta[a0][b0]*delta[b1][c0]*delta[c1][d1] + delta[a0][b0]*delta[b1][c1]*delta[c0][d1] + delta[a0][b0]*delta[b1][d1]*delta[c0][c1] + delta[a0][b1]*delta[b0][c0]*delta[c1][d1] + delta[a0][b1]*delta[b0][c1]*delta[c0][d1] + delta[a0][b1]*delta[b0][d1]*delta[c0][c1] + delta[a0][c0]*delta[b0][b1]*delta[c1][d1] + delta[a0][c1]*delta[b0][b1]*delta[c0][d1] + delta[a0][d1]*delta[b0][b1]*delta[c0][c1]) + (-PA_1*PQ[d1] + PQ[a1]*PQ[d1])*(delta[a0][b0]*delta[b1][c0]*delta[c1][d0] + delta[a0][b0]*delta[b1][c1]*delta[c0][d0] + delta[a0][b0]*delta[b1][d0]*delta[c0][c1] + delta[a0][b1]*delta[b0][c0]*delta[c1][d0] + delta[a0][b1]*delta[b0][c1]*delta[c0][d0] + delta[a0][b1]*delta[b0][d0]*delta[c0][c1] + delta[a0][c0]*delta[b0][b1]*delta[c1][d0] + delta[a0][c1]*delta[b0][b1]*delta[c0][d0] + delta[a0][d0]*delta[b0][b1]*delta[c0][c1]) + (-PB_0*PQ[c0] + PQ[b0]*PQ[c0])*(delta[a0][a1]*delta[b1][c1]*delta[d0][d1] + delta[a0][a1]*delta[b1][d0]*delta[c1][d1] + delta[a0][a1]*delta[b1][d1]*delta[c1][d0] + delta[a0][b1]*delta[a1][c1]*delta[d0][d1] + delta[a0][b1]*delta[a1][d0]*delta[c1][d1] + delta[a0][b1]*delta[a1][d1]*delta[c1][d0] + delta[a0][c1]*delta[a1][b1]*delta[d0][d1] + delta[a0][d0]*delta[a1][b1]*delta[c1][d1] + delta[a0][d1]*delta[a1][b1]*delta[c1][d0]) + (-PB_0*PQ[c1] + PQ[b0]*PQ[c1])*(delta[a0][a1]*delta[b1][c0]*delta[d0][d1] + delta[a0][a1]*delta[b1][d0]*delta[c0][d1] + delta[a0][a1]*delta[b1][d1]*delta[c0][d0] + delta[a0][b1]*delta[a1][c0]*delta[d0][d1] + delta[a0][b1]*delta[a1][d0]*delta[c0][d1] + delta[a0][b1]*delta[a1][d1]*delta[c0][d0] + delta[a0][c0]*delta[a1][b1]*delta[d0][d1] + delta[a0][d0]*delta[a1][b1]*delta[c0][d1] + delta[a0][d1]*delta[a1][b1]*delta[c0][d0]) + (-PB_0*PQ[d0] + PQ[b0]*PQ[d0])*(delta[a0][a1]*delta[b1][c0]*delta[c1][d1] + delta[a0][a1]*delta[b1][c1]*delta[c0][d1] + delta[a0][a1]*delta[b1][d1]*delta[c0][c1] + delta[a0][b1]*delta[a1][c0]*delta[c1][d1] + delta[a0][b1]*delta[a1][c1]*delta[c0][d1] + delta[a0][b1]*delta[a1][d1]*delta[c0][c1] + delta[a0][c0]*delta[a1][b1]*delta[c1][d1] + delta[a0][c1]*delta[a1][b1]*delta[c0][d1] + delta[a0][d1]*delta[a1][b1]*delta[c0][c1]) + (-PB_0*PQ[d1] + PQ[b0]*PQ[d1])*(delta[a0][a1]*delta[b1][c0]*delta[c1][d0] + delta[a0][a1]*delta[b1][c1]*delta[c0][d0] + delta[a0][a1]*delta[b1][d0]*delta[c0][c1] + delta[a0][b1]*delta[a1][c0]*delta[c1][d0] + delta[a0][b1]*delta[a1][c1]*delta[c0][d0] + delta[a0][b1]*delta[a1][d0]*delta[c0][c1] + delta[a0][c0]*delta[a1][b1]*delta[c1][d0] + delta[a0][c1]*delta[a1][b1]*delta[c0][d0] + delta[a0][d0]*delta[a1][b1]*delta[c0][c1]) + (-PB_1*PQ[c0] + PQ[b1]*PQ[c0])*(delta[a0][a1]*delta[b0][c1]*delta[d0][d1] + delta[a0][a1]*delta[b0][d0]*delta[c1][d1] + delta[a0][a1]*delta[b0][d1]*delta[c1][d0] + delta[a0][b0]*delta[a1][c1]*delta[d0][d1] + delta[a0][b0]*delta[a1][d0]*delta[c1][d1] + delta[a0][b0]*delta[a1][d1]*delta[c1][d0] + delta[a0][c1]*delta[a1][b0]*delta[d0][d1] + delta[a0][d0]*delta[a1][b0]*delta[c1][d1] + delta[a0][d1]*delta[a1][b0]*delta[c1][d0]) + (-PB_1*PQ[c1] + PQ[b1]*PQ[c1])*(delta[a0][a1]*delta[b0][c0]*delta[d0][d1] + delta[a0][a1]*delta[b0][d0]*delta[c0][d1] + delta[a0][a1]*delta[b0][d1]*delta[c0][d0] + delta[a0][b0]*delta[a1][c0]*delta[d0][d1] + delta[a0][b0]*delta[a1][d0]*delta[c0][d1] + delta[a0][b0]*delta[a1][d1]*delta[c0][d0] + delta[a0][c0]*delta[a1][b0]*delta[d0][d1] + delta[a0][d0]*delta[a1][b0]*delta[c0][d1] + delta[a0][d1]*delta[a1][b0]*delta[c0][d0]) + (-PB_1*PQ[d0] + PQ[b1]*PQ[d0])*(delta[a0][a1]*delta[b0][c0]*delta[c1][d1] + delta[a0][a1]*delta[b0][c1]*delta[c0][d1] + delta[a0][a1]*delta[b0][d1]*delta[c0][c1] + delta[a0][b0]*delta[a1][c0]*delta[c1][d1] + delta[a0][b0]*delta[a1][c1]*delta[c0][d1] + delta[a0][b0]*delta[a1][d1]*delta[c0][c1] + delta[a0][c0]*delta[a1][b0]*delta[c1][d1] + delta[a0][c1]*delta[a1][b0]*delta[c0][d1] + delta[a0][d1]*delta[a1][b0]*delta[c0][c1]) + (-PB_1*PQ[d1] + PQ[b1]*PQ[d1])*(delta[a0][a1]*delta[b0][c0]*delta[c1][d0] + delta[a0][a1]*delta[b0][c1]*delta[c0][d0] + delta[a0][a1]*delta[b0][d0]*delta[c0][c1] + delta[a0][b0]*delta[a1][c0]*delta[c1][d0] + delta[a0][b0]*delta[a1][c1]*delta[c0][d0] + delta[a0][b0]*delta[a1][d0]*delta[c0][c1] + delta[a0][c0]*delta[a1][b0]*delta[c1][d0] + delta[a0][c1]*delta[a1][b0]*delta[c0][d0] + delta[a0][d0]*delta[a1][b0]*delta[c0][c1])

                        -PA_0*(PQ[c0]*(delta[a1][c1]*(delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + delta[a1][d0]*(delta[b0][c1]*delta[b1][d1] + delta[b0][d1]*delta[b1][c1]) + delta[a1][d1]*(delta[b0][c1]*delta[b1][d0] + delta[b0][d0]*delta[b1][c1])) + PQ[c1]*(delta[a1][c0]*(delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + delta[a1][d0]*(delta[b0][c0]*delta[b1][d1] + delta[b0][d1]*delta[b1][c0]) + delta[a1][d1]*(delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0])) + PQ[d0]*(delta[a1][c0]*(delta[b0][c1]*delta[b1][d1] + delta[b0][d1]*delta[b1][c1]) + delta[a1][c1]*(delta[b0][c0]*delta[b1][d1] + delta[b0][d1]*delta[b1][c0]) + delta[a1][d1]*(delta[b0][c0]*delta[b1][c1] + delta[b0][c1]*delta[b1][c0])) + PQ[d1]*(delta[a1][c0]*(delta[b0][c1]*delta[b1][d0] + delta[b0][d0]*delta[b1][c1]) + delta[a1][c1]*(delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0]) + delta[a1][d0]*(delta[b0][c0]*delta[b1][c1] + delta[b0][c1]*delta[b1][c0]))) - PA_1*(PQ[c0]*(delta[a0][c1]*(delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + delta[a0][d0]*(delta[b0][c1]*delta[b1][d1] + delta[b0][d1]*delta[b1][c1]) + delta[a0][d1]*(delta[b0][c1]*delta[b1][d0] + delta[b0][d0]*delta[b1][c1])) + PQ[c1]*(delta[a0][c0]*(delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + delta[a0][d0]*(delta[b0][c0]*delta[b1][d1] + delta[b0][d1]*delta[b1][c0]) + delta[a0][d1]*(delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0])) + PQ[d0]*(delta[a0][c0]*(delta[b0][c1]*delta[b1][d1] + delta[b0][d1]*delta[b1][c1]) + delta[a0][c1]*(delta[b0][c0]*delta[b1][d1] + delta[b0][d1]*delta[b1][c0]) + delta[a0][d1]*(delta[b0][c0]*delta[b1][c1] + delta[b0][c1]*delta[b1][c0])) + PQ[d1]*(delta[a0][c0]*(delta[b0][c1]*delta[b1][d0] + delta[b0][d0]*delta[b1][c1]) + delta[a0][c1]*(delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0]) + delta[a0][d0]*(delta[b0][c0]*delta[b1][c1] + delta[b0][c1]*delta[b1][c0]))) - PB_0*(PQ[c0]*(delta[a0][c1]*(delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + delta[a0][d0]*(delta[a1][c1]*delta[b1][d1] + delta[a1][d1]*delta[b1][c1]) + delta[a0][d1]*(delta[a1][c1]*delta[b1][d0] + delta[a1][d0]*delta[b1][c1])) + PQ[c1]*(delta[a0][c0]*(delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + delta[a0][d0]*(delta[a1][c0]*delta[b1][d1] + delta[a1][d1]*delta[b1][c0]) + delta[a0][d1]*(delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0])) + PQ[d0]*(delta[a0][c0]*(delta[a1][c1]*delta[b1][d1] + delta[a1][d1]*delta[b1][c1]) + delta[a0][c1]*(delta[a1][c0]*delta[b1][d1] + delta[a1][d1]*delta[b1][c0]) + delta[a0][d1]*(delta[a1][c0]*delta[b1][c1] + delta[a1][c1]*delta[b1][c0])) + PQ[d1]*(delta[a0][c0]*(delta[a1][c1]*delta[b1][d0] + delta[a1][d0]*delta[b1][c1]) + delta[a0][c1]*(delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + delta[a0][d0]*(delta[a1][c0]*delta[b1][c1] + delta[a1][c1]*delta[b1][c0]))) - PB_1*(PQ[c0]*(delta[a0][c1]*(delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + delta[a0][d0]*(delta[a1][c1]*delta[b0][d1] + delta[a1][d1]*delta[b0][c1]) + delta[a0][d1]*(delta[a1][c1]*delta[b0][d0] + delta[a1][d0]*delta[b0][c1])) + PQ[c1]*(delta[a0][c0]*(delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + delta[a0][d0]*(delta[a1][c0]*delta[b0][d1] + delta[a1][d1]*delta[b0][c0]) + delta[a0][d1]*(delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0])) + PQ[d0]*(delta[a0][c0]*(delta[a1][c1]*delta[b0][d1] + delta[a1][d1]*delta[b0][c1]) + delta[a0][c1]*(delta[a1][c0]*delta[b0][d1] + delta[a1][d1]*delta[b0][c0]) + delta[a0][d1]*(delta[a1][c0]*delta[b0][c1] + delta[a1][c1]*delta[b0][c0])) + PQ[d1]*(delta[a0][c0]*(delta[a1][c1]*delta[b0][d0] + delta[a1][d0]*delta[b0][c1]) + delta[a0][c1]*(delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0]) + delta[a0][d0]*(delta[a1][c0]*delta[b0][c1] + delta[a1][c1]*delta[b0][c0])))

                        +2.0*(PQ[c0]*(PQ[c1]*delta[d0][d1] + PQ[d0]*delta[c1][d1] + PQ[d1]*delta[c1][d0]) + PQ[c1]*(PQ[d0]*delta[c0][d1] + PQ[d1]*delta[c0][d0]) + PQ[d0]*PQ[d1]*delta[c0][c1])*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])

                        +PQ[c0]*(PQ[c1]*(delta[a0][b0]*(delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + delta[a0][d0]*(delta[a1][b0]*delta[b1][d1] + delta[a1][d1]*delta[b0][b1]) + delta[a0][d1]*(delta[a1][b0]*delta[b1][d0] + delta[a1][d0]*delta[b0][b1]) + delta[b0][d0]*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + delta[b0][d1]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1])) + PQ[d0]*(delta[a0][b0]*(delta[a1][c1]*delta[b1][d1] + delta[a1][d1]*delta[b1][c1]) + delta[a0][c1]*(delta[a1][b0]*delta[b1][d1] + delta[a1][d1]*delta[b0][b1]) + delta[a0][d1]*(delta[a1][b0]*delta[b1][c1] + delta[a1][c1]*delta[b0][b1]) + delta[b0][c1]*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + delta[b0][d1]*(delta[a0][a1]*delta[b1][c1] + delta[a0][b1]*delta[a1][c1] + delta[a0][c1]*delta[a1][b1])) + PQ[d1]*(delta[a0][b0]*(delta[a1][c1]*delta[b1][d0] + delta[a1][d0]*delta[b1][c1]) + delta[a0][c1]*(delta[a1][b0]*delta[b1][d0] + delta[a1][d0]*delta[b0][b1]) + delta[a0][d0]*(delta[a1][b0]*delta[b1][c1] + delta[a1][c1]*delta[b0][b1]) + delta[b0][c1]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + delta[b0][d0]*(delta[a0][a1]*delta[b1][c1] + delta[a0][b1]*delta[a1][c1] + delta[a0][c1]*delta[a1][b1]))) + PQ[c1]*(PQ[d0]*(delta[a0][b0]*(delta[a1][c0]*delta[b1][d1] + delta[a1][d1]*delta[b1][c0]) + delta[a0][c0]*(delta[a1][b0]*delta[b1][d1] + delta[a1][d1]*delta[b0][b1]) + delta[a0][d1]*(delta[a1][b0]*delta[b1][c0] + delta[a1][c0]*delta[b0][b1]) + delta[b0][c0]*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + delta[b0][d1]*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1])) + PQ[d1]*(delta[a0][b0]*(delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + delta[a0][c0]*(delta[a1][b0]*delta[b1][d0] + delta[a1][d0]*delta[b0][b1]) + delta[a0][d0]*(delta[a1][b0]*delta[b1][c0] + delta[a1][c0]*delta[b0][b1]) + delta[b0][c0]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + delta[b0][d0]*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]))) + PQ[d0]*PQ[d1]*(delta[a0][b0]*(delta[a1][c0]*delta[b1][c1] + delta[a1][c1]*delta[b1][c0]) + delta[a0][c0]*(delta[a1][b0]*delta[b1][c1] + delta[a1][c1]*delta[b0][b1]) + delta[a0][c1]*(delta[a1][b0]*delta[b1][c0] + delta[a1][c0]*delta[b0][b1]) + delta[b0][c0]*(delta[a0][a1]*delta[b1][c1] + delta[a0][b1]*delta[a1][c1] + delta[a0][c1]*delta[a1][b1]) + delta[b0][c1]*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]))

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * dd_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDD14_FP64(double*         mat_J,
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
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ double   delta[3][3];

    __shared__ double a_i, a_j, r_i[3], r_j[3], S_ij_00, S1, inv_S1;
    __shared__ double PA_0, PA_1, PB_0, PB_1;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta[0][0] = 1.0; delta[0][1] = 0.0; delta[0][2] = 0.0;
        delta[1][0] = 0.0; delta[1][1] = 1.0; delta[1][2] = 0.0;
        delta[2][0] = 0.0; delta[2][1] = 0.0; delta[2][2] = 1.0;

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
            inv_S1 = 1.0 / S1;

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

    for (uint32_t m = 0; m < prec_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= dd_prim_pair_count))
        {
            break;
        }

        const auto k = dd_first_inds[kl];
        const auto l = dd_second_inds[kl];

        const auto a_k = d_prim_info[k / 6 + d_prim_count * 0];

        const double r_k[3] = {d_prim_info[k / 6 + d_prim_count * 2],
                               d_prim_info[k / 6 + d_prim_count * 3],
                               d_prim_info[k / 6 + d_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = dd_pair_data[kl];

        const auto c0 = d_cart_inds[k % 6][0];
        const auto c1 = d_cart_inds[k % 6][1];
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F8_t[5];

        gpu::computeBoysFunction(F8_t, S1 * S2 * inv_S4 * r2_PQ, 4, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QC_1 = (a_l * inv_S2) * (r_l[c1] - r_k[c1]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                    + F8_t[4] * 0.125 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        +PQ[a0]*(delta[a1][b0]*(delta[b1][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[b1][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[b1][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[b1][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))) + delta[a1][b1]*(delta[b0][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[b0][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[b0][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[b0][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))) + delta[b0][b1]*(delta[a1][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[a1][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[a1][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[a1][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0)))) + PQ[a1]*(delta[a0][b0]*(delta[b1][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[b1][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[b1][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[b1][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))) + delta[a0][b1]*(delta[b0][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[b0][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[b0][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[b0][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))) + delta[b0][b1]*(delta[a0][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[a0][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[a0][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[a0][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0)))) + PQ[b0]*(delta[a0][a1]*(delta[b1][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[b1][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[b1][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[b1][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))) + delta[a0][b1]*(delta[a1][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[a1][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[a1][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[a1][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))) + delta[a1][b1]*(delta[a0][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[a0][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[a0][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[a0][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0)))) + PQ[b1]*(delta[a0][a1]*(delta[b0][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[b0][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[b0][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[b0][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))) + delta[a0][b0]*(delta[a1][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[a1][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[a1][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[a1][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))) + delta[a1][b0]*(delta[a0][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[a0][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[a0][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[a0][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))))

                        +PQ[a0]*(QC_0*(delta[a1][c1]*(delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + delta[a1][d0]*(delta[b0][c1]*delta[b1][d1] + delta[b0][d1]*delta[b1][c1]) + delta[a1][d1]*(delta[b0][c1]*delta[b1][d0] + delta[b0][d0]*delta[b1][c1])) + QC_1*(delta[a1][c0]*(delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + delta[a1][d0]*(delta[b0][c0]*delta[b1][d1] + delta[b0][d1]*delta[b1][c0]) + delta[a1][d1]*(delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0])) + QD_0*(delta[a1][c0]*(delta[b0][c1]*delta[b1][d1] + delta[b0][d1]*delta[b1][c1]) + delta[a1][c1]*(delta[b0][c0]*delta[b1][d1] + delta[b0][d1]*delta[b1][c0]) + delta[a1][d1]*(delta[b0][c0]*delta[b1][c1] + delta[b0][c1]*delta[b1][c0])) + QD_1*(delta[a1][c0]*(delta[b0][c1]*delta[b1][d0] + delta[b0][d0]*delta[b1][c1]) + delta[a1][c1]*(delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0]) + delta[a1][d0]*(delta[b0][c0]*delta[b1][c1] + delta[b0][c1]*delta[b1][c0]))) + PQ[a1]*(QC_0*(delta[a0][c1]*(delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + delta[a0][d0]*(delta[b0][c1]*delta[b1][d1] + delta[b0][d1]*delta[b1][c1]) + delta[a0][d1]*(delta[b0][c1]*delta[b1][d0] + delta[b0][d0]*delta[b1][c1])) + QC_1*(delta[a0][c0]*(delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + delta[a0][d0]*(delta[b0][c0]*delta[b1][d1] + delta[b0][d1]*delta[b1][c0]) + delta[a0][d1]*(delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0])) + QD_0*(delta[a0][c0]*(delta[b0][c1]*delta[b1][d1] + delta[b0][d1]*delta[b1][c1]) + delta[a0][c1]*(delta[b0][c0]*delta[b1][d1] + delta[b0][d1]*delta[b1][c0]) + delta[a0][d1]*(delta[b0][c0]*delta[b1][c1] + delta[b0][c1]*delta[b1][c0])) + QD_1*(delta[a0][c0]*(delta[b0][c1]*delta[b1][d0] + delta[b0][d0]*delta[b1][c1]) + delta[a0][c1]*(delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0]) + delta[a0][d0]*(delta[b0][c0]*delta[b1][c1] + delta[b0][c1]*delta[b1][c0]))) + PQ[b0]*(QC_0*(delta[a0][c1]*(delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + delta[a0][d0]*(delta[a1][c1]*delta[b1][d1] + delta[a1][d1]*delta[b1][c1]) + delta[a0][d1]*(delta[a1][c1]*delta[b1][d0] + delta[a1][d0]*delta[b1][c1])) + QC_1*(delta[a0][c0]*(delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + delta[a0][d0]*(delta[a1][c0]*delta[b1][d1] + delta[a1][d1]*delta[b1][c0]) + delta[a0][d1]*(delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0])) + QD_0*(delta[a0][c0]*(delta[a1][c1]*delta[b1][d1] + delta[a1][d1]*delta[b1][c1]) + delta[a0][c1]*(delta[a1][c0]*delta[b1][d1] + delta[a1][d1]*delta[b1][c0]) + delta[a0][d1]*(delta[a1][c0]*delta[b1][c1] + delta[a1][c1]*delta[b1][c0])) + QD_1*(delta[a0][c0]*(delta[a1][c1]*delta[b1][d0] + delta[a1][d0]*delta[b1][c1]) + delta[a0][c1]*(delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + delta[a0][d0]*(delta[a1][c0]*delta[b1][c1] + delta[a1][c1]*delta[b1][c0]))) + PQ[b1]*(QC_0*(delta[a0][c1]*(delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + delta[a0][d0]*(delta[a1][c1]*delta[b0][d1] + delta[a1][d1]*delta[b0][c1]) + delta[a0][d1]*(delta[a1][c1]*delta[b0][d0] + delta[a1][d0]*delta[b0][c1])) + QC_1*(delta[a0][c0]*(delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + delta[a0][d0]*(delta[a1][c0]*delta[b0][d1] + delta[a1][d1]*delta[b0][c0]) + delta[a0][d1]*(delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0])) + QD_0*(delta[a0][c0]*(delta[a1][c1]*delta[b0][d1] + delta[a1][d1]*delta[b0][c1]) + delta[a0][c1]*(delta[a1][c0]*delta[b0][d1] + delta[a1][d1]*delta[b0][c0]) + delta[a0][d1]*(delta[a1][c0]*delta[b0][c1] + delta[a1][c1]*delta[b0][c0])) + QD_1*(delta[a0][c0]*(delta[a1][c1]*delta[b0][d0] + delta[a1][d0]*delta[b0][c1]) + delta[a0][c1]*(delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0]) + delta[a0][d0]*(delta[a1][c0]*delta[b0][c1] + delta[a1][c1]*delta[b0][c0])))

                        +(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])*(PQ[c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + PQ[c1]*(QC_0*delta[d0][d1] + delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0)) + PQ[d0]*(QC_0*delta[c1][d1] + QC_1*delta[c0][d1] + delta[c0][c1]*(PQ[d1] + QD_1)) + PQ[d1]*(QC_0*delta[c1][d0] + QC_1*delta[c0][d0] + QD_0*delta[c0][c1]))

                        +(PQ[c0]*QC_1 + PQ[c1]*QC_0)*(delta[a0][a1]*delta[b0][d0]*delta[b1][d1] + delta[a0][a1]*delta[b0][d1]*delta[b1][d0] + delta[a0][b0]*delta[a1][d0]*delta[b1][d1] + delta[a0][b0]*delta[a1][d1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0]*delta[b0][d1] + delta[a0][b1]*delta[a1][d1]*delta[b0][d0] + delta[a0][d0]*delta[a1][b0]*delta[b1][d1] + delta[a0][d0]*delta[a1][b1]*delta[b0][d1] + delta[a0][d0]*delta[a1][d1]*delta[b0][b1] + delta[a0][d1]*delta[a1][b0]*delta[b1][d0] + delta[a0][d1]*delta[a1][b1]*delta[b0][d0] + delta[a0][d1]*delta[a1][d0]*delta[b0][b1]) + (PQ[c0]*QD_0 + PQ[d0]*QC_0)*(delta[a0][a1]*delta[b0][c1]*delta[b1][d1] + delta[a0][a1]*delta[b0][d1]*delta[b1][c1] + delta[a0][b0]*delta[a1][c1]*delta[b1][d1] + delta[a0][b0]*delta[a1][d1]*delta[b1][c1] + delta[a0][b1]*delta[a1][c1]*delta[b0][d1] + delta[a0][b1]*delta[a1][d1]*delta[b0][c1] + delta[a0][c1]*delta[a1][b0]*delta[b1][d1] + delta[a0][c1]*delta[a1][b1]*delta[b0][d1] + delta[a0][c1]*delta[a1][d1]*delta[b0][b1] + delta[a0][d1]*delta[a1][b0]*delta[b1][c1] + delta[a0][d1]*delta[a1][b1]*delta[b0][c1] + delta[a0][d1]*delta[a1][c1]*delta[b0][b1]) + (PQ[c0]*QD_1 + PQ[d1]*QC_0)*(delta[a0][a1]*delta[b0][c1]*delta[b1][d0] + delta[a0][a1]*delta[b0][d0]*delta[b1][c1] + delta[a0][b0]*delta[a1][c1]*delta[b1][d0] + delta[a0][b0]*delta[a1][d0]*delta[b1][c1] + delta[a0][b1]*delta[a1][c1]*delta[b0][d0] + delta[a0][b1]*delta[a1][d0]*delta[b0][c1] + delta[a0][c1]*delta[a1][b0]*delta[b1][d0] + delta[a0][c1]*delta[a1][b1]*delta[b0][d0] + delta[a0][c1]*delta[a1][d0]*delta[b0][b1] + delta[a0][d0]*delta[a1][b0]*delta[b1][c1] + delta[a0][d0]*delta[a1][b1]*delta[b0][c1] + delta[a0][d0]*delta[a1][c1]*delta[b0][b1]) + (PQ[c1]*QD_0 + PQ[d0]*QC_1)*(delta[a0][a1]*delta[b0][c0]*delta[b1][d1] + delta[a0][a1]*delta[b0][d1]*delta[b1][c0] + delta[a0][b0]*delta[a1][c0]*delta[b1][d1] + delta[a0][b0]*delta[a1][d1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0]*delta[b0][d1] + delta[a0][b1]*delta[a1][d1]*delta[b0][c0] + delta[a0][c0]*delta[a1][b0]*delta[b1][d1] + delta[a0][c0]*delta[a1][b1]*delta[b0][d1] + delta[a0][c0]*delta[a1][d1]*delta[b0][b1] + delta[a0][d1]*delta[a1][b0]*delta[b1][c0] + delta[a0][d1]*delta[a1][b1]*delta[b0][c0] + delta[a0][d1]*delta[a1][c0]*delta[b0][b1]) + (PQ[c1]*QD_1 + PQ[d1]*QC_1)*(delta[a0][a1]*delta[b0][c0]*delta[b1][d0] + delta[a0][a1]*delta[b0][d0]*delta[b1][c0] + delta[a0][b0]*delta[a1][c0]*delta[b1][d0] + delta[a0][b0]*delta[a1][d0]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0]*delta[b0][d0] + delta[a0][b1]*delta[a1][d0]*delta[b0][c0] + delta[a0][c0]*delta[a1][b0]*delta[b1][d0] + delta[a0][c0]*delta[a1][b1]*delta[b0][d0] + delta[a0][c0]*delta[a1][d0]*delta[b0][b1] + delta[a0][d0]*delta[a1][b0]*delta[b1][c0] + delta[a0][d0]*delta[a1][b1]*delta[b0][c0] + delta[a0][d0]*delta[a1][c0]*delta[b0][b1]) + (PQ[d0]*QD_1 + PQ[d1]*QD_0)*(delta[a0][a1]*delta[b0][c0]*delta[b1][c1] + delta[a0][a1]*delta[b0][c1]*delta[b1][c0] + delta[a0][b0]*delta[a1][c0]*delta[b1][c1] + delta[a0][b0]*delta[a1][c1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0]*delta[b0][c1] + delta[a0][b1]*delta[a1][c1]*delta[b0][c0] + delta[a0][c0]*delta[a1][b0]*delta[b1][c1] + delta[a0][c0]*delta[a1][b1]*delta[b0][c1] + delta[a0][c0]*delta[a1][c1]*delta[b0][b1] + delta[a0][c1]*delta[a1][b0]*delta[b1][c0] + delta[a0][c1]*delta[a1][b1]*delta[b0][c0] + delta[a0][c1]*delta[a1][c0]*delta[b0][b1])

                        +2.0*(PQ[a0]*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PQ[a1]*(PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PQ[b0]*PQ[b1]*delta[a0][a1])*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])

                        +PQ[a0]*(PQ[a1]*(delta[b0][c0]*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + delta[b0][c1]*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + delta[b0][d0]*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + delta[b0][d1]*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) + PQ[b0]*(delta[a1][c0]*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + delta[a1][c1]*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + delta[a1][d0]*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + delta[a1][d1]*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) + PQ[b1]*(delta[a1][c0]*(delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0]) + delta[a1][c1]*(delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0]) + delta[a1][d0]*(delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1]) + delta[a1][d1]*(delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1]))) + PQ[a1]*(PQ[b0]*(delta[a0][c0]*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + delta[a0][c1]*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + delta[a0][d0]*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + delta[a0][d1]*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) + PQ[b1]*(delta[a0][c0]*(delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0]) + delta[a0][c1]*(delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0]) + delta[a0][d0]*(delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1]) + delta[a0][d1]*(delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1]))) + PQ[b0]*PQ[b1]*(delta[a0][c0]*(delta[a1][c1]*delta[d0][d1] + delta[a1][d0]*delta[c1][d1] + delta[a1][d1]*delta[c1][d0]) + delta[a0][c1]*(delta[a1][c0]*delta[d0][d1] + delta[a1][d0]*delta[c0][d1] + delta[a1][d1]*delta[c0][d0]) + delta[a0][d0]*(delta[a1][c0]*delta[c1][d1] + delta[a1][c1]*delta[c0][d1] + delta[a1][d1]*delta[c0][c1]) + delta[a0][d1]*(delta[a1][c0]*delta[c1][d0] + delta[a1][c1]*delta[c0][d0] + delta[a1][d0]*delta[c0][c1]))

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * dd_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDD15_FP64(double*         mat_J,
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
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ double   delta[3][3];

    __shared__ double a_i, a_j, r_i[3], r_j[3], S_ij_00, S1, inv_S1;
    __shared__ double PA_0, PA_1, PB_0, PB_1;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta[0][0] = 1.0; delta[0][1] = 0.0; delta[0][2] = 0.0;
        delta[1][0] = 0.0; delta[1][1] = 1.0; delta[1][2] = 0.0;
        delta[2][0] = 0.0; delta[2][1] = 0.0; delta[2][2] = 1.0;

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
            inv_S1 = 1.0 / S1;

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

    for (uint32_t m = 0; m < prec_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= dd_prim_pair_count))
        {
            break;
        }

        const auto k = dd_first_inds[kl];
        const auto l = dd_second_inds[kl];

        const auto a_k = d_prim_info[k / 6 + d_prim_count * 0];

        const double r_k[3] = {d_prim_info[k / 6 + d_prim_count * 2],
                               d_prim_info[k / 6 + d_prim_count * 3],
                               d_prim_info[k / 6 + d_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = dd_pair_data[kl];

        const auto c0 = d_cart_inds[k % 6][0];
        const auto c1 = d_cart_inds[k % 6][1];
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F8_t[5];

        gpu::computeBoysFunction(F8_t, S1 * S2 * inv_S4 * r2_PQ, 4, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QC_1 = (a_l * inv_S2) * (r_l[c1] - r_k[c1]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                    + F8_t[4] * 0.25 * S1 * S1 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        +(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))

                        +(delta[a0][c0]*delta[c1][d0] + delta[a0][c1]*delta[c0][d0] + delta[a0][d0]*delta[c0][c1])*(PA_1*PB_0*PQ[b1]*PQ[d1] + PA_1*PB_1*PQ[b0]*PQ[d1] + PB_0*PB_1*PQ[a1]*PQ[d1]) + (delta[a0][c0]*delta[c1][d1] + delta[a0][c1]*delta[c0][d1] + delta[a0][d1]*delta[c0][c1])*(PA_1*PB_0*PQ[b1]*PQ[d0] + PA_1*PB_1*PQ[b0]*PQ[d0] + PB_0*PB_1*PQ[a1]*PQ[d0]) + (delta[a0][c0]*delta[d0][d1] + delta[a0][d0]*delta[c0][d1] + delta[a0][d1]*delta[c0][d0])*(PA_1*PB_0*PQ[b1]*PQ[c1] + PA_1*PB_1*PQ[b0]*PQ[c1] + PB_0*PB_1*PQ[a1]*PQ[c1]) + (delta[a0][c1]*delta[d0][d1] + delta[a0][d0]*delta[c1][d1] + delta[a0][d1]*delta[c1][d0])*(PA_1*PB_0*PQ[b1]*PQ[c0] + PA_1*PB_1*PQ[b0]*PQ[c0] + PB_0*PB_1*PQ[a1]*PQ[c0]) + (delta[a1][c0]*delta[c1][d0] + delta[a1][c1]*delta[c0][d0] + delta[a1][d0]*delta[c0][c1])*(PA_0*PB_0*PQ[b1]*PQ[d1] + PA_0*PB_1*PQ[b0]*PQ[d1] + PB_0*PB_1*PQ[a0]*PQ[d1]) + (delta[a1][c0]*delta[c1][d1] + delta[a1][c1]*delta[c0][d1] + delta[a1][d1]*delta[c0][c1])*(PA_0*PB_0*PQ[b1]*PQ[d0] + PA_0*PB_1*PQ[b0]*PQ[d0] + PB_0*PB_1*PQ[a0]*PQ[d0]) + (delta[a1][c0]*delta[d0][d1] + delta[a1][d0]*delta[c0][d1] + delta[a1][d1]*delta[c0][d0])*(PA_0*PB_0*PQ[b1]*PQ[c1] + PA_0*PB_1*PQ[b0]*PQ[c1] + PB_0*PB_1*PQ[a0]*PQ[c1]) + (delta[a1][c1]*delta[d0][d1] + delta[a1][d0]*delta[c1][d1] + delta[a1][d1]*delta[c1][d0])*(PA_0*PB_0*PQ[b1]*PQ[c0] + PA_0*PB_1*PQ[b0]*PQ[c0] + PB_0*PB_1*PQ[a0]*PQ[c0]) + (delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1])*(PA_0*PA_1*PQ[b1]*PQ[d1] + PA_0*PB_1*PQ[a1]*PQ[d1] + PA_1*PB_1*PQ[a0]*PQ[d1]) + (delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1])*(PA_0*PA_1*PQ[b1]*PQ[d0] + PA_0*PB_1*PQ[a1]*PQ[d0] + PA_1*PB_1*PQ[a0]*PQ[d0]) + (delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0])*(PA_0*PA_1*PQ[b1]*PQ[c1] + PA_0*PB_1*PQ[a1]*PQ[c1] + PA_1*PB_1*PQ[a0]*PQ[c1]) + (delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0])*(PA_0*PA_1*PQ[b1]*PQ[c0] + PA_0*PB_1*PQ[a1]*PQ[c0] + PA_1*PB_1*PQ[a0]*PQ[c0]) + (delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])*(PA_0*PA_1*PQ[b0]*PQ[d1] + PA_0*PB_0*PQ[a1]*PQ[d1] + PA_1*PB_0*PQ[a0]*PQ[d1]) + (delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1])*(PA_0*PA_1*PQ[b0]*PQ[d0] + PA_0*PB_0*PQ[a1]*PQ[d0] + PA_1*PB_0*PQ[a0]*PQ[d0]) + (delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0])*(PA_0*PA_1*PQ[b0]*PQ[c1] + PA_0*PB_0*PQ[a1]*PQ[c1] + PA_1*PB_0*PQ[a0]*PQ[c1]) + (delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0])*(PA_0*PA_1*PQ[b0]*PQ[c0] + PA_0*PB_0*PQ[a1]*PQ[c0] + PA_1*PB_0*PQ[a0]*PQ[c0])

                        +(PQ[c0]*(PQ[c1]*delta[d0][d1] + PQ[d0]*delta[c1][d1] + PQ[d1]*delta[c1][d0]) + PQ[c1]*(PQ[d0]*delta[c0][d1] + PQ[d1]*delta[c0][d0]) + PQ[d0]*PQ[d1]*delta[c0][c1])*(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) - PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) - PA_1*(PQ[a0]*delta[b0][b1] + PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1] - PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) - PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]))

                        +PA_0*(PQ[c0]*(PA_1*(PQ[c1]*(delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + PQ[d0]*(delta[b0][c1]*delta[b1][d1] + delta[b0][d1]*delta[b1][c1]) + PQ[d1]*(delta[b0][c1]*delta[b1][d0] + delta[b0][d0]*delta[b1][c1])) + PB_0*(PQ[c1]*(delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + PQ[d0]*(delta[a1][c1]*delta[b1][d1] + delta[a1][d1]*delta[b1][c1]) + PQ[d1]*(delta[a1][c1]*delta[b1][d0] + delta[a1][d0]*delta[b1][c1])) + PB_1*(PQ[c1]*(delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + PQ[d0]*(delta[a1][c1]*delta[b0][d1] + delta[a1][d1]*delta[b0][c1]) + PQ[d1]*(delta[a1][c1]*delta[b0][d0] + delta[a1][d0]*delta[b0][c1]))) + PQ[c1]*(PQ[d0]*(PA_1*(delta[b0][c0]*delta[b1][d1] + delta[b0][d1]*delta[b1][c0]) + PB_0*(delta[a1][c0]*delta[b1][d1] + delta[a1][d1]*delta[b1][c0]) + PB_1*(delta[a1][c0]*delta[b0][d1] + delta[a1][d1]*delta[b0][c0])) + PQ[d1]*(PA_1*(delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0]) + PB_0*(delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + PB_1*(delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0])))) + PA_1*(PB_0*(PQ[c0]*(PQ[c1]*(delta[a0][d0]*delta[b1][d1] + delta[a0][d1]*delta[b1][d0]) + PQ[d0]*(delta[a0][c1]*delta[b1][d1] + delta[a0][d1]*delta[b1][c1]) + PQ[d1]*(delta[a0][c1]*delta[b1][d0] + delta[a0][d0]*delta[b1][c1])) + PQ[c1]*(PQ[d0]*(delta[a0][c0]*delta[b1][d1] + delta[a0][d1]*delta[b1][c0]) + PQ[d1]*(delta[a0][c0]*delta[b1][d0] + delta[a0][d0]*delta[b1][c0]))) + PB_1*(PQ[c0]*(PQ[c1]*(delta[a0][d0]*delta[b0][d1] + delta[a0][d1]*delta[b0][d0]) + PQ[d0]*(delta[a0][c1]*delta[b0][d1] + delta[a0][d1]*delta[b0][c1]) + PQ[d1]*(delta[a0][c1]*delta[b0][d0] + delta[a0][d0]*delta[b0][c1])) + PQ[c1]*(PQ[d0]*(delta[a0][c0]*delta[b0][d1] + delta[a0][d1]*delta[b0][c0]) + PQ[d1]*(delta[a0][c0]*delta[b0][d0] + delta[a0][d0]*delta[b0][c0])))) + PB_0*PB_1*(PQ[c0]*(PQ[c1]*(delta[a0][d0]*delta[a1][d1] + delta[a0][d1]*delta[a1][d0]) + PQ[d0]*(delta[a0][c1]*delta[a1][d1] + delta[a0][d1]*delta[a1][c1]) + PQ[d1]*(delta[a0][c1]*delta[a1][d0] + delta[a0][d0]*delta[a1][c1])) + PQ[c1]*(PQ[d0]*(delta[a0][c0]*delta[a1][d1] + delta[a0][d1]*delta[a1][c0]) + PQ[d1]*(delta[a0][c0]*delta[a1][d0] + delta[a0][d0]*delta[a1][c0]))) + PQ[d0]*PQ[d1]*(PA_0*(PA_1*(delta[b0][c0]*delta[b1][c1] + delta[b0][c1]*delta[b1][c0]) + PB_0*(delta[a1][c0]*delta[b1][c1] + delta[a1][c1]*delta[b1][c0]) + PB_1*(delta[a1][c0]*delta[b0][c1] + delta[a1][c1]*delta[b0][c0])) + PA_1*(PB_0*(delta[a0][c0]*delta[b1][c1] + delta[a0][c1]*delta[b1][c0]) + PB_1*(delta[a0][c0]*delta[b0][c1] + delta[a0][c1]*delta[b0][c0])) + PB_0*PB_1*(delta[a0][c0]*delta[a1][c1] + delta[a0][c1]*delta[a1][c0]))

                        -PQ[c0]*PQ[c1]*(PQ[d0]*(PA_0*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0])) + PQ[d1]*(PA_0*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]))) - PQ[d0]*PQ[d1]*(PQ[c0]*(PA_0*(delta[a1][b0]*delta[b1][c1] + delta[a1][b1]*delta[b0][c1] + delta[a1][c1]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][c1] + delta[a0][b1]*delta[b0][c1] + delta[a0][c1]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][c1] + delta[a0][b1]*delta[a1][c1] + delta[a0][c1]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][c1] + delta[a0][b0]*delta[a1][c1] + delta[a0][c1]*delta[a1][b0])) + PQ[c1]*(PA_0*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0])))

                        +PQ[c0]*PQ[c1]*PQ[d0]*PQ[d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * dd_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDD16_FP64(double*         mat_J,
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
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ double   delta[3][3];

    __shared__ double a_i, a_j, r_i[3], r_j[3], S_ij_00, S1, inv_S1;
    __shared__ double PA_0, PA_1, PB_0, PB_1;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta[0][0] = 1.0; delta[0][1] = 0.0; delta[0][2] = 0.0;
        delta[1][0] = 0.0; delta[1][1] = 1.0; delta[1][2] = 0.0;
        delta[2][0] = 0.0; delta[2][1] = 0.0; delta[2][2] = 1.0;

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
            inv_S1 = 1.0 / S1;

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

    for (uint32_t m = 0; m < prec_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= dd_prim_pair_count))
        {
            break;
        }

        const auto k = dd_first_inds[kl];
        const auto l = dd_second_inds[kl];

        const auto a_k = d_prim_info[k / 6 + d_prim_count * 0];

        const double r_k[3] = {d_prim_info[k / 6 + d_prim_count * 2],
                               d_prim_info[k / 6 + d_prim_count * 3],
                               d_prim_info[k / 6 + d_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = dd_pair_data[kl];

        const auto c0 = d_cart_inds[k % 6][0];
        const auto c1 = d_cart_inds[k % 6][1];
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F8_t[5];

        gpu::computeBoysFunction(F8_t, S1 * S2 * inv_S4 * r2_PQ, 4, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QC_1 = (a_l * inv_S2) * (r_l[c1] - r_k[c1]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                    + F8_t[4] * 0.25 * S1 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        -2*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])

                        -PQ[a0]*(PA_1*(PQ[b0]*(delta[b1][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[b1][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[b1][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[b1][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))) + PQ[b1]*(delta[b0][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[b0][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[b0][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[b0][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0)))) + PB_0*(PQ[a1]*(delta[b1][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[b1][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[b1][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[b1][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))) + PQ[b1]*(delta[a1][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[a1][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[a1][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[a1][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0)))) + PB_1*(PQ[a1]*(delta[b0][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[b0][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[b0][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[b0][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))) + PQ[b0]*(delta[a1][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[a1][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[a1][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[a1][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))))) - PQ[a1]*(PA_0*(PQ[b0]*(delta[b1][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[b1][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[b1][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[b1][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))) + PQ[b1]*(delta[b0][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[b0][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[b0][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[b0][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0)))) + (PB_0*PQ[b1] + PB_1*PQ[b0])*(delta[a0][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[a0][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[a0][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[a0][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0)))) - PQ[b0]*PQ[b1]*(PA_0*(delta[a1][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[a1][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[a1][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[a1][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))) + PA_1*(delta[a0][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[a0][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[a0][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[a0][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))))

                        +(PQ[c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + PQ[c1]*(QC_0*delta[d0][d1] + delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0)) + PQ[d0]*(QC_0*delta[c1][d1] + QC_1*delta[c0][d1] + delta[c0][c1]*(PQ[d1] + QD_1)) + PQ[d1]*(QC_0*delta[c1][d0] + QC_1*delta[c0][d0] + QD_0*delta[c0][c1]))*(-PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) - PA_1*(PQ[a0]*delta[b0][b1] + PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) - PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) - PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]) + PQ[a0]*PQ[a1]*delta[b0][b1] + PQ[a0]*PQ[b0]*delta[a1][b1] + PQ[a0]*PQ[b1]*delta[a1][b0] + PQ[a1]*PQ[b0]*delta[a0][b1] + PQ[a1]*PQ[b1]*delta[a0][b0] + PQ[b0]*PQ[b1]*delta[a0][a1])

                        +(delta[a0][c0]*delta[a1][c1] + delta[a0][c1]*delta[a1][c0])*(-PB_0*PQ[b1]*PQ[d0]*QD_1 - PB_0*PQ[b1]*PQ[d1]*QD_0 - PB_1*PQ[b0]*PQ[d0]*QD_1 - PB_1*PQ[b0]*PQ[d1]*QD_0) + (delta[a0][c0]*delta[a1][d0] + delta[a0][d0]*delta[a1][c0])*(-PB_0*PQ[b1]*PQ[c1]*QD_1 - PB_0*PQ[b1]*PQ[d1]*QC_1 - PB_1*PQ[b0]*PQ[c1]*QD_1 - PB_1*PQ[b0]*PQ[d1]*QC_1) + (delta[a0][c0]*delta[a1][d1] + delta[a0][d1]*delta[a1][c0])*(-PB_0*PQ[b1]*PQ[c1]*QD_0 - PB_0*PQ[b1]*PQ[d0]*QC_1 - PB_1*PQ[b0]*PQ[c1]*QD_0 - PB_1*PQ[b0]*PQ[d0]*QC_1) + (delta[a0][c0]*delta[b0][c1] + delta[a0][c1]*delta[b0][c0])*(-PA_1*PQ[b1]*PQ[d0]*QD_1 - PA_1*PQ[b1]*PQ[d1]*QD_0 - PB_1*PQ[a1]*PQ[d0]*QD_1 - PB_1*PQ[a1]*PQ[d1]*QD_0) + (delta[a0][c0]*delta[b0][d0] + delta[a0][d0]*delta[b0][c0])*(-PA_1*PQ[b1]*PQ[c1]*QD_1 - PA_1*PQ[b1]*PQ[d1]*QC_1 - PB_1*PQ[a1]*PQ[c1]*QD_1 - PB_1*PQ[a1]*PQ[d1]*QC_1) + (delta[a0][c0]*delta[b0][d1] + delta[a0][d1]*delta[b0][c0])*(-PA_1*PQ[b1]*PQ[c1]*QD_0 - PA_1*PQ[b1]*PQ[d0]*QC_1 - PB_1*PQ[a1]*PQ[c1]*QD_0 - PB_1*PQ[a1]*PQ[d0]*QC_1) + (delta[a0][c0]*delta[b1][c1] + delta[a0][c1]*delta[b1][c0])*(-PA_1*PQ[b0]*PQ[d0]*QD_1 - PA_1*PQ[b0]*PQ[d1]*QD_0 - PB_0*PQ[a1]*PQ[d0]*QD_1 - PB_0*PQ[a1]*PQ[d1]*QD_0) + (delta[a0][c0]*delta[b1][d0] + delta[a0][d0]*delta[b1][c0])*(-PA_1*PQ[b0]*PQ[c1]*QD_1 - PA_1*PQ[b0]*PQ[d1]*QC_1 - PB_0*PQ[a1]*PQ[c1]*QD_1 - PB_0*PQ[a1]*PQ[d1]*QC_1) + (delta[a0][c0]*delta[b1][d1] + delta[a0][d1]*delta[b1][c0])*(-PA_1*PQ[b0]*PQ[c1]*QD_0 - PA_1*PQ[b0]*PQ[d0]*QC_1 - PB_0*PQ[a1]*PQ[c1]*QD_0 - PB_0*PQ[a1]*PQ[d0]*QC_1) + (delta[a0][c1]*delta[a1][d0] + delta[a0][d0]*delta[a1][c1])*(-PB_0*PQ[b1]*PQ[c0]*QD_1 - PB_0*PQ[b1]*PQ[d1]*QC_0 - PB_1*PQ[b0]*PQ[c0]*QD_1 - PB_1*PQ[b0]*PQ[d1]*QC_0) + (delta[a0][c1]*delta[a1][d1] + delta[a0][d1]*delta[a1][c1])*(-PB_0*PQ[b1]*PQ[c0]*QD_0 - PB_0*PQ[b1]*PQ[d0]*QC_0 - PB_1*PQ[b0]*PQ[c0]*QD_0 - PB_1*PQ[b0]*PQ[d0]*QC_0) + (delta[a0][c1]*delta[b0][d0] + delta[a0][d0]*delta[b0][c1])*(-PA_1*PQ[b1]*PQ[c0]*QD_1 - PA_1*PQ[b1]*PQ[d1]*QC_0 - PB_1*PQ[a1]*PQ[c0]*QD_1 - PB_1*PQ[a1]*PQ[d1]*QC_0) + (delta[a0][c1]*delta[b0][d1] + delta[a0][d1]*delta[b0][c1])*(-PA_1*PQ[b1]*PQ[c0]*QD_0 - PA_1*PQ[b1]*PQ[d0]*QC_0 - PB_1*PQ[a1]*PQ[c0]*QD_0 - PB_1*PQ[a1]*PQ[d0]*QC_0) + (delta[a0][c1]*delta[b1][d0] + delta[a0][d0]*delta[b1][c1])*(-PA_1*PQ[b0]*PQ[c0]*QD_1 - PA_1*PQ[b0]*PQ[d1]*QC_0 - PB_0*PQ[a1]*PQ[c0]*QD_1 - PB_0*PQ[a1]*PQ[d1]*QC_0) + (delta[a0][c1]*delta[b1][d1] + delta[a0][d1]*delta[b1][c1])*(-PA_1*PQ[b0]*PQ[c0]*QD_0 - PA_1*PQ[b0]*PQ[d0]*QC_0 - PB_0*PQ[a1]*PQ[c0]*QD_0 - PB_0*PQ[a1]*PQ[d0]*QC_0) + (delta[a0][d0]*delta[a1][d1] + delta[a0][d1]*delta[a1][d0])*(-PB_0*PQ[b1]*PQ[c0]*QC_1 - PB_0*PQ[b1]*PQ[c1]*QC_0 - PB_1*PQ[b0]*PQ[c0]*QC_1 - PB_1*PQ[b0]*PQ[c1]*QC_0) + (delta[a0][d0]*delta[b0][d1] + delta[a0][d1]*delta[b0][d0])*(-PA_1*PQ[b1]*PQ[c0]*QC_1 - PA_1*PQ[b1]*PQ[c1]*QC_0 - PB_1*PQ[a1]*PQ[c0]*QC_1 - PB_1*PQ[a1]*PQ[c1]*QC_0) + (delta[a0][d0]*delta[b1][d1] + delta[a0][d1]*delta[b1][d0])*(-PA_1*PQ[b0]*PQ[c0]*QC_1 - PA_1*PQ[b0]*PQ[c1]*QC_0 - PB_0*PQ[a1]*PQ[c0]*QC_1 - PB_0*PQ[a1]*PQ[c1]*QC_0) + (delta[a1][c0]*delta[b0][c1] + delta[a1][c1]*delta[b0][c0])*(-PA_0*PQ[b1]*PQ[d0]*QD_1 - PA_0*PQ[b1]*PQ[d1]*QD_0 - PB_1*PQ[a0]*PQ[d0]*QD_1 - PB_1*PQ[a0]*PQ[d1]*QD_0) + (delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0])*(-PA_0*PQ[b1]*PQ[c1]*QD_1 - PA_0*PQ[b1]*PQ[d1]*QC_1 - PB_1*PQ[a0]*PQ[c1]*QD_1 - PB_1*PQ[a0]*PQ[d1]*QC_1) + (delta[a1][c0]*delta[b0][d1] + delta[a1][d1]*delta[b0][c0])*(-PA_0*PQ[b1]*PQ[c1]*QD_0 - PA_0*PQ[b1]*PQ[d0]*QC_1 - PB_1*PQ[a0]*PQ[c1]*QD_0 - PB_1*PQ[a0]*PQ[d0]*QC_1) + (delta[a1][c0]*delta[b1][c1] + delta[a1][c1]*delta[b1][c0])*(-PA_0*PQ[b0]*PQ[d0]*QD_1 - PA_0*PQ[b0]*PQ[d1]*QD_0 - PB_0*PQ[a0]*PQ[d0]*QD_1 - PB_0*PQ[a0]*PQ[d1]*QD_0) + (delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0])*(-PA_0*PQ[b0]*PQ[c1]*QD_1 - PA_0*PQ[b0]*PQ[d1]*QC_1 - PB_0*PQ[a0]*PQ[c1]*QD_1 - PB_0*PQ[a0]*PQ[d1]*QC_1) + (delta[a1][c0]*delta[b1][d1] + delta[a1][d1]*delta[b1][c0])*(-PA_0*PQ[b0]*PQ[c1]*QD_0 - PA_0*PQ[b0]*PQ[d0]*QC_1 - PB_0*PQ[a0]*PQ[c1]*QD_0 - PB_0*PQ[a0]*PQ[d0]*QC_1) + (delta[a1][c1]*delta[b0][d0] + delta[a1][d0]*delta[b0][c1])*(-PA_0*PQ[b1]*PQ[c0]*QD_1 - PA_0*PQ[b1]*PQ[d1]*QC_0 - PB_1*PQ[a0]*PQ[c0]*QD_1 - PB_1*PQ[a0]*PQ[d1]*QC_0) + (delta[a1][c1]*delta[b0][d1] + delta[a1][d1]*delta[b0][c1])*(-PA_0*PQ[b1]*PQ[c0]*QD_0 - PA_0*PQ[b1]*PQ[d0]*QC_0 - PB_1*PQ[a0]*PQ[c0]*QD_0 - PB_1*PQ[a0]*PQ[d0]*QC_0) + (delta[a1][c1]*delta[b1][d0] + delta[a1][d0]*delta[b1][c1])*(-PA_0*PQ[b0]*PQ[c0]*QD_1 - PA_0*PQ[b0]*PQ[d1]*QC_0 - PB_0*PQ[a0]*PQ[c0]*QD_1 - PB_0*PQ[a0]*PQ[d1]*QC_0) + (delta[a1][c1]*delta[b1][d1] + delta[a1][d1]*delta[b1][c1])*(-PA_0*PQ[b0]*PQ[c0]*QD_0 - PA_0*PQ[b0]*PQ[d0]*QC_0 - PB_0*PQ[a0]*PQ[c0]*QD_0 - PB_0*PQ[a0]*PQ[d0]*QC_0) + (delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0])*(-PA_0*PQ[b1]*PQ[c0]*QC_1 - PA_0*PQ[b1]*PQ[c1]*QC_0 - PB_1*PQ[a0]*PQ[c0]*QC_1 - PB_1*PQ[a0]*PQ[c1]*QC_0) + (delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0])*(-PA_0*PQ[b0]*PQ[c0]*QC_1 - PA_0*PQ[b0]*PQ[c1]*QC_0 - PB_0*PQ[a0]*PQ[c0]*QC_1 - PB_0*PQ[a0]*PQ[c1]*QC_0) + (delta[b0][c0]*delta[b1][c1] + delta[b0][c1]*delta[b1][c0])*(-PA_0*PQ[a1]*PQ[d0]*QD_1 - PA_0*PQ[a1]*PQ[d1]*QD_0 - PA_1*PQ[a0]*PQ[d0]*QD_1 - PA_1*PQ[a0]*PQ[d1]*QD_0) + (delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0])*(-PA_0*PQ[a1]*PQ[c1]*QD_1 - PA_0*PQ[a1]*PQ[d1]*QC_1 - PA_1*PQ[a0]*PQ[c1]*QD_1 - PA_1*PQ[a0]*PQ[d1]*QC_1) + (delta[b0][c0]*delta[b1][d1] + delta[b0][d1]*delta[b1][c0])*(-PA_0*PQ[a1]*PQ[c1]*QD_0 - PA_0*PQ[a1]*PQ[d0]*QC_1 - PA_1*PQ[a0]*PQ[c1]*QD_0 - PA_1*PQ[a0]*PQ[d0]*QC_1) + (delta[b0][c1]*delta[b1][d0] + delta[b0][d0]*delta[b1][c1])*(-PA_0*PQ[a1]*PQ[c0]*QD_1 - PA_0*PQ[a1]*PQ[d1]*QC_0 - PA_1*PQ[a0]*PQ[c0]*QD_1 - PA_1*PQ[a0]*PQ[d1]*QC_0) + (delta[b0][c1]*delta[b1][d1] + delta[b0][d1]*delta[b1][c1])*(-PA_0*PQ[a1]*PQ[c0]*QD_0 - PA_0*PQ[a1]*PQ[d0]*QC_0 - PA_1*PQ[a0]*PQ[c0]*QD_0 - PA_1*PQ[a0]*PQ[d0]*QC_0) + (delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0])*(-PA_0*PQ[a1]*PQ[c0]*QC_1 - PA_0*PQ[a1]*PQ[c1]*QC_0 - PA_1*PQ[a0]*PQ[c0]*QC_1 - PA_1*PQ[a0]*PQ[c1]*QC_0)

                        +(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0])*(-PB_1*PQ[c1]*PQ[d0]*QD_1 - PB_1*PQ[c1]*PQ[d1]*QD_0 - PB_1*PQ[d0]*PQ[d1]*QC_1 + PQ[b1]*PQ[c1]*PQ[d0]*QD_1 + PQ[b1]*PQ[c1]*PQ[d1]*QD_0 + PQ[b1]*PQ[d0]*PQ[d1]*QC_1) + (delta[a0][a1]*delta[b0][c1] + delta[a0][b0]*delta[a1][c1] + delta[a0][c1]*delta[a1][b0])*(-PB_1*PQ[c0]*PQ[d0]*QD_1 - PB_1*PQ[c0]*PQ[d1]*QD_0 - PB_1*PQ[d0]*PQ[d1]*QC_0 + PQ[b1]*PQ[c0]*PQ[d0]*QD_1 + PQ[b1]*PQ[c0]*PQ[d1]*QD_0 + PQ[b1]*PQ[d0]*PQ[d1]*QC_0) + (delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])*(-PB_1*PQ[c0]*PQ[c1]*QD_1 - PB_1*PQ[c0]*PQ[d1]*QC_1 - PB_1*PQ[c1]*PQ[d1]*QC_0 + PQ[b1]*PQ[c0]*PQ[c1]*QD_1 + PQ[b1]*PQ[c0]*PQ[d1]*QC_1 + PQ[b1]*PQ[c1]*PQ[d1]*QC_0) + (delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0])*(-PB_1*PQ[c0]*PQ[c1]*QD_0 - PB_1*PQ[c0]*PQ[d0]*QC_1 - PB_1*PQ[c1]*PQ[d0]*QC_0 + PQ[b1]*PQ[c0]*PQ[c1]*QD_0 + PQ[b1]*PQ[c0]*PQ[d0]*QC_1 + PQ[b1]*PQ[c1]*PQ[d0]*QC_0) + (delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1])*(-PB_0*PQ[c1]*PQ[d0]*QD_1 - PB_0*PQ[c1]*PQ[d1]*QD_0 - PB_0*PQ[d0]*PQ[d1]*QC_1 + PQ[b0]*PQ[c1]*PQ[d0]*QD_1 + PQ[b0]*PQ[c1]*PQ[d1]*QD_0 + PQ[b0]*PQ[d0]*PQ[d1]*QC_1) + (delta[a0][a1]*delta[b1][c1] + delta[a0][b1]*delta[a1][c1] + delta[a0][c1]*delta[a1][b1])*(-PB_0*PQ[c0]*PQ[d0]*QD_1 - PB_0*PQ[c0]*PQ[d1]*QD_0 - PB_0*PQ[d0]*PQ[d1]*QC_0 + PQ[b0]*PQ[c0]*PQ[d0]*QD_1 + PQ[b0]*PQ[c0]*PQ[d1]*QD_0 + PQ[b0]*PQ[d0]*PQ[d1]*QC_0) + (delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1])*(-PB_0*PQ[c0]*PQ[c1]*QD_1 - PB_0*PQ[c0]*PQ[d1]*QC_1 - PB_0*PQ[c1]*PQ[d1]*QC_0 + PQ[b0]*PQ[c0]*PQ[c1]*QD_1 + PQ[b0]*PQ[c0]*PQ[d1]*QC_1 + PQ[b0]*PQ[c1]*PQ[d1]*QC_0) + (delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1])*(-PB_0*PQ[c0]*PQ[c1]*QD_0 - PB_0*PQ[c0]*PQ[d0]*QC_1 - PB_0*PQ[c1]*PQ[d0]*QC_0 + PQ[b0]*PQ[c0]*PQ[c1]*QD_0 + PQ[b0]*PQ[c0]*PQ[d0]*QC_1 + PQ[b0]*PQ[c1]*PQ[d0]*QC_0) + (delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1])*(-PA_1*PQ[c1]*PQ[d0]*QD_1 - PA_1*PQ[c1]*PQ[d1]*QD_0 - PA_1*PQ[d0]*PQ[d1]*QC_1 + PQ[a1]*PQ[c1]*PQ[d0]*QD_1 + PQ[a1]*PQ[c1]*PQ[d1]*QD_0 + PQ[a1]*PQ[d0]*PQ[d1]*QC_1) + (delta[a0][b0]*delta[b1][c1] + delta[a0][b1]*delta[b0][c1] + delta[a0][c1]*delta[b0][b1])*(-PA_1*PQ[c0]*PQ[d0]*QD_1 - PA_1*PQ[c0]*PQ[d1]*QD_0 - PA_1*PQ[d0]*PQ[d1]*QC_0 + PQ[a1]*PQ[c0]*PQ[d0]*QD_1 + PQ[a1]*PQ[c0]*PQ[d1]*QD_0 + PQ[a1]*PQ[d0]*PQ[d1]*QC_0) + (delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1])*(-PA_1*PQ[c0]*PQ[c1]*QD_1 - PA_1*PQ[c0]*PQ[d1]*QC_1 - PA_1*PQ[c1]*PQ[d1]*QC_0 + PQ[a1]*PQ[c0]*PQ[c1]*QD_1 + PQ[a1]*PQ[c0]*PQ[d1]*QC_1 + PQ[a1]*PQ[c1]*PQ[d1]*QC_0) + (delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1])*(-PA_1*PQ[c0]*PQ[c1]*QD_0 - PA_1*PQ[c0]*PQ[d0]*QC_1 - PA_1*PQ[c1]*PQ[d0]*QC_0 + PQ[a1]*PQ[c0]*PQ[c1]*QD_0 + PQ[a1]*PQ[c0]*PQ[d0]*QC_1 + PQ[a1]*PQ[c1]*PQ[d0]*QC_0) + (delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1])*(-PA_0*PQ[c1]*PQ[d0]*QD_1 - PA_0*PQ[c1]*PQ[d1]*QD_0 - PA_0*PQ[d0]*PQ[d1]*QC_1 + PQ[a0]*PQ[c1]*PQ[d0]*QD_1 + PQ[a0]*PQ[c1]*PQ[d1]*QD_0 + PQ[a0]*PQ[d0]*PQ[d1]*QC_1) + (delta[a1][b0]*delta[b1][c1] + delta[a1][b1]*delta[b0][c1] + delta[a1][c1]*delta[b0][b1])*(-PA_0*PQ[c0]*PQ[d0]*QD_1 - PA_0*PQ[c0]*PQ[d1]*QD_0 - PA_0*PQ[d0]*PQ[d1]*QC_0 + PQ[a0]*PQ[c0]*PQ[d0]*QD_1 + PQ[a0]*PQ[c0]*PQ[d1]*QD_0 + PQ[a0]*PQ[d0]*PQ[d1]*QC_0) + (delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1])*(-PA_0*PQ[c0]*PQ[c1]*QD_1 - PA_0*PQ[c0]*PQ[d1]*QC_1 - PA_0*PQ[c1]*PQ[d1]*QC_0 + PQ[a0]*PQ[c0]*PQ[c1]*QD_1 + PQ[a0]*PQ[c0]*PQ[d1]*QC_1 + PQ[a0]*PQ[c1]*PQ[d1]*QC_0) + (delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1])*(-PA_0*PQ[c0]*PQ[c1]*QD_0 - PA_0*PQ[c0]*PQ[d0]*QC_1 - PA_0*PQ[c1]*PQ[d0]*QC_0 + PQ[a0]*PQ[c0]*PQ[c1]*QD_0 + PQ[a0]*PQ[c0]*PQ[d0]*QC_1 + PQ[a0]*PQ[c1]*PQ[d0]*QC_0)

                        +2.0*(PQ[c0]*PQ[c1]*(PQ[d0]*QD_1 + PQ[d1]*QD_0) + PQ[d0]*PQ[d1]*(PQ[c0]*QC_1 + PQ[c1]*QC_0))*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * dd_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDD17_FP64(double*         mat_J,
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
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ double   delta[3][3];

    __shared__ double a_i, a_j, r_i[3], r_j[3], S_ij_00, S1, inv_S1;
    __shared__ double PA_0, PA_1, PB_0, PB_1;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta[0][0] = 1.0; delta[0][1] = 0.0; delta[0][2] = 0.0;
        delta[1][0] = 0.0; delta[1][1] = 1.0; delta[1][2] = 0.0;
        delta[2][0] = 0.0; delta[2][1] = 0.0; delta[2][2] = 1.0;

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
            inv_S1 = 1.0 / S1;

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

    for (uint32_t m = 0; m < prec_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= dd_prim_pair_count))
        {
            break;
        }

        const auto k = dd_first_inds[kl];
        const auto l = dd_second_inds[kl];

        const auto a_k = d_prim_info[k / 6 + d_prim_count * 0];

        const double r_k[3] = {d_prim_info[k / 6 + d_prim_count * 2],
                               d_prim_info[k / 6 + d_prim_count * 3],
                               d_prim_info[k / 6 + d_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = dd_pair_data[kl];

        const auto c0 = d_cart_inds[k % 6][0];
        const auto c1 = d_cart_inds[k % 6][1];
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F8_t[5];

        gpu::computeBoysFunction(F8_t, S1 * S2 * inv_S4 * r2_PQ, 4, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QC_1 = (a_l * inv_S2) * (r_l[c1] - r_k[c1]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                    + F8_t[4] * 0.25 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        +PQ[a0]*(PQ[a1]*(PQ[b0]*(QC_0*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + QC_1*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + QD_0*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + QD_1*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) + PQ[b1]*(QC_0*(delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0]) + QC_1*(delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0]) + QD_0*(delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1]) + QD_1*(delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1])) + delta[b0][b1]*(QC_0*(PQ[c1]*delta[d0][d1] + PQ[d0]*delta[c1][d1] + PQ[d1]*delta[c1][d0]) + QC_1*(PQ[d0]*delta[c0][d1] + PQ[d1]*delta[c0][d0] + delta[d0][d1]*(PQ[c0] + QC_0)) + QD_0*(PQ[d1]*delta[c0][c1] + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + QD_1*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0)))) + (PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0])*(QC_0*(PQ[c1]*delta[d0][d1] + PQ[d0]*delta[c1][d1] + PQ[d1]*delta[c1][d0]) + QC_1*(PQ[d0]*delta[c0][d1] + PQ[d1]*delta[c0][d0] + delta[d0][d1]*(PQ[c0] + QC_0)) + QD_0*(PQ[d1]*delta[c0][c1] + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + QD_1*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0)))) + PQ[a1]*(PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0])*(QC_0*(PQ[c1]*delta[d0][d1] + PQ[d0]*delta[c1][d1] + PQ[d1]*delta[c1][d0]) + QC_1*(PQ[d0]*delta[c0][d1] + PQ[d1]*delta[c0][d0] + delta[d0][d1]*(PQ[c0] + QC_0)) + QD_0*(PQ[d1]*delta[c0][c1] + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + QD_1*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))) + PQ[b0]*PQ[b1]*(PQ[a0]*(QC_0*(delta[a1][c1]*delta[d0][d1] + delta[a1][d0]*delta[c1][d1] + delta[a1][d1]*delta[c1][d0]) + QC_1*(delta[a1][c0]*delta[d0][d1] + delta[a1][d0]*delta[c0][d1] + delta[a1][d1]*delta[c0][d0]) + QD_0*(delta[a1][c0]*delta[c1][d1] + delta[a1][c1]*delta[c0][d1] + delta[a1][d1]*delta[c0][c1]) + QD_1*(delta[a1][c0]*delta[c1][d0] + delta[a1][c1]*delta[c0][d0] + delta[a1][d0]*delta[c0][c1])) + PQ[a1]*(QC_0*(delta[a0][c1]*delta[d0][d1] + delta[a0][d0]*delta[c1][d1] + delta[a0][d1]*delta[c1][d0]) + QC_1*(delta[a0][c0]*delta[d0][d1] + delta[a0][d0]*delta[c0][d1] + delta[a0][d1]*delta[c0][d0]) + QD_0*(delta[a0][c0]*delta[c1][d1] + delta[a0][c1]*delta[c0][d1] + delta[a0][d1]*delta[c0][c1]) + QD_1*(delta[a0][c0]*delta[c1][d0] + delta[a0][c1]*delta[c0][d0] + delta[a0][d0]*delta[c0][c1])) + delta[a0][a1]*(QC_0*(PQ[c1]*delta[d0][d1] + PQ[d0]*delta[c1][d1] + PQ[d1]*delta[c1][d0]) + QC_1*(PQ[d0]*delta[c0][d1] + PQ[d1]*delta[c0][d0] + delta[d0][d1]*(PQ[c0] + QC_0)) + QD_0*(PQ[d1]*delta[c0][c1] + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + QD_1*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))))

                        +PQ[a0]*(QC_0*(PQ[a1]*(QC_1*(delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + QD_0*(delta[b0][c1]*delta[b1][d1] + delta[b0][d1]*delta[b1][c1]) + QD_1*(delta[b0][c1]*delta[b1][d0] + delta[b0][d0]*delta[b1][c1])) + PQ[b0]*(QC_1*(delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + QD_0*(delta[a1][c1]*delta[b1][d1] + delta[a1][d1]*delta[b1][c1]) + QD_1*(delta[a1][c1]*delta[b1][d0] + delta[a1][d0]*delta[b1][c1])) + PQ[b1]*(QC_1*(delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + QD_0*(delta[a1][c1]*delta[b0][d1] + delta[a1][d1]*delta[b0][c1]) + QD_1*(delta[a1][c1]*delta[b0][d0] + delta[a1][d0]*delta[b0][c1]))) + QC_1*(QD_0*(PQ[a1]*(delta[b0][c0]*delta[b1][d1] + delta[b0][d1]*delta[b1][c0]) + PQ[b0]*(delta[a1][c0]*delta[b1][d1] + delta[a1][d1]*delta[b1][c0]) + PQ[b1]*(delta[a1][c0]*delta[b0][d1] + delta[a1][d1]*delta[b0][c0])) + QD_1*(PQ[a1]*(delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0]) + PQ[b0]*(delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + PQ[b1]*(delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0])))) + PQ[a1]*(PQ[b0]*(QC_0*(QC_1*(delta[a0][d0]*delta[b1][d1] + delta[a0][d1]*delta[b1][d0]) + QD_0*(delta[a0][c1]*delta[b1][d1] + delta[a0][d1]*delta[b1][c1]) + QD_1*(delta[a0][c1]*delta[b1][d0] + delta[a0][d0]*delta[b1][c1])) + QC_1*(QD_0*(delta[a0][c0]*delta[b1][d1] + delta[a0][d1]*delta[b1][c0]) + QD_1*(delta[a0][c0]*delta[b1][d0] + delta[a0][d0]*delta[b1][c0]))) + PQ[b1]*(QC_0*(QC_1*(delta[a0][d0]*delta[b0][d1] + delta[a0][d1]*delta[b0][d0]) + QD_0*(delta[a0][c1]*delta[b0][d1] + delta[a0][d1]*delta[b0][c1]) + QD_1*(delta[a0][c1]*delta[b0][d0] + delta[a0][d0]*delta[b0][c1])) + QC_1*(QD_0*(delta[a0][c0]*delta[b0][d1] + delta[a0][d1]*delta[b0][c0]) + QD_1*(delta[a0][c0]*delta[b0][d0] + delta[a0][d0]*delta[b0][c0])))) + PQ[b0]*PQ[b1]*(QC_0*(QC_1*(delta[a0][d0]*delta[a1][d1] + delta[a0][d1]*delta[a1][d0]) + QD_0*(delta[a0][c1]*delta[a1][d1] + delta[a0][d1]*delta[a1][c1]) + QD_1*(delta[a0][c1]*delta[a1][d0] + delta[a0][d0]*delta[a1][c1])) + QC_1*(QD_0*(delta[a0][c0]*delta[a1][d1] + delta[a0][d1]*delta[a1][c0]) + QD_1*(delta[a0][c0]*delta[a1][d0] + delta[a0][d0]*delta[a1][c0]))) + QD_0*QD_1*(PQ[a0]*(PQ[a1]*(delta[b0][c0]*delta[b1][c1] + delta[b0][c1]*delta[b1][c0]) + PQ[b0]*(delta[a1][c0]*delta[b1][c1] + delta[a1][c1]*delta[b1][c0]) + PQ[b1]*(delta[a1][c0]*delta[b0][c1] + delta[a1][c1]*delta[b0][c0])) + PQ[a1]*(PQ[b0]*(delta[a0][c0]*delta[b1][c1] + delta[a0][c1]*delta[b1][c0]) + PQ[b1]*(delta[a0][c0]*delta[b0][c1] + delta[a0][c1]*delta[b0][c0])) + PQ[b0]*PQ[b1]*(delta[a0][c0]*delta[a1][c1] + delta[a0][c1]*delta[a1][c0]))

                        +(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0])*(PQ[b1]*PQ[c1]*QD_0*QD_1 + PQ[b1]*PQ[d0]*QC_1*QD_1 + PQ[b1]*PQ[d1]*QC_1*QD_0) + (delta[a0][a1]*delta[b0][c1] + delta[a0][b0]*delta[a1][c1] + delta[a0][c1]*delta[a1][b0])*(PQ[b1]*PQ[c0]*QD_0*QD_1 + PQ[b1]*PQ[d0]*QC_0*QD_1 + PQ[b1]*PQ[d1]*QC_0*QD_0) + (delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])*(PQ[b1]*PQ[c0]*QC_1*QD_1 + PQ[b1]*PQ[c1]*QC_0*QD_1 + PQ[b1]*PQ[d1]*QC_0*QC_1) + (delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0])*(PQ[b1]*PQ[c0]*QC_1*QD_0 + PQ[b1]*PQ[c1]*QC_0*QD_0 + PQ[b1]*PQ[d0]*QC_0*QC_1) + (delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1])*(PQ[b0]*PQ[c1]*QD_0*QD_1 + PQ[b0]*PQ[d0]*QC_1*QD_1 + PQ[b0]*PQ[d1]*QC_1*QD_0) + (delta[a0][a1]*delta[b1][c1] + delta[a0][b1]*delta[a1][c1] + delta[a0][c1]*delta[a1][b1])*(PQ[b0]*PQ[c0]*QD_0*QD_1 + PQ[b0]*PQ[d0]*QC_0*QD_1 + PQ[b0]*PQ[d1]*QC_0*QD_0) + (delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1])*(PQ[b0]*PQ[c0]*QC_1*QD_1 + PQ[b0]*PQ[c1]*QC_0*QD_1 + PQ[b0]*PQ[d1]*QC_0*QC_1) + (delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1])*(PQ[b0]*PQ[c0]*QC_1*QD_0 + PQ[b0]*PQ[c1]*QC_0*QD_0 + PQ[b0]*PQ[d0]*QC_0*QC_1) + (delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1])*(PQ[a1]*PQ[c1]*QD_0*QD_1 + PQ[a1]*PQ[d0]*QC_1*QD_1 + PQ[a1]*PQ[d1]*QC_1*QD_0) + (delta[a0][b0]*delta[b1][c1] + delta[a0][b1]*delta[b0][c1] + delta[a0][c1]*delta[b0][b1])*(PQ[a1]*PQ[c0]*QD_0*QD_1 + PQ[a1]*PQ[d0]*QC_0*QD_1 + PQ[a1]*PQ[d1]*QC_0*QD_0) + (delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1])*(PQ[a1]*PQ[c0]*QC_1*QD_1 + PQ[a1]*PQ[c1]*QC_0*QD_1 + PQ[a1]*PQ[d1]*QC_0*QC_1) + (delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1])*(PQ[a1]*PQ[c0]*QC_1*QD_0 + PQ[a1]*PQ[c1]*QC_0*QD_0 + PQ[a1]*PQ[d0]*QC_0*QC_1) + (delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1])*(PQ[a0]*PQ[c1]*QD_0*QD_1 + PQ[a0]*PQ[d0]*QC_1*QD_1 + PQ[a0]*PQ[d1]*QC_1*QD_0) + (delta[a1][b0]*delta[b1][c1] + delta[a1][b1]*delta[b0][c1] + delta[a1][c1]*delta[b0][b1])*(PQ[a0]*PQ[c0]*QD_0*QD_1 + PQ[a0]*PQ[d0]*QC_0*QD_1 + PQ[a0]*PQ[d1]*QC_0*QD_0) + (delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1])*(PQ[a0]*PQ[c0]*QC_1*QD_1 + PQ[a0]*PQ[c1]*QC_0*QD_1 + PQ[a0]*PQ[d1]*QC_0*QC_1) + (delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1])*(PQ[a0]*PQ[c0]*QC_1*QD_0 + PQ[a0]*PQ[c1]*QC_0*QD_0 + PQ[a0]*PQ[d0]*QC_0*QC_1)

                        +(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])*(PQ[c0]*QD_0*(PQ[c1]*QD_1 + PQ[d1]*QC_1) + PQ[c1]*QC_0*(PQ[d0]*QD_1 + PQ[d1]*QD_0) + PQ[d0]*QC_1*(PQ[c0]*QD_1 + PQ[d1]*QC_0))

                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * dd_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDD18_FP64(double*         mat_J,
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
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ double   delta[3][3];

    __shared__ double a_i, a_j, r_i[3], r_j[3], S_ij_00, S1, inv_S1;
    __shared__ double PA_0, PA_1, PB_0, PB_1;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta[0][0] = 1.0; delta[0][1] = 0.0; delta[0][2] = 0.0;
        delta[1][0] = 0.0; delta[1][1] = 1.0; delta[1][2] = 0.0;
        delta[2][0] = 0.0; delta[2][1] = 0.0; delta[2][2] = 1.0;

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
            inv_S1 = 1.0 / S1;

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

    for (uint32_t m = 0; m < prec_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= dd_prim_pair_count))
        {
            break;
        }

        const auto k = dd_first_inds[kl];
        const auto l = dd_second_inds[kl];

        const auto a_k = d_prim_info[k / 6 + d_prim_count * 0];

        const double r_k[3] = {d_prim_info[k / 6 + d_prim_count * 2],
                               d_prim_info[k / 6 + d_prim_count * 3],
                               d_prim_info[k / 6 + d_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = dd_pair_data[kl];

        const auto c0 = d_cart_inds[k % 6][0];
        const auto c1 = d_cart_inds[k % 6][1];
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F8_t[5];

        gpu::computeBoysFunction(F8_t, S1 * S2 * inv_S4 * r2_PQ, 4, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QC_1 = (a_l * inv_S2) * (r_l[c1] - r_k[c1]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                    + F8_t[4] * 0.5 * S1 * S1 * S1 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        -(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))*(PQ[c0]*(PQ[c1]*delta[d0][d1] + PQ[d0]*delta[c1][d1] + PQ[d1]*delta[c1][d0]) + PQ[c1]*(PQ[d0]*delta[c0][d1] + PQ[d1]*delta[c0][d0]) + PQ[d0]*PQ[d1]*delta[c0][c1])

                        -PA_0*PA_1*(PQ[c0]*PQ[c1]*(PB_0*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PB_1*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0])) + PQ[d0]*PQ[d1]*(PB_0*(PQ[c0]*delta[b1][c1] + PQ[c1]*delta[b1][c0]) + PB_1*(PQ[c0]*delta[b0][c1] + PQ[c1]*delta[b0][c0]))) - PB_0*PB_1*(PQ[c0]*PQ[c1]*(PA_0*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0]) + PA_1*(PQ[d0]*delta[a0][d1] + PQ[d1]*delta[a0][d0])) + PQ[d0]*PQ[d1]*(PA_0*(PQ[c0]*delta[a1][c1] + PQ[c1]*delta[a1][c0]) + PA_1*(PQ[c0]*delta[a0][c1] + PQ[c1]*delta[a0][c0])))

                        +PQ[c0]*PQ[c1]*PQ[d0]*PQ[d1]*(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1])

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * dd_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDD19_FP64(double*         mat_J,
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
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ double   delta[3][3];

    __shared__ double a_i, a_j, r_i[3], r_j[3], S_ij_00, S1, inv_S1;
    __shared__ double PA_0, PA_1, PB_0, PB_1;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta[0][0] = 1.0; delta[0][1] = 0.0; delta[0][2] = 0.0;
        delta[1][0] = 0.0; delta[1][1] = 1.0; delta[1][2] = 0.0;
        delta[2][0] = 0.0; delta[2][1] = 0.0; delta[2][2] = 1.0;

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
            inv_S1 = 1.0 / S1;

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

    for (uint32_t m = 0; m < prec_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= dd_prim_pair_count))
        {
            break;
        }

        const auto k = dd_first_inds[kl];
        const auto l = dd_second_inds[kl];

        const auto a_k = d_prim_info[k / 6 + d_prim_count * 0];

        const double r_k[3] = {d_prim_info[k / 6 + d_prim_count * 2],
                               d_prim_info[k / 6 + d_prim_count * 3],
                               d_prim_info[k / 6 + d_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = dd_pair_data[kl];

        const auto c0 = d_cart_inds[k % 6][0];
        const auto c1 = d_cart_inds[k % 6][1];
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F8_t[5];

        gpu::computeBoysFunction(F8_t, S1 * S2 * inv_S4 * r2_PQ, 4, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QC_1 = (a_l * inv_S2) * (r_l[c1] - r_k[c1]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                    + F8_t[4] * 0.5 * S1 * S1 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        +(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))*(PQ[c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + PQ[c1]*(QC_0*delta[d0][d1] + delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0)) + PQ[d0]*(QC_0*delta[c1][d1] + QC_1*delta[c0][d1] + delta[c0][c1]*(PQ[d1] + QD_1)) + PQ[d1]*(QC_0*delta[c1][d0] + QC_1*delta[c0][d0] + QD_0*delta[c0][c1]))

                        +PA_0*(PA_1*(PQ[b0]*(PQ[c0]*(PQ[c1]*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + PQ[d0]*(QC_1*delta[b1][d1] + QD_1*delta[b1][c1]) + PQ[d1]*(QC_1*delta[b1][d0] + QD_0*delta[b1][c1])) + PQ[c1]*(PQ[d0]*(QC_0*delta[b1][d1] + QD_1*delta[b1][c0]) + PQ[d1]*(QC_0*delta[b1][d0] + QD_0*delta[b1][c0]))) + PQ[b1]*(PQ[c0]*(PQ[c1]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + PQ[d0]*(QC_1*delta[b0][d1] + QD_1*delta[b0][c1]) + PQ[d1]*(QC_1*delta[b0][d0] + QD_0*delta[b0][c1])) + PQ[c1]*(PQ[d0]*(QC_0*delta[b0][d1] + QD_1*delta[b0][c0]) + PQ[d1]*(QC_0*delta[b0][d0] + QD_0*delta[b0][c0])))) + PB_0*(PQ[a1]*(PQ[c0]*(PQ[c1]*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + PQ[d0]*(QC_1*delta[b1][d1] + QD_1*delta[b1][c1]) + PQ[d1]*(QC_1*delta[b1][d0] + QD_0*delta[b1][c1])) + PQ[c1]*(PQ[d0]*(QC_0*delta[b1][d1] + QD_1*delta[b1][c0]) + PQ[d1]*(QC_0*delta[b1][d0] + QD_0*delta[b1][c0]))) + PQ[b1]*(PQ[c0]*(PQ[c1]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]) + PQ[d0]*(QC_1*delta[a1][d1] + QD_1*delta[a1][c1]) + PQ[d1]*(QC_1*delta[a1][d0] + QD_0*delta[a1][c1])) + PQ[c1]*(PQ[d0]*(QC_0*delta[a1][d1] + QD_1*delta[a1][c0]) + PQ[d1]*(QC_0*delta[a1][d0] + QD_0*delta[a1][c0])))) + PB_1*(PQ[a1]*(PQ[c0]*(PQ[c1]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + PQ[d0]*(QC_1*delta[b0][d1] + QD_1*delta[b0][c1]) + PQ[d1]*(QC_1*delta[b0][d0] + QD_0*delta[b0][c1])) + PQ[c1]*(PQ[d0]*(QC_0*delta[b0][d1] + QD_1*delta[b0][c0]) + PQ[d1]*(QC_0*delta[b0][d0] + QD_0*delta[b0][c0]))) + PQ[b0]*(PQ[c0]*(PQ[c1]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]) + PQ[d0]*(QC_1*delta[a1][d1] + QD_1*delta[a1][c1]) + PQ[d1]*(QC_1*delta[a1][d0] + QD_0*delta[a1][c1])) + PQ[c1]*(PQ[d0]*(QC_0*delta[a1][d1] + QD_1*delta[a1][c0]) + PQ[d1]*(QC_0*delta[a1][d0] + QD_0*delta[a1][c0]))))) + PA_1*(PB_0*(PQ[a0]*(PQ[c0]*(PQ[c1]*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + PQ[d0]*(QC_1*delta[b1][d1] + QD_1*delta[b1][c1]) + PQ[d1]*(QC_1*delta[b1][d0] + QD_0*delta[b1][c1])) + PQ[c1]*(PQ[d0]*(QC_0*delta[b1][d1] + QD_1*delta[b1][c0]) + PQ[d1]*(QC_0*delta[b1][d0] + QD_0*delta[b1][c0]))) + PQ[b1]*(PQ[c0]*(PQ[c1]*(QD_0*delta[a0][d1] + QD_1*delta[a0][d0]) + PQ[d0]*(QC_1*delta[a0][d1] + QD_1*delta[a0][c1]) + PQ[d1]*(QC_1*delta[a0][d0] + QD_0*delta[a0][c1])) + PQ[c1]*(PQ[d0]*(QC_0*delta[a0][d1] + QD_1*delta[a0][c0]) + PQ[d1]*(QC_0*delta[a0][d0] + QD_0*delta[a0][c0])))) + PB_1*(PQ[a0]*(PQ[c0]*(PQ[c1]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + PQ[d0]*(QC_1*delta[b0][d1] + QD_1*delta[b0][c1]) + PQ[d1]*(QC_1*delta[b0][d0] + QD_0*delta[b0][c1])) + PQ[c1]*(PQ[d0]*(QC_0*delta[b0][d1] + QD_1*delta[b0][c0]) + PQ[d1]*(QC_0*delta[b0][d0] + QD_0*delta[b0][c0]))) + PQ[b0]*(PQ[c0]*(PQ[c1]*(QD_0*delta[a0][d1] + QD_1*delta[a0][d0]) + PQ[d0]*(QC_1*delta[a0][d1] + QD_1*delta[a0][c1]) + PQ[d1]*(QC_1*delta[a0][d0] + QD_0*delta[a0][c1])) + PQ[c1]*(PQ[d0]*(QC_0*delta[a0][d1] + QD_1*delta[a0][c0]) + PQ[d1]*(QC_0*delta[a0][d0] + QD_0*delta[a0][c0]))))) + PB_0*PB_1*(PQ[a0]*(PQ[c0]*(PQ[c1]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]) + PQ[d0]*(QC_1*delta[a1][d1] + QD_1*delta[a1][c1]) + PQ[d1]*(QC_1*delta[a1][d0] + QD_0*delta[a1][c1])) + PQ[c1]*(PQ[d0]*(QC_0*delta[a1][d1] + QD_1*delta[a1][c0]) + PQ[d1]*(QC_0*delta[a1][d0] + QD_0*delta[a1][c0]))) + PQ[a1]*(PQ[c0]*(PQ[c1]*(QD_0*delta[a0][d1] + QD_1*delta[a0][d0]) + PQ[d0]*(QC_1*delta[a0][d1] + QD_1*delta[a0][c1]) + PQ[d1]*(QC_1*delta[a0][d0] + QD_0*delta[a0][c1])) + PQ[c1]*(PQ[d0]*(QC_0*delta[a0][d1] + QD_1*delta[a0][c0]) + PQ[d1]*(QC_0*delta[a0][d0] + QD_0*delta[a0][c0])))) + PQ[d0]*PQ[d1]*(PA_0*(PA_1*(PQ[b0]*(QC_0*delta[b1][c1] + QC_1*delta[b1][c0]) + PQ[b1]*(QC_0*delta[b0][c1] + QC_1*delta[b0][c0])) + PB_0*(PQ[a1]*(QC_0*delta[b1][c1] + QC_1*delta[b1][c0]) + PQ[b1]*(QC_0*delta[a1][c1] + QC_1*delta[a1][c0])) + PB_1*(PQ[a1]*(QC_0*delta[b0][c1] + QC_1*delta[b0][c0]) + PQ[b0]*(QC_0*delta[a1][c1] + QC_1*delta[a1][c0]))) + PA_1*(PB_0*(PQ[a0]*(QC_0*delta[b1][c1] + QC_1*delta[b1][c0]) + PQ[b1]*(QC_0*delta[a0][c1] + QC_1*delta[a0][c0])) + PB_1*(PQ[a0]*(QC_0*delta[b0][c1] + QC_1*delta[b0][c0]) + PQ[b0]*(QC_0*delta[a0][c1] + QC_1*delta[a0][c0]))) + PB_0*PB_1*(PQ[a0]*(QC_0*delta[a1][c1] + QC_1*delta[a1][c0]) + PQ[a1]*(QC_0*delta[a0][c1] + QC_1*delta[a0][c0])))

                        +(PQ[c0]*PQ[c1]*(PQ[d0]*QD_1 + PQ[d1]*QD_0) + PQ[d0]*PQ[d1]*(PQ[c0]*QC_1 + PQ[c1]*QC_0))*(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) - PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) - PA_1*(PQ[a0]*delta[b0][b1] + PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1] - PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) - PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]))

                    )

                    + F8_t[4] * 0.5 * S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        -(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))*(QC_0*(PQ[c1]*delta[d0][d1] + PQ[d0]*delta[c1][d1] + PQ[d1]*delta[c1][d0]) + QC_1*(PQ[d0]*delta[c0][d1] + PQ[d1]*delta[c0][d0] + delta[d0][d1]*(PQ[c0] + QC_0)) + QD_0*(PQ[d1]*delta[c0][c1] + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + QD_1*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0)))

                        -PQ[a0]*(PA_1*(PQ[b0]*(QC_0*(PQ[c1]*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + PQ[d0]*(QC_1*delta[b1][d1] + QD_1*delta[b1][c1]) + PQ[d1]*(QC_1*delta[b1][d0] + QD_0*delta[b1][c1])) + QC_1*(PQ[c0]*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + delta[b1][c0]*(PQ[d0]*QD_1 + PQ[d1]*QD_0))) + PQ[b1]*(QC_0*(PQ[c1]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + PQ[d0]*(QC_1*delta[b0][d1] + QD_1*delta[b0][c1]) + PQ[d1]*(QC_1*delta[b0][d0] + QD_0*delta[b0][c1])) + QC_1*(PQ[c0]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + delta[b0][c0]*(PQ[d0]*QD_1 + PQ[d1]*QD_0)))) + PB_0*(PQ[a1]*(QC_0*(PQ[c1]*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + PQ[d0]*(QC_1*delta[b1][d1] + QD_1*delta[b1][c1]) + PQ[d1]*(QC_1*delta[b1][d0] + QD_0*delta[b1][c1])) + QC_1*(PQ[c0]*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + delta[b1][c0]*(PQ[d0]*QD_1 + PQ[d1]*QD_0))) + PQ[b1]*(QC_0*(PQ[c1]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]) + PQ[d0]*(QC_1*delta[a1][d1] + QD_1*delta[a1][c1]) + PQ[d1]*(QC_1*delta[a1][d0] + QD_0*delta[a1][c1])) + QC_1*(PQ[c0]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]) + delta[a1][c0]*(PQ[d0]*QD_1 + PQ[d1]*QD_0)))) + PB_1*(PQ[a1]*(QC_0*(PQ[c1]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + PQ[d0]*(QC_1*delta[b0][d1] + QD_1*delta[b0][c1]) + PQ[d1]*(QC_1*delta[b0][d0] + QD_0*delta[b0][c1])) + QC_1*(PQ[c0]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + delta[b0][c0]*(PQ[d0]*QD_1 + PQ[d1]*QD_0))) + PQ[b0]*(QC_0*(PQ[c1]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]) + PQ[d0]*(QC_1*delta[a1][d1] + QD_1*delta[a1][c1]) + PQ[d1]*(QC_1*delta[a1][d0] + QD_0*delta[a1][c1])) + QC_1*(PQ[c0]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]) + delta[a1][c0]*(PQ[d0]*QD_1 + PQ[d1]*QD_0))))) - PQ[a1]*(PA_0*(PQ[b0]*(QC_0*(PQ[c1]*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + PQ[d0]*(QC_1*delta[b1][d1] + QD_1*delta[b1][c1]) + PQ[d1]*(QC_1*delta[b1][d0] + QD_0*delta[b1][c1])) + QC_1*(PQ[c0]*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + delta[b1][c0]*(PQ[d0]*QD_1 + PQ[d1]*QD_0))) + PQ[b1]*(QC_0*(PQ[c1]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + PQ[d0]*(QC_1*delta[b0][d1] + QD_1*delta[b0][c1]) + PQ[d1]*(QC_1*delta[b0][d0] + QD_0*delta[b0][c1])) + QC_1*(PQ[c0]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + delta[b0][c0]*(PQ[d0]*QD_1 + PQ[d1]*QD_0)))) + (PB_0*PQ[b1] + PB_1*PQ[b0])*(QC_0*(PQ[c1]*(QD_0*delta[a0][d1] + QD_1*delta[a0][d0]) + PQ[d0]*(QC_1*delta[a0][d1] + QD_1*delta[a0][c1]) + PQ[d1]*(QC_1*delta[a0][d0] + QD_0*delta[a0][c1])) + QC_1*(PQ[c0]*(QD_0*delta[a0][d1] + QD_1*delta[a0][d0]) + delta[a0][c0]*(PQ[d0]*QD_1 + PQ[d1]*QD_0)))) - PQ[b0]*PQ[b1]*(PA_0*(QC_0*(PQ[c1]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]) + PQ[d0]*(QC_1*delta[a1][d1] + QD_1*delta[a1][c1]) + PQ[d1]*(QC_1*delta[a1][d0] + QD_0*delta[a1][c1])) + QC_1*(PQ[c0]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]) + delta[a1][c0]*(PQ[d0]*QD_1 + PQ[d1]*QD_0))) + PA_1*(QC_0*(PQ[c1]*(QD_0*delta[a0][d1] + QD_1*delta[a0][d0]) + PQ[d0]*(QC_1*delta[a0][d1] + QD_1*delta[a0][c1]) + PQ[d1]*(QC_1*delta[a0][d0] + QD_0*delta[a0][c1])) + QC_1*(PQ[c0]*(QD_0*delta[a0][d1] + QD_1*delta[a0][d0]) + delta[a0][c0]*(PQ[d0]*QD_1 + PQ[d1]*QD_0))) + QD_0*QD_1*(PA_0*(PQ[c0]*delta[a1][c1] + PQ[c1]*delta[a1][c0]) + PA_1*(PQ[c0]*delta[a0][c1] + PQ[c1]*delta[a0][c0]))) - QD_0*QD_1*(PQ[a0]*(PA_1*(PQ[b0]*(PQ[c0]*delta[b1][c1] + PQ[c1]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[b0][c1] + PQ[c1]*delta[b0][c0])) + PB_0*(PQ[a1]*(PQ[c0]*delta[b1][c1] + PQ[c1]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[a1][c1] + PQ[c1]*delta[a1][c0])) + PB_1*(PQ[a1]*(PQ[c0]*delta[b0][c1] + PQ[c1]*delta[b0][c0]) + PQ[b0]*(PQ[c0]*delta[a1][c1] + PQ[c1]*delta[a1][c0]))) + PQ[a1]*(PA_0*(PQ[b0]*(PQ[c0]*delta[b1][c1] + PQ[c1]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[b0][c1] + PQ[c1]*delta[b0][c0])) + (PB_0*PQ[b1] + PB_1*PQ[b0])*(PQ[c0]*delta[a0][c1] + PQ[c1]*delta[a0][c0])))

                        +(PQ[c0]*QD_0*(PQ[c1]*QD_1 + PQ[d1]*QC_1) + PQ[c1]*QC_0*(PQ[d0]*QD_1 + PQ[d1]*QD_0) + PQ[d0]*QC_1*(PQ[c0]*QD_1 + PQ[d1]*QC_0))*(-PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) - PA_1*(PQ[a0]*delta[b0][b1] + PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) - PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) - PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]) + PQ[a0]*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PQ[a1]*(PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PQ[b0]*PQ[b1]*delta[a0][a1])

                    )

                    + F8_t[4] * 0.5 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*(QC_0*(QC_1*delta[d0][d1] + QD_0*delta[c1][d1] + QD_1*delta[c1][d0]) + QC_1*(QD_0*delta[c0][d1] + QD_1*delta[c0][d0]) + QD_0*QD_1*delta[c0][c1])

                        +PQ[a0]*PQ[a1]*(QC_0*QC_1*(PQ[b0]*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + PQ[b1]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0])) + QD_0*QD_1*(PQ[b0]*(QC_0*delta[b1][c1] + QC_1*delta[b1][c0]) + PQ[b1]*(QC_0*delta[b0][c1] + QC_1*delta[b0][c0]))) + PQ[b0]*PQ[b1]*(QC_0*QC_1*(PQ[a0]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]) + PQ[a1]*(QD_0*delta[a0][d1] + QD_1*delta[a0][d0])) + QD_0*QD_1*(PQ[a0]*(QC_0*delta[a1][c1] + QC_1*delta[a1][c0]) + PQ[a1]*(QC_0*delta[a0][c1] + QC_1*delta[a0][c0])))

                        +(QC_0*QD_1*(PQ[c1]*QD_0 + PQ[d0]*QC_1) + QC_1*QD_0*(PQ[c0]*QD_1 + PQ[d1]*QC_0))*(PQ[a0]*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PQ[a1]*(PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PQ[b0]*PQ[b1]*delta[a0][a1])

                    )

                    + F8_t[4] * S1 * S1 * S1 * S1 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        +PA_0*PA_1*PB_0*PB_1*PQ[c0]*PQ[c1]*PQ[d0]*PQ[d1]

                    )

                    + F8_t[4] * S1 * S1 * S1 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        -(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))*(PQ[c0]*PQ[c1]*(PQ[d0]*QD_1 + PQ[d1]*QD_0) + PQ[d0]*PQ[d1]*(PQ[c0]*QC_1 + PQ[c1]*QC_0))

                    )

                    + F8_t[4] * S1 * S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        +(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))*(PQ[c0]*QD_0*(PQ[c1]*QD_1 + PQ[d1]*QC_1) + PQ[c1]*QC_0*(PQ[d0]*QD_1 + PQ[d1]*QD_0) + PQ[d0]*QC_1*(PQ[c0]*QD_1 + PQ[d1]*QC_0))

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * dd_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDD20_FP64(double*         mat_J,
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
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ double   delta[3][3];

    __shared__ double a_i, a_j, r_i[3], r_j[3], S_ij_00, S1, inv_S1;
    __shared__ double PA_0, PA_1, PB_0, PB_1;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta[0][0] = 1.0; delta[0][1] = 0.0; delta[0][2] = 0.0;
        delta[1][0] = 0.0; delta[1][1] = 1.0; delta[1][2] = 0.0;
        delta[2][0] = 0.0; delta[2][1] = 0.0; delta[2][2] = 1.0;

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
            inv_S1 = 1.0 / S1;

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

    for (uint32_t m = 0; m < prec_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= dd_prim_pair_count))
        {
            break;
        }

        const auto k = dd_first_inds[kl];
        const auto l = dd_second_inds[kl];

        const auto a_k = d_prim_info[k / 6 + d_prim_count * 0];

        const double r_k[3] = {d_prim_info[k / 6 + d_prim_count * 2],
                               d_prim_info[k / 6 + d_prim_count * 3],
                               d_prim_info[k / 6 + d_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = dd_pair_data[kl];

        const auto c0 = d_cart_inds[k % 6][0];
        const auto c1 = d_cart_inds[k % 6][1];
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F8_t[5];

        gpu::computeBoysFunction(F8_t, S1 * S2 * inv_S4 * r2_PQ, 4, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QC_1 = (a_l * inv_S2) * (r_l[c1] - r_k[c1]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                    + F8_t[4] * S1 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        -(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))*(QC_0*QD_1*(PQ[c1]*QD_0 + PQ[d0]*QC_1) + QC_1*QD_0*(PQ[c0]*QD_1 + PQ[d1]*QC_0))

                    )

                    + F8_t[4] * S2 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*QC_0*QC_1*QD_0*QD_1

                    )

                    + F8_t[4] * 0.0625 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        +delta[a0][a1]*(delta[b0][b1]*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0]) + delta[b0][c0]*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + delta[b0][c1]*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + delta[b0][d0]*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + delta[b0][d1]*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) + delta[a0][b0]*(delta[a1][b1]*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0]) + delta[a1][c0]*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + delta[a1][c1]*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + delta[a1][d0]*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + delta[a1][d1]*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) + delta[a0][b1]*(delta[a1][b0]*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0]) + delta[a1][c0]*(delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0]) + delta[a1][c1]*(delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0]) + delta[a1][d0]*(delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1]) + delta[a1][d1]*(delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1])) + delta[a0][c0]*(delta[a1][b0]*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + delta[a1][b1]*(delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0]) + delta[a1][c1]*(delta[b0][b1]*delta[d0][d1] + delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + delta[a1][d0]*(delta[b0][b1]*delta[c1][d1] + delta[b0][c1]*delta[b1][d1] + delta[b0][d1]*delta[b1][c1]) + delta[a1][d1]*(delta[b0][b1]*delta[c1][d0] + delta[b0][c1]*delta[b1][d0] + delta[b0][d0]*delta[b1][c1])) + delta[a0][c1]*(delta[a1][b0]*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + delta[a1][b1]*(delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0]) + delta[a1][c0]*(delta[b0][b1]*delta[d0][d1] + delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + delta[a1][d0]*(delta[b0][b1]*delta[c0][d1] + delta[b0][c0]*delta[b1][d1] + delta[b0][d1]*delta[b1][c0]) + delta[a1][d1]*(delta[b0][b1]*delta[c0][d0] + delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0])) + delta[a0][d0]*(delta[a1][b0]*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + delta[a1][b1]*(delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1]) + delta[a1][c0]*(delta[b0][b1]*delta[c1][d1] + delta[b0][c1]*delta[b1][d1] + delta[b0][d1]*delta[b1][c1]) + delta[a1][c1]*(delta[b0][b1]*delta[c0][d1] + delta[b0][c0]*delta[b1][d1] + delta[b0][d1]*delta[b1][c0]) + delta[a1][d1]*(delta[b0][b1]*delta[c0][c1] + delta[b0][c0]*delta[b1][c1] + delta[b0][c1]*delta[b1][c0])) + delta[a0][d1]*(delta[a1][b0]*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1]) + delta[a1][b1]*(delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1]) + delta[a1][c0]*(delta[b0][b1]*delta[c1][d0] + delta[b0][c1]*delta[b1][d0] + delta[b0][d0]*delta[b1][c1]) + delta[a1][c1]*(delta[b0][b1]*delta[c0][d0] + delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0]) + delta[a1][d0]*(delta[b0][b1]*delta[c0][c1] + delta[b0][c0]*delta[b1][c1] + delta[b0][c1]*delta[b1][c0]))

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * dd_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDD21_FP64(double*         mat_J,
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
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ double   delta[3][3];

    __shared__ double a_i, a_j, r_i[3], r_j[3], S_ij_00, S1, inv_S1;
    __shared__ double PA_0, PA_1, PB_0, PB_1;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta[0][0] = 1.0; delta[0][1] = 0.0; delta[0][2] = 0.0;
        delta[1][0] = 0.0; delta[1][1] = 1.0; delta[1][2] = 0.0;
        delta[2][0] = 0.0; delta[2][1] = 0.0; delta[2][2] = 1.0;

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
            inv_S1 = 1.0 / S1;

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

    for (uint32_t m = 0; m < prec_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= dd_prim_pair_count))
        {
            break;
        }

        const auto k = dd_first_inds[kl];
        const auto l = dd_second_inds[kl];

        const auto a_k = d_prim_info[k / 6 + d_prim_count * 0];

        const double r_k[3] = {d_prim_info[k / 6 + d_prim_count * 2],
                               d_prim_info[k / 6 + d_prim_count * 3],
                               d_prim_info[k / 6 + d_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = dd_pair_data[kl];

        const auto c0 = d_cart_inds[k % 6][0];
        const auto c1 = d_cart_inds[k % 6][1];
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F8_t[6];

        gpu::computeBoysFunction(F8_t, S1 * S2 * inv_S4 * r2_PQ, 5, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QC_1 = (a_l * inv_S2) * (r_l[c1] - r_k[c1]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                    + F8_t[5] * (-0.125) * S1 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        +PQ[a0]*(PQ[a1]*(delta[b0][b1]*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0]) + delta[b0][c0]*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + delta[b0][c1]*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + delta[b0][d0]*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + delta[b0][d1]*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) + PQ[b0]*(delta[a1][b1]*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0]) + delta[a1][c0]*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + delta[a1][c1]*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + delta[a1][d0]*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + delta[a1][d1]*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) + PQ[b1]*(delta[a1][b0]*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0]) + delta[a1][c0]*(delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0]) + delta[a1][c1]*(delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0]) + delta[a1][d0]*(delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1]) + delta[a1][d1]*(delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1])) + PQ[c0]*(delta[a1][b0]*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + delta[a1][b1]*(delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0]) + delta[a1][c1]*(delta[b0][b1]*delta[d0][d1] + delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + delta[a1][d0]*(delta[b0][b1]*delta[c1][d1] + delta[b0][c1]*delta[b1][d1] + delta[b0][d1]*delta[b1][c1]) + delta[a1][d1]*(delta[b0][b1]*delta[c1][d0] + delta[b0][c1]*delta[b1][d0] + delta[b0][d0]*delta[b1][c1])) + PQ[c1]*(delta[a1][b0]*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + delta[a1][b1]*(delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0]) + delta[a1][c0]*(delta[b0][b1]*delta[d0][d1] + delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + delta[a1][d0]*(delta[b0][b1]*delta[c0][d1] + delta[b0][c0]*delta[b1][d1] + delta[b0][d1]*delta[b1][c0]) + delta[a1][d1]*(delta[b0][b1]*delta[c0][d0] + delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0])) + PQ[d0]*(delta[a1][b0]*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + delta[a1][b1]*(delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1]) + delta[a1][c0]*(delta[b0][b1]*delta[c1][d1] + delta[b0][c1]*delta[b1][d1] + delta[b0][d1]*delta[b1][c1]) + delta[a1][c1]*(delta[b0][b1]*delta[c0][d1] + delta[b0][c0]*delta[b1][d1] + delta[b0][d1]*delta[b1][c0]) + delta[a1][d1]*(delta[b0][b1]*delta[c0][c1] + delta[b0][c0]*delta[b1][c1] + delta[b0][c1]*delta[b1][c0])) + PQ[d1]*(delta[a1][b0]*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1]) + delta[a1][b1]*(delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1]) + delta[a1][c0]*(delta[b0][b1]*delta[c1][d0] + delta[b0][c1]*delta[b1][d0] + delta[b0][d0]*delta[b1][c1]) + delta[a1][c1]*(delta[b0][b1]*delta[c0][d0] + delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0]) + delta[a1][d0]*(delta[b0][b1]*delta[c0][c1] + delta[b0][c0]*delta[b1][c1] + delta[b0][c1]*delta[b1][c0]))) + PQ[a1]*(PQ[b0]*(delta[a0][b1]*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0]) + delta[a0][c0]*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + delta[a0][c1]*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + delta[a0][d0]*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + delta[a0][d1]*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) + PQ[b1]*(delta[a0][b0]*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0]) + delta[a0][c0]*(delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0]) + delta[a0][c1]*(delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0]) + delta[a0][d0]*(delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1]) + delta[a0][d1]*(delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1])) + PQ[c0]*(delta[a0][b0]*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + delta[a0][b1]*(delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0]) + delta[a0][c1]*(delta[b0][b1]*delta[d0][d1] + delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + delta[a0][d0]*(delta[b0][b1]*delta[c1][d1] + delta[b0][c1]*delta[b1][d1] + delta[b0][d1]*delta[b1][c1]) + delta[a0][d1]*(delta[b0][b1]*delta[c1][d0] + delta[b0][c1]*delta[b1][d0] + delta[b0][d0]*delta[b1][c1])) + PQ[c1]*(delta[a0][b0]*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + delta[a0][b1]*(delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0]) + delta[a0][c0]*(delta[b0][b1]*delta[d0][d1] + delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + delta[a0][d0]*(delta[b0][b1]*delta[c0][d1] + delta[b0][c0]*delta[b1][d1] + delta[b0][d1]*delta[b1][c0]) + delta[a0][d1]*(delta[b0][b1]*delta[c0][d0] + delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0])) + PQ[d0]*(delta[a0][b0]*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + delta[a0][b1]*(delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1]) + delta[a0][c0]*(delta[b0][b1]*delta[c1][d1] + delta[b0][c1]*delta[b1][d1] + delta[b0][d1]*delta[b1][c1]) + delta[a0][c1]*(delta[b0][b1]*delta[c0][d1] + delta[b0][c0]*delta[b1][d1] + delta[b0][d1]*delta[b1][c0]) + delta[a0][d1]*(delta[b0][b1]*delta[c0][c1] + delta[b0][c0]*delta[b1][c1] + delta[b0][c1]*delta[b1][c0])) + PQ[d1]*(delta[a0][b0]*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1]) + delta[a0][b1]*(delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1]) + delta[a0][c0]*(delta[b0][b1]*delta[c1][d0] + delta[b0][c1]*delta[b1][d0] + delta[b0][d0]*delta[b1][c1]) + delta[a0][c1]*(delta[b0][b1]*delta[c0][d0] + delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0]) + delta[a0][d0]*(delta[b0][b1]*delta[c0][c1] + delta[b0][c0]*delta[b1][c1] + delta[b0][c1]*delta[b1][c0]))) + PQ[b0]*(PQ[b1]*(delta[a0][a1]*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0]) + delta[a0][c0]*(delta[a1][c1]*delta[d0][d1] + delta[a1][d0]*delta[c1][d1] + delta[a1][d1]*delta[c1][d0]) + delta[a0][c1]*(delta[a1][c0]*delta[d0][d1] + delta[a1][d0]*delta[c0][d1] + delta[a1][d1]*delta[c0][d0]) + delta[a0][d0]*(delta[a1][c0]*delta[c1][d1] + delta[a1][c1]*delta[c0][d1] + delta[a1][d1]*delta[c0][c1]) + delta[a0][d1]*(delta[a1][c0]*delta[c1][d0] + delta[a1][c1]*delta[c0][d0] + delta[a1][d0]*delta[c0][c1])) + PQ[c0]*(delta[a0][a1]*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + delta[a0][b1]*(delta[a1][c1]*delta[d0][d1] + delta[a1][d0]*delta[c1][d1] + delta[a1][d1]*delta[c1][d0]) + delta[a0][c1]*(delta[a1][b1]*delta[d0][d1] + delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + delta[a0][d0]*(delta[a1][b1]*delta[c1][d1] + delta[a1][c1]*delta[b1][d1] + delta[a1][d1]*delta[b1][c1]) + delta[a0][d1]*(delta[a1][b1]*delta[c1][d0] + delta[a1][c1]*delta[b1][d0] + delta[a1][d0]*delta[b1][c1])) + PQ[c1]*(delta[a0][a1]*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + delta[a0][b1]*(delta[a1][c0]*delta[d0][d1] + delta[a1][d0]*delta[c0][d1] + delta[a1][d1]*delta[c0][d0]) + delta[a0][c0]*(delta[a1][b1]*delta[d0][d1] + delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + delta[a0][d0]*(delta[a1][b1]*delta[c0][d1] + delta[a1][c0]*delta[b1][d1] + delta[a1][d1]*delta[b1][c0]) + delta[a0][d1]*(delta[a1][b1]*delta[c0][d0] + delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0])) + PQ[d0]*(delta[a0][a1]*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + delta[a0][b1]*(delta[a1][c0]*delta[c1][d1] + delta[a1][c1]*delta[c0][d1] + delta[a1][d1]*delta[c0][c1]) + delta[a0][c0]*(delta[a1][b1]*delta[c1][d1] + delta[a1][c1]*delta[b1][d1] + delta[a1][d1]*delta[b1][c1]) + delta[a0][c1]*(delta[a1][b1]*delta[c0][d1] + delta[a1][c0]*delta[b1][d1] + delta[a1][d1]*delta[b1][c0]) + delta[a0][d1]*(delta[a1][b1]*delta[c0][c1] + delta[a1][c0]*delta[b1][c1] + delta[a1][c1]*delta[b1][c0])) + PQ[d1]*(delta[a0][a1]*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1]) + delta[a0][b1]*(delta[a1][c0]*delta[c1][d0] + delta[a1][c1]*delta[c0][d0] + delta[a1][d0]*delta[c0][c1]) + delta[a0][c0]*(delta[a1][b1]*delta[c1][d0] + delta[a1][c1]*delta[b1][d0] + delta[a1][d0]*delta[b1][c1]) + delta[a0][c1]*(delta[a1][b1]*delta[c0][d0] + delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + delta[a0][d0]*(delta[a1][b1]*delta[c0][c1] + delta[a1][c0]*delta[b1][c1] + delta[a1][c1]*delta[b1][c0]))) + PQ[b1]*(PQ[c0]*(delta[a0][a1]*(delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0]) + delta[a0][b0]*(delta[a1][c1]*delta[d0][d1] + delta[a1][d0]*delta[c1][d1] + delta[a1][d1]*delta[c1][d0]) + delta[a0][c1]*(delta[a1][b0]*delta[d0][d1] + delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + delta[a0][d0]*(delta[a1][b0]*delta[c1][d1] + delta[a1][c1]*delta[b0][d1] + delta[a1][d1]*delta[b0][c1]) + delta[a0][d1]*(delta[a1][b0]*delta[c1][d0] + delta[a1][c1]*delta[b0][d0] + delta[a1][d0]*delta[b0][c1])) + PQ[c1]*(delta[a0][a1]*(delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0]) + delta[a0][b0]*(delta[a1][c0]*delta[d0][d1] + delta[a1][d0]*delta[c0][d1] + delta[a1][d1]*delta[c0][d0]) + delta[a0][c0]*(delta[a1][b0]*delta[d0][d1] + delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + delta[a0][d0]*(delta[a1][b0]*delta[c0][d1] + delta[a1][c0]*delta[b0][d1] + delta[a1][d1]*delta[b0][c0]) + delta[a0][d1]*(delta[a1][b0]*delta[c0][d0] + delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0])) + PQ[d0]*(delta[a0][a1]*(delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1]) + delta[a0][b0]*(delta[a1][c0]*delta[c1][d1] + delta[a1][c1]*delta[c0][d1] + delta[a1][d1]*delta[c0][c1]) + delta[a0][c0]*(delta[a1][b0]*delta[c1][d1] + delta[a1][c1]*delta[b0][d1] + delta[a1][d1]*delta[b0][c1]) + delta[a0][c1]*(delta[a1][b0]*delta[c0][d1] + delta[a1][c0]*delta[b0][d1] + delta[a1][d1]*delta[b0][c0]) + delta[a0][d1]*(delta[a1][b0]*delta[c0][c1] + delta[a1][c0]*delta[b0][c1] + delta[a1][c1]*delta[b0][c0])) + PQ[d1]*(delta[a0][a1]*(delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1]) + delta[a0][b0]*(delta[a1][c0]*delta[c1][d0] + delta[a1][c1]*delta[c0][d0] + delta[a1][d0]*delta[c0][c1]) + delta[a0][c0]*(delta[a1][b0]*delta[c1][d0] + delta[a1][c1]*delta[b0][d0] + delta[a1][d0]*delta[b0][c1]) + delta[a0][c1]*(delta[a1][b0]*delta[c0][d0] + delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0]) + delta[a0][d0]*(delta[a1][b0]*delta[c0][c1] + delta[a1][c0]*delta[b0][c1] + delta[a1][c1]*delta[b0][c0]))) + PQ[c0]*(PQ[c1]*(delta[a0][a1]*(delta[b0][b1]*delta[d0][d1] + delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + delta[a0][b0]*(delta[a1][b1]*delta[d0][d1] + delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + delta[a0][b1]*(delta[a1][b0]*delta[d0][d1] + delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + delta[a0][d0]*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + delta[a0][d1]*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1])) + PQ[d0]*(delta[a0][a1]*(delta[b0][b1]*delta[c1][d1] + delta[b0][c1]*delta[b1][d1] + delta[b0][d1]*delta[b1][c1]) + delta[a0][b0]*(delta[a1][b1]*delta[c1][d1] + delta[a1][c1]*delta[b1][d1] + delta[a1][d1]*delta[b1][c1]) + delta[a0][b1]*(delta[a1][b0]*delta[c1][d1] + delta[a1][c1]*delta[b0][d1] + delta[a1][d1]*delta[b0][c1]) + delta[a0][c1]*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + delta[a0][d1]*(delta[a1][b0]*delta[b1][c1] + delta[a1][b1]*delta[b0][c1] + delta[a1][c1]*delta[b0][b1])) + PQ[d1]*(delta[a0][a1]*(delta[b0][b1]*delta[c1][d0] + delta[b0][c1]*delta[b1][d0] + delta[b0][d0]*delta[b1][c1]) + delta[a0][b0]*(delta[a1][b1]*delta[c1][d0] + delta[a1][c1]*delta[b1][d0] + delta[a1][d0]*delta[b1][c1]) + delta[a0][b1]*(delta[a1][b0]*delta[c1][d0] + delta[a1][c1]*delta[b0][d0] + delta[a1][d0]*delta[b0][c1]) + delta[a0][c1]*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + delta[a0][d0]*(delta[a1][b0]*delta[b1][c1] + delta[a1][b1]*delta[b0][c1] + delta[a1][c1]*delta[b0][b1]))) + PQ[c1]*(PQ[d0]*(delta[a0][a1]*(delta[b0][b1]*delta[c0][d1] + delta[b0][c0]*delta[b1][d1] + delta[b0][d1]*delta[b1][c0]) + delta[a0][b0]*(delta[a1][b1]*delta[c0][d1] + delta[a1][c0]*delta[b1][d1] + delta[a1][d1]*delta[b1][c0]) + delta[a0][b1]*(delta[a1][b0]*delta[c0][d1] + delta[a1][c0]*delta[b0][d1] + delta[a1][d1]*delta[b0][c0]) + delta[a0][c0]*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + delta[a0][d1]*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1])) + PQ[d1]*(delta[a0][a1]*(delta[b0][b1]*delta[c0][d0] + delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0]) + delta[a0][b0]*(delta[a1][b1]*delta[c0][d0] + delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + delta[a0][b1]*(delta[a1][b0]*delta[c0][d0] + delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0]) + delta[a0][c0]*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + delta[a0][d0]*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1]))) + PQ[d0]*PQ[d1]*(delta[a0][a1]*(delta[b0][b1]*delta[c0][c1] + delta[b0][c0]*delta[b1][c1] + delta[b0][c1]*delta[b1][c0]) + delta[a0][b0]*(delta[a1][b1]*delta[c0][c1] + delta[a1][c0]*delta[b1][c1] + delta[a1][c1]*delta[b1][c0]) + delta[a0][b1]*(delta[a1][b0]*delta[c0][c1] + delta[a1][c0]*delta[b0][c1] + delta[a1][c1]*delta[b0][c0]) + delta[a0][c0]*(delta[a1][b0]*delta[b1][c1] + delta[a1][b1]*delta[b0][c1] + delta[a1][c1]*delta[b0][b1]) + delta[a0][c1]*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1]))

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * dd_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDD22_FP64(double*         mat_J,
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
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ double   delta[3][3];

    __shared__ double a_i, a_j, r_i[3], r_j[3], S_ij_00, S1, inv_S1;
    __shared__ double PA_0, PA_1, PB_0, PB_1;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta[0][0] = 1.0; delta[0][1] = 0.0; delta[0][2] = 0.0;
        delta[1][0] = 0.0; delta[1][1] = 1.0; delta[1][2] = 0.0;
        delta[2][0] = 0.0; delta[2][1] = 0.0; delta[2][2] = 1.0;

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
            inv_S1 = 1.0 / S1;

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

    for (uint32_t m = 0; m < prec_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= dd_prim_pair_count))
        {
            break;
        }

        const auto k = dd_first_inds[kl];
        const auto l = dd_second_inds[kl];

        const auto a_k = d_prim_info[k / 6 + d_prim_count * 0];

        const double r_k[3] = {d_prim_info[k / 6 + d_prim_count * 2],
                               d_prim_info[k / 6 + d_prim_count * 3],
                               d_prim_info[k / 6 + d_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = dd_pair_data[kl];

        const auto c0 = d_cart_inds[k % 6][0];
        const auto c1 = d_cart_inds[k % 6][1];
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F8_t[6];

        gpu::computeBoysFunction(F8_t, S1 * S2 * inv_S4 * r2_PQ, 5, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QC_1 = (a_l * inv_S2) * (r_l[c1] - r_k[c1]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                    + F8_t[5] * 0.25 * S1 * S1 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        +(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])

                        +(delta[a0][c0]*delta[c1][d0] + delta[a0][c1]*delta[c0][d0] + delta[a0][d0]*delta[c0][c1])*(PA_1*PQ[b0]*PQ[b1]*PQ[d1] + PB_0*PQ[a1]*PQ[b1]*PQ[d1] + PB_1*PQ[a1]*PQ[b0]*PQ[d1]) + (delta[a0][c0]*delta[c1][d1] + delta[a0][c1]*delta[c0][d1] + delta[a0][d1]*delta[c0][c1])*(PA_1*PQ[b0]*PQ[b1]*PQ[d0] + PB_0*PQ[a1]*PQ[b1]*PQ[d0] + PB_1*PQ[a1]*PQ[b0]*PQ[d0]) + (delta[a0][c0]*delta[d0][d1] + delta[a0][d0]*delta[c0][d1] + delta[a0][d1]*delta[c0][d0])*(PA_1*PQ[b0]*PQ[b1]*PQ[c1] + PB_0*PQ[a1]*PQ[b1]*PQ[c1] + PB_1*PQ[a1]*PQ[b0]*PQ[c1]) + (delta[a0][c1]*delta[d0][d1] + delta[a0][d0]*delta[c1][d1] + delta[a0][d1]*delta[c1][d0])*(PA_1*PQ[b0]*PQ[b1]*PQ[c0] + PB_0*PQ[a1]*PQ[b1]*PQ[c0] + PB_1*PQ[a1]*PQ[b0]*PQ[c0]) + (delta[a1][c0]*delta[c1][d0] + delta[a1][c1]*delta[c0][d0] + delta[a1][d0]*delta[c0][c1])*(PA_0*PQ[b0]*PQ[b1]*PQ[d1] + PB_0*PQ[a0]*PQ[b1]*PQ[d1] + PB_1*PQ[a0]*PQ[b0]*PQ[d1]) + (delta[a1][c0]*delta[c1][d1] + delta[a1][c1]*delta[c0][d1] + delta[a1][d1]*delta[c0][c1])*(PA_0*PQ[b0]*PQ[b1]*PQ[d0] + PB_0*PQ[a0]*PQ[b1]*PQ[d0] + PB_1*PQ[a0]*PQ[b0]*PQ[d0]) + (delta[a1][c0]*delta[d0][d1] + delta[a1][d0]*delta[c0][d1] + delta[a1][d1]*delta[c0][d0])*(PA_0*PQ[b0]*PQ[b1]*PQ[c1] + PB_0*PQ[a0]*PQ[b1]*PQ[c1] + PB_1*PQ[a0]*PQ[b0]*PQ[c1]) + (delta[a1][c1]*delta[d0][d1] + delta[a1][d0]*delta[c1][d1] + delta[a1][d1]*delta[c1][d0])*(PA_0*PQ[b0]*PQ[b1]*PQ[c0] + PB_0*PQ[a0]*PQ[b1]*PQ[c0] + PB_1*PQ[a0]*PQ[b0]*PQ[c0]) + (delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1])*(PA_0*PQ[a1]*PQ[b1]*PQ[d1] + PA_1*PQ[a0]*PQ[b1]*PQ[d1] + PB_1*PQ[a0]*PQ[a1]*PQ[d1]) + (delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1])*(PA_0*PQ[a1]*PQ[b1]*PQ[d0] + PA_1*PQ[a0]*PQ[b1]*PQ[d0] + PB_1*PQ[a0]*PQ[a1]*PQ[d0]) + (delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0])*(PA_0*PQ[a1]*PQ[b1]*PQ[c1] + PA_1*PQ[a0]*PQ[b1]*PQ[c1] + PB_1*PQ[a0]*PQ[a1]*PQ[c1]) + (delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0])*(PA_0*PQ[a1]*PQ[b1]*PQ[c0] + PA_1*PQ[a0]*PQ[b1]*PQ[c0] + PB_1*PQ[a0]*PQ[a1]*PQ[c0]) + (delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])*(PA_0*PQ[a1]*PQ[b0]*PQ[d1] + PA_1*PQ[a0]*PQ[b0]*PQ[d1] + PB_0*PQ[a0]*PQ[a1]*PQ[d1]) + (delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1])*(PA_0*PQ[a1]*PQ[b0]*PQ[d0] + PA_1*PQ[a0]*PQ[b0]*PQ[d0] + PB_0*PQ[a0]*PQ[a1]*PQ[d0]) + (delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0])*(PA_0*PQ[a1]*PQ[b0]*PQ[c1] + PA_1*PQ[a0]*PQ[b0]*PQ[c1] + PB_0*PQ[a0]*PQ[a1]*PQ[c1]) + (delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0])*(PA_0*PQ[a1]*PQ[b0]*PQ[c0] + PA_1*PQ[a0]*PQ[b0]*PQ[c0] + PB_0*PQ[a0]*PQ[a1]*PQ[c0])

                        +(PQ[c0]*(PQ[c1]*delta[d0][d1] + PQ[d0]*delta[c1][d1] + PQ[d1]*delta[c1][d0]) + PQ[c1]*(PQ[d0]*delta[c0][d1] + PQ[d1]*delta[c0][d0]) + PQ[d0]*PQ[d1]*delta[c0][c1])*(PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PQ[a0]*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0] - PQ[a1]*delta[b0][b1] - PQ[b0]*delta[a1][b1] - PQ[b1]*delta[a1][b0]) + PQ[a1]*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PQ[b0]*(PB_1*delta[a0][a1] + delta[a0][b1]*(PA_1 - PQ[a1])) + PQ[b1]*(delta[a0][a1]*(PB_0 - PQ[b0]) + delta[a0][b0]*(PA_1 - PQ[a1])))

                        +PQ[c0]*(PA_0*(PQ[a1]*(PQ[c1]*(delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + PQ[d0]*(delta[b0][c1]*delta[b1][d1] + delta[b0][d1]*delta[b1][c1]) + PQ[d1]*(delta[b0][c1]*delta[b1][d0] + delta[b0][d0]*delta[b1][c1])) + PQ[b0]*(PQ[c1]*(delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + PQ[d0]*(delta[a1][c1]*delta[b1][d1] + delta[a1][d1]*delta[b1][c1]) + PQ[d1]*(delta[a1][c1]*delta[b1][d0] + delta[a1][d0]*delta[b1][c1])) + PQ[b1]*(PQ[c1]*(delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + PQ[d0]*(delta[a1][c1]*delta[b0][d1] + delta[a1][d1]*delta[b0][c1]) + PQ[d1]*(delta[a1][c1]*delta[b0][d0] + delta[a1][d0]*delta[b0][c1]))) + PA_1*(PQ[a0]*(PQ[c1]*(delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + PQ[d0]*(delta[b0][c1]*delta[b1][d1] + delta[b0][d1]*delta[b1][c1]) + PQ[d1]*(delta[b0][c1]*delta[b1][d0] + delta[b0][d0]*delta[b1][c1])) + PQ[b0]*(PQ[c1]*(delta[a0][d0]*delta[b1][d1] + delta[a0][d1]*delta[b1][d0]) + PQ[d0]*(delta[a0][c1]*delta[b1][d1] + delta[a0][d1]*delta[b1][c1]) + PQ[d1]*(delta[a0][c1]*delta[b1][d0] + delta[a0][d0]*delta[b1][c1])) + PQ[b1]*(PQ[c1]*(delta[a0][d0]*delta[b0][d1] + delta[a0][d1]*delta[b0][d0]) + PQ[d0]*(delta[a0][c1]*delta[b0][d1] + delta[a0][d1]*delta[b0][c1]) + PQ[d1]*(delta[a0][c1]*delta[b0][d0] + delta[a0][d0]*delta[b0][c1]))) + PB_0*(PQ[a0]*(PQ[c1]*(delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + PQ[d0]*(delta[a1][c1]*delta[b1][d1] + delta[a1][d1]*delta[b1][c1]) + PQ[d1]*(delta[a1][c1]*delta[b1][d0] + delta[a1][d0]*delta[b1][c1])) + PQ[a1]*(PQ[c1]*(delta[a0][d0]*delta[b1][d1] + delta[a0][d1]*delta[b1][d0]) + PQ[d0]*(delta[a0][c1]*delta[b1][d1] + delta[a0][d1]*delta[b1][c1]) + PQ[d1]*(delta[a0][c1]*delta[b1][d0] + delta[a0][d0]*delta[b1][c1])) + PQ[b1]*(PQ[c1]*(delta[a0][d0]*delta[a1][d1] + delta[a0][d1]*delta[a1][d0]) + PQ[d0]*(delta[a0][c1]*delta[a1][d1] + delta[a0][d1]*delta[a1][c1]) + PQ[d1]*(delta[a0][c1]*delta[a1][d0] + delta[a0][d0]*delta[a1][c1]))) + PB_1*(PQ[a0]*(PQ[c1]*(delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + PQ[d0]*(delta[a1][c1]*delta[b0][d1] + delta[a1][d1]*delta[b0][c1]) + PQ[d1]*(delta[a1][c1]*delta[b0][d0] + delta[a1][d0]*delta[b0][c1])) + PQ[a1]*(PQ[c1]*(delta[a0][d0]*delta[b0][d1] + delta[a0][d1]*delta[b0][d0]) + PQ[d0]*(delta[a0][c1]*delta[b0][d1] + delta[a0][d1]*delta[b0][c1]) + PQ[d1]*(delta[a0][c1]*delta[b0][d0] + delta[a0][d0]*delta[b0][c1])) + PQ[b0]*(PQ[c1]*(delta[a0][d0]*delta[a1][d1] + delta[a0][d1]*delta[a1][d0]) + PQ[d0]*(delta[a0][c1]*delta[a1][d1] + delta[a0][d1]*delta[a1][c1]) + PQ[d1]*(delta[a0][c1]*delta[a1][d0] + delta[a0][d0]*delta[a1][c1])))) + PQ[c1]*(PQ[d0]*(PA_0*(PQ[a1]*(delta[b0][c0]*delta[b1][d1] + delta[b0][d1]*delta[b1][c0]) + PQ[b0]*(delta[a1][c0]*delta[b1][d1] + delta[a1][d1]*delta[b1][c0]) + PQ[b1]*(delta[a1][c0]*delta[b0][d1] + delta[a1][d1]*delta[b0][c0])) + PA_1*(PQ[a0]*(delta[b0][c0]*delta[b1][d1] + delta[b0][d1]*delta[b1][c0]) + PQ[b0]*(delta[a0][c0]*delta[b1][d1] + delta[a0][d1]*delta[b1][c0]) + PQ[b1]*(delta[a0][c0]*delta[b0][d1] + delta[a0][d1]*delta[b0][c0])) + PB_0*(PQ[a0]*(delta[a1][c0]*delta[b1][d1] + delta[a1][d1]*delta[b1][c0]) + PQ[a1]*(delta[a0][c0]*delta[b1][d1] + delta[a0][d1]*delta[b1][c0]) + PQ[b1]*(delta[a0][c0]*delta[a1][d1] + delta[a0][d1]*delta[a1][c0])) + PB_1*(PQ[a0]*(delta[a1][c0]*delta[b0][d1] + delta[a1][d1]*delta[b0][c0]) + PQ[a1]*(delta[a0][c0]*delta[b0][d1] + delta[a0][d1]*delta[b0][c0]) + PQ[b0]*(delta[a0][c0]*delta[a1][d1] + delta[a0][d1]*delta[a1][c0]))) + PQ[d1]*(PA_0*(PQ[a1]*(delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0]) + PQ[b0]*(delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + PQ[b1]*(delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0])) + PA_1*(PQ[a0]*(delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0]) + PQ[b0]*(delta[a0][c0]*delta[b1][d0] + delta[a0][d0]*delta[b1][c0]) + PQ[b1]*(delta[a0][c0]*delta[b0][d0] + delta[a0][d0]*delta[b0][c0])) + PB_0*(PQ[a0]*(delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + PQ[a1]*(delta[a0][c0]*delta[b1][d0] + delta[a0][d0]*delta[b1][c0]) + PQ[b1]*(delta[a0][c0]*delta[a1][d0] + delta[a0][d0]*delta[a1][c0])) + PB_1*(PQ[a0]*(delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0]) + PQ[a1]*(delta[a0][c0]*delta[b0][d0] + delta[a0][d0]*delta[b0][c0]) + PQ[b0]*(delta[a0][c0]*delta[a1][d0] + delta[a0][d0]*delta[a1][c0])))) + PQ[d0]*PQ[d1]*(PA_0*(PQ[a1]*(delta[b0][c0]*delta[b1][c1] + delta[b0][c1]*delta[b1][c0]) + PQ[b0]*(delta[a1][c0]*delta[b1][c1] + delta[a1][c1]*delta[b1][c0]) + PQ[b1]*(delta[a1][c0]*delta[b0][c1] + delta[a1][c1]*delta[b0][c0])) + PA_1*(PQ[a0]*(delta[b0][c0]*delta[b1][c1] + delta[b0][c1]*delta[b1][c0]) + PQ[b0]*(delta[a0][c0]*delta[b1][c1] + delta[a0][c1]*delta[b1][c0]) + PQ[b1]*(delta[a0][c0]*delta[b0][c1] + delta[a0][c1]*delta[b0][c0])) + PB_0*(PQ[a0]*(delta[a1][c0]*delta[b1][c1] + delta[a1][c1]*delta[b1][c0]) + PQ[a1]*(delta[a0][c0]*delta[b1][c1] + delta[a0][c1]*delta[b1][c0]) + PQ[b1]*(delta[a0][c0]*delta[a1][c1] + delta[a0][c1]*delta[a1][c0])) + PB_1*(PQ[a0]*(delta[a1][c0]*delta[b0][c1] + delta[a1][c1]*delta[b0][c0]) + PQ[a1]*(delta[a0][c0]*delta[b0][c1] + delta[a0][c1]*delta[b0][c0]) + PQ[b0]*(delta[a0][c0]*delta[a1][c1] + delta[a0][c1]*delta[a1][c0])))

                        +(PA_0*PQ[c0]*PQ[c1]*PQ[d0] - PQ[a0]*PQ[c0]*PQ[c1]*PQ[d0])*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + (PA_0*PQ[c0]*PQ[c1]*PQ[d1] - PQ[a0]*PQ[c0]*PQ[c1]*PQ[d1])*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + (PA_0*PQ[c0]*PQ[d0]*PQ[d1] - PQ[a0]*PQ[c0]*PQ[d0]*PQ[d1])*(delta[a1][b0]*delta[b1][c1] + delta[a1][b1]*delta[b0][c1] + delta[a1][c1]*delta[b0][b1]) + (PA_0*PQ[c1]*PQ[d0]*PQ[d1] - PQ[a0]*PQ[c1]*PQ[d0]*PQ[d1])*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1]) + (PA_1*PQ[c0]*PQ[c1]*PQ[d0] - PQ[a1]*PQ[c0]*PQ[c1]*PQ[d0])*(delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1]) + (PA_1*PQ[c0]*PQ[c1]*PQ[d1] - PQ[a1]*PQ[c0]*PQ[c1]*PQ[d1])*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + (PA_1*PQ[c0]*PQ[d0]*PQ[d1] - PQ[a1]*PQ[c0]*PQ[d0]*PQ[d1])*(delta[a0][b0]*delta[b1][c1] + delta[a0][b1]*delta[b0][c1] + delta[a0][c1]*delta[b0][b1]) + (PA_1*PQ[c1]*PQ[d0]*PQ[d1] - PQ[a1]*PQ[c1]*PQ[d0]*PQ[d1])*(delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1]) + (PB_0*PQ[c0]*PQ[c1]*PQ[d0] - PQ[b0]*PQ[c0]*PQ[c1]*PQ[d0])*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + (PB_0*PQ[c0]*PQ[c1]*PQ[d1] - PQ[b0]*PQ[c0]*PQ[c1]*PQ[d1])*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + (PB_0*PQ[c0]*PQ[d0]*PQ[d1] - PQ[b0]*PQ[c0]*PQ[d0]*PQ[d1])*(delta[a0][a1]*delta[b1][c1] + delta[a0][b1]*delta[a1][c1] + delta[a0][c1]*delta[a1][b1]) + (PB_0*PQ[c1]*PQ[d0]*PQ[d1] - PQ[b0]*PQ[c1]*PQ[d0]*PQ[d1])*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]) + (PB_1*PQ[c0]*PQ[c1]*PQ[d0] - PQ[b1]*PQ[c0]*PQ[c1]*PQ[d0])*(delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0]) + (PB_1*PQ[c0]*PQ[c1]*PQ[d1] - PQ[b1]*PQ[c0]*PQ[c1]*PQ[d1])*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]) + (PB_1*PQ[c0]*PQ[d0]*PQ[d1] - PQ[b1]*PQ[c0]*PQ[d0]*PQ[d1])*(delta[a0][a1]*delta[b0][c1] + delta[a0][b0]*delta[a1][c1] + delta[a0][c1]*delta[a1][b0]) + (PB_1*PQ[c1]*PQ[d0]*PQ[d1] - PQ[b1]*PQ[c1]*PQ[d0]*PQ[d1])*(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0])

                        -2*PQ[c0]*PQ[c1]*PQ[d0]*PQ[d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * dd_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDD23_FP64(double*         mat_J,
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
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ double   delta[3][3];

    __shared__ double a_i, a_j, r_i[3], r_j[3], S_ij_00, S1, inv_S1;
    __shared__ double PA_0, PA_1, PB_0, PB_1;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta[0][0] = 1.0; delta[0][1] = 0.0; delta[0][2] = 0.0;
        delta[1][0] = 0.0; delta[1][1] = 1.0; delta[1][2] = 0.0;
        delta[2][0] = 0.0; delta[2][1] = 0.0; delta[2][2] = 1.0;

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
            inv_S1 = 1.0 / S1;

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

    for (uint32_t m = 0; m < prec_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= dd_prim_pair_count))
        {
            break;
        }

        const auto k = dd_first_inds[kl];
        const auto l = dd_second_inds[kl];

        const auto a_k = d_prim_info[k / 6 + d_prim_count * 0];

        const double r_k[3] = {d_prim_info[k / 6 + d_prim_count * 2],
                               d_prim_info[k / 6 + d_prim_count * 3],
                               d_prim_info[k / 6 + d_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = dd_pair_data[kl];

        const auto c0 = d_cart_inds[k % 6][0];
        const auto c1 = d_cart_inds[k % 6][1];
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F8_t[6];

        gpu::computeBoysFunction(F8_t, S1 * S2 * inv_S4 * r2_PQ, 5, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QC_1 = (a_l * inv_S2) * (r_l[c1] - r_k[c1]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                    + F8_t[5] * 0.25 * S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        -2*PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0])

                        -PQ[a0]*PQ[a1]*(PQ[b0]*(delta[b1][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[b1][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[b1][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[b1][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))) + PQ[b1]*(delta[b0][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[b0][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[b0][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[b0][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0)))) - PQ[b0]*PQ[b1]*(PQ[a0]*(delta[a1][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[a1][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[a1][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[a1][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))) + PQ[a1]*(delta[a0][c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + delta[a0][c1]*(delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c0] + QC_0)) + delta[a0][d0]*(delta[c0][c1]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + delta[a0][d1]*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0))))

                        -(PQ[a0]*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PQ[a1]*(PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PQ[b0]*PQ[b1]*delta[a0][a1])*(PQ[c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + PQ[c1]*(QC_0*delta[d0][d1] + delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0)) + PQ[d0]*(QC_0*delta[c1][d1] + QC_1*delta[c0][d1] + delta[c0][c1]*(PQ[d1] + QD_1)) + PQ[d1]*(QC_0*delta[c1][d0] + QC_1*delta[c0][d0] + QD_0*delta[c0][c1]))

                        -PQ[a0]*(PQ[a1]*(PQ[c0]*(QC_1*(delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + QD_0*(delta[b0][c1]*delta[b1][d1] + delta[b0][d1]*delta[b1][c1]) + QD_1*(delta[b0][c1]*delta[b1][d0] + delta[b0][d0]*delta[b1][c1])) + PQ[c1]*(QC_0*(delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + QD_0*(delta[b0][c0]*delta[b1][d1] + delta[b0][d1]*delta[b1][c0]) + QD_1*(delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0])) + PQ[d0]*(QC_0*(delta[b0][c1]*delta[b1][d1] + delta[b0][d1]*delta[b1][c1]) + QC_1*(delta[b0][c0]*delta[b1][d1] + delta[b0][d1]*delta[b1][c0]) + QD_1*(delta[b0][c0]*delta[b1][c1] + delta[b0][c1]*delta[b1][c0])) + PQ[d1]*(QC_0*(delta[b0][c1]*delta[b1][d0] + delta[b0][d0]*delta[b1][c1]) + QC_1*(delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0]) + QD_0*(delta[b0][c0]*delta[b1][c1] + delta[b0][c1]*delta[b1][c0]))) + PQ[b0]*(PQ[c0]*(QC_1*(delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + QD_0*(delta[a1][c1]*delta[b1][d1] + delta[a1][d1]*delta[b1][c1]) + QD_1*(delta[a1][c1]*delta[b1][d0] + delta[a1][d0]*delta[b1][c1])) + PQ[c1]*(QC_0*(delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + QD_0*(delta[a1][c0]*delta[b1][d1] + delta[a1][d1]*delta[b1][c0]) + QD_1*(delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0])) + PQ[d0]*(QC_0*(delta[a1][c1]*delta[b1][d1] + delta[a1][d1]*delta[b1][c1]) + QC_1*(delta[a1][c0]*delta[b1][d1] + delta[a1][d1]*delta[b1][c0]) + QD_1*(delta[a1][c0]*delta[b1][c1] + delta[a1][c1]*delta[b1][c0])) + PQ[d1]*(QC_0*(delta[a1][c1]*delta[b1][d0] + delta[a1][d0]*delta[b1][c1]) + QC_1*(delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + QD_0*(delta[a1][c0]*delta[b1][c1] + delta[a1][c1]*delta[b1][c0]))) + PQ[b1]*(PQ[c0]*(QC_1*(delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + QD_0*(delta[a1][c1]*delta[b0][d1] + delta[a1][d1]*delta[b0][c1]) + QD_1*(delta[a1][c1]*delta[b0][d0] + delta[a1][d0]*delta[b0][c1])) + PQ[c1]*(QC_0*(delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + QD_0*(delta[a1][c0]*delta[b0][d1] + delta[a1][d1]*delta[b0][c0]) + QD_1*(delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0])) + PQ[d0]*(QC_0*(delta[a1][c1]*delta[b0][d1] + delta[a1][d1]*delta[b0][c1]) + QC_1*(delta[a1][c0]*delta[b0][d1] + delta[a1][d1]*delta[b0][c0]) + QD_1*(delta[a1][c0]*delta[b0][c1] + delta[a1][c1]*delta[b0][c0])) + PQ[d1]*(QC_0*(delta[a1][c1]*delta[b0][d0] + delta[a1][d0]*delta[b0][c1]) + QC_1*(delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0]) + QD_0*(delta[a1][c0]*delta[b0][c1] + delta[a1][c1]*delta[b0][c0])))) - PQ[a1]*(PQ[b0]*(PQ[c0]*(QC_1*(delta[a0][d0]*delta[b1][d1] + delta[a0][d1]*delta[b1][d0]) + QD_0*(delta[a0][c1]*delta[b1][d1] + delta[a0][d1]*delta[b1][c1]) + QD_1*(delta[a0][c1]*delta[b1][d0] + delta[a0][d0]*delta[b1][c1])) + PQ[c1]*(QC_0*(delta[a0][d0]*delta[b1][d1] + delta[a0][d1]*delta[b1][d0]) + QD_0*(delta[a0][c0]*delta[b1][d1] + delta[a0][d1]*delta[b1][c0]) + QD_1*(delta[a0][c0]*delta[b1][d0] + delta[a0][d0]*delta[b1][c0])) + PQ[d0]*(QC_0*(delta[a0][c1]*delta[b1][d1] + delta[a0][d1]*delta[b1][c1]) + QC_1*(delta[a0][c0]*delta[b1][d1] + delta[a0][d1]*delta[b1][c0]) + QD_1*(delta[a0][c0]*delta[b1][c1] + delta[a0][c1]*delta[b1][c0])) + PQ[d1]*(QC_0*(delta[a0][c1]*delta[b1][d0] + delta[a0][d0]*delta[b1][c1]) + QC_1*(delta[a0][c0]*delta[b1][d0] + delta[a0][d0]*delta[b1][c0]) + QD_0*(delta[a0][c0]*delta[b1][c1] + delta[a0][c1]*delta[b1][c0]))) + PQ[b1]*(PQ[c0]*(QC_1*(delta[a0][d0]*delta[b0][d1] + delta[a0][d1]*delta[b0][d0]) + QD_0*(delta[a0][c1]*delta[b0][d1] + delta[a0][d1]*delta[b0][c1]) + QD_1*(delta[a0][c1]*delta[b0][d0] + delta[a0][d0]*delta[b0][c1])) + PQ[c1]*(QC_0*(delta[a0][d0]*delta[b0][d1] + delta[a0][d1]*delta[b0][d0]) + QD_0*(delta[a0][c0]*delta[b0][d1] + delta[a0][d1]*delta[b0][c0]) + QD_1*(delta[a0][c0]*delta[b0][d0] + delta[a0][d0]*delta[b0][c0])) + PQ[d0]*(QC_0*(delta[a0][c1]*delta[b0][d1] + delta[a0][d1]*delta[b0][c1]) + QC_1*(delta[a0][c0]*delta[b0][d1] + delta[a0][d1]*delta[b0][c0]) + QD_1*(delta[a0][c0]*delta[b0][c1] + delta[a0][c1]*delta[b0][c0])) + PQ[d1]*(QC_0*(delta[a0][c1]*delta[b0][d0] + delta[a0][d0]*delta[b0][c1]) + QC_1*(delta[a0][c0]*delta[b0][d0] + delta[a0][d0]*delta[b0][c0]) + QD_0*(delta[a0][c0]*delta[b0][c1] + delta[a0][c1]*delta[b0][c0])))) - PQ[b0]*PQ[b1]*(PQ[c0]*(QC_1*(delta[a0][d0]*delta[a1][d1] + delta[a0][d1]*delta[a1][d0]) + QD_0*(delta[a0][c1]*delta[a1][d1] + delta[a0][d1]*delta[a1][c1]) + QD_1*(delta[a0][c1]*delta[a1][d0] + delta[a0][d0]*delta[a1][c1])) + PQ[c1]*(QC_0*(delta[a0][d0]*delta[a1][d1] + delta[a0][d1]*delta[a1][d0]) + QD_0*(delta[a0][c0]*delta[a1][d1] + delta[a0][d1]*delta[a1][c0]) + QD_1*(delta[a0][c0]*delta[a1][d0] + delta[a0][d0]*delta[a1][c0])) + PQ[d0]*(QC_0*(delta[a0][c1]*delta[a1][d1] + delta[a0][d1]*delta[a1][c1]) + QC_1*(delta[a0][c0]*delta[a1][d1] + delta[a0][d1]*delta[a1][c0]) + QD_1*(delta[a0][c0]*delta[a1][c1] + delta[a0][c1]*delta[a1][c0])) + PQ[d1]*(QC_0*(delta[a0][c1]*delta[a1][d0] + delta[a0][d0]*delta[a1][c1]) + QC_1*(delta[a0][c0]*delta[a1][d0] + delta[a0][d0]*delta[a1][c0]) + QD_0*(delta[a0][c0]*delta[a1][c1] + delta[a0][c1]*delta[a1][c0])))

                        +(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0])*(-PQ[b1]*PQ[c1]*PQ[d0]*QD_1 - PQ[b1]*PQ[c1]*PQ[d1]*QD_0 - PQ[b1]*PQ[d0]*PQ[d1]*QC_1) + (delta[a0][a1]*delta[b0][c1] + delta[a0][b0]*delta[a1][c1] + delta[a0][c1]*delta[a1][b0])*(-PQ[b1]*PQ[c0]*PQ[d0]*QD_1 - PQ[b1]*PQ[c0]*PQ[d1]*QD_0 - PQ[b1]*PQ[d0]*PQ[d1]*QC_0) + (delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])*(-PQ[b1]*PQ[c0]*PQ[c1]*QD_1 - PQ[b1]*PQ[c0]*PQ[d1]*QC_1 - PQ[b1]*PQ[c1]*PQ[d1]*QC_0) + (delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0])*(-PQ[b1]*PQ[c0]*PQ[c1]*QD_0 - PQ[b1]*PQ[c0]*PQ[d0]*QC_1 - PQ[b1]*PQ[c1]*PQ[d0]*QC_0) + (delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1])*(-PQ[b0]*PQ[c1]*PQ[d0]*QD_1 - PQ[b0]*PQ[c1]*PQ[d1]*QD_0 - PQ[b0]*PQ[d0]*PQ[d1]*QC_1) + (delta[a0][a1]*delta[b1][c1] + delta[a0][b1]*delta[a1][c1] + delta[a0][c1]*delta[a1][b1])*(-PQ[b0]*PQ[c0]*PQ[d0]*QD_1 - PQ[b0]*PQ[c0]*PQ[d1]*QD_0 - PQ[b0]*PQ[d0]*PQ[d1]*QC_0) + (delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1])*(-PQ[b0]*PQ[c0]*PQ[c1]*QD_1 - PQ[b0]*PQ[c0]*PQ[d1]*QC_1 - PQ[b0]*PQ[c1]*PQ[d1]*QC_0) + (delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1])*(-PQ[b0]*PQ[c0]*PQ[c1]*QD_0 - PQ[b0]*PQ[c0]*PQ[d0]*QC_1 - PQ[b0]*PQ[c1]*PQ[d0]*QC_0) + (delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1])*(-PQ[a1]*PQ[c1]*PQ[d0]*QD_1 - PQ[a1]*PQ[c1]*PQ[d1]*QD_0 - PQ[a1]*PQ[d0]*PQ[d1]*QC_1) + (delta[a0][b0]*delta[b1][c1] + delta[a0][b1]*delta[b0][c1] + delta[a0][c1]*delta[b0][b1])*(-PQ[a1]*PQ[c0]*PQ[d0]*QD_1 - PQ[a1]*PQ[c0]*PQ[d1]*QD_0 - PQ[a1]*PQ[d0]*PQ[d1]*QC_0) + (delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1])*(-PQ[a1]*PQ[c0]*PQ[c1]*QD_1 - PQ[a1]*PQ[c0]*PQ[d1]*QC_1 - PQ[a1]*PQ[c1]*PQ[d1]*QC_0) + (delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1])*(-PQ[a1]*PQ[c0]*PQ[c1]*QD_0 - PQ[a1]*PQ[c0]*PQ[d0]*QC_1 - PQ[a1]*PQ[c1]*PQ[d0]*QC_0) + (delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1])*(-PQ[a0]*PQ[c1]*PQ[d0]*QD_1 - PQ[a0]*PQ[c1]*PQ[d1]*QD_0 - PQ[a0]*PQ[d0]*PQ[d1]*QC_1) + (delta[a1][b0]*delta[b1][c1] + delta[a1][b1]*delta[b0][c1] + delta[a1][c1]*delta[b0][b1])*(-PQ[a0]*PQ[c0]*PQ[d0]*QD_1 - PQ[a0]*PQ[c0]*PQ[d1]*QD_0 - PQ[a0]*PQ[d0]*PQ[d1]*QC_0) + (delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1])*(-PQ[a0]*PQ[c0]*PQ[c1]*QD_1 - PQ[a0]*PQ[c0]*PQ[d1]*QC_1 - PQ[a0]*PQ[c1]*PQ[d1]*QC_0) + (delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1])*(-PQ[a0]*PQ[c0]*PQ[c1]*QD_0 - PQ[a0]*PQ[c0]*PQ[d0]*QC_1 - PQ[a0]*PQ[c1]*PQ[d0]*QC_0)

                        -(PQ[c0]*PQ[c1]*(PQ[d0]*QD_1 + PQ[d1]*QD_0) + PQ[d0]*PQ[d1]*(PQ[c0]*QC_1 + PQ[c1]*QC_0))*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * dd_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDD24_FP64(double*         mat_J,
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
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ double   delta[3][3];

    __shared__ double a_i, a_j, r_i[3], r_j[3], S_ij_00, S1, inv_S1;
    __shared__ double PA_0, PA_1, PB_0, PB_1;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta[0][0] = 1.0; delta[0][1] = 0.0; delta[0][2] = 0.0;
        delta[1][0] = 0.0; delta[1][1] = 1.0; delta[1][2] = 0.0;
        delta[2][0] = 0.0; delta[2][1] = 0.0; delta[2][2] = 1.0;

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
            inv_S1 = 1.0 / S1;

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

    for (uint32_t m = 0; m < prec_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= dd_prim_pair_count))
        {
            break;
        }

        const auto k = dd_first_inds[kl];
        const auto l = dd_second_inds[kl];

        const auto a_k = d_prim_info[k / 6 + d_prim_count * 0];

        const double r_k[3] = {d_prim_info[k / 6 + d_prim_count * 2],
                               d_prim_info[k / 6 + d_prim_count * 3],
                               d_prim_info[k / 6 + d_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = dd_pair_data[kl];

        const auto c0 = d_cart_inds[k % 6][0];
        const auto c1 = d_cart_inds[k % 6][1];
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F8_t[6];

        gpu::computeBoysFunction(F8_t, S1 * S2 * inv_S4 * r2_PQ, 5, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QC_1 = (a_l * inv_S2) * (r_l[c1] - r_k[c1]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                    + F8_t[5] * 0.5 * S1 * S1 * S1 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        -(PQ[c0]*PQ[c1]*delta[d0][d1] + PQ[c0]*PQ[d0]*delta[c1][d1] + PQ[c0]*PQ[d1]*delta[c1][d0] + PQ[c1]*PQ[d0]*delta[c0][d1] + PQ[c1]*PQ[d1]*delta[c0][d0] + PQ[d0]*PQ[d1]*delta[c0][c1])*(PA_0*PA_1*PQ[b0]*PQ[b1] + PA_0*PB_0*PQ[a1]*PQ[b1] + PA_0*PB_1*PQ[a1]*PQ[b0] + PA_1*PB_0*PQ[a0]*PQ[b1] + PA_1*PB_1*PQ[a0]*PQ[b0] + PB_0*PB_1*PQ[a0]*PQ[a1])

                        -PQ[c0]*PQ[c1]*(PA_0*(PA_1*(PQ[b0]*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PQ[b1]*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0])) + PB_0*(PQ[a1]*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PQ[b1]*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0])) + PB_1*(PQ[a1]*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0]) + PQ[b0]*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0]))) + PA_1*(PB_0*(PQ[a0]*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PQ[b1]*(PQ[d0]*delta[a0][d1] + PQ[d1]*delta[a0][d0])) + PB_1*(PQ[a0]*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0]) + PQ[b0]*(PQ[d0]*delta[a0][d1] + PQ[d1]*delta[a0][d0]))) + PB_0*PB_1*(PQ[a0]*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0]) + PQ[a1]*(PQ[d0]*delta[a0][d1] + PQ[d1]*delta[a0][d0]))) - PQ[d0]*PQ[d1]*(PA_0*(PA_1*(PQ[b0]*(PQ[c0]*delta[b1][c1] + PQ[c1]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[b0][c1] + PQ[c1]*delta[b0][c0])) + PB_0*(PQ[a1]*(PQ[c0]*delta[b1][c1] + PQ[c1]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[a1][c1] + PQ[c1]*delta[a1][c0])) + PB_1*(PQ[a1]*(PQ[c0]*delta[b0][c1] + PQ[c1]*delta[b0][c0]) + PQ[b0]*(PQ[c0]*delta[a1][c1] + PQ[c1]*delta[a1][c0]))) + PA_1*(PB_0*(PQ[a0]*(PQ[c0]*delta[b1][c1] + PQ[c1]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[a0][c1] + PQ[c1]*delta[a0][c0])) + PB_1*(PQ[a0]*(PQ[c0]*delta[b0][c1] + PQ[c1]*delta[b0][c0]) + PQ[b0]*(PQ[c0]*delta[a0][c1] + PQ[c1]*delta[a0][c0]))) + PB_0*PB_1*(PQ[a0]*(PQ[c0]*delta[a1][c1] + PQ[c1]*delta[a1][c0]) + PQ[a1]*(PQ[c0]*delta[a0][c1] + PQ[c1]*delta[a0][c0])))

                        +PQ[c0]*PQ[c1]*PQ[d0]*PQ[d1]*(PA_0*(-PA_1*delta[b0][b1] - PB_0*delta[a1][b1] - PB_1*delta[a1][b0] + PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PA_1*(PQ[a0]*delta[b0][b1] + delta[a0][b0]*(-PB_1 + PQ[b1]) + delta[a0][b1]*(-PB_0 + PQ[b0])) + PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + delta[a0][a1]*(-PB_1 + PQ[b1])) + PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]))

                    )

                    + F8_t[5] * 0.5 * S1 * S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        +(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))*(PQ[c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + PQ[c1]*(QC_0*delta[d0][d1] + delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0)) + PQ[d0]*(QC_0*delta[c1][d1] + QC_1*delta[c0][d1] + delta[c0][c1]*(PQ[d1] + QD_1)) + PQ[d1]*(QC_0*delta[c1][d0] + QC_1*delta[c0][d0] + QD_0*delta[c0][c1]))

                        +PQ[a0]*(PA_1*(PQ[b0]*(PQ[c0]*(PQ[c1]*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + PQ[d0]*(QC_1*delta[b1][d1] + QD_1*delta[b1][c1]) + PQ[d1]*(QC_1*delta[b1][d0] + QD_0*delta[b1][c1])) + PQ[c1]*(PQ[d0]*(QC_0*delta[b1][d1] + QD_1*delta[b1][c0]) + PQ[d1]*(QC_0*delta[b1][d0] + QD_0*delta[b1][c0]))) + PQ[b1]*(PQ[c0]*(PQ[c1]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + PQ[d0]*(QC_1*delta[b0][d1] + QD_1*delta[b0][c1]) + PQ[d1]*(QC_1*delta[b0][d0] + QD_0*delta[b0][c1])) + PQ[c1]*(PQ[d0]*(QC_0*delta[b0][d1] + QD_1*delta[b0][c0]) + PQ[d1]*(QC_0*delta[b0][d0] + QD_0*delta[b0][c0])))) + PB_0*(PQ[a1]*(PQ[c0]*(PQ[c1]*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + PQ[d0]*(QC_1*delta[b1][d1] + QD_1*delta[b1][c1]) + PQ[d1]*(QC_1*delta[b1][d0] + QD_0*delta[b1][c1])) + PQ[c1]*(PQ[d0]*(QC_0*delta[b1][d1] + QD_1*delta[b1][c0]) + PQ[d1]*(QC_0*delta[b1][d0] + QD_0*delta[b1][c0]))) + PQ[b1]*(PQ[c0]*(PQ[c1]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]) + PQ[d0]*(QC_1*delta[a1][d1] + QD_1*delta[a1][c1]) + PQ[d1]*(QC_1*delta[a1][d0] + QD_0*delta[a1][c1])) + PQ[c1]*(PQ[d0]*(QC_0*delta[a1][d1] + QD_1*delta[a1][c0]) + PQ[d1]*(QC_0*delta[a1][d0] + QD_0*delta[a1][c0])))) + PB_1*(PQ[a1]*(PQ[c0]*(PQ[c1]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + PQ[d0]*(QC_1*delta[b0][d1] + QD_1*delta[b0][c1]) + PQ[d1]*(QC_1*delta[b0][d0] + QD_0*delta[b0][c1])) + PQ[c1]*(PQ[d0]*(QC_0*delta[b0][d1] + QD_1*delta[b0][c0]) + PQ[d1]*(QC_0*delta[b0][d0] + QD_0*delta[b0][c0]))) + PQ[b0]*(PQ[c0]*(PQ[c1]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]) + PQ[d0]*(QC_1*delta[a1][d1] + QD_1*delta[a1][c1]) + PQ[d1]*(QC_1*delta[a1][d0] + QD_0*delta[a1][c1])) + PQ[c1]*(PQ[d0]*(QC_0*delta[a1][d1] + QD_1*delta[a1][c0]) + PQ[d1]*(QC_0*delta[a1][d0] + QD_0*delta[a1][c0]))))) + PQ[a1]*(PA_0*(PQ[b0]*(PQ[c0]*(PQ[c1]*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + PQ[d0]*(QC_1*delta[b1][d1] + QD_1*delta[b1][c1]) + PQ[d1]*(QC_1*delta[b1][d0] + QD_0*delta[b1][c1])) + PQ[c1]*(PQ[d0]*(QC_0*delta[b1][d1] + QD_1*delta[b1][c0]) + PQ[d1]*(QC_0*delta[b1][d0] + QD_0*delta[b1][c0]))) + PQ[b1]*(PQ[c0]*(PQ[c1]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + PQ[d0]*(QC_1*delta[b0][d1] + QD_1*delta[b0][c1]) + PQ[d1]*(QC_1*delta[b0][d0] + QD_0*delta[b0][c1])) + PQ[c1]*(PQ[d0]*(QC_0*delta[b0][d1] + QD_1*delta[b0][c0]) + PQ[d1]*(QC_0*delta[b0][d0] + QD_0*delta[b0][c0])))) + (PB_0*PQ[b1] + PB_1*PQ[b0])*(PQ[c0]*(PQ[c1]*(QD_0*delta[a0][d1] + QD_1*delta[a0][d0]) + PQ[d0]*(QC_1*delta[a0][d1] + QD_1*delta[a0][c1]) + PQ[d1]*(QC_1*delta[a0][d0] + QD_0*delta[a0][c1])) + PQ[c1]*(PQ[d0]*(QC_0*delta[a0][d1] + QD_1*delta[a0][c0]) + PQ[d1]*(QC_0*delta[a0][d0] + QD_0*delta[a0][c0])))) + PQ[b0]*PQ[b1]*(PA_0*(PQ[c0]*(PQ[c1]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]) + PQ[d0]*(QC_1*delta[a1][d1] + QD_1*delta[a1][c1]) + PQ[d1]*(QC_1*delta[a1][d0] + QD_0*delta[a1][c1])) + PQ[c1]*(PQ[d0]*(QC_0*delta[a1][d1] + QD_1*delta[a1][c0]) + PQ[d1]*(QC_0*delta[a1][d0] + QD_0*delta[a1][c0]))) + PA_1*(PQ[c0]*(PQ[c1]*(QD_0*delta[a0][d1] + QD_1*delta[a0][d0]) + PQ[d0]*(QC_1*delta[a0][d1] + QD_1*delta[a0][c1]) + PQ[d1]*(QC_1*delta[a0][d0] + QD_0*delta[a0][c1])) + PQ[c1]*(PQ[d0]*(QC_0*delta[a0][d1] + QD_1*delta[a0][c0]) + PQ[d1]*(QC_0*delta[a0][d0] + QD_0*delta[a0][c0]))) + PQ[d0]*PQ[d1]*(PA_0*(QC_0*delta[a1][c1] + QC_1*delta[a1][c0]) + PA_1*(QC_0*delta[a0][c1] + QC_1*delta[a0][c0]))) + PQ[d0]*PQ[d1]*(PQ[a0]*(PA_1*(PQ[b0]*(QC_0*delta[b1][c1] + QC_1*delta[b1][c0]) + PQ[b1]*(QC_0*delta[b0][c1] + QC_1*delta[b0][c0])) + PB_0*(PQ[a1]*(QC_0*delta[b1][c1] + QC_1*delta[b1][c0]) + PQ[b1]*(QC_0*delta[a1][c1] + QC_1*delta[a1][c0])) + PB_1*(PQ[a1]*(QC_0*delta[b0][c1] + QC_1*delta[b0][c0]) + PQ[b0]*(QC_0*delta[a1][c1] + QC_1*delta[a1][c0]))) + PQ[a1]*(PA_0*(PQ[b0]*(QC_0*delta[b1][c1] + QC_1*delta[b1][c0]) + PQ[b1]*(QC_0*delta[b0][c1] + QC_1*delta[b0][c0])) + (PB_0*PQ[b1] + PB_1*PQ[b0])*(QC_0*delta[a0][c1] + QC_1*delta[a0][c0])))

                        +(PQ[c0]*PQ[c1]*(PQ[d0]*QD_1 + PQ[d1]*QD_0) + PQ[d0]*PQ[d1]*(PQ[c0]*QC_1 + PQ[c1]*QC_0))*(PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PQ[a0]*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0] - PQ[a1]*delta[b0][b1] - PQ[b0]*delta[a1][b1] - PQ[b1]*delta[a1][b0]) + PQ[a1]*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PQ[b0]*(PB_1*delta[a0][a1] + delta[a0][b1]*(PA_1 - PQ[a1])) + PQ[b1]*(delta[a0][a1]*(PB_0 - PQ[b0]) + delta[a0][b0]*(PA_1 - PQ[a1])))

                    )

                    + F8_t[5] * (-0.5) * S1 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*(QC_0*(PQ[c1]*delta[d0][d1] + PQ[d0]*delta[c1][d1] + PQ[d1]*delta[c1][d0]) + QC_1*(PQ[d0]*delta[c0][d1] + PQ[d1]*delta[c0][d0] + delta[d0][d1]*(PQ[c0] + QC_0)) + QD_0*(PQ[d1]*delta[c0][c1] + delta[c0][d1]*(PQ[c1] + QC_1) + delta[c1][d1]*(PQ[c0] + QC_0)) + QD_1*(delta[c0][c1]*(PQ[d0] + QD_0) + delta[c0][d0]*(PQ[c1] + QC_1) + delta[c1][d0]*(PQ[c0] + QC_0)))

                        +PQ[a0]*PQ[a1]*(PQ[b0]*(QC_0*(PQ[c1]*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + PQ[d0]*(QC_1*delta[b1][d1] + QD_1*delta[b1][c1]) + PQ[d1]*(QC_1*delta[b1][d0] + QD_0*delta[b1][c1])) + QC_1*(PQ[c0]*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + delta[b1][c0]*(PQ[d0]*QD_1 + PQ[d1]*QD_0))) + PQ[b1]*(QC_0*(PQ[c1]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + PQ[d0]*(QC_1*delta[b0][d1] + QD_1*delta[b0][c1]) + PQ[d1]*(QC_1*delta[b0][d0] + QD_0*delta[b0][c1])) + QC_1*(PQ[c0]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + delta[b0][c0]*(PQ[d0]*QD_1 + PQ[d1]*QD_0))) + QD_0*QD_1*(PQ[b0]*(PQ[c0]*delta[b1][c1] + PQ[c1]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[b0][c1] + PQ[c1]*delta[b0][c0]))) + PQ[b0]*PQ[b1]*(PQ[a0]*(QC_0*(PQ[c1]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]) + PQ[d0]*(QC_1*delta[a1][d1] + QD_1*delta[a1][c1]) + PQ[d1]*(QC_1*delta[a1][d0] + QD_0*delta[a1][c1])) + QC_1*(PQ[c0]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]) + delta[a1][c0]*(PQ[d0]*QD_1 + PQ[d1]*QD_0))) + PQ[a1]*(QC_0*(PQ[c1]*(QD_0*delta[a0][d1] + QD_1*delta[a0][d0]) + PQ[d0]*(QC_1*delta[a0][d1] + QD_1*delta[a0][c1]) + PQ[d1]*(QC_1*delta[a0][d0] + QD_0*delta[a0][c1])) + QC_1*(PQ[c0]*(QD_0*delta[a0][d1] + QD_1*delta[a0][d0]) + delta[a0][c0]*(PQ[d0]*QD_1 + PQ[d1]*QD_0))) + QD_0*QD_1*(PQ[a0]*(PQ[c0]*delta[a1][c1] + PQ[c1]*delta[a1][c0]) + PQ[a1]*(PQ[c0]*delta[a0][c1] + PQ[c1]*delta[a0][c0])))

                        +(PQ[a0]*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PQ[a1]*(PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PQ[b0]*PQ[b1]*delta[a0][a1])*(PQ[c0]*QD_0*(PQ[c1]*QD_1 + PQ[d1]*QC_1) + PQ[c1]*QC_0*(PQ[d0]*QD_1 + PQ[d1]*QD_0) + PQ[d0]*QC_1*(PQ[c0]*QD_1 + PQ[d1]*QC_0))

                    )

                    + F8_t[5] * S1 * S1 * S1 * S1 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        +PQ[c0]*PQ[c1]*PQ[d0]*PQ[d1]*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))

                    )

                    + F8_t[5] * S1 * S1 * S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        -(PQ[c0]*PQ[c1]*(PQ[d0]*QD_1 + PQ[d1]*QD_0) + PQ[d0]*PQ[d1]*(PQ[c0]*QC_1 + PQ[c1]*QC_0))*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))

                    )

                    + F8_t[5] * S1 * S1 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        +(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))*(PQ[c0]*QD_0*(PQ[c1]*QD_1 + PQ[d1]*QC_1) + PQ[c1]*QC_0*(PQ[d0]*QD_1 + PQ[d1]*QD_0) + PQ[d0]*QC_1*(PQ[c0]*QD_1 + PQ[d1]*QC_0))

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * dd_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDDD25_FP64(double*         mat_J,
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
{
    // each thread row scans over [ij|??] and sum up to a primitive J matrix element
    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015

    __shared__ double   ERIs[TILE_DIM_LARGE + 1];
    __shared__ uint32_t d_cart_inds[6][2];
    __shared__ double   delta[3][3];

    __shared__ double a_i, a_j, r_i[3], r_j[3], S_ij_00, S1, inv_S1;
    __shared__ double PA_0, PA_1, PB_0, PB_1;
    __shared__ uint32_t i, j, a0, a1, b0, b1;

    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ij_tile = blockIdx.x;
    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];

    if ((threadIdx.y == 0) && (threadIdx.x == 0))
    {

        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;
        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;
        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;
        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;
        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;
        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;

        delta[0][0] = 1.0; delta[0][1] = 0.0; delta[0][2] = 0.0;
        delta[1][0] = 0.0; delta[1][1] = 1.0; delta[1][2] = 0.0;
        delta[2][0] = 0.0; delta[2][1] = 0.0; delta[2][2] = 1.0;

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
            inv_S1 = 1.0 / S1;

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

    for (uint32_t m = 0; m < prec_cut; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= dd_prim_pair_count))
        {
            break;
        }

        const auto k = dd_first_inds[kl];
        const auto l = dd_second_inds[kl];

        const auto a_k = d_prim_info[k / 6 + d_prim_count * 0];

        const double r_k[3] = {d_prim_info[k / 6 + d_prim_count * 2],
                               d_prim_info[k / 6 + d_prim_count * 3],
                               d_prim_info[k / 6 + d_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = dd_pair_data[kl];

        const auto c0 = d_cart_inds[k % 6][0];
        const auto c1 = d_cart_inds[k % 6][1];
        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F8_t[9];

        gpu::computeBoysFunction(F8_t, S1 * S2 * inv_S4 * r2_PQ, 8, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QC_1 = (a_l * inv_S2) * (r_l[c1] - r_k[c1]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                    + F8_t[5] * S1 * S2 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        -PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*(QC_0*QD_1*(PQ[c1]*QD_0 + PQ[d0]*QC_1) + QC_1*QD_0*(PQ[c0]*QD_1 + PQ[d1]*QC_0))

                    )

                    + F8_t[6] * 0.5 * S1 * S1 * S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        -(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))*(PQ[c0]*(PQ[c1]*delta[d0][d1] + PQ[d0]*delta[c1][d1] + PQ[d1]*delta[c1][d0]) + PQ[c1]*(PQ[d0]*delta[c0][d1] + PQ[d1]*delta[c0][d0]) + PQ[d0]*PQ[d1]*delta[c0][c1])

                        -PQ[c0]*PQ[c1]*(PQ[a0]*(PA_1*(PQ[b0]*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PQ[b1]*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0])) + PB_0*(PQ[a1]*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PQ[b1]*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0])) + PB_1*(PQ[a1]*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0]) + PQ[b0]*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0]))) + PQ[a1]*(PA_0*(PQ[b0]*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PQ[b1]*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0])) + (PB_0*PQ[b1] + PB_1*PQ[b0])*(PQ[d0]*delta[a0][d1] + PQ[d1]*delta[a0][d0])) + PQ[b0]*PQ[b1]*(PA_0*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0]) + PA_1*(PQ[d0]*delta[a0][d1] + PQ[d1]*delta[a0][d0]))) - PQ[d0]*PQ[d1]*(PQ[a0]*(PA_1*(PQ[b0]*(PQ[c0]*delta[b1][c1] + PQ[c1]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[b0][c1] + PQ[c1]*delta[b0][c0])) + PB_0*(PQ[a1]*(PQ[c0]*delta[b1][c1] + PQ[c1]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[a1][c1] + PQ[c1]*delta[a1][c0])) + PB_1*(PQ[a1]*(PQ[c0]*delta[b0][c1] + PQ[c1]*delta[b0][c0]) + PQ[b0]*(PQ[c0]*delta[a1][c1] + PQ[c1]*delta[a1][c0]))) + PQ[a1]*(PA_0*(PQ[b0]*(PQ[c0]*delta[b1][c1] + PQ[c1]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[b0][c1] + PQ[c1]*delta[b0][c0])) + (PB_0*PQ[b1] + PB_1*PQ[b0])*(PQ[c0]*delta[a0][c1] + PQ[c1]*delta[a0][c0])) + PQ[b0]*PQ[b1]*(PA_0*(PQ[c0]*delta[a1][c1] + PQ[c1]*delta[a1][c0]) + PA_1*(PQ[c0]*delta[a0][c1] + PQ[c1]*delta[a0][c0])))

                        +PQ[c0]*PQ[c1]*PQ[d0]*PQ[d1]*(-PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) - PA_1*(PQ[a0]*delta[b0][b1] + PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) - PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) - PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]) + PQ[a0]*PQ[a1]*delta[b0][b1] + PQ[a0]*PQ[b0]*delta[a1][b1] + PQ[a0]*PQ[b1]*delta[a1][b0] + PQ[a1]*PQ[b0]*delta[a0][b1] + PQ[a1]*PQ[b1]*delta[a0][b0] + PQ[b0]*PQ[b1]*delta[a0][a1])

                    )

                    + F8_t[6] * 0.5 * S1 * S1 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*(PQ[c0]*(delta[c1][d0]*(PQ[d1] + QD_1) + delta[c1][d1]*(PQ[d0] + QD_0) + delta[d0][d1]*(PQ[c1] + QC_1)) + PQ[c1]*(QC_0*delta[d0][d1] + delta[c0][d0]*(PQ[d1] + QD_1) + delta[c0][d1]*(PQ[d0] + QD_0)) + PQ[d0]*(QC_0*delta[c1][d1] + QC_1*delta[c0][d1] + delta[c0][c1]*(PQ[d1] + QD_1)) + PQ[d1]*(QC_0*delta[c1][d0] + QC_1*delta[c0][d0] + QD_0*delta[c0][c1]))

                        +PQ[a0]*PQ[a1]*(PQ[b0]*(PQ[c0]*(PQ[c1]*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + PQ[d0]*(QC_1*delta[b1][d1] + QD_1*delta[b1][c1]) + PQ[d1]*(QC_1*delta[b1][d0] + QD_0*delta[b1][c1])) + PQ[c1]*(PQ[d0]*(QC_0*delta[b1][d1] + QD_1*delta[b1][c0]) + PQ[d1]*(QC_0*delta[b1][d0] + QD_0*delta[b1][c0]))) + PQ[b1]*(PQ[c0]*(PQ[c1]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + PQ[d0]*(QC_1*delta[b0][d1] + QD_1*delta[b0][c1]) + PQ[d1]*(QC_1*delta[b0][d0] + QD_0*delta[b0][c1])) + PQ[c1]*(PQ[d0]*(QC_0*delta[b0][d1] + QD_1*delta[b0][c0]) + PQ[d1]*(QC_0*delta[b0][d0] + QD_0*delta[b0][c0]))) + PQ[d0]*PQ[d1]*(PQ[b0]*(QC_0*delta[b1][c1] + QC_1*delta[b1][c0]) + PQ[b1]*(QC_0*delta[b0][c1] + QC_1*delta[b0][c0]))) + PQ[b0]*PQ[b1]*(PQ[a0]*(PQ[c0]*(PQ[c1]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]) + PQ[d0]*(QC_1*delta[a1][d1] + QD_1*delta[a1][c1]) + PQ[d1]*(QC_1*delta[a1][d0] + QD_0*delta[a1][c1])) + PQ[c1]*(PQ[d0]*(QC_0*delta[a1][d1] + QD_1*delta[a1][c0]) + PQ[d1]*(QC_0*delta[a1][d0] + QD_0*delta[a1][c0]))) + PQ[a1]*(PQ[c0]*(PQ[c1]*(QD_0*delta[a0][d1] + QD_1*delta[a0][d0]) + PQ[d0]*(QC_1*delta[a0][d1] + QD_1*delta[a0][c1]) + PQ[d1]*(QC_1*delta[a0][d0] + QD_0*delta[a0][c1])) + PQ[c1]*(PQ[d0]*(QC_0*delta[a0][d1] + QD_1*delta[a0][c0]) + PQ[d1]*(QC_0*delta[a0][d0] + QD_0*delta[a0][c0]))) + PQ[d0]*PQ[d1]*(PQ[a0]*(QC_0*delta[a1][c1] + QC_1*delta[a1][c0]) + PQ[a1]*(QC_0*delta[a0][c1] + QC_1*delta[a0][c0])))

                        +(PQ[c0]*PQ[c1]*(PQ[d0]*QD_1 + PQ[d1]*QD_0) + PQ[d0]*PQ[d1]*(PQ[c0]*QC_1 + PQ[c1]*QC_0))*(PQ[a0]*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PQ[a1]*(PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PQ[b0]*PQ[b1]*delta[a0][a1])

                    )

                    + F8_t[6] * S1 * S1 * S1 * S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        +PQ[c0]*PQ[c1]*PQ[d0]*PQ[d1]*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))

                    )

                    + F8_t[6] * S1 * S1 * S1 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        -(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))*(PQ[c0]*PQ[c1]*(PQ[d0]*QD_1 + PQ[d1]*QD_0) + PQ[d0]*PQ[d1]*(PQ[c0]*QC_1 + PQ[c1]*QC_0))

                    )

                    + F8_t[6] * S1 * S1 * S2 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*(PQ[c0]*QD_0*(PQ[c1]*QD_1 + PQ[d1]*QC_1) + PQ[c1]*QC_0*(PQ[d0]*QD_1 + PQ[d1]*QD_0) + PQ[d0]*QC_1*(PQ[c0]*QD_1 + PQ[d1]*QC_0))

                    )

                    + F8_t[6] * 0.25 * S1 * S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        +PQ[a0]*PQ[b0]*(PQ[a1]*(PQ[b1]*(delta[c0][c1]*delta[d0][d1] + delta[c0][d0]*delta[c1][d1] + delta[c0][d1]*delta[c1][d0]) + PQ[c0]*(delta[b1][c1]*delta[d0][d1] + delta[b1][d0]*delta[c1][d1] + delta[b1][d1]*delta[c1][d0]) + PQ[c1]*(delta[b1][c0]*delta[d0][d1] + delta[b1][d0]*delta[c0][d1] + delta[b1][d1]*delta[c0][d0]) + PQ[d0]*(delta[b1][c0]*delta[c1][d1] + delta[b1][c1]*delta[c0][d1] + delta[b1][d1]*delta[c0][c1]) + PQ[d1]*(delta[b1][c0]*delta[c1][d0] + delta[b1][c1]*delta[c0][d0] + delta[b1][d0]*delta[c0][c1])) + PQ[b1]*PQ[c1]*(delta[a1][c0]*delta[d0][d1] + delta[a1][d0]*delta[c0][d1] + delta[a1][d1]*delta[c0][d0]) + PQ[c0]*PQ[d0]*(delta[a1][b1]*delta[c1][d1] + delta[a1][c1]*delta[b1][d1] + delta[a1][d1]*delta[b1][c1]) + PQ[d1]*(PQ[b1]*(delta[a1][c0]*delta[c1][d0] + delta[a1][c1]*delta[c0][d0] + delta[a1][d0]*delta[c0][c1]) + PQ[c0]*(delta[a1][b1]*delta[c1][d0] + delta[a1][c1]*delta[b1][d0] + delta[a1][d0]*delta[b1][c1]) + PQ[c1]*(delta[a1][b1]*delta[c0][d0] + delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + PQ[d0]*(delta[a1][b1]*delta[c0][c1] + delta[a1][c0]*delta[b1][c1] + delta[a1][c1]*delta[b1][c0]))) + PQ[a1]*PQ[d1]*(PQ[a0]*(PQ[b1]*(delta[b0][c0]*delta[c1][d0] + delta[b0][c1]*delta[c0][d0] + delta[b0][d0]*delta[c0][c1]) + PQ[c0]*(delta[b0][b1]*delta[c1][d0] + delta[b0][c1]*delta[b1][d0] + delta[b0][d0]*delta[b1][c1]) + PQ[c1]*(delta[b0][b1]*delta[c0][d0] + delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0]) + PQ[d0]*(delta[b0][b1]*delta[c0][c1] + delta[b0][c0]*delta[b1][c1] + delta[b0][c1]*delta[b1][c0])) + PQ[b0]*(PQ[b1]*(delta[a0][c0]*delta[c1][d0] + delta[a0][c1]*delta[c0][d0] + delta[a0][d0]*delta[c0][c1]) + PQ[c0]*(delta[a0][b1]*delta[c1][d0] + delta[a0][c1]*delta[b1][d0] + delta[a0][d0]*delta[b1][c1]) + PQ[c1]*(delta[a0][b1]*delta[c0][d0] + delta[a0][c0]*delta[b1][d0] + delta[a0][d0]*delta[b1][c0]) + PQ[d0]*(delta[a0][b1]*delta[c0][c1] + delta[a0][c0]*delta[b1][c1] + delta[a0][c1]*delta[b1][c0])) + PQ[b1]*PQ[d0]*(delta[a0][b0]*delta[c0][c1] + delta[a0][c0]*delta[b0][c1] + delta[a0][c1]*delta[b0][c0]) + PQ[c0]*PQ[c1]*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1])) + PQ[b1]*(PQ[c0]*(PQ[a0]*(PQ[a1]*(delta[b0][c1]*delta[d0][d1] + delta[b0][d0]*delta[c1][d1] + delta[b0][d1]*delta[c1][d0]) + PQ[b0]*(delta[a1][c1]*delta[d0][d1] + delta[a1][d0]*delta[c1][d1] + delta[a1][d1]*delta[c1][d0]) + PQ[c1]*(delta[a1][b0]*delta[d0][d1] + delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + PQ[d0]*(delta[a1][b0]*delta[c1][d1] + delta[a1][c1]*delta[b0][d1] + delta[a1][d1]*delta[b0][c1]) + PQ[d1]*(delta[a1][b0]*delta[c1][d0] + delta[a1][c1]*delta[b0][d0] + delta[a1][d0]*delta[b0][c1])) + PQ[a1]*(PQ[b0]*(delta[a0][c1]*delta[d0][d1] + delta[a0][d0]*delta[c1][d1] + delta[a0][d1]*delta[c1][d0]) + PQ[d1]*(delta[a0][b0]*delta[c1][d0] + delta[a0][c1]*delta[b0][d0] + delta[a0][d0]*delta[b0][c1])) + PQ[b0]*PQ[d1]*(delta[a0][a1]*delta[c1][d0] + delta[a0][c1]*delta[a1][d0] + delta[a0][d0]*delta[a1][c1]) + PQ[c1]*PQ[d0]*(delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0])) + PQ[c1]*(PQ[a1]*(PQ[a0]*(delta[b0][c0]*delta[d0][d1] + delta[b0][d0]*delta[c0][d1] + delta[b0][d1]*delta[c0][d0]) + PQ[b0]*(delta[a0][c0]*delta[d0][d1] + delta[a0][d0]*delta[c0][d1] + delta[a0][d1]*delta[c0][d0]) + PQ[c0]*(delta[a0][b0]*delta[d0][d1] + delta[a0][d0]*delta[b0][d1] + delta[a0][d1]*delta[b0][d0]) + PQ[d0]*(delta[a0][b0]*delta[c0][d1] + delta[a0][c0]*delta[b0][d1] + delta[a0][d1]*delta[b0][c0]) + PQ[d1]*(delta[a0][b0]*delta[c0][d0] + delta[a0][c0]*delta[b0][d0] + delta[a0][d0]*delta[b0][c0])) + PQ[d1]*(PQ[a0]*(delta[a1][b0]*delta[c0][d0] + delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0]) + PQ[b0]*(delta[a0][a1]*delta[c0][d0] + delta[a0][c0]*delta[a1][d0] + delta[a0][d0]*delta[a1][c0]) + PQ[c0]*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]) + PQ[d0]*(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0]))) + PQ[d0]*(PQ[a0]*(PQ[a1]*(delta[b0][c0]*delta[c1][d1] + delta[b0][c1]*delta[c0][d1] + delta[b0][d1]*delta[c0][c1]) + PQ[d1]*(delta[a1][b0]*delta[c0][c1] + delta[a1][c0]*delta[b0][c1] + delta[a1][c1]*delta[b0][c0])) + PQ[b0]*(PQ[a0]*(delta[a1][c0]*delta[c1][d1] + delta[a1][c1]*delta[c0][d1] + delta[a1][d1]*delta[c0][c1]) + PQ[a1]*(delta[a0][c0]*delta[c1][d1] + delta[a0][c1]*delta[c0][d1] + delta[a0][d1]*delta[c0][c1]) + PQ[c0]*(delta[a0][a1]*delta[c1][d1] + delta[a0][c1]*delta[a1][d1] + delta[a0][d1]*delta[a1][c1]) + PQ[c1]*(delta[a0][a1]*delta[c0][d1] + delta[a0][c0]*delta[a1][d1] + delta[a0][d1]*delta[a1][c0]) + PQ[d1]*(delta[a0][a1]*delta[c0][c1] + delta[a0][c0]*delta[a1][c1] + delta[a0][c1]*delta[a1][c0])))) + PQ[c0]*(PQ[c1]*(PQ[a0]*(PQ[a1]*(delta[b0][b1]*delta[d0][d1] + delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + PQ[d1]*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1])) + PQ[b0]*(PQ[a0]*(delta[a1][b1]*delta[d0][d1] + delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + PQ[a1]*(delta[a0][b1]*delta[d0][d1] + delta[a0][d0]*delta[b1][d1] + delta[a0][d1]*delta[b1][d0]) + PQ[b1]*(delta[a0][a1]*delta[d0][d1] + delta[a0][d0]*delta[a1][d1] + delta[a0][d1]*delta[a1][d0]) + PQ[d0]*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + PQ[d1]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]))) + PQ[d0]*(PQ[a1]*(PQ[a0]*(delta[b0][b1]*delta[c1][d1] + delta[b0][c1]*delta[b1][d1] + delta[b0][d1]*delta[b1][c1]) + PQ[b0]*(delta[a0][b1]*delta[c1][d1] + delta[a0][c1]*delta[b1][d1] + delta[a0][d1]*delta[b1][c1]) + PQ[b1]*(delta[a0][b0]*delta[c1][d1] + delta[a0][c1]*delta[b0][d1] + delta[a0][d1]*delta[b0][c1]) + PQ[c1]*(delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1]) + PQ[d1]*(delta[a0][b0]*delta[b1][c1] + delta[a0][b1]*delta[b0][c1] + delta[a0][c1]*delta[b0][b1])) + PQ[d1]*(PQ[a0]*(delta[a1][b0]*delta[b1][c1] + delta[a1][b1]*delta[b0][c1] + delta[a1][c1]*delta[b0][b1]) + PQ[b0]*(delta[a0][a1]*delta[b1][c1] + delta[a0][b1]*delta[a1][c1] + delta[a0][c1]*delta[a1][b1]) + PQ[b1]*(delta[a0][a1]*delta[b0][c1] + delta[a0][b0]*delta[a1][c1] + delta[a0][c1]*delta[a1][b0]) + PQ[c1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])))) + PQ[c1]*PQ[d0]*(PQ[a0]*(PQ[a1]*(delta[b0][b1]*delta[c0][d1] + delta[b0][c0]*delta[b1][d1] + delta[b0][d1]*delta[b1][c0]) + PQ[b0]*(delta[a1][b1]*delta[c0][d1] + delta[a1][c0]*delta[b1][d1] + delta[a1][d1]*delta[b1][c0]) + PQ[b1]*(delta[a1][b0]*delta[c0][d1] + delta[a1][c0]*delta[b0][d1] + delta[a1][d1]*delta[b0][c0]) + PQ[c0]*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + PQ[d1]*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1])) + PQ[a1]*(PQ[b0]*(delta[a0][b1]*delta[c0][d1] + delta[a0][c0]*delta[b1][d1] + delta[a0][d1]*delta[b1][c0]) + PQ[d1]*(delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1])) + PQ[b0]*PQ[d1]*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]))

                    )

                    + F8_t[7] * (-0.5) * S1 * S1 * S1 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        +PQ[a0]*PQ[c1]*(PQ[a1]*(PQ[b0]*PQ[d0]*(PQ[b1]*delta[c0][d1] + PQ[c0]*delta[b1][d1] + PQ[d1]*delta[b1][c0]) + PQ[b1]*PQ[d1]*(PQ[b0]*delta[c0][d0] + PQ[c0]*delta[b0][d0] + PQ[d0]*delta[b0][c0])) + PQ[c0]*(PQ[b0]*PQ[d1]*(PQ[a1]*delta[b1][d0] + PQ[b1]*delta[a1][d0] + PQ[d0]*delta[a1][b1]) + PQ[b1]*PQ[d0]*(PQ[a1]*delta[b0][d1] + PQ[b0]*delta[a1][d1] + PQ[d1]*delta[a1][b0]))) + PQ[a1]*PQ[c0]*(PQ[b0]*PQ[b1]*(PQ[a0]*(PQ[c1]*delta[d0][d1] + PQ[d0]*delta[c1][d1] + PQ[d1]*delta[c1][d0]) + PQ[c1]*(PQ[d0]*delta[a0][d1] + PQ[d1]*delta[a0][d0])) + PQ[d0]*PQ[d1]*(PQ[a0]*(PQ[b0]*delta[b1][c1] + PQ[b1]*delta[b0][c1] + PQ[c1]*delta[b0][b1]) + PQ[c1]*(PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]))) + PQ[b0]*PQ[b1]*PQ[d0]*PQ[d1]*(PQ[a0]*(PQ[a1]*delta[c0][c1] + PQ[c0]*delta[a1][c1] + PQ[c1]*delta[a1][c0]) + PQ[a1]*(PQ[c0]*delta[a0][c1] + PQ[c1]*delta[a0][c0]) + PQ[c0]*PQ[c1]*delta[a0][a1])

                    )

                    + F8_t[7] * S1 * S1 * S1 * S1 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        +PQ[c0]*PQ[c1]*PQ[d0]*PQ[d1]*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))

                    )

                    + F8_t[7] * S1 * S1 * S1 * S2 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        -PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*(PQ[c0]*PQ[c1]*(PQ[d0]*QD_1 + PQ[d1]*QD_0) + PQ[d0]*PQ[d1]*(PQ[c0]*QC_1 + PQ[c1]*QC_0))

                    )

                    + F8_t[8] * S1 * S1 * S1 * S1 * S2 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (

                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*PQ[c0]*PQ[c1]*PQ[d0]*PQ[d1]

                    )


                );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * dd_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

