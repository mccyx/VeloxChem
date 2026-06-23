// DDSD altA DDSD0 BASE
__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDSD_altA0(double*         mat_J,
                       const double*   s_prim_info,
                       const uint32_t  s_prim_count,
                       const double*   d_prim_info,
                       const uint32_t  d_prim_count,
                       const double*   sd_mat_D,
                       const double*   dd_mat_Q_local,
                       const double*   sd_mat_Q,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const double*   dd_pair_data_local,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* sd_first_inds,
                       const uint32_t* sd_second_inds,
                       const double*   sd_pair_data,
                       const uint32_t  sd_prim_pair_count,
                       const double*   boys_func_table,
                       const double*   boys_func_ft,
                       const double    eri_threshold)
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

    for (uint32_t m = 0; m < (sd_prim_pair_count + TILE_DIM_LARGE - 1) / TILE_DIM_LARGE; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= sd_prim_pair_count) || (fabs(dd_mat_Q_local[ij] * sd_mat_Q[kl] * sd_mat_D[kl]) <= eri_threshold))
        {
            break;
        }

        const auto k = sd_first_inds[kl];
        const auto l = sd_second_inds[kl];

        const auto a_k = s_prim_info[k + s_prim_count * 0];

        const double r_k[3] = {s_prim_info[k + s_prim_count * 2],
                               s_prim_info[k + s_prim_count * 3],
                               s_prim_info[k + s_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = sd_pair_data[kl];

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

        double F6_t[2];

        gpu::computeBoysFunction(F6_t, S1 * S2 * inv_S4 * r2_PQ, 1, boys_func_table, boys_func_ft);

        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                + F6_t[0] * (

                    0.25 * inv_S1 * inv_S1 * (
                        +QD_0*QD_1*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * inv_S1 * inv_S2 * (
                        +PA_0*PB_0*delta[a1][b1]*delta[d0][d1]

                        +delta[d0][d1]*(PA_1*(PA_0*delta[b0][b1] + PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_1*(PA_0*delta[a1][b0] + PB_0*delta[a0][a1]))
                    )

                    + 0.5 * inv_S1 * (
                        +PA_0*PA_1*QD_0*QD_1*delta[b0][b1]

                        +QD_0*QD_1*(PB_0*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) + PB_1*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0]))
                    )

                    + 0.5 * inv_S2 * (
                        +PA_0*PA_1*PB_0*PB_1*delta[d0][d1]
                    )

                    + (
                        +PA_0*PA_1*PB_0*PB_1*QD_0*QD_1
                    )

                    + 0.125 * inv_S1 * inv_S1 * inv_S2 * (
                        +delta[d0][d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                )

                + F6_t[1] * (

                    (-0.25) * inv_S1 * inv_S1 * inv_S4 * (
                        +delta[d0][d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + (-0.125) * inv_S1 * inv_S2 * inv_S4 * (
                        +delta[d0][d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + (-0.5) * S2 * inv_S1 * inv_S1 * inv_S4 * (
                        +QD_0*QD_1*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * inv_S1 * inv_S4 * (
                        +delta[b0][b1]*delta[d0][d1]*(PA_0*(-PA_1 + PQ[a1]) + PA_1*PQ[a0])

                        +delta[d0][d1]*(PB_0*(-PA_0*delta[a1][b1] - PA_1*delta[a0][b1] - PB_1*delta[a0][a1] + PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + PQ[b0]*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) + delta[a0][b0]*(PA_1*(-PB_1 + PQ[b1]) + PB_1*PQ[a1]) + delta[a1][b0]*(PA_0*(-PB_1 + PQ[b1]) + PB_1*PQ[a0]))

                        +QD_0*(PA_0*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0])) + QD_1*(PA_0*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]))

                        -(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + (-0.25) * inv_S2 * inv_S4 * (
                        +PA_0*PA_1*delta[b0][b1]*delta[d0][d1]

                        +delta[d0][d1]*(PB_0*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) + PB_1*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0]))
                    )

                    + (-0.5) * S1 * inv_S2 * inv_S4 * (
                        +PA_0*PA_1*PB_0*PB_1*delta[d0][d1]
                    )

                    + 0.5 * S2 * inv_S1 * inv_S4 * (
                        +QD_0*QD_1*delta[b0][b1]*(PA_0*(-PA_1 + PQ[a1]) + PA_1*PQ[a0])

                        +QD_0*QD_1*(PB_0*(-PA_0*delta[a1][b1] - PA_1*delta[a0][b1] - PB_1*delta[a0][a1] + PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + PQ[b0]*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) + delta[a0][b0]*(PA_1*(-PB_1 + PQ[b1]) + PB_1*PQ[a1]) + delta[a1][b0]*(PA_0*(-PB_1 + PQ[b1]) + PB_1*PQ[a0]))
                    )

                    + 0.5 * inv_S4 * (
                        +delta[d0][d1]*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))

                        +PA_0*PA_1*(PB_0*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + PB_1*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0])) + PB_0*PB_1*(PA_0*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]) + PA_1*(QD_0*delta[a0][d1] + QD_1*delta[a0][d0]))

                        -(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1])
                    )

                    + S1 * inv_S4 * (
                        +PA_0*PA_1*PB_0*PB_1*(-PQ[d0]*QD_1 - PQ[d1]*QD_0)
                    )

                    + S2 * inv_S4 * (
                        +QD_0*QD_1*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))
                    )

                )

            );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        ERIs[threadIdx.y] += eri_ijkl * sd_mat_D[kl] * 2.0;
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

// DDSD altA DDSD0 FP64
__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDSD_altA0_FP64(double*         mat_J,
                       const double*   s_prim_info,
                       const uint32_t  s_prim_count,
                       const double*   d_prim_info,
                       const uint32_t  d_prim_count,
                       const double*   sd_mat_D,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const double*   dd_pair_data_local,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* sd_first_inds,
                       const uint32_t* sd_second_inds,
                       const double*   sd_pair_data,
                       const uint32_t  sd_prim_pair_count,
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

        if ((ij >= dd_prim_pair_count_local) || (kl >= sd_prim_pair_count))
        {
            break;
        }

        const auto k = sd_first_inds[kl];
        const auto l = sd_second_inds[kl];

        const auto a_k = s_prim_info[k + s_prim_count * 0];

        const double r_k[3] = {s_prim_info[k + s_prim_count * 2],
                               s_prim_info[k + s_prim_count * 3],
                               s_prim_info[k + s_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = sd_pair_data[kl];

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

        double F6_t[2];

        gpu::computeBoysFunction(F6_t, S1 * S2 * inv_S4 * r2_PQ, 1, boys_func_table, boys_func_ft);

        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                + F6_t[0] * (

                    0.25 * inv_S1 * inv_S1 * (
                        +QD_0*QD_1*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * inv_S1 * inv_S2 * (
                        +PA_0*PB_0*delta[a1][b1]*delta[d0][d1]

                        +delta[d0][d1]*(PA_1*(PA_0*delta[b0][b1] + PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_1*(PA_0*delta[a1][b0] + PB_0*delta[a0][a1]))
                    )

                    + 0.5 * inv_S1 * (
                        +PA_0*PA_1*QD_0*QD_1*delta[b0][b1]

                        +QD_0*QD_1*(PB_0*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) + PB_1*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0]))
                    )

                    + 0.5 * inv_S2 * (
                        +PA_0*PA_1*PB_0*PB_1*delta[d0][d1]
                    )

                    + (
                        +PA_0*PA_1*PB_0*PB_1*QD_0*QD_1
                    )

                    + 0.125 * inv_S1 * inv_S1 * inv_S2 * (
                        +delta[d0][d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                )

                + F6_t[1] * (

                    (-0.25) * inv_S1 * inv_S1 * inv_S4 * (
                        +delta[d0][d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + (-0.125) * inv_S1 * inv_S2 * inv_S4 * (
                        +delta[d0][d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + (-0.5) * S2 * inv_S1 * inv_S1 * inv_S4 * (
                        +QD_0*QD_1*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * inv_S1 * inv_S4 * (
                        +delta[b0][b1]*delta[d0][d1]*(PA_0*(-PA_1 + PQ[a1]) + PA_1*PQ[a0])

                        +delta[d0][d1]*(PB_0*(-PA_0*delta[a1][b1] - PA_1*delta[a0][b1] - PB_1*delta[a0][a1] + PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + PQ[b0]*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) + delta[a0][b0]*(PA_1*(-PB_1 + PQ[b1]) + PB_1*PQ[a1]) + delta[a1][b0]*(PA_0*(-PB_1 + PQ[b1]) + PB_1*PQ[a0]))

                        +QD_0*(PA_0*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0])) + QD_1*(PA_0*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]))

                        -(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + (-0.25) * inv_S2 * inv_S4 * (
                        +PA_0*PA_1*delta[b0][b1]*delta[d0][d1]

                        +delta[d0][d1]*(PB_0*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) + PB_1*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0]))
                    )

                    + (-0.5) * S1 * inv_S2 * inv_S4 * (
                        +PA_0*PA_1*PB_0*PB_1*delta[d0][d1]
                    )

                    + 0.5 * S2 * inv_S1 * inv_S4 * (
                        +QD_0*QD_1*delta[b0][b1]*(PA_0*(-PA_1 + PQ[a1]) + PA_1*PQ[a0])

                        +QD_0*QD_1*(PB_0*(-PA_0*delta[a1][b1] - PA_1*delta[a0][b1] - PB_1*delta[a0][a1] + PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + PQ[b0]*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) + delta[a0][b0]*(PA_1*(-PB_1 + PQ[b1]) + PB_1*PQ[a1]) + delta[a1][b0]*(PA_0*(-PB_1 + PQ[b1]) + PB_1*PQ[a0]))
                    )

                    + 0.5 * inv_S4 * (
                        +delta[d0][d1]*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))

                        +PA_0*PA_1*(PB_0*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + PB_1*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0])) + PB_0*PB_1*(PA_0*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]) + PA_1*(QD_0*delta[a0][d1] + QD_1*delta[a0][d0]))

                        -(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1])
                    )

                    + S1 * inv_S4 * (
                        +PA_0*PA_1*PB_0*PB_1*(-PQ[d0]*QD_1 - PQ[d1]*QD_0)
                    )

                    + S2 * inv_S4 * (
                        +QD_0*QD_1*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))
                    )

                )

            );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        ERIs[threadIdx.y] += eri_ijkl * sd_mat_D[kl] * 2.0;
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

// DDSD altA DDSD0 FP32
__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDSD_altA0_FP32(double*         mat_J,
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

        const float r_k_f[3] = {s_prim_info_f[k + s_prim_count * 2],
                               s_prim_info_f[k + s_prim_count * 3],
                               s_prim_info_f[k + s_prim_count * 4]};

        const auto a_l_f = d_prim_info_f[l / 6 + d_prim_count * 0];

        const float r_l_f[3] = {d_prim_info_f[l / 6 + d_prim_count * 2],
                               d_prim_info_f[l / 6 + d_prim_count * 3],
                               d_prim_info_f[l / 6 + d_prim_count * 4]};

        const auto S_kl_00_f = sd_pair_data_f[kl];

        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2_f = a_k_f + a_l_f;

        const auto inv_S2_f = 1.0f / S2_f;
        const auto inv_S4_f = 1.0f / (S1_f + S2_f);

        const float PQ_f[3] = {(a_k_f * r_k_f[0] + a_l_f * r_l_f[0]) * inv_S2_f - (a_i_f * r_i_f[0] + a_j_f * r_j_f[0]) * inv_S1_f,
                              (a_k_f * r_k_f[1] + a_l_f * r_l_f[1]) * inv_S2_f - (a_i_f * r_i_f[1] + a_j_f * r_j_f[1]) * inv_S1_f,
                              (a_k_f * r_k_f[2] + a_l_f * r_l_f[2]) * inv_S2_f - (a_i_f * r_i_f[2] + a_j_f * r_j_f[2]) * inv_S1_f};

        const auto r2_PQ_f = PQ_f[0] * PQ_f[0] + PQ_f[1] * PQ_f[1] + PQ_f[2] * PQ_f[2];

        const auto Lambda_f = sqrtf(4.0f * S1_f * S2_f * MATH_CONST_INV_PI_F * inv_S4_f);

        float F6_t_f[2];

        gpu::computeBoysFunction_f(F6_t_f, S1_f * S2_f * inv_S4_f * r2_PQ_f, 1, boys_func_table_f, boys_func_ft_f);

        const auto QD_0_f = (-a_k_f * inv_S2_f) * (r_l_f[d0] - r_k_f[d0]);
        const auto QD_1_f = (-a_k_f * inv_S2_f) * (r_l_f[d1] - r_k_f[d1]);

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
                        +delta_f[b0][b1]*delta_f[d0][d1]*(PA_0_f*(-PA_1_f + PQ_f[a1]) + PA_1_f*PQ_f[a0])

                        +delta_f[d0][d1]*(PB_0_f*(-PA_0_f*delta_f[a1][b1] - PA_1_f*delta_f[a0][b1] - PB_1_f*delta_f[a0][a1] + PQ_f[a0]*delta_f[a1][b1] + PQ_f[a1]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][a1]) + PQ_f[b0]*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) + delta_f[a0][b0]*(PA_1_f*(-PB_1_f + PQ_f[b1]) + PB_1_f*PQ_f[a1]) + delta_f[a1][b0]*(PA_0_f*(-PB_1_f + PQ_f[b1]) + PB_1_f*PQ_f[a0]))

                        +QD_0_f*(PA_0_f*(delta_f[a1][b0]*delta_f[b1][d1] + delta_f[a1][b1]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][b1]) + PA_1_f*(delta_f[a0][b0]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][b1]) + PB_0_f*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + PB_1_f*(delta_f[a0][a1]*delta_f[b0][d1] + delta_f[a0][b0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b0])) + QD_1_f*(PA_0_f*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + PA_1_f*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + PB_0_f*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + PB_1_f*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0]))

                        -(PQ_f[d0]*QD_1_f + PQ_f[d1]*QD_0_f)*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + (-0.25f) * inv_S2_f * inv_S4_f * (
                        +PA_0_f*PA_1_f*delta_f[b0][b1]*delta_f[d0][d1]

                        +delta_f[d0][d1]*(PB_0_f*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) + PB_1_f*(PA_0_f*delta_f[a1][b0] + PA_1_f*delta_f[a0][b0]))
                    )

                    + (-0.5f) * S1_f * inv_S2_f * inv_S4_f * (
                        +PA_0_f*PA_1_f*PB_0_f*PB_1_f*delta_f[d0][d1]
                    )

                    + 0.5f * S2_f * inv_S1_f * inv_S4_f * (
                        +QD_0_f*QD_1_f*delta_f[b0][b1]*(PA_0_f*(-PA_1_f + PQ_f[a1]) + PA_1_f*PQ_f[a0])

                        +QD_0_f*QD_1_f*(PB_0_f*(-PA_0_f*delta_f[a1][b1] - PA_1_f*delta_f[a0][b1] - PB_1_f*delta_f[a0][a1] + PQ_f[a0]*delta_f[a1][b1] + PQ_f[a1]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][a1]) + PQ_f[b0]*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) + delta_f[a0][b0]*(PA_1_f*(-PB_1_f + PQ_f[b1]) + PB_1_f*PQ_f[a1]) + delta_f[a1][b0]*(PA_0_f*(-PB_1_f + PQ_f[b1]) + PB_1_f*PQ_f[a0]))
                    )

                    + 0.5f * inv_S4_f * (
                        +delta_f[d0][d1]*(PA_0_f*PA_1_f*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PB_1_f*(PA_0_f*PQ_f[a1] + PA_1_f*PQ_f[a0]))

                        +PA_0_f*PA_1_f*(PB_0_f*(QD_0_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][d0]) + PB_1_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0])) + PB_0_f*PB_1_f*(PA_0_f*(QD_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][d0]) + PA_1_f*(QD_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][d0]))

                        -(PQ_f[d0]*QD_1_f + PQ_f[d1]*QD_0_f)*(PA_0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) + PA_1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PB_0_f*PB_1_f*delta_f[a0][a1])
                    )

                    + S1_f * inv_S4_f * (
                        +PA_0_f*PA_1_f*PB_0_f*PB_1_f*(-PQ_f[d0]*QD_1_f - PQ_f[d1]*QD_0_f)
                    )

                    + S2_f * inv_S4_f * (
                        +QD_0_f*QD_1_f*(PA_0_f*PA_1_f*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PB_1_f*(PA_0_f*PQ_f[a1] + PA_1_f*PQ_f[a0]))
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
        double J_ij = 0.0;

        for (uint32_t n = 0; n < TILE_DIM_LARGE; n++)
        {
            J_ij += ERIs[n];
        }

        mat_J[ij] += J_ij;
    }
}

// DDSD altA DDSD1 BASE
__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDSD_altA1(double*         mat_J,
                       const double*   s_prim_info,
                       const uint32_t  s_prim_count,
                       const double*   d_prim_info,
                       const uint32_t  d_prim_count,
                       const double*   sd_mat_D,
                       const double*   dd_mat_Q_local,
                       const double*   sd_mat_Q,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const double*   dd_pair_data_local,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* sd_first_inds,
                       const uint32_t* sd_second_inds,
                       const double*   sd_pair_data,
                       const uint32_t  sd_prim_pair_count,
                       const double*   boys_func_table,
                       const double*   boys_func_ft,
                       const double    eri_threshold)
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

    for (uint32_t m = 0; m < (sd_prim_pair_count + TILE_DIM_LARGE - 1) / TILE_DIM_LARGE; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= sd_prim_pair_count) || (fabs(dd_mat_Q_local[ij] * sd_mat_Q[kl] * sd_mat_D[kl]) <= eri_threshold))
        {
            break;
        }

        const auto k = sd_first_inds[kl];
        const auto l = sd_second_inds[kl];

        const auto a_k = s_prim_info[k + s_prim_count * 0];

        const double r_k[3] = {s_prim_info[k + s_prim_count * 2],
                               s_prim_info[k + s_prim_count * 3],
                               s_prim_info[k + s_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = sd_pair_data[kl];

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

        double F6_t[7];

        gpu::computeBoysFunction(F6_t, S1 * S2 * inv_S4 * r2_PQ, 6, boys_func_table, boys_func_ft);

        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                + F6_t[2] * (

                    0.25 * S2 * S2 * inv_S1 * inv_S1 * inv_S4 * inv_S4 * (
                        +QD_0*QD_1*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * S2 * inv_S1 * inv_S4 * inv_S4 * (
                        +delta[b0][b1]*delta[d0][d1]*(-PA_0*PQ[a1] + PQ[a0]*(-PA_1 + PQ[a1]))

                        +delta[d0][d1]*(PQ[a0]*delta[a1][b0]*(-PB_1 + PQ[b1]) + PQ[a0]*delta[a1][b1]*(-PB_0 + PQ[b0]) + PQ[a1]*delta[a0][b0]*(-PB_1 + PQ[b1]) + PQ[a1]*delta[a0][b1]*(-PB_0 + PQ[b0]) + PQ[b0]*PQ[b1]*delta[a0][a1] - PQ[b0]*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) - PQ[b1]*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0] + PB_0*delta[a0][a1]))

                        +(-PA_0*QD_0 + PQ[a0]*QD_0)*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + (-PA_0*QD_1 + PQ[a0]*QD_1)*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + (-PA_1*QD_0 + PQ[a1]*QD_0)*(delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1]) + (-PA_1*QD_1 + PQ[a1]*QD_1)*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + (-PB_0*QD_0 + PQ[b0]*QD_0)*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + (-PB_0*QD_1 + PQ[b0]*QD_1)*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + (-PB_1*QD_0 + PQ[b1]*QD_0)*(delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0]) + (-PB_1*QD_1 + PQ[b1]*QD_1)*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])

                        +2.0*(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * inv_S4 * inv_S4 * (
                        +delta[a1][b1]*delta[d0][d1]*(PA_0*(PB_0 - PQ[b0]) - PB_0*PQ[a0])

                        +PA_0*(PA_1*(delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + PB_0*(delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + PB_1*(delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0])) + PA_1*(PB_0*(delta[a0][d0]*delta[b1][d1] + delta[a0][d1]*delta[b1][d0]) + PB_1*(delta[a0][d0]*delta[b0][d1] + delta[a0][d1]*delta[b0][d0])) + PB_0*PB_1*(delta[a0][d0]*delta[a1][d1] + delta[a0][d1]*delta[a1][d0])

                        +delta[d0][d1]*(PA_1*(PB_1*delta[a0][b0] + delta[a0][b1]*(PB_0 - PQ[b0]) + delta[b0][b1]*(PA_0 - PQ[a0])) + PB_1*delta[a0][a1]*(PB_0 - PQ[b0]) + PB_1*delta[a1][b0]*(PA_0 - PQ[a0]) - PQ[a1]*(PA_0*delta[b0][b1] + PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) - PQ[b1]*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0] + PB_0*delta[a0][a1]))

                        -PQ[d0]*(PA_0*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0])) - PQ[d1]*(PA_0*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]))

                        +PQ[d0]*PQ[d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.5 * S1 * inv_S4 * inv_S4 * (
                        -delta[d0][d1]*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))

                        -PA_0*PA_1*(PB_0*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PB_1*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0])) - PB_0*PB_1*(PA_0*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0]) + PA_1*(PQ[d0]*delta[a0][d1] + PQ[d1]*delta[a0][d0]))

                        +PQ[d0]*PQ[d1]*(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1])
                    )

                    + 0.5 * S2 * S2 * inv_S1 * inv_S4 * inv_S4 * (
                        +QD_0*QD_1*delta[b0][b1]*(-PA_0*PQ[a1] + PQ[a0]*(-PA_1 + PQ[a1]))

                        +QD_0*QD_1*(PQ[a0]*delta[a1][b0]*(-PB_1 + PQ[b1]) + PQ[a0]*delta[a1][b1]*(-PB_0 + PQ[b0]) + PQ[a1]*delta[a0][b0]*(-PB_1 + PQ[b1]) + PQ[a1]*delta[a0][b1]*(-PB_0 + PQ[b0]) + PQ[b0]*PQ[b1]*delta[a0][a1] - PQ[b0]*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) - PQ[b1]*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0] + PB_0*delta[a0][a1]))
                    )

                    + 0.5 * S2 * inv_S4 * inv_S4 * (
                        +QD_0*delta[b1][d1]*(PA_0*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PA_1*PB_0*PQ[a0])

                        +PB_1*(PA_0*(PQ[a1]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + PQ[b0]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0])) + PQ[a0]*(PA_1*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + PB_0*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]))) + QD_0*(PA_0*PQ[b1]*(PA_1*delta[b0][d1] + PB_0*delta[a1][d1]) + delta[a0][d1]*(PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*PQ[a1])) + QD_1*(PA_0*(PA_1*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0]) + PB_0*(PQ[a1]*delta[b1][d0] + PQ[b1]*delta[a1][d0])) + PA_1*PB_0*(PQ[a0]*delta[b1][d0] + PQ[b1]*delta[a0][d0]) + PB_1*delta[a0][d0]*(PA_1*PQ[b0] + PB_0*PQ[a1]))

                        +(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) - PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) - PA_1*(PQ[a0]*delta[b0][b1] + PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1] - PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) - PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]))

                        +delta[d0][d1]*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S1 * S1 * inv_S4 * inv_S4 * (
                        +PA_0*PA_1*PB_0*PB_1*PQ[d0]*PQ[d1]
                    )

                    + S1 * S2 * inv_S4 * inv_S4 * (
                        -(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))
                    )

                    + S2 * S2 * inv_S4 * inv_S4 * (
                        +QD_0*QD_1*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + 0.125 * S2 * inv_S1 * inv_S1 * inv_S4 * inv_S4 * (
                        +delta[d0][d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.125 * inv_S1 * inv_S4 * inv_S4 * (
                        +2.0*delta[d0][d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])

                        +delta[b0][b1]*(delta[a0][d0]*delta[a1][d1] + delta[a0][d1]*delta[a1][d0]) + delta[b0][d0]*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + delta[b0][d1]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + delta[b1][d0]*(delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0]) + delta[b1][d1]*(delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])
                    )

                )

                + F6_t[3] * (

                    (-0.125) * S2 * inv_S1 * inv_S4 * inv_S4 * inv_S4 * (
                        +delta[a0][a1]*(delta[b0][b1]*delta[d0][d1] + delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + delta[a0][b0]*(delta[a1][b1]*delta[d0][d1] + delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + delta[a0][b1]*(delta[a1][b0]*delta[d0][d1] + delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + delta[a0][d0]*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + delta[a0][d1]*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1])
                    )

                    + (-0.25) * S2 * S2 * inv_S1 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*delta[b0][b1]*delta[d0][d1]

                        +delta[d0][d1]*(PQ[b0]*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + PQ[b1]*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0]))

                        +QD_0*(PQ[a0]*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1]) + PQ[b0]*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + PQ[b1]*(delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0])) + QD_1*(PQ[a0]*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + PQ[b0]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + PQ[b1]*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]))

                        +(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        +delta[a1][b1]*delta[d0][d1]*(PB_0*PQ[a0] + PQ[b0]*(PA_0 - PQ[a0]))

                        +(PA_0*PQ[a1] + PA_1*PQ[a0])*(delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + (PA_0*PQ[b0] + PB_0*PQ[a0])*(delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + (PA_0*PQ[b1] + PB_1*PQ[a0])*(delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + (PA_1*PQ[b0] + PB_0*PQ[a1])*(delta[a0][d0]*delta[b1][d1] + delta[a0][d1]*delta[b1][d0]) + (PA_1*PQ[b1] + PB_1*PQ[a1])*(delta[a0][d0]*delta[b0][d1] + delta[a0][d1]*delta[b0][d0]) + (PB_0*PQ[b1] + PB_1*PQ[b0])*(delta[a0][d0]*delta[a1][d1] + delta[a0][d1]*delta[a1][d0])

                        +delta[d0][d1]*(PA_1*(PQ[a0]*delta[b0][b1] + PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PB_1*(PQ[a0]*delta[a1][b0] + PQ[b0]*delta[a0][a1]) + PQ[a1]*(PA_0*delta[b0][b1] + PB_0*delta[a0][b1] + PB_1*delta[a0][b0] - PQ[a0]*delta[b0][b1] - PQ[b0]*delta[a0][b1] - PQ[b1]*delta[a0][b0]) + PQ[b1]*(delta[a0][a1]*(PB_0 - PQ[b0]) + delta[a1][b0]*(PA_0 - PQ[a0])))

                        +(PA_0*PQ[d0] - PQ[a0]*PQ[d0])*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + (PA_0*PQ[d1] - PQ[a0]*PQ[d1])*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + (PA_1*PQ[d0] - PQ[a1]*PQ[d0])*(delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1]) + (PA_1*PQ[d1] - PQ[a1]*PQ[d1])*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + (PB_0*PQ[d0] - PQ[b0]*PQ[d0])*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + (PB_0*PQ[d1] - PQ[b0]*PQ[d1])*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + (PB_1*PQ[d0] - PQ[b1]*PQ[d0])*(delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0]) + (PB_1*PQ[d1] - PQ[b1]*PQ[d1])*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])

                        -2*PQ[d0]*PQ[d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.5 * S1 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        -delta[d0][d1]*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))

                        -PA_0*(PA_1*(PQ[b0]*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PQ[b1]*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0])) + PB_0*(PQ[a1]*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PQ[b1]*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0])) + PB_1*(PQ[a1]*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0]) + PQ[b0]*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0]))) - PA_1*(PB_0*(PQ[a0]*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PQ[b1]*(PQ[d0]*delta[a0][d1] + PQ[d1]*delta[a0][d0])) + PB_1*(PQ[a0]*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0]) + PQ[b0]*(PQ[d0]*delta[a0][d1] + PQ[d1]*delta[a0][d0]))) - PB_0*PB_1*(PQ[a0]*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0]) + PQ[a1]*(PQ[d0]*delta[a0][d1] + PQ[d1]*delta[a0][d0]))

                        +PQ[d0]*PQ[d1]*(PA_0*(-PA_1*delta[b0][b1] - PB_0*delta[a1][b1] - PB_1*delta[a1][b0] + PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PA_1*(PQ[a0]*delta[b0][b1] + delta[a0][b0]*(-PB_1 + PQ[b1]) + delta[a0][b1]*(-PB_0 + PQ[b0])) + PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + delta[a0][a1]*(-PB_1 + PQ[b1])) + PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]))
                    )

                    + (-0.5) * S2 * S2 * S2 * inv_S1 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*QD_0*QD_1*delta[b0][b1]

                        +QD_0*QD_1*(PQ[b0]*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + PQ[b1]*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0]))
                    )

                    + 0.5 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        +QD_0*delta[b1][d1]*(PA_1*PQ[a0]*PQ[b0] + PQ[a1]*(PA_0*PQ[b0] + PB_0*PQ[a0]))

                        +PB_1*PQ[a0]*(PQ[a1]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + PQ[b0]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0])) + QD_0*(PB_1*PQ[a1]*PQ[b0]*delta[a0][d1] + PQ[b1]*(PA_0*(PQ[a1]*delta[b0][d1] + PQ[b0]*delta[a1][d1]) + PA_1*(PQ[a0]*delta[b0][d1] + PQ[b0]*delta[a0][d1]) + PB_0*(PQ[a0]*delta[a1][d1] + PQ[a1]*delta[a0][d1]))) + QD_1*(PQ[a0]*(PA_1*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0]) + PB_0*(PQ[a1]*delta[b1][d0] + PQ[b1]*delta[a1][d0])) + PQ[a1]*(PA_0*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0]) + delta[a0][d0]*(PB_0*PQ[b1] + PB_1*PQ[b0])) + PQ[b0]*PQ[b1]*(PA_0*delta[a1][d0] + PA_1*delta[a0][d0]))

                        +(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PQ[a0]*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0] - PQ[a1]*delta[b0][b1] - PQ[b0]*delta[a1][b1] - PQ[b1]*delta[a1][b0]) + PQ[a1]*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PQ[b0]*(PB_1*delta[a0][a1] + delta[a0][b1]*(PA_1 - PQ[a1])) + PQ[b1]*(delta[a0][a1]*(PB_0 - PQ[b0]) + delta[a0][b0]*(PA_1 - PQ[a1])))

                        +delta[d0][d1]*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S1 * S1 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[d0]*PQ[d1]*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))
                    )

                    + S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        -(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        +QD_0*QD_1*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                )

                + F6_t[4] * (

                    0.5 * S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        -delta[d0][d1]*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))

                        -PQ[a0]*(PA_1*(PQ[b0]*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PQ[b1]*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0])) + PB_0*(PQ[a1]*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PQ[b1]*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0])) + PB_1*(PQ[a1]*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0]) + PQ[b0]*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0]))) - PQ[a1]*(PA_0*(PQ[b0]*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PQ[b1]*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0])) + (PB_0*PQ[b1] + PB_1*PQ[b0])*(PQ[d0]*delta[a0][d1] + PQ[d1]*delta[a0][d0])) - PQ[b0]*PQ[b1]*(PA_0*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0]) + PA_1*(PQ[d0]*delta[a0][d1] + PQ[d1]*delta[a0][d0]))

                        +PQ[d0]*PQ[d1]*(-PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) - PQ[a0]*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PQ[a0]*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) - PQ[a1]*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PQ[a1]*(PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PQ[b0]*PQ[b1]*delta[a0][a1] - PQ[b0]*(PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) - PQ[b1]*(PA_1*delta[a0][b0] + PB_0*delta[a0][a1]))
                    )

                    + 0.5 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*QD_0*delta[b1][d1]

                        +PQ[a0]*PQ[a1]*QD_1*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0]) + PQ[b1]*(PQ[a0]*QD_0*(PQ[a1]*delta[b0][d1] + PQ[b0]*delta[a1][d1]) + PQ[b0]*(PQ[a1]*QD_0*delta[a0][d1] + QD_1*(PQ[a0]*delta[a1][d0] + PQ[a1]*delta[a0][d0])))

                        +(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(PQ[a0]*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PQ[a1]*(PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PQ[b0]*PQ[b1]*delta[a0][a1])

                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*delta[d0][d1]
                    )

                    + S1 * S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[d0]*PQ[d1]*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S1 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        -(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S2 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*QD_0*QD_1
                    )

                    + 0.25 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*(delta[b0][b1]*delta[d0][d1] + delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0])

                        +PQ[b0]*(PQ[a0]*(delta[a1][b1]*delta[d0][d1] + delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + PQ[a1]*(delta[a0][b1]*delta[d0][d1] + delta[a0][d0]*delta[b1][d1] + delta[a0][d1]*delta[b1][d0]) + PQ[b1]*(delta[a0][a1]*delta[d0][d1] + delta[a0][d0]*delta[a1][d1] + delta[a0][d1]*delta[a1][d0]) + PQ[d0]*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + PQ[d1]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1])) + PQ[b1]*(PQ[a0]*(delta[a1][b0]*delta[d0][d1] + delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + PQ[a1]*(delta[a0][b0]*delta[d0][d1] + delta[a0][d0]*delta[b0][d1] + delta[a0][d1]*delta[b0][d0]) + PQ[d0]*(delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0]) + PQ[d1]*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])) + PQ[d0]*(PQ[a0]*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1]) + PQ[d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])) + PQ[d1]*(PQ[a0]*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]))
                    )

                )

                + F6_t[5] * (

                    (-0.5) * S1 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*delta[d0][d1]

                        +PQ[d0]*(PQ[a0]*(PQ[a1]*(PQ[b0]*delta[b1][d1] + PQ[b1]*delta[b0][d1] + PQ[d1]*delta[b0][b1]) + PQ[b1]*PQ[d1]*delta[a1][b0]) + PQ[b0]*PQ[b1]*(PQ[a0]*delta[a1][d1] + PQ[a1]*delta[a0][d1] + PQ[d1]*delta[a0][a1])) + PQ[d1]*(PQ[a1]*PQ[b1]*(PQ[a0]*delta[b0][d0] + PQ[b0]*delta[a0][d0] + PQ[d0]*delta[a0][b0]) + PQ[b0]*(PQ[a0]*(PQ[a1]*delta[b1][d0] + PQ[b1]*delta[a1][d0] + PQ[d0]*delta[a1][b1]) + PQ[a1]*PQ[d0]*delta[a0][b1]))
                    )

                    + S1 * S1 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[d0]*PQ[d1]*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S1 * S2 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*(-PQ[d0]*QD_1 - PQ[d1]*QD_0)
                    )

                )

                + F6_t[6] * (

                    S1 * S1 * S2 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*PQ[d0]*PQ[d1]
                    )

                )

            );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        ERIs[threadIdx.y] += eri_ijkl * sd_mat_D[kl] * 2.0;
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

// DDSD altA DDSD1 FP64
__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDSD_altA1_FP64(double*         mat_J,
                       const double*   s_prim_info,
                       const uint32_t  s_prim_count,
                       const double*   d_prim_info,
                       const uint32_t  d_prim_count,
                       const double*   sd_mat_D,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const double*   dd_pair_data_local,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* sd_first_inds,
                       const uint32_t* sd_second_inds,
                       const double*   sd_pair_data,
                       const uint32_t  sd_prim_pair_count,
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

        if ((ij >= dd_prim_pair_count_local) || (kl >= sd_prim_pair_count))
        {
            break;
        }

        const auto k = sd_first_inds[kl];
        const auto l = sd_second_inds[kl];

        const auto a_k = s_prim_info[k + s_prim_count * 0];

        const double r_k[3] = {s_prim_info[k + s_prim_count * 2],
                               s_prim_info[k + s_prim_count * 3],
                               s_prim_info[k + s_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = sd_pair_data[kl];

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

        double F6_t[7];

        gpu::computeBoysFunction(F6_t, S1 * S2 * inv_S4 * r2_PQ, 6, boys_func_table, boys_func_ft);

        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                + F6_t[2] * (

                    0.25 * S2 * S2 * inv_S1 * inv_S1 * inv_S4 * inv_S4 * (
                        +QD_0*QD_1*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * S2 * inv_S1 * inv_S4 * inv_S4 * (
                        +delta[b0][b1]*delta[d0][d1]*(-PA_0*PQ[a1] + PQ[a0]*(-PA_1 + PQ[a1]))

                        +delta[d0][d1]*(PQ[a0]*delta[a1][b0]*(-PB_1 + PQ[b1]) + PQ[a0]*delta[a1][b1]*(-PB_0 + PQ[b0]) + PQ[a1]*delta[a0][b0]*(-PB_1 + PQ[b1]) + PQ[a1]*delta[a0][b1]*(-PB_0 + PQ[b0]) + PQ[b0]*PQ[b1]*delta[a0][a1] - PQ[b0]*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) - PQ[b1]*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0] + PB_0*delta[a0][a1]))

                        +(-PA_0*QD_0 + PQ[a0]*QD_0)*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + (-PA_0*QD_1 + PQ[a0]*QD_1)*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + (-PA_1*QD_0 + PQ[a1]*QD_0)*(delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1]) + (-PA_1*QD_1 + PQ[a1]*QD_1)*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + (-PB_0*QD_0 + PQ[b0]*QD_0)*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + (-PB_0*QD_1 + PQ[b0]*QD_1)*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + (-PB_1*QD_0 + PQ[b1]*QD_0)*(delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0]) + (-PB_1*QD_1 + PQ[b1]*QD_1)*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])

                        +2.0*(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * inv_S4 * inv_S4 * (
                        +delta[a1][b1]*delta[d0][d1]*(PA_0*(PB_0 - PQ[b0]) - PB_0*PQ[a0])

                        +PA_0*(PA_1*(delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + PB_0*(delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + PB_1*(delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0])) + PA_1*(PB_0*(delta[a0][d0]*delta[b1][d1] + delta[a0][d1]*delta[b1][d0]) + PB_1*(delta[a0][d0]*delta[b0][d1] + delta[a0][d1]*delta[b0][d0])) + PB_0*PB_1*(delta[a0][d0]*delta[a1][d1] + delta[a0][d1]*delta[a1][d0])

                        +delta[d0][d1]*(PA_1*(PB_1*delta[a0][b0] + delta[a0][b1]*(PB_0 - PQ[b0]) + delta[b0][b1]*(PA_0 - PQ[a0])) + PB_1*delta[a0][a1]*(PB_0 - PQ[b0]) + PB_1*delta[a1][b0]*(PA_0 - PQ[a0]) - PQ[a1]*(PA_0*delta[b0][b1] + PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) - PQ[b1]*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0] + PB_0*delta[a0][a1]))

                        -PQ[d0]*(PA_0*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0])) - PQ[d1]*(PA_0*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]))

                        +PQ[d0]*PQ[d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.5 * S1 * inv_S4 * inv_S4 * (
                        -delta[d0][d1]*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))

                        -PA_0*PA_1*(PB_0*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PB_1*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0])) - PB_0*PB_1*(PA_0*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0]) + PA_1*(PQ[d0]*delta[a0][d1] + PQ[d1]*delta[a0][d0]))

                        +PQ[d0]*PQ[d1]*(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1])
                    )

                    + 0.5 * S2 * S2 * inv_S1 * inv_S4 * inv_S4 * (
                        +QD_0*QD_1*delta[b0][b1]*(-PA_0*PQ[a1] + PQ[a0]*(-PA_1 + PQ[a1]))

                        +QD_0*QD_1*(PQ[a0]*delta[a1][b0]*(-PB_1 + PQ[b1]) + PQ[a0]*delta[a1][b1]*(-PB_0 + PQ[b0]) + PQ[a1]*delta[a0][b0]*(-PB_1 + PQ[b1]) + PQ[a1]*delta[a0][b1]*(-PB_0 + PQ[b0]) + PQ[b0]*PQ[b1]*delta[a0][a1] - PQ[b0]*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) - PQ[b1]*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0] + PB_0*delta[a0][a1]))
                    )

                    + 0.5 * S2 * inv_S4 * inv_S4 * (
                        +QD_0*delta[b1][d1]*(PA_0*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PA_1*PB_0*PQ[a0])

                        +PB_1*(PA_0*(PQ[a1]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + PQ[b0]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0])) + PQ[a0]*(PA_1*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + PB_0*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]))) + QD_0*(PA_0*PQ[b1]*(PA_1*delta[b0][d1] + PB_0*delta[a1][d1]) + delta[a0][d1]*(PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*PQ[a1])) + QD_1*(PA_0*(PA_1*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0]) + PB_0*(PQ[a1]*delta[b1][d0] + PQ[b1]*delta[a1][d0])) + PA_1*PB_0*(PQ[a0]*delta[b1][d0] + PQ[b1]*delta[a0][d0]) + PB_1*delta[a0][d0]*(PA_1*PQ[b0] + PB_0*PQ[a1]))

                        +(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) - PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) - PA_1*(PQ[a0]*delta[b0][b1] + PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1] - PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) - PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]))

                        +delta[d0][d1]*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S1 * S1 * inv_S4 * inv_S4 * (
                        +PA_0*PA_1*PB_0*PB_1*PQ[d0]*PQ[d1]
                    )

                    + S1 * S2 * inv_S4 * inv_S4 * (
                        -(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))
                    )

                    + S2 * S2 * inv_S4 * inv_S4 * (
                        +QD_0*QD_1*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + 0.125 * S2 * inv_S1 * inv_S1 * inv_S4 * inv_S4 * (
                        +delta[d0][d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.125 * inv_S1 * inv_S4 * inv_S4 * (
                        +2.0*delta[d0][d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])

                        +delta[b0][b1]*(delta[a0][d0]*delta[a1][d1] + delta[a0][d1]*delta[a1][d0]) + delta[b0][d0]*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + delta[b0][d1]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + delta[b1][d0]*(delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0]) + delta[b1][d1]*(delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])
                    )

                )

                + F6_t[3] * (

                    (-0.125) * S2 * inv_S1 * inv_S4 * inv_S4 * inv_S4 * (
                        +delta[a0][a1]*(delta[b0][b1]*delta[d0][d1] + delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + delta[a0][b0]*(delta[a1][b1]*delta[d0][d1] + delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + delta[a0][b1]*(delta[a1][b0]*delta[d0][d1] + delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + delta[a0][d0]*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + delta[a0][d1]*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1])
                    )

                    + (-0.25) * S2 * S2 * inv_S1 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*delta[b0][b1]*delta[d0][d1]

                        +delta[d0][d1]*(PQ[b0]*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + PQ[b1]*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0]))

                        +QD_0*(PQ[a0]*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1]) + PQ[b0]*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + PQ[b1]*(delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0])) + QD_1*(PQ[a0]*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + PQ[b0]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + PQ[b1]*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]))

                        +(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        +delta[a1][b1]*delta[d0][d1]*(PB_0*PQ[a0] + PQ[b0]*(PA_0 - PQ[a0]))

                        +(PA_0*PQ[a1] + PA_1*PQ[a0])*(delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + (PA_0*PQ[b0] + PB_0*PQ[a0])*(delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + (PA_0*PQ[b1] + PB_1*PQ[a0])*(delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + (PA_1*PQ[b0] + PB_0*PQ[a1])*(delta[a0][d0]*delta[b1][d1] + delta[a0][d1]*delta[b1][d0]) + (PA_1*PQ[b1] + PB_1*PQ[a1])*(delta[a0][d0]*delta[b0][d1] + delta[a0][d1]*delta[b0][d0]) + (PB_0*PQ[b1] + PB_1*PQ[b0])*(delta[a0][d0]*delta[a1][d1] + delta[a0][d1]*delta[a1][d0])

                        +delta[d0][d1]*(PA_1*(PQ[a0]*delta[b0][b1] + PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PB_1*(PQ[a0]*delta[a1][b0] + PQ[b0]*delta[a0][a1]) + PQ[a1]*(PA_0*delta[b0][b1] + PB_0*delta[a0][b1] + PB_1*delta[a0][b0] - PQ[a0]*delta[b0][b1] - PQ[b0]*delta[a0][b1] - PQ[b1]*delta[a0][b0]) + PQ[b1]*(delta[a0][a1]*(PB_0 - PQ[b0]) + delta[a1][b0]*(PA_0 - PQ[a0])))

                        +(PA_0*PQ[d0] - PQ[a0]*PQ[d0])*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + (PA_0*PQ[d1] - PQ[a0]*PQ[d1])*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + (PA_1*PQ[d0] - PQ[a1]*PQ[d0])*(delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1]) + (PA_1*PQ[d1] - PQ[a1]*PQ[d1])*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + (PB_0*PQ[d0] - PQ[b0]*PQ[d0])*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + (PB_0*PQ[d1] - PQ[b0]*PQ[d1])*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + (PB_1*PQ[d0] - PQ[b1]*PQ[d0])*(delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0]) + (PB_1*PQ[d1] - PQ[b1]*PQ[d1])*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])

                        -2*PQ[d0]*PQ[d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.5 * S1 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        -delta[d0][d1]*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))

                        -PA_0*(PA_1*(PQ[b0]*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PQ[b1]*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0])) + PB_0*(PQ[a1]*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PQ[b1]*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0])) + PB_1*(PQ[a1]*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0]) + PQ[b0]*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0]))) - PA_1*(PB_0*(PQ[a0]*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PQ[b1]*(PQ[d0]*delta[a0][d1] + PQ[d1]*delta[a0][d0])) + PB_1*(PQ[a0]*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0]) + PQ[b0]*(PQ[d0]*delta[a0][d1] + PQ[d1]*delta[a0][d0]))) - PB_0*PB_1*(PQ[a0]*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0]) + PQ[a1]*(PQ[d0]*delta[a0][d1] + PQ[d1]*delta[a0][d0]))

                        +PQ[d0]*PQ[d1]*(PA_0*(-PA_1*delta[b0][b1] - PB_0*delta[a1][b1] - PB_1*delta[a1][b0] + PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PA_1*(PQ[a0]*delta[b0][b1] + delta[a0][b0]*(-PB_1 + PQ[b1]) + delta[a0][b1]*(-PB_0 + PQ[b0])) + PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + delta[a0][a1]*(-PB_1 + PQ[b1])) + PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]))
                    )

                    + (-0.5) * S2 * S2 * S2 * inv_S1 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*QD_0*QD_1*delta[b0][b1]

                        +QD_0*QD_1*(PQ[b0]*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + PQ[b1]*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0]))
                    )

                    + 0.5 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        +QD_0*delta[b1][d1]*(PA_1*PQ[a0]*PQ[b0] + PQ[a1]*(PA_0*PQ[b0] + PB_0*PQ[a0]))

                        +PB_1*PQ[a0]*(PQ[a1]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + PQ[b0]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0])) + QD_0*(PB_1*PQ[a1]*PQ[b0]*delta[a0][d1] + PQ[b1]*(PA_0*(PQ[a1]*delta[b0][d1] + PQ[b0]*delta[a1][d1]) + PA_1*(PQ[a0]*delta[b0][d1] + PQ[b0]*delta[a0][d1]) + PB_0*(PQ[a0]*delta[a1][d1] + PQ[a1]*delta[a0][d1]))) + QD_1*(PQ[a0]*(PA_1*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0]) + PB_0*(PQ[a1]*delta[b1][d0] + PQ[b1]*delta[a1][d0])) + PQ[a1]*(PA_0*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0]) + delta[a0][d0]*(PB_0*PQ[b1] + PB_1*PQ[b0])) + PQ[b0]*PQ[b1]*(PA_0*delta[a1][d0] + PA_1*delta[a0][d0]))

                        +(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PQ[a0]*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0] - PQ[a1]*delta[b0][b1] - PQ[b0]*delta[a1][b1] - PQ[b1]*delta[a1][b0]) + PQ[a1]*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PQ[b0]*(PB_1*delta[a0][a1] + delta[a0][b1]*(PA_1 - PQ[a1])) + PQ[b1]*(delta[a0][a1]*(PB_0 - PQ[b0]) + delta[a0][b0]*(PA_1 - PQ[a1])))

                        +delta[d0][d1]*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S1 * S1 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[d0]*PQ[d1]*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))
                    )

                    + S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        -(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        +QD_0*QD_1*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                )

                + F6_t[4] * (

                    0.5 * S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        -delta[d0][d1]*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))

                        -PQ[a0]*(PA_1*(PQ[b0]*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PQ[b1]*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0])) + PB_0*(PQ[a1]*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PQ[b1]*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0])) + PB_1*(PQ[a1]*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0]) + PQ[b0]*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0]))) - PQ[a1]*(PA_0*(PQ[b0]*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PQ[b1]*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0])) + (PB_0*PQ[b1] + PB_1*PQ[b0])*(PQ[d0]*delta[a0][d1] + PQ[d1]*delta[a0][d0])) - PQ[b0]*PQ[b1]*(PA_0*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0]) + PA_1*(PQ[d0]*delta[a0][d1] + PQ[d1]*delta[a0][d0]))

                        +PQ[d0]*PQ[d1]*(-PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) - PQ[a0]*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PQ[a0]*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) - PQ[a1]*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PQ[a1]*(PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PQ[b0]*PQ[b1]*delta[a0][a1] - PQ[b0]*(PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) - PQ[b1]*(PA_1*delta[a0][b0] + PB_0*delta[a0][a1]))
                    )

                    + 0.5 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*QD_0*delta[b1][d1]

                        +PQ[a0]*PQ[a1]*QD_1*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0]) + PQ[b1]*(PQ[a0]*QD_0*(PQ[a1]*delta[b0][d1] + PQ[b0]*delta[a1][d1]) + PQ[b0]*(PQ[a1]*QD_0*delta[a0][d1] + QD_1*(PQ[a0]*delta[a1][d0] + PQ[a1]*delta[a0][d0])))

                        +(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(PQ[a0]*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PQ[a1]*(PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PQ[b0]*PQ[b1]*delta[a0][a1])

                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*delta[d0][d1]
                    )

                    + S1 * S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[d0]*PQ[d1]*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S1 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        -(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S2 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*QD_0*QD_1
                    )

                    + 0.25 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*(delta[b0][b1]*delta[d0][d1] + delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0])

                        +PQ[b0]*(PQ[a0]*(delta[a1][b1]*delta[d0][d1] + delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + PQ[a1]*(delta[a0][b1]*delta[d0][d1] + delta[a0][d0]*delta[b1][d1] + delta[a0][d1]*delta[b1][d0]) + PQ[b1]*(delta[a0][a1]*delta[d0][d1] + delta[a0][d0]*delta[a1][d1] + delta[a0][d1]*delta[a1][d0]) + PQ[d0]*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + PQ[d1]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1])) + PQ[b1]*(PQ[a0]*(delta[a1][b0]*delta[d0][d1] + delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + PQ[a1]*(delta[a0][b0]*delta[d0][d1] + delta[a0][d0]*delta[b0][d1] + delta[a0][d1]*delta[b0][d0]) + PQ[d0]*(delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0]) + PQ[d1]*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])) + PQ[d0]*(PQ[a0]*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1]) + PQ[d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])) + PQ[d1]*(PQ[a0]*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]))
                    )

                )

                + F6_t[5] * (

                    (-0.5) * S1 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*delta[d0][d1]

                        +PQ[d0]*(PQ[a0]*(PQ[a1]*(PQ[b0]*delta[b1][d1] + PQ[b1]*delta[b0][d1] + PQ[d1]*delta[b0][b1]) + PQ[b1]*PQ[d1]*delta[a1][b0]) + PQ[b0]*PQ[b1]*(PQ[a0]*delta[a1][d1] + PQ[a1]*delta[a0][d1] + PQ[d1]*delta[a0][a1])) + PQ[d1]*(PQ[a1]*PQ[b1]*(PQ[a0]*delta[b0][d0] + PQ[b0]*delta[a0][d0] + PQ[d0]*delta[a0][b0]) + PQ[b0]*(PQ[a0]*(PQ[a1]*delta[b1][d0] + PQ[b1]*delta[a1][d0] + PQ[d0]*delta[a1][b1]) + PQ[a1]*PQ[d0]*delta[a0][b1]))
                    )

                    + S1 * S1 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[d0]*PQ[d1]*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S1 * S2 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*(-PQ[d0]*QD_1 - PQ[d1]*QD_0)
                    )

                )

                + F6_t[6] * (

                    S1 * S1 * S2 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*PQ[d0]*PQ[d1]
                    )

                )

            );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        ERIs[threadIdx.y] += eri_ijkl * sd_mat_D[kl] * 2.0;
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

// DDSD altA DDSD1 FP32
__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDSD_altA1_FP32(double*         mat_J,
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

        const float r_k_f[3] = {s_prim_info_f[k + s_prim_count * 2],
                               s_prim_info_f[k + s_prim_count * 3],
                               s_prim_info_f[k + s_prim_count * 4]};

        const auto a_l_f = d_prim_info_f[l / 6 + d_prim_count * 0];

        const float r_l_f[3] = {d_prim_info_f[l / 6 + d_prim_count * 2],
                               d_prim_info_f[l / 6 + d_prim_count * 3],
                               d_prim_info_f[l / 6 + d_prim_count * 4]};

        const auto S_kl_00_f = sd_pair_data_f[kl];

        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2_f = a_k_f + a_l_f;

        const auto inv_S2_f = 1.0f / S2_f;
        const auto inv_S4_f = 1.0f / (S1_f + S2_f);

        const float PQ_f[3] = {(a_k_f * r_k_f[0] + a_l_f * r_l_f[0]) * inv_S2_f - (a_i_f * r_i_f[0] + a_j_f * r_j_f[0]) * inv_S1_f,
                              (a_k_f * r_k_f[1] + a_l_f * r_l_f[1]) * inv_S2_f - (a_i_f * r_i_f[1] + a_j_f * r_j_f[1]) * inv_S1_f,
                              (a_k_f * r_k_f[2] + a_l_f * r_l_f[2]) * inv_S2_f - (a_i_f * r_i_f[2] + a_j_f * r_j_f[2]) * inv_S1_f};

        const auto r2_PQ_f = PQ_f[0] * PQ_f[0] + PQ_f[1] * PQ_f[1] + PQ_f[2] * PQ_f[2];

        const auto Lambda_f = sqrtf(4.0f * S1_f * S2_f * MATH_CONST_INV_PI_F * inv_S4_f);

        float F6_t_f[7];

        gpu::computeBoysFunction_f(F6_t_f, S1_f * S2_f * inv_S4_f * r2_PQ_f, 6, boys_func_table_f, boys_func_ft_f);

        const auto QD_0_f = (-a_k_f * inv_S2_f) * (r_l_f[d0] - r_k_f[d0]);
        const auto QD_1_f = (-a_k_f * inv_S2_f) * (r_l_f[d1] - r_k_f[d1]);

        const float eri_ijkl_f = Lambda_f * S_ij_00_f * S_kl_00_f * (

                + F6_t_f[2] * (

                    0.25f * S2_f * S2_f * inv_S1_f * inv_S1_f * inv_S4_f * inv_S4_f * (
                        +QD_0_f*QD_1_f*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.25f * S2_f * inv_S1_f * inv_S4_f * inv_S4_f * (
                        +delta_f[b0][b1]*delta_f[d0][d1]*(-PA_0_f*PQ_f[a1] + PQ_f[a0]*(-PA_1_f + PQ_f[a1]))

                        +delta_f[d0][d1]*(PQ_f[a0]*delta_f[a1][b0]*(-PB_1_f + PQ_f[b1]) + PQ_f[a0]*delta_f[a1][b1]*(-PB_0_f + PQ_f[b0]) + PQ_f[a1]*delta_f[a0][b0]*(-PB_1_f + PQ_f[b1]) + PQ_f[a1]*delta_f[a0][b1]*(-PB_0_f + PQ_f[b0]) + PQ_f[b0]*PQ_f[b1]*delta_f[a0][a1] - PQ_f[b0]*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) - PQ_f[b1]*(PA_0_f*delta_f[a1][b0] + PA_1_f*delta_f[a0][b0] + PB_0_f*delta_f[a0][a1]))

                        +(-PA_0_f*QD_0_f + PQ_f[a0]*QD_0_f)*(delta_f[a1][b0]*delta_f[b1][d1] + delta_f[a1][b1]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][b1]) + (-PA_0_f*QD_1_f + PQ_f[a0]*QD_1_f)*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + (-PA_1_f*QD_0_f + PQ_f[a1]*QD_0_f)*(delta_f[a0][b0]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][b1]) + (-PA_1_f*QD_1_f + PQ_f[a1]*QD_1_f)*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + (-PB_0_f*QD_0_f + PQ_f[b0]*QD_0_f)*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + (-PB_0_f*QD_1_f + PQ_f[b0]*QD_1_f)*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + (-PB_1_f*QD_0_f + PQ_f[b1]*QD_0_f)*(delta_f[a0][a1]*delta_f[b0][d1] + delta_f[a0][b0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b0]) + (-PB_1_f*QD_1_f + PQ_f[b1]*QD_1_f)*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0])

                        +2.0f*(PQ_f[d0]*QD_1_f + PQ_f[d1]*QD_0_f)*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.25f * inv_S4_f * inv_S4_f * (
                        +delta_f[a1][b1]*delta_f[d0][d1]*(PA_0_f*(PB_0_f - PQ_f[b0]) - PB_0_f*PQ_f[a0])

                        +PA_0_f*(PA_1_f*(delta_f[b0][d0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][d0]) + PB_0_f*(delta_f[a1][d0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b1][d0]) + PB_1_f*(delta_f[a1][d0]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][d0])) + PA_1_f*(PB_0_f*(delta_f[a0][d0]*delta_f[b1][d1] + delta_f[a0][d1]*delta_f[b1][d0]) + PB_1_f*(delta_f[a0][d0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][d0])) + PB_0_f*PB_1_f*(delta_f[a0][d0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][d0])

                        +delta_f[d0][d1]*(PA_1_f*(PB_1_f*delta_f[a0][b0] + delta_f[a0][b1]*(PB_0_f - PQ_f[b0]) + delta_f[b0][b1]*(PA_0_f - PQ_f[a0])) + PB_1_f*delta_f[a0][a1]*(PB_0_f - PQ_f[b0]) + PB_1_f*delta_f[a1][b0]*(PA_0_f - PQ_f[a0]) - PQ_f[a1]*(PA_0_f*delta_f[b0][b1] + PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) - PQ_f[b1]*(PA_0_f*delta_f[a1][b0] + PA_1_f*delta_f[a0][b0] + PB_0_f*delta_f[a0][a1]))

                        -PQ_f[d0]*(PA_0_f*(delta_f[a1][b0]*delta_f[b1][d1] + delta_f[a1][b1]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][b1]) + PA_1_f*(delta_f[a0][b0]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][b1]) + PB_0_f*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + PB_1_f*(delta_f[a0][a1]*delta_f[b0][d1] + delta_f[a0][b0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b0])) - PQ_f[d1]*(PA_0_f*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + PA_1_f*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + PB_0_f*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + PB_1_f*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0]))

                        +PQ_f[d0]*PQ_f[d1]*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.5f * S1_f * inv_S4_f * inv_S4_f * (
                        -delta_f[d0][d1]*(PA_0_f*PA_1_f*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PB_1_f*(PA_0_f*PQ_f[a1] + PA_1_f*PQ_f[a0]))

                        -PA_0_f*PA_1_f*(PB_0_f*(PQ_f[d0]*delta_f[b1][d1] + PQ_f[d1]*delta_f[b1][d0]) + PB_1_f*(PQ_f[d0]*delta_f[b0][d1] + PQ_f[d1]*delta_f[b0][d0])) - PB_0_f*PB_1_f*(PA_0_f*(PQ_f[d0]*delta_f[a1][d1] + PQ_f[d1]*delta_f[a1][d0]) + PA_1_f*(PQ_f[d0]*delta_f[a0][d1] + PQ_f[d1]*delta_f[a0][d0]))

                        +PQ_f[d0]*PQ_f[d1]*(PA_0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) + PA_1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PB_0_f*PB_1_f*delta_f[a0][a1])
                    )

                    + 0.5f * S2_f * S2_f * inv_S1_f * inv_S4_f * inv_S4_f * (
                        +QD_0_f*QD_1_f*delta_f[b0][b1]*(-PA_0_f*PQ_f[a1] + PQ_f[a0]*(-PA_1_f + PQ_f[a1]))

                        +QD_0_f*QD_1_f*(PQ_f[a0]*delta_f[a1][b0]*(-PB_1_f + PQ_f[b1]) + PQ_f[a0]*delta_f[a1][b1]*(-PB_0_f + PQ_f[b0]) + PQ_f[a1]*delta_f[a0][b0]*(-PB_1_f + PQ_f[b1]) + PQ_f[a1]*delta_f[a0][b1]*(-PB_0_f + PQ_f[b0]) + PQ_f[b0]*PQ_f[b1]*delta_f[a0][a1] - PQ_f[b0]*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) - PQ_f[b1]*(PA_0_f*delta_f[a1][b0] + PA_1_f*delta_f[a0][b0] + PB_0_f*delta_f[a0][a1]))
                    )

                    + 0.5f * S2_f * inv_S4_f * inv_S4_f * (
                        +QD_0_f*delta_f[b1][d1]*(PA_0_f*(PA_1_f*PQ_f[b0] + PB_0_f*PQ_f[a1]) + PA_1_f*PB_0_f*PQ_f[a0])

                        +PB_1_f*(PA_0_f*(PQ_f[a1]*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0]) + PQ_f[b0]*(QD_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][d0])) + PQ_f[a0]*(PA_1_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0]) + PB_0_f*(QD_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][d0]))) + QD_0_f*(PA_0_f*PQ_f[b1]*(PA_1_f*delta_f[b0][d1] + PB_0_f*delta_f[a1][d1]) + delta_f[a0][d1]*(PA_1_f*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PB_1_f*PQ_f[a1])) + QD_1_f*(PA_0_f*(PA_1_f*(PQ_f[b0]*delta_f[b1][d0] + PQ_f[b1]*delta_f[b0][d0]) + PB_0_f*(PQ_f[a1]*delta_f[b1][d0] + PQ_f[b1]*delta_f[a1][d0])) + PA_1_f*PB_0_f*(PQ_f[a0]*delta_f[b1][d0] + PQ_f[b1]*delta_f[a0][d0]) + PB_1_f*delta_f[a0][d0]*(PA_1_f*PQ_f[b0] + PB_0_f*PQ_f[a1]))

                        +(PQ_f[d0]*QD_1_f + PQ_f[d1]*QD_0_f)*(PA_0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) - PA_0_f*(PQ_f[a1]*delta_f[b0][b1] + PQ_f[b0]*delta_f[a1][b1] + PQ_f[b1]*delta_f[a1][b0]) + PA_1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) - PA_1_f*(PQ_f[a0]*delta_f[b0][b1] + PQ_f[b0]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][b0]) + PB_0_f*PB_1_f*delta_f[a0][a1] - PB_0_f*(PQ_f[a0]*delta_f[a1][b1] + PQ_f[a1]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][a1]) - PB_1_f*(PQ_f[a0]*delta_f[a1][b0] + PQ_f[a1]*delta_f[a0][b0] + PQ_f[b0]*delta_f[a0][a1]))

                        +delta_f[d0][d1]*(PA_0_f*PQ_f[b0]*(PA_1_f*PQ_f[b1] + PB_1_f*PQ_f[a1]) + PA_1_f*PQ_f[a0]*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PQ_f[a1]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))
                    )

                    + S1_f * S1_f * inv_S4_f * inv_S4_f * (
                        +PA_0_f*PA_1_f*PB_0_f*PB_1_f*PQ_f[d0]*PQ_f[d1]
                    )

                    + S1_f * S2_f * inv_S4_f * inv_S4_f * (
                        -(PQ_f[d0]*QD_1_f + PQ_f[d1]*QD_0_f)*(PA_0_f*PA_1_f*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PB_1_f*(PA_0_f*PQ_f[a1] + PA_1_f*PQ_f[a0]))
                    )

                    + S2_f * S2_f * inv_S4_f * inv_S4_f * (
                        +QD_0_f*QD_1_f*(PA_0_f*PQ_f[b0]*(PA_1_f*PQ_f[b1] + PB_1_f*PQ_f[a1]) + PA_1_f*PQ_f[a0]*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PQ_f[a1]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))
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
                        +PQ_f[a0]*PQ_f[a1]*delta_f[b0][b1]*delta_f[d0][d1]

                        +delta_f[d0][d1]*(PQ_f[b0]*(PQ_f[a0]*delta_f[a1][b1] + PQ_f[a1]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][a1]) + PQ_f[b1]*(PQ_f[a0]*delta_f[a1][b0] + PQ_f[a1]*delta_f[a0][b0]))

                        +QD_0_f*(PQ_f[a0]*(delta_f[a1][b0]*delta_f[b1][d1] + delta_f[a1][b1]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][b1]) + PQ_f[a1]*(delta_f[a0][b0]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][b1]) + PQ_f[b0]*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + PQ_f[b1]*(delta_f[a0][a1]*delta_f[b0][d1] + delta_f[a0][b0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b0])) + QD_1_f*(PQ_f[a0]*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + PQ_f[a1]*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + PQ_f[b0]*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + PQ_f[b1]*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0]))

                        +(PQ_f[d0]*QD_1_f + PQ_f[d1]*QD_0_f)*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.25f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +delta_f[a1][b1]*delta_f[d0][d1]*(PB_0_f*PQ_f[a0] + PQ_f[b0]*(PA_0_f - PQ_f[a0]))

                        +(PA_0_f*PQ_f[a1] + PA_1_f*PQ_f[a0])*(delta_f[b0][d0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][d0]) + (PA_0_f*PQ_f[b0] + PB_0_f*PQ_f[a0])*(delta_f[a1][d0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b1][d0]) + (PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0])*(delta_f[a1][d0]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][d0]) + (PA_1_f*PQ_f[b0] + PB_0_f*PQ_f[a1])*(delta_f[a0][d0]*delta_f[b1][d1] + delta_f[a0][d1]*delta_f[b1][d0]) + (PA_1_f*PQ_f[b1] + PB_1_f*PQ_f[a1])*(delta_f[a0][d0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][d0]) + (PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0])*(delta_f[a0][d0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][d0])

                        +delta_f[d0][d1]*(PA_1_f*(PQ_f[a0]*delta_f[b0][b1] + PQ_f[b0]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][b0]) + PB_1_f*(PQ_f[a0]*delta_f[a1][b0] + PQ_f[b0]*delta_f[a0][a1]) + PQ_f[a1]*(PA_0_f*delta_f[b0][b1] + PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0] - PQ_f[a0]*delta_f[b0][b1] - PQ_f[b0]*delta_f[a0][b1] - PQ_f[b1]*delta_f[a0][b0]) + PQ_f[b1]*(delta_f[a0][a1]*(PB_0_f - PQ_f[b0]) + delta_f[a1][b0]*(PA_0_f - PQ_f[a0])))

                        +(PA_0_f*PQ_f[d0] - PQ_f[a0]*PQ_f[d0])*(delta_f[a1][b0]*delta_f[b1][d1] + delta_f[a1][b1]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][b1]) + (PA_0_f*PQ_f[d1] - PQ_f[a0]*PQ_f[d1])*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + (PA_1_f*PQ_f[d0] - PQ_f[a1]*PQ_f[d0])*(delta_f[a0][b0]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][b1]) + (PA_1_f*PQ_f[d1] - PQ_f[a1]*PQ_f[d1])*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + (PB_0_f*PQ_f[d0] - PQ_f[b0]*PQ_f[d0])*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + (PB_0_f*PQ_f[d1] - PQ_f[b0]*PQ_f[d1])*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + (PB_1_f*PQ_f[d0] - PQ_f[b1]*PQ_f[d0])*(delta_f[a0][a1]*delta_f[b0][d1] + delta_f[a0][b0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b0]) + (PB_1_f*PQ_f[d1] - PQ_f[b1]*PQ_f[d1])*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0])

                        -2*PQ_f[d0]*PQ_f[d1]*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.5f * S1_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        -delta_f[d0][d1]*(PA_0_f*PQ_f[b0]*(PA_1_f*PQ_f[b1] + PB_1_f*PQ_f[a1]) + PA_1_f*PQ_f[a0]*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PQ_f[a1]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))

                        -PA_0_f*(PA_1_f*(PQ_f[b0]*(PQ_f[d0]*delta_f[b1][d1] + PQ_f[d1]*delta_f[b1][d0]) + PQ_f[b1]*(PQ_f[d0]*delta_f[b0][d1] + PQ_f[d1]*delta_f[b0][d0])) + PB_0_f*(PQ_f[a1]*(PQ_f[d0]*delta_f[b1][d1] + PQ_f[d1]*delta_f[b1][d0]) + PQ_f[b1]*(PQ_f[d0]*delta_f[a1][d1] + PQ_f[d1]*delta_f[a1][d0])) + PB_1_f*(PQ_f[a1]*(PQ_f[d0]*delta_f[b0][d1] + PQ_f[d1]*delta_f[b0][d0]) + PQ_f[b0]*(PQ_f[d0]*delta_f[a1][d1] + PQ_f[d1]*delta_f[a1][d0]))) - PA_1_f*(PB_0_f*(PQ_f[a0]*(PQ_f[d0]*delta_f[b1][d1] + PQ_f[d1]*delta_f[b1][d0]) + PQ_f[b1]*(PQ_f[d0]*delta_f[a0][d1] + PQ_f[d1]*delta_f[a0][d0])) + PB_1_f*(PQ_f[a0]*(PQ_f[d0]*delta_f[b0][d1] + PQ_f[d1]*delta_f[b0][d0]) + PQ_f[b0]*(PQ_f[d0]*delta_f[a0][d1] + PQ_f[d1]*delta_f[a0][d0]))) - PB_0_f*PB_1_f*(PQ_f[a0]*(PQ_f[d0]*delta_f[a1][d1] + PQ_f[d1]*delta_f[a1][d0]) + PQ_f[a1]*(PQ_f[d0]*delta_f[a0][d1] + PQ_f[d1]*delta_f[a0][d0]))

                        +PQ_f[d0]*PQ_f[d1]*(PA_0_f*(-PA_1_f*delta_f[b0][b1] - PB_0_f*delta_f[a1][b1] - PB_1_f*delta_f[a1][b0] + PQ_f[a1]*delta_f[b0][b1] + PQ_f[b0]*delta_f[a1][b1] + PQ_f[b1]*delta_f[a1][b0]) + PA_1_f*(PQ_f[a0]*delta_f[b0][b1] + delta_f[a0][b0]*(-PB_1_f + PQ_f[b1]) + delta_f[a0][b1]*(-PB_0_f + PQ_f[b0])) + PB_0_f*(PQ_f[a0]*delta_f[a1][b1] + PQ_f[a1]*delta_f[a0][b1] + delta_f[a0][a1]*(-PB_1_f + PQ_f[b1])) + PB_1_f*(PQ_f[a0]*delta_f[a1][b0] + PQ_f[a1]*delta_f[a0][b0] + PQ_f[b0]*delta_f[a0][a1]))
                    )

                    + (-0.5f) * S2_f * S2_f * S2_f * inv_S1_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[a0]*PQ_f[a1]*QD_0_f*QD_1_f*delta_f[b0][b1]

                        +QD_0_f*QD_1_f*(PQ_f[b0]*(PQ_f[a0]*delta_f[a1][b1] + PQ_f[a1]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][a1]) + PQ_f[b1]*(PQ_f[a0]*delta_f[a1][b0] + PQ_f[a1]*delta_f[a0][b0]))
                    )

                    + 0.5f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +QD_0_f*delta_f[b1][d1]*(PA_1_f*PQ_f[a0]*PQ_f[b0] + PQ_f[a1]*(PA_0_f*PQ_f[b0] + PB_0_f*PQ_f[a0]))

                        +PB_1_f*PQ_f[a0]*(PQ_f[a1]*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0]) + PQ_f[b0]*(QD_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][d0])) + QD_0_f*(PB_1_f*PQ_f[a1]*PQ_f[b0]*delta_f[a0][d1] + PQ_f[b1]*(PA_0_f*(PQ_f[a1]*delta_f[b0][d1] + PQ_f[b0]*delta_f[a1][d1]) + PA_1_f*(PQ_f[a0]*delta_f[b0][d1] + PQ_f[b0]*delta_f[a0][d1]) + PB_0_f*(PQ_f[a0]*delta_f[a1][d1] + PQ_f[a1]*delta_f[a0][d1]))) + QD_1_f*(PQ_f[a0]*(PA_1_f*(PQ_f[b0]*delta_f[b1][d0] + PQ_f[b1]*delta_f[b0][d0]) + PB_0_f*(PQ_f[a1]*delta_f[b1][d0] + PQ_f[b1]*delta_f[a1][d0])) + PQ_f[a1]*(PA_0_f*(PQ_f[b0]*delta_f[b1][d0] + PQ_f[b1]*delta_f[b0][d0]) + delta_f[a0][d0]*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0])) + PQ_f[b0]*PQ_f[b1]*(PA_0_f*delta_f[a1][d0] + PA_1_f*delta_f[a0][d0]))

                        +(PQ_f[d0]*QD_1_f + PQ_f[d1]*QD_0_f)*(PA_0_f*(PQ_f[a1]*delta_f[b0][b1] + PQ_f[b0]*delta_f[a1][b1] + PQ_f[b1]*delta_f[a1][b0]) + PQ_f[a0]*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0] - PQ_f[a1]*delta_f[b0][b1] - PQ_f[b0]*delta_f[a1][b1] - PQ_f[b1]*delta_f[a1][b0]) + PQ_f[a1]*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PQ_f[b0]*(PB_1_f*delta_f[a0][a1] + delta_f[a0][b1]*(PA_1_f - PQ_f[a1])) + PQ_f[b1]*(delta_f[a0][a1]*(PB_0_f - PQ_f[b0]) + delta_f[a0][b0]*(PA_1_f - PQ_f[a1])))

                        +delta_f[d0][d1]*(PQ_f[a0]*PQ_f[b1]*(PA_1_f*PQ_f[b0] + PB_0_f*PQ_f[a1]) + PQ_f[a1]*PQ_f[b0]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))
                    )

                    + S1_f * S1_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[d0]*PQ_f[d1]*(PA_0_f*PA_1_f*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PB_1_f*(PA_0_f*PQ_f[a1] + PA_1_f*PQ_f[a0]))
                    )

                    + S1_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        -(PQ_f[d0]*QD_1_f + PQ_f[d1]*QD_0_f)*(PA_0_f*PQ_f[b0]*(PA_1_f*PQ_f[b1] + PB_1_f*PQ_f[a1]) + PA_1_f*PQ_f[a0]*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PQ_f[a1]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))
                    )

                    + S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +QD_0_f*QD_1_f*(PQ_f[a0]*PQ_f[b1]*(PA_1_f*PQ_f[b0] + PB_0_f*PQ_f[a1]) + PQ_f[a1]*PQ_f[b0]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))
                    )

                )

                + F6_t_f[4] * (

                    0.5f * S1_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        -delta_f[d0][d1]*(PQ_f[a0]*PQ_f[b1]*(PA_1_f*PQ_f[b0] + PB_0_f*PQ_f[a1]) + PQ_f[a1]*PQ_f[b0]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))

                        -PQ_f[a0]*(PA_1_f*(PQ_f[b0]*(PQ_f[d0]*delta_f[b1][d1] + PQ_f[d1]*delta_f[b1][d0]) + PQ_f[b1]*(PQ_f[d0]*delta_f[b0][d1] + PQ_f[d1]*delta_f[b0][d0])) + PB_0_f*(PQ_f[a1]*(PQ_f[d0]*delta_f[b1][d1] + PQ_f[d1]*delta_f[b1][d0]) + PQ_f[b1]*(PQ_f[d0]*delta_f[a1][d1] + PQ_f[d1]*delta_f[a1][d0])) + PB_1_f*(PQ_f[a1]*(PQ_f[d0]*delta_f[b0][d1] + PQ_f[d1]*delta_f[b0][d0]) + PQ_f[b0]*(PQ_f[d0]*delta_f[a1][d1] + PQ_f[d1]*delta_f[a1][d0]))) - PQ_f[a1]*(PA_0_f*(PQ_f[b0]*(PQ_f[d0]*delta_f[b1][d1] + PQ_f[d1]*delta_f[b1][d0]) + PQ_f[b1]*(PQ_f[d0]*delta_f[b0][d1] + PQ_f[d1]*delta_f[b0][d0])) + (PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0])*(PQ_f[d0]*delta_f[a0][d1] + PQ_f[d1]*delta_f[a0][d0])) - PQ_f[b0]*PQ_f[b1]*(PA_0_f*(PQ_f[d0]*delta_f[a1][d1] + PQ_f[d1]*delta_f[a1][d0]) + PA_1_f*(PQ_f[d0]*delta_f[a0][d1] + PQ_f[d1]*delta_f[a0][d0]))

                        +PQ_f[d0]*PQ_f[d1]*(-PA_0_f*(PQ_f[a1]*delta_f[b0][b1] + PQ_f[b0]*delta_f[a1][b1] + PQ_f[b1]*delta_f[a1][b0]) - PQ_f[a0]*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) + PQ_f[a0]*(PQ_f[a1]*delta_f[b0][b1] + PQ_f[b0]*delta_f[a1][b1] + PQ_f[b1]*delta_f[a1][b0]) - PQ_f[a1]*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PQ_f[a1]*(PQ_f[b0]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][b0]) + PQ_f[b0]*PQ_f[b1]*delta_f[a0][a1] - PQ_f[b0]*(PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) - PQ_f[b1]*(PA_1_f*delta_f[a0][b0] + PB_0_f*delta_f[a0][a1]))
                    )

                    + 0.5f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[a0]*PQ_f[a1]*PQ_f[b0]*QD_0_f*delta_f[b1][d1]

                        +PQ_f[a0]*PQ_f[a1]*QD_1_f*(PQ_f[b0]*delta_f[b1][d0] + PQ_f[b1]*delta_f[b0][d0]) + PQ_f[b1]*(PQ_f[a0]*QD_0_f*(PQ_f[a1]*delta_f[b0][d1] + PQ_f[b0]*delta_f[a1][d1]) + PQ_f[b0]*(PQ_f[a1]*QD_0_f*delta_f[a0][d1] + QD_1_f*(PQ_f[a0]*delta_f[a1][d0] + PQ_f[a1]*delta_f[a0][d0])))

                        +(PQ_f[d0]*QD_1_f + PQ_f[d1]*QD_0_f)*(PQ_f[a0]*(PQ_f[a1]*delta_f[b0][b1] + PQ_f[b0]*delta_f[a1][b1] + PQ_f[b1]*delta_f[a1][b0]) + PQ_f[a1]*(PQ_f[b0]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][b0]) + PQ_f[b0]*PQ_f[b1]*delta_f[a0][a1])

                        +PQ_f[a0]*PQ_f[a1]*PQ_f[b0]*PQ_f[b1]*delta_f[d0][d1]
                    )

                    + S1_f * S1_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[d0]*PQ_f[d1]*(PA_0_f*PQ_f[b0]*(PA_1_f*PQ_f[b1] + PB_1_f*PQ_f[a1]) + PA_1_f*PQ_f[a0]*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PQ_f[a1]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))
                    )

                    + S1_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        -(PQ_f[d0]*QD_1_f + PQ_f[d1]*QD_0_f)*(PQ_f[a0]*PQ_f[b1]*(PA_1_f*PQ_f[b0] + PB_0_f*PQ_f[a1]) + PQ_f[a1]*PQ_f[b0]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))
                    )

                    + S2_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[a0]*PQ_f[a1]*PQ_f[b0]*PQ_f[b1]*QD_0_f*QD_1_f
                    )

                    + 0.25f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[a0]*PQ_f[a1]*(delta_f[b0][b1]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][d0])

                        +PQ_f[b0]*(PQ_f[a0]*(delta_f[a1][b1]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b1][d0]) + PQ_f[a1]*(delta_f[a0][b1]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[b1][d1] + delta_f[a0][d1]*delta_f[b1][d0]) + PQ_f[b1]*(delta_f[a0][a1]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][d0]) + PQ_f[d0]*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + PQ_f[d1]*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1])) + PQ_f[b1]*(PQ_f[a0]*(delta_f[a1][b0]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][d0]) + PQ_f[a1]*(delta_f[a0][b0]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][d0]) + PQ_f[d0]*(delta_f[a0][a1]*delta_f[b0][d1] + delta_f[a0][b0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b0]) + PQ_f[d1]*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0])) + PQ_f[d0]*(PQ_f[a0]*(delta_f[a1][b0]*delta_f[b1][d1] + delta_f[a1][b1]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][b1]) + PQ_f[a1]*(delta_f[a0][b0]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][b1]) + PQ_f[d1]*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])) + PQ_f[d1]*(PQ_f[a0]*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + PQ_f[a1]*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]))
                    )

                )

                + F6_t_f[5] * (

                    (-0.5f) * S1_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[a0]*PQ_f[a1]*PQ_f[b0]*PQ_f[b1]*delta_f[d0][d1]

                        +PQ_f[d0]*(PQ_f[a0]*(PQ_f[a1]*(PQ_f[b0]*delta_f[b1][d1] + PQ_f[b1]*delta_f[b0][d1] + PQ_f[d1]*delta_f[b0][b1]) + PQ_f[b1]*PQ_f[d1]*delta_f[a1][b0]) + PQ_f[b0]*PQ_f[b1]*(PQ_f[a0]*delta_f[a1][d1] + PQ_f[a1]*delta_f[a0][d1] + PQ_f[d1]*delta_f[a0][a1])) + PQ_f[d1]*(PQ_f[a1]*PQ_f[b1]*(PQ_f[a0]*delta_f[b0][d0] + PQ_f[b0]*delta_f[a0][d0] + PQ_f[d0]*delta_f[a0][b0]) + PQ_f[b0]*(PQ_f[a0]*(PQ_f[a1]*delta_f[b1][d0] + PQ_f[b1]*delta_f[a1][d0] + PQ_f[d0]*delta_f[a1][b1]) + PQ_f[a1]*PQ_f[d0]*delta_f[a0][b1]))
                    )

                    + S1_f * S1_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[d0]*PQ_f[d1]*(PQ_f[a0]*PQ_f[b1]*(PA_1_f*PQ_f[b0] + PB_0_f*PQ_f[a1]) + PQ_f[a1]*PQ_f[b0]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))
                    )

                    + S1_f * S2_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[a0]*PQ_f[a1]*PQ_f[b0]*PQ_f[b1]*(-PQ_f[d0]*QD_1_f - PQ_f[d1]*QD_0_f)
                    )

                )

                + F6_t_f[6] * (

                    S1_f * S1_f * S2_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[a0]*PQ_f[a1]*PQ_f[b0]*PQ_f[b1]*PQ_f[d0]*PQ_f[d1]
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
        double J_ij = 0.0;

        for (uint32_t n = 0; n < TILE_DIM_LARGE; n++)
        {
            J_ij += ERIs[n];
        }

        mat_J[ij] += J_ij;
    }
}

// DDSD altB DDSD0 BASE
__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDSD_altB0(double*         mat_J,
                       const double*   s_prim_info,
                       const uint32_t  s_prim_count,
                       const double*   d_prim_info,
                       const uint32_t  d_prim_count,
                       const double*   sd_mat_D,
                       const double*   dd_mat_Q_local,
                       const double*   sd_mat_Q,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const double*   dd_pair_data_local,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* sd_first_inds,
                       const uint32_t* sd_second_inds,
                       const double*   sd_pair_data,
                       const uint32_t  sd_prim_pair_count,
                       const double*   boys_func_table,
                       const double*   boys_func_ft,
                       const double    eri_threshold)
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

    for (uint32_t m = 0; m < (sd_prim_pair_count + TILE_DIM_LARGE - 1) / TILE_DIM_LARGE; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= sd_prim_pair_count) || (fabs(dd_mat_Q_local[ij] * sd_mat_Q[kl] * sd_mat_D[kl]) <= eri_threshold))
        {
            break;
        }

        const auto k = sd_first_inds[kl];
        const auto l = sd_second_inds[kl];

        const auto a_k = s_prim_info[k + s_prim_count * 0];

        const double r_k[3] = {s_prim_info[k + s_prim_count * 2],
                               s_prim_info[k + s_prim_count * 3],
                               s_prim_info[k + s_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = sd_pair_data[kl];

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

        double F6_t[2];

        gpu::computeBoysFunction(F6_t, S1 * S2 * inv_S4 * r2_PQ, 1, boys_func_table, boys_func_ft);

        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                + F6_t[0] * (

                    0.25 * inv_S1 * inv_S1 * (
                        +QD_0*QD_1*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * inv_S1 * inv_S2 * (
                        +PA_0*PB_0*delta[a1][b1]*delta[d0][d1]

                        +delta[d0][d1]*(PA_1*(PA_0*delta[b0][b1] + PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_1*(PA_0*delta[a1][b0] + PB_0*delta[a0][a1]))
                    )

                    + 0.5 * inv_S1 * (
                        +PA_0*PA_1*QD_0*QD_1*delta[b0][b1]

                        +QD_0*QD_1*(PB_0*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) + PB_1*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0]))
                    )

                    + 0.5 * inv_S2 * (
                        +PA_0*PA_1*PB_0*PB_1*delta[d0][d1]
                    )

                    + (
                        +PA_0*PA_1*PB_0*PB_1*QD_0*QD_1
                    )

                    + 0.125 * inv_S1 * inv_S1 * inv_S2 * (
                        +delta[d0][d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                )

                + F6_t[1] * (

                    (-0.25) * inv_S1 * inv_S1 * inv_S4 * (
                        +delta[d0][d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + (-0.125) * inv_S1 * inv_S2 * inv_S4 * (
                        +delta[d0][d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + (-0.5) * S2 * inv_S1 * inv_S1 * inv_S4 * (
                        +QD_0*QD_1*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * inv_S1 * inv_S4 * (
                        +delta[b0][b1]*delta[d0][d1]*(PA_0*(-PA_1 + PQ[a1]) + PA_1*PQ[a0])

                        +delta[d0][d1]*(PB_0*(-PA_0*delta[a1][b1] - PA_1*delta[a0][b1] - PB_1*delta[a0][a1] + PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + PQ[b0]*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) + delta[a0][b0]*(PA_1*(-PB_1 + PQ[b1]) + PB_1*PQ[a1]) + delta[a1][b0]*(PA_0*(-PB_1 + PQ[b1]) + PB_1*PQ[a0]))

                        +QD_0*(PA_0*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0])) + QD_1*(PA_0*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]))

                        -(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + (-0.25) * inv_S2 * inv_S4 * (
                        +PA_0*PA_1*delta[b0][b1]*delta[d0][d1]

                        +delta[d0][d1]*(PB_0*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) + PB_1*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0]))
                    )

                    + (-0.5) * S1 * inv_S2 * inv_S4 * (
                        +PA_0*PA_1*PB_0*PB_1*delta[d0][d1]
                    )

                    + 0.5 * S2 * inv_S1 * inv_S4 * (
                        +QD_0*QD_1*delta[b0][b1]*(PA_0*(-PA_1 + PQ[a1]) + PA_1*PQ[a0])

                        +QD_0*QD_1*(PB_0*(-PA_0*delta[a1][b1] - PA_1*delta[a0][b1] - PB_1*delta[a0][a1] + PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + PQ[b0]*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) + delta[a0][b0]*(PA_1*(-PB_1 + PQ[b1]) + PB_1*PQ[a1]) + delta[a1][b0]*(PA_0*(-PB_1 + PQ[b1]) + PB_1*PQ[a0]))
                    )

                    + 0.5 * inv_S4 * (
                        +delta[d0][d1]*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))

                        +PA_0*PA_1*(PB_0*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + PB_1*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0])) + PB_0*PB_1*(PA_0*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]) + PA_1*(QD_0*delta[a0][d1] + QD_1*delta[a0][d0]))

                        -(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1])
                    )

                    + S1 * inv_S4 * (
                        +PA_0*PA_1*PB_0*PB_1*(-PQ[d0]*QD_1 - PQ[d1]*QD_0)
                    )

                    + S2 * inv_S4 * (
                        +QD_0*QD_1*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))
                    )

                )

            );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        ERIs[threadIdx.y] += eri_ijkl * sd_mat_D[kl] * 2.0;
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

// DDSD altB DDSD0 FP64
__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDSD_altB0_FP64(double*         mat_J,
                       const double*   s_prim_info,
                       const uint32_t  s_prim_count,
                       const double*   d_prim_info,
                       const uint32_t  d_prim_count,
                       const double*   sd_mat_D,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const double*   dd_pair_data_local,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* sd_first_inds,
                       const uint32_t* sd_second_inds,
                       const double*   sd_pair_data,
                       const uint32_t  sd_prim_pair_count,
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

        if ((ij >= dd_prim_pair_count_local) || (kl >= sd_prim_pair_count))
        {
            break;
        }

        const auto k = sd_first_inds[kl];
        const auto l = sd_second_inds[kl];

        const auto a_k = s_prim_info[k + s_prim_count * 0];

        const double r_k[3] = {s_prim_info[k + s_prim_count * 2],
                               s_prim_info[k + s_prim_count * 3],
                               s_prim_info[k + s_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = sd_pair_data[kl];

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

        double F6_t[2];

        gpu::computeBoysFunction(F6_t, S1 * S2 * inv_S4 * r2_PQ, 1, boys_func_table, boys_func_ft);

        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                + F6_t[0] * (

                    0.25 * inv_S1 * inv_S1 * (
                        +QD_0*QD_1*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * inv_S1 * inv_S2 * (
                        +PA_0*PB_0*delta[a1][b1]*delta[d0][d1]

                        +delta[d0][d1]*(PA_1*(PA_0*delta[b0][b1] + PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_1*(PA_0*delta[a1][b0] + PB_0*delta[a0][a1]))
                    )

                    + 0.5 * inv_S1 * (
                        +PA_0*PA_1*QD_0*QD_1*delta[b0][b1]

                        +QD_0*QD_1*(PB_0*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) + PB_1*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0]))
                    )

                    + 0.5 * inv_S2 * (
                        +PA_0*PA_1*PB_0*PB_1*delta[d0][d1]
                    )

                    + (
                        +PA_0*PA_1*PB_0*PB_1*QD_0*QD_1
                    )

                    + 0.125 * inv_S1 * inv_S1 * inv_S2 * (
                        +delta[d0][d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                )

                + F6_t[1] * (

                    (-0.25) * inv_S1 * inv_S1 * inv_S4 * (
                        +delta[d0][d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + (-0.125) * inv_S1 * inv_S2 * inv_S4 * (
                        +delta[d0][d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + (-0.5) * S2 * inv_S1 * inv_S1 * inv_S4 * (
                        +QD_0*QD_1*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * inv_S1 * inv_S4 * (
                        +delta[b0][b1]*delta[d0][d1]*(PA_0*(-PA_1 + PQ[a1]) + PA_1*PQ[a0])

                        +delta[d0][d1]*(PB_0*(-PA_0*delta[a1][b1] - PA_1*delta[a0][b1] - PB_1*delta[a0][a1] + PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + PQ[b0]*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) + delta[a0][b0]*(PA_1*(-PB_1 + PQ[b1]) + PB_1*PQ[a1]) + delta[a1][b0]*(PA_0*(-PB_1 + PQ[b1]) + PB_1*PQ[a0]))

                        +QD_0*(PA_0*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0])) + QD_1*(PA_0*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]))

                        -(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + (-0.25) * inv_S2 * inv_S4 * (
                        +PA_0*PA_1*delta[b0][b1]*delta[d0][d1]

                        +delta[d0][d1]*(PB_0*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) + PB_1*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0]))
                    )

                    + (-0.5) * S1 * inv_S2 * inv_S4 * (
                        +PA_0*PA_1*PB_0*PB_1*delta[d0][d1]
                    )

                    + 0.5 * S2 * inv_S1 * inv_S4 * (
                        +QD_0*QD_1*delta[b0][b1]*(PA_0*(-PA_1 + PQ[a1]) + PA_1*PQ[a0])

                        +QD_0*QD_1*(PB_0*(-PA_0*delta[a1][b1] - PA_1*delta[a0][b1] - PB_1*delta[a0][a1] + PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + PQ[b0]*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) + delta[a0][b0]*(PA_1*(-PB_1 + PQ[b1]) + PB_1*PQ[a1]) + delta[a1][b0]*(PA_0*(-PB_1 + PQ[b1]) + PB_1*PQ[a0]))
                    )

                    + 0.5 * inv_S4 * (
                        +delta[d0][d1]*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))

                        +PA_0*PA_1*(PB_0*(QD_0*delta[b1][d1] + QD_1*delta[b1][d0]) + PB_1*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0])) + PB_0*PB_1*(PA_0*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]) + PA_1*(QD_0*delta[a0][d1] + QD_1*delta[a0][d0]))

                        -(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1])
                    )

                    + S1 * inv_S4 * (
                        +PA_0*PA_1*PB_0*PB_1*(-PQ[d0]*QD_1 - PQ[d1]*QD_0)
                    )

                    + S2 * inv_S4 * (
                        +QD_0*QD_1*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))
                    )

                )

            );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        ERIs[threadIdx.y] += eri_ijkl * sd_mat_D[kl] * 2.0;
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

// DDSD altB DDSD0 FP32
__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDSD_altB0_FP32(double*         mat_J,
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

        const float r_k_f[3] = {s_prim_info_f[k + s_prim_count * 2],
                               s_prim_info_f[k + s_prim_count * 3],
                               s_prim_info_f[k + s_prim_count * 4]};

        const auto a_l_f = d_prim_info_f[l / 6 + d_prim_count * 0];

        const float r_l_f[3] = {d_prim_info_f[l / 6 + d_prim_count * 2],
                               d_prim_info_f[l / 6 + d_prim_count * 3],
                               d_prim_info_f[l / 6 + d_prim_count * 4]};

        const auto S_kl_00_f = sd_pair_data_f[kl];

        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2_f = a_k_f + a_l_f;

        const auto inv_S2_f = 1.0f / S2_f;
        const auto inv_S4_f = 1.0f / (S1_f + S2_f);

        const float PQ_f[3] = {(a_k_f * r_k_f[0] + a_l_f * r_l_f[0]) * inv_S2_f - (a_i_f * r_i_f[0] + a_j_f * r_j_f[0]) * inv_S1_f,
                              (a_k_f * r_k_f[1] + a_l_f * r_l_f[1]) * inv_S2_f - (a_i_f * r_i_f[1] + a_j_f * r_j_f[1]) * inv_S1_f,
                              (a_k_f * r_k_f[2] + a_l_f * r_l_f[2]) * inv_S2_f - (a_i_f * r_i_f[2] + a_j_f * r_j_f[2]) * inv_S1_f};

        const auto r2_PQ_f = PQ_f[0] * PQ_f[0] + PQ_f[1] * PQ_f[1] + PQ_f[2] * PQ_f[2];

        const auto Lambda_f = sqrtf(4.0f * S1_f * S2_f * MATH_CONST_INV_PI_F * inv_S4_f);

        float F6_t_f[2];

        gpu::computeBoysFunction_f(F6_t_f, S1_f * S2_f * inv_S4_f * r2_PQ_f, 1, boys_func_table_f, boys_func_ft_f);

        const auto QD_0_f = (-a_k_f * inv_S2_f) * (r_l_f[d0] - r_k_f[d0]);
        const auto QD_1_f = (-a_k_f * inv_S2_f) * (r_l_f[d1] - r_k_f[d1]);

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
                        +delta_f[b0][b1]*delta_f[d0][d1]*(PA_0_f*(-PA_1_f + PQ_f[a1]) + PA_1_f*PQ_f[a0])

                        +delta_f[d0][d1]*(PB_0_f*(-PA_0_f*delta_f[a1][b1] - PA_1_f*delta_f[a0][b1] - PB_1_f*delta_f[a0][a1] + PQ_f[a0]*delta_f[a1][b1] + PQ_f[a1]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][a1]) + PQ_f[b0]*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) + delta_f[a0][b0]*(PA_1_f*(-PB_1_f + PQ_f[b1]) + PB_1_f*PQ_f[a1]) + delta_f[a1][b0]*(PA_0_f*(-PB_1_f + PQ_f[b1]) + PB_1_f*PQ_f[a0]))

                        +QD_0_f*(PA_0_f*(delta_f[a1][b0]*delta_f[b1][d1] + delta_f[a1][b1]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][b1]) + PA_1_f*(delta_f[a0][b0]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][b1]) + PB_0_f*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + PB_1_f*(delta_f[a0][a1]*delta_f[b0][d1] + delta_f[a0][b0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b0])) + QD_1_f*(PA_0_f*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + PA_1_f*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + PB_0_f*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + PB_1_f*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0]))

                        -(PQ_f[d0]*QD_1_f + PQ_f[d1]*QD_0_f)*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + (-0.25f) * inv_S2_f * inv_S4_f * (
                        +PA_0_f*PA_1_f*delta_f[b0][b1]*delta_f[d0][d1]

                        +delta_f[d0][d1]*(PB_0_f*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) + PB_1_f*(PA_0_f*delta_f[a1][b0] + PA_1_f*delta_f[a0][b0]))
                    )

                    + (-0.5f) * S1_f * inv_S2_f * inv_S4_f * (
                        +PA_0_f*PA_1_f*PB_0_f*PB_1_f*delta_f[d0][d1]
                    )

                    + 0.5f * S2_f * inv_S1_f * inv_S4_f * (
                        +QD_0_f*QD_1_f*delta_f[b0][b1]*(PA_0_f*(-PA_1_f + PQ_f[a1]) + PA_1_f*PQ_f[a0])

                        +QD_0_f*QD_1_f*(PB_0_f*(-PA_0_f*delta_f[a1][b1] - PA_1_f*delta_f[a0][b1] - PB_1_f*delta_f[a0][a1] + PQ_f[a0]*delta_f[a1][b1] + PQ_f[a1]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][a1]) + PQ_f[b0]*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) + delta_f[a0][b0]*(PA_1_f*(-PB_1_f + PQ_f[b1]) + PB_1_f*PQ_f[a1]) + delta_f[a1][b0]*(PA_0_f*(-PB_1_f + PQ_f[b1]) + PB_1_f*PQ_f[a0]))
                    )

                    + 0.5f * inv_S4_f * (
                        +delta_f[d0][d1]*(PA_0_f*PA_1_f*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PB_1_f*(PA_0_f*PQ_f[a1] + PA_1_f*PQ_f[a0]))

                        +PA_0_f*PA_1_f*(PB_0_f*(QD_0_f*delta_f[b1][d1] + QD_1_f*delta_f[b1][d0]) + PB_1_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0])) + PB_0_f*PB_1_f*(PA_0_f*(QD_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][d0]) + PA_1_f*(QD_0_f*delta_f[a0][d1] + QD_1_f*delta_f[a0][d0]))

                        -(PQ_f[d0]*QD_1_f + PQ_f[d1]*QD_0_f)*(PA_0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) + PA_1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PB_0_f*PB_1_f*delta_f[a0][a1])
                    )

                    + S1_f * inv_S4_f * (
                        +PA_0_f*PA_1_f*PB_0_f*PB_1_f*(-PQ_f[d0]*QD_1_f - PQ_f[d1]*QD_0_f)
                    )

                    + S2_f * inv_S4_f * (
                        +QD_0_f*QD_1_f*(PA_0_f*PA_1_f*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PB_1_f*(PA_0_f*PQ_f[a1] + PA_1_f*PQ_f[a0]))
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
        double J_ij = 0.0;

        for (uint32_t n = 0; n < TILE_DIM_LARGE; n++)
        {
            J_ij += ERIs[n];
        }

        mat_J[ij] += J_ij;
    }
}

// DDSD altB DDSD1 BASE
__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDSD_altB1(double*         mat_J,
                       const double*   s_prim_info,
                       const uint32_t  s_prim_count,
                       const double*   d_prim_info,
                       const uint32_t  d_prim_count,
                       const double*   sd_mat_D,
                       const double*   dd_mat_Q_local,
                       const double*   sd_mat_Q,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const double*   dd_pair_data_local,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* sd_first_inds,
                       const uint32_t* sd_second_inds,
                       const double*   sd_pair_data,
                       const uint32_t  sd_prim_pair_count,
                       const double*   boys_func_table,
                       const double*   boys_func_ft,
                       const double    eri_threshold)
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

    for (uint32_t m = 0; m < (sd_prim_pair_count + TILE_DIM_LARGE - 1) / TILE_DIM_LARGE; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= sd_prim_pair_count) || (fabs(dd_mat_Q_local[ij] * sd_mat_Q[kl] * sd_mat_D[kl]) <= eri_threshold))
        {
            break;
        }

        const auto k = sd_first_inds[kl];
        const auto l = sd_second_inds[kl];

        const auto a_k = s_prim_info[k + s_prim_count * 0];

        const double r_k[3] = {s_prim_info[k + s_prim_count * 2],
                               s_prim_info[k + s_prim_count * 3],
                               s_prim_info[k + s_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = sd_pair_data[kl];

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

        double F6_t[4];

        gpu::computeBoysFunction(F6_t, S1 * S2 * inv_S4 * r2_PQ, 3, boys_func_table, boys_func_ft);

        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                + F6_t[2] * (

                    0.25 * S2 * S2 * inv_S1 * inv_S1 * inv_S4 * inv_S4 * (
                        +QD_0*QD_1*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * S2 * inv_S1 * inv_S4 * inv_S4 * (
                        +delta[b0][b1]*delta[d0][d1]*(-PA_0*PQ[a1] + PQ[a0]*(-PA_1 + PQ[a1]))

                        +delta[d0][d1]*(PQ[a0]*delta[a1][b0]*(-PB_1 + PQ[b1]) + PQ[a0]*delta[a1][b1]*(-PB_0 + PQ[b0]) + PQ[a1]*delta[a0][b0]*(-PB_1 + PQ[b1]) + PQ[a1]*delta[a0][b1]*(-PB_0 + PQ[b0]) + PQ[b0]*PQ[b1]*delta[a0][a1] - PQ[b0]*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) - PQ[b1]*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0] + PB_0*delta[a0][a1]))

                        +(-PA_0*QD_0 + PQ[a0]*QD_0)*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + (-PA_0*QD_1 + PQ[a0]*QD_1)*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + (-PA_1*QD_0 + PQ[a1]*QD_0)*(delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1]) + (-PA_1*QD_1 + PQ[a1]*QD_1)*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + (-PB_0*QD_0 + PQ[b0]*QD_0)*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + (-PB_0*QD_1 + PQ[b0]*QD_1)*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + (-PB_1*QD_0 + PQ[b1]*QD_0)*(delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0]) + (-PB_1*QD_1 + PQ[b1]*QD_1)*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])

                        +2.0*(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * inv_S4 * inv_S4 * (
                        +delta[a1][b1]*delta[d0][d1]*(PA_0*(PB_0 - PQ[b0]) - PB_0*PQ[a0])

                        +PA_0*(PA_1*(delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + PB_0*(delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + PB_1*(delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0])) + PA_1*(PB_0*(delta[a0][d0]*delta[b1][d1] + delta[a0][d1]*delta[b1][d0]) + PB_1*(delta[a0][d0]*delta[b0][d1] + delta[a0][d1]*delta[b0][d0])) + PB_0*PB_1*(delta[a0][d0]*delta[a1][d1] + delta[a0][d1]*delta[a1][d0])

                        +delta[d0][d1]*(PA_1*(PB_1*delta[a0][b0] + delta[a0][b1]*(PB_0 - PQ[b0]) + delta[b0][b1]*(PA_0 - PQ[a0])) + PB_1*delta[a0][a1]*(PB_0 - PQ[b0]) + PB_1*delta[a1][b0]*(PA_0 - PQ[a0]) - PQ[a1]*(PA_0*delta[b0][b1] + PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) - PQ[b1]*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0] + PB_0*delta[a0][a1]))

                        -PQ[d0]*(PA_0*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0])) - PQ[d1]*(PA_0*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]))

                        +PQ[d0]*PQ[d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.5 * S1 * inv_S4 * inv_S4 * (
                        -delta[d0][d1]*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))

                        -PA_0*PA_1*(PB_0*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PB_1*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0])) - PB_0*PB_1*(PA_0*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0]) + PA_1*(PQ[d0]*delta[a0][d1] + PQ[d1]*delta[a0][d0]))

                        +PQ[d0]*PQ[d1]*(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1])
                    )

                    + 0.5 * S2 * S2 * inv_S1 * inv_S4 * inv_S4 * (
                        +QD_0*QD_1*delta[b0][b1]*(-PA_0*PQ[a1] + PQ[a0]*(-PA_1 + PQ[a1]))

                        +QD_0*QD_1*(PQ[a0]*delta[a1][b0]*(-PB_1 + PQ[b1]) + PQ[a0]*delta[a1][b1]*(-PB_0 + PQ[b0]) + PQ[a1]*delta[a0][b0]*(-PB_1 + PQ[b1]) + PQ[a1]*delta[a0][b1]*(-PB_0 + PQ[b0]) + PQ[b0]*PQ[b1]*delta[a0][a1] - PQ[b0]*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) - PQ[b1]*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0] + PB_0*delta[a0][a1]))
                    )

                    + 0.5 * S2 * inv_S4 * inv_S4 * (
                        +QD_0*delta[b1][d1]*(PA_0*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PA_1*PB_0*PQ[a0])

                        +PB_1*(PA_0*(PQ[a1]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + PQ[b0]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0])) + PQ[a0]*(PA_1*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + PB_0*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]))) + QD_0*(PA_0*PQ[b1]*(PA_1*delta[b0][d1] + PB_0*delta[a1][d1]) + delta[a0][d1]*(PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*PQ[a1])) + QD_1*(PA_0*(PA_1*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0]) + PB_0*(PQ[a1]*delta[b1][d0] + PQ[b1]*delta[a1][d0])) + PA_1*PB_0*(PQ[a0]*delta[b1][d0] + PQ[b1]*delta[a0][d0]) + PB_1*delta[a0][d0]*(PA_1*PQ[b0] + PB_0*PQ[a1]))

                        +(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) - PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) - PA_1*(PQ[a0]*delta[b0][b1] + PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1] - PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) - PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]))

                        +delta[d0][d1]*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S1 * S1 * inv_S4 * inv_S4 * (
                        +PA_0*PA_1*PB_0*PB_1*PQ[d0]*PQ[d1]
                    )

                    + S1 * S2 * inv_S4 * inv_S4 * (
                        -(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))
                    )

                    + S2 * S2 * inv_S4 * inv_S4 * (
                        +QD_0*QD_1*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + 0.125 * S2 * inv_S1 * inv_S1 * inv_S4 * inv_S4 * (
                        +delta[d0][d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.125 * inv_S1 * inv_S4 * inv_S4 * (
                        +2.0*delta[d0][d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])

                        +delta[b0][b1]*(delta[a0][d0]*delta[a1][d1] + delta[a0][d1]*delta[a1][d0]) + delta[b0][d0]*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + delta[b0][d1]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + delta[b1][d0]*(delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0]) + delta[b1][d1]*(delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])
                    )

                )

                + F6_t[3] * (

                    (-0.125) * S2 * inv_S1 * inv_S4 * inv_S4 * inv_S4 * (
                        +delta[a0][a1]*(delta[b0][b1]*delta[d0][d1] + delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + delta[a0][b0]*(delta[a1][b1]*delta[d0][d1] + delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + delta[a0][b1]*(delta[a1][b0]*delta[d0][d1] + delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + delta[a0][d0]*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + delta[a0][d1]*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1])
                    )

                    + (-0.25) * S2 * S2 * inv_S1 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*delta[b0][b1]*delta[d0][d1]

                        +delta[d0][d1]*(PQ[b0]*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + PQ[b1]*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0]))

                        +QD_0*(PQ[a0]*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1]) + PQ[b0]*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + PQ[b1]*(delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0])) + QD_1*(PQ[a0]*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + PQ[b0]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + PQ[b1]*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]))

                        +(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        +delta[a1][b1]*delta[d0][d1]*(PB_0*PQ[a0] + PQ[b0]*(PA_0 - PQ[a0]))

                        +(PA_0*PQ[a1] + PA_1*PQ[a0])*(delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + (PA_0*PQ[b0] + PB_0*PQ[a0])*(delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + (PA_0*PQ[b1] + PB_1*PQ[a0])*(delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + (PA_1*PQ[b0] + PB_0*PQ[a1])*(delta[a0][d0]*delta[b1][d1] + delta[a0][d1]*delta[b1][d0]) + (PA_1*PQ[b1] + PB_1*PQ[a1])*(delta[a0][d0]*delta[b0][d1] + delta[a0][d1]*delta[b0][d0]) + (PB_0*PQ[b1] + PB_1*PQ[b0])*(delta[a0][d0]*delta[a1][d1] + delta[a0][d1]*delta[a1][d0])

                        +delta[d0][d1]*(PA_1*(PQ[a0]*delta[b0][b1] + PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PB_1*(PQ[a0]*delta[a1][b0] + PQ[b0]*delta[a0][a1]) + PQ[a1]*(PA_0*delta[b0][b1] + PB_0*delta[a0][b1] + PB_1*delta[a0][b0] - PQ[a0]*delta[b0][b1] - PQ[b0]*delta[a0][b1] - PQ[b1]*delta[a0][b0]) + PQ[b1]*(delta[a0][a1]*(PB_0 - PQ[b0]) + delta[a1][b0]*(PA_0 - PQ[a0])))

                        +(PA_0*PQ[d0] - PQ[a0]*PQ[d0])*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + (PA_0*PQ[d1] - PQ[a0]*PQ[d1])*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + (PA_1*PQ[d0] - PQ[a1]*PQ[d0])*(delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1]) + (PA_1*PQ[d1] - PQ[a1]*PQ[d1])*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + (PB_0*PQ[d0] - PQ[b0]*PQ[d0])*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + (PB_0*PQ[d1] - PQ[b0]*PQ[d1])*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + (PB_1*PQ[d0] - PQ[b1]*PQ[d0])*(delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0]) + (PB_1*PQ[d1] - PQ[b1]*PQ[d1])*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])

                        -2*PQ[d0]*PQ[d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.5 * S1 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        -delta[d0][d1]*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))

                        -PA_0*(PA_1*(PQ[b0]*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PQ[b1]*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0])) + PB_0*(PQ[a1]*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PQ[b1]*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0])) + PB_1*(PQ[a1]*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0]) + PQ[b0]*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0]))) - PA_1*(PB_0*(PQ[a0]*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PQ[b1]*(PQ[d0]*delta[a0][d1] + PQ[d1]*delta[a0][d0])) + PB_1*(PQ[a0]*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0]) + PQ[b0]*(PQ[d0]*delta[a0][d1] + PQ[d1]*delta[a0][d0]))) - PB_0*PB_1*(PQ[a0]*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0]) + PQ[a1]*(PQ[d0]*delta[a0][d1] + PQ[d1]*delta[a0][d0]))

                        +PQ[d0]*PQ[d1]*(PA_0*(-PA_1*delta[b0][b1] - PB_0*delta[a1][b1] - PB_1*delta[a1][b0] + PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PA_1*(PQ[a0]*delta[b0][b1] + delta[a0][b0]*(-PB_1 + PQ[b1]) + delta[a0][b1]*(-PB_0 + PQ[b0])) + PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + delta[a0][a1]*(-PB_1 + PQ[b1])) + PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]))
                    )

                    + (-0.5) * S2 * S2 * S2 * inv_S1 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*QD_0*QD_1*delta[b0][b1]

                        +QD_0*QD_1*(PQ[b0]*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + PQ[b1]*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0]))
                    )

                    + 0.5 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        +QD_0*delta[b1][d1]*(PA_1*PQ[a0]*PQ[b0] + PQ[a1]*(PA_0*PQ[b0] + PB_0*PQ[a0]))

                        +PB_1*PQ[a0]*(PQ[a1]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + PQ[b0]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0])) + QD_0*(PB_1*PQ[a1]*PQ[b0]*delta[a0][d1] + PQ[b1]*(PA_0*(PQ[a1]*delta[b0][d1] + PQ[b0]*delta[a1][d1]) + PA_1*(PQ[a0]*delta[b0][d1] + PQ[b0]*delta[a0][d1]) + PB_0*(PQ[a0]*delta[a1][d1] + PQ[a1]*delta[a0][d1]))) + QD_1*(PQ[a0]*(PA_1*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0]) + PB_0*(PQ[a1]*delta[b1][d0] + PQ[b1]*delta[a1][d0])) + PQ[a1]*(PA_0*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0]) + delta[a0][d0]*(PB_0*PQ[b1] + PB_1*PQ[b0])) + PQ[b0]*PQ[b1]*(PA_0*delta[a1][d0] + PA_1*delta[a0][d0]))

                        +(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PQ[a0]*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0] - PQ[a1]*delta[b0][b1] - PQ[b0]*delta[a1][b1] - PQ[b1]*delta[a1][b0]) + PQ[a1]*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PQ[b0]*(PB_1*delta[a0][a1] + delta[a0][b1]*(PA_1 - PQ[a1])) + PQ[b1]*(delta[a0][a1]*(PB_0 - PQ[b0]) + delta[a0][b0]*(PA_1 - PQ[a1])))

                        +delta[d0][d1]*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S1 * S1 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[d0]*PQ[d1]*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))
                    )

                    + S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        -(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        +QD_0*QD_1*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                )

            );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        ERIs[threadIdx.y] += eri_ijkl * sd_mat_D[kl] * 2.0;
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

// DDSD altB DDSD1 FP64
__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDSD_altB1_FP64(double*         mat_J,
                       const double*   s_prim_info,
                       const uint32_t  s_prim_count,
                       const double*   d_prim_info,
                       const uint32_t  d_prim_count,
                       const double*   sd_mat_D,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const double*   dd_pair_data_local,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* sd_first_inds,
                       const uint32_t* sd_second_inds,
                       const double*   sd_pair_data,
                       const uint32_t  sd_prim_pair_count,
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

        if ((ij >= dd_prim_pair_count_local) || (kl >= sd_prim_pair_count))
        {
            break;
        }

        const auto k = sd_first_inds[kl];
        const auto l = sd_second_inds[kl];

        const auto a_k = s_prim_info[k + s_prim_count * 0];

        const double r_k[3] = {s_prim_info[k + s_prim_count * 2],
                               s_prim_info[k + s_prim_count * 3],
                               s_prim_info[k + s_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = sd_pair_data[kl];

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

        double F6_t[4];

        gpu::computeBoysFunction(F6_t, S1 * S2 * inv_S4 * r2_PQ, 3, boys_func_table, boys_func_ft);

        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                + F6_t[2] * (

                    0.25 * S2 * S2 * inv_S1 * inv_S1 * inv_S4 * inv_S4 * (
                        +QD_0*QD_1*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * S2 * inv_S1 * inv_S4 * inv_S4 * (
                        +delta[b0][b1]*delta[d0][d1]*(-PA_0*PQ[a1] + PQ[a0]*(-PA_1 + PQ[a1]))

                        +delta[d0][d1]*(PQ[a0]*delta[a1][b0]*(-PB_1 + PQ[b1]) + PQ[a0]*delta[a1][b1]*(-PB_0 + PQ[b0]) + PQ[a1]*delta[a0][b0]*(-PB_1 + PQ[b1]) + PQ[a1]*delta[a0][b1]*(-PB_0 + PQ[b0]) + PQ[b0]*PQ[b1]*delta[a0][a1] - PQ[b0]*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) - PQ[b1]*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0] + PB_0*delta[a0][a1]))

                        +(-PA_0*QD_0 + PQ[a0]*QD_0)*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + (-PA_0*QD_1 + PQ[a0]*QD_1)*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + (-PA_1*QD_0 + PQ[a1]*QD_0)*(delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1]) + (-PA_1*QD_1 + PQ[a1]*QD_1)*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + (-PB_0*QD_0 + PQ[b0]*QD_0)*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + (-PB_0*QD_1 + PQ[b0]*QD_1)*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + (-PB_1*QD_0 + PQ[b1]*QD_0)*(delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0]) + (-PB_1*QD_1 + PQ[b1]*QD_1)*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])

                        +2.0*(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * inv_S4 * inv_S4 * (
                        +delta[a1][b1]*delta[d0][d1]*(PA_0*(PB_0 - PQ[b0]) - PB_0*PQ[a0])

                        +PA_0*(PA_1*(delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + PB_0*(delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + PB_1*(delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0])) + PA_1*(PB_0*(delta[a0][d0]*delta[b1][d1] + delta[a0][d1]*delta[b1][d0]) + PB_1*(delta[a0][d0]*delta[b0][d1] + delta[a0][d1]*delta[b0][d0])) + PB_0*PB_1*(delta[a0][d0]*delta[a1][d1] + delta[a0][d1]*delta[a1][d0])

                        +delta[d0][d1]*(PA_1*(PB_1*delta[a0][b0] + delta[a0][b1]*(PB_0 - PQ[b0]) + delta[b0][b1]*(PA_0 - PQ[a0])) + PB_1*delta[a0][a1]*(PB_0 - PQ[b0]) + PB_1*delta[a1][b0]*(PA_0 - PQ[a0]) - PQ[a1]*(PA_0*delta[b0][b1] + PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) - PQ[b1]*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0] + PB_0*delta[a0][a1]))

                        -PQ[d0]*(PA_0*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0])) - PQ[d1]*(PA_0*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]))

                        +PQ[d0]*PQ[d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.5 * S1 * inv_S4 * inv_S4 * (
                        -delta[d0][d1]*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))

                        -PA_0*PA_1*(PB_0*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PB_1*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0])) - PB_0*PB_1*(PA_0*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0]) + PA_1*(PQ[d0]*delta[a0][d1] + PQ[d1]*delta[a0][d0]))

                        +PQ[d0]*PQ[d1]*(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1])
                    )

                    + 0.5 * S2 * S2 * inv_S1 * inv_S4 * inv_S4 * (
                        +QD_0*QD_1*delta[b0][b1]*(-PA_0*PQ[a1] + PQ[a0]*(-PA_1 + PQ[a1]))

                        +QD_0*QD_1*(PQ[a0]*delta[a1][b0]*(-PB_1 + PQ[b1]) + PQ[a0]*delta[a1][b1]*(-PB_0 + PQ[b0]) + PQ[a1]*delta[a0][b0]*(-PB_1 + PQ[b1]) + PQ[a1]*delta[a0][b1]*(-PB_0 + PQ[b0]) + PQ[b0]*PQ[b1]*delta[a0][a1] - PQ[b0]*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) - PQ[b1]*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0] + PB_0*delta[a0][a1]))
                    )

                    + 0.5 * S2 * inv_S4 * inv_S4 * (
                        +QD_0*delta[b1][d1]*(PA_0*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PA_1*PB_0*PQ[a0])

                        +PB_1*(PA_0*(PQ[a1]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + PQ[b0]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0])) + PQ[a0]*(PA_1*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + PB_0*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0]))) + QD_0*(PA_0*PQ[b1]*(PA_1*delta[b0][d1] + PB_0*delta[a1][d1]) + delta[a0][d1]*(PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*PQ[a1])) + QD_1*(PA_0*(PA_1*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0]) + PB_0*(PQ[a1]*delta[b1][d0] + PQ[b1]*delta[a1][d0])) + PA_1*PB_0*(PQ[a0]*delta[b1][d0] + PQ[b1]*delta[a0][d0]) + PB_1*delta[a0][d0]*(PA_1*PQ[b0] + PB_0*PQ[a1]))

                        +(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) - PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) - PA_1*(PQ[a0]*delta[b0][b1] + PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1] - PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) - PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]))

                        +delta[d0][d1]*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S1 * S1 * inv_S4 * inv_S4 * (
                        +PA_0*PA_1*PB_0*PB_1*PQ[d0]*PQ[d1]
                    )

                    + S1 * S2 * inv_S4 * inv_S4 * (
                        -(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))
                    )

                    + S2 * S2 * inv_S4 * inv_S4 * (
                        +QD_0*QD_1*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + 0.125 * S2 * inv_S1 * inv_S1 * inv_S4 * inv_S4 * (
                        +delta[d0][d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.125 * inv_S1 * inv_S4 * inv_S4 * (
                        +2.0*delta[d0][d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])

                        +delta[b0][b1]*(delta[a0][d0]*delta[a1][d1] + delta[a0][d1]*delta[a1][d0]) + delta[b0][d0]*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + delta[b0][d1]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + delta[b1][d0]*(delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0]) + delta[b1][d1]*(delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])
                    )

                )

                + F6_t[3] * (

                    (-0.125) * S2 * inv_S1 * inv_S4 * inv_S4 * inv_S4 * (
                        +delta[a0][a1]*(delta[b0][b1]*delta[d0][d1] + delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + delta[a0][b0]*(delta[a1][b1]*delta[d0][d1] + delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + delta[a0][b1]*(delta[a1][b0]*delta[d0][d1] + delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + delta[a0][d0]*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + delta[a0][d1]*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1])
                    )

                    + (-0.25) * S2 * S2 * inv_S1 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*delta[b0][b1]*delta[d0][d1]

                        +delta[d0][d1]*(PQ[b0]*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + PQ[b1]*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0]))

                        +QD_0*(PQ[a0]*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1]) + PQ[b0]*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + PQ[b1]*(delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0])) + QD_1*(PQ[a0]*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + PQ[b0]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + PQ[b1]*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]))

                        +(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        +delta[a1][b1]*delta[d0][d1]*(PB_0*PQ[a0] + PQ[b0]*(PA_0 - PQ[a0]))

                        +(PA_0*PQ[a1] + PA_1*PQ[a0])*(delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0]) + (PA_0*PQ[b0] + PB_0*PQ[a0])*(delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + (PA_0*PQ[b1] + PB_1*PQ[a0])*(delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + (PA_1*PQ[b0] + PB_0*PQ[a1])*(delta[a0][d0]*delta[b1][d1] + delta[a0][d1]*delta[b1][d0]) + (PA_1*PQ[b1] + PB_1*PQ[a1])*(delta[a0][d0]*delta[b0][d1] + delta[a0][d1]*delta[b0][d0]) + (PB_0*PQ[b1] + PB_1*PQ[b0])*(delta[a0][d0]*delta[a1][d1] + delta[a0][d1]*delta[a1][d0])

                        +delta[d0][d1]*(PA_1*(PQ[a0]*delta[b0][b1] + PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PB_1*(PQ[a0]*delta[a1][b0] + PQ[b0]*delta[a0][a1]) + PQ[a1]*(PA_0*delta[b0][b1] + PB_0*delta[a0][b1] + PB_1*delta[a0][b0] - PQ[a0]*delta[b0][b1] - PQ[b0]*delta[a0][b1] - PQ[b1]*delta[a0][b0]) + PQ[b1]*(delta[a0][a1]*(PB_0 - PQ[b0]) + delta[a1][b0]*(PA_0 - PQ[a0])))

                        +(PA_0*PQ[d0] - PQ[a0]*PQ[d0])*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + (PA_0*PQ[d1] - PQ[a0]*PQ[d1])*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + (PA_1*PQ[d0] - PQ[a1]*PQ[d0])*(delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1]) + (PA_1*PQ[d1] - PQ[a1]*PQ[d1])*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + (PB_0*PQ[d0] - PQ[b0]*PQ[d0])*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + (PB_0*PQ[d1] - PQ[b0]*PQ[d1])*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + (PB_1*PQ[d0] - PQ[b1]*PQ[d0])*(delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0]) + (PB_1*PQ[d1] - PQ[b1]*PQ[d1])*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])

                        -2*PQ[d0]*PQ[d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.5 * S1 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        -delta[d0][d1]*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))

                        -PA_0*(PA_1*(PQ[b0]*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PQ[b1]*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0])) + PB_0*(PQ[a1]*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PQ[b1]*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0])) + PB_1*(PQ[a1]*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0]) + PQ[b0]*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0]))) - PA_1*(PB_0*(PQ[a0]*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PQ[b1]*(PQ[d0]*delta[a0][d1] + PQ[d1]*delta[a0][d0])) + PB_1*(PQ[a0]*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0]) + PQ[b0]*(PQ[d0]*delta[a0][d1] + PQ[d1]*delta[a0][d0]))) - PB_0*PB_1*(PQ[a0]*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0]) + PQ[a1]*(PQ[d0]*delta[a0][d1] + PQ[d1]*delta[a0][d0]))

                        +PQ[d0]*PQ[d1]*(PA_0*(-PA_1*delta[b0][b1] - PB_0*delta[a1][b1] - PB_1*delta[a1][b0] + PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PA_1*(PQ[a0]*delta[b0][b1] + delta[a0][b0]*(-PB_1 + PQ[b1]) + delta[a0][b1]*(-PB_0 + PQ[b0])) + PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + delta[a0][a1]*(-PB_1 + PQ[b1])) + PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]))
                    )

                    + (-0.5) * S2 * S2 * S2 * inv_S1 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*QD_0*QD_1*delta[b0][b1]

                        +QD_0*QD_1*(PQ[b0]*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + PQ[b1]*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0]))
                    )

                    + 0.5 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        +QD_0*delta[b1][d1]*(PA_1*PQ[a0]*PQ[b0] + PQ[a1]*(PA_0*PQ[b0] + PB_0*PQ[a0]))

                        +PB_1*PQ[a0]*(PQ[a1]*(QD_0*delta[b0][d1] + QD_1*delta[b0][d0]) + PQ[b0]*(QD_0*delta[a1][d1] + QD_1*delta[a1][d0])) + QD_0*(PB_1*PQ[a1]*PQ[b0]*delta[a0][d1] + PQ[b1]*(PA_0*(PQ[a1]*delta[b0][d1] + PQ[b0]*delta[a1][d1]) + PA_1*(PQ[a0]*delta[b0][d1] + PQ[b0]*delta[a0][d1]) + PB_0*(PQ[a0]*delta[a1][d1] + PQ[a1]*delta[a0][d1]))) + QD_1*(PQ[a0]*(PA_1*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0]) + PB_0*(PQ[a1]*delta[b1][d0] + PQ[b1]*delta[a1][d0])) + PQ[a1]*(PA_0*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0]) + delta[a0][d0]*(PB_0*PQ[b1] + PB_1*PQ[b0])) + PQ[b0]*PQ[b1]*(PA_0*delta[a1][d0] + PA_1*delta[a0][d0]))

                        +(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PQ[a0]*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0] - PQ[a1]*delta[b0][b1] - PQ[b0]*delta[a1][b1] - PQ[b1]*delta[a1][b0]) + PQ[a1]*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PQ[b0]*(PB_1*delta[a0][a1] + delta[a0][b1]*(PA_1 - PQ[a1])) + PQ[b1]*(delta[a0][a1]*(PB_0 - PQ[b0]) + delta[a0][b0]*(PA_1 - PQ[a1])))

                        +delta[d0][d1]*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S1 * S1 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[d0]*PQ[d1]*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))
                    )

                    + S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        -(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        +QD_0*QD_1*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                )

            );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        ERIs[threadIdx.y] += eri_ijkl * sd_mat_D[kl] * 2.0;
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

// DDSD altB DDSD1 FP32
__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDSD_altB1_FP32(double*         mat_J,
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

        const float r_k_f[3] = {s_prim_info_f[k + s_prim_count * 2],
                               s_prim_info_f[k + s_prim_count * 3],
                               s_prim_info_f[k + s_prim_count * 4]};

        const auto a_l_f = d_prim_info_f[l / 6 + d_prim_count * 0];

        const float r_l_f[3] = {d_prim_info_f[l / 6 + d_prim_count * 2],
                               d_prim_info_f[l / 6 + d_prim_count * 3],
                               d_prim_info_f[l / 6 + d_prim_count * 4]};

        const auto S_kl_00_f = sd_pair_data_f[kl];

        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2_f = a_k_f + a_l_f;

        const auto inv_S2_f = 1.0f / S2_f;
        const auto inv_S4_f = 1.0f / (S1_f + S2_f);

        const float PQ_f[3] = {(a_k_f * r_k_f[0] + a_l_f * r_l_f[0]) * inv_S2_f - (a_i_f * r_i_f[0] + a_j_f * r_j_f[0]) * inv_S1_f,
                              (a_k_f * r_k_f[1] + a_l_f * r_l_f[1]) * inv_S2_f - (a_i_f * r_i_f[1] + a_j_f * r_j_f[1]) * inv_S1_f,
                              (a_k_f * r_k_f[2] + a_l_f * r_l_f[2]) * inv_S2_f - (a_i_f * r_i_f[2] + a_j_f * r_j_f[2]) * inv_S1_f};

        const auto r2_PQ_f = PQ_f[0] * PQ_f[0] + PQ_f[1] * PQ_f[1] + PQ_f[2] * PQ_f[2];

        const auto Lambda_f = sqrtf(4.0f * S1_f * S2_f * MATH_CONST_INV_PI_F * inv_S4_f);

        float F6_t_f[4];

        gpu::computeBoysFunction_f(F6_t_f, S1_f * S2_f * inv_S4_f * r2_PQ_f, 3, boys_func_table_f, boys_func_ft_f);

        const auto QD_0_f = (-a_k_f * inv_S2_f) * (r_l_f[d0] - r_k_f[d0]);
        const auto QD_1_f = (-a_k_f * inv_S2_f) * (r_l_f[d1] - r_k_f[d1]);

        const float eri_ijkl_f = Lambda_f * S_ij_00_f * S_kl_00_f * (

                + F6_t_f[2] * (

                    0.25f * S2_f * S2_f * inv_S1_f * inv_S1_f * inv_S4_f * inv_S4_f * (
                        +QD_0_f*QD_1_f*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.25f * S2_f * inv_S1_f * inv_S4_f * inv_S4_f * (
                        +delta_f[b0][b1]*delta_f[d0][d1]*(-PA_0_f*PQ_f[a1] + PQ_f[a0]*(-PA_1_f + PQ_f[a1]))

                        +delta_f[d0][d1]*(PQ_f[a0]*delta_f[a1][b0]*(-PB_1_f + PQ_f[b1]) + PQ_f[a0]*delta_f[a1][b1]*(-PB_0_f + PQ_f[b0]) + PQ_f[a1]*delta_f[a0][b0]*(-PB_1_f + PQ_f[b1]) + PQ_f[a1]*delta_f[a0][b1]*(-PB_0_f + PQ_f[b0]) + PQ_f[b0]*PQ_f[b1]*delta_f[a0][a1] - PQ_f[b0]*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) - PQ_f[b1]*(PA_0_f*delta_f[a1][b0] + PA_1_f*delta_f[a0][b0] + PB_0_f*delta_f[a0][a1]))

                        +(-PA_0_f*QD_0_f + PQ_f[a0]*QD_0_f)*(delta_f[a1][b0]*delta_f[b1][d1] + delta_f[a1][b1]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][b1]) + (-PA_0_f*QD_1_f + PQ_f[a0]*QD_1_f)*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + (-PA_1_f*QD_0_f + PQ_f[a1]*QD_0_f)*(delta_f[a0][b0]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][b1]) + (-PA_1_f*QD_1_f + PQ_f[a1]*QD_1_f)*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + (-PB_0_f*QD_0_f + PQ_f[b0]*QD_0_f)*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + (-PB_0_f*QD_1_f + PQ_f[b0]*QD_1_f)*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + (-PB_1_f*QD_0_f + PQ_f[b1]*QD_0_f)*(delta_f[a0][a1]*delta_f[b0][d1] + delta_f[a0][b0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b0]) + (-PB_1_f*QD_1_f + PQ_f[b1]*QD_1_f)*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0])

                        +2.0f*(PQ_f[d0]*QD_1_f + PQ_f[d1]*QD_0_f)*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.25f * inv_S4_f * inv_S4_f * (
                        +delta_f[a1][b1]*delta_f[d0][d1]*(PA_0_f*(PB_0_f - PQ_f[b0]) - PB_0_f*PQ_f[a0])

                        +PA_0_f*(PA_1_f*(delta_f[b0][d0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][d0]) + PB_0_f*(delta_f[a1][d0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b1][d0]) + PB_1_f*(delta_f[a1][d0]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][d0])) + PA_1_f*(PB_0_f*(delta_f[a0][d0]*delta_f[b1][d1] + delta_f[a0][d1]*delta_f[b1][d0]) + PB_1_f*(delta_f[a0][d0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][d0])) + PB_0_f*PB_1_f*(delta_f[a0][d0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][d0])

                        +delta_f[d0][d1]*(PA_1_f*(PB_1_f*delta_f[a0][b0] + delta_f[a0][b1]*(PB_0_f - PQ_f[b0]) + delta_f[b0][b1]*(PA_0_f - PQ_f[a0])) + PB_1_f*delta_f[a0][a1]*(PB_0_f - PQ_f[b0]) + PB_1_f*delta_f[a1][b0]*(PA_0_f - PQ_f[a0]) - PQ_f[a1]*(PA_0_f*delta_f[b0][b1] + PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) - PQ_f[b1]*(PA_0_f*delta_f[a1][b0] + PA_1_f*delta_f[a0][b0] + PB_0_f*delta_f[a0][a1]))

                        -PQ_f[d0]*(PA_0_f*(delta_f[a1][b0]*delta_f[b1][d1] + delta_f[a1][b1]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][b1]) + PA_1_f*(delta_f[a0][b0]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][b1]) + PB_0_f*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + PB_1_f*(delta_f[a0][a1]*delta_f[b0][d1] + delta_f[a0][b0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b0])) - PQ_f[d1]*(PA_0_f*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + PA_1_f*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + PB_0_f*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + PB_1_f*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0]))

                        +PQ_f[d0]*PQ_f[d1]*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.5f * S1_f * inv_S4_f * inv_S4_f * (
                        -delta_f[d0][d1]*(PA_0_f*PA_1_f*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PB_1_f*(PA_0_f*PQ_f[a1] + PA_1_f*PQ_f[a0]))

                        -PA_0_f*PA_1_f*(PB_0_f*(PQ_f[d0]*delta_f[b1][d1] + PQ_f[d1]*delta_f[b1][d0]) + PB_1_f*(PQ_f[d0]*delta_f[b0][d1] + PQ_f[d1]*delta_f[b0][d0])) - PB_0_f*PB_1_f*(PA_0_f*(PQ_f[d0]*delta_f[a1][d1] + PQ_f[d1]*delta_f[a1][d0]) + PA_1_f*(PQ_f[d0]*delta_f[a0][d1] + PQ_f[d1]*delta_f[a0][d0]))

                        +PQ_f[d0]*PQ_f[d1]*(PA_0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) + PA_1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PB_0_f*PB_1_f*delta_f[a0][a1])
                    )

                    + 0.5f * S2_f * S2_f * inv_S1_f * inv_S4_f * inv_S4_f * (
                        +QD_0_f*QD_1_f*delta_f[b0][b1]*(-PA_0_f*PQ_f[a1] + PQ_f[a0]*(-PA_1_f + PQ_f[a1]))

                        +QD_0_f*QD_1_f*(PQ_f[a0]*delta_f[a1][b0]*(-PB_1_f + PQ_f[b1]) + PQ_f[a0]*delta_f[a1][b1]*(-PB_0_f + PQ_f[b0]) + PQ_f[a1]*delta_f[a0][b0]*(-PB_1_f + PQ_f[b1]) + PQ_f[a1]*delta_f[a0][b1]*(-PB_0_f + PQ_f[b0]) + PQ_f[b0]*PQ_f[b1]*delta_f[a0][a1] - PQ_f[b0]*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) - PQ_f[b1]*(PA_0_f*delta_f[a1][b0] + PA_1_f*delta_f[a0][b0] + PB_0_f*delta_f[a0][a1]))
                    )

                    + 0.5f * S2_f * inv_S4_f * inv_S4_f * (
                        +QD_0_f*delta_f[b1][d1]*(PA_0_f*(PA_1_f*PQ_f[b0] + PB_0_f*PQ_f[a1]) + PA_1_f*PB_0_f*PQ_f[a0])

                        +PB_1_f*(PA_0_f*(PQ_f[a1]*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0]) + PQ_f[b0]*(QD_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][d0])) + PQ_f[a0]*(PA_1_f*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0]) + PB_0_f*(QD_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][d0]))) + QD_0_f*(PA_0_f*PQ_f[b1]*(PA_1_f*delta_f[b0][d1] + PB_0_f*delta_f[a1][d1]) + delta_f[a0][d1]*(PA_1_f*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PB_1_f*PQ_f[a1])) + QD_1_f*(PA_0_f*(PA_1_f*(PQ_f[b0]*delta_f[b1][d0] + PQ_f[b1]*delta_f[b0][d0]) + PB_0_f*(PQ_f[a1]*delta_f[b1][d0] + PQ_f[b1]*delta_f[a1][d0])) + PA_1_f*PB_0_f*(PQ_f[a0]*delta_f[b1][d0] + PQ_f[b1]*delta_f[a0][d0]) + PB_1_f*delta_f[a0][d0]*(PA_1_f*PQ_f[b0] + PB_0_f*PQ_f[a1]))

                        +(PQ_f[d0]*QD_1_f + PQ_f[d1]*QD_0_f)*(PA_0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) - PA_0_f*(PQ_f[a1]*delta_f[b0][b1] + PQ_f[b0]*delta_f[a1][b1] + PQ_f[b1]*delta_f[a1][b0]) + PA_1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) - PA_1_f*(PQ_f[a0]*delta_f[b0][b1] + PQ_f[b0]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][b0]) + PB_0_f*PB_1_f*delta_f[a0][a1] - PB_0_f*(PQ_f[a0]*delta_f[a1][b1] + PQ_f[a1]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][a1]) - PB_1_f*(PQ_f[a0]*delta_f[a1][b0] + PQ_f[a1]*delta_f[a0][b0] + PQ_f[b0]*delta_f[a0][a1]))

                        +delta_f[d0][d1]*(PA_0_f*PQ_f[b0]*(PA_1_f*PQ_f[b1] + PB_1_f*PQ_f[a1]) + PA_1_f*PQ_f[a0]*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PQ_f[a1]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))
                    )

                    + S1_f * S1_f * inv_S4_f * inv_S4_f * (
                        +PA_0_f*PA_1_f*PB_0_f*PB_1_f*PQ_f[d0]*PQ_f[d1]
                    )

                    + S1_f * S2_f * inv_S4_f * inv_S4_f * (
                        -(PQ_f[d0]*QD_1_f + PQ_f[d1]*QD_0_f)*(PA_0_f*PA_1_f*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PB_1_f*(PA_0_f*PQ_f[a1] + PA_1_f*PQ_f[a0]))
                    )

                    + S2_f * S2_f * inv_S4_f * inv_S4_f * (
                        +QD_0_f*QD_1_f*(PA_0_f*PQ_f[b0]*(PA_1_f*PQ_f[b1] + PB_1_f*PQ_f[a1]) + PA_1_f*PQ_f[a0]*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PQ_f[a1]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))
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
                        +PQ_f[a0]*PQ_f[a1]*delta_f[b0][b1]*delta_f[d0][d1]

                        +delta_f[d0][d1]*(PQ_f[b0]*(PQ_f[a0]*delta_f[a1][b1] + PQ_f[a1]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][a1]) + PQ_f[b1]*(PQ_f[a0]*delta_f[a1][b0] + PQ_f[a1]*delta_f[a0][b0]))

                        +QD_0_f*(PQ_f[a0]*(delta_f[a1][b0]*delta_f[b1][d1] + delta_f[a1][b1]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][b1]) + PQ_f[a1]*(delta_f[a0][b0]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][b1]) + PQ_f[b0]*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + PQ_f[b1]*(delta_f[a0][a1]*delta_f[b0][d1] + delta_f[a0][b0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b0])) + QD_1_f*(PQ_f[a0]*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + PQ_f[a1]*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + PQ_f[b0]*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + PQ_f[b1]*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0]))

                        +(PQ_f[d0]*QD_1_f + PQ_f[d1]*QD_0_f)*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.25f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +delta_f[a1][b1]*delta_f[d0][d1]*(PB_0_f*PQ_f[a0] + PQ_f[b0]*(PA_0_f - PQ_f[a0]))

                        +(PA_0_f*PQ_f[a1] + PA_1_f*PQ_f[a0])*(delta_f[b0][d0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][d0]) + (PA_0_f*PQ_f[b0] + PB_0_f*PQ_f[a0])*(delta_f[a1][d0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b1][d0]) + (PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0])*(delta_f[a1][d0]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][d0]) + (PA_1_f*PQ_f[b0] + PB_0_f*PQ_f[a1])*(delta_f[a0][d0]*delta_f[b1][d1] + delta_f[a0][d1]*delta_f[b1][d0]) + (PA_1_f*PQ_f[b1] + PB_1_f*PQ_f[a1])*(delta_f[a0][d0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][d0]) + (PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0])*(delta_f[a0][d0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][d0])

                        +delta_f[d0][d1]*(PA_1_f*(PQ_f[a0]*delta_f[b0][b1] + PQ_f[b0]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][b0]) + PB_1_f*(PQ_f[a0]*delta_f[a1][b0] + PQ_f[b0]*delta_f[a0][a1]) + PQ_f[a1]*(PA_0_f*delta_f[b0][b1] + PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0] - PQ_f[a0]*delta_f[b0][b1] - PQ_f[b0]*delta_f[a0][b1] - PQ_f[b1]*delta_f[a0][b0]) + PQ_f[b1]*(delta_f[a0][a1]*(PB_0_f - PQ_f[b0]) + delta_f[a1][b0]*(PA_0_f - PQ_f[a0])))

                        +(PA_0_f*PQ_f[d0] - PQ_f[a0]*PQ_f[d0])*(delta_f[a1][b0]*delta_f[b1][d1] + delta_f[a1][b1]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][b1]) + (PA_0_f*PQ_f[d1] - PQ_f[a0]*PQ_f[d1])*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + (PA_1_f*PQ_f[d0] - PQ_f[a1]*PQ_f[d0])*(delta_f[a0][b0]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][b1]) + (PA_1_f*PQ_f[d1] - PQ_f[a1]*PQ_f[d1])*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + (PB_0_f*PQ_f[d0] - PQ_f[b0]*PQ_f[d0])*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + (PB_0_f*PQ_f[d1] - PQ_f[b0]*PQ_f[d1])*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + (PB_1_f*PQ_f[d0] - PQ_f[b1]*PQ_f[d0])*(delta_f[a0][a1]*delta_f[b0][d1] + delta_f[a0][b0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b0]) + (PB_1_f*PQ_f[d1] - PQ_f[b1]*PQ_f[d1])*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0])

                        -2*PQ_f[d0]*PQ_f[d1]*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.5f * S1_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        -delta_f[d0][d1]*(PA_0_f*PQ_f[b0]*(PA_1_f*PQ_f[b1] + PB_1_f*PQ_f[a1]) + PA_1_f*PQ_f[a0]*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PQ_f[a1]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))

                        -PA_0_f*(PA_1_f*(PQ_f[b0]*(PQ_f[d0]*delta_f[b1][d1] + PQ_f[d1]*delta_f[b1][d0]) + PQ_f[b1]*(PQ_f[d0]*delta_f[b0][d1] + PQ_f[d1]*delta_f[b0][d0])) + PB_0_f*(PQ_f[a1]*(PQ_f[d0]*delta_f[b1][d1] + PQ_f[d1]*delta_f[b1][d0]) + PQ_f[b1]*(PQ_f[d0]*delta_f[a1][d1] + PQ_f[d1]*delta_f[a1][d0])) + PB_1_f*(PQ_f[a1]*(PQ_f[d0]*delta_f[b0][d1] + PQ_f[d1]*delta_f[b0][d0]) + PQ_f[b0]*(PQ_f[d0]*delta_f[a1][d1] + PQ_f[d1]*delta_f[a1][d0]))) - PA_1_f*(PB_0_f*(PQ_f[a0]*(PQ_f[d0]*delta_f[b1][d1] + PQ_f[d1]*delta_f[b1][d0]) + PQ_f[b1]*(PQ_f[d0]*delta_f[a0][d1] + PQ_f[d1]*delta_f[a0][d0])) + PB_1_f*(PQ_f[a0]*(PQ_f[d0]*delta_f[b0][d1] + PQ_f[d1]*delta_f[b0][d0]) + PQ_f[b0]*(PQ_f[d0]*delta_f[a0][d1] + PQ_f[d1]*delta_f[a0][d0]))) - PB_0_f*PB_1_f*(PQ_f[a0]*(PQ_f[d0]*delta_f[a1][d1] + PQ_f[d1]*delta_f[a1][d0]) + PQ_f[a1]*(PQ_f[d0]*delta_f[a0][d1] + PQ_f[d1]*delta_f[a0][d0]))

                        +PQ_f[d0]*PQ_f[d1]*(PA_0_f*(-PA_1_f*delta_f[b0][b1] - PB_0_f*delta_f[a1][b1] - PB_1_f*delta_f[a1][b0] + PQ_f[a1]*delta_f[b0][b1] + PQ_f[b0]*delta_f[a1][b1] + PQ_f[b1]*delta_f[a1][b0]) + PA_1_f*(PQ_f[a0]*delta_f[b0][b1] + delta_f[a0][b0]*(-PB_1_f + PQ_f[b1]) + delta_f[a0][b1]*(-PB_0_f + PQ_f[b0])) + PB_0_f*(PQ_f[a0]*delta_f[a1][b1] + PQ_f[a1]*delta_f[a0][b1] + delta_f[a0][a1]*(-PB_1_f + PQ_f[b1])) + PB_1_f*(PQ_f[a0]*delta_f[a1][b0] + PQ_f[a1]*delta_f[a0][b0] + PQ_f[b0]*delta_f[a0][a1]))
                    )

                    + (-0.5f) * S2_f * S2_f * S2_f * inv_S1_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[a0]*PQ_f[a1]*QD_0_f*QD_1_f*delta_f[b0][b1]

                        +QD_0_f*QD_1_f*(PQ_f[b0]*(PQ_f[a0]*delta_f[a1][b1] + PQ_f[a1]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][a1]) + PQ_f[b1]*(PQ_f[a0]*delta_f[a1][b0] + PQ_f[a1]*delta_f[a0][b0]))
                    )

                    + 0.5f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +QD_0_f*delta_f[b1][d1]*(PA_1_f*PQ_f[a0]*PQ_f[b0] + PQ_f[a1]*(PA_0_f*PQ_f[b0] + PB_0_f*PQ_f[a0]))

                        +PB_1_f*PQ_f[a0]*(PQ_f[a1]*(QD_0_f*delta_f[b0][d1] + QD_1_f*delta_f[b0][d0]) + PQ_f[b0]*(QD_0_f*delta_f[a1][d1] + QD_1_f*delta_f[a1][d0])) + QD_0_f*(PB_1_f*PQ_f[a1]*PQ_f[b0]*delta_f[a0][d1] + PQ_f[b1]*(PA_0_f*(PQ_f[a1]*delta_f[b0][d1] + PQ_f[b0]*delta_f[a1][d1]) + PA_1_f*(PQ_f[a0]*delta_f[b0][d1] + PQ_f[b0]*delta_f[a0][d1]) + PB_0_f*(PQ_f[a0]*delta_f[a1][d1] + PQ_f[a1]*delta_f[a0][d1]))) + QD_1_f*(PQ_f[a0]*(PA_1_f*(PQ_f[b0]*delta_f[b1][d0] + PQ_f[b1]*delta_f[b0][d0]) + PB_0_f*(PQ_f[a1]*delta_f[b1][d0] + PQ_f[b1]*delta_f[a1][d0])) + PQ_f[a1]*(PA_0_f*(PQ_f[b0]*delta_f[b1][d0] + PQ_f[b1]*delta_f[b0][d0]) + delta_f[a0][d0]*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0])) + PQ_f[b0]*PQ_f[b1]*(PA_0_f*delta_f[a1][d0] + PA_1_f*delta_f[a0][d0]))

                        +(PQ_f[d0]*QD_1_f + PQ_f[d1]*QD_0_f)*(PA_0_f*(PQ_f[a1]*delta_f[b0][b1] + PQ_f[b0]*delta_f[a1][b1] + PQ_f[b1]*delta_f[a1][b0]) + PQ_f[a0]*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0] - PQ_f[a1]*delta_f[b0][b1] - PQ_f[b0]*delta_f[a1][b1] - PQ_f[b1]*delta_f[a1][b0]) + PQ_f[a1]*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PQ_f[b0]*(PB_1_f*delta_f[a0][a1] + delta_f[a0][b1]*(PA_1_f - PQ_f[a1])) + PQ_f[b1]*(delta_f[a0][a1]*(PB_0_f - PQ_f[b0]) + delta_f[a0][b0]*(PA_1_f - PQ_f[a1])))

                        +delta_f[d0][d1]*(PQ_f[a0]*PQ_f[b1]*(PA_1_f*PQ_f[b0] + PB_0_f*PQ_f[a1]) + PQ_f[a1]*PQ_f[b0]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))
                    )

                    + S1_f * S1_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[d0]*PQ_f[d1]*(PA_0_f*PA_1_f*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PB_1_f*(PA_0_f*PQ_f[a1] + PA_1_f*PQ_f[a0]))
                    )

                    + S1_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        -(PQ_f[d0]*QD_1_f + PQ_f[d1]*QD_0_f)*(PA_0_f*PQ_f[b0]*(PA_1_f*PQ_f[b1] + PB_1_f*PQ_f[a1]) + PA_1_f*PQ_f[a0]*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PQ_f[a1]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))
                    )

                    + S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +QD_0_f*QD_1_f*(PQ_f[a0]*PQ_f[b1]*(PA_1_f*PQ_f[b0] + PB_0_f*PQ_f[a1]) + PQ_f[a1]*PQ_f[b0]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))
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
        double J_ij = 0.0;

        for (uint32_t n = 0; n < TILE_DIM_LARGE; n++)
        {
            J_ij += ERIs[n];
        }

        mat_J[ij] += J_ij;
    }
}

// DDSD altB DDSD2 BASE
__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDSD_altB2(double*         mat_J,
                       const double*   s_prim_info,
                       const uint32_t  s_prim_count,
                       const double*   d_prim_info,
                       const uint32_t  d_prim_count,
                       const double*   sd_mat_D,
                       const double*   dd_mat_Q_local,
                       const double*   sd_mat_Q,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const double*   dd_pair_data_local,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* sd_first_inds,
                       const uint32_t* sd_second_inds,
                       const double*   sd_pair_data,
                       const uint32_t  sd_prim_pair_count,
                       const double*   boys_func_table,
                       const double*   boys_func_ft,
                       const double    eri_threshold)
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

    for (uint32_t m = 0; m < (sd_prim_pair_count + TILE_DIM_LARGE - 1) / TILE_DIM_LARGE; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= sd_prim_pair_count) || (fabs(dd_mat_Q_local[ij] * sd_mat_Q[kl] * sd_mat_D[kl]) <= eri_threshold))
        {
            break;
        }

        const auto k = sd_first_inds[kl];
        const auto l = sd_second_inds[kl];

        const auto a_k = s_prim_info[k + s_prim_count * 0];

        const double r_k[3] = {s_prim_info[k + s_prim_count * 2],
                               s_prim_info[k + s_prim_count * 3],
                               s_prim_info[k + s_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = sd_pair_data[kl];

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

        double F6_t[7];

        gpu::computeBoysFunction(F6_t, S1 * S2 * inv_S4 * r2_PQ, 6, boys_func_table, boys_func_ft);

        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                + F6_t[4] * (

                    0.5 * S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        -delta[d0][d1]*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))

                        -PQ[a0]*(PA_1*(PQ[b0]*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PQ[b1]*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0])) + PB_0*(PQ[a1]*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PQ[b1]*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0])) + PB_1*(PQ[a1]*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0]) + PQ[b0]*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0]))) - PQ[a1]*(PA_0*(PQ[b0]*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PQ[b1]*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0])) + (PB_0*PQ[b1] + PB_1*PQ[b0])*(PQ[d0]*delta[a0][d1] + PQ[d1]*delta[a0][d0])) - PQ[b0]*PQ[b1]*(PA_0*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0]) + PA_1*(PQ[d0]*delta[a0][d1] + PQ[d1]*delta[a0][d0]))

                        +PQ[d0]*PQ[d1]*(-PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) - PQ[a0]*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PQ[a0]*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) - PQ[a1]*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PQ[a1]*(PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PQ[b0]*PQ[b1]*delta[a0][a1] - PQ[b0]*(PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) - PQ[b1]*(PA_1*delta[a0][b0] + PB_0*delta[a0][a1]))
                    )

                    + 0.5 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*QD_0*delta[b1][d1]

                        +PQ[a0]*PQ[a1]*QD_1*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0]) + PQ[b1]*(PQ[a0]*QD_0*(PQ[a1]*delta[b0][d1] + PQ[b0]*delta[a1][d1]) + PQ[b0]*(PQ[a1]*QD_0*delta[a0][d1] + QD_1*(PQ[a0]*delta[a1][d0] + PQ[a1]*delta[a0][d0])))

                        +(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(PQ[a0]*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PQ[a1]*(PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PQ[b0]*PQ[b1]*delta[a0][a1])

                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*delta[d0][d1]
                    )

                    + S1 * S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[d0]*PQ[d1]*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S1 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        -(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S2 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*QD_0*QD_1
                    )

                    + 0.25 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*(delta[b0][b1]*delta[d0][d1] + delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0])

                        +PQ[b0]*(PQ[a0]*(delta[a1][b1]*delta[d0][d1] + delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + PQ[a1]*(delta[a0][b1]*delta[d0][d1] + delta[a0][d0]*delta[b1][d1] + delta[a0][d1]*delta[b1][d0]) + PQ[b1]*(delta[a0][a1]*delta[d0][d1] + delta[a0][d0]*delta[a1][d1] + delta[a0][d1]*delta[a1][d0]) + PQ[d0]*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + PQ[d1]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1])) + PQ[b1]*(PQ[a0]*(delta[a1][b0]*delta[d0][d1] + delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + PQ[a1]*(delta[a0][b0]*delta[d0][d1] + delta[a0][d0]*delta[b0][d1] + delta[a0][d1]*delta[b0][d0]) + PQ[d0]*(delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0]) + PQ[d1]*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])) + PQ[d0]*(PQ[a0]*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1]) + PQ[d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])) + PQ[d1]*(PQ[a0]*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]))
                    )

                )

                + F6_t[5] * (

                    (-0.5) * S1 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*delta[d0][d1]

                        +PQ[d0]*(PQ[a0]*(PQ[a1]*(PQ[b0]*delta[b1][d1] + PQ[b1]*delta[b0][d1] + PQ[d1]*delta[b0][b1]) + PQ[b1]*PQ[d1]*delta[a1][b0]) + PQ[b0]*PQ[b1]*(PQ[a0]*delta[a1][d1] + PQ[a1]*delta[a0][d1] + PQ[d1]*delta[a0][a1])) + PQ[d1]*(PQ[a1]*PQ[b1]*(PQ[a0]*delta[b0][d0] + PQ[b0]*delta[a0][d0] + PQ[d0]*delta[a0][b0]) + PQ[b0]*(PQ[a0]*(PQ[a1]*delta[b1][d0] + PQ[b1]*delta[a1][d0] + PQ[d0]*delta[a1][b1]) + PQ[a1]*PQ[d0]*delta[a0][b1]))
                    )

                    + S1 * S1 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[d0]*PQ[d1]*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S1 * S2 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*(-PQ[d0]*QD_1 - PQ[d1]*QD_0)
                    )

                )

                + F6_t[6] * (

                    S1 * S1 * S2 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*PQ[d0]*PQ[d1]
                    )

                )

            );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        ERIs[threadIdx.y] += eri_ijkl * sd_mat_D[kl] * 2.0;
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

// DDSD altB DDSD2 FP64
__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDSD_altB2_FP64(double*         mat_J,
                       const double*   s_prim_info,
                       const uint32_t  s_prim_count,
                       const double*   d_prim_info,
                       const uint32_t  d_prim_count,
                       const double*   sd_mat_D,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const double*   dd_pair_data_local,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* sd_first_inds,
                       const uint32_t* sd_second_inds,
                       const double*   sd_pair_data,
                       const uint32_t  sd_prim_pair_count,
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

        if ((ij >= dd_prim_pair_count_local) || (kl >= sd_prim_pair_count))
        {
            break;
        }

        const auto k = sd_first_inds[kl];
        const auto l = sd_second_inds[kl];

        const auto a_k = s_prim_info[k + s_prim_count * 0];

        const double r_k[3] = {s_prim_info[k + s_prim_count * 2],
                               s_prim_info[k + s_prim_count * 3],
                               s_prim_info[k + s_prim_count * 4]};

        const auto a_l = d_prim_info[l / 6 + d_prim_count * 0];

        const double r_l[3] = {d_prim_info[l / 6 + d_prim_count * 2],
                               d_prim_info[l / 6 + d_prim_count * 3],
                               d_prim_info[l / 6 + d_prim_count * 4]};

        const auto S_kl_00 = sd_pair_data[kl];

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

        double F6_t[7];

        gpu::computeBoysFunction(F6_t, S1 * S2 * inv_S4 * r2_PQ, 6, boys_func_table, boys_func_ft);

        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);
        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                + F6_t[4] * (

                    0.5 * S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        -delta[d0][d1]*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))

                        -PQ[a0]*(PA_1*(PQ[b0]*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PQ[b1]*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0])) + PB_0*(PQ[a1]*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PQ[b1]*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0])) + PB_1*(PQ[a1]*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0]) + PQ[b0]*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0]))) - PQ[a1]*(PA_0*(PQ[b0]*(PQ[d0]*delta[b1][d1] + PQ[d1]*delta[b1][d0]) + PQ[b1]*(PQ[d0]*delta[b0][d1] + PQ[d1]*delta[b0][d0])) + (PB_0*PQ[b1] + PB_1*PQ[b0])*(PQ[d0]*delta[a0][d1] + PQ[d1]*delta[a0][d0])) - PQ[b0]*PQ[b1]*(PA_0*(PQ[d0]*delta[a1][d1] + PQ[d1]*delta[a1][d0]) + PA_1*(PQ[d0]*delta[a0][d1] + PQ[d1]*delta[a0][d0]))

                        +PQ[d0]*PQ[d1]*(-PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) - PQ[a0]*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PQ[a0]*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) - PQ[a1]*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PQ[a1]*(PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PQ[b0]*PQ[b1]*delta[a0][a1] - PQ[b0]*(PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) - PQ[b1]*(PA_1*delta[a0][b0] + PB_0*delta[a0][a1]))
                    )

                    + 0.5 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*QD_0*delta[b1][d1]

                        +PQ[a0]*PQ[a1]*QD_1*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0]) + PQ[b1]*(PQ[a0]*QD_0*(PQ[a1]*delta[b0][d1] + PQ[b0]*delta[a1][d1]) + PQ[b0]*(PQ[a1]*QD_0*delta[a0][d1] + QD_1*(PQ[a0]*delta[a1][d0] + PQ[a1]*delta[a0][d0])))

                        +(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(PQ[a0]*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PQ[a1]*(PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PQ[b0]*PQ[b1]*delta[a0][a1])

                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*delta[d0][d1]
                    )

                    + S1 * S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[d0]*PQ[d1]*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S1 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        -(PQ[d0]*QD_1 + PQ[d1]*QD_0)*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S2 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*QD_0*QD_1
                    )

                    + 0.25 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*(delta[b0][b1]*delta[d0][d1] + delta[b0][d0]*delta[b1][d1] + delta[b0][d1]*delta[b1][d0])

                        +PQ[b0]*(PQ[a0]*(delta[a1][b1]*delta[d0][d1] + delta[a1][d0]*delta[b1][d1] + delta[a1][d1]*delta[b1][d0]) + PQ[a1]*(delta[a0][b1]*delta[d0][d1] + delta[a0][d0]*delta[b1][d1] + delta[a0][d1]*delta[b1][d0]) + PQ[b1]*(delta[a0][a1]*delta[d0][d1] + delta[a0][d0]*delta[a1][d1] + delta[a0][d1]*delta[a1][d0]) + PQ[d0]*(delta[a0][a1]*delta[b1][d1] + delta[a0][b1]*delta[a1][d1] + delta[a0][d1]*delta[a1][b1]) + PQ[d1]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1])) + PQ[b1]*(PQ[a0]*(delta[a1][b0]*delta[d0][d1] + delta[a1][d0]*delta[b0][d1] + delta[a1][d1]*delta[b0][d0]) + PQ[a1]*(delta[a0][b0]*delta[d0][d1] + delta[a0][d0]*delta[b0][d1] + delta[a0][d1]*delta[b0][d0]) + PQ[d0]*(delta[a0][a1]*delta[b0][d1] + delta[a0][b0]*delta[a1][d1] + delta[a0][d1]*delta[a1][b0]) + PQ[d1]*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])) + PQ[d0]*(PQ[a0]*(delta[a1][b0]*delta[b1][d1] + delta[a1][b1]*delta[b0][d1] + delta[a1][d1]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][d1] + delta[a0][b1]*delta[b0][d1] + delta[a0][d1]*delta[b0][b1]) + PQ[d1]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])) + PQ[d1]*(PQ[a0]*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]))
                    )

                )

                + F6_t[5] * (

                    (-0.5) * S1 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*delta[d0][d1]

                        +PQ[d0]*(PQ[a0]*(PQ[a1]*(PQ[b0]*delta[b1][d1] + PQ[b1]*delta[b0][d1] + PQ[d1]*delta[b0][b1]) + PQ[b1]*PQ[d1]*delta[a1][b0]) + PQ[b0]*PQ[b1]*(PQ[a0]*delta[a1][d1] + PQ[a1]*delta[a0][d1] + PQ[d1]*delta[a0][a1])) + PQ[d1]*(PQ[a1]*PQ[b1]*(PQ[a0]*delta[b0][d0] + PQ[b0]*delta[a0][d0] + PQ[d0]*delta[a0][b0]) + PQ[b0]*(PQ[a0]*(PQ[a1]*delta[b1][d0] + PQ[b1]*delta[a1][d0] + PQ[d0]*delta[a1][b1]) + PQ[a1]*PQ[d0]*delta[a0][b1]))
                    )

                    + S1 * S1 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[d0]*PQ[d1]*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S1 * S2 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*(-PQ[d0]*QD_1 - PQ[d1]*QD_0)
                    )

                )

                + F6_t[6] * (

                    S1 * S1 * S2 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*PQ[d0]*PQ[d1]
                    )

                )

            );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        ERIs[threadIdx.y] += eri_ijkl * sd_mat_D[kl] * 2.0;
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

// DDSD altB DDSD2 FP32
__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDSD_altB2_FP32(double*         mat_J,
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

        const float r_k_f[3] = {s_prim_info_f[k + s_prim_count * 2],
                               s_prim_info_f[k + s_prim_count * 3],
                               s_prim_info_f[k + s_prim_count * 4]};

        const auto a_l_f = d_prim_info_f[l / 6 + d_prim_count * 0];

        const float r_l_f[3] = {d_prim_info_f[l / 6 + d_prim_count * 2],
                               d_prim_info_f[l / 6 + d_prim_count * 3],
                               d_prim_info_f[l / 6 + d_prim_count * 4]};

        const auto S_kl_00_f = sd_pair_data_f[kl];

        const auto d0 = d_cart_inds[l % 6][0];
        const auto d1 = d_cart_inds[l % 6][1];

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2_f = a_k_f + a_l_f;

        const auto inv_S2_f = 1.0f / S2_f;
        const auto inv_S4_f = 1.0f / (S1_f + S2_f);

        const float PQ_f[3] = {(a_k_f * r_k_f[0] + a_l_f * r_l_f[0]) * inv_S2_f - (a_i_f * r_i_f[0] + a_j_f * r_j_f[0]) * inv_S1_f,
                              (a_k_f * r_k_f[1] + a_l_f * r_l_f[1]) * inv_S2_f - (a_i_f * r_i_f[1] + a_j_f * r_j_f[1]) * inv_S1_f,
                              (a_k_f * r_k_f[2] + a_l_f * r_l_f[2]) * inv_S2_f - (a_i_f * r_i_f[2] + a_j_f * r_j_f[2]) * inv_S1_f};

        const auto r2_PQ_f = PQ_f[0] * PQ_f[0] + PQ_f[1] * PQ_f[1] + PQ_f[2] * PQ_f[2];

        const auto Lambda_f = sqrtf(4.0f * S1_f * S2_f * MATH_CONST_INV_PI_F * inv_S4_f);

        float F6_t_f[7];

        gpu::computeBoysFunction_f(F6_t_f, S1_f * S2_f * inv_S4_f * r2_PQ_f, 6, boys_func_table_f, boys_func_ft_f);

        const auto QD_0_f = (-a_k_f * inv_S2_f) * (r_l_f[d0] - r_k_f[d0]);
        const auto QD_1_f = (-a_k_f * inv_S2_f) * (r_l_f[d1] - r_k_f[d1]);

        const float eri_ijkl_f = Lambda_f * S_ij_00_f * S_kl_00_f * (

                + F6_t_f[4] * (

                    0.5f * S1_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        -delta_f[d0][d1]*(PQ_f[a0]*PQ_f[b1]*(PA_1_f*PQ_f[b0] + PB_0_f*PQ_f[a1]) + PQ_f[a1]*PQ_f[b0]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))

                        -PQ_f[a0]*(PA_1_f*(PQ_f[b0]*(PQ_f[d0]*delta_f[b1][d1] + PQ_f[d1]*delta_f[b1][d0]) + PQ_f[b1]*(PQ_f[d0]*delta_f[b0][d1] + PQ_f[d1]*delta_f[b0][d0])) + PB_0_f*(PQ_f[a1]*(PQ_f[d0]*delta_f[b1][d1] + PQ_f[d1]*delta_f[b1][d0]) + PQ_f[b1]*(PQ_f[d0]*delta_f[a1][d1] + PQ_f[d1]*delta_f[a1][d0])) + PB_1_f*(PQ_f[a1]*(PQ_f[d0]*delta_f[b0][d1] + PQ_f[d1]*delta_f[b0][d0]) + PQ_f[b0]*(PQ_f[d0]*delta_f[a1][d1] + PQ_f[d1]*delta_f[a1][d0]))) - PQ_f[a1]*(PA_0_f*(PQ_f[b0]*(PQ_f[d0]*delta_f[b1][d1] + PQ_f[d1]*delta_f[b1][d0]) + PQ_f[b1]*(PQ_f[d0]*delta_f[b0][d1] + PQ_f[d1]*delta_f[b0][d0])) + (PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0])*(PQ_f[d0]*delta_f[a0][d1] + PQ_f[d1]*delta_f[a0][d0])) - PQ_f[b0]*PQ_f[b1]*(PA_0_f*(PQ_f[d0]*delta_f[a1][d1] + PQ_f[d1]*delta_f[a1][d0]) + PA_1_f*(PQ_f[d0]*delta_f[a0][d1] + PQ_f[d1]*delta_f[a0][d0]))

                        +PQ_f[d0]*PQ_f[d1]*(-PA_0_f*(PQ_f[a1]*delta_f[b0][b1] + PQ_f[b0]*delta_f[a1][b1] + PQ_f[b1]*delta_f[a1][b0]) - PQ_f[a0]*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) + PQ_f[a0]*(PQ_f[a1]*delta_f[b0][b1] + PQ_f[b0]*delta_f[a1][b1] + PQ_f[b1]*delta_f[a1][b0]) - PQ_f[a1]*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PQ_f[a1]*(PQ_f[b0]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][b0]) + PQ_f[b0]*PQ_f[b1]*delta_f[a0][a1] - PQ_f[b0]*(PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) - PQ_f[b1]*(PA_1_f*delta_f[a0][b0] + PB_0_f*delta_f[a0][a1]))
                    )

                    + 0.5f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[a0]*PQ_f[a1]*PQ_f[b0]*QD_0_f*delta_f[b1][d1]

                        +PQ_f[a0]*PQ_f[a1]*QD_1_f*(PQ_f[b0]*delta_f[b1][d0] + PQ_f[b1]*delta_f[b0][d0]) + PQ_f[b1]*(PQ_f[a0]*QD_0_f*(PQ_f[a1]*delta_f[b0][d1] + PQ_f[b0]*delta_f[a1][d1]) + PQ_f[b0]*(PQ_f[a1]*QD_0_f*delta_f[a0][d1] + QD_1_f*(PQ_f[a0]*delta_f[a1][d0] + PQ_f[a1]*delta_f[a0][d0])))

                        +(PQ_f[d0]*QD_1_f + PQ_f[d1]*QD_0_f)*(PQ_f[a0]*(PQ_f[a1]*delta_f[b0][b1] + PQ_f[b0]*delta_f[a1][b1] + PQ_f[b1]*delta_f[a1][b0]) + PQ_f[a1]*(PQ_f[b0]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][b0]) + PQ_f[b0]*PQ_f[b1]*delta_f[a0][a1])

                        +PQ_f[a0]*PQ_f[a1]*PQ_f[b0]*PQ_f[b1]*delta_f[d0][d1]
                    )

                    + S1_f * S1_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[d0]*PQ_f[d1]*(PA_0_f*PQ_f[b0]*(PA_1_f*PQ_f[b1] + PB_1_f*PQ_f[a1]) + PA_1_f*PQ_f[a0]*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PQ_f[a1]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))
                    )

                    + S1_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        -(PQ_f[d0]*QD_1_f + PQ_f[d1]*QD_0_f)*(PQ_f[a0]*PQ_f[b1]*(PA_1_f*PQ_f[b0] + PB_0_f*PQ_f[a1]) + PQ_f[a1]*PQ_f[b0]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))
                    )

                    + S2_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[a0]*PQ_f[a1]*PQ_f[b0]*PQ_f[b1]*QD_0_f*QD_1_f
                    )

                    + 0.25f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[a0]*PQ_f[a1]*(delta_f[b0][b1]*delta_f[d0][d1] + delta_f[b0][d0]*delta_f[b1][d1] + delta_f[b0][d1]*delta_f[b1][d0])

                        +PQ_f[b0]*(PQ_f[a0]*(delta_f[a1][b1]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[b1][d1] + delta_f[a1][d1]*delta_f[b1][d0]) + PQ_f[a1]*(delta_f[a0][b1]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[b1][d1] + delta_f[a0][d1]*delta_f[b1][d0]) + PQ_f[b1]*(delta_f[a0][a1]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][d0]) + PQ_f[d0]*(delta_f[a0][a1]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b1]) + PQ_f[d1]*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1])) + PQ_f[b1]*(PQ_f[a0]*(delta_f[a1][b0]*delta_f[d0][d1] + delta_f[a1][d0]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][d0]) + PQ_f[a1]*(delta_f[a0][b0]*delta_f[d0][d1] + delta_f[a0][d0]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][d0]) + PQ_f[d0]*(delta_f[a0][a1]*delta_f[b0][d1] + delta_f[a0][b0]*delta_f[a1][d1] + delta_f[a0][d1]*delta_f[a1][b0]) + PQ_f[d1]*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0])) + PQ_f[d0]*(PQ_f[a0]*(delta_f[a1][b0]*delta_f[b1][d1] + delta_f[a1][b1]*delta_f[b0][d1] + delta_f[a1][d1]*delta_f[b0][b1]) + PQ_f[a1]*(delta_f[a0][b0]*delta_f[b1][d1] + delta_f[a0][b1]*delta_f[b0][d1] + delta_f[a0][d1]*delta_f[b0][b1]) + PQ_f[d1]*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])) + PQ_f[d1]*(PQ_f[a0]*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + PQ_f[a1]*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]))
                    )

                )

                + F6_t_f[5] * (

                    (-0.5f) * S1_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[a0]*PQ_f[a1]*PQ_f[b0]*PQ_f[b1]*delta_f[d0][d1]

                        +PQ_f[d0]*(PQ_f[a0]*(PQ_f[a1]*(PQ_f[b0]*delta_f[b1][d1] + PQ_f[b1]*delta_f[b0][d1] + PQ_f[d1]*delta_f[b0][b1]) + PQ_f[b1]*PQ_f[d1]*delta_f[a1][b0]) + PQ_f[b0]*PQ_f[b1]*(PQ_f[a0]*delta_f[a1][d1] + PQ_f[a1]*delta_f[a0][d1] + PQ_f[d1]*delta_f[a0][a1])) + PQ_f[d1]*(PQ_f[a1]*PQ_f[b1]*(PQ_f[a0]*delta_f[b0][d0] + PQ_f[b0]*delta_f[a0][d0] + PQ_f[d0]*delta_f[a0][b0]) + PQ_f[b0]*(PQ_f[a0]*(PQ_f[a1]*delta_f[b1][d0] + PQ_f[b1]*delta_f[a1][d0] + PQ_f[d0]*delta_f[a1][b1]) + PQ_f[a1]*PQ_f[d0]*delta_f[a0][b1]))
                    )

                    + S1_f * S1_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[d0]*PQ_f[d1]*(PQ_f[a0]*PQ_f[b1]*(PA_1_f*PQ_f[b0] + PB_0_f*PQ_f[a1]) + PQ_f[a1]*PQ_f[b0]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))
                    )

                    + S1_f * S2_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[a0]*PQ_f[a1]*PQ_f[b0]*PQ_f[b1]*(-PQ_f[d0]*QD_1_f - PQ_f[d1]*QD_0_f)
                    )

                )

                + F6_t_f[6] * (

                    S1_f * S1_f * S2_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[a0]*PQ_f[a1]*PQ_f[b0]*PQ_f[b1]*PQ_f[d0]*PQ_f[d1]
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
        double J_ij = 0.0;

        for (uint32_t n = 0; n < TILE_DIM_LARGE; n++)
        {
            J_ij += ERIs[n];
        }

        mat_J[ij] += J_ij;
    }
}

// DDPP altA DDPP0 BASE
__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDPP_altA0(double*         mat_J,
                       const double*   p_prim_info,
                       const uint32_t  p_prim_count,
                       const double*   d_prim_info,
                       const uint32_t  d_prim_count,
                       const double*   pp_mat_D,
                       const double*   dd_mat_Q_local,
                       const double*   pp_mat_Q,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const double*   dd_pair_data_local,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* pp_first_inds,
                       const uint32_t* pp_second_inds,
                       const double*   pp_pair_data,
                       const uint32_t  pp_prim_pair_count,
                       const double*   boys_func_table,
                       const double*   boys_func_ft,
                       const double    eri_threshold)
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

    for (uint32_t m = 0; m < (pp_prim_pair_count + TILE_DIM_LARGE - 1) / TILE_DIM_LARGE; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= pp_prim_pair_count) || (fabs(dd_mat_Q_local[ij] * pp_mat_Q[kl] * pp_mat_D[kl]) <= eri_threshold))
        {
            break;
        }

        const auto k = pp_first_inds[kl];
        const auto l = pp_second_inds[kl];

        const auto a_k = p_prim_info[k / 3 + p_prim_count * 0];

        const double r_k[3] = {p_prim_info[k / 3 + p_prim_count * 2],
                               p_prim_info[k / 3 + p_prim_count * 3],
                               p_prim_info[k / 3 + p_prim_count * 4]};

        const auto a_l = p_prim_info[l / 3 + p_prim_count * 0];

        const double r_l[3] = {p_prim_info[l / 3 + p_prim_count * 2],
                               p_prim_info[l / 3 + p_prim_count * 3],
                               p_prim_info[l / 3 + p_prim_count * 4]};

        const auto S_kl_00 = pp_pair_data[kl];

        const auto c0 = k % 3;
        const auto d0 = l % 3;

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F6_t[2];

        gpu::computeBoysFunction(F6_t, S1 * S2 * inv_S4 * r2_PQ, 1, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                + F6_t[0] * (

                    0.25 * inv_S1 * inv_S1 * (
                        +QC_0*QD_0*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * inv_S1 * inv_S2 * (
                        +PA_0*PB_0*delta[a1][b1]*delta[c0][d0]

                        +delta[c0][d0]*(PA_1*(PA_0*delta[b0][b1] + PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_1*(PA_0*delta[a1][b0] + PB_0*delta[a0][a1]))
                    )

                    + 0.5 * inv_S1 * (
                        +PA_0*PA_1*QC_0*QD_0*delta[b0][b1]

                        +QC_0*QD_0*(PB_0*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) + PB_1*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0]))
                    )

                    + 0.5 * inv_S2 * (
                        +PA_0*PA_1*PB_0*PB_1*delta[c0][d0]
                    )

                    + (
                        +PA_0*PA_1*PB_0*PB_1*QC_0*QD_0
                    )

                    + 0.125 * inv_S1 * inv_S1 * inv_S2 * (
                        +delta[c0][d0]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                )

                + F6_t[1] * (

                    (-0.25) * inv_S1 * inv_S1 * inv_S4 * (
                        +delta[c0][d0]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + (-0.125) * inv_S1 * inv_S2 * inv_S4 * (
                        +delta[c0][d0]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + (-0.5) * S2 * inv_S1 * inv_S1 * inv_S4 * (
                        +QC_0*QD_0*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * inv_S1 * inv_S4 * (
                        +delta[b0][b1]*delta[c0][d0]*(PA_0*(-PA_1 + PQ[a1]) + PA_1*PQ[a0])

                        +delta[c0][d0]*(PB_0*(-PA_0*delta[a1][b1] - PA_1*delta[a0][b1] - PB_1*delta[a0][a1] + PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + PQ[b0]*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) + delta[a0][b0]*(PA_1*(-PB_1 + PQ[b1]) + PB_1*PQ[a1]) + delta[a1][b0]*(PA_0*(-PB_1 + PQ[b1]) + PB_1*PQ[a0]))

                        +QC_0*(PA_0*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])) + QD_0*(PA_0*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0]))

                        -(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + (-0.25) * inv_S2 * inv_S4 * (
                        +PA_0*PA_1*delta[b0][b1]*delta[c0][d0]

                        +delta[c0][d0]*(PB_0*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) + PB_1*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0]))
                    )

                    + (-0.5) * S1 * inv_S2 * inv_S4 * (
                        +PA_0*PA_1*PB_0*PB_1*delta[c0][d0]
                    )

                    + 0.5 * S2 * inv_S1 * inv_S4 * (
                        +QC_0*QD_0*delta[b0][b1]*(PA_0*(-PA_1 + PQ[a1]) + PA_1*PQ[a0])

                        +QC_0*QD_0*(PB_0*(-PA_0*delta[a1][b1] - PA_1*delta[a0][b1] - PB_1*delta[a0][a1] + PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + PQ[b0]*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) + delta[a0][b0]*(PA_1*(-PB_1 + PQ[b1]) + PB_1*PQ[a1]) + delta[a1][b0]*(PA_0*(-PB_1 + PQ[b1]) + PB_1*PQ[a0]))
                    )

                    + 0.5 * inv_S4 * (
                        +delta[c0][d0]*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))

                        +PA_0*PA_1*(PB_0*(QC_0*delta[b1][d0] + QD_0*delta[b1][c0]) + PB_1*(QC_0*delta[b0][d0] + QD_0*delta[b0][c0])) + PB_0*PB_1*(PA_0*(QC_0*delta[a1][d0] + QD_0*delta[a1][c0]) + PA_1*(QC_0*delta[a0][d0] + QD_0*delta[a0][c0]))

                        -(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1])
                    )

                    + S1 * inv_S4 * (
                        +PA_0*PA_1*PB_0*PB_1*(-PQ[c0]*QD_0 - PQ[d0]*QC_0)
                    )

                    + S2 * inv_S4 * (
                        +QC_0*QD_0*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))
                    )

                )

            );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * pp_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

// DDPP altA DDPP0 FP64
__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDPP_altA0_FP64(double*         mat_J,
                       const double*   p_prim_info,
                       const uint32_t  p_prim_count,
                       const double*   d_prim_info,
                       const uint32_t  d_prim_count,
                       const double*   pp_mat_D,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const double*   dd_pair_data_local,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* pp_first_inds,
                       const uint32_t* pp_second_inds,
                       const double*   pp_pair_data,
                       const uint32_t  pp_prim_pair_count,
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

        if ((ij >= dd_prim_pair_count_local) || (kl >= pp_prim_pair_count))
        {
            break;
        }

        const auto k = pp_first_inds[kl];
        const auto l = pp_second_inds[kl];

        const auto a_k = p_prim_info[k / 3 + p_prim_count * 0];

        const double r_k[3] = {p_prim_info[k / 3 + p_prim_count * 2],
                               p_prim_info[k / 3 + p_prim_count * 3],
                               p_prim_info[k / 3 + p_prim_count * 4]};

        const auto a_l = p_prim_info[l / 3 + p_prim_count * 0];

        const double r_l[3] = {p_prim_info[l / 3 + p_prim_count * 2],
                               p_prim_info[l / 3 + p_prim_count * 3],
                               p_prim_info[l / 3 + p_prim_count * 4]};

        const auto S_kl_00 = pp_pair_data[kl];

        const auto c0 = k % 3;
        const auto d0 = l % 3;

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F6_t[2];

        gpu::computeBoysFunction(F6_t, S1 * S2 * inv_S4 * r2_PQ, 1, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                + F6_t[0] * (

                    0.25 * inv_S1 * inv_S1 * (
                        +QC_0*QD_0*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * inv_S1 * inv_S2 * (
                        +PA_0*PB_0*delta[a1][b1]*delta[c0][d0]

                        +delta[c0][d0]*(PA_1*(PA_0*delta[b0][b1] + PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_1*(PA_0*delta[a1][b0] + PB_0*delta[a0][a1]))
                    )

                    + 0.5 * inv_S1 * (
                        +PA_0*PA_1*QC_0*QD_0*delta[b0][b1]

                        +QC_0*QD_0*(PB_0*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) + PB_1*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0]))
                    )

                    + 0.5 * inv_S2 * (
                        +PA_0*PA_1*PB_0*PB_1*delta[c0][d0]
                    )

                    + (
                        +PA_0*PA_1*PB_0*PB_1*QC_0*QD_0
                    )

                    + 0.125 * inv_S1 * inv_S1 * inv_S2 * (
                        +delta[c0][d0]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                )

                + F6_t[1] * (

                    (-0.25) * inv_S1 * inv_S1 * inv_S4 * (
                        +delta[c0][d0]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + (-0.125) * inv_S1 * inv_S2 * inv_S4 * (
                        +delta[c0][d0]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + (-0.5) * S2 * inv_S1 * inv_S1 * inv_S4 * (
                        +QC_0*QD_0*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * inv_S1 * inv_S4 * (
                        +delta[b0][b1]*delta[c0][d0]*(PA_0*(-PA_1 + PQ[a1]) + PA_1*PQ[a0])

                        +delta[c0][d0]*(PB_0*(-PA_0*delta[a1][b1] - PA_1*delta[a0][b1] - PB_1*delta[a0][a1] + PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + PQ[b0]*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) + delta[a0][b0]*(PA_1*(-PB_1 + PQ[b1]) + PB_1*PQ[a1]) + delta[a1][b0]*(PA_0*(-PB_1 + PQ[b1]) + PB_1*PQ[a0]))

                        +QC_0*(PA_0*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])) + QD_0*(PA_0*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0]))

                        -(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + (-0.25) * inv_S2 * inv_S4 * (
                        +PA_0*PA_1*delta[b0][b1]*delta[c0][d0]

                        +delta[c0][d0]*(PB_0*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) + PB_1*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0]))
                    )

                    + (-0.5) * S1 * inv_S2 * inv_S4 * (
                        +PA_0*PA_1*PB_0*PB_1*delta[c0][d0]
                    )

                    + 0.5 * S2 * inv_S1 * inv_S4 * (
                        +QC_0*QD_0*delta[b0][b1]*(PA_0*(-PA_1 + PQ[a1]) + PA_1*PQ[a0])

                        +QC_0*QD_0*(PB_0*(-PA_0*delta[a1][b1] - PA_1*delta[a0][b1] - PB_1*delta[a0][a1] + PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + PQ[b0]*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) + delta[a0][b0]*(PA_1*(-PB_1 + PQ[b1]) + PB_1*PQ[a1]) + delta[a1][b0]*(PA_0*(-PB_1 + PQ[b1]) + PB_1*PQ[a0]))
                    )

                    + 0.5 * inv_S4 * (
                        +delta[c0][d0]*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))

                        +PA_0*PA_1*(PB_0*(QC_0*delta[b1][d0] + QD_0*delta[b1][c0]) + PB_1*(QC_0*delta[b0][d0] + QD_0*delta[b0][c0])) + PB_0*PB_1*(PA_0*(QC_0*delta[a1][d0] + QD_0*delta[a1][c0]) + PA_1*(QC_0*delta[a0][d0] + QD_0*delta[a0][c0]))

                        -(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1])
                    )

                    + S1 * inv_S4 * (
                        +PA_0*PA_1*PB_0*PB_1*(-PQ[c0]*QD_0 - PQ[d0]*QC_0)
                    )

                    + S2 * inv_S4 * (
                        +QC_0*QD_0*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))
                    )

                )

            );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * pp_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

// DDPP altA DDPP0 FP32
__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDPP_altA0_FP32(double*         mat_J,
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

        const float r_k_f[3] = {p_prim_info_f[k / 3 + p_prim_count * 2],
                               p_prim_info_f[k / 3 + p_prim_count * 3],
                               p_prim_info_f[k / 3 + p_prim_count * 4]};

        const auto a_l_f = p_prim_info_f[l / 3 + p_prim_count * 0];

        const float r_l_f[3] = {p_prim_info_f[l / 3 + p_prim_count * 2],
                               p_prim_info_f[l / 3 + p_prim_count * 3],
                               p_prim_info_f[l / 3 + p_prim_count * 4]};

        const auto S_kl_00_f = pp_pair_data_f[kl];

        const auto c0 = k % 3;
        const auto d0 = l % 3;

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2_f = a_k_f + a_l_f;

        const auto inv_S2_f = 1.0f / S2_f;
        const auto inv_S4_f = 1.0f / (S1_f + S2_f);

        const float PQ_f[3] = {(a_k_f * r_k_f[0] + a_l_f * r_l_f[0]) * inv_S2_f - (a_i_f * r_i_f[0] + a_j_f * r_j_f[0]) * inv_S1_f,
                              (a_k_f * r_k_f[1] + a_l_f * r_l_f[1]) * inv_S2_f - (a_i_f * r_i_f[1] + a_j_f * r_j_f[1]) * inv_S1_f,
                              (a_k_f * r_k_f[2] + a_l_f * r_l_f[2]) * inv_S2_f - (a_i_f * r_i_f[2] + a_j_f * r_j_f[2]) * inv_S1_f};

        const auto r2_PQ_f = PQ_f[0] * PQ_f[0] + PQ_f[1] * PQ_f[1] + PQ_f[2] * PQ_f[2];

        const auto Lambda_f = sqrtf(4.0f * S1_f * S2_f * MATH_CONST_INV_PI_F * inv_S4_f);

        float F6_t_f[2];

        gpu::computeBoysFunction_f(F6_t_f, S1_f * S2_f * inv_S4_f * r2_PQ_f, 1, boys_func_table_f, boys_func_ft_f);

        const auto QC_0_f = (a_l_f * inv_S2_f) * (r_l_f[c0] - r_k_f[c0]);
        const auto QD_0_f = (-a_k_f * inv_S2_f) * (r_l_f[d0] - r_k_f[d0]);

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
                        +delta_f[b0][b1]*delta_f[c0][d0]*(PA_0_f*(-PA_1_f + PQ_f[a1]) + PA_1_f*PQ_f[a0])

                        +delta_f[c0][d0]*(PB_0_f*(-PA_0_f*delta_f[a1][b1] - PA_1_f*delta_f[a0][b1] - PB_1_f*delta_f[a0][a1] + PQ_f[a0]*delta_f[a1][b1] + PQ_f[a1]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][a1]) + PQ_f[b0]*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) + delta_f[a0][b0]*(PA_1_f*(-PB_1_f + PQ_f[b1]) + PB_1_f*PQ_f[a1]) + delta_f[a1][b0]*(PA_0_f*(-PB_1_f + PQ_f[b1]) + PB_1_f*PQ_f[a0]))

                        +QC_0_f*(PA_0_f*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + PA_1_f*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + PB_0_f*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + PB_1_f*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0])) + QD_0_f*(PA_0_f*(delta_f[a1][b0]*delta_f[b1][c0] + delta_f[a1][b1]*delta_f[b0][c0] + delta_f[a1][c0]*delta_f[b0][b1]) + PA_1_f*(delta_f[a0][b0]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[b0][c0] + delta_f[a0][c0]*delta_f[b0][b1]) + PB_0_f*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1]) + PB_1_f*(delta_f[a0][a1]*delta_f[b0][c0] + delta_f[a0][b0]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b0]))

                        -(PQ_f[c0]*QD_0_f + PQ_f[d0]*QC_0_f)*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + (-0.25f) * inv_S2_f * inv_S4_f * (
                        +PA_0_f*PA_1_f*delta_f[b0][b1]*delta_f[c0][d0]

                        +delta_f[c0][d0]*(PB_0_f*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) + PB_1_f*(PA_0_f*delta_f[a1][b0] + PA_1_f*delta_f[a0][b0]))
                    )

                    + (-0.5f) * S1_f * inv_S2_f * inv_S4_f * (
                        +PA_0_f*PA_1_f*PB_0_f*PB_1_f*delta_f[c0][d0]
                    )

                    + 0.5f * S2_f * inv_S1_f * inv_S4_f * (
                        +QC_0_f*QD_0_f*delta_f[b0][b1]*(PA_0_f*(-PA_1_f + PQ_f[a1]) + PA_1_f*PQ_f[a0])

                        +QC_0_f*QD_0_f*(PB_0_f*(-PA_0_f*delta_f[a1][b1] - PA_1_f*delta_f[a0][b1] - PB_1_f*delta_f[a0][a1] + PQ_f[a0]*delta_f[a1][b1] + PQ_f[a1]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][a1]) + PQ_f[b0]*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) + delta_f[a0][b0]*(PA_1_f*(-PB_1_f + PQ_f[b1]) + PB_1_f*PQ_f[a1]) + delta_f[a1][b0]*(PA_0_f*(-PB_1_f + PQ_f[b1]) + PB_1_f*PQ_f[a0]))
                    )

                    + 0.5f * inv_S4_f * (
                        +delta_f[c0][d0]*(PA_0_f*PA_1_f*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PB_1_f*(PA_0_f*PQ_f[a1] + PA_1_f*PQ_f[a0]))

                        +PA_0_f*PA_1_f*(PB_0_f*(QC_0_f*delta_f[b1][d0] + QD_0_f*delta_f[b1][c0]) + PB_1_f*(QC_0_f*delta_f[b0][d0] + QD_0_f*delta_f[b0][c0])) + PB_0_f*PB_1_f*(PA_0_f*(QC_0_f*delta_f[a1][d0] + QD_0_f*delta_f[a1][c0]) + PA_1_f*(QC_0_f*delta_f[a0][d0] + QD_0_f*delta_f[a0][c0]))

                        -(PQ_f[c0]*QD_0_f + PQ_f[d0]*QC_0_f)*(PA_0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) + PA_1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PB_0_f*PB_1_f*delta_f[a0][a1])
                    )

                    + S1_f * inv_S4_f * (
                        +PA_0_f*PA_1_f*PB_0_f*PB_1_f*(-PQ_f[c0]*QD_0_f - PQ_f[d0]*QC_0_f)
                    )

                    + S2_f * inv_S4_f * (
                        +QC_0_f*QD_0_f*(PA_0_f*PA_1_f*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PB_1_f*(PA_0_f*PQ_f[a1] + PA_1_f*PQ_f[a0]))
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
        double J_ij = 0.0;

        for (uint32_t n = 0; n < TILE_DIM_LARGE; n++)
        {
            J_ij += ERIs[n];
        }

        mat_J[ij] += J_ij;
    }
}

// DDPP altA DDPP1 BASE
__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDPP_altA1(double*         mat_J,
                       const double*   p_prim_info,
                       const uint32_t  p_prim_count,
                       const double*   d_prim_info,
                       const uint32_t  d_prim_count,
                       const double*   pp_mat_D,
                       const double*   dd_mat_Q_local,
                       const double*   pp_mat_Q,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const double*   dd_pair_data_local,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* pp_first_inds,
                       const uint32_t* pp_second_inds,
                       const double*   pp_pair_data,
                       const uint32_t  pp_prim_pair_count,
                       const double*   boys_func_table,
                       const double*   boys_func_ft,
                       const double    eri_threshold)
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

    for (uint32_t m = 0; m < (pp_prim_pair_count + TILE_DIM_LARGE - 1) / TILE_DIM_LARGE; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= pp_prim_pair_count) || (fabs(dd_mat_Q_local[ij] * pp_mat_Q[kl] * pp_mat_D[kl]) <= eri_threshold))
        {
            break;
        }

        const auto k = pp_first_inds[kl];
        const auto l = pp_second_inds[kl];

        const auto a_k = p_prim_info[k / 3 + p_prim_count * 0];

        const double r_k[3] = {p_prim_info[k / 3 + p_prim_count * 2],
                               p_prim_info[k / 3 + p_prim_count * 3],
                               p_prim_info[k / 3 + p_prim_count * 4]};

        const auto a_l = p_prim_info[l / 3 + p_prim_count * 0];

        const double r_l[3] = {p_prim_info[l / 3 + p_prim_count * 2],
                               p_prim_info[l / 3 + p_prim_count * 3],
                               p_prim_info[l / 3 + p_prim_count * 4]};

        const auto S_kl_00 = pp_pair_data[kl];

        const auto c0 = k % 3;
        const auto d0 = l % 3;

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F6_t[7];

        gpu::computeBoysFunction(F6_t, S1 * S2 * inv_S4 * r2_PQ, 6, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                + F6_t[2] * (

                    0.25 * S2 * S2 * inv_S1 * inv_S1 * inv_S4 * inv_S4 * (
                        +QC_0*QD_0*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * S2 * inv_S1 * inv_S4 * inv_S4 * (
                        +delta[b0][b1]*delta[c0][d0]*(-PA_0*PQ[a1] + PQ[a0]*(-PA_1 + PQ[a1]))

                        +delta[c0][d0]*(PQ[a0]*delta[a1][b0]*(-PB_1 + PQ[b1]) + PQ[a0]*delta[a1][b1]*(-PB_0 + PQ[b0]) + PQ[a1]*delta[a0][b0]*(-PB_1 + PQ[b1]) + PQ[a1]*delta[a0][b1]*(-PB_0 + PQ[b0]) + PQ[b0]*PQ[b1]*delta[a0][a1] - PQ[b0]*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) - PQ[b1]*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0] + PB_0*delta[a0][a1]))

                        +(-PA_0*QC_0 + PQ[a0]*QC_0)*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + (-PA_0*QD_0 + PQ[a0]*QD_0)*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1]) + (-PA_1*QC_0 + PQ[a1]*QC_0)*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + (-PA_1*QD_0 + PQ[a1]*QD_0)*(delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1]) + (-PB_0*QC_0 + PQ[b0]*QC_0)*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + (-PB_0*QD_0 + PQ[b0]*QD_0)*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]) + (-PB_1*QC_0 + PQ[b1]*QC_0)*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]) + (-PB_1*QD_0 + PQ[b1]*QD_0)*(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0])

                        +2.0*(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * inv_S4 * inv_S4 * (
                        +delta[a1][b1]*delta[c0][d0]*(PA_0*(PB_0 - PQ[b0]) - PB_0*PQ[a0])

                        +PA_0*(PA_1*(delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0]) + PB_0*(delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + PB_1*(delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0])) + PA_1*(PB_0*(delta[a0][c0]*delta[b1][d0] + delta[a0][d0]*delta[b1][c0]) + PB_1*(delta[a0][c0]*delta[b0][d0] + delta[a0][d0]*delta[b0][c0])) + PB_0*PB_1*(delta[a0][c0]*delta[a1][d0] + delta[a0][d0]*delta[a1][c0])

                        +delta[c0][d0]*(PA_1*(PB_1*delta[a0][b0] + delta[a0][b1]*(PB_0 - PQ[b0]) + delta[b0][b1]*(PA_0 - PQ[a0])) + PB_1*delta[a0][a1]*(PB_0 - PQ[b0]) + PB_1*delta[a1][b0]*(PA_0 - PQ[a0]) - PQ[a1]*(PA_0*delta[b0][b1] + PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) - PQ[b1]*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0] + PB_0*delta[a0][a1]))

                        -PQ[c0]*(PA_0*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])) - PQ[d0]*(PA_0*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0]))

                        +PQ[c0]*PQ[d0]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.5 * S1 * inv_S4 * inv_S4 * (
                        -delta[c0][d0]*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))

                        -PA_0*PA_1*(PB_0*(PQ[c0]*delta[b1][d0] + PQ[d0]*delta[b1][c0]) + PB_1*(PQ[c0]*delta[b0][d0] + PQ[d0]*delta[b0][c0])) - PB_0*PB_1*(PA_0*(PQ[c0]*delta[a1][d0] + PQ[d0]*delta[a1][c0]) + PA_1*(PQ[c0]*delta[a0][d0] + PQ[d0]*delta[a0][c0]))

                        +PQ[c0]*PQ[d0]*(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1])
                    )

                    + 0.5 * S2 * S2 * inv_S1 * inv_S4 * inv_S4 * (
                        +QC_0*QD_0*delta[b0][b1]*(-PA_0*PQ[a1] + PQ[a0]*(-PA_1 + PQ[a1]))

                        +QC_0*QD_0*(PQ[a0]*delta[a1][b0]*(-PB_1 + PQ[b1]) + PQ[a0]*delta[a1][b1]*(-PB_0 + PQ[b0]) + PQ[a1]*delta[a0][b0]*(-PB_1 + PQ[b1]) + PQ[a1]*delta[a0][b1]*(-PB_0 + PQ[b0]) + PQ[b0]*PQ[b1]*delta[a0][a1] - PQ[b0]*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) - PQ[b1]*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0] + PB_0*delta[a0][a1]))
                    )

                    + 0.5 * S2 * inv_S4 * inv_S4 * (
                        +QD_0*delta[b1][c0]*(PA_0*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PA_1*PB_0*PQ[a0])

                        +PB_1*(PA_0*(PQ[a1]*(QC_0*delta[b0][d0] + QD_0*delta[b0][c0]) + PQ[b0]*(QC_0*delta[a1][d0] + QD_0*delta[a1][c0])) + PQ[a0]*(PA_1*(QC_0*delta[b0][d0] + QD_0*delta[b0][c0]) + PB_0*(QC_0*delta[a1][d0] + QD_0*delta[a1][c0]))) + QC_0*(PA_0*(PA_1*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0]) + PB_0*(PQ[a1]*delta[b1][d0] + PQ[b1]*delta[a1][d0])) + PA_1*PB_0*(PQ[a0]*delta[b1][d0] + PQ[b1]*delta[a0][d0]) + PB_1*delta[a0][d0]*(PA_1*PQ[b0] + PB_0*PQ[a1])) + QD_0*(PA_0*PQ[b1]*(PA_1*delta[b0][c0] + PB_0*delta[a1][c0]) + delta[a0][c0]*(PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*PQ[a1]))

                        +(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) - PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) - PA_1*(PQ[a0]*delta[b0][b1] + PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1] - PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) - PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]))

                        +delta[c0][d0]*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S1 * S1 * inv_S4 * inv_S4 * (
                        +PA_0*PA_1*PB_0*PB_1*PQ[c0]*PQ[d0]
                    )

                    + S1 * S2 * inv_S4 * inv_S4 * (
                        -(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))
                    )

                    + S2 * S2 * inv_S4 * inv_S4 * (
                        +QC_0*QD_0*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + 0.125 * S2 * inv_S1 * inv_S1 * inv_S4 * inv_S4 * (
                        +delta[c0][d0]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.125 * inv_S1 * inv_S4 * inv_S4 * (
                        +2.0*delta[c0][d0]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])

                        +delta[b0][b1]*(delta[a0][c0]*delta[a1][d0] + delta[a0][d0]*delta[a1][c0]) + delta[b0][c0]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + delta[b0][d0]*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]) + delta[b1][c0]*(delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]) + delta[b1][d0]*(delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0])
                    )

                )

                + F6_t[3] * (

                    (-0.125) * S2 * inv_S1 * inv_S4 * inv_S4 * inv_S4 * (
                        +delta[a0][a1]*(delta[b0][b1]*delta[c0][d0] + delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0]) + delta[a0][b0]*(delta[a1][b1]*delta[c0][d0] + delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + delta[a0][b1]*(delta[a1][b0]*delta[c0][d0] + delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0]) + delta[a0][c0]*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + delta[a0][d0]*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1])
                    )

                    + (-0.25) * S2 * S2 * inv_S1 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*delta[b0][b1]*delta[c0][d0]

                        +delta[c0][d0]*(PQ[b0]*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + PQ[b1]*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0]))

                        +QC_0*(PQ[a0]*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + PQ[b0]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + PQ[b1]*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])) + QD_0*(PQ[a0]*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1]) + PQ[b0]*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]) + PQ[b1]*(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0]))

                        +(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        +delta[a1][b1]*delta[c0][d0]*(PB_0*PQ[a0] + PQ[b0]*(PA_0 - PQ[a0]))

                        +(PA_0*PQ[a1] + PA_1*PQ[a0])*(delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0]) + (PA_0*PQ[b0] + PB_0*PQ[a0])*(delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + (PA_0*PQ[b1] + PB_1*PQ[a0])*(delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0]) + (PA_1*PQ[b0] + PB_0*PQ[a1])*(delta[a0][c0]*delta[b1][d0] + delta[a0][d0]*delta[b1][c0]) + (PA_1*PQ[b1] + PB_1*PQ[a1])*(delta[a0][c0]*delta[b0][d0] + delta[a0][d0]*delta[b0][c0]) + (PB_0*PQ[b1] + PB_1*PQ[b0])*(delta[a0][c0]*delta[a1][d0] + delta[a0][d0]*delta[a1][c0])

                        +delta[c0][d0]*(PA_1*(PQ[a0]*delta[b0][b1] + PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PB_1*(PQ[a0]*delta[a1][b0] + PQ[b0]*delta[a0][a1]) + PQ[a1]*(PA_0*delta[b0][b1] + PB_0*delta[a0][b1] + PB_1*delta[a0][b0] - PQ[a0]*delta[b0][b1] - PQ[b0]*delta[a0][b1] - PQ[b1]*delta[a0][b0]) + PQ[b1]*(delta[a0][a1]*(PB_0 - PQ[b0]) + delta[a1][b0]*(PA_0 - PQ[a0])))

                        +(PA_0*PQ[c0] - PQ[a0]*PQ[c0])*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + (PA_0*PQ[d0] - PQ[a0]*PQ[d0])*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1]) + (PA_1*PQ[c0] - PQ[a1]*PQ[c0])*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + (PA_1*PQ[d0] - PQ[a1]*PQ[d0])*(delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1]) + (PB_0*PQ[c0] - PQ[b0]*PQ[c0])*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + (PB_0*PQ[d0] - PQ[b0]*PQ[d0])*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]) + (PB_1*PQ[c0] - PQ[b1]*PQ[c0])*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]) + (PB_1*PQ[d0] - PQ[b1]*PQ[d0])*(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0])

                        -2*PQ[c0]*PQ[d0]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.5 * S1 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        -delta[c0][d0]*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))

                        -PA_0*(PA_1*(PQ[b0]*(PQ[c0]*delta[b1][d0] + PQ[d0]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[b0][d0] + PQ[d0]*delta[b0][c0])) + PB_0*(PQ[a1]*(PQ[c0]*delta[b1][d0] + PQ[d0]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[a1][d0] + PQ[d0]*delta[a1][c0])) + PB_1*(PQ[a1]*(PQ[c0]*delta[b0][d0] + PQ[d0]*delta[b0][c0]) + PQ[b0]*(PQ[c0]*delta[a1][d0] + PQ[d0]*delta[a1][c0]))) - PA_1*(PB_0*(PQ[a0]*(PQ[c0]*delta[b1][d0] + PQ[d0]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[a0][d0] + PQ[d0]*delta[a0][c0])) + PB_1*(PQ[a0]*(PQ[c0]*delta[b0][d0] + PQ[d0]*delta[b0][c0]) + PQ[b0]*(PQ[c0]*delta[a0][d0] + PQ[d0]*delta[a0][c0]))) - PB_0*PB_1*(PQ[a0]*(PQ[c0]*delta[a1][d0] + PQ[d0]*delta[a1][c0]) + PQ[a1]*(PQ[c0]*delta[a0][d0] + PQ[d0]*delta[a0][c0]))

                        +PQ[c0]*PQ[d0]*(PA_0*(-PA_1*delta[b0][b1] - PB_0*delta[a1][b1] - PB_1*delta[a1][b0] + PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PA_1*(PQ[a0]*delta[b0][b1] + delta[a0][b0]*(-PB_1 + PQ[b1]) + delta[a0][b1]*(-PB_0 + PQ[b0])) + PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + delta[a0][a1]*(-PB_1 + PQ[b1])) + PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]))
                    )

                    + (-0.5) * S2 * S2 * S2 * inv_S1 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*QC_0*QD_0*delta[b0][b1]

                        +QC_0*QD_0*(PQ[b0]*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + PQ[b1]*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0]))
                    )

                    + 0.5 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        +QD_0*delta[b1][c0]*(PA_1*PQ[a0]*PQ[b0] + PQ[a1]*(PA_0*PQ[b0] + PB_0*PQ[a0]))

                        +PB_1*PQ[a0]*(PQ[a1]*(QC_0*delta[b0][d0] + QD_0*delta[b0][c0]) + PQ[b0]*(QC_0*delta[a1][d0] + QD_0*delta[a1][c0])) + QC_0*(PQ[a0]*(PA_1*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0]) + PB_0*(PQ[a1]*delta[b1][d0] + PQ[b1]*delta[a1][d0])) + PQ[a1]*(PA_0*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0]) + delta[a0][d0]*(PB_0*PQ[b1] + PB_1*PQ[b0])) + PQ[b0]*PQ[b1]*(PA_0*delta[a1][d0] + PA_1*delta[a0][d0])) + QD_0*(PB_1*PQ[a1]*PQ[b0]*delta[a0][c0] + PQ[b1]*(PA_0*(PQ[a1]*delta[b0][c0] + PQ[b0]*delta[a1][c0]) + PA_1*(PQ[a0]*delta[b0][c0] + PQ[b0]*delta[a0][c0]) + PB_0*(PQ[a0]*delta[a1][c0] + PQ[a1]*delta[a0][c0])))

                        +(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PQ[a0]*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0] - PQ[a1]*delta[b0][b1] - PQ[b0]*delta[a1][b1] - PQ[b1]*delta[a1][b0]) + PQ[a1]*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PQ[b0]*(PB_1*delta[a0][a1] + delta[a0][b1]*(PA_1 - PQ[a1])) + PQ[b1]*(delta[a0][a1]*(PB_0 - PQ[b0]) + delta[a0][b0]*(PA_1 - PQ[a1])))

                        +delta[c0][d0]*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S1 * S1 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[c0]*PQ[d0]*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))
                    )

                    + S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        -(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        +QC_0*QD_0*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                )

                + F6_t[4] * (

                    0.5 * S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        -delta[c0][d0]*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))

                        -PQ[a0]*(PA_1*(PQ[b0]*(PQ[c0]*delta[b1][d0] + PQ[d0]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[b0][d0] + PQ[d0]*delta[b0][c0])) + PB_0*(PQ[a1]*(PQ[c0]*delta[b1][d0] + PQ[d0]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[a1][d0] + PQ[d0]*delta[a1][c0])) + PB_1*(PQ[a1]*(PQ[c0]*delta[b0][d0] + PQ[d0]*delta[b0][c0]) + PQ[b0]*(PQ[c0]*delta[a1][d0] + PQ[d0]*delta[a1][c0]))) - PQ[a1]*(PA_0*(PQ[b0]*(PQ[c0]*delta[b1][d0] + PQ[d0]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[b0][d0] + PQ[d0]*delta[b0][c0])) + (PB_0*PQ[b1] + PB_1*PQ[b0])*(PQ[c0]*delta[a0][d0] + PQ[d0]*delta[a0][c0])) - PQ[b0]*PQ[b1]*(PA_0*(PQ[c0]*delta[a1][d0] + PQ[d0]*delta[a1][c0]) + PA_1*(PQ[c0]*delta[a0][d0] + PQ[d0]*delta[a0][c0]))

                        +PQ[c0]*PQ[d0]*(-PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) - PQ[a0]*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PQ[a0]*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) - PQ[a1]*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PQ[a1]*(PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PQ[b0]*PQ[b1]*delta[a0][a1] - PQ[b0]*(PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) - PQ[b1]*(PA_1*delta[a0][b0] + PB_0*delta[a0][a1]))
                    )

                    + 0.5 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*QD_0*delta[b1][c0]

                        +PQ[a0]*PQ[a1]*QC_0*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0]) + PQ[b1]*(PQ[a0]*QD_0*(PQ[a1]*delta[b0][c0] + PQ[b0]*delta[a1][c0]) + PQ[b0]*(PQ[a1]*QD_0*delta[a0][c0] + QC_0*(PQ[a0]*delta[a1][d0] + PQ[a1]*delta[a0][d0])))

                        +(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(PQ[a0]*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PQ[a1]*(PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PQ[b0]*PQ[b1]*delta[a0][a1])

                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*delta[c0][d0]
                    )

                    + S1 * S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[c0]*PQ[d0]*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S1 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        -(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S2 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*QC_0*QD_0
                    )

                    + 0.25 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*(delta[b0][b1]*delta[c0][d0] + delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0])

                        +PQ[b0]*(PQ[a0]*(delta[a1][b1]*delta[c0][d0] + delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + PQ[a1]*(delta[a0][b1]*delta[c0][d0] + delta[a0][c0]*delta[b1][d0] + delta[a0][d0]*delta[b1][c0]) + PQ[b1]*(delta[a0][a1]*delta[c0][d0] + delta[a0][c0]*delta[a1][d0] + delta[a0][d0]*delta[a1][c0]) + PQ[c0]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + PQ[d0]*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1])) + PQ[b1]*(PQ[a0]*(delta[a1][b0]*delta[c0][d0] + delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0]) + PQ[a1]*(delta[a0][b0]*delta[c0][d0] + delta[a0][c0]*delta[b0][d0] + delta[a0][d0]*delta[b0][c0]) + PQ[c0]*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]) + PQ[d0]*(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0])) + PQ[c0]*(PQ[a0]*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + PQ[d0]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])) + PQ[d0]*(PQ[a0]*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1]))
                    )

                )

                + F6_t[5] * (

                    (-0.5) * S1 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*delta[c0][d0]

                        +PQ[c0]*(PQ[a0]*(PQ[a1]*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0] + PQ[d0]*delta[b0][b1]) + PQ[b1]*PQ[d0]*delta[a1][b0]) + PQ[b0]*PQ[b1]*(PQ[a0]*delta[a1][d0] + PQ[a1]*delta[a0][d0] + PQ[d0]*delta[a0][a1])) + PQ[d0]*(PQ[a1]*PQ[b1]*(PQ[a0]*delta[b0][c0] + PQ[b0]*delta[a0][c0] + PQ[c0]*delta[a0][b0]) + PQ[b0]*(PQ[a0]*(PQ[a1]*delta[b1][c0] + PQ[b1]*delta[a1][c0] + PQ[c0]*delta[a1][b1]) + PQ[a1]*PQ[c0]*delta[a0][b1]))
                    )

                    + S1 * S1 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[c0]*PQ[d0]*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S1 * S2 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*(-PQ[c0]*QD_0 - PQ[d0]*QC_0)
                    )

                )

                + F6_t[6] * (

                    S1 * S1 * S2 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*PQ[c0]*PQ[d0]
                    )

                )

            );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * pp_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

// DDPP altA DDPP1 FP64
__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDPP_altA1_FP64(double*         mat_J,
                       const double*   p_prim_info,
                       const uint32_t  p_prim_count,
                       const double*   d_prim_info,
                       const uint32_t  d_prim_count,
                       const double*   pp_mat_D,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const double*   dd_pair_data_local,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* pp_first_inds,
                       const uint32_t* pp_second_inds,
                       const double*   pp_pair_data,
                       const uint32_t  pp_prim_pair_count,
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

        if ((ij >= dd_prim_pair_count_local) || (kl >= pp_prim_pair_count))
        {
            break;
        }

        const auto k = pp_first_inds[kl];
        const auto l = pp_second_inds[kl];

        const auto a_k = p_prim_info[k / 3 + p_prim_count * 0];

        const double r_k[3] = {p_prim_info[k / 3 + p_prim_count * 2],
                               p_prim_info[k / 3 + p_prim_count * 3],
                               p_prim_info[k / 3 + p_prim_count * 4]};

        const auto a_l = p_prim_info[l / 3 + p_prim_count * 0];

        const double r_l[3] = {p_prim_info[l / 3 + p_prim_count * 2],
                               p_prim_info[l / 3 + p_prim_count * 3],
                               p_prim_info[l / 3 + p_prim_count * 4]};

        const auto S_kl_00 = pp_pair_data[kl];

        const auto c0 = k % 3;
        const auto d0 = l % 3;

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F6_t[7];

        gpu::computeBoysFunction(F6_t, S1 * S2 * inv_S4 * r2_PQ, 6, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                + F6_t[2] * (

                    0.25 * S2 * S2 * inv_S1 * inv_S1 * inv_S4 * inv_S4 * (
                        +QC_0*QD_0*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * S2 * inv_S1 * inv_S4 * inv_S4 * (
                        +delta[b0][b1]*delta[c0][d0]*(-PA_0*PQ[a1] + PQ[a0]*(-PA_1 + PQ[a1]))

                        +delta[c0][d0]*(PQ[a0]*delta[a1][b0]*(-PB_1 + PQ[b1]) + PQ[a0]*delta[a1][b1]*(-PB_0 + PQ[b0]) + PQ[a1]*delta[a0][b0]*(-PB_1 + PQ[b1]) + PQ[a1]*delta[a0][b1]*(-PB_0 + PQ[b0]) + PQ[b0]*PQ[b1]*delta[a0][a1] - PQ[b0]*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) - PQ[b1]*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0] + PB_0*delta[a0][a1]))

                        +(-PA_0*QC_0 + PQ[a0]*QC_0)*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + (-PA_0*QD_0 + PQ[a0]*QD_0)*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1]) + (-PA_1*QC_0 + PQ[a1]*QC_0)*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + (-PA_1*QD_0 + PQ[a1]*QD_0)*(delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1]) + (-PB_0*QC_0 + PQ[b0]*QC_0)*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + (-PB_0*QD_0 + PQ[b0]*QD_0)*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]) + (-PB_1*QC_0 + PQ[b1]*QC_0)*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]) + (-PB_1*QD_0 + PQ[b1]*QD_0)*(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0])

                        +2.0*(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * inv_S4 * inv_S4 * (
                        +delta[a1][b1]*delta[c0][d0]*(PA_0*(PB_0 - PQ[b0]) - PB_0*PQ[a0])

                        +PA_0*(PA_1*(delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0]) + PB_0*(delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + PB_1*(delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0])) + PA_1*(PB_0*(delta[a0][c0]*delta[b1][d0] + delta[a0][d0]*delta[b1][c0]) + PB_1*(delta[a0][c0]*delta[b0][d0] + delta[a0][d0]*delta[b0][c0])) + PB_0*PB_1*(delta[a0][c0]*delta[a1][d0] + delta[a0][d0]*delta[a1][c0])

                        +delta[c0][d0]*(PA_1*(PB_1*delta[a0][b0] + delta[a0][b1]*(PB_0 - PQ[b0]) + delta[b0][b1]*(PA_0 - PQ[a0])) + PB_1*delta[a0][a1]*(PB_0 - PQ[b0]) + PB_1*delta[a1][b0]*(PA_0 - PQ[a0]) - PQ[a1]*(PA_0*delta[b0][b1] + PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) - PQ[b1]*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0] + PB_0*delta[a0][a1]))

                        -PQ[c0]*(PA_0*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])) - PQ[d0]*(PA_0*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0]))

                        +PQ[c0]*PQ[d0]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.5 * S1 * inv_S4 * inv_S4 * (
                        -delta[c0][d0]*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))

                        -PA_0*PA_1*(PB_0*(PQ[c0]*delta[b1][d0] + PQ[d0]*delta[b1][c0]) + PB_1*(PQ[c0]*delta[b0][d0] + PQ[d0]*delta[b0][c0])) - PB_0*PB_1*(PA_0*(PQ[c0]*delta[a1][d0] + PQ[d0]*delta[a1][c0]) + PA_1*(PQ[c0]*delta[a0][d0] + PQ[d0]*delta[a0][c0]))

                        +PQ[c0]*PQ[d0]*(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1])
                    )

                    + 0.5 * S2 * S2 * inv_S1 * inv_S4 * inv_S4 * (
                        +QC_0*QD_0*delta[b0][b1]*(-PA_0*PQ[a1] + PQ[a0]*(-PA_1 + PQ[a1]))

                        +QC_0*QD_0*(PQ[a0]*delta[a1][b0]*(-PB_1 + PQ[b1]) + PQ[a0]*delta[a1][b1]*(-PB_0 + PQ[b0]) + PQ[a1]*delta[a0][b0]*(-PB_1 + PQ[b1]) + PQ[a1]*delta[a0][b1]*(-PB_0 + PQ[b0]) + PQ[b0]*PQ[b1]*delta[a0][a1] - PQ[b0]*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) - PQ[b1]*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0] + PB_0*delta[a0][a1]))
                    )

                    + 0.5 * S2 * inv_S4 * inv_S4 * (
                        +QD_0*delta[b1][c0]*(PA_0*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PA_1*PB_0*PQ[a0])

                        +PB_1*(PA_0*(PQ[a1]*(QC_0*delta[b0][d0] + QD_0*delta[b0][c0]) + PQ[b0]*(QC_0*delta[a1][d0] + QD_0*delta[a1][c0])) + PQ[a0]*(PA_1*(QC_0*delta[b0][d0] + QD_0*delta[b0][c0]) + PB_0*(QC_0*delta[a1][d0] + QD_0*delta[a1][c0]))) + QC_0*(PA_0*(PA_1*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0]) + PB_0*(PQ[a1]*delta[b1][d0] + PQ[b1]*delta[a1][d0])) + PA_1*PB_0*(PQ[a0]*delta[b1][d0] + PQ[b1]*delta[a0][d0]) + PB_1*delta[a0][d0]*(PA_1*PQ[b0] + PB_0*PQ[a1])) + QD_0*(PA_0*PQ[b1]*(PA_1*delta[b0][c0] + PB_0*delta[a1][c0]) + delta[a0][c0]*(PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*PQ[a1]))

                        +(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) - PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) - PA_1*(PQ[a0]*delta[b0][b1] + PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1] - PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) - PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]))

                        +delta[c0][d0]*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S1 * S1 * inv_S4 * inv_S4 * (
                        +PA_0*PA_1*PB_0*PB_1*PQ[c0]*PQ[d0]
                    )

                    + S1 * S2 * inv_S4 * inv_S4 * (
                        -(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))
                    )

                    + S2 * S2 * inv_S4 * inv_S4 * (
                        +QC_0*QD_0*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + 0.125 * S2 * inv_S1 * inv_S1 * inv_S4 * inv_S4 * (
                        +delta[c0][d0]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.125 * inv_S1 * inv_S4 * inv_S4 * (
                        +2.0*delta[c0][d0]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])

                        +delta[b0][b1]*(delta[a0][c0]*delta[a1][d0] + delta[a0][d0]*delta[a1][c0]) + delta[b0][c0]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + delta[b0][d0]*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]) + delta[b1][c0]*(delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]) + delta[b1][d0]*(delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0])
                    )

                )

                + F6_t[3] * (

                    (-0.125) * S2 * inv_S1 * inv_S4 * inv_S4 * inv_S4 * (
                        +delta[a0][a1]*(delta[b0][b1]*delta[c0][d0] + delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0]) + delta[a0][b0]*(delta[a1][b1]*delta[c0][d0] + delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + delta[a0][b1]*(delta[a1][b0]*delta[c0][d0] + delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0]) + delta[a0][c0]*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + delta[a0][d0]*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1])
                    )

                    + (-0.25) * S2 * S2 * inv_S1 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*delta[b0][b1]*delta[c0][d0]

                        +delta[c0][d0]*(PQ[b0]*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + PQ[b1]*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0]))

                        +QC_0*(PQ[a0]*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + PQ[b0]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + PQ[b1]*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])) + QD_0*(PQ[a0]*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1]) + PQ[b0]*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]) + PQ[b1]*(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0]))

                        +(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        +delta[a1][b1]*delta[c0][d0]*(PB_0*PQ[a0] + PQ[b0]*(PA_0 - PQ[a0]))

                        +(PA_0*PQ[a1] + PA_1*PQ[a0])*(delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0]) + (PA_0*PQ[b0] + PB_0*PQ[a0])*(delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + (PA_0*PQ[b1] + PB_1*PQ[a0])*(delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0]) + (PA_1*PQ[b0] + PB_0*PQ[a1])*(delta[a0][c0]*delta[b1][d0] + delta[a0][d0]*delta[b1][c0]) + (PA_1*PQ[b1] + PB_1*PQ[a1])*(delta[a0][c0]*delta[b0][d0] + delta[a0][d0]*delta[b0][c0]) + (PB_0*PQ[b1] + PB_1*PQ[b0])*(delta[a0][c0]*delta[a1][d0] + delta[a0][d0]*delta[a1][c0])

                        +delta[c0][d0]*(PA_1*(PQ[a0]*delta[b0][b1] + PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PB_1*(PQ[a0]*delta[a1][b0] + PQ[b0]*delta[a0][a1]) + PQ[a1]*(PA_0*delta[b0][b1] + PB_0*delta[a0][b1] + PB_1*delta[a0][b0] - PQ[a0]*delta[b0][b1] - PQ[b0]*delta[a0][b1] - PQ[b1]*delta[a0][b0]) + PQ[b1]*(delta[a0][a1]*(PB_0 - PQ[b0]) + delta[a1][b0]*(PA_0 - PQ[a0])))

                        +(PA_0*PQ[c0] - PQ[a0]*PQ[c0])*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + (PA_0*PQ[d0] - PQ[a0]*PQ[d0])*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1]) + (PA_1*PQ[c0] - PQ[a1]*PQ[c0])*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + (PA_1*PQ[d0] - PQ[a1]*PQ[d0])*(delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1]) + (PB_0*PQ[c0] - PQ[b0]*PQ[c0])*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + (PB_0*PQ[d0] - PQ[b0]*PQ[d0])*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]) + (PB_1*PQ[c0] - PQ[b1]*PQ[c0])*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]) + (PB_1*PQ[d0] - PQ[b1]*PQ[d0])*(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0])

                        -2*PQ[c0]*PQ[d0]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.5 * S1 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        -delta[c0][d0]*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))

                        -PA_0*(PA_1*(PQ[b0]*(PQ[c0]*delta[b1][d0] + PQ[d0]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[b0][d0] + PQ[d0]*delta[b0][c0])) + PB_0*(PQ[a1]*(PQ[c0]*delta[b1][d0] + PQ[d0]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[a1][d0] + PQ[d0]*delta[a1][c0])) + PB_1*(PQ[a1]*(PQ[c0]*delta[b0][d0] + PQ[d0]*delta[b0][c0]) + PQ[b0]*(PQ[c0]*delta[a1][d0] + PQ[d0]*delta[a1][c0]))) - PA_1*(PB_0*(PQ[a0]*(PQ[c0]*delta[b1][d0] + PQ[d0]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[a0][d0] + PQ[d0]*delta[a0][c0])) + PB_1*(PQ[a0]*(PQ[c0]*delta[b0][d0] + PQ[d0]*delta[b0][c0]) + PQ[b0]*(PQ[c0]*delta[a0][d0] + PQ[d0]*delta[a0][c0]))) - PB_0*PB_1*(PQ[a0]*(PQ[c0]*delta[a1][d0] + PQ[d0]*delta[a1][c0]) + PQ[a1]*(PQ[c0]*delta[a0][d0] + PQ[d0]*delta[a0][c0]))

                        +PQ[c0]*PQ[d0]*(PA_0*(-PA_1*delta[b0][b1] - PB_0*delta[a1][b1] - PB_1*delta[a1][b0] + PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PA_1*(PQ[a0]*delta[b0][b1] + delta[a0][b0]*(-PB_1 + PQ[b1]) + delta[a0][b1]*(-PB_0 + PQ[b0])) + PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + delta[a0][a1]*(-PB_1 + PQ[b1])) + PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]))
                    )

                    + (-0.5) * S2 * S2 * S2 * inv_S1 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*QC_0*QD_0*delta[b0][b1]

                        +QC_0*QD_0*(PQ[b0]*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + PQ[b1]*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0]))
                    )

                    + 0.5 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        +QD_0*delta[b1][c0]*(PA_1*PQ[a0]*PQ[b0] + PQ[a1]*(PA_0*PQ[b0] + PB_0*PQ[a0]))

                        +PB_1*PQ[a0]*(PQ[a1]*(QC_0*delta[b0][d0] + QD_0*delta[b0][c0]) + PQ[b0]*(QC_0*delta[a1][d0] + QD_0*delta[a1][c0])) + QC_0*(PQ[a0]*(PA_1*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0]) + PB_0*(PQ[a1]*delta[b1][d0] + PQ[b1]*delta[a1][d0])) + PQ[a1]*(PA_0*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0]) + delta[a0][d0]*(PB_0*PQ[b1] + PB_1*PQ[b0])) + PQ[b0]*PQ[b1]*(PA_0*delta[a1][d0] + PA_1*delta[a0][d0])) + QD_0*(PB_1*PQ[a1]*PQ[b0]*delta[a0][c0] + PQ[b1]*(PA_0*(PQ[a1]*delta[b0][c0] + PQ[b0]*delta[a1][c0]) + PA_1*(PQ[a0]*delta[b0][c0] + PQ[b0]*delta[a0][c0]) + PB_0*(PQ[a0]*delta[a1][c0] + PQ[a1]*delta[a0][c0])))

                        +(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PQ[a0]*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0] - PQ[a1]*delta[b0][b1] - PQ[b0]*delta[a1][b1] - PQ[b1]*delta[a1][b0]) + PQ[a1]*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PQ[b0]*(PB_1*delta[a0][a1] + delta[a0][b1]*(PA_1 - PQ[a1])) + PQ[b1]*(delta[a0][a1]*(PB_0 - PQ[b0]) + delta[a0][b0]*(PA_1 - PQ[a1])))

                        +delta[c0][d0]*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S1 * S1 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[c0]*PQ[d0]*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))
                    )

                    + S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        -(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        +QC_0*QD_0*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                )

                + F6_t[4] * (

                    0.5 * S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        -delta[c0][d0]*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))

                        -PQ[a0]*(PA_1*(PQ[b0]*(PQ[c0]*delta[b1][d0] + PQ[d0]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[b0][d0] + PQ[d0]*delta[b0][c0])) + PB_0*(PQ[a1]*(PQ[c0]*delta[b1][d0] + PQ[d0]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[a1][d0] + PQ[d0]*delta[a1][c0])) + PB_1*(PQ[a1]*(PQ[c0]*delta[b0][d0] + PQ[d0]*delta[b0][c0]) + PQ[b0]*(PQ[c0]*delta[a1][d0] + PQ[d0]*delta[a1][c0]))) - PQ[a1]*(PA_0*(PQ[b0]*(PQ[c0]*delta[b1][d0] + PQ[d0]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[b0][d0] + PQ[d0]*delta[b0][c0])) + (PB_0*PQ[b1] + PB_1*PQ[b0])*(PQ[c0]*delta[a0][d0] + PQ[d0]*delta[a0][c0])) - PQ[b0]*PQ[b1]*(PA_0*(PQ[c0]*delta[a1][d0] + PQ[d0]*delta[a1][c0]) + PA_1*(PQ[c0]*delta[a0][d0] + PQ[d0]*delta[a0][c0]))

                        +PQ[c0]*PQ[d0]*(-PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) - PQ[a0]*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PQ[a0]*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) - PQ[a1]*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PQ[a1]*(PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PQ[b0]*PQ[b1]*delta[a0][a1] - PQ[b0]*(PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) - PQ[b1]*(PA_1*delta[a0][b0] + PB_0*delta[a0][a1]))
                    )

                    + 0.5 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*QD_0*delta[b1][c0]

                        +PQ[a0]*PQ[a1]*QC_0*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0]) + PQ[b1]*(PQ[a0]*QD_0*(PQ[a1]*delta[b0][c0] + PQ[b0]*delta[a1][c0]) + PQ[b0]*(PQ[a1]*QD_0*delta[a0][c0] + QC_0*(PQ[a0]*delta[a1][d0] + PQ[a1]*delta[a0][d0])))

                        +(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(PQ[a0]*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PQ[a1]*(PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PQ[b0]*PQ[b1]*delta[a0][a1])

                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*delta[c0][d0]
                    )

                    + S1 * S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[c0]*PQ[d0]*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S1 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        -(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S2 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*QC_0*QD_0
                    )

                    + 0.25 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*(delta[b0][b1]*delta[c0][d0] + delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0])

                        +PQ[b0]*(PQ[a0]*(delta[a1][b1]*delta[c0][d0] + delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + PQ[a1]*(delta[a0][b1]*delta[c0][d0] + delta[a0][c0]*delta[b1][d0] + delta[a0][d0]*delta[b1][c0]) + PQ[b1]*(delta[a0][a1]*delta[c0][d0] + delta[a0][c0]*delta[a1][d0] + delta[a0][d0]*delta[a1][c0]) + PQ[c0]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + PQ[d0]*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1])) + PQ[b1]*(PQ[a0]*(delta[a1][b0]*delta[c0][d0] + delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0]) + PQ[a1]*(delta[a0][b0]*delta[c0][d0] + delta[a0][c0]*delta[b0][d0] + delta[a0][d0]*delta[b0][c0]) + PQ[c0]*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]) + PQ[d0]*(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0])) + PQ[c0]*(PQ[a0]*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + PQ[d0]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])) + PQ[d0]*(PQ[a0]*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1]))
                    )

                )

                + F6_t[5] * (

                    (-0.5) * S1 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*delta[c0][d0]

                        +PQ[c0]*(PQ[a0]*(PQ[a1]*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0] + PQ[d0]*delta[b0][b1]) + PQ[b1]*PQ[d0]*delta[a1][b0]) + PQ[b0]*PQ[b1]*(PQ[a0]*delta[a1][d0] + PQ[a1]*delta[a0][d0] + PQ[d0]*delta[a0][a1])) + PQ[d0]*(PQ[a1]*PQ[b1]*(PQ[a0]*delta[b0][c0] + PQ[b0]*delta[a0][c0] + PQ[c0]*delta[a0][b0]) + PQ[b0]*(PQ[a0]*(PQ[a1]*delta[b1][c0] + PQ[b1]*delta[a1][c0] + PQ[c0]*delta[a1][b1]) + PQ[a1]*PQ[c0]*delta[a0][b1]))
                    )

                    + S1 * S1 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[c0]*PQ[d0]*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S1 * S2 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*(-PQ[c0]*QD_0 - PQ[d0]*QC_0)
                    )

                )

                + F6_t[6] * (

                    S1 * S1 * S2 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*PQ[c0]*PQ[d0]
                    )

                )

            );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * pp_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

// DDPP altA DDPP1 FP32
__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDPP_altA1_FP32(double*         mat_J,
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

        const float r_k_f[3] = {p_prim_info_f[k / 3 + p_prim_count * 2],
                               p_prim_info_f[k / 3 + p_prim_count * 3],
                               p_prim_info_f[k / 3 + p_prim_count * 4]};

        const auto a_l_f = p_prim_info_f[l / 3 + p_prim_count * 0];

        const float r_l_f[3] = {p_prim_info_f[l / 3 + p_prim_count * 2],
                               p_prim_info_f[l / 3 + p_prim_count * 3],
                               p_prim_info_f[l / 3 + p_prim_count * 4]};

        const auto S_kl_00_f = pp_pair_data_f[kl];

        const auto c0 = k % 3;
        const auto d0 = l % 3;

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2_f = a_k_f + a_l_f;

        const auto inv_S2_f = 1.0f / S2_f;
        const auto inv_S4_f = 1.0f / (S1_f + S2_f);

        const float PQ_f[3] = {(a_k_f * r_k_f[0] + a_l_f * r_l_f[0]) * inv_S2_f - (a_i_f * r_i_f[0] + a_j_f * r_j_f[0]) * inv_S1_f,
                              (a_k_f * r_k_f[1] + a_l_f * r_l_f[1]) * inv_S2_f - (a_i_f * r_i_f[1] + a_j_f * r_j_f[1]) * inv_S1_f,
                              (a_k_f * r_k_f[2] + a_l_f * r_l_f[2]) * inv_S2_f - (a_i_f * r_i_f[2] + a_j_f * r_j_f[2]) * inv_S1_f};

        const auto r2_PQ_f = PQ_f[0] * PQ_f[0] + PQ_f[1] * PQ_f[1] + PQ_f[2] * PQ_f[2];

        const auto Lambda_f = sqrtf(4.0f * S1_f * S2_f * MATH_CONST_INV_PI_F * inv_S4_f);

        float F6_t_f[7];

        gpu::computeBoysFunction_f(F6_t_f, S1_f * S2_f * inv_S4_f * r2_PQ_f, 6, boys_func_table_f, boys_func_ft_f);

        const auto QC_0_f = (a_l_f * inv_S2_f) * (r_l_f[c0] - r_k_f[c0]);
        const auto QD_0_f = (-a_k_f * inv_S2_f) * (r_l_f[d0] - r_k_f[d0]);

        const float eri_ijkl_f = Lambda_f * S_ij_00_f * S_kl_00_f * (

                + F6_t_f[2] * (

                    0.25f * S2_f * S2_f * inv_S1_f * inv_S1_f * inv_S4_f * inv_S4_f * (
                        +QC_0_f*QD_0_f*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.25f * S2_f * inv_S1_f * inv_S4_f * inv_S4_f * (
                        +delta_f[b0][b1]*delta_f[c0][d0]*(-PA_0_f*PQ_f[a1] + PQ_f[a0]*(-PA_1_f + PQ_f[a1]))

                        +delta_f[c0][d0]*(PQ_f[a0]*delta_f[a1][b0]*(-PB_1_f + PQ_f[b1]) + PQ_f[a0]*delta_f[a1][b1]*(-PB_0_f + PQ_f[b0]) + PQ_f[a1]*delta_f[a0][b0]*(-PB_1_f + PQ_f[b1]) + PQ_f[a1]*delta_f[a0][b1]*(-PB_0_f + PQ_f[b0]) + PQ_f[b0]*PQ_f[b1]*delta_f[a0][a1] - PQ_f[b0]*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) - PQ_f[b1]*(PA_0_f*delta_f[a1][b0] + PA_1_f*delta_f[a0][b0] + PB_0_f*delta_f[a0][a1]))

                        +(-PA_0_f*QC_0_f + PQ_f[a0]*QC_0_f)*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + (-PA_0_f*QD_0_f + PQ_f[a0]*QD_0_f)*(delta_f[a1][b0]*delta_f[b1][c0] + delta_f[a1][b1]*delta_f[b0][c0] + delta_f[a1][c0]*delta_f[b0][b1]) + (-PA_1_f*QC_0_f + PQ_f[a1]*QC_0_f)*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + (-PA_1_f*QD_0_f + PQ_f[a1]*QD_0_f)*(delta_f[a0][b0]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[b0][c0] + delta_f[a0][c0]*delta_f[b0][b1]) + (-PB_0_f*QC_0_f + PQ_f[b0]*QC_0_f)*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + (-PB_0_f*QD_0_f + PQ_f[b0]*QD_0_f)*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1]) + (-PB_1_f*QC_0_f + PQ_f[b1]*QC_0_f)*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0]) + (-PB_1_f*QD_0_f + PQ_f[b1]*QD_0_f)*(delta_f[a0][a1]*delta_f[b0][c0] + delta_f[a0][b0]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b0])

                        +2.0f*(PQ_f[c0]*QD_0_f + PQ_f[d0]*QC_0_f)*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.25f * inv_S4_f * inv_S4_f * (
                        +delta_f[a1][b1]*delta_f[c0][d0]*(PA_0_f*(PB_0_f - PQ_f[b0]) - PB_0_f*PQ_f[a0])

                        +PA_0_f*(PA_1_f*(delta_f[b0][c0]*delta_f[b1][d0] + delta_f[b0][d0]*delta_f[b1][c0]) + PB_0_f*(delta_f[a1][c0]*delta_f[b1][d0] + delta_f[a1][d0]*delta_f[b1][c0]) + PB_1_f*(delta_f[a1][c0]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][c0])) + PA_1_f*(PB_0_f*(delta_f[a0][c0]*delta_f[b1][d0] + delta_f[a0][d0]*delta_f[b1][c0]) + PB_1_f*(delta_f[a0][c0]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][c0])) + PB_0_f*PB_1_f*(delta_f[a0][c0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][c0])

                        +delta_f[c0][d0]*(PA_1_f*(PB_1_f*delta_f[a0][b0] + delta_f[a0][b1]*(PB_0_f - PQ_f[b0]) + delta_f[b0][b1]*(PA_0_f - PQ_f[a0])) + PB_1_f*delta_f[a0][a1]*(PB_0_f - PQ_f[b0]) + PB_1_f*delta_f[a1][b0]*(PA_0_f - PQ_f[a0]) - PQ_f[a1]*(PA_0_f*delta_f[b0][b1] + PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) - PQ_f[b1]*(PA_0_f*delta_f[a1][b0] + PA_1_f*delta_f[a0][b0] + PB_0_f*delta_f[a0][a1]))

                        -PQ_f[c0]*(PA_0_f*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + PA_1_f*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + PB_0_f*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + PB_1_f*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0])) - PQ_f[d0]*(PA_0_f*(delta_f[a1][b0]*delta_f[b1][c0] + delta_f[a1][b1]*delta_f[b0][c0] + delta_f[a1][c0]*delta_f[b0][b1]) + PA_1_f*(delta_f[a0][b0]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[b0][c0] + delta_f[a0][c0]*delta_f[b0][b1]) + PB_0_f*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1]) + PB_1_f*(delta_f[a0][a1]*delta_f[b0][c0] + delta_f[a0][b0]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b0]))

                        +PQ_f[c0]*PQ_f[d0]*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.5f * S1_f * inv_S4_f * inv_S4_f * (
                        -delta_f[c0][d0]*(PA_0_f*PA_1_f*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PB_1_f*(PA_0_f*PQ_f[a1] + PA_1_f*PQ_f[a0]))

                        -PA_0_f*PA_1_f*(PB_0_f*(PQ_f[c0]*delta_f[b1][d0] + PQ_f[d0]*delta_f[b1][c0]) + PB_1_f*(PQ_f[c0]*delta_f[b0][d0] + PQ_f[d0]*delta_f[b0][c0])) - PB_0_f*PB_1_f*(PA_0_f*(PQ_f[c0]*delta_f[a1][d0] + PQ_f[d0]*delta_f[a1][c0]) + PA_1_f*(PQ_f[c0]*delta_f[a0][d0] + PQ_f[d0]*delta_f[a0][c0]))

                        +PQ_f[c0]*PQ_f[d0]*(PA_0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) + PA_1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PB_0_f*PB_1_f*delta_f[a0][a1])
                    )

                    + 0.5f * S2_f * S2_f * inv_S1_f * inv_S4_f * inv_S4_f * (
                        +QC_0_f*QD_0_f*delta_f[b0][b1]*(-PA_0_f*PQ_f[a1] + PQ_f[a0]*(-PA_1_f + PQ_f[a1]))

                        +QC_0_f*QD_0_f*(PQ_f[a0]*delta_f[a1][b0]*(-PB_1_f + PQ_f[b1]) + PQ_f[a0]*delta_f[a1][b1]*(-PB_0_f + PQ_f[b0]) + PQ_f[a1]*delta_f[a0][b0]*(-PB_1_f + PQ_f[b1]) + PQ_f[a1]*delta_f[a0][b1]*(-PB_0_f + PQ_f[b0]) + PQ_f[b0]*PQ_f[b1]*delta_f[a0][a1] - PQ_f[b0]*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) - PQ_f[b1]*(PA_0_f*delta_f[a1][b0] + PA_1_f*delta_f[a0][b0] + PB_0_f*delta_f[a0][a1]))
                    )

                    + 0.5f * S2_f * inv_S4_f * inv_S4_f * (
                        +QD_0_f*delta_f[b1][c0]*(PA_0_f*(PA_1_f*PQ_f[b0] + PB_0_f*PQ_f[a1]) + PA_1_f*PB_0_f*PQ_f[a0])

                        +PB_1_f*(PA_0_f*(PQ_f[a1]*(QC_0_f*delta_f[b0][d0] + QD_0_f*delta_f[b0][c0]) + PQ_f[b0]*(QC_0_f*delta_f[a1][d0] + QD_0_f*delta_f[a1][c0])) + PQ_f[a0]*(PA_1_f*(QC_0_f*delta_f[b0][d0] + QD_0_f*delta_f[b0][c0]) + PB_0_f*(QC_0_f*delta_f[a1][d0] + QD_0_f*delta_f[a1][c0]))) + QC_0_f*(PA_0_f*(PA_1_f*(PQ_f[b0]*delta_f[b1][d0] + PQ_f[b1]*delta_f[b0][d0]) + PB_0_f*(PQ_f[a1]*delta_f[b1][d0] + PQ_f[b1]*delta_f[a1][d0])) + PA_1_f*PB_0_f*(PQ_f[a0]*delta_f[b1][d0] + PQ_f[b1]*delta_f[a0][d0]) + PB_1_f*delta_f[a0][d0]*(PA_1_f*PQ_f[b0] + PB_0_f*PQ_f[a1])) + QD_0_f*(PA_0_f*PQ_f[b1]*(PA_1_f*delta_f[b0][c0] + PB_0_f*delta_f[a1][c0]) + delta_f[a0][c0]*(PA_1_f*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PB_1_f*PQ_f[a1]))

                        +(PQ_f[c0]*QD_0_f + PQ_f[d0]*QC_0_f)*(PA_0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) - PA_0_f*(PQ_f[a1]*delta_f[b0][b1] + PQ_f[b0]*delta_f[a1][b1] + PQ_f[b1]*delta_f[a1][b0]) + PA_1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) - PA_1_f*(PQ_f[a0]*delta_f[b0][b1] + PQ_f[b0]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][b0]) + PB_0_f*PB_1_f*delta_f[a0][a1] - PB_0_f*(PQ_f[a0]*delta_f[a1][b1] + PQ_f[a1]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][a1]) - PB_1_f*(PQ_f[a0]*delta_f[a1][b0] + PQ_f[a1]*delta_f[a0][b0] + PQ_f[b0]*delta_f[a0][a1]))

                        +delta_f[c0][d0]*(PA_0_f*PQ_f[b0]*(PA_1_f*PQ_f[b1] + PB_1_f*PQ_f[a1]) + PA_1_f*PQ_f[a0]*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PQ_f[a1]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))
                    )

                    + S1_f * S1_f * inv_S4_f * inv_S4_f * (
                        +PA_0_f*PA_1_f*PB_0_f*PB_1_f*PQ_f[c0]*PQ_f[d0]
                    )

                    + S1_f * S2_f * inv_S4_f * inv_S4_f * (
                        -(PQ_f[c0]*QD_0_f + PQ_f[d0]*QC_0_f)*(PA_0_f*PA_1_f*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PB_1_f*(PA_0_f*PQ_f[a1] + PA_1_f*PQ_f[a0]))
                    )

                    + S2_f * S2_f * inv_S4_f * inv_S4_f * (
                        +QC_0_f*QD_0_f*(PA_0_f*PQ_f[b0]*(PA_1_f*PQ_f[b1] + PB_1_f*PQ_f[a1]) + PA_1_f*PQ_f[a0]*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PQ_f[a1]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))
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
                        +PQ_f[a0]*PQ_f[a1]*delta_f[b0][b1]*delta_f[c0][d0]

                        +delta_f[c0][d0]*(PQ_f[b0]*(PQ_f[a0]*delta_f[a1][b1] + PQ_f[a1]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][a1]) + PQ_f[b1]*(PQ_f[a0]*delta_f[a1][b0] + PQ_f[a1]*delta_f[a0][b0]))

                        +QC_0_f*(PQ_f[a0]*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + PQ_f[a1]*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + PQ_f[b0]*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + PQ_f[b1]*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0])) + QD_0_f*(PQ_f[a0]*(delta_f[a1][b0]*delta_f[b1][c0] + delta_f[a1][b1]*delta_f[b0][c0] + delta_f[a1][c0]*delta_f[b0][b1]) + PQ_f[a1]*(delta_f[a0][b0]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[b0][c0] + delta_f[a0][c0]*delta_f[b0][b1]) + PQ_f[b0]*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1]) + PQ_f[b1]*(delta_f[a0][a1]*delta_f[b0][c0] + delta_f[a0][b0]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b0]))

                        +(PQ_f[c0]*QD_0_f + PQ_f[d0]*QC_0_f)*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.25f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +delta_f[a1][b1]*delta_f[c0][d0]*(PB_0_f*PQ_f[a0] + PQ_f[b0]*(PA_0_f - PQ_f[a0]))

                        +(PA_0_f*PQ_f[a1] + PA_1_f*PQ_f[a0])*(delta_f[b0][c0]*delta_f[b1][d0] + delta_f[b0][d0]*delta_f[b1][c0]) + (PA_0_f*PQ_f[b0] + PB_0_f*PQ_f[a0])*(delta_f[a1][c0]*delta_f[b1][d0] + delta_f[a1][d0]*delta_f[b1][c0]) + (PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0])*(delta_f[a1][c0]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][c0]) + (PA_1_f*PQ_f[b0] + PB_0_f*PQ_f[a1])*(delta_f[a0][c0]*delta_f[b1][d0] + delta_f[a0][d0]*delta_f[b1][c0]) + (PA_1_f*PQ_f[b1] + PB_1_f*PQ_f[a1])*(delta_f[a0][c0]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][c0]) + (PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0])*(delta_f[a0][c0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][c0])

                        +delta_f[c0][d0]*(PA_1_f*(PQ_f[a0]*delta_f[b0][b1] + PQ_f[b0]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][b0]) + PB_1_f*(PQ_f[a0]*delta_f[a1][b0] + PQ_f[b0]*delta_f[a0][a1]) + PQ_f[a1]*(PA_0_f*delta_f[b0][b1] + PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0] - PQ_f[a0]*delta_f[b0][b1] - PQ_f[b0]*delta_f[a0][b1] - PQ_f[b1]*delta_f[a0][b0]) + PQ_f[b1]*(delta_f[a0][a1]*(PB_0_f - PQ_f[b0]) + delta_f[a1][b0]*(PA_0_f - PQ_f[a0])))

                        +(PA_0_f*PQ_f[c0] - PQ_f[a0]*PQ_f[c0])*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + (PA_0_f*PQ_f[d0] - PQ_f[a0]*PQ_f[d0])*(delta_f[a1][b0]*delta_f[b1][c0] + delta_f[a1][b1]*delta_f[b0][c0] + delta_f[a1][c0]*delta_f[b0][b1]) + (PA_1_f*PQ_f[c0] - PQ_f[a1]*PQ_f[c0])*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + (PA_1_f*PQ_f[d0] - PQ_f[a1]*PQ_f[d0])*(delta_f[a0][b0]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[b0][c0] + delta_f[a0][c0]*delta_f[b0][b1]) + (PB_0_f*PQ_f[c0] - PQ_f[b0]*PQ_f[c0])*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + (PB_0_f*PQ_f[d0] - PQ_f[b0]*PQ_f[d0])*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1]) + (PB_1_f*PQ_f[c0] - PQ_f[b1]*PQ_f[c0])*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0]) + (PB_1_f*PQ_f[d0] - PQ_f[b1]*PQ_f[d0])*(delta_f[a0][a1]*delta_f[b0][c0] + delta_f[a0][b0]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b0])

                        -2*PQ_f[c0]*PQ_f[d0]*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.5f * S1_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        -delta_f[c0][d0]*(PA_0_f*PQ_f[b0]*(PA_1_f*PQ_f[b1] + PB_1_f*PQ_f[a1]) + PA_1_f*PQ_f[a0]*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PQ_f[a1]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))

                        -PA_0_f*(PA_1_f*(PQ_f[b0]*(PQ_f[c0]*delta_f[b1][d0] + PQ_f[d0]*delta_f[b1][c0]) + PQ_f[b1]*(PQ_f[c0]*delta_f[b0][d0] + PQ_f[d0]*delta_f[b0][c0])) + PB_0_f*(PQ_f[a1]*(PQ_f[c0]*delta_f[b1][d0] + PQ_f[d0]*delta_f[b1][c0]) + PQ_f[b1]*(PQ_f[c0]*delta_f[a1][d0] + PQ_f[d0]*delta_f[a1][c0])) + PB_1_f*(PQ_f[a1]*(PQ_f[c0]*delta_f[b0][d0] + PQ_f[d0]*delta_f[b0][c0]) + PQ_f[b0]*(PQ_f[c0]*delta_f[a1][d0] + PQ_f[d0]*delta_f[a1][c0]))) - PA_1_f*(PB_0_f*(PQ_f[a0]*(PQ_f[c0]*delta_f[b1][d0] + PQ_f[d0]*delta_f[b1][c0]) + PQ_f[b1]*(PQ_f[c0]*delta_f[a0][d0] + PQ_f[d0]*delta_f[a0][c0])) + PB_1_f*(PQ_f[a0]*(PQ_f[c0]*delta_f[b0][d0] + PQ_f[d0]*delta_f[b0][c0]) + PQ_f[b0]*(PQ_f[c0]*delta_f[a0][d0] + PQ_f[d0]*delta_f[a0][c0]))) - PB_0_f*PB_1_f*(PQ_f[a0]*(PQ_f[c0]*delta_f[a1][d0] + PQ_f[d0]*delta_f[a1][c0]) + PQ_f[a1]*(PQ_f[c0]*delta_f[a0][d0] + PQ_f[d0]*delta_f[a0][c0]))

                        +PQ_f[c0]*PQ_f[d0]*(PA_0_f*(-PA_1_f*delta_f[b0][b1] - PB_0_f*delta_f[a1][b1] - PB_1_f*delta_f[a1][b0] + PQ_f[a1]*delta_f[b0][b1] + PQ_f[b0]*delta_f[a1][b1] + PQ_f[b1]*delta_f[a1][b0]) + PA_1_f*(PQ_f[a0]*delta_f[b0][b1] + delta_f[a0][b0]*(-PB_1_f + PQ_f[b1]) + delta_f[a0][b1]*(-PB_0_f + PQ_f[b0])) + PB_0_f*(PQ_f[a0]*delta_f[a1][b1] + PQ_f[a1]*delta_f[a0][b1] + delta_f[a0][a1]*(-PB_1_f + PQ_f[b1])) + PB_1_f*(PQ_f[a0]*delta_f[a1][b0] + PQ_f[a1]*delta_f[a0][b0] + PQ_f[b0]*delta_f[a0][a1]))
                    )

                    + (-0.5f) * S2_f * S2_f * S2_f * inv_S1_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[a0]*PQ_f[a1]*QC_0_f*QD_0_f*delta_f[b0][b1]

                        +QC_0_f*QD_0_f*(PQ_f[b0]*(PQ_f[a0]*delta_f[a1][b1] + PQ_f[a1]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][a1]) + PQ_f[b1]*(PQ_f[a0]*delta_f[a1][b0] + PQ_f[a1]*delta_f[a0][b0]))
                    )

                    + 0.5f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +QD_0_f*delta_f[b1][c0]*(PA_1_f*PQ_f[a0]*PQ_f[b0] + PQ_f[a1]*(PA_0_f*PQ_f[b0] + PB_0_f*PQ_f[a0]))

                        +PB_1_f*PQ_f[a0]*(PQ_f[a1]*(QC_0_f*delta_f[b0][d0] + QD_0_f*delta_f[b0][c0]) + PQ_f[b0]*(QC_0_f*delta_f[a1][d0] + QD_0_f*delta_f[a1][c0])) + QC_0_f*(PQ_f[a0]*(PA_1_f*(PQ_f[b0]*delta_f[b1][d0] + PQ_f[b1]*delta_f[b0][d0]) + PB_0_f*(PQ_f[a1]*delta_f[b1][d0] + PQ_f[b1]*delta_f[a1][d0])) + PQ_f[a1]*(PA_0_f*(PQ_f[b0]*delta_f[b1][d0] + PQ_f[b1]*delta_f[b0][d0]) + delta_f[a0][d0]*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0])) + PQ_f[b0]*PQ_f[b1]*(PA_0_f*delta_f[a1][d0] + PA_1_f*delta_f[a0][d0])) + QD_0_f*(PB_1_f*PQ_f[a1]*PQ_f[b0]*delta_f[a0][c0] + PQ_f[b1]*(PA_0_f*(PQ_f[a1]*delta_f[b0][c0] + PQ_f[b0]*delta_f[a1][c0]) + PA_1_f*(PQ_f[a0]*delta_f[b0][c0] + PQ_f[b0]*delta_f[a0][c0]) + PB_0_f*(PQ_f[a0]*delta_f[a1][c0] + PQ_f[a1]*delta_f[a0][c0])))

                        +(PQ_f[c0]*QD_0_f + PQ_f[d0]*QC_0_f)*(PA_0_f*(PQ_f[a1]*delta_f[b0][b1] + PQ_f[b0]*delta_f[a1][b1] + PQ_f[b1]*delta_f[a1][b0]) + PQ_f[a0]*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0] - PQ_f[a1]*delta_f[b0][b1] - PQ_f[b0]*delta_f[a1][b1] - PQ_f[b1]*delta_f[a1][b0]) + PQ_f[a1]*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PQ_f[b0]*(PB_1_f*delta_f[a0][a1] + delta_f[a0][b1]*(PA_1_f - PQ_f[a1])) + PQ_f[b1]*(delta_f[a0][a1]*(PB_0_f - PQ_f[b0]) + delta_f[a0][b0]*(PA_1_f - PQ_f[a1])))

                        +delta_f[c0][d0]*(PQ_f[a0]*PQ_f[b1]*(PA_1_f*PQ_f[b0] + PB_0_f*PQ_f[a1]) + PQ_f[a1]*PQ_f[b0]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))
                    )

                    + S1_f * S1_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[c0]*PQ_f[d0]*(PA_0_f*PA_1_f*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PB_1_f*(PA_0_f*PQ_f[a1] + PA_1_f*PQ_f[a0]))
                    )

                    + S1_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        -(PQ_f[c0]*QD_0_f + PQ_f[d0]*QC_0_f)*(PA_0_f*PQ_f[b0]*(PA_1_f*PQ_f[b1] + PB_1_f*PQ_f[a1]) + PA_1_f*PQ_f[a0]*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PQ_f[a1]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))
                    )

                    + S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +QC_0_f*QD_0_f*(PQ_f[a0]*PQ_f[b1]*(PA_1_f*PQ_f[b0] + PB_0_f*PQ_f[a1]) + PQ_f[a1]*PQ_f[b0]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))
                    )

                )

                + F6_t_f[4] * (

                    0.5f * S1_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        -delta_f[c0][d0]*(PQ_f[a0]*PQ_f[b1]*(PA_1_f*PQ_f[b0] + PB_0_f*PQ_f[a1]) + PQ_f[a1]*PQ_f[b0]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))

                        -PQ_f[a0]*(PA_1_f*(PQ_f[b0]*(PQ_f[c0]*delta_f[b1][d0] + PQ_f[d0]*delta_f[b1][c0]) + PQ_f[b1]*(PQ_f[c0]*delta_f[b0][d0] + PQ_f[d0]*delta_f[b0][c0])) + PB_0_f*(PQ_f[a1]*(PQ_f[c0]*delta_f[b1][d0] + PQ_f[d0]*delta_f[b1][c0]) + PQ_f[b1]*(PQ_f[c0]*delta_f[a1][d0] + PQ_f[d0]*delta_f[a1][c0])) + PB_1_f*(PQ_f[a1]*(PQ_f[c0]*delta_f[b0][d0] + PQ_f[d0]*delta_f[b0][c0]) + PQ_f[b0]*(PQ_f[c0]*delta_f[a1][d0] + PQ_f[d0]*delta_f[a1][c0]))) - PQ_f[a1]*(PA_0_f*(PQ_f[b0]*(PQ_f[c0]*delta_f[b1][d0] + PQ_f[d0]*delta_f[b1][c0]) + PQ_f[b1]*(PQ_f[c0]*delta_f[b0][d0] + PQ_f[d0]*delta_f[b0][c0])) + (PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0])*(PQ_f[c0]*delta_f[a0][d0] + PQ_f[d0]*delta_f[a0][c0])) - PQ_f[b0]*PQ_f[b1]*(PA_0_f*(PQ_f[c0]*delta_f[a1][d0] + PQ_f[d0]*delta_f[a1][c0]) + PA_1_f*(PQ_f[c0]*delta_f[a0][d0] + PQ_f[d0]*delta_f[a0][c0]))

                        +PQ_f[c0]*PQ_f[d0]*(-PA_0_f*(PQ_f[a1]*delta_f[b0][b1] + PQ_f[b0]*delta_f[a1][b1] + PQ_f[b1]*delta_f[a1][b0]) - PQ_f[a0]*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) + PQ_f[a0]*(PQ_f[a1]*delta_f[b0][b1] + PQ_f[b0]*delta_f[a1][b1] + PQ_f[b1]*delta_f[a1][b0]) - PQ_f[a1]*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PQ_f[a1]*(PQ_f[b0]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][b0]) + PQ_f[b0]*PQ_f[b1]*delta_f[a0][a1] - PQ_f[b0]*(PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) - PQ_f[b1]*(PA_1_f*delta_f[a0][b0] + PB_0_f*delta_f[a0][a1]))
                    )

                    + 0.5f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[a0]*PQ_f[a1]*PQ_f[b0]*QD_0_f*delta_f[b1][c0]

                        +PQ_f[a0]*PQ_f[a1]*QC_0_f*(PQ_f[b0]*delta_f[b1][d0] + PQ_f[b1]*delta_f[b0][d0]) + PQ_f[b1]*(PQ_f[a0]*QD_0_f*(PQ_f[a1]*delta_f[b0][c0] + PQ_f[b0]*delta_f[a1][c0]) + PQ_f[b0]*(PQ_f[a1]*QD_0_f*delta_f[a0][c0] + QC_0_f*(PQ_f[a0]*delta_f[a1][d0] + PQ_f[a1]*delta_f[a0][d0])))

                        +(PQ_f[c0]*QD_0_f + PQ_f[d0]*QC_0_f)*(PQ_f[a0]*(PQ_f[a1]*delta_f[b0][b1] + PQ_f[b0]*delta_f[a1][b1] + PQ_f[b1]*delta_f[a1][b0]) + PQ_f[a1]*(PQ_f[b0]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][b0]) + PQ_f[b0]*PQ_f[b1]*delta_f[a0][a1])

                        +PQ_f[a0]*PQ_f[a1]*PQ_f[b0]*PQ_f[b1]*delta_f[c0][d0]
                    )

                    + S1_f * S1_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[c0]*PQ_f[d0]*(PA_0_f*PQ_f[b0]*(PA_1_f*PQ_f[b1] + PB_1_f*PQ_f[a1]) + PA_1_f*PQ_f[a0]*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PQ_f[a1]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))
                    )

                    + S1_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        -(PQ_f[c0]*QD_0_f + PQ_f[d0]*QC_0_f)*(PQ_f[a0]*PQ_f[b1]*(PA_1_f*PQ_f[b0] + PB_0_f*PQ_f[a1]) + PQ_f[a1]*PQ_f[b0]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))
                    )

                    + S2_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[a0]*PQ_f[a1]*PQ_f[b0]*PQ_f[b1]*QC_0_f*QD_0_f
                    )

                    + 0.25f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[a0]*PQ_f[a1]*(delta_f[b0][b1]*delta_f[c0][d0] + delta_f[b0][c0]*delta_f[b1][d0] + delta_f[b0][d0]*delta_f[b1][c0])

                        +PQ_f[b0]*(PQ_f[a0]*(delta_f[a1][b1]*delta_f[c0][d0] + delta_f[a1][c0]*delta_f[b1][d0] + delta_f[a1][d0]*delta_f[b1][c0]) + PQ_f[a1]*(delta_f[a0][b1]*delta_f[c0][d0] + delta_f[a0][c0]*delta_f[b1][d0] + delta_f[a0][d0]*delta_f[b1][c0]) + PQ_f[b1]*(delta_f[a0][a1]*delta_f[c0][d0] + delta_f[a0][c0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][c0]) + PQ_f[c0]*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + PQ_f[d0]*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1])) + PQ_f[b1]*(PQ_f[a0]*(delta_f[a1][b0]*delta_f[c0][d0] + delta_f[a1][c0]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][c0]) + PQ_f[a1]*(delta_f[a0][b0]*delta_f[c0][d0] + delta_f[a0][c0]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][c0]) + PQ_f[c0]*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0]) + PQ_f[d0]*(delta_f[a0][a1]*delta_f[b0][c0] + delta_f[a0][b0]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b0])) + PQ_f[c0]*(PQ_f[a0]*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + PQ_f[a1]*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + PQ_f[d0]*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])) + PQ_f[d0]*(PQ_f[a0]*(delta_f[a1][b0]*delta_f[b1][c0] + delta_f[a1][b1]*delta_f[b0][c0] + delta_f[a1][c0]*delta_f[b0][b1]) + PQ_f[a1]*(delta_f[a0][b0]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[b0][c0] + delta_f[a0][c0]*delta_f[b0][b1]))
                    )

                )

                + F6_t_f[5] * (

                    (-0.5f) * S1_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[a0]*PQ_f[a1]*PQ_f[b0]*PQ_f[b1]*delta_f[c0][d0]

                        +PQ_f[c0]*(PQ_f[a0]*(PQ_f[a1]*(PQ_f[b0]*delta_f[b1][d0] + PQ_f[b1]*delta_f[b0][d0] + PQ_f[d0]*delta_f[b0][b1]) + PQ_f[b1]*PQ_f[d0]*delta_f[a1][b0]) + PQ_f[b0]*PQ_f[b1]*(PQ_f[a0]*delta_f[a1][d0] + PQ_f[a1]*delta_f[a0][d0] + PQ_f[d0]*delta_f[a0][a1])) + PQ_f[d0]*(PQ_f[a1]*PQ_f[b1]*(PQ_f[a0]*delta_f[b0][c0] + PQ_f[b0]*delta_f[a0][c0] + PQ_f[c0]*delta_f[a0][b0]) + PQ_f[b0]*(PQ_f[a0]*(PQ_f[a1]*delta_f[b1][c0] + PQ_f[b1]*delta_f[a1][c0] + PQ_f[c0]*delta_f[a1][b1]) + PQ_f[a1]*PQ_f[c0]*delta_f[a0][b1]))
                    )

                    + S1_f * S1_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[c0]*PQ_f[d0]*(PQ_f[a0]*PQ_f[b1]*(PA_1_f*PQ_f[b0] + PB_0_f*PQ_f[a1]) + PQ_f[a1]*PQ_f[b0]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))
                    )

                    + S1_f * S2_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[a0]*PQ_f[a1]*PQ_f[b0]*PQ_f[b1]*(-PQ_f[c0]*QD_0_f - PQ_f[d0]*QC_0_f)
                    )

                )

                + F6_t_f[6] * (

                    S1_f * S1_f * S2_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[a0]*PQ_f[a1]*PQ_f[b0]*PQ_f[b1]*PQ_f[c0]*PQ_f[d0]
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
        double J_ij = 0.0;

        for (uint32_t n = 0; n < TILE_DIM_LARGE; n++)
        {
            J_ij += ERIs[n];
        }

        mat_J[ij] += J_ij;
    }
}

// DDPP altB DDPP0 BASE
__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDPP_altB0(double*         mat_J,
                       const double*   p_prim_info,
                       const uint32_t  p_prim_count,
                       const double*   d_prim_info,
                       const uint32_t  d_prim_count,
                       const double*   pp_mat_D,
                       const double*   dd_mat_Q_local,
                       const double*   pp_mat_Q,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const double*   dd_pair_data_local,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* pp_first_inds,
                       const uint32_t* pp_second_inds,
                       const double*   pp_pair_data,
                       const uint32_t  pp_prim_pair_count,
                       const double*   boys_func_table,
                       const double*   boys_func_ft,
                       const double    eri_threshold)
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

    for (uint32_t m = 0; m < (pp_prim_pair_count + TILE_DIM_LARGE - 1) / TILE_DIM_LARGE; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= pp_prim_pair_count) || (fabs(dd_mat_Q_local[ij] * pp_mat_Q[kl] * pp_mat_D[kl]) <= eri_threshold))
        {
            break;
        }

        const auto k = pp_first_inds[kl];
        const auto l = pp_second_inds[kl];

        const auto a_k = p_prim_info[k / 3 + p_prim_count * 0];

        const double r_k[3] = {p_prim_info[k / 3 + p_prim_count * 2],
                               p_prim_info[k / 3 + p_prim_count * 3],
                               p_prim_info[k / 3 + p_prim_count * 4]};

        const auto a_l = p_prim_info[l / 3 + p_prim_count * 0];

        const double r_l[3] = {p_prim_info[l / 3 + p_prim_count * 2],
                               p_prim_info[l / 3 + p_prim_count * 3],
                               p_prim_info[l / 3 + p_prim_count * 4]};

        const auto S_kl_00 = pp_pair_data[kl];

        const auto c0 = k % 3;
        const auto d0 = l % 3;

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F6_t[2];

        gpu::computeBoysFunction(F6_t, S1 * S2 * inv_S4 * r2_PQ, 1, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                + F6_t[0] * (

                    0.25 * inv_S1 * inv_S1 * (
                        +QC_0*QD_0*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * inv_S1 * inv_S2 * (
                        +PA_0*PB_0*delta[a1][b1]*delta[c0][d0]

                        +delta[c0][d0]*(PA_1*(PA_0*delta[b0][b1] + PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_1*(PA_0*delta[a1][b0] + PB_0*delta[a0][a1]))
                    )

                    + 0.5 * inv_S1 * (
                        +PA_0*PA_1*QC_0*QD_0*delta[b0][b1]

                        +QC_0*QD_0*(PB_0*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) + PB_1*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0]))
                    )

                    + 0.5 * inv_S2 * (
                        +PA_0*PA_1*PB_0*PB_1*delta[c0][d0]
                    )

                    + (
                        +PA_0*PA_1*PB_0*PB_1*QC_0*QD_0
                    )

                    + 0.125 * inv_S1 * inv_S1 * inv_S2 * (
                        +delta[c0][d0]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                )

                + F6_t[1] * (

                    (-0.25) * inv_S1 * inv_S1 * inv_S4 * (
                        +delta[c0][d0]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + (-0.125) * inv_S1 * inv_S2 * inv_S4 * (
                        +delta[c0][d0]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + (-0.5) * S2 * inv_S1 * inv_S1 * inv_S4 * (
                        +QC_0*QD_0*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * inv_S1 * inv_S4 * (
                        +delta[b0][b1]*delta[c0][d0]*(PA_0*(-PA_1 + PQ[a1]) + PA_1*PQ[a0])

                        +delta[c0][d0]*(PB_0*(-PA_0*delta[a1][b1] - PA_1*delta[a0][b1] - PB_1*delta[a0][a1] + PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + PQ[b0]*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) + delta[a0][b0]*(PA_1*(-PB_1 + PQ[b1]) + PB_1*PQ[a1]) + delta[a1][b0]*(PA_0*(-PB_1 + PQ[b1]) + PB_1*PQ[a0]))

                        +QC_0*(PA_0*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])) + QD_0*(PA_0*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0]))

                        -(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + (-0.25) * inv_S2 * inv_S4 * (
                        +PA_0*PA_1*delta[b0][b1]*delta[c0][d0]

                        +delta[c0][d0]*(PB_0*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) + PB_1*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0]))
                    )

                    + (-0.5) * S1 * inv_S2 * inv_S4 * (
                        +PA_0*PA_1*PB_0*PB_1*delta[c0][d0]
                    )

                    + 0.5 * S2 * inv_S1 * inv_S4 * (
                        +QC_0*QD_0*delta[b0][b1]*(PA_0*(-PA_1 + PQ[a1]) + PA_1*PQ[a0])

                        +QC_0*QD_0*(PB_0*(-PA_0*delta[a1][b1] - PA_1*delta[a0][b1] - PB_1*delta[a0][a1] + PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + PQ[b0]*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) + delta[a0][b0]*(PA_1*(-PB_1 + PQ[b1]) + PB_1*PQ[a1]) + delta[a1][b0]*(PA_0*(-PB_1 + PQ[b1]) + PB_1*PQ[a0]))
                    )

                    + 0.5 * inv_S4 * (
                        +delta[c0][d0]*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))

                        +PA_0*PA_1*(PB_0*(QC_0*delta[b1][d0] + QD_0*delta[b1][c0]) + PB_1*(QC_0*delta[b0][d0] + QD_0*delta[b0][c0])) + PB_0*PB_1*(PA_0*(QC_0*delta[a1][d0] + QD_0*delta[a1][c0]) + PA_1*(QC_0*delta[a0][d0] + QD_0*delta[a0][c0]))

                        -(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1])
                    )

                    + S1 * inv_S4 * (
                        +PA_0*PA_1*PB_0*PB_1*(-PQ[c0]*QD_0 - PQ[d0]*QC_0)
                    )

                    + S2 * inv_S4 * (
                        +QC_0*QD_0*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))
                    )

                )

            );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * pp_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

// DDPP altB DDPP0 FP64
__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDPP_altB0_FP64(double*         mat_J,
                       const double*   p_prim_info,
                       const uint32_t  p_prim_count,
                       const double*   d_prim_info,
                       const uint32_t  d_prim_count,
                       const double*   pp_mat_D,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const double*   dd_pair_data_local,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* pp_first_inds,
                       const uint32_t* pp_second_inds,
                       const double*   pp_pair_data,
                       const uint32_t  pp_prim_pair_count,
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

        if ((ij >= dd_prim_pair_count_local) || (kl >= pp_prim_pair_count))
        {
            break;
        }

        const auto k = pp_first_inds[kl];
        const auto l = pp_second_inds[kl];

        const auto a_k = p_prim_info[k / 3 + p_prim_count * 0];

        const double r_k[3] = {p_prim_info[k / 3 + p_prim_count * 2],
                               p_prim_info[k / 3 + p_prim_count * 3],
                               p_prim_info[k / 3 + p_prim_count * 4]};

        const auto a_l = p_prim_info[l / 3 + p_prim_count * 0];

        const double r_l[3] = {p_prim_info[l / 3 + p_prim_count * 2],
                               p_prim_info[l / 3 + p_prim_count * 3],
                               p_prim_info[l / 3 + p_prim_count * 4]};

        const auto S_kl_00 = pp_pair_data[kl];

        const auto c0 = k % 3;
        const auto d0 = l % 3;

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F6_t[2];

        gpu::computeBoysFunction(F6_t, S1 * S2 * inv_S4 * r2_PQ, 1, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                + F6_t[0] * (

                    0.25 * inv_S1 * inv_S1 * (
                        +QC_0*QD_0*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * inv_S1 * inv_S2 * (
                        +PA_0*PB_0*delta[a1][b1]*delta[c0][d0]

                        +delta[c0][d0]*(PA_1*(PA_0*delta[b0][b1] + PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_1*(PA_0*delta[a1][b0] + PB_0*delta[a0][a1]))
                    )

                    + 0.5 * inv_S1 * (
                        +PA_0*PA_1*QC_0*QD_0*delta[b0][b1]

                        +QC_0*QD_0*(PB_0*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) + PB_1*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0]))
                    )

                    + 0.5 * inv_S2 * (
                        +PA_0*PA_1*PB_0*PB_1*delta[c0][d0]
                    )

                    + (
                        +PA_0*PA_1*PB_0*PB_1*QC_0*QD_0
                    )

                    + 0.125 * inv_S1 * inv_S1 * inv_S2 * (
                        +delta[c0][d0]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                )

                + F6_t[1] * (

                    (-0.25) * inv_S1 * inv_S1 * inv_S4 * (
                        +delta[c0][d0]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + (-0.125) * inv_S1 * inv_S2 * inv_S4 * (
                        +delta[c0][d0]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + (-0.5) * S2 * inv_S1 * inv_S1 * inv_S4 * (
                        +QC_0*QD_0*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * inv_S1 * inv_S4 * (
                        +delta[b0][b1]*delta[c0][d0]*(PA_0*(-PA_1 + PQ[a1]) + PA_1*PQ[a0])

                        +delta[c0][d0]*(PB_0*(-PA_0*delta[a1][b1] - PA_1*delta[a0][b1] - PB_1*delta[a0][a1] + PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + PQ[b0]*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) + delta[a0][b0]*(PA_1*(-PB_1 + PQ[b1]) + PB_1*PQ[a1]) + delta[a1][b0]*(PA_0*(-PB_1 + PQ[b1]) + PB_1*PQ[a0]))

                        +QC_0*(PA_0*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])) + QD_0*(PA_0*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0]))

                        -(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + (-0.25) * inv_S2 * inv_S4 * (
                        +PA_0*PA_1*delta[b0][b1]*delta[c0][d0]

                        +delta[c0][d0]*(PB_0*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) + PB_1*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0]))
                    )

                    + (-0.5) * S1 * inv_S2 * inv_S4 * (
                        +PA_0*PA_1*PB_0*PB_1*delta[c0][d0]
                    )

                    + 0.5 * S2 * inv_S1 * inv_S4 * (
                        +QC_0*QD_0*delta[b0][b1]*(PA_0*(-PA_1 + PQ[a1]) + PA_1*PQ[a0])

                        +QC_0*QD_0*(PB_0*(-PA_0*delta[a1][b1] - PA_1*delta[a0][b1] - PB_1*delta[a0][a1] + PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + PQ[b0]*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) + delta[a0][b0]*(PA_1*(-PB_1 + PQ[b1]) + PB_1*PQ[a1]) + delta[a1][b0]*(PA_0*(-PB_1 + PQ[b1]) + PB_1*PQ[a0]))
                    )

                    + 0.5 * inv_S4 * (
                        +delta[c0][d0]*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))

                        +PA_0*PA_1*(PB_0*(QC_0*delta[b1][d0] + QD_0*delta[b1][c0]) + PB_1*(QC_0*delta[b0][d0] + QD_0*delta[b0][c0])) + PB_0*PB_1*(PA_0*(QC_0*delta[a1][d0] + QD_0*delta[a1][c0]) + PA_1*(QC_0*delta[a0][d0] + QD_0*delta[a0][c0]))

                        -(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1])
                    )

                    + S1 * inv_S4 * (
                        +PA_0*PA_1*PB_0*PB_1*(-PQ[c0]*QD_0 - PQ[d0]*QC_0)
                    )

                    + S2 * inv_S4 * (
                        +QC_0*QD_0*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))
                    )

                )

            );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * pp_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

// DDPP altB DDPP0 FP32
__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDPP_altB0_FP32(double*         mat_J,
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

        const float r_k_f[3] = {p_prim_info_f[k / 3 + p_prim_count * 2],
                               p_prim_info_f[k / 3 + p_prim_count * 3],
                               p_prim_info_f[k / 3 + p_prim_count * 4]};

        const auto a_l_f = p_prim_info_f[l / 3 + p_prim_count * 0];

        const float r_l_f[3] = {p_prim_info_f[l / 3 + p_prim_count * 2],
                               p_prim_info_f[l / 3 + p_prim_count * 3],
                               p_prim_info_f[l / 3 + p_prim_count * 4]};

        const auto S_kl_00_f = pp_pair_data_f[kl];

        const auto c0 = k % 3;
        const auto d0 = l % 3;

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2_f = a_k_f + a_l_f;

        const auto inv_S2_f = 1.0f / S2_f;
        const auto inv_S4_f = 1.0f / (S1_f + S2_f);

        const float PQ_f[3] = {(a_k_f * r_k_f[0] + a_l_f * r_l_f[0]) * inv_S2_f - (a_i_f * r_i_f[0] + a_j_f * r_j_f[0]) * inv_S1_f,
                              (a_k_f * r_k_f[1] + a_l_f * r_l_f[1]) * inv_S2_f - (a_i_f * r_i_f[1] + a_j_f * r_j_f[1]) * inv_S1_f,
                              (a_k_f * r_k_f[2] + a_l_f * r_l_f[2]) * inv_S2_f - (a_i_f * r_i_f[2] + a_j_f * r_j_f[2]) * inv_S1_f};

        const auto r2_PQ_f = PQ_f[0] * PQ_f[0] + PQ_f[1] * PQ_f[1] + PQ_f[2] * PQ_f[2];

        const auto Lambda_f = sqrtf(4.0f * S1_f * S2_f * MATH_CONST_INV_PI_F * inv_S4_f);

        float F6_t_f[2];

        gpu::computeBoysFunction_f(F6_t_f, S1_f * S2_f * inv_S4_f * r2_PQ_f, 1, boys_func_table_f, boys_func_ft_f);

        const auto QC_0_f = (a_l_f * inv_S2_f) * (r_l_f[c0] - r_k_f[c0]);
        const auto QD_0_f = (-a_k_f * inv_S2_f) * (r_l_f[d0] - r_k_f[d0]);

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
                        +delta_f[b0][b1]*delta_f[c0][d0]*(PA_0_f*(-PA_1_f + PQ_f[a1]) + PA_1_f*PQ_f[a0])

                        +delta_f[c0][d0]*(PB_0_f*(-PA_0_f*delta_f[a1][b1] - PA_1_f*delta_f[a0][b1] - PB_1_f*delta_f[a0][a1] + PQ_f[a0]*delta_f[a1][b1] + PQ_f[a1]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][a1]) + PQ_f[b0]*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) + delta_f[a0][b0]*(PA_1_f*(-PB_1_f + PQ_f[b1]) + PB_1_f*PQ_f[a1]) + delta_f[a1][b0]*(PA_0_f*(-PB_1_f + PQ_f[b1]) + PB_1_f*PQ_f[a0]))

                        +QC_0_f*(PA_0_f*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + PA_1_f*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + PB_0_f*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + PB_1_f*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0])) + QD_0_f*(PA_0_f*(delta_f[a1][b0]*delta_f[b1][c0] + delta_f[a1][b1]*delta_f[b0][c0] + delta_f[a1][c0]*delta_f[b0][b1]) + PA_1_f*(delta_f[a0][b0]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[b0][c0] + delta_f[a0][c0]*delta_f[b0][b1]) + PB_0_f*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1]) + PB_1_f*(delta_f[a0][a1]*delta_f[b0][c0] + delta_f[a0][b0]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b0]))

                        -(PQ_f[c0]*QD_0_f + PQ_f[d0]*QC_0_f)*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + (-0.25f) * inv_S2_f * inv_S4_f * (
                        +PA_0_f*PA_1_f*delta_f[b0][b1]*delta_f[c0][d0]

                        +delta_f[c0][d0]*(PB_0_f*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) + PB_1_f*(PA_0_f*delta_f[a1][b0] + PA_1_f*delta_f[a0][b0]))
                    )

                    + (-0.5f) * S1_f * inv_S2_f * inv_S4_f * (
                        +PA_0_f*PA_1_f*PB_0_f*PB_1_f*delta_f[c0][d0]
                    )

                    + 0.5f * S2_f * inv_S1_f * inv_S4_f * (
                        +QC_0_f*QD_0_f*delta_f[b0][b1]*(PA_0_f*(-PA_1_f + PQ_f[a1]) + PA_1_f*PQ_f[a0])

                        +QC_0_f*QD_0_f*(PB_0_f*(-PA_0_f*delta_f[a1][b1] - PA_1_f*delta_f[a0][b1] - PB_1_f*delta_f[a0][a1] + PQ_f[a0]*delta_f[a1][b1] + PQ_f[a1]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][a1]) + PQ_f[b0]*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) + delta_f[a0][b0]*(PA_1_f*(-PB_1_f + PQ_f[b1]) + PB_1_f*PQ_f[a1]) + delta_f[a1][b0]*(PA_0_f*(-PB_1_f + PQ_f[b1]) + PB_1_f*PQ_f[a0]))
                    )

                    + 0.5f * inv_S4_f * (
                        +delta_f[c0][d0]*(PA_0_f*PA_1_f*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PB_1_f*(PA_0_f*PQ_f[a1] + PA_1_f*PQ_f[a0]))

                        +PA_0_f*PA_1_f*(PB_0_f*(QC_0_f*delta_f[b1][d0] + QD_0_f*delta_f[b1][c0]) + PB_1_f*(QC_0_f*delta_f[b0][d0] + QD_0_f*delta_f[b0][c0])) + PB_0_f*PB_1_f*(PA_0_f*(QC_0_f*delta_f[a1][d0] + QD_0_f*delta_f[a1][c0]) + PA_1_f*(QC_0_f*delta_f[a0][d0] + QD_0_f*delta_f[a0][c0]))

                        -(PQ_f[c0]*QD_0_f + PQ_f[d0]*QC_0_f)*(PA_0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) + PA_1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PB_0_f*PB_1_f*delta_f[a0][a1])
                    )

                    + S1_f * inv_S4_f * (
                        +PA_0_f*PA_1_f*PB_0_f*PB_1_f*(-PQ_f[c0]*QD_0_f - PQ_f[d0]*QC_0_f)
                    )

                    + S2_f * inv_S4_f * (
                        +QC_0_f*QD_0_f*(PA_0_f*PA_1_f*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PB_1_f*(PA_0_f*PQ_f[a1] + PA_1_f*PQ_f[a0]))
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
        double J_ij = 0.0;

        for (uint32_t n = 0; n < TILE_DIM_LARGE; n++)
        {
            J_ij += ERIs[n];
        }

        mat_J[ij] += J_ij;
    }
}

// DDPP altB DDPP1 BASE
__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDPP_altB1(double*         mat_J,
                       const double*   p_prim_info,
                       const uint32_t  p_prim_count,
                       const double*   d_prim_info,
                       const uint32_t  d_prim_count,
                       const double*   pp_mat_D,
                       const double*   dd_mat_Q_local,
                       const double*   pp_mat_Q,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const double*   dd_pair_data_local,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* pp_first_inds,
                       const uint32_t* pp_second_inds,
                       const double*   pp_pair_data,
                       const uint32_t  pp_prim_pair_count,
                       const double*   boys_func_table,
                       const double*   boys_func_ft,
                       const double    eri_threshold)
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

    for (uint32_t m = 0; m < (pp_prim_pair_count + TILE_DIM_LARGE - 1) / TILE_DIM_LARGE; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= pp_prim_pair_count) || (fabs(dd_mat_Q_local[ij] * pp_mat_Q[kl] * pp_mat_D[kl]) <= eri_threshold))
        {
            break;
        }

        const auto k = pp_first_inds[kl];
        const auto l = pp_second_inds[kl];

        const auto a_k = p_prim_info[k / 3 + p_prim_count * 0];

        const double r_k[3] = {p_prim_info[k / 3 + p_prim_count * 2],
                               p_prim_info[k / 3 + p_prim_count * 3],
                               p_prim_info[k / 3 + p_prim_count * 4]};

        const auto a_l = p_prim_info[l / 3 + p_prim_count * 0];

        const double r_l[3] = {p_prim_info[l / 3 + p_prim_count * 2],
                               p_prim_info[l / 3 + p_prim_count * 3],
                               p_prim_info[l / 3 + p_prim_count * 4]};

        const auto S_kl_00 = pp_pair_data[kl];

        const auto c0 = k % 3;
        const auto d0 = l % 3;

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F6_t[4];

        gpu::computeBoysFunction(F6_t, S1 * S2 * inv_S4 * r2_PQ, 3, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                + F6_t[2] * (

                    0.25 * S2 * S2 * inv_S1 * inv_S1 * inv_S4 * inv_S4 * (
                        +QC_0*QD_0*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * S2 * inv_S1 * inv_S4 * inv_S4 * (
                        +delta[b0][b1]*delta[c0][d0]*(-PA_0*PQ[a1] + PQ[a0]*(-PA_1 + PQ[a1]))

                        +delta[c0][d0]*(PQ[a0]*delta[a1][b0]*(-PB_1 + PQ[b1]) + PQ[a0]*delta[a1][b1]*(-PB_0 + PQ[b0]) + PQ[a1]*delta[a0][b0]*(-PB_1 + PQ[b1]) + PQ[a1]*delta[a0][b1]*(-PB_0 + PQ[b0]) + PQ[b0]*PQ[b1]*delta[a0][a1] - PQ[b0]*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) - PQ[b1]*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0] + PB_0*delta[a0][a1]))

                        +(-PA_0*QC_0 + PQ[a0]*QC_0)*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + (-PA_0*QD_0 + PQ[a0]*QD_0)*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1]) + (-PA_1*QC_0 + PQ[a1]*QC_0)*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + (-PA_1*QD_0 + PQ[a1]*QD_0)*(delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1]) + (-PB_0*QC_0 + PQ[b0]*QC_0)*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + (-PB_0*QD_0 + PQ[b0]*QD_0)*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]) + (-PB_1*QC_0 + PQ[b1]*QC_0)*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]) + (-PB_1*QD_0 + PQ[b1]*QD_0)*(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0])

                        +2.0*(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * inv_S4 * inv_S4 * (
                        +delta[a1][b1]*delta[c0][d0]*(PA_0*(PB_0 - PQ[b0]) - PB_0*PQ[a0])

                        +PA_0*(PA_1*(delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0]) + PB_0*(delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + PB_1*(delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0])) + PA_1*(PB_0*(delta[a0][c0]*delta[b1][d0] + delta[a0][d0]*delta[b1][c0]) + PB_1*(delta[a0][c0]*delta[b0][d0] + delta[a0][d0]*delta[b0][c0])) + PB_0*PB_1*(delta[a0][c0]*delta[a1][d0] + delta[a0][d0]*delta[a1][c0])

                        +delta[c0][d0]*(PA_1*(PB_1*delta[a0][b0] + delta[a0][b1]*(PB_0 - PQ[b0]) + delta[b0][b1]*(PA_0 - PQ[a0])) + PB_1*delta[a0][a1]*(PB_0 - PQ[b0]) + PB_1*delta[a1][b0]*(PA_0 - PQ[a0]) - PQ[a1]*(PA_0*delta[b0][b1] + PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) - PQ[b1]*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0] + PB_0*delta[a0][a1]))

                        -PQ[c0]*(PA_0*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])) - PQ[d0]*(PA_0*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0]))

                        +PQ[c0]*PQ[d0]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.5 * S1 * inv_S4 * inv_S4 * (
                        -delta[c0][d0]*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))

                        -PA_0*PA_1*(PB_0*(PQ[c0]*delta[b1][d0] + PQ[d0]*delta[b1][c0]) + PB_1*(PQ[c0]*delta[b0][d0] + PQ[d0]*delta[b0][c0])) - PB_0*PB_1*(PA_0*(PQ[c0]*delta[a1][d0] + PQ[d0]*delta[a1][c0]) + PA_1*(PQ[c0]*delta[a0][d0] + PQ[d0]*delta[a0][c0]))

                        +PQ[c0]*PQ[d0]*(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1])
                    )

                    + 0.5 * S2 * S2 * inv_S1 * inv_S4 * inv_S4 * (
                        +QC_0*QD_0*delta[b0][b1]*(-PA_0*PQ[a1] + PQ[a0]*(-PA_1 + PQ[a1]))

                        +QC_0*QD_0*(PQ[a0]*delta[a1][b0]*(-PB_1 + PQ[b1]) + PQ[a0]*delta[a1][b1]*(-PB_0 + PQ[b0]) + PQ[a1]*delta[a0][b0]*(-PB_1 + PQ[b1]) + PQ[a1]*delta[a0][b1]*(-PB_0 + PQ[b0]) + PQ[b0]*PQ[b1]*delta[a0][a1] - PQ[b0]*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) - PQ[b1]*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0] + PB_0*delta[a0][a1]))
                    )

                    + 0.5 * S2 * inv_S4 * inv_S4 * (
                        +QD_0*delta[b1][c0]*(PA_0*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PA_1*PB_0*PQ[a0])

                        +PB_1*(PA_0*(PQ[a1]*(QC_0*delta[b0][d0] + QD_0*delta[b0][c0]) + PQ[b0]*(QC_0*delta[a1][d0] + QD_0*delta[a1][c0])) + PQ[a0]*(PA_1*(QC_0*delta[b0][d0] + QD_0*delta[b0][c0]) + PB_0*(QC_0*delta[a1][d0] + QD_0*delta[a1][c0]))) + QC_0*(PA_0*(PA_1*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0]) + PB_0*(PQ[a1]*delta[b1][d0] + PQ[b1]*delta[a1][d0])) + PA_1*PB_0*(PQ[a0]*delta[b1][d0] + PQ[b1]*delta[a0][d0]) + PB_1*delta[a0][d0]*(PA_1*PQ[b0] + PB_0*PQ[a1])) + QD_0*(PA_0*PQ[b1]*(PA_1*delta[b0][c0] + PB_0*delta[a1][c0]) + delta[a0][c0]*(PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*PQ[a1]))

                        +(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) - PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) - PA_1*(PQ[a0]*delta[b0][b1] + PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1] - PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) - PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]))

                        +delta[c0][d0]*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S1 * S1 * inv_S4 * inv_S4 * (
                        +PA_0*PA_1*PB_0*PB_1*PQ[c0]*PQ[d0]
                    )

                    + S1 * S2 * inv_S4 * inv_S4 * (
                        -(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))
                    )

                    + S2 * S2 * inv_S4 * inv_S4 * (
                        +QC_0*QD_0*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + 0.125 * S2 * inv_S1 * inv_S1 * inv_S4 * inv_S4 * (
                        +delta[c0][d0]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.125 * inv_S1 * inv_S4 * inv_S4 * (
                        +2.0*delta[c0][d0]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])

                        +delta[b0][b1]*(delta[a0][c0]*delta[a1][d0] + delta[a0][d0]*delta[a1][c0]) + delta[b0][c0]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + delta[b0][d0]*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]) + delta[b1][c0]*(delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]) + delta[b1][d0]*(delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0])
                    )

                )

                + F6_t[3] * (

                    (-0.125) * S2 * inv_S1 * inv_S4 * inv_S4 * inv_S4 * (
                        +delta[a0][a1]*(delta[b0][b1]*delta[c0][d0] + delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0]) + delta[a0][b0]*(delta[a1][b1]*delta[c0][d0] + delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + delta[a0][b1]*(delta[a1][b0]*delta[c0][d0] + delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0]) + delta[a0][c0]*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + delta[a0][d0]*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1])
                    )

                    + (-0.25) * S2 * S2 * inv_S1 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*delta[b0][b1]*delta[c0][d0]

                        +delta[c0][d0]*(PQ[b0]*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + PQ[b1]*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0]))

                        +QC_0*(PQ[a0]*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + PQ[b0]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + PQ[b1]*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])) + QD_0*(PQ[a0]*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1]) + PQ[b0]*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]) + PQ[b1]*(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0]))

                        +(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        +delta[a1][b1]*delta[c0][d0]*(PB_0*PQ[a0] + PQ[b0]*(PA_0 - PQ[a0]))

                        +(PA_0*PQ[a1] + PA_1*PQ[a0])*(delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0]) + (PA_0*PQ[b0] + PB_0*PQ[a0])*(delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + (PA_0*PQ[b1] + PB_1*PQ[a0])*(delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0]) + (PA_1*PQ[b0] + PB_0*PQ[a1])*(delta[a0][c0]*delta[b1][d0] + delta[a0][d0]*delta[b1][c0]) + (PA_1*PQ[b1] + PB_1*PQ[a1])*(delta[a0][c0]*delta[b0][d0] + delta[a0][d0]*delta[b0][c0]) + (PB_0*PQ[b1] + PB_1*PQ[b0])*(delta[a0][c0]*delta[a1][d0] + delta[a0][d0]*delta[a1][c0])

                        +delta[c0][d0]*(PA_1*(PQ[a0]*delta[b0][b1] + PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PB_1*(PQ[a0]*delta[a1][b0] + PQ[b0]*delta[a0][a1]) + PQ[a1]*(PA_0*delta[b0][b1] + PB_0*delta[a0][b1] + PB_1*delta[a0][b0] - PQ[a0]*delta[b0][b1] - PQ[b0]*delta[a0][b1] - PQ[b1]*delta[a0][b0]) + PQ[b1]*(delta[a0][a1]*(PB_0 - PQ[b0]) + delta[a1][b0]*(PA_0 - PQ[a0])))

                        +(PA_0*PQ[c0] - PQ[a0]*PQ[c0])*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + (PA_0*PQ[d0] - PQ[a0]*PQ[d0])*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1]) + (PA_1*PQ[c0] - PQ[a1]*PQ[c0])*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + (PA_1*PQ[d0] - PQ[a1]*PQ[d0])*(delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1]) + (PB_0*PQ[c0] - PQ[b0]*PQ[c0])*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + (PB_0*PQ[d0] - PQ[b0]*PQ[d0])*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]) + (PB_1*PQ[c0] - PQ[b1]*PQ[c0])*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]) + (PB_1*PQ[d0] - PQ[b1]*PQ[d0])*(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0])

                        -2*PQ[c0]*PQ[d0]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.5 * S1 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        -delta[c0][d0]*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))

                        -PA_0*(PA_1*(PQ[b0]*(PQ[c0]*delta[b1][d0] + PQ[d0]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[b0][d0] + PQ[d0]*delta[b0][c0])) + PB_0*(PQ[a1]*(PQ[c0]*delta[b1][d0] + PQ[d0]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[a1][d0] + PQ[d0]*delta[a1][c0])) + PB_1*(PQ[a1]*(PQ[c0]*delta[b0][d0] + PQ[d0]*delta[b0][c0]) + PQ[b0]*(PQ[c0]*delta[a1][d0] + PQ[d0]*delta[a1][c0]))) - PA_1*(PB_0*(PQ[a0]*(PQ[c0]*delta[b1][d0] + PQ[d0]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[a0][d0] + PQ[d0]*delta[a0][c0])) + PB_1*(PQ[a0]*(PQ[c0]*delta[b0][d0] + PQ[d0]*delta[b0][c0]) + PQ[b0]*(PQ[c0]*delta[a0][d0] + PQ[d0]*delta[a0][c0]))) - PB_0*PB_1*(PQ[a0]*(PQ[c0]*delta[a1][d0] + PQ[d0]*delta[a1][c0]) + PQ[a1]*(PQ[c0]*delta[a0][d0] + PQ[d0]*delta[a0][c0]))

                        +PQ[c0]*PQ[d0]*(PA_0*(-PA_1*delta[b0][b1] - PB_0*delta[a1][b1] - PB_1*delta[a1][b0] + PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PA_1*(PQ[a0]*delta[b0][b1] + delta[a0][b0]*(-PB_1 + PQ[b1]) + delta[a0][b1]*(-PB_0 + PQ[b0])) + PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + delta[a0][a1]*(-PB_1 + PQ[b1])) + PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]))
                    )

                    + (-0.5) * S2 * S2 * S2 * inv_S1 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*QC_0*QD_0*delta[b0][b1]

                        +QC_0*QD_0*(PQ[b0]*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + PQ[b1]*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0]))
                    )

                    + 0.5 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        +QD_0*delta[b1][c0]*(PA_1*PQ[a0]*PQ[b0] + PQ[a1]*(PA_0*PQ[b0] + PB_0*PQ[a0]))

                        +PB_1*PQ[a0]*(PQ[a1]*(QC_0*delta[b0][d0] + QD_0*delta[b0][c0]) + PQ[b0]*(QC_0*delta[a1][d0] + QD_0*delta[a1][c0])) + QC_0*(PQ[a0]*(PA_1*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0]) + PB_0*(PQ[a1]*delta[b1][d0] + PQ[b1]*delta[a1][d0])) + PQ[a1]*(PA_0*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0]) + delta[a0][d0]*(PB_0*PQ[b1] + PB_1*PQ[b0])) + PQ[b0]*PQ[b1]*(PA_0*delta[a1][d0] + PA_1*delta[a0][d0])) + QD_0*(PB_1*PQ[a1]*PQ[b0]*delta[a0][c0] + PQ[b1]*(PA_0*(PQ[a1]*delta[b0][c0] + PQ[b0]*delta[a1][c0]) + PA_1*(PQ[a0]*delta[b0][c0] + PQ[b0]*delta[a0][c0]) + PB_0*(PQ[a0]*delta[a1][c0] + PQ[a1]*delta[a0][c0])))

                        +(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PQ[a0]*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0] - PQ[a1]*delta[b0][b1] - PQ[b0]*delta[a1][b1] - PQ[b1]*delta[a1][b0]) + PQ[a1]*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PQ[b0]*(PB_1*delta[a0][a1] + delta[a0][b1]*(PA_1 - PQ[a1])) + PQ[b1]*(delta[a0][a1]*(PB_0 - PQ[b0]) + delta[a0][b0]*(PA_1 - PQ[a1])))

                        +delta[c0][d0]*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S1 * S1 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[c0]*PQ[d0]*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))
                    )

                    + S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        -(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        +QC_0*QD_0*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                )

            );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * pp_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

// DDPP altB DDPP1 FP64
__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDPP_altB1_FP64(double*         mat_J,
                       const double*   p_prim_info,
                       const uint32_t  p_prim_count,
                       const double*   d_prim_info,
                       const uint32_t  d_prim_count,
                       const double*   pp_mat_D,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const double*   dd_pair_data_local,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* pp_first_inds,
                       const uint32_t* pp_second_inds,
                       const double*   pp_pair_data,
                       const uint32_t  pp_prim_pair_count,
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

        if ((ij >= dd_prim_pair_count_local) || (kl >= pp_prim_pair_count))
        {
            break;
        }

        const auto k = pp_first_inds[kl];
        const auto l = pp_second_inds[kl];

        const auto a_k = p_prim_info[k / 3 + p_prim_count * 0];

        const double r_k[3] = {p_prim_info[k / 3 + p_prim_count * 2],
                               p_prim_info[k / 3 + p_prim_count * 3],
                               p_prim_info[k / 3 + p_prim_count * 4]};

        const auto a_l = p_prim_info[l / 3 + p_prim_count * 0];

        const double r_l[3] = {p_prim_info[l / 3 + p_prim_count * 2],
                               p_prim_info[l / 3 + p_prim_count * 3],
                               p_prim_info[l / 3 + p_prim_count * 4]};

        const auto S_kl_00 = pp_pair_data[kl];

        const auto c0 = k % 3;
        const auto d0 = l % 3;

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F6_t[4];

        gpu::computeBoysFunction(F6_t, S1 * S2 * inv_S4 * r2_PQ, 3, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                + F6_t[2] * (

                    0.25 * S2 * S2 * inv_S1 * inv_S1 * inv_S4 * inv_S4 * (
                        +QC_0*QD_0*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * S2 * inv_S1 * inv_S4 * inv_S4 * (
                        +delta[b0][b1]*delta[c0][d0]*(-PA_0*PQ[a1] + PQ[a0]*(-PA_1 + PQ[a1]))

                        +delta[c0][d0]*(PQ[a0]*delta[a1][b0]*(-PB_1 + PQ[b1]) + PQ[a0]*delta[a1][b1]*(-PB_0 + PQ[b0]) + PQ[a1]*delta[a0][b0]*(-PB_1 + PQ[b1]) + PQ[a1]*delta[a0][b1]*(-PB_0 + PQ[b0]) + PQ[b0]*PQ[b1]*delta[a0][a1] - PQ[b0]*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) - PQ[b1]*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0] + PB_0*delta[a0][a1]))

                        +(-PA_0*QC_0 + PQ[a0]*QC_0)*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + (-PA_0*QD_0 + PQ[a0]*QD_0)*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1]) + (-PA_1*QC_0 + PQ[a1]*QC_0)*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + (-PA_1*QD_0 + PQ[a1]*QD_0)*(delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1]) + (-PB_0*QC_0 + PQ[b0]*QC_0)*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + (-PB_0*QD_0 + PQ[b0]*QD_0)*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]) + (-PB_1*QC_0 + PQ[b1]*QC_0)*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]) + (-PB_1*QD_0 + PQ[b1]*QD_0)*(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0])

                        +2.0*(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * inv_S4 * inv_S4 * (
                        +delta[a1][b1]*delta[c0][d0]*(PA_0*(PB_0 - PQ[b0]) - PB_0*PQ[a0])

                        +PA_0*(PA_1*(delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0]) + PB_0*(delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + PB_1*(delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0])) + PA_1*(PB_0*(delta[a0][c0]*delta[b1][d0] + delta[a0][d0]*delta[b1][c0]) + PB_1*(delta[a0][c0]*delta[b0][d0] + delta[a0][d0]*delta[b0][c0])) + PB_0*PB_1*(delta[a0][c0]*delta[a1][d0] + delta[a0][d0]*delta[a1][c0])

                        +delta[c0][d0]*(PA_1*(PB_1*delta[a0][b0] + delta[a0][b1]*(PB_0 - PQ[b0]) + delta[b0][b1]*(PA_0 - PQ[a0])) + PB_1*delta[a0][a1]*(PB_0 - PQ[b0]) + PB_1*delta[a1][b0]*(PA_0 - PQ[a0]) - PQ[a1]*(PA_0*delta[b0][b1] + PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) - PQ[b1]*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0] + PB_0*delta[a0][a1]))

                        -PQ[c0]*(PA_0*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])) - PQ[d0]*(PA_0*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1]) + PA_1*(delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1]) + PB_0*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]) + PB_1*(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0]))

                        +PQ[c0]*PQ[d0]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.5 * S1 * inv_S4 * inv_S4 * (
                        -delta[c0][d0]*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))

                        -PA_0*PA_1*(PB_0*(PQ[c0]*delta[b1][d0] + PQ[d0]*delta[b1][c0]) + PB_1*(PQ[c0]*delta[b0][d0] + PQ[d0]*delta[b0][c0])) - PB_0*PB_1*(PA_0*(PQ[c0]*delta[a1][d0] + PQ[d0]*delta[a1][c0]) + PA_1*(PQ[c0]*delta[a0][d0] + PQ[d0]*delta[a0][c0]))

                        +PQ[c0]*PQ[d0]*(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1])
                    )

                    + 0.5 * S2 * S2 * inv_S1 * inv_S4 * inv_S4 * (
                        +QC_0*QD_0*delta[b0][b1]*(-PA_0*PQ[a1] + PQ[a0]*(-PA_1 + PQ[a1]))

                        +QC_0*QD_0*(PQ[a0]*delta[a1][b0]*(-PB_1 + PQ[b1]) + PQ[a0]*delta[a1][b1]*(-PB_0 + PQ[b0]) + PQ[a1]*delta[a0][b0]*(-PB_1 + PQ[b1]) + PQ[a1]*delta[a0][b1]*(-PB_0 + PQ[b0]) + PQ[b0]*PQ[b1]*delta[a0][a1] - PQ[b0]*(PA_0*delta[a1][b1] + PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) - PQ[b1]*(PA_0*delta[a1][b0] + PA_1*delta[a0][b0] + PB_0*delta[a0][a1]))
                    )

                    + 0.5 * S2 * inv_S4 * inv_S4 * (
                        +QD_0*delta[b1][c0]*(PA_0*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PA_1*PB_0*PQ[a0])

                        +PB_1*(PA_0*(PQ[a1]*(QC_0*delta[b0][d0] + QD_0*delta[b0][c0]) + PQ[b0]*(QC_0*delta[a1][d0] + QD_0*delta[a1][c0])) + PQ[a0]*(PA_1*(QC_0*delta[b0][d0] + QD_0*delta[b0][c0]) + PB_0*(QC_0*delta[a1][d0] + QD_0*delta[a1][c0]))) + QC_0*(PA_0*(PA_1*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0]) + PB_0*(PQ[a1]*delta[b1][d0] + PQ[b1]*delta[a1][d0])) + PA_1*PB_0*(PQ[a0]*delta[b1][d0] + PQ[b1]*delta[a0][d0]) + PB_1*delta[a0][d0]*(PA_1*PQ[b0] + PB_0*PQ[a1])) + QD_0*(PA_0*PQ[b1]*(PA_1*delta[b0][c0] + PB_0*delta[a1][c0]) + delta[a0][c0]*(PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*PQ[a1]))

                        +(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(PA_0*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) - PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PA_1*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) - PA_1*(PQ[a0]*delta[b0][b1] + PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PB_0*PB_1*delta[a0][a1] - PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) - PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]))

                        +delta[c0][d0]*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S1 * S1 * inv_S4 * inv_S4 * (
                        +PA_0*PA_1*PB_0*PB_1*PQ[c0]*PQ[d0]
                    )

                    + S1 * S2 * inv_S4 * inv_S4 * (
                        -(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))
                    )

                    + S2 * S2 * inv_S4 * inv_S4 * (
                        +QC_0*QD_0*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + 0.125 * S2 * inv_S1 * inv_S1 * inv_S4 * inv_S4 * (
                        +delta[c0][d0]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.125 * inv_S1 * inv_S4 * inv_S4 * (
                        +2.0*delta[c0][d0]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])

                        +delta[b0][b1]*(delta[a0][c0]*delta[a1][d0] + delta[a0][d0]*delta[a1][c0]) + delta[b0][c0]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + delta[b0][d0]*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]) + delta[b1][c0]*(delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]) + delta[b1][d0]*(delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0])
                    )

                )

                + F6_t[3] * (

                    (-0.125) * S2 * inv_S1 * inv_S4 * inv_S4 * inv_S4 * (
                        +delta[a0][a1]*(delta[b0][b1]*delta[c0][d0] + delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0]) + delta[a0][b0]*(delta[a1][b1]*delta[c0][d0] + delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + delta[a0][b1]*(delta[a1][b0]*delta[c0][d0] + delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0]) + delta[a0][c0]*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + delta[a0][d0]*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1])
                    )

                    + (-0.25) * S2 * S2 * inv_S1 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*delta[b0][b1]*delta[c0][d0]

                        +delta[c0][d0]*(PQ[b0]*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + PQ[b1]*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0]))

                        +QC_0*(PQ[a0]*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + PQ[b0]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + PQ[b1]*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0])) + QD_0*(PQ[a0]*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1]) + PQ[b0]*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]) + PQ[b1]*(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0]))

                        +(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.25 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        +delta[a1][b1]*delta[c0][d0]*(PB_0*PQ[a0] + PQ[b0]*(PA_0 - PQ[a0]))

                        +(PA_0*PQ[a1] + PA_1*PQ[a0])*(delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0]) + (PA_0*PQ[b0] + PB_0*PQ[a0])*(delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + (PA_0*PQ[b1] + PB_1*PQ[a0])*(delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0]) + (PA_1*PQ[b0] + PB_0*PQ[a1])*(delta[a0][c0]*delta[b1][d0] + delta[a0][d0]*delta[b1][c0]) + (PA_1*PQ[b1] + PB_1*PQ[a1])*(delta[a0][c0]*delta[b0][d0] + delta[a0][d0]*delta[b0][c0]) + (PB_0*PQ[b1] + PB_1*PQ[b0])*(delta[a0][c0]*delta[a1][d0] + delta[a0][d0]*delta[a1][c0])

                        +delta[c0][d0]*(PA_1*(PQ[a0]*delta[b0][b1] + PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PB_1*(PQ[a0]*delta[a1][b0] + PQ[b0]*delta[a0][a1]) + PQ[a1]*(PA_0*delta[b0][b1] + PB_0*delta[a0][b1] + PB_1*delta[a0][b0] - PQ[a0]*delta[b0][b1] - PQ[b0]*delta[a0][b1] - PQ[b1]*delta[a0][b0]) + PQ[b1]*(delta[a0][a1]*(PB_0 - PQ[b0]) + delta[a1][b0]*(PA_0 - PQ[a0])))

                        +(PA_0*PQ[c0] - PQ[a0]*PQ[c0])*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + (PA_0*PQ[d0] - PQ[a0]*PQ[d0])*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1]) + (PA_1*PQ[c0] - PQ[a1]*PQ[c0])*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + (PA_1*PQ[d0] - PQ[a1]*PQ[d0])*(delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1]) + (PB_0*PQ[c0] - PQ[b0]*PQ[c0])*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + (PB_0*PQ[d0] - PQ[b0]*PQ[d0])*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1]) + (PB_1*PQ[c0] - PQ[b1]*PQ[c0])*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]) + (PB_1*PQ[d0] - PQ[b1]*PQ[d0])*(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0])

                        -2*PQ[c0]*PQ[d0]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])
                    )

                    + 0.5 * S1 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        -delta[c0][d0]*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))

                        -PA_0*(PA_1*(PQ[b0]*(PQ[c0]*delta[b1][d0] + PQ[d0]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[b0][d0] + PQ[d0]*delta[b0][c0])) + PB_0*(PQ[a1]*(PQ[c0]*delta[b1][d0] + PQ[d0]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[a1][d0] + PQ[d0]*delta[a1][c0])) + PB_1*(PQ[a1]*(PQ[c0]*delta[b0][d0] + PQ[d0]*delta[b0][c0]) + PQ[b0]*(PQ[c0]*delta[a1][d0] + PQ[d0]*delta[a1][c0]))) - PA_1*(PB_0*(PQ[a0]*(PQ[c0]*delta[b1][d0] + PQ[d0]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[a0][d0] + PQ[d0]*delta[a0][c0])) + PB_1*(PQ[a0]*(PQ[c0]*delta[b0][d0] + PQ[d0]*delta[b0][c0]) + PQ[b0]*(PQ[c0]*delta[a0][d0] + PQ[d0]*delta[a0][c0]))) - PB_0*PB_1*(PQ[a0]*(PQ[c0]*delta[a1][d0] + PQ[d0]*delta[a1][c0]) + PQ[a1]*(PQ[c0]*delta[a0][d0] + PQ[d0]*delta[a0][c0]))

                        +PQ[c0]*PQ[d0]*(PA_0*(-PA_1*delta[b0][b1] - PB_0*delta[a1][b1] - PB_1*delta[a1][b0] + PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PA_1*(PQ[a0]*delta[b0][b1] + delta[a0][b0]*(-PB_1 + PQ[b1]) + delta[a0][b1]*(-PB_0 + PQ[b0])) + PB_0*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + delta[a0][a1]*(-PB_1 + PQ[b1])) + PB_1*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0] + PQ[b0]*delta[a0][a1]))
                    )

                    + (-0.5) * S2 * S2 * S2 * inv_S1 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*QC_0*QD_0*delta[b0][b1]

                        +QC_0*QD_0*(PQ[b0]*(PQ[a0]*delta[a1][b1] + PQ[a1]*delta[a0][b1] + PQ[b1]*delta[a0][a1]) + PQ[b1]*(PQ[a0]*delta[a1][b0] + PQ[a1]*delta[a0][b0]))
                    )

                    + 0.5 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        +QD_0*delta[b1][c0]*(PA_1*PQ[a0]*PQ[b0] + PQ[a1]*(PA_0*PQ[b0] + PB_0*PQ[a0]))

                        +PB_1*PQ[a0]*(PQ[a1]*(QC_0*delta[b0][d0] + QD_0*delta[b0][c0]) + PQ[b0]*(QC_0*delta[a1][d0] + QD_0*delta[a1][c0])) + QC_0*(PQ[a0]*(PA_1*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0]) + PB_0*(PQ[a1]*delta[b1][d0] + PQ[b1]*delta[a1][d0])) + PQ[a1]*(PA_0*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0]) + delta[a0][d0]*(PB_0*PQ[b1] + PB_1*PQ[b0])) + PQ[b0]*PQ[b1]*(PA_0*delta[a1][d0] + PA_1*delta[a0][d0])) + QD_0*(PB_1*PQ[a1]*PQ[b0]*delta[a0][c0] + PQ[b1]*(PA_0*(PQ[a1]*delta[b0][c0] + PQ[b0]*delta[a1][c0]) + PA_1*(PQ[a0]*delta[b0][c0] + PQ[b0]*delta[a0][c0]) + PB_0*(PQ[a0]*delta[a1][c0] + PQ[a1]*delta[a0][c0])))

                        +(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PQ[a0]*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0] - PQ[a1]*delta[b0][b1] - PQ[b0]*delta[a1][b1] - PQ[b1]*delta[a1][b0]) + PQ[a1]*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PQ[b0]*(PB_1*delta[a0][a1] + delta[a0][b1]*(PA_1 - PQ[a1])) + PQ[b1]*(delta[a0][a1]*(PB_0 - PQ[b0]) + delta[a0][b0]*(PA_1 - PQ[a1])))

                        +delta[c0][d0]*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S1 * S1 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[c0]*PQ[d0]*(PA_0*PA_1*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PB_1*(PA_0*PQ[a1] + PA_1*PQ[a0]))
                    )

                    + S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        -(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * (
                        +QC_0*QD_0*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                )

            );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * pp_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

// DDPP altB DDPP1 FP32
__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDPP_altB1_FP32(double*         mat_J,
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

        const float r_k_f[3] = {p_prim_info_f[k / 3 + p_prim_count * 2],
                               p_prim_info_f[k / 3 + p_prim_count * 3],
                               p_prim_info_f[k / 3 + p_prim_count * 4]};

        const auto a_l_f = p_prim_info_f[l / 3 + p_prim_count * 0];

        const float r_l_f[3] = {p_prim_info_f[l / 3 + p_prim_count * 2],
                               p_prim_info_f[l / 3 + p_prim_count * 3],
                               p_prim_info_f[l / 3 + p_prim_count * 4]};

        const auto S_kl_00_f = pp_pair_data_f[kl];

        const auto c0 = k % 3;
        const auto d0 = l % 3;

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2_f = a_k_f + a_l_f;

        const auto inv_S2_f = 1.0f / S2_f;
        const auto inv_S4_f = 1.0f / (S1_f + S2_f);

        const float PQ_f[3] = {(a_k_f * r_k_f[0] + a_l_f * r_l_f[0]) * inv_S2_f - (a_i_f * r_i_f[0] + a_j_f * r_j_f[0]) * inv_S1_f,
                              (a_k_f * r_k_f[1] + a_l_f * r_l_f[1]) * inv_S2_f - (a_i_f * r_i_f[1] + a_j_f * r_j_f[1]) * inv_S1_f,
                              (a_k_f * r_k_f[2] + a_l_f * r_l_f[2]) * inv_S2_f - (a_i_f * r_i_f[2] + a_j_f * r_j_f[2]) * inv_S1_f};

        const auto r2_PQ_f = PQ_f[0] * PQ_f[0] + PQ_f[1] * PQ_f[1] + PQ_f[2] * PQ_f[2];

        const auto Lambda_f = sqrtf(4.0f * S1_f * S2_f * MATH_CONST_INV_PI_F * inv_S4_f);

        float F6_t_f[4];

        gpu::computeBoysFunction_f(F6_t_f, S1_f * S2_f * inv_S4_f * r2_PQ_f, 3, boys_func_table_f, boys_func_ft_f);

        const auto QC_0_f = (a_l_f * inv_S2_f) * (r_l_f[c0] - r_k_f[c0]);
        const auto QD_0_f = (-a_k_f * inv_S2_f) * (r_l_f[d0] - r_k_f[d0]);

        const float eri_ijkl_f = Lambda_f * S_ij_00_f * S_kl_00_f * (

                + F6_t_f[2] * (

                    0.25f * S2_f * S2_f * inv_S1_f * inv_S1_f * inv_S4_f * inv_S4_f * (
                        +QC_0_f*QD_0_f*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.25f * S2_f * inv_S1_f * inv_S4_f * inv_S4_f * (
                        +delta_f[b0][b1]*delta_f[c0][d0]*(-PA_0_f*PQ_f[a1] + PQ_f[a0]*(-PA_1_f + PQ_f[a1]))

                        +delta_f[c0][d0]*(PQ_f[a0]*delta_f[a1][b0]*(-PB_1_f + PQ_f[b1]) + PQ_f[a0]*delta_f[a1][b1]*(-PB_0_f + PQ_f[b0]) + PQ_f[a1]*delta_f[a0][b0]*(-PB_1_f + PQ_f[b1]) + PQ_f[a1]*delta_f[a0][b1]*(-PB_0_f + PQ_f[b0]) + PQ_f[b0]*PQ_f[b1]*delta_f[a0][a1] - PQ_f[b0]*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) - PQ_f[b1]*(PA_0_f*delta_f[a1][b0] + PA_1_f*delta_f[a0][b0] + PB_0_f*delta_f[a0][a1]))

                        +(-PA_0_f*QC_0_f + PQ_f[a0]*QC_0_f)*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + (-PA_0_f*QD_0_f + PQ_f[a0]*QD_0_f)*(delta_f[a1][b0]*delta_f[b1][c0] + delta_f[a1][b1]*delta_f[b0][c0] + delta_f[a1][c0]*delta_f[b0][b1]) + (-PA_1_f*QC_0_f + PQ_f[a1]*QC_0_f)*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + (-PA_1_f*QD_0_f + PQ_f[a1]*QD_0_f)*(delta_f[a0][b0]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[b0][c0] + delta_f[a0][c0]*delta_f[b0][b1]) + (-PB_0_f*QC_0_f + PQ_f[b0]*QC_0_f)*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + (-PB_0_f*QD_0_f + PQ_f[b0]*QD_0_f)*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1]) + (-PB_1_f*QC_0_f + PQ_f[b1]*QC_0_f)*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0]) + (-PB_1_f*QD_0_f + PQ_f[b1]*QD_0_f)*(delta_f[a0][a1]*delta_f[b0][c0] + delta_f[a0][b0]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b0])

                        +2.0f*(PQ_f[c0]*QD_0_f + PQ_f[d0]*QC_0_f)*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.25f * inv_S4_f * inv_S4_f * (
                        +delta_f[a1][b1]*delta_f[c0][d0]*(PA_0_f*(PB_0_f - PQ_f[b0]) - PB_0_f*PQ_f[a0])

                        +PA_0_f*(PA_1_f*(delta_f[b0][c0]*delta_f[b1][d0] + delta_f[b0][d0]*delta_f[b1][c0]) + PB_0_f*(delta_f[a1][c0]*delta_f[b1][d0] + delta_f[a1][d0]*delta_f[b1][c0]) + PB_1_f*(delta_f[a1][c0]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][c0])) + PA_1_f*(PB_0_f*(delta_f[a0][c0]*delta_f[b1][d0] + delta_f[a0][d0]*delta_f[b1][c0]) + PB_1_f*(delta_f[a0][c0]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][c0])) + PB_0_f*PB_1_f*(delta_f[a0][c0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][c0])

                        +delta_f[c0][d0]*(PA_1_f*(PB_1_f*delta_f[a0][b0] + delta_f[a0][b1]*(PB_0_f - PQ_f[b0]) + delta_f[b0][b1]*(PA_0_f - PQ_f[a0])) + PB_1_f*delta_f[a0][a1]*(PB_0_f - PQ_f[b0]) + PB_1_f*delta_f[a1][b0]*(PA_0_f - PQ_f[a0]) - PQ_f[a1]*(PA_0_f*delta_f[b0][b1] + PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) - PQ_f[b1]*(PA_0_f*delta_f[a1][b0] + PA_1_f*delta_f[a0][b0] + PB_0_f*delta_f[a0][a1]))

                        -PQ_f[c0]*(PA_0_f*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + PA_1_f*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + PB_0_f*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + PB_1_f*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0])) - PQ_f[d0]*(PA_0_f*(delta_f[a1][b0]*delta_f[b1][c0] + delta_f[a1][b1]*delta_f[b0][c0] + delta_f[a1][c0]*delta_f[b0][b1]) + PA_1_f*(delta_f[a0][b0]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[b0][c0] + delta_f[a0][c0]*delta_f[b0][b1]) + PB_0_f*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1]) + PB_1_f*(delta_f[a0][a1]*delta_f[b0][c0] + delta_f[a0][b0]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b0]))

                        +PQ_f[c0]*PQ_f[d0]*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.5f * S1_f * inv_S4_f * inv_S4_f * (
                        -delta_f[c0][d0]*(PA_0_f*PA_1_f*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PB_1_f*(PA_0_f*PQ_f[a1] + PA_1_f*PQ_f[a0]))

                        -PA_0_f*PA_1_f*(PB_0_f*(PQ_f[c0]*delta_f[b1][d0] + PQ_f[d0]*delta_f[b1][c0]) + PB_1_f*(PQ_f[c0]*delta_f[b0][d0] + PQ_f[d0]*delta_f[b0][c0])) - PB_0_f*PB_1_f*(PA_0_f*(PQ_f[c0]*delta_f[a1][d0] + PQ_f[d0]*delta_f[a1][c0]) + PA_1_f*(PQ_f[c0]*delta_f[a0][d0] + PQ_f[d0]*delta_f[a0][c0]))

                        +PQ_f[c0]*PQ_f[d0]*(PA_0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) + PA_1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PB_0_f*PB_1_f*delta_f[a0][a1])
                    )

                    + 0.5f * S2_f * S2_f * inv_S1_f * inv_S4_f * inv_S4_f * (
                        +QC_0_f*QD_0_f*delta_f[b0][b1]*(-PA_0_f*PQ_f[a1] + PQ_f[a0]*(-PA_1_f + PQ_f[a1]))

                        +QC_0_f*QD_0_f*(PQ_f[a0]*delta_f[a1][b0]*(-PB_1_f + PQ_f[b1]) + PQ_f[a0]*delta_f[a1][b1]*(-PB_0_f + PQ_f[b0]) + PQ_f[a1]*delta_f[a0][b0]*(-PB_1_f + PQ_f[b1]) + PQ_f[a1]*delta_f[a0][b1]*(-PB_0_f + PQ_f[b0]) + PQ_f[b0]*PQ_f[b1]*delta_f[a0][a1] - PQ_f[b0]*(PA_0_f*delta_f[a1][b1] + PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) - PQ_f[b1]*(PA_0_f*delta_f[a1][b0] + PA_1_f*delta_f[a0][b0] + PB_0_f*delta_f[a0][a1]))
                    )

                    + 0.5f * S2_f * inv_S4_f * inv_S4_f * (
                        +QD_0_f*delta_f[b1][c0]*(PA_0_f*(PA_1_f*PQ_f[b0] + PB_0_f*PQ_f[a1]) + PA_1_f*PB_0_f*PQ_f[a0])

                        +PB_1_f*(PA_0_f*(PQ_f[a1]*(QC_0_f*delta_f[b0][d0] + QD_0_f*delta_f[b0][c0]) + PQ_f[b0]*(QC_0_f*delta_f[a1][d0] + QD_0_f*delta_f[a1][c0])) + PQ_f[a0]*(PA_1_f*(QC_0_f*delta_f[b0][d0] + QD_0_f*delta_f[b0][c0]) + PB_0_f*(QC_0_f*delta_f[a1][d0] + QD_0_f*delta_f[a1][c0]))) + QC_0_f*(PA_0_f*(PA_1_f*(PQ_f[b0]*delta_f[b1][d0] + PQ_f[b1]*delta_f[b0][d0]) + PB_0_f*(PQ_f[a1]*delta_f[b1][d0] + PQ_f[b1]*delta_f[a1][d0])) + PA_1_f*PB_0_f*(PQ_f[a0]*delta_f[b1][d0] + PQ_f[b1]*delta_f[a0][d0]) + PB_1_f*delta_f[a0][d0]*(PA_1_f*PQ_f[b0] + PB_0_f*PQ_f[a1])) + QD_0_f*(PA_0_f*PQ_f[b1]*(PA_1_f*delta_f[b0][c0] + PB_0_f*delta_f[a1][c0]) + delta_f[a0][c0]*(PA_1_f*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PB_1_f*PQ_f[a1]))

                        +(PQ_f[c0]*QD_0_f + PQ_f[d0]*QC_0_f)*(PA_0_f*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) - PA_0_f*(PQ_f[a1]*delta_f[b0][b1] + PQ_f[b0]*delta_f[a1][b1] + PQ_f[b1]*delta_f[a1][b0]) + PA_1_f*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) - PA_1_f*(PQ_f[a0]*delta_f[b0][b1] + PQ_f[b0]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][b0]) + PB_0_f*PB_1_f*delta_f[a0][a1] - PB_0_f*(PQ_f[a0]*delta_f[a1][b1] + PQ_f[a1]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][a1]) - PB_1_f*(PQ_f[a0]*delta_f[a1][b0] + PQ_f[a1]*delta_f[a0][b0] + PQ_f[b0]*delta_f[a0][a1]))

                        +delta_f[c0][d0]*(PA_0_f*PQ_f[b0]*(PA_1_f*PQ_f[b1] + PB_1_f*PQ_f[a1]) + PA_1_f*PQ_f[a0]*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PQ_f[a1]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))
                    )

                    + S1_f * S1_f * inv_S4_f * inv_S4_f * (
                        +PA_0_f*PA_1_f*PB_0_f*PB_1_f*PQ_f[c0]*PQ_f[d0]
                    )

                    + S1_f * S2_f * inv_S4_f * inv_S4_f * (
                        -(PQ_f[c0]*QD_0_f + PQ_f[d0]*QC_0_f)*(PA_0_f*PA_1_f*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PB_1_f*(PA_0_f*PQ_f[a1] + PA_1_f*PQ_f[a0]))
                    )

                    + S2_f * S2_f * inv_S4_f * inv_S4_f * (
                        +QC_0_f*QD_0_f*(PA_0_f*PQ_f[b0]*(PA_1_f*PQ_f[b1] + PB_1_f*PQ_f[a1]) + PA_1_f*PQ_f[a0]*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PQ_f[a1]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))
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
                        +PQ_f[a0]*PQ_f[a1]*delta_f[b0][b1]*delta_f[c0][d0]

                        +delta_f[c0][d0]*(PQ_f[b0]*(PQ_f[a0]*delta_f[a1][b1] + PQ_f[a1]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][a1]) + PQ_f[b1]*(PQ_f[a0]*delta_f[a1][b0] + PQ_f[a1]*delta_f[a0][b0]))

                        +QC_0_f*(PQ_f[a0]*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + PQ_f[a1]*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + PQ_f[b0]*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + PQ_f[b1]*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0])) + QD_0_f*(PQ_f[a0]*(delta_f[a1][b0]*delta_f[b1][c0] + delta_f[a1][b1]*delta_f[b0][c0] + delta_f[a1][c0]*delta_f[b0][b1]) + PQ_f[a1]*(delta_f[a0][b0]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[b0][c0] + delta_f[a0][c0]*delta_f[b0][b1]) + PQ_f[b0]*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1]) + PQ_f[b1]*(delta_f[a0][a1]*delta_f[b0][c0] + delta_f[a0][b0]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b0]))

                        +(PQ_f[c0]*QD_0_f + PQ_f[d0]*QC_0_f)*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.25f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +delta_f[a1][b1]*delta_f[c0][d0]*(PB_0_f*PQ_f[a0] + PQ_f[b0]*(PA_0_f - PQ_f[a0]))

                        +(PA_0_f*PQ_f[a1] + PA_1_f*PQ_f[a0])*(delta_f[b0][c0]*delta_f[b1][d0] + delta_f[b0][d0]*delta_f[b1][c0]) + (PA_0_f*PQ_f[b0] + PB_0_f*PQ_f[a0])*(delta_f[a1][c0]*delta_f[b1][d0] + delta_f[a1][d0]*delta_f[b1][c0]) + (PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0])*(delta_f[a1][c0]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][c0]) + (PA_1_f*PQ_f[b0] + PB_0_f*PQ_f[a1])*(delta_f[a0][c0]*delta_f[b1][d0] + delta_f[a0][d0]*delta_f[b1][c0]) + (PA_1_f*PQ_f[b1] + PB_1_f*PQ_f[a1])*(delta_f[a0][c0]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][c0]) + (PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0])*(delta_f[a0][c0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][c0])

                        +delta_f[c0][d0]*(PA_1_f*(PQ_f[a0]*delta_f[b0][b1] + PQ_f[b0]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][b0]) + PB_1_f*(PQ_f[a0]*delta_f[a1][b0] + PQ_f[b0]*delta_f[a0][a1]) + PQ_f[a1]*(PA_0_f*delta_f[b0][b1] + PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0] - PQ_f[a0]*delta_f[b0][b1] - PQ_f[b0]*delta_f[a0][b1] - PQ_f[b1]*delta_f[a0][b0]) + PQ_f[b1]*(delta_f[a0][a1]*(PB_0_f - PQ_f[b0]) + delta_f[a1][b0]*(PA_0_f - PQ_f[a0])))

                        +(PA_0_f*PQ_f[c0] - PQ_f[a0]*PQ_f[c0])*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + (PA_0_f*PQ_f[d0] - PQ_f[a0]*PQ_f[d0])*(delta_f[a1][b0]*delta_f[b1][c0] + delta_f[a1][b1]*delta_f[b0][c0] + delta_f[a1][c0]*delta_f[b0][b1]) + (PA_1_f*PQ_f[c0] - PQ_f[a1]*PQ_f[c0])*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + (PA_1_f*PQ_f[d0] - PQ_f[a1]*PQ_f[d0])*(delta_f[a0][b0]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[b0][c0] + delta_f[a0][c0]*delta_f[b0][b1]) + (PB_0_f*PQ_f[c0] - PQ_f[b0]*PQ_f[c0])*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + (PB_0_f*PQ_f[d0] - PQ_f[b0]*PQ_f[d0])*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1]) + (PB_1_f*PQ_f[c0] - PQ_f[b1]*PQ_f[c0])*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0]) + (PB_1_f*PQ_f[d0] - PQ_f[b1]*PQ_f[d0])*(delta_f[a0][a1]*delta_f[b0][c0] + delta_f[a0][b0]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b0])

                        -2*PQ_f[c0]*PQ_f[d0]*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])
                    )

                    + 0.5f * S1_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        -delta_f[c0][d0]*(PA_0_f*PQ_f[b0]*(PA_1_f*PQ_f[b1] + PB_1_f*PQ_f[a1]) + PA_1_f*PQ_f[a0]*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PQ_f[a1]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))

                        -PA_0_f*(PA_1_f*(PQ_f[b0]*(PQ_f[c0]*delta_f[b1][d0] + PQ_f[d0]*delta_f[b1][c0]) + PQ_f[b1]*(PQ_f[c0]*delta_f[b0][d0] + PQ_f[d0]*delta_f[b0][c0])) + PB_0_f*(PQ_f[a1]*(PQ_f[c0]*delta_f[b1][d0] + PQ_f[d0]*delta_f[b1][c0]) + PQ_f[b1]*(PQ_f[c0]*delta_f[a1][d0] + PQ_f[d0]*delta_f[a1][c0])) + PB_1_f*(PQ_f[a1]*(PQ_f[c0]*delta_f[b0][d0] + PQ_f[d0]*delta_f[b0][c0]) + PQ_f[b0]*(PQ_f[c0]*delta_f[a1][d0] + PQ_f[d0]*delta_f[a1][c0]))) - PA_1_f*(PB_0_f*(PQ_f[a0]*(PQ_f[c0]*delta_f[b1][d0] + PQ_f[d0]*delta_f[b1][c0]) + PQ_f[b1]*(PQ_f[c0]*delta_f[a0][d0] + PQ_f[d0]*delta_f[a0][c0])) + PB_1_f*(PQ_f[a0]*(PQ_f[c0]*delta_f[b0][d0] + PQ_f[d0]*delta_f[b0][c0]) + PQ_f[b0]*(PQ_f[c0]*delta_f[a0][d0] + PQ_f[d0]*delta_f[a0][c0]))) - PB_0_f*PB_1_f*(PQ_f[a0]*(PQ_f[c0]*delta_f[a1][d0] + PQ_f[d0]*delta_f[a1][c0]) + PQ_f[a1]*(PQ_f[c0]*delta_f[a0][d0] + PQ_f[d0]*delta_f[a0][c0]))

                        +PQ_f[c0]*PQ_f[d0]*(PA_0_f*(-PA_1_f*delta_f[b0][b1] - PB_0_f*delta_f[a1][b1] - PB_1_f*delta_f[a1][b0] + PQ_f[a1]*delta_f[b0][b1] + PQ_f[b0]*delta_f[a1][b1] + PQ_f[b1]*delta_f[a1][b0]) + PA_1_f*(PQ_f[a0]*delta_f[b0][b1] + delta_f[a0][b0]*(-PB_1_f + PQ_f[b1]) + delta_f[a0][b1]*(-PB_0_f + PQ_f[b0])) + PB_0_f*(PQ_f[a0]*delta_f[a1][b1] + PQ_f[a1]*delta_f[a0][b1] + delta_f[a0][a1]*(-PB_1_f + PQ_f[b1])) + PB_1_f*(PQ_f[a0]*delta_f[a1][b0] + PQ_f[a1]*delta_f[a0][b0] + PQ_f[b0]*delta_f[a0][a1]))
                    )

                    + (-0.5f) * S2_f * S2_f * S2_f * inv_S1_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[a0]*PQ_f[a1]*QC_0_f*QD_0_f*delta_f[b0][b1]

                        +QC_0_f*QD_0_f*(PQ_f[b0]*(PQ_f[a0]*delta_f[a1][b1] + PQ_f[a1]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][a1]) + PQ_f[b1]*(PQ_f[a0]*delta_f[a1][b0] + PQ_f[a1]*delta_f[a0][b0]))
                    )

                    + 0.5f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +QD_0_f*delta_f[b1][c0]*(PA_1_f*PQ_f[a0]*PQ_f[b0] + PQ_f[a1]*(PA_0_f*PQ_f[b0] + PB_0_f*PQ_f[a0]))

                        +PB_1_f*PQ_f[a0]*(PQ_f[a1]*(QC_0_f*delta_f[b0][d0] + QD_0_f*delta_f[b0][c0]) + PQ_f[b0]*(QC_0_f*delta_f[a1][d0] + QD_0_f*delta_f[a1][c0])) + QC_0_f*(PQ_f[a0]*(PA_1_f*(PQ_f[b0]*delta_f[b1][d0] + PQ_f[b1]*delta_f[b0][d0]) + PB_0_f*(PQ_f[a1]*delta_f[b1][d0] + PQ_f[b1]*delta_f[a1][d0])) + PQ_f[a1]*(PA_0_f*(PQ_f[b0]*delta_f[b1][d0] + PQ_f[b1]*delta_f[b0][d0]) + delta_f[a0][d0]*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0])) + PQ_f[b0]*PQ_f[b1]*(PA_0_f*delta_f[a1][d0] + PA_1_f*delta_f[a0][d0])) + QD_0_f*(PB_1_f*PQ_f[a1]*PQ_f[b0]*delta_f[a0][c0] + PQ_f[b1]*(PA_0_f*(PQ_f[a1]*delta_f[b0][c0] + PQ_f[b0]*delta_f[a1][c0]) + PA_1_f*(PQ_f[a0]*delta_f[b0][c0] + PQ_f[b0]*delta_f[a0][c0]) + PB_0_f*(PQ_f[a0]*delta_f[a1][c0] + PQ_f[a1]*delta_f[a0][c0])))

                        +(PQ_f[c0]*QD_0_f + PQ_f[d0]*QC_0_f)*(PA_0_f*(PQ_f[a1]*delta_f[b0][b1] + PQ_f[b0]*delta_f[a1][b1] + PQ_f[b1]*delta_f[a1][b0]) + PQ_f[a0]*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0] - PQ_f[a1]*delta_f[b0][b1] - PQ_f[b0]*delta_f[a1][b1] - PQ_f[b1]*delta_f[a1][b0]) + PQ_f[a1]*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PQ_f[b0]*(PB_1_f*delta_f[a0][a1] + delta_f[a0][b1]*(PA_1_f - PQ_f[a1])) + PQ_f[b1]*(delta_f[a0][a1]*(PB_0_f - PQ_f[b0]) + delta_f[a0][b0]*(PA_1_f - PQ_f[a1])))

                        +delta_f[c0][d0]*(PQ_f[a0]*PQ_f[b1]*(PA_1_f*PQ_f[b0] + PB_0_f*PQ_f[a1]) + PQ_f[a1]*PQ_f[b0]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))
                    )

                    + S1_f * S1_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[c0]*PQ_f[d0]*(PA_0_f*PA_1_f*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PB_1_f*(PA_0_f*PQ_f[a1] + PA_1_f*PQ_f[a0]))
                    )

                    + S1_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        -(PQ_f[c0]*QD_0_f + PQ_f[d0]*QC_0_f)*(PA_0_f*PQ_f[b0]*(PA_1_f*PQ_f[b1] + PB_1_f*PQ_f[a1]) + PA_1_f*PQ_f[a0]*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PQ_f[a1]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))
                    )

                    + S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +QC_0_f*QD_0_f*(PQ_f[a0]*PQ_f[b1]*(PA_1_f*PQ_f[b0] + PB_0_f*PQ_f[a1]) + PQ_f[a1]*PQ_f[b0]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))
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
        double J_ij = 0.0;

        for (uint32_t n = 0; n < TILE_DIM_LARGE; n++)
        {
            J_ij += ERIs[n];
        }

        mat_J[ij] += J_ij;
    }
}

// DDPP altB DDPP2 BASE
__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDPP_altB2(double*         mat_J,
                       const double*   p_prim_info,
                       const uint32_t  p_prim_count,
                       const double*   d_prim_info,
                       const uint32_t  d_prim_count,
                       const double*   pp_mat_D,
                       const double*   dd_mat_Q_local,
                       const double*   pp_mat_Q,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const double*   dd_pair_data_local,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* pp_first_inds,
                       const uint32_t* pp_second_inds,
                       const double*   pp_pair_data,
                       const uint32_t  pp_prim_pair_count,
                       const double*   boys_func_table,
                       const double*   boys_func_ft,
                       const double    eri_threshold)
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

    for (uint32_t m = 0; m < (pp_prim_pair_count + TILE_DIM_LARGE - 1) / TILE_DIM_LARGE; m++)
    {
        const uint32_t kl = m * TILE_DIM_LARGE + threadIdx.y;

        if ((ij >= dd_prim_pair_count_local) || (kl >= pp_prim_pair_count) || (fabs(dd_mat_Q_local[ij] * pp_mat_Q[kl] * pp_mat_D[kl]) <= eri_threshold))
        {
            break;
        }

        const auto k = pp_first_inds[kl];
        const auto l = pp_second_inds[kl];

        const auto a_k = p_prim_info[k / 3 + p_prim_count * 0];

        const double r_k[3] = {p_prim_info[k / 3 + p_prim_count * 2],
                               p_prim_info[k / 3 + p_prim_count * 3],
                               p_prim_info[k / 3 + p_prim_count * 4]};

        const auto a_l = p_prim_info[l / 3 + p_prim_count * 0];

        const double r_l[3] = {p_prim_info[l / 3 + p_prim_count * 2],
                               p_prim_info[l / 3 + p_prim_count * 3],
                               p_prim_info[l / 3 + p_prim_count * 4]};

        const auto S_kl_00 = pp_pair_data[kl];

        const auto c0 = k % 3;
        const auto d0 = l % 3;

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F6_t[7];

        gpu::computeBoysFunction(F6_t, S1 * S2 * inv_S4 * r2_PQ, 6, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                + F6_t[4] * (

                    0.5 * S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        -delta[c0][d0]*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))

                        -PQ[a0]*(PA_1*(PQ[b0]*(PQ[c0]*delta[b1][d0] + PQ[d0]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[b0][d0] + PQ[d0]*delta[b0][c0])) + PB_0*(PQ[a1]*(PQ[c0]*delta[b1][d0] + PQ[d0]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[a1][d0] + PQ[d0]*delta[a1][c0])) + PB_1*(PQ[a1]*(PQ[c0]*delta[b0][d0] + PQ[d0]*delta[b0][c0]) + PQ[b0]*(PQ[c0]*delta[a1][d0] + PQ[d0]*delta[a1][c0]))) - PQ[a1]*(PA_0*(PQ[b0]*(PQ[c0]*delta[b1][d0] + PQ[d0]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[b0][d0] + PQ[d0]*delta[b0][c0])) + (PB_0*PQ[b1] + PB_1*PQ[b0])*(PQ[c0]*delta[a0][d0] + PQ[d0]*delta[a0][c0])) - PQ[b0]*PQ[b1]*(PA_0*(PQ[c0]*delta[a1][d0] + PQ[d0]*delta[a1][c0]) + PA_1*(PQ[c0]*delta[a0][d0] + PQ[d0]*delta[a0][c0]))

                        +PQ[c0]*PQ[d0]*(-PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) - PQ[a0]*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PQ[a0]*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) - PQ[a1]*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PQ[a1]*(PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PQ[b0]*PQ[b1]*delta[a0][a1] - PQ[b0]*(PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) - PQ[b1]*(PA_1*delta[a0][b0] + PB_0*delta[a0][a1]))
                    )

                    + 0.5 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*QD_0*delta[b1][c0]

                        +PQ[a0]*PQ[a1]*QC_0*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0]) + PQ[b1]*(PQ[a0]*QD_0*(PQ[a1]*delta[b0][c0] + PQ[b0]*delta[a1][c0]) + PQ[b0]*(PQ[a1]*QD_0*delta[a0][c0] + QC_0*(PQ[a0]*delta[a1][d0] + PQ[a1]*delta[a0][d0])))

                        +(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(PQ[a0]*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PQ[a1]*(PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PQ[b0]*PQ[b1]*delta[a0][a1])

                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*delta[c0][d0]
                    )

                    + S1 * S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[c0]*PQ[d0]*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S1 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        -(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S2 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*QC_0*QD_0
                    )

                    + 0.25 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*(delta[b0][b1]*delta[c0][d0] + delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0])

                        +PQ[b0]*(PQ[a0]*(delta[a1][b1]*delta[c0][d0] + delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + PQ[a1]*(delta[a0][b1]*delta[c0][d0] + delta[a0][c0]*delta[b1][d0] + delta[a0][d0]*delta[b1][c0]) + PQ[b1]*(delta[a0][a1]*delta[c0][d0] + delta[a0][c0]*delta[a1][d0] + delta[a0][d0]*delta[a1][c0]) + PQ[c0]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + PQ[d0]*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1])) + PQ[b1]*(PQ[a0]*(delta[a1][b0]*delta[c0][d0] + delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0]) + PQ[a1]*(delta[a0][b0]*delta[c0][d0] + delta[a0][c0]*delta[b0][d0] + delta[a0][d0]*delta[b0][c0]) + PQ[c0]*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]) + PQ[d0]*(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0])) + PQ[c0]*(PQ[a0]*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + PQ[d0]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])) + PQ[d0]*(PQ[a0]*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1]))
                    )

                )

                + F6_t[5] * (

                    (-0.5) * S1 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*delta[c0][d0]

                        +PQ[c0]*(PQ[a0]*(PQ[a1]*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0] + PQ[d0]*delta[b0][b1]) + PQ[b1]*PQ[d0]*delta[a1][b0]) + PQ[b0]*PQ[b1]*(PQ[a0]*delta[a1][d0] + PQ[a1]*delta[a0][d0] + PQ[d0]*delta[a0][a1])) + PQ[d0]*(PQ[a1]*PQ[b1]*(PQ[a0]*delta[b0][c0] + PQ[b0]*delta[a0][c0] + PQ[c0]*delta[a0][b0]) + PQ[b0]*(PQ[a0]*(PQ[a1]*delta[b1][c0] + PQ[b1]*delta[a1][c0] + PQ[c0]*delta[a1][b1]) + PQ[a1]*PQ[c0]*delta[a0][b1]))
                    )

                    + S1 * S1 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[c0]*PQ[d0]*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S1 * S2 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*(-PQ[c0]*QD_0 - PQ[d0]*QC_0)
                    )

                )

                + F6_t[6] * (

                    S1 * S1 * S2 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*PQ[c0]*PQ[d0]
                    )

                )

            );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * pp_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

// DDPP altB DDPP2 FP64
__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDPP_altB2_FP64(double*         mat_J,
                       const double*   p_prim_info,
                       const uint32_t  p_prim_count,
                       const double*   d_prim_info,
                       const uint32_t  d_prim_count,
                       const double*   pp_mat_D,
                       const uint32_t* dd_first_inds_local,
                       const uint32_t* dd_second_inds_local,
                       const double*   dd_pair_data_local,
                       const uint32_t  dd_prim_pair_count_local,
                       const uint32_t* pp_first_inds,
                       const uint32_t* pp_second_inds,
                       const double*   pp_pair_data,
                       const uint32_t  pp_prim_pair_count,
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

        if ((ij >= dd_prim_pair_count_local) || (kl >= pp_prim_pair_count))
        {
            break;
        }

        const auto k = pp_first_inds[kl];
        const auto l = pp_second_inds[kl];

        const auto a_k = p_prim_info[k / 3 + p_prim_count * 0];

        const double r_k[3] = {p_prim_info[k / 3 + p_prim_count * 2],
                               p_prim_info[k / 3 + p_prim_count * 3],
                               p_prim_info[k / 3 + p_prim_count * 4]};

        const auto a_l = p_prim_info[l / 3 + p_prim_count * 0];

        const double r_l[3] = {p_prim_info[l / 3 + p_prim_count * 2],
                               p_prim_info[l / 3 + p_prim_count * 3],
                               p_prim_info[l / 3 + p_prim_count * 4]};

        const auto S_kl_00 = pp_pair_data[kl];

        const auto c0 = k % 3;
        const auto d0 = l % 3;

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2 = a_k + a_l;

        const auto inv_S2 = 1.0 / S2;
        const auto inv_S4 = 1.0 / (S1 + S2);

        const double PQ[3] = {(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,
                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,
                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1};

        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];

        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);

        double F6_t[7];

        gpu::computeBoysFunction(F6_t, S1 * S2 * inv_S4 * r2_PQ, 6, boys_func_table, boys_func_ft);

        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);
        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);

        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (

                + F6_t[4] * (

                    0.5 * S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        -delta[c0][d0]*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))

                        -PQ[a0]*(PA_1*(PQ[b0]*(PQ[c0]*delta[b1][d0] + PQ[d0]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[b0][d0] + PQ[d0]*delta[b0][c0])) + PB_0*(PQ[a1]*(PQ[c0]*delta[b1][d0] + PQ[d0]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[a1][d0] + PQ[d0]*delta[a1][c0])) + PB_1*(PQ[a1]*(PQ[c0]*delta[b0][d0] + PQ[d0]*delta[b0][c0]) + PQ[b0]*(PQ[c0]*delta[a1][d0] + PQ[d0]*delta[a1][c0]))) - PQ[a1]*(PA_0*(PQ[b0]*(PQ[c0]*delta[b1][d0] + PQ[d0]*delta[b1][c0]) + PQ[b1]*(PQ[c0]*delta[b0][d0] + PQ[d0]*delta[b0][c0])) + (PB_0*PQ[b1] + PB_1*PQ[b0])*(PQ[c0]*delta[a0][d0] + PQ[d0]*delta[a0][c0])) - PQ[b0]*PQ[b1]*(PA_0*(PQ[c0]*delta[a1][d0] + PQ[d0]*delta[a1][c0]) + PA_1*(PQ[c0]*delta[a0][d0] + PQ[d0]*delta[a0][c0]))

                        +PQ[c0]*PQ[d0]*(-PA_0*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) - PQ[a0]*(PA_1*delta[b0][b1] + PB_0*delta[a1][b1] + PB_1*delta[a1][b0]) + PQ[a0]*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) - PQ[a1]*(PB_0*delta[a0][b1] + PB_1*delta[a0][b0]) + PQ[a1]*(PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PQ[b0]*PQ[b1]*delta[a0][a1] - PQ[b0]*(PA_1*delta[a0][b1] + PB_1*delta[a0][a1]) - PQ[b1]*(PA_1*delta[a0][b0] + PB_0*delta[a0][a1]))
                    )

                    + 0.5 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*QD_0*delta[b1][c0]

                        +PQ[a0]*PQ[a1]*QC_0*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0]) + PQ[b1]*(PQ[a0]*QD_0*(PQ[a1]*delta[b0][c0] + PQ[b0]*delta[a1][c0]) + PQ[b0]*(PQ[a1]*QD_0*delta[a0][c0] + QC_0*(PQ[a0]*delta[a1][d0] + PQ[a1]*delta[a0][d0])))

                        +(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(PQ[a0]*(PQ[a1]*delta[b0][b1] + PQ[b0]*delta[a1][b1] + PQ[b1]*delta[a1][b0]) + PQ[a1]*(PQ[b0]*delta[a0][b1] + PQ[b1]*delta[a0][b0]) + PQ[b0]*PQ[b1]*delta[a0][a1])

                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*delta[c0][d0]
                    )

                    + S1 * S1 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[c0]*PQ[d0]*(PA_0*PQ[b0]*(PA_1*PQ[b1] + PB_1*PQ[a1]) + PA_1*PQ[a0]*(PB_0*PQ[b1] + PB_1*PQ[b0]) + PB_0*PQ[a1]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S1 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        -(PQ[c0]*QD_0 + PQ[d0]*QC_0)*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S2 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*QC_0*QD_0
                    )

                    + 0.25 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*(delta[b0][b1]*delta[c0][d0] + delta[b0][c0]*delta[b1][d0] + delta[b0][d0]*delta[b1][c0])

                        +PQ[b0]*(PQ[a0]*(delta[a1][b1]*delta[c0][d0] + delta[a1][c0]*delta[b1][d0] + delta[a1][d0]*delta[b1][c0]) + PQ[a1]*(delta[a0][b1]*delta[c0][d0] + delta[a0][c0]*delta[b1][d0] + delta[a0][d0]*delta[b1][c0]) + PQ[b1]*(delta[a0][a1]*delta[c0][d0] + delta[a0][c0]*delta[a1][d0] + delta[a0][d0]*delta[a1][c0]) + PQ[c0]*(delta[a0][a1]*delta[b1][d0] + delta[a0][b1]*delta[a1][d0] + delta[a0][d0]*delta[a1][b1]) + PQ[d0]*(delta[a0][a1]*delta[b1][c0] + delta[a0][b1]*delta[a1][c0] + delta[a0][c0]*delta[a1][b1])) + PQ[b1]*(PQ[a0]*(delta[a1][b0]*delta[c0][d0] + delta[a1][c0]*delta[b0][d0] + delta[a1][d0]*delta[b0][c0]) + PQ[a1]*(delta[a0][b0]*delta[c0][d0] + delta[a0][c0]*delta[b0][d0] + delta[a0][d0]*delta[b0][c0]) + PQ[c0]*(delta[a0][a1]*delta[b0][d0] + delta[a0][b0]*delta[a1][d0] + delta[a0][d0]*delta[a1][b0]) + PQ[d0]*(delta[a0][a1]*delta[b0][c0] + delta[a0][b0]*delta[a1][c0] + delta[a0][c0]*delta[a1][b0])) + PQ[c0]*(PQ[a0]*(delta[a1][b0]*delta[b1][d0] + delta[a1][b1]*delta[b0][d0] + delta[a1][d0]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][d0] + delta[a0][b1]*delta[b0][d0] + delta[a0][d0]*delta[b0][b1]) + PQ[d0]*(delta[a0][a1]*delta[b0][b1] + delta[a0][b0]*delta[a1][b1] + delta[a0][b1]*delta[a1][b0])) + PQ[d0]*(PQ[a0]*(delta[a1][b0]*delta[b1][c0] + delta[a1][b1]*delta[b0][c0] + delta[a1][c0]*delta[b0][b1]) + PQ[a1]*(delta[a0][b0]*delta[b1][c0] + delta[a0][b1]*delta[b0][c0] + delta[a0][c0]*delta[b0][b1]))
                    )

                )

                + F6_t[5] * (

                    (-0.5) * S1 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*delta[c0][d0]

                        +PQ[c0]*(PQ[a0]*(PQ[a1]*(PQ[b0]*delta[b1][d0] + PQ[b1]*delta[b0][d0] + PQ[d0]*delta[b0][b1]) + PQ[b1]*PQ[d0]*delta[a1][b0]) + PQ[b0]*PQ[b1]*(PQ[a0]*delta[a1][d0] + PQ[a1]*delta[a0][d0] + PQ[d0]*delta[a0][a1])) + PQ[d0]*(PQ[a1]*PQ[b1]*(PQ[a0]*delta[b0][c0] + PQ[b0]*delta[a0][c0] + PQ[c0]*delta[a0][b0]) + PQ[b0]*(PQ[a0]*(PQ[a1]*delta[b1][c0] + PQ[b1]*delta[a1][c0] + PQ[c0]*delta[a1][b1]) + PQ[a1]*PQ[c0]*delta[a0][b1]))
                    )

                    + S1 * S1 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[c0]*PQ[d0]*(PQ[a0]*PQ[b1]*(PA_1*PQ[b0] + PB_0*PQ[a1]) + PQ[a1]*PQ[b0]*(PA_0*PQ[b1] + PB_1*PQ[a0]))
                    )

                    + S1 * S2 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*(-PQ[c0]*QD_0 - PQ[d0]*QC_0)
                    )

                )

                + F6_t[6] * (

                    S1 * S1 * S2 * S2 * S2 * S2 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * inv_S4 * (
                        +PQ[a0]*PQ[a1]*PQ[b0]*PQ[b1]*PQ[c0]*PQ[d0]
                    )

                )

            );

        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry
        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)
        ERIs[threadIdx.y] += eri_ijkl * pp_mat_D[kl] * (static_cast<double>(k != l) + 1.0);
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

// DDPP altB DDPP2 FP32
__global__ void __launch_bounds__(TILE_SIZE_J)
computeCoulombFockDDPP_altB2_FP32(double*         mat_J,
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

        const float r_k_f[3] = {p_prim_info_f[k / 3 + p_prim_count * 2],
                               p_prim_info_f[k / 3 + p_prim_count * 3],
                               p_prim_info_f[k / 3 + p_prim_count * 4]};

        const auto a_l_f = p_prim_info_f[l / 3 + p_prim_count * 0];

        const float r_l_f[3] = {p_prim_info_f[l / 3 + p_prim_count * 2],
                               p_prim_info_f[l / 3 + p_prim_count * 3],
                               p_prim_info_f[l / 3 + p_prim_count * 4]};

        const auto S_kl_00_f = pp_pair_data_f[kl];

        const auto c0 = k % 3;
        const auto d0 = l % 3;

        // J. Chem. Phys. 84, 3963-3974 (1986)

        const auto S2_f = a_k_f + a_l_f;

        const auto inv_S2_f = 1.0f / S2_f;
        const auto inv_S4_f = 1.0f / (S1_f + S2_f);

        const float PQ_f[3] = {(a_k_f * r_k_f[0] + a_l_f * r_l_f[0]) * inv_S2_f - (a_i_f * r_i_f[0] + a_j_f * r_j_f[0]) * inv_S1_f,
                              (a_k_f * r_k_f[1] + a_l_f * r_l_f[1]) * inv_S2_f - (a_i_f * r_i_f[1] + a_j_f * r_j_f[1]) * inv_S1_f,
                              (a_k_f * r_k_f[2] + a_l_f * r_l_f[2]) * inv_S2_f - (a_i_f * r_i_f[2] + a_j_f * r_j_f[2]) * inv_S1_f};

        const auto r2_PQ_f = PQ_f[0] * PQ_f[0] + PQ_f[1] * PQ_f[1] + PQ_f[2] * PQ_f[2];

        const auto Lambda_f = sqrtf(4.0f * S1_f * S2_f * MATH_CONST_INV_PI_F * inv_S4_f);

        float F6_t_f[7];

        gpu::computeBoysFunction_f(F6_t_f, S1_f * S2_f * inv_S4_f * r2_PQ_f, 6, boys_func_table_f, boys_func_ft_f);

        const auto QC_0_f = (a_l_f * inv_S2_f) * (r_l_f[c0] - r_k_f[c0]);
        const auto QD_0_f = (-a_k_f * inv_S2_f) * (r_l_f[d0] - r_k_f[d0]);

        const float eri_ijkl_f = Lambda_f * S_ij_00_f * S_kl_00_f * (

                + F6_t_f[4] * (

                    0.5f * S1_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        -delta_f[c0][d0]*(PQ_f[a0]*PQ_f[b1]*(PA_1_f*PQ_f[b0] + PB_0_f*PQ_f[a1]) + PQ_f[a1]*PQ_f[b0]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))

                        -PQ_f[a0]*(PA_1_f*(PQ_f[b0]*(PQ_f[c0]*delta_f[b1][d0] + PQ_f[d0]*delta_f[b1][c0]) + PQ_f[b1]*(PQ_f[c0]*delta_f[b0][d0] + PQ_f[d0]*delta_f[b0][c0])) + PB_0_f*(PQ_f[a1]*(PQ_f[c0]*delta_f[b1][d0] + PQ_f[d0]*delta_f[b1][c0]) + PQ_f[b1]*(PQ_f[c0]*delta_f[a1][d0] + PQ_f[d0]*delta_f[a1][c0])) + PB_1_f*(PQ_f[a1]*(PQ_f[c0]*delta_f[b0][d0] + PQ_f[d0]*delta_f[b0][c0]) + PQ_f[b0]*(PQ_f[c0]*delta_f[a1][d0] + PQ_f[d0]*delta_f[a1][c0]))) - PQ_f[a1]*(PA_0_f*(PQ_f[b0]*(PQ_f[c0]*delta_f[b1][d0] + PQ_f[d0]*delta_f[b1][c0]) + PQ_f[b1]*(PQ_f[c0]*delta_f[b0][d0] + PQ_f[d0]*delta_f[b0][c0])) + (PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0])*(PQ_f[c0]*delta_f[a0][d0] + PQ_f[d0]*delta_f[a0][c0])) - PQ_f[b0]*PQ_f[b1]*(PA_0_f*(PQ_f[c0]*delta_f[a1][d0] + PQ_f[d0]*delta_f[a1][c0]) + PA_1_f*(PQ_f[c0]*delta_f[a0][d0] + PQ_f[d0]*delta_f[a0][c0]))

                        +PQ_f[c0]*PQ_f[d0]*(-PA_0_f*(PQ_f[a1]*delta_f[b0][b1] + PQ_f[b0]*delta_f[a1][b1] + PQ_f[b1]*delta_f[a1][b0]) - PQ_f[a0]*(PA_1_f*delta_f[b0][b1] + PB_0_f*delta_f[a1][b1] + PB_1_f*delta_f[a1][b0]) + PQ_f[a0]*(PQ_f[a1]*delta_f[b0][b1] + PQ_f[b0]*delta_f[a1][b1] + PQ_f[b1]*delta_f[a1][b0]) - PQ_f[a1]*(PB_0_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][b0]) + PQ_f[a1]*(PQ_f[b0]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][b0]) + PQ_f[b0]*PQ_f[b1]*delta_f[a0][a1] - PQ_f[b0]*(PA_1_f*delta_f[a0][b1] + PB_1_f*delta_f[a0][a1]) - PQ_f[b1]*(PA_1_f*delta_f[a0][b0] + PB_0_f*delta_f[a0][a1]))
                    )

                    + 0.5f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[a0]*PQ_f[a1]*PQ_f[b0]*QD_0_f*delta_f[b1][c0]

                        +PQ_f[a0]*PQ_f[a1]*QC_0_f*(PQ_f[b0]*delta_f[b1][d0] + PQ_f[b1]*delta_f[b0][d0]) + PQ_f[b1]*(PQ_f[a0]*QD_0_f*(PQ_f[a1]*delta_f[b0][c0] + PQ_f[b0]*delta_f[a1][c0]) + PQ_f[b0]*(PQ_f[a1]*QD_0_f*delta_f[a0][c0] + QC_0_f*(PQ_f[a0]*delta_f[a1][d0] + PQ_f[a1]*delta_f[a0][d0])))

                        +(PQ_f[c0]*QD_0_f + PQ_f[d0]*QC_0_f)*(PQ_f[a0]*(PQ_f[a1]*delta_f[b0][b1] + PQ_f[b0]*delta_f[a1][b1] + PQ_f[b1]*delta_f[a1][b0]) + PQ_f[a1]*(PQ_f[b0]*delta_f[a0][b1] + PQ_f[b1]*delta_f[a0][b0]) + PQ_f[b0]*PQ_f[b1]*delta_f[a0][a1])

                        +PQ_f[a0]*PQ_f[a1]*PQ_f[b0]*PQ_f[b1]*delta_f[c0][d0]
                    )

                    + S1_f * S1_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[c0]*PQ_f[d0]*(PA_0_f*PQ_f[b0]*(PA_1_f*PQ_f[b1] + PB_1_f*PQ_f[a1]) + PA_1_f*PQ_f[a0]*(PB_0_f*PQ_f[b1] + PB_1_f*PQ_f[b0]) + PB_0_f*PQ_f[a1]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))
                    )

                    + S1_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        -(PQ_f[c0]*QD_0_f + PQ_f[d0]*QC_0_f)*(PQ_f[a0]*PQ_f[b1]*(PA_1_f*PQ_f[b0] + PB_0_f*PQ_f[a1]) + PQ_f[a1]*PQ_f[b0]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))
                    )

                    + S2_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[a0]*PQ_f[a1]*PQ_f[b0]*PQ_f[b1]*QC_0_f*QD_0_f
                    )

                    + 0.25f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[a0]*PQ_f[a1]*(delta_f[b0][b1]*delta_f[c0][d0] + delta_f[b0][c0]*delta_f[b1][d0] + delta_f[b0][d0]*delta_f[b1][c0])

                        +PQ_f[b0]*(PQ_f[a0]*(delta_f[a1][b1]*delta_f[c0][d0] + delta_f[a1][c0]*delta_f[b1][d0] + delta_f[a1][d0]*delta_f[b1][c0]) + PQ_f[a1]*(delta_f[a0][b1]*delta_f[c0][d0] + delta_f[a0][c0]*delta_f[b1][d0] + delta_f[a0][d0]*delta_f[b1][c0]) + PQ_f[b1]*(delta_f[a0][a1]*delta_f[c0][d0] + delta_f[a0][c0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][c0]) + PQ_f[c0]*(delta_f[a0][a1]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b1]) + PQ_f[d0]*(delta_f[a0][a1]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b1])) + PQ_f[b1]*(PQ_f[a0]*(delta_f[a1][b0]*delta_f[c0][d0] + delta_f[a1][c0]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][c0]) + PQ_f[a1]*(delta_f[a0][b0]*delta_f[c0][d0] + delta_f[a0][c0]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][c0]) + PQ_f[c0]*(delta_f[a0][a1]*delta_f[b0][d0] + delta_f[a0][b0]*delta_f[a1][d0] + delta_f[a0][d0]*delta_f[a1][b0]) + PQ_f[d0]*(delta_f[a0][a1]*delta_f[b0][c0] + delta_f[a0][b0]*delta_f[a1][c0] + delta_f[a0][c0]*delta_f[a1][b0])) + PQ_f[c0]*(PQ_f[a0]*(delta_f[a1][b0]*delta_f[b1][d0] + delta_f[a1][b1]*delta_f[b0][d0] + delta_f[a1][d0]*delta_f[b0][b1]) + PQ_f[a1]*(delta_f[a0][b0]*delta_f[b1][d0] + delta_f[a0][b1]*delta_f[b0][d0] + delta_f[a0][d0]*delta_f[b0][b1]) + PQ_f[d0]*(delta_f[a0][a1]*delta_f[b0][b1] + delta_f[a0][b0]*delta_f[a1][b1] + delta_f[a0][b1]*delta_f[a1][b0])) + PQ_f[d0]*(PQ_f[a0]*(delta_f[a1][b0]*delta_f[b1][c0] + delta_f[a1][b1]*delta_f[b0][c0] + delta_f[a1][c0]*delta_f[b0][b1]) + PQ_f[a1]*(delta_f[a0][b0]*delta_f[b1][c0] + delta_f[a0][b1]*delta_f[b0][c0] + delta_f[a0][c0]*delta_f[b0][b1]))
                    )

                )

                + F6_t_f[5] * (

                    (-0.5f) * S1_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[a0]*PQ_f[a1]*PQ_f[b0]*PQ_f[b1]*delta_f[c0][d0]

                        +PQ_f[c0]*(PQ_f[a0]*(PQ_f[a1]*(PQ_f[b0]*delta_f[b1][d0] + PQ_f[b1]*delta_f[b0][d0] + PQ_f[d0]*delta_f[b0][b1]) + PQ_f[b1]*PQ_f[d0]*delta_f[a1][b0]) + PQ_f[b0]*PQ_f[b1]*(PQ_f[a0]*delta_f[a1][d0] + PQ_f[a1]*delta_f[a0][d0] + PQ_f[d0]*delta_f[a0][a1])) + PQ_f[d0]*(PQ_f[a1]*PQ_f[b1]*(PQ_f[a0]*delta_f[b0][c0] + PQ_f[b0]*delta_f[a0][c0] + PQ_f[c0]*delta_f[a0][b0]) + PQ_f[b0]*(PQ_f[a0]*(PQ_f[a1]*delta_f[b1][c0] + PQ_f[b1]*delta_f[a1][c0] + PQ_f[c0]*delta_f[a1][b1]) + PQ_f[a1]*PQ_f[c0]*delta_f[a0][b1]))
                    )

                    + S1_f * S1_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[c0]*PQ_f[d0]*(PQ_f[a0]*PQ_f[b1]*(PA_1_f*PQ_f[b0] + PB_0_f*PQ_f[a1]) + PQ_f[a1]*PQ_f[b0]*(PA_0_f*PQ_f[b1] + PB_1_f*PQ_f[a0]))
                    )

                    + S1_f * S2_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[a0]*PQ_f[a1]*PQ_f[b0]*PQ_f[b1]*(-PQ_f[c0]*QD_0_f - PQ_f[d0]*QC_0_f)
                    )

                )

                + F6_t_f[6] * (

                    S1_f * S1_f * S2_f * S2_f * S2_f * S2_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * (
                        +PQ_f[a0]*PQ_f[a1]*PQ_f[b0]*PQ_f[b1]*PQ_f[c0]*PQ_f[d0]
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
        double J_ij = 0.0;

        for (uint32_t n = 0; n < TILE_DIM_LARGE; n++)
        {
            J_ij += ERIs[n];
        }

        mat_J[ij] += J_ij;
    }
}
