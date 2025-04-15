from mpi4py import MPI
import numpy as np

# --- MPI 初期化 ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

# --- 領域分割数の指定 ---
nprocr = 3  # R方向（列方向）に分割
nprocz = 2  # Z方向（行方向）に分割

assert nproc == nprocr * nprocz, "nproc と nprocr × nprocz の積が一致しません"

# --- 各ランクの2次元インデックス ---
rankz = rank // nprocr
rankr = rank % nprocr

print(f"[rank {rank}] → (rankz, rankr) = ({rankz}, {rankr})")
comm.Barrier()

def get_neighbors(rankz, rankr, nprocz, nprocr):
    neighbors_2dmap = {
        'up':    (rankz + 1, rankr) if rankz < nprocz - 1 else None,
        'down':  (rankz - 1, rankr) if rankz > 0 else None,
        'left':  (rankz, rankr - 1) if rankr > 0 else None,
        'right': (rankz, rankr + 1) if rankr < nprocr - 1 else None,
    }
    
    # ランクに変換
    neighbors = {}
    for key, value in neighbors_2dmap.items():
        if value is not None:
            rankz, rankr = value
            neighbors[key] = rankz * nprocr + rankr
        else:
            neighbors[key] = MPI.PROC_NULL
    return neighbors

neighbors = get_neighbors(rankz, rankr, nprocz, nprocr)
print(f"[rank {rank}] neighbors: {neighbors}")
comm.Barrier()


nr_global = 120  # 有効点数（ゴーストなし）
nz_global = 100

assert nr_global % nprocr == 0, f"nr_global={nr_global} is not divisible by nprocr={nprocr}"
assert nz_global % nprocz == 0, f"nz_global={nz_global} is not divisible by nprocz={nprocz}"
nr_local = nr_global // nprocr
nz_local = nz_global // nprocz
ir_start = rankr * nr_local
iz_start = rankz * nz_local

# --- Parameters ---
liter = 1000
eps = 0.01
rm = 1.0
rp = 0.5
kappa = 1.0
delta = 0.0
b0 = 1.0
q0 = 1.4

# --- Derived ---
r_0 = np.sqrt(rm**2 + rp**2)
aa = rm * rp / r_0
r_max = r_0 - delta * aa
z_max = kappa * aa
E_sol = 2.0 * r_0 * z_max / np.sqrt((r_0**2 - r_max**2)**2 + 4.0 * aa**2 * r_0**2)
d_sol = 1.0 - 2.0 * r_0**2 * (r_0**2 - r_max**2) / ((r_0**2 - r_max**2)**2 + 4.0 * aa**2 * r_0**2)
psi_s = aa**2 * E_sol * b0 / (2.0 * q0)

# --- Grad–Shafranov solver ---
def solve_grad_shafranov():
  
    # --- Grid ---
    r_left = r_0 * np.sqrt(1.0 - 2.0 * aa / r_0)
    r_right = r_0 * np.sqrt(1.0 + 2.0 * aa / r_0)
    dr = (r_right - r_left) / (nr_global - 1)
    z_top = z_max
    z_bottom = -z_max
    dz = (z_top - z_bottom) / (nz_global - 1)
    # # nr+2 点、nz+2 点。前後１点ずつゴーストグリッド。
    # rr = np.array([r_left + dr * (i - 1) for i in range(nr + 2)])
    # zz = np.array([dz * (j - nz // 2 - 1) for j in range(nz + 2)])
    # --- R方向: [ir_start -1, ..., ir_start + nr_local] → 長さ nr_local + 2
    rr = np.array([r_left + dr * (i - 1) for i in range(ir_start, ir_start + nr_local + 2)])
    # --- Z方向: [iz_start -1, ..., iz_start + nz_local] → 長さ nz_local + 2
    zz = np.array([dz * (j - nz_global // 2 - 1) for j in range(iz_start, iz_start + nz_local + 2)])
    RR, ZZ = np.meshgrid(rr, zz, indexing='xy')

    # --- Mask ---
    def inside_mask(rr, zz):
        t1 = ((1 - d_sol) * rr**2 + d_sol * r_0**2) * zz**2 / E_sol**2
        t2 = 0.25 * (rr**2 - r_0**2)**2
        return t1 + t2 < aa**2 * r_0**2

    mask = inside_mask(RR, ZZ)

    # --- RHS and Initial ---
    rhs = np.zeros_like(RR)
    rhs[mask] = (
        2.0 * RR[mask]**2 * psi_s / (aa**2 * r_0**2) * (1.0 + (1.0 - d_sol) / E_sol**2)
        + 2.0 * d_sol * psi_s / (aa**2 * E_sol**2)
    )
    psi = np.zeros_like(rhs)

    # --- Operator ---
    def exchange_ghost_cells(psi):
        sendbuf = psi[-2, :].copy()
        recvbuf = np.zeros_like(sendbuf)
        comm.Sendrecv(sendbuf=sendbuf, dest=neighbors['up'],
                      recvbuf=recvbuf, source=neighbors['down'])
        psi[0, :] = recvbuf

        sendbuf = psi[1, :].copy()
        recvbuf = np.zeros_like(sendbuf)
        comm.Sendrecv(sendbuf=sendbuf, dest=neighbors['down'],
                      recvbuf=recvbuf, source=neighbors['up'])
        psi[-1, :] = recvbuf

        sendbuf = np.ascontiguousarray(psi[:, 1])
        recvbuf = np.zeros_like(sendbuf)
        comm.Sendrecv(sendbuf=sendbuf, dest=neighbors['left'],
                      recvbuf=recvbuf, source=neighbors['right'])
        psi[:, -1] = recvbuf

        sendbuf = np.ascontiguousarray(psi[:, -2])
        recvbuf = np.zeros_like(sendbuf)
        comm.Sendrecv(sendbuf=sendbuf, dest=neighbors['right'],
                      recvbuf=recvbuf, source=neighbors['left'])
        psi[:, 0] = recvbuf

    def apply_operator(phi):
        exchange_ghost_cells(phi)
        out = np.zeros_like(phi)
        out[1:-1, 1:-1] = (
            (1.0 / dr**2 - 0.5 / (dr * RR[1:-1, 1:-1])) * phi[2:, 1:-1] +
            (1.0 / dr**2 + 0.5 / (dr * RR[1:-1, 1:-1])) * phi[:-2, 1:-1] +
            1.0 / dz**2 * (phi[1:-1, 2:] + phi[1:-1, :-2]) -
            2.0 * (1.0 / dr**2 + 1.0 / dz**2) * phi[1:-1, 1:-1]
        )
        return out * mask

    def integrate(array):
        local_sum = np.sum(array[1:-1,1:-1]) # Avoiding ghost cells
        global_sum = comm.allreduce(local_sum, op=MPI.SUM)
        return global_sum

    def cg_step(psi, r, p, rs_old):
        Ap = apply_operator(p)
        alpha = rs_old / integrate(p * Ap)
        psi += alpha * p
        r_new = r - alpha * Ap
        rs_new = integrate(r_new * r_new)
        beta = rs_new / rs_old
        p_new = r_new + beta * p
        return psi, r_new, p_new, rs_new

    iterative_solver = cg_step

    # --- Iterative loop ---
    r = rhs - apply_operator(psi)
    p = r.copy()
    rs_old = integrate(r * r)
    errors = []
    psi_all = [psi.copy()]

    for i in range(liter):
        psi, r, p, rs_new = iterative_solver(psi, r, p, rs_old)
        rel_error = np.sqrt(rs_new) / np.sqrt(integrate(rhs**2))
        errors.append(rel_error)
        psi_all.append(psi.copy())
        if rel_error < eps:
            print(f"Converged at iter = {i}, error = {rel_error}")
            break
        rs_old = rs_new
    else:
        print(f"Did not converge within {liter} iterations. Final error = {rel_error}")

    return psi_all, errors, RR, ZZ

# --- Run iterative solvers ---
psi_all, errors, RR, ZZ = solve_grad_shafranov()
psi = psi_all[-1]

if rank == 0:
    print("Grad-Shafranov CG solve complete.")

print(psi.shape, RR.shape, ZZ.shape)
import xarray as xr
filename = f"psi.{rank:03d}.nc"
da = xr.DataArray(psi[1:-1,1:-1], coords={"r": RR[0,1:-1], "z": ZZ[1:-1,0]}, dims=("z", "r"), name="psi")
ds = xr.Dataset({"psi": da})
ds.to_netcdf(filename)
print(f"Saved local psi to {filename}")



# from mpi4py import MPI
# import numpy as np
# import xarray as xr

# # --- Physical parameters ---
# liter = 1000
# eps = 0.01
# rm = 1.0
# rp = 0.5
# kappa = 1.0
# delta = 0.0
# b0 = 1.0
# q0 = 1.4

# # --- Derived ---
# r_0 = np.sqrt(rm**2 + rp**2)
# aa = rm * rp / r_0
# r_max = r_0 - delta * aa
# z_max = kappa * aa
# E_sol = 2.0 * r_0 * z_max / np.sqrt((r_0**2 - r_max**2)**2 + 4.0 * aa**2 * r_0**2)
# d_sol = 1.0 - 2.0 * r_0**2 * (r_0**2 - r_max**2) / ((r_0**2 - r_max**2)**2 + 4.0 * aa**2 * r_0**2)
# psi_s = aa**2 * E_sol * b0 / (2.0 * q0)

# # --- Grid parameters (global) ---
# Nr_global = 120
# Nz_global = 100

# # --- MPI setup (2D Cartesian) ---
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
# dims = MPI.Compute_dims(size, 2)
# periods = [False, False]
# cart_comm = comm.Create_cart(dims, periods=periods, reorder=True)
# coords = cart_comm.Get_coords(rank)
# nbrs = dict()
# nbrs['up'], nbrs['down'] = cart_comm.Shift(1, 1)
# nbrs['left'], nbrs['right'] = cart_comm.Shift(0, 1)

# # --- Local grid size (with halo) ---
# Nr_local = Nr_global // dims[0]
# Nz_local = Nz_global // dims[1]
# Nr = Nr_local + 2
# Nz = Nz_local + 2

# # --- Grad–Shafranov solver ---
# def solve_grad_shafranov(flag_solver="cg"):
#     assert flag_solver in ["cg", "sd", "jacobi", "gauss_seidel", "sor"], f"Unknown solver: {flag_solver}"

#     # --- Grid ---
#     r_left = r_0 * np.sqrt(1.0 - 2.0 * aa / r_0)
#     r_right = r_0 * np.sqrt(1.0 + 2.0 * aa / r_0)
#     dr = (r_right - r_left) / (nr - 2)
#     z_top = z_max
#     z_bottom = -z_max
#     dz = (z_top - z_bottom) / (nz - 2)
#     rr = np.array([r_left + dr * (ir - 1) for ir in range(nr + 1)])
#     zz = np.array([dz * (iz - nz // 2) for iz in range(nz + 1)])
#     RR, ZZ = np.meshgrid(rr, zz, indexing='xy')

#     # --- Mask ---
#     def inside_mask(rr, zz):
#         t1 = ((1 - d_sol) * rr**2 + d_sol * r_0**2) * zz**2 / E_sol**2
#         t2 = 0.25 * (rr**2 - r_0**2)**2
#         return t1 + t2 < aa**2 * r_0**2

#     mask = inside_mask(RR, ZZ)

#     # --- RHS and Initial ---
#     rhs = np.zeros_like(RR)
#     rhs[mask] = (
#         2.0 * RR[mask]**2 * psi_s / (aa**2 * r_0**2) * (1.0 + (1.0 - d_sol) / E_sol**2)
#         + 2.0 * d_sol * psi_s / (aa**2 * E_sol**2)
#     )
#     psi = np.zeros_like(rhs)

#     # --- Operator ---
#     def apply_operator(phi):
#         out = np.zeros_like(phi)
#         out[1:-1, 1:-1] = (
#             (1.0 / dr**2 - 0.5 / (dr * RR[1:-1, 1:-1])) * phi[2:, 1:-1] +
#             (1.0 / dr**2 + 0.5 / (dr * RR[1:-1, 1:-1])) * phi[:-2, 1:-1] +
#             1.0 / dz**2 * (phi[1:-1, 2:] + phi[1:-1, :-2]) -
#             2.0 * (1.0 / dr**2 + 1.0 / dz**2) * phi[1:-1, 1:-1]
#         )
#         return out * mask

#     # --- Solver steps ---
#     def sd_step(psi, r, p, rs_old):
#         Ar = apply_operator(r)
#         alpha = np.sum(r * r) / np.sum(r * Ar)
#         psi += alpha * r
#         r_new = r - alpha * Ar
#         return psi, r_new, r_new, np.sum(r_new * r_new)

#     def cg_step(psi, r, p, rs_old):
#         Ap = apply_operator(p)
#         alpha = rs_old / np.sum(p * Ap)
#         psi += alpha * p
#         r_new = r - alpha * Ap
#         rs_new = np.sum(r_new * r_new)
#         beta = rs_new / rs_old
#         p_new = r_new + beta * p
#         return psi, r_new, p_new, rs_new

#     def jacobi_step(psi, _, __, ___):
#         psi_new = psi.copy()
#         coef1 = dz**2 / (2 * (dr**2 + dz**2))
#         coef2 = dr**2 / (2 * (dr**2 + dz**2))
#         coef3 = dr**2 * dz**2 / (2 * (dr**2 + dz**2))
#         for iz in range(1, nz):
#             for ir in range(1, nr):
#                 if mask[iz, ir]:
#                     psi_new[iz, ir] = coef1 * (1 - 0.5 * dr / RR[iz, ir]) * psi[iz, ir + 1] + \
#                                        coef1 * (1 + 0.5 * dr / RR[iz, ir]) * psi[iz, ir - 1] + \
#                                        coef2 * (psi[iz + 1, ir] + psi[iz - 1, ir]) - \
#                                        coef3 * rhs[iz, ir]
#         r_new = rhs - apply_operator(psi_new)
#         return psi_new, r_new, None, np.sum(r_new * r_new)

#     def gauss_seidel_step(psi, _, __, ___):
#         coef1 = dz**2 / (2 * (dr**2 + dz**2))
#         coef2 = dr**2 / (2 * (dr**2 + dz**2))
#         coef3 = dr**2 * dz**2 / (2 * (dr**2 + dz**2))
#         for iz in range(1, nz):
#             for ir in range(1, nr):
#                 if mask[iz, ir]:
#                     psi[iz, ir] = coef1 * (1 - 0.5 * dr / RR[iz, ir]) * psi[iz, ir + 1] + \
#                                   coef1 * (1 + 0.5 * dr / RR[iz, ir]) * psi[iz, ir - 1] + \
#                                   coef2 * (psi[iz + 1, ir] + psi[iz - 1, ir]) - \
#                                   coef3 * rhs[iz, ir]
#         r_new = rhs - apply_operator(psi)
#         return psi, r_new, None, np.sum(r_new * r_new)

#     def sor_step(psi, _, __, ___):
#         coef1 = dz**2 / (2 * (dr**2 + dz**2))
#         coef2 = dr**2 / (2 * (dr**2 + dz**2))
#         coef3 = dr**2 * dz**2 / (2 * (dr**2 + dz**2))
#         alpha = 1.82
#         for iz in range(1, nz):
#             for ir in range(1, nr):
#                 if mask[iz, ir]:
#                     update_val = coef1 * (1 - 0.5 * dr / RR[iz, ir]) * psi[iz, ir + 1] + \
#                                  coef1 * (1 + 0.5 * dr / RR[iz, ir]) * psi[iz, ir - 1] + \
#                                  coef2 * (psi[iz + 1, ir] + psi[iz - 1, ir]) - \
#                                  coef3 * rhs[iz, ir]
#                     psi[iz, ir] = (1 - alpha) * psi[iz, ir] + alpha * update_val
#         r_new = rhs - apply_operator(psi)
#         return psi, r_new, None, np.sum(r_new * r_new)

#     solver_dict = {
#         "cg": cg_step,
#         "sd": sd_step,
#         "jacobi": jacobi_step,
#         "gauss_seidel": gauss_seidel_step,
#         "sor": sor_step
#     }
#     iterative_solver = solver_dict[flag_solver]

#     # --- Iterative loop ---
#     r = rhs - apply_operator(psi)
#     p = r.copy()
#     rs_old = np.sum(r * r)
#     errors = []
#     psi_all = [psi.copy()]

#     for i in range(liter):
#         psi, r, p, rs_new = iterative_solver(psi, r, p, rs_old)
#         rel_error = np.sqrt(rs_new) / np.sqrt(np.sum(rhs**2))
#         errors.append(rel_error)
#         psi_all.append(psi.copy())
#         if rel_error < eps:
#             print(f"Converged at iter = {i}, error = {rel_error}")
#             break
#         rs_old = rs_new
#     else:
#         print(f"Did not converge within {liter} iterations. Final error = {rel_error}")

#     return psi_all, errors, RR, ZZ

# def compute_boundary_curve(num_points=500):
#     """
#     Return rr_curve, zz_curve as parametric boundary based on Solov'ev parameters.
#     """
#     t = np.linspace(0, 2 * np.pi, num_points)
#     eps_val = aa / r_0
#     sqrt_p = np.sqrt(psi_s / psi_s)  # =1.0
#     rr_curve = r_0 * np.sqrt(1.0 + 2.0 * eps_val * sqrt_p * np.cos(t))
#     zz_curve = (aa * E_sol * sqrt_p * np.sin(t)) / np.sqrt(1.0 + 2.0 * eps_val * sqrt_p * (1.0 - d_sol) * np.cos(t))
#     return rr_curve, zz_curve

# def solovev_solution(rr, zz):
#     """
#     Return Solov'ev analytic solution components: psi, pressure, B^2, rhs.
#     All outputs are masked: outside the plasma domain, values are zero.
#     """
#     term = ((1 - d_sol) * rr**2 + d_sol * r_0**2) * zz**2 / E_sol**2 + 0.25 * (rr**2 - r_0**2)**2
#     mask = term < aa**2 * r_0**2

#     psi_sol = np.zeros_like(rr)
#     pres_sol = np.zeros_like(rr)
#     T2_sol = np.zeros_like(rr)
#     rhs_sol = np.zeros_like(rr)

#     psi_tmp = psi_s / (aa**2 * r_0**2) * (
#         ((1 - d_sol) * rr[mask]**2 + d_sol * r_0**2) * zz[mask]**2 / E_sol**2 +
#         0.25 * (rr[mask]**2 - r_0**2)**2
#     ) - psi_s

#     psi_sol[mask] = psi_tmp
#     pres_sol[mask] = 4.0 * psi_s / (aa**2 * r_0**2) * (1 + (1 - d_sol) / E_sol**2) * (-psi_tmp)
#     T2_sol[mask] = (r_0 * b0)**2 - 4.0 * d_sol * psi_s * (psi_tmp + psi_s) / (aa**2 * E_sol**2)
#     rhs_sol[mask] = (
#         2.0 * rr[mask]**2 * psi_s / (aa**2 * r_0**2) * (1 + (1 - d_sol) / E_sol**2) +
#         2.0 * d_sol * psi_s / (aa**2 * E_sol**2)
#     )
#     return psi_sol, pres_sol, T2_sol, rhs_sol, mask

# # --- Run iterative solvers ---
# psi_all, errors, RR, ZZ = solve_grad_shafranov(flag_solver="cg")
# psi = psi_all[-1]


# # --- Verification against analytic solution ---
# rr_curve, zz_curve = compute_boundary_curve()
# psi_sol, pres_sol, T2_sol, rhs_sol, mask = solovev_solution(RR, ZZ)



# # --- Domain decomposition info ---
# r_0 = np.sqrt(rm**2 + rp**2)
# aa = rm * rp / r_0
# r_max = r_0 - delta * aa
# z_max = kappa * aa
# E_sol = 2.0 * r_0 * z_max / np.sqrt((r_0**2 - r_max**2)**2 + 4.0 * aa**2 * r_0**2)
# d_sol = 1.0 - 2.0 * r_0**2 * (r_0**2 - r_max**2) / ((r_0**2 - r_max**2)**2 + 4.0 * aa**2 * r_0**2)
# psi_s = aa**2 * E_sol * b0 / (2.0 * q0)

# r_left = r_0 * np.sqrt(1.0 - 2.0 * aa / r_0)
# r_right = r_0 * np.sqrt(1.0 + 2.0 * aa / r_0)
# dr = (r_right - r_left) / (Nr_global - 2)
# dz = 2 * z_max / (Nz_global - 2)

# ir0 = coords[0] * Nr_local
# iz0 = coords[1] * Nz_local

# r = np.array([r_left + dr * (ir0 + i - 1) for i in range(Nr)])
# z = np.array([dz * (iz0 + j - Nz_global // 2) for j in range(Nz)])
# RR, ZZ = np.meshgrid(r, z, indexing='ij')

# # --- Domain mask ---
# def inside_mask(rr, zz):
#     t1 = ((1 - d_sol) * rr**2 + d_sol * r_0**2) * zz**2 / E_sol**2
#     t2 = 0.25 * (rr**2 - r_0**2)**2
#     return t1 + t2 < aa**2 * r_0**2

# mask = inside_mask(RR, ZZ)

# # --- Right-hand side and initial guess ---
# rhs = np.zeros_like(RR)
# rhs[mask] = (
#     2.0 * RR[mask]**2 * psi_s / (aa**2 * r_0**2) * (1.0 + (1.0 - d_sol) / E_sol**2)
#     + 2.0 * d_sol * psi_s / (aa**2 * E_sol**2)
# )
# psi = np.zeros_like(rhs)

# # --- Halo exchange ---
# def halo_exchange(field):
#     reqs = []

#     if nbrs['up'] != MPI.PROC_NULL:
#         recvbuf_up = np.ascontiguousarray(field[:, -1])
#         reqs.append(cart_comm.Irecv(recvbuf_up, source=nbrs['up'], tag=1))
#         reqs.append(cart_comm.Isend(np.ascontiguousarray(field[:, -2]), dest=nbrs['up'], tag=2))

#     if nbrs['down'] != MPI.PROC_NULL:
#         recvbuf_dn = np.ascontiguousarray(field[:, 0])
#         reqs.append(cart_comm.Irecv(recvbuf_dn, source=nbrs['down'], tag=2))
#         reqs.append(cart_comm.Isend(np.ascontiguousarray(field[:, 1]), dest=nbrs['down'], tag=1))

#     if nbrs['left'] != MPI.PROC_NULL:
#         recvbuf_lf = np.ascontiguousarray(field[0, :])
#         reqs.append(cart_comm.Irecv(recvbuf_lf, source=nbrs['left'], tag=3))
#         reqs.append(cart_comm.Isend(np.ascontiguousarray(field[1, :]), dest=nbrs['left'], tag=4))

#     if nbrs['right'] != MPI.PROC_NULL:
#         recvbuf_rt = np.ascontiguousarray(field[-1, :])
#         reqs.append(cart_comm.Irecv(recvbuf_rt, source=nbrs['right'], tag=4))
#         reqs.append(cart_comm.Isend(np.ascontiguousarray(field[-2, :]), dest=nbrs['right'], tag=3))

#     MPI.Request.Waitall(reqs)

#     # 書き戻し
#     if nbrs['up'] != MPI.PROC_NULL:
#         field[:, -1] = recvbuf_up
#     if nbrs['down'] != MPI.PROC_NULL:
#         field[:, 0] = recvbuf_dn
#     if nbrs['left'] != MPI.PROC_NULL:
#         field[0, :] = recvbuf_lf
#     if nbrs['right'] != MPI.PROC_NULL:
#         field[-1, :] = recvbuf_rt


# # --- Operator application ---
# def apply_operator(phi):
#     halo_exchange(phi)
#     out = np.zeros_like(phi)
#     out[1:-1, 1:-1] = (
#         (1.0 / dr**2 - 0.5 / (dr * RR[1:-1, 1:-1])) * phi[2:, 1:-1] +
#         (1.0 / dr**2 + 0.5 / (dr * RR[1:-1, 1:-1])) * phi[:-2, 1:-1] +
#         1.0 / dz**2 * (phi[1:-1, 2:] + phi[1:-1, :-2]) -
#         2.0 * (1.0 / dr**2 + 1.0 / dz**2) * phi[1:-1, 1:-1]
#     )
#     return out * mask

# # --- Conjugate Gradient solver ---
# def solve_cg(psi, rhs):
#     r = rhs - apply_operator(psi)
#     p = r.copy()
#     rs_old = np.sum(r * r)
#     rs_global = comm.allreduce(rs_old, op=MPI.SUM)
#     for it in range(liter):
#         Ap = apply_operator(p)
#         dot = np.sum(p * Ap)
#         dot_global = comm.allreduce(dot, op=MPI.SUM)
#         alpha = rs_global / dot_global
#         psi += alpha * p
#         r -= alpha * Ap
#         rs_new = np.sum(r * r)
#         rs_new_global = comm.allreduce(rs_new, op=MPI.SUM)
#         if rank == 0:
#             print(f"[{it}] Residual = {np.sqrt(rs_new_global):.3e}")
#         if np.sqrt(rs_new_global) < eps:
#             break
#         beta = rs_new_global / rs_global
#         p = r + beta * p
#         rs_global = rs_new_global
#     return psi

# # --- Main execution ---
# psi = solve_cg(psi, rhs)

# if rank == 0:
#     print("Grad-Shafranov CG solve complete.")

# filename = f"psi.{rank:03d}.nc"
# da = xr.DataArray(psi, coords={"r": r, "z": z}, dims=("r", "z"), name="psi")
# ds = xr.Dataset({"psi": da})
# ds.to_netcdf(filename)
# print(f"Saved local psi to {filename}")
