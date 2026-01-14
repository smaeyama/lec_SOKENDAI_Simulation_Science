from mpi4py import MPI
import numpy as np

# --- MPI initialization ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # MPI rank of each MPI process
nproc = comm.Get_size() # Total number of MPI processes

# --- Numbers of domain decompositions ---
nprocr = 3  # Decomposition in R
nprocz = 2  # Decomposition in Z

assert nproc == nprocr * nprocz, "nproc == nprocr * nprocz is required."

# --- 2D indices for each MPI process ---
rankz = rank // nprocr
rankr = rank % nprocr

print(f"[rank {rank}] → (rankz, rankr) = ({rankz}, {rankr})")
comm.Barrier()

def get_neighbors(rankz, rankr, nprocz, nprocr):
    # Set neighboring MPI processes in 2D indices (rankz, rankr)
    neighbors_2dmap = {
        'up':    (rankz + 1, rankr) if rankz < nprocz - 1 else None,
        'down':  (rankz - 1, rankr) if rankz > 0 else None,
        'left':  (rankz, rankr - 1) if rankr > 0 else None,
        'right': (rankz, rankr + 1) if rankr < nprocr - 1 else None,
    }
    
    # Transrate (rankz, rankr) -> rank
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

# --- Global grid number (without ghost cells) ---
nr_global = 90
nz_global = 80

assert nr_global % nprocr == 0, f"nr_global={nr_global} is not divisible by nprocr={nprocr}"
assert nz_global % nprocz == 0, f"nz_global={nz_global} is not divisible by nprocz={nprocz}"
nr_local = nr_global // nprocr
nz_local = nz_global // nprocz
ir_start = rankr * nr_local
iz_start = rankz * nz_local


# --- Numerical parameters ---
niter = 4000   # Iteration limit
nskip = 1      # Interval for storing history
eps = 0.01     # Convergence tolerance (relative residual)

# --- Physical parameters -
rm = 1.0      # Major radius
rp = 0.5      # Minor radius
kappa = 1.4   # Elongation
delta = 0.5   # Triangularity
b0 = 1.0      # Toroidal magnetic field
q0 = 3.0      # Safety factor

r_0 = np.sqrt(rm**2 + rp**2)  # R0
aa = rm * rp / r_0            # a
r_max = r_0 - delta * aa
z_max = kappa * aa
E_sol = 2.0 * r_0 * z_max / np.sqrt((r_0**2 - r_max**2)**2 + 4.0 * aa**2 * r_0**2) # E
d_sol = 1.0 - 2.0 * r_0**2 * (r_0**2 - r_max**2) / ((r_0**2 - r_max**2)**2 + 4.0 * aa**2 * r_0**2) # d
psi_s = aa**2 * E_sol * b0 / (2.0 * q0) # psi_s

# --- Grad–Shafranov solver ---
def solve_grad_shafranov():
    """
    Grad-Shafranov solver, parallelized by MPI
    """
    # --- Rectangular grid in (Z,R) ---
    r_left = r_0 * np.sqrt(1.0 - 2.0 * aa / r_0)
    r_right = r_0 * np.sqrt(1.0 + 2.0 * aa / r_0)
    dr = (r_right - r_left) / (nr_global - 1)  # Grid spacing in R
    z_top = z_max
    z_bottom = -z_max
    dz = (z_top - z_bottom) / (nz_global - 1)  # Grid spacing in Z

    # (nz+2, nr+2) grid points with one-layer ghost cells on each side
    # r1 = np.array([r_left + dr * (i - 1) for i in range(nr + 2)])
    # z1 = np.array([z_bottom + dz * (j - 1) for j in range(nz + 2)])
    # --- Local R grid (size: nr_local+2) [ir_start -1, ..., ir_start + nr_local] ---
    r1 = np.array([r_left + dr * (i - 1) for i in range(ir_start, ir_start + nr_local + 2)])
    # --- Local Z grid (size: nz_local+2) [iz_start -1, ..., iz_start + nz_local] ---
    z1 = np.array([z_bottom + dz * (j - 1) for j in range(iz_start, iz_start + nz_local + 2)])
    RR, ZZ = np.meshgrid(r1, z1, indexing='xy')  # Shape: (Z,R)
    
    # --- Boundary mask ---
    def inside_mask(rr, zz):
        """
        Boolean mask defining the plasma region:
        True inside the plasma boundary, False outside.
        """
        t1 = ((1 - d_sol) * rr**2 + d_sol * r_0**2) * zz**2 / E_sol**2
        t2 = 0.25 * (rr**2 - r_0**2)**2
        return t1 + t2 < aa**2 * r_0**2

    mask = inside_mask(RR, ZZ)

    # --- Right-hand side and initial condition ---
    rhs = np.zeros_like(RR)
    rhs[mask] = (
        2.0 * RR[mask]**2 * psi_s / (aa**2 * r_0**2) * (1.0 + (1.0 - d_sol) / E_sol**2)
        + 2.0 * d_sol * psi_s / (aa**2 * E_sol**2)
    )
    psi = np.zeros_like(rhs)  # Initial guess for the poloidal flux

    # --- Discrete Grad–Shafranov operator ---
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

    def apply_operator(psi):
        exchange_ghost_cells(psi)
        out = np.zeros_like(psi)
        out[1:-1, 1:-1] = (
            (1.0 / dr**2 - 0.5 / (dr * RR[1:-1, 1:-1])) * psi[1:-1, 2:] +   # psi[iz,ir+1] term
            (1.0 / dr**2 + 0.5 / (dr * RR[1:-1, 1:-1])) * psi[1:-1, :-2] +  # psi[iz,ir-1] term
            1.0 / dz**2 * (psi[2:, 1:-1] + psi[:-2, 1:-1]) - # psi[iz+1,ir] and psi[iz-1,ir] terms
            2.0 * (1.0 / dr**2 + 1.0 / dz**2) * psi[1:-1, 1:-1]             # psi[iz,ir] term
        )
        return out * mask

    def mpi_sum(array):
        local_sum = np.sum(array[1:-1,1:-1]) # Avoiding ghost cells
        global_sum = comm.allreduce(local_sum, op=MPI.SUM)
        return global_sum

    def cg_step(psi, r, p, rs_old):
        """
        Conjugate gradient algorithm
        """
        Ap = apply_operator(p)
        alpha = rs_old / mpi_sum(p * Ap)
        psi = psi + alpha * p
        r_new = r - alpha * Ap
        rs_new = mpi_sum(r_new * r_new)
        beta = rs_new / rs_old
        p_new = r_new + beta * p
        return psi, r_new, p_new, rs_new

    iterative_solver = cg_step

    # --- Main iteration loop ---
    r = rhs - apply_operator(psi)         # Initial residual
    p = r.copy()                          # Initial search direction (for CG/SD)
    rs_new = mpi_sum(r * r)               # Squared residual norm
    rhs_rms = np.sqrt(mpi_sum(rhs**2))    # Root mean square or RHS
    rel_error = np.sqrt(rs_new) / rhs_rms # Relative residual
    errors = [rel_error]                  # Convergence history
    psi_all = [psi.copy()]                # Stored solutions

    for i in range(1,niter):
        psi, r, p, rs_new = iterative_solver(psi, r, p, rs_new)
        rel_error = np.sqrt(rs_new) / rhs_rms
        if i % nskip == 0:
            errors.append(rel_error)
            psi_all.append(psi.copy())
        if rel_error < eps:
            print(f"Converged at iter = {i}, error = {rel_error}")
            break
    else:
        print(f"Did not converge within {niter} iterations. Final error = {rel_error}")

    return psi_all, errors, RR, ZZ

# --- Run solver ---
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
