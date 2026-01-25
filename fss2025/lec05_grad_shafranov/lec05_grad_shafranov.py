#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import Video
from time import time as timer
from numba import njit

# --- Numerical parameters ---
niter = 1000   # Iteration limit
nskip = 1      # Interval for storing history
eps = 0.01     # Convergence tolerance (relative residual)
nr = 32        # Grid points in R
nz = 32        # Grid points in Z

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
def solve_grad_shafranov(flag_solver="sor"):
    """
    Grad-Shafranov solver

    Parameters
    ----------
    flag_solver : str
        Solver type: "jacobi", "gauss_seidel", "sor", "sd", "cg"
    """
    assert flag_solver in ["jacobi", "gauss_seidel", "sor", "sd", "cg"], f"Unknown solver: {flag_solver}"

    # --- Rectangular grid in (Z,R) ---
    r_left = r_0 * np.sqrt(1.0 - 2.0 * aa / r_0)
    r_right = r_0 * np.sqrt(1.0 + 2.0 * aa / r_0)
    dr = (r_right - r_left) / (nr - 1)  # Grid spacing in R
    z_top = z_max
    z_bottom = -z_max
    dz = (z_top - z_bottom) / (nz - 1)  # Grid spacing in Z

    # (nz+2, nr+2) grid points with one-layer ghost cells on each side
    r1 = np.array([r_left + dr * (i - 1) for i in range(nr + 2)])
    z1 = np.array([z_bottom + dz * (j - 1) for j in range(nz + 2)])
    RR, ZZ = np.meshgrid(r1, z1, indexing='xy')  # Shape: (Z,R)

    # --- Boundary mask ---
    def inside_mask(RR, ZZ):
        """
        Boolean mask defining the plasma region:
        True inside the plasma boundary, False outside.
        """
        t1 = ((1 - d_sol) * RR**2 + d_sol * r_0**2) * ZZ**2 / E_sol**2
        t2 = 0.25 * (RR**2 - r_0**2)**2
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
    @njit
    def apply_operator(psi):
        out = np.zeros_like(psi)
        out[1:-1, 1:-1] = (
            (1.0 / dr**2 - 0.5 / (dr * RR[1:-1, 1:-1])) * psi[1:-1, 2:] +   # psi[iz,ir+1] term
            (1.0 / dr**2 + 0.5 / (dr * RR[1:-1, 1:-1])) * psi[1:-1, :-2] +  # psi[iz,ir-1] term
            1.0 / dz**2 * (psi[2:, 1:-1] + psi[:-2, 1:-1]) - # psi[iz+1,ir] and psi[iz-1,ir] terms
            2.0 * (1.0 / dr**2 + 1.0 / dz**2) * psi[1:-1, 1:-1]             # psi[iz,ir] term
        )
        return out * mask

    # --- Iterative solver kernels ---
    def sd_step(psi, r, p, rs_old):
        """
        Steepest descent algorithm
        """
        Ar = apply_operator(r)
        alpha = np.sum(r * r) / np.sum(r * Ar)
        psi = psi + alpha * r
        r_new = r - alpha * Ar
        return psi, r_new, r_new, np.sum(r_new * r_new)

    def cg_step(psi, r, p, rs_old):
        """
        Conjugate gradient algorithm
        """
        Ap = apply_operator(p)
        alpha = rs_old / np.sum(p * Ap)
        psi = psi + alpha * p
        r_new = r - alpha * Ap
        rs_new = np.sum(r_new * r_new)
        beta = rs_new / rs_old
        p_new = r_new + beta * p
        return psi, r_new, p_new, rs_new

    @njit
    def jacobi_step(psi, _, __, ___):
        """
        Jacobi algorithm
        """
        psi_new = psi.copy()
        D_inv = - dr**2 * dz**2 / (2.0 * (dr**2 + dz**2))  # Inverse diagonal coefficient
        coef1 = D_inv / dr**2
        coef2 = D_inv / dz**2
        for iz in range(1, nz+1):
            for ir in range(1, nr+1):
                if mask[iz, ir]:
                    psi_new[iz, ir] = (
                        D_inv * rhs[iz, ir]  # RHS term
                        - coef1 * (1 - 0.5 * dr / RR[iz, ir]) * psi[iz, ir+1]  # psi[iz,ir+1] term
                        - coef1 * (1 + 0.5 * dr / RR[iz, ir]) * psi[iz, ir-1]  # psi[iz,ir-1] term
                        - coef2 * (psi[iz+1, ir] + psi[iz-1, ir])  # psi[iz+1,ir] and psi[iz-1,ir] terms
                    )
        r_new = rhs - apply_operator(psi_new)
        return psi_new, r_new, None, np.sum(r_new * r_new)

    @njit
    def gauss_seidel_step(psi, _, __, ___):
        """
        Gauss-Seidel algorithm
        """
        D_inv = - dr**2 * dz**2 / (2.0 * (dr**2 + dz**2))  # Inverse diagonal coefficient
        coef1 = D_inv / dr**2
        coef2 = D_inv / dz**2
        for iz in range(1, nz+1):
            for ir in range(1, nr+1):
                if mask[iz, ir]:
                    psi[iz, ir] = (
                        D_inv * rhs[iz, ir]  # rhs term
                        - coef1 * (1 - 0.5 * dr / RR[iz, ir]) * psi[iz, ir+1]  # psi[iz,ir+1] term
                        - coef1 * (1 + 0.5 * dr / RR[iz, ir]) * psi[iz, ir-1]  # psi[iz,ir-1] term
                        - coef2 * (psi[iz+1, ir] + psi[iz-1, ir])  # psi[iz+1,ir] and psi[iz-1,ir] terms
                    )
        r_new = rhs - apply_operator(psi)
        return psi, r_new, None, np.sum(r_new * r_new)

    @njit
    def sor_step(psi, _, __, ___):
        """
        Successive over-relaxation (SOR) algorithm
        """
        D_inv = - dr**2 * dz**2 / (2.0 * (dr**2 + dz**2))   # Inverse diagonal coefficient
        coef1 = D_inv / dr**2
        coef2 = D_inv / dz**2
        alpha = 1.8  # Relaxation paramter
        for iz in range(1, nz+1):
            for ir in range(1, nr+1):
                if mask[iz, ir]:
                    update_val = (
                        D_inv * rhs[iz, ir]  # rhs term
                        - coef1 * (1 - 0.5 * dr / RR[iz, ir]) * psi[iz, ir+1]  # psi[iz,ir+1] term
                        - coef1 * (1 + 0.5 * dr / RR[iz, ir]) * psi[iz, ir-1]  # psi[iz,ir-1] term
                        - coef2 * (psi[iz+1, ir] + psi[iz-1, ir])  # psi[iz+1,ir] and psi[iz-1,ir] terms
                    )
                    psi[iz, ir] = (1 - alpha) * psi[iz, ir] + alpha * update_val
        r_new = rhs - apply_operator(psi)
        return psi, r_new, None, np.sum(r_new * r_new)

    solver_dict = {
        "jacobi": jacobi_step,
        "gauss_seidel": gauss_seidel_step,
        "sor": sor_step,
        "sd": sd_step,
        "cg": cg_step
    }
    iterative_solver = solver_dict[flag_solver]

    # --- Main iteration loop ---
    r = rhs - apply_operator(psi)      # Initial residual
    p = r.copy()                       # Initial search direction (for CG/SD)
    rs_new = np.sum(r * r)             # Squared residual norm
    rel_error = np.sqrt(rs_new) / np.sqrt(np.sum(rhs**2))  # Relative residual
    errors = [rel_error]               # Convergence history
    psi_all = [psi.copy()]             # Stored solutions

    for i in range(1,niter):
        psi, r, p, rs_new = iterative_solver(psi, r, p, rs_new)
        rel_error = np.sqrt(rs_new) / np.sqrt(np.sum(rhs**2))
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
tsta = timer()
psi_all, errors, RR, ZZ = solve_grad_shafranov(flag_solver="jacobi")
psi = psi_all[-1]
tend = timer()
print(f"Elapsed time: {tend-tsta:} [sec]")
print(len(psi_all), len(errors))


# In[21]:


def compute_boundary_curve(num_points=500):
    """
    Return rr_curve, zz_curve as parametric boundary based on Solov'ev parameters.
    The boundary corresponds to the last closed magnetic surface psi=0.
    """
    t = np.linspace(0, 2 * np.pi, num_points)
    eps_val = aa / r_0
    ss = 1.0
    rr_curve = r_0 * np.sqrt(1.0 + 2.0 * eps_val * ss * np.cos(t))
    zz_curve = (aa * E_sol * ss * np.sin(t)) / np.sqrt(1.0 + 2.0 * eps_val * ss * (1.0 - d_sol) * np.cos(t))
    return rr_curve, zz_curve

def solovev_solution(rr, zz):
    """
    Return Solov'ev analytic solution components: psi, pressure, B^2, rhs.
    All outputs are masked: outside the plasma domain, values are zero.
    """
    term = ((1 - d_sol) * rr**2 + d_sol * r_0**2) * zz**2 / E_sol**2 + 0.25 * (rr**2 - r_0**2)**2
    mask = term < aa**2 * r_0**2

    psi_sol = np.zeros_like(rr)
    pres_sol = np.zeros_like(rr)
    F2_sol = np.zeros_like(rr)
    rhs_sol = np.zeros_like(rr)

    # Analytic Solov'ev solution for psi inside the plasma
    psi_tmp = psi_s / (aa**2 * r_0**2) * (
        ((1 - d_sol) * rr[mask]**2 + d_sol * r_0**2) * zz[mask]**2 / E_sol**2 +
        0.25 * (rr[mask]**2 - r_0**2)**2
    ) - psi_s

    psi_sol[mask] = psi_tmp
    pres_sol[mask] = 4.0 * psi_s / (aa**2 * r_0**2) * (1 + (1 - d_sol) / E_sol**2) * (-psi_tmp)
    F2_sol[mask] = (r_0 * b0)**2 - 4.0 * d_sol * psi_s * (psi_tmp + psi_s) / (aa**2 * E_sol**2)
    rhs_sol[mask] = (
        2.0 * rr[mask]**2 * psi_s / (aa**2 * r_0**2) * (1 + (1 - d_sol) / E_sol**2) +
        2.0 * d_sol * psi_s / (aa**2 * E_sol**2)
    )
    return psi_sol, pres_sol, F2_sol, rhs_sol, mask

# --- Verification against analytic solution ---
rr_curve, zz_curve = compute_boundary_curve()
psi_sol, pres_sol, F2_sol, rhs_sol, mask = solovev_solution(RR, ZZ)


# In[22]:


# --- Plot convergence history ---
plt.figure()
plt.semilogy(errors)
plt.xlabel('Iteration')
plt.ylabel(r'Relative residual error $r = b - Ax$')
plt.title('Residual history')
plt.grid(True)
plt.show()

# --- Plot with pcolormesh ---
vmax = np.max(np.abs(psi))
res_vmax = np.max(np.abs(psi - psi_sol))
# res_vmax = 0.1 * vmax
fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
pc1 = axes[0].pcolormesh(RR, ZZ, psi, shading='auto')
axes[0].plot(rr_curve, zz_curve, color='black', linewidth=0.5)
axes[0].set_title(r'Numerical Solution $\psi(R,Z)$')
fig.colorbar(pc1, ax=axes[0], shrink=0.85)
pc2 = axes[1].pcolormesh(RR, ZZ, psi_sol, shading='auto')
axes[1].plot(rr_curve, zz_curve, color='black', linewidth=0.5)
axes[1].set_title(r"Analytic Solov'ev Solution $\psi_\mathrm{sol}(R,Z)$")
fig.colorbar(pc2, ax=axes[1], shrink=0.85)
pc3 = axes[2].pcolormesh(RR, ZZ, psi - psi_sol, cmap='RdBu_r',
                         shading='auto', vmin=-res_vmax, vmax=+res_vmax)
axes[2].plot(rr_curve, zz_curve, color='black', linewidth=0.5)
axes[2].set_title(r'Error $\psi - \psi_\mathrm{sol}$')
fig.colorbar(pc3, ax=axes[2], shrink=0.85)
for ax in axes:
    ax.set_xlabel(r'Major radius $R$')
    ax.set_xlim(RR.min(), RR.max())
    ax.set_ylabel(r'Height $Z$')
    ax.set_ylim(ZZ.min(), ZZ.max())
    ax.set_aspect('equal', adjustable='box')
plt.suptitle("Comparison of Numerical and Analytic Solov'ev Solution (pcolormesh)", fontsize=14)
plt.show()


# In[23]:


# --- Animation for convergence ---
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

wire = None  # for clearing
zmin = np.min(psi_all[-1])
zmax = np.max(psi_all[-1])
def update(frame):
    global wire
    ax.clear()
    ax.set_title(f"Iteration {frame}")
    ax.set_xlabel("r")
    ax.set_ylabel("z")
    ax.set_zlabel("psi")
    ax.set_xlim(RR.min(), RR.max())
    ax.set_ylim(ZZ.min(), ZZ.max())
    ax.set_zlim(zmin,zmax)
    wire = ax.plot_surface(RR, ZZ, psi_all[frame], cmap="viridis", edgecolor='k', linewidth=0.3, antialiased=True)
    return wire,

ani = FuncAnimation(fig, update, frames=range(0, len(psi_all), 10), interval=100, blit=False)
ani.save("gs_wireframe_animation.mp4", writer="ffmpeg", fps=10)
plt.close(fig)

Video("gs_wireframe_animation.mp4", embed=True)


# In[4]:




