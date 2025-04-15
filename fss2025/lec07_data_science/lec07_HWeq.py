#!/usr/bin/env python
# coding: utf-8

# import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.numpy.fft import fft2, ifft2, fftfreq
import os
from time import time as timer

def hasegawa_wakatani_simulation(flag_adiabaticity="constant"):
    # --- Parameters from param.namelist ---
    pi = jnp.pi
    Nx, Ny = 108, 96
    Lx, Ly = pi * 10, pi * 10
    dx = 2 * Lx / Nx
    dy = 2 * Ly / Ny

    x = jnp.linspace(-Lx, Lx, Nx, endpoint=False, dtype=jnp.float64)
    y = jnp.linspace(-Ly, Ly, Ny, endpoint=False, dtype=jnp.float64)
    kx = fftfreq(Nx, d=dx/(2*pi))
    ky = fftfreq(Ny, d=dy/(2*pi))
    KX, KY = jnp.meshgrid(kx, ky, indexing='xy')
    ksq = KX**2 + KY**2
    ksq = ksq.at[0, 0].set(1.0)
    poisson_fac = jnp.where(ksq != 0.0, -1.0 / ksq, 0.0)
    kquad = ksq**2

    kx_max = jnp.max(jnp.abs(kx))
    ky_max = jnp.max(jnp.abs(ky))
    dealias_mask =  (jnp.abs(KX) < (2/3)*kx_max) &  (jnp.abs(KY) < (2/3)*ky_max)

    @jax.jit
    def dealias(f_hat):
        return f_hat * dealias_mask

    dt = 0.005
    dt_out = 0.1
    nt = int(150.0 / dt)
    dns_init_ampl = 1e-4
    eta = 5.0
    ca = 3.0
    nu = 0.02
    nskip = int(dt_out / dt)
    nsave = nt // nskip

    if flag_adiabaticity == "constant":
        ca_factor = ca
    elif flag_adiabaticity == "modified":
        ca_mask = jnp.ones_like(KY, dtype=jnp.float64)
        ca_mask = ca_mask.at[0, :].set(0.0)
        ca_factor = ca * ca_mask
    else:
        raise ValueError(f"Unknown flag_adiabaticity: {flag_adiabaticity}")

    @jax.jit
    def poisson_solver(omg):
        omg_hat = dealias(fft2(omg)) / (Nx*Ny)
        phi_hat = poisson_fac * omg_hat
        phi_hat = phi_hat.at[0, 0].set(0.0)
        return jnp.real(ifft2(phi_hat)) * (Nx*Ny)

    @jax.jit
    def calc_poisson_bracket(f, g):
        f_hat = dealias(fft2(f)) / (Nx*Ny)
        g_hat = dealias(fft2(g)) / (Nx*Ny)
        df_dx = jnp.real(ifft2(1j * KX * f_hat)) * (Nx*Ny)
        df_dy = jnp.real(ifft2(1j * KY * f_hat)) * (Nx*Ny)
        dg_dx = jnp.real(ifft2(1j * KX * g_hat)) * (Nx*Ny)
        dg_dy = jnp.real(ifft2(1j * KY * g_hat)) * (Nx*Ny)
        nonlinear_term = df_dx * dg_dy - df_dy * dg_dx
        return jnp.real(ifft2(dealias(fft2(nonlinear_term))))

    @jax.jit
    def time_derivatives(dns, omg, phi):
        dphi_dy = jnp.real(ifft2(1j * KY * dealias(fft2(phi))))
        pb_phidns = calc_poisson_bracket(phi, dns)
        pb_phiomg = calc_poisson_bracket(phi, omg)

        dns_phi_term = ca_factor * (dns - phi) # Depending on flag_adiabaticity

        rhs_dns = -pb_phidns - eta * dphi_dy - dns_phi_term + nu * jnp.real(ifft2(- kquad * dealias(fft2(dns))))
        rhs_omg = -pb_phiomg - dns_phi_term + nu * jnp.real(ifft2(- kquad * dealias(fft2(omg))))
        return rhs_dns, rhs_omg

    @jax.jit
    def rkg4_step(dns, omg, phi, qq_dns, qq_omg):
        df1, df2 = time_derivatives(dns, omg, phi)
        k1_dns = dt * df1; r1_dns = 0.5 * (k1_dns - 2 * qq_dns); s_dns = dns.copy()
        k1_omg = dt * df2; r1_omg = 0.5 * (k1_omg - 2 * qq_omg); s_omg = omg.copy()

        dns = s_dns + r1_dns; omg = s_omg + r1_omg
        qq_dns += 3 * (dns - s_dns) - 0.5 * k1_dns
        qq_omg += 3 * (omg - s_omg) - 0.5 * k1_omg
        phi = poisson_solver(omg)

        df1, df2 = time_derivatives(dns, omg, phi)
        k2_dns = dt * df1; r2_dns = (1 - jnp.sqrt(0.5)) * (k2_dns - qq_dns); s_dns = dns.copy()
        k2_omg = dt * df2; r2_omg = (1 - jnp.sqrt(0.5)) * (k2_omg - qq_omg); s_omg = omg.copy()

        dns = s_dns + r2_dns; omg = s_omg + r2_omg
        qq_dns += 3 * (dns - s_dns) - (1 - jnp.sqrt(0.5)) * k2_dns
        qq_omg += 3 * (omg - s_omg) - (1 - jnp.sqrt(0.5)) * k2_omg
        phi = poisson_solver(omg)

        df1, df2 = time_derivatives(dns, omg, phi)
        k3_dns = dt * df1; r3_dns = (1 + jnp.sqrt(0.5)) * (k3_dns - qq_dns); s_dns = dns.copy()
        k3_omg = dt * df2; r3_omg = (1 + jnp.sqrt(0.5)) * (k3_omg - qq_omg); s_omg = omg.copy()

        dns = s_dns + r3_dns; omg = s_omg + r3_omg
        qq_dns += 3 * (dns - s_dns) - (1 + jnp.sqrt(0.5)) * k3_dns
        qq_omg += 3 * (omg - s_omg) - (1 + jnp.sqrt(0.5)) * k3_omg
        phi = poisson_solver(omg)

        df1, df2 = time_derivatives(dns, omg, phi)
        k4_dns = dt * df1; r4_dns = (1/6) * (k4_dns - 2 * qq_dns); s_dns = dns.copy()
        k4_omg = dt * df2; r4_omg = (1/6) * (k4_omg - 2 * qq_omg); s_omg = omg.copy()

        dns = s_dns + r4_dns; omg = s_omg + r4_omg
        qq_dns += 3 * (dns - s_dns) - 0.5 * k4_dns
        qq_omg += 3 * (omg - s_omg) - 0.5 * k4_omg
        phi = poisson_solver(omg)

        return dns, omg, phi, qq_dns, qq_omg

    # --- Initial values ---
    random_key = jax.random.PRNGKey(0)
    rand_phase = jax.random.uniform(random_key, shape=(Ny, Nx), minval=0.0, maxval=2*pi, dtype=jnp.float64)
    f_hat = jnp.exp(1j * rand_phase) / (1.0 + ksq)
    f_hat = dealias(f_hat)
    t = 0.0
    dns = jnp.real(ifft2(f_hat * dns_init_ampl)) * (Nx*Ny)
    omg =  -jnp.real(ifft2(dealias(ksq * fft2(dns))))
    phi = poisson_solver(omg)

    qq_dns = jnp.zeros_like(dns)
    qq_omg = jnp.zeros_like(omg)

    # --- Time integration ---
    # t_all, dns_all, phi_all = [], [], []
    # for it in range(nt):
    #     if it % nskip == 0:
    #         t_all.append(it * dt)
    #         dns_all.append(dns.copy())
    #         phi_all.append(phi.copy())
    #     dns, omg, phi, qq_dns, qq_omg = rkg4_step(dns, omg, phi, qq_dns, qq_omg)
    # t_all = jnp.array(t_all)
    # dns_all = jnp.array(dns_all)
    # phi_all = jnp.array(phi_all)
    # dns_da = xr.DataArray(dns_all, coords={'t': t_all, 'y': y, 'x': x}, dims=('t', 'y', 'x'))
    # phi_da = xr.DataArray(phi_all, coords={'t': t_all, 'y': y, 'x': x}, dims=('t', 'y', 'x'))
    # ds = xr.Dataset({'dns': dns_da, 'phi': phi_da})
    # ds.to_netcdf("hasegawa_wakatani.nc")
    # return

    def output_netcdf(t, dns, phi, x, y, netcdf_path="hasegawa_wakatani.nc", first=False):
        """Save output as NetCDF format"""
        import netCDF4 as nc

        if first: # Create a new NetCDF
            if os.path.exists(netcdf_path):
                os.remove(netcdf_path)

            with nc.Dataset(netcdf_path, "w") as ds:
                ds.createDimension("t", None) # Unlimited dimension
                ds.createDimension("y", len(y))
                ds.createDimension("x", len(x))

                ds.createVariable("t", "f8", ("t",))
                ds.createVariable("y", "f8", ("y",))
                ds.createVariable("x", "f8", ("x",))
                ds.createVariable("dns", "f8", ("t", "y", "x"))
                ds.createVariable("phi", "f8", ("t", "y", "x"))

                ds.variables["x"][:] = jnp.asarray(x)
                ds.variables["y"][:] = jnp.asarray(y)
                ds.variables["t"][0] = float(t)
                ds.variables["dns"][0, :, :] = jnp.asarray(dns)
                ds.variables["phi"][0, :, :] = jnp.asarray(phi)
        else:
            with nc.Dataset(netcdf_path, "a") as ds:
                it = ds.dimensions["t"].size
                ds.variables["t"][it] = float(t)
                ds.variables["dns"][it, :, :] = jnp.asarray(dns)
                ds.variables["phi"][it, :, :] = jnp.asarray(phi)
        return

    # def output_zarr(t, dns, phi, x, y, zarr_path="hasegawa_wakatani.zarr", first=False):
    #     """Save output as Zarr format: Similar to NetCDF, scalable for Cloud system"""
    # import xarray as xr
    # import zarr
    #     t_np = jnp.array([t])
    #     step_ds = xr.Dataset({"dns": (("t", "y", "x"), dns[jnp.newaxis,:,:]),
    #                           "phi": (("t", "y", "x"), phi[jnp.newaxis,:,:]),},
    #                           coords={"t": jnp.array([t]), "y": y, "x": x,},)
    #     if first: # Create a new zarr
    #         step_ds.to_zarr(zarr_path, mode="w")
    #     else: # Append to the existing zarr
    #         step_ds.to_zarr(zarr_path, mode="a", append_dim="t")
    #     return

    @jax.jit
    def inner_nskiploop_jit(carry):
        def body_fun(i, carry):
            t, dns, omg, phi, qq_dns, qq_omg = carry
            dns, omg, phi, qq_dns, qq_omg = rkg4_step(dns, omg, phi, qq_dns, qq_omg)
            t += dt
            return (t, dns, omg, phi, qq_dns, qq_omg)
        return jax.lax.fori_loop(0, nskip, body_fun=body_fun, init_val=carry)
    for it in range(nsave):
        output_netcdf(t, dns, phi, x, y, first=(it == 0))
        t, dns, omg, phi, qq_dns, qq_omg = inner_nskiploop_jit((t, dns, omg, phi, qq_dns, qq_omg))
    return

if __name__ == "__main__":
    # --- Run the simulation ---
    elt0 = timer()
    print("Start 1D Hasegawa-Wakatani simulation.")
    hasegawa_wakatani_simulation()
    print("Simulation complete.")
    elt1 = timer(); print("Elapsed time [sec] :", elt1 - elt0)


# from google.colab import drive
# drive.mount('/content/drive')
# get_ipython().system('jupyter nbconvert --to python "/content/drive/MyDrive/Colab Notebooks/lec07_HWeq.ipynb"')




