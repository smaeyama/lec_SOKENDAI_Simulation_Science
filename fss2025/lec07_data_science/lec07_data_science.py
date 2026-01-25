#!/usr/bin/env python
# coding: utf-8

# In[1]:


#--- Install required package ---

get_ipython().system('pip install netCDF4')


# In[2]:


# --- Download lec07_HWeq.py ---

get_ipython().system('wget https://raw.githubusercontent.com/smaeyama/lec_SOKENDAI_Simulation_Science/main/fss2025/lec07_data_science/lec07_HWeq.py')


# ## 1. Run Hasegawa-Wakatani plasma turbulence simulation

# In[3]:


get_ipython().system('python3 "lec07_HWeq.py"')


# In[4]:


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import Video
ds = xr.open_dataset("hasegawa_wakatani.nc")
print(ds)
t = ds["t"].values
x = ds["x"].values
y = ds["y"].values
dns = ds["dns"].values
phi = ds["phi"].values
ds.close()
print(phi.shape)


# In[5]:


# --- Animation ---
fig, ax = plt.subplots()
cax = ax.pcolormesh(x, y, dns[0], shading='auto', cmap='turbo')
cbar = fig.colorbar(cax)
ax.set_title(f"t = {t[0]:.2f}")

def update(i):
    cax.set_array(dns[i].ravel())
    vmax = np.max([dns[i].max(), -dns[i].min()])
    cax.set_clim(-vmax, vmax)
    ax.set_title(f"t = {t[i]:.2f}")
    return cax,

ani = FuncAnimation(fig, update, frames=range(0,len(t),1), interval=50)
ani.save("hasegawa_wakatani.mp4", writer="ffmpeg", dpi=100)
plt.close()

# --- Display in notebook ---
from IPython.display import Video
Video("hasegawa_wakatani.mp4", embed=True)


# In[6]:


nx = len(x)
ny = len(y)
dx = x[1] - x[0]
dy = y[1] - y[0]
kx = 2*np.pi * np.fft.fftfreq(nx, dx)  # kx = 0, kx_min, 2*kx_min, ..., -2*kx_min, -kx_min
ky = 2*np.pi * np.fft.fftfreq(ny, dy)  # ky = 0, ky_min, 2*ky_min, ..., -2*ky_min, -ky_min
#print(kx)
print("Minimum wavenumber kx_min = ", kx[1])#, "(= 2pi/Lx =", 2.0*np.pi/(dx*nx),")")
print("Nyquist wavenumber kx_max = ", np.abs(kx).max())#, "(= pi/dx =", np.pi/dx,")")
#print(ky)
print("Minimum wavenumber ky_min = ", ky[1])#, "(= 2pi/Ly =", 2.0*np.pi/(dy*ny),")")
print("Nyquist wavenumber ky_max = ", np.abs(ky).max())#, "(= pi/dy =", np.pi/dy,")")

# Forward FFT with normalization (for our definition)
dnsk = np.fft.fft2(dns, axes=(-2,-1)) / (nx*ny)
phik = np.fft.fft2(phi, axes=(-2,-1)) / (nx*ny)
kx2, ky2 = np.meshgrid(kx, ky, indexing="xy")
ksq = kx2**2 + ky2**2


# In[7]:


ca = 3.0           # adiabaticity parameter
eta = 5.0          # Density gradient
nu = 0.02          # Viscosity
flag_adiabaticity = "constant"


### Calculate energy balance equation ###
entrk = 0.5 * np.abs(dnsk)**2                         # Entropy
eenek = 0.5 * ksq.reshape(1,ny,nx) * np.abs(phik)**2  # Electric field energy
entr = np.sum(entrk,axis = (1,2))
eene = np.sum(eenek,axis = (1,2))

enefluxk = eta * (1j * ky[None,:,None] * dnsk * phik.conjugate()).real  # Particle flux
eneresik = - ca * np.abs(phik - dnsk)**2                                # Dissipation by resistivity
enevisck = - nu * (ksq[None,:,:]**2 * np.abs(dnsk)**2
                   + ksq[None,:,:]**3 * np.abs(phik)**2)                # Dissipation by viscosity
eneflux = np.sum(enefluxk,axis = (1,2))
eneresi = np.sum(eneresik,axis = (1,2))
enevisc = np.sum(enevisck,axis = (1,2))


# d/dt (Entropy + Electric field energy)
#   = Particle flux + Dissipation by resistivity + Dissipation by viscosity
# evaluated at (t+dt + t)/2
dt = t[1] - t[0]
t_mid = (t[1:] + t[:-1]) / 2

dentr = np.diff(entr)
deene = np.diff(eene)
dedt = (dentr + deene) / dt

eneflux_mid = (eneflux[1:] + eneflux[:-1])/2
eneresi_mid = (eneresi[1:] + eneresi[:-1])/2
enevisc_mid = (enevisc[1:] + enevisc[:-1])/2

# Calculate energy balance error = LHS - RHS
eneerror = dedt - (eneflux_mid + eneresi_mid + enevisc_mid)

plt.plot(t_mid,dedt,label=r"$d(E_n+E_E)/dt$")
plt.plot(t_mid,eneflux_mid,label=r"Particle flux $\Gamma_n$")
plt.plot(t_mid,eneresi_mid,label=r"Resistivity $D_a$")
plt.plot(t_mid,enevisc_mid,label=r"Viscosity $D_E$")
plt.plot(t_mid,eneerror,label="Error")
plt.xlabel("Time t")
plt.ylabel("Energy balance")
plt.legend()
plt.show()
#plt.savefig("energy_balance.png")


# In[8]:


### Calculate enstrophy balance equation ###
enstk = 0.5 * np.abs(ksq[None,:,:] * phik + dnsk)**2  # Enstrophy
enst = np.sum(enstk,axis = (1,2))

enstfluxk = eta * (1j * ky[None,:,None] * (dnsk * phik.conjugate())).real     # Particle flux
enstvisck = - nu * ksq[None,:,:]**2 * np.abs(ksq[None,:,:] * phik + dnsk)**2  # Dissipation by viscosity
enstflux = np.sum(enstfluxk,axis = (1,2))
enstvisc = np.sum(enstvisck,axis = (1,2))

# d/dt (Enstrophy)　= Particle flux + Dissipation by viscosity
# evaluated at (t+dt + t)/2
denst = np.diff(enst)
dudt = denst/dt

enstflux_mid = (enstflux[1:] + enstflux[:-1])/2
enstvisc_mid = (enstvisc[1:] + enstvisc[:-1])/2

# Calculate enstrophy balance error = LHS - RHS
ensterror = dudt - (enstflux_mid + enstvisc_mid)

plt.plot(t_mid,dudt,label=r"$dU/dt$")
plt.plot(t_mid,enstflux_mid,label=r"Particle flux $\Gamma_n$")
plt.plot(t_mid,enstvisc_mid,label=r"Viscosity $D_E$")
plt.plot(t_mid,ensterror,label="Error")
plt.xlabel("Time t")
plt.ylabel("Enstrophy balance")
plt.legend()
plt.show()
#plt.savefig("enstrophy_balance.png")


# ## 2. Data analysis: Spectrogram

# In[9]:


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
ds = xr.open_dataset("hasegawa_wakatani.nc")
print(ds)
t = ds["t"].values
x = ds["x"].values
y = ds["y"].values
dns = ds["dns"].values
phi = ds["phi"].values
ds.close()
print(phi.shape)

# --- Fourier transform ---
#  We physically assume the following equations:
#    Backward Fourier transform:
#      phi(t,y,x) = \sum_kx \sum_ky \sum_omega phi(omega,ky,kx) * exp(i*kx*x + i*ky*y - i*omega*t)
#    Forward Fourier transform:
#      phi(omega,ky,kx) = 1/(Lx*Ly*T) \int_dx \int_dy \int_dt phi(t,y,x) * exp(- i*kx*x - i*ky*y + i*omega*t)
#  Be careful with signature and definition of numpy.fft.fft and numpy.fft.ifft

# Wavenumber kx, ky
nx = len(x)
ny = len(y)
dx = x[1]-x[0]
dy = y[1]-y[0]
kx = 2*np.pi * np.fft.fftfreq(nx, dx)        # kx = 0, kx_min, 2*kx_min, ..., -2*kx_min, -kx_min
ky = 2*np.pi * np.fft.fftfreq(ny, dy)        # ky = 0, ky_min, 2*ky_min, ..., -2*ky_min, -ky_min

# Forward FFT in x, y (- sign with normalization)
phi_tkykx = np.fft.fft2(phi, axes=(1,2)) / (nx*ny)


def short_time_Fourier_transform(itsta, nwindow, t, phi_tkykx):
    # Frequency omega under a window function
    window_function = np.hanning(nwindow)
    phi_tkykx_window = phi_tkykx[itsta:itsta+nwindow,:,:] * window_function[:,None,None]

    dt = t[1]-t[0]
    omega = 2*np.pi * np.fft.fftfreq(nwindow, dt) # omega = 0, omega_min, ..., -omega_min

    # Forward FFT in t (+ sign with normalization)
    phi_omegakykx = np.fft.ifft(phi_tkykx_window, axis=0)

    energy_spectrum = np.sum(np.abs(phi_omegakykx)**2, axis=(1,2))
    return t[itsta+nwindow//2], omega, energy_spectrum

nt=len(t)
dt = t[1] - t[0]
itsta = nt//3
nwindow = nt//2
t_window, omega, energy_spectrum = short_time_Fourier_transform(itsta, nwindow, t, phi_tkykx)
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(omega[0:nwindow//2],energy_spectrum[0:nwindow//2])
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Frequency $\omega$")
ax.set_ylabel(r"Energy spectrum")
plt.show()


# In[10]:


# --- Spectrogram
#   Apply short-time Fourier transform for various time slices
nt=len(t)
nwindow = int(10.0 / dt)  # Window length in time
t_spectrogram = []
spectrogram = []
for itsta in range(0, nt-nwindow, nwindow//10):  # Time slide (90% overlap)
    # Short-time Fourier transform around time window
    t_window, omega, energy_spectrum = short_time_Fourier_transform(itsta, nwindow, t, phi_tkykx)
    t_spectrogram.append(t_window)       # Center time of the window
    spectrogram.append(energy_spectrum)  # Frequency spectrum at that time

spectrogram = np.array(spectrogram)   # (ntime_window, nomega)
t_spectrogram = np.array(t_spectrogram)

import matplotlib.colors as colors
omega_max = 20.0                     # Maximum frequency to visualize
nmax = np.argmax(omega > omega_max)  # Index for omega cutoff
vmax = np.max(spectrogram)
plt.figure(figsize=(10, 6))
plt.pcolormesh(
    omega[0:nmax],                   # Frequency axis
    t_spectrogram,                   # Time axis
    spectrogram[:, 0:nmax],          # Energy spectrum
    norm=colors.LogNorm(vmin=vmax*1e-6, vmax=vmax),
    shading='auto'
)
plt.xlabel(r"Frequency $\omega$")
plt.ylabel("Time")
plt.colorbar(label="Energy spectrum")
plt.title(r"$t$-$\omega$ Spectrogram")
plt.show()


# ## 3. Data analysis: Singular value decomposition

# In[11]:


import xarray as xr
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
ds = xr.open_dataset("hasegawa_wakatani.nc")
print(ds)
t = ds["t"].values
x = ds["x"].values
y = ds["y"].values
dns = ds["dns"].values
phi = ds["phi"].values
ds.close()
print(phi.shape)

itsta = 500
wk_t = t[itsta:]
data = phi[itsta:]
ntime, ny, nx = data.shape
data = data.reshape(ntime,ny*nx)

# SVD (Singular Value Decomposition): A = U @ S @ V_H
U, Sigma, V_H = sp.linalg.svd(data,full_matrices=False)
print(data.shape,U.shape,Sigma.shape,V_H.shape)


# In[12]:


plt.plot(Sigma,".-")                    # Plot singular values vs mode index
plt.xscale("log")                       # Log scale for mode number (wide dynamic range)
plt.yscale("log")                       # Log scale for singular values (power-law visibility)
plt.xlabel("Mode number")               # SVD mode index r
plt.ylabel(r"Singular value $\sigma$")  # Mode amplitude (sqrt of energy)
plt.ylim(1e-3*Sigma.max(),Sigma.max())
plt.show()


# In[13]:


def plot_mode_structure(idx, t, y, x, U, Sigma, V_H):
    SVD_amplitude = U[:,idx]*Sigma[idx] # Time evolution of amplitude
    SVD_mode = V_H[idx,:]               # Spatial mode structures, which are orthonormal.
    fig = plt.figure(figsize=(6,3))
    fig.suptitle(f"{idx}-th SVD mode structure and amplitude: "+r"$\sigma =$"+f"{Sigma[idx]:.2f}")
    ax = fig.add_subplot(121)
    ax.pcolormesh(x,y,SVD_mode.reshape(len(y),len(x)),cmap="jet")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax = fig.add_subplot(122)
    ax.plot(t,SVD_amplitude)
    ax.set_xlabel("t")
    ax.set_ylabel("SVD mode amplitude")
    fig.tight_layout()
    plt.show()

for idx in range(4):
    plot_mode_structure(idx, wk_t, y, x, U, Sigma, V_H)


# In[14]:


# --- Truncation for low-rank approximation ---
truncation=50
print("Truncated rank = ", truncation)

U_trunc=U[:,:truncation]       # Temporal variations
Sigma_trunc=Sigma[:truncation] # Singular values
V_H_trunc = V_H[:truncation,:] # Spactial modes structures, flattened (y,x)

SVD_amplitude_trunc = U_trunc*Sigma_trunc[np.newaxis,:] # Time evolution of amplitude
SVD_mode_trunc = V_H_trunc                              # Spatial mode structures, which are orthonormal.
print(SVD_amplitude_trunc.shape,SVD_mode_trunc.shape)
print("Raw data size = (ntime)x(ny)x(nx) = ({},{},{}) = {:,}".format(ntime,ny,nx,ntime*ny*nx))
print("Truncated size = (ntime)x(truncated_rank) + (truncated_rank)x(ny*nx) = ({},{})+({},{}) = {:,}".format(ntime,truncation,truncation,ny*nx,ntime*truncation+truncation*(ny*nx)))

reconstructed = SVD_amplitude_trunc @ SVD_mode_trunc
# print(reconstructed.shape, data.shape)

phi_rec = reconstructed.reshape(ntime, ny, nx).real   # Reconstructed field (assumes real-valued phi)
phi_data = data.reshape(ntime, ny, nx)                # Original snapshots (same window)
# print(phi_rec.shape, phi_data.shape)


# In[15]:


# --- Animation ---
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

x2, y2 = np.meshgrid(x,y)
fig = plt.figure(figsize=(8,4))
# - ax1: original phi -
ax1 = fig.add_subplot(121)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
title=fig.suptitle(r"$\phi(x,y)$ Time = {:5.2f}".format(wk_t[0]))
quad1=ax1.pcolormesh(x2, y2, phi_data[0,:,:],shading="auto",cmap="jet")
title1=ax1.set_title("Raw data")
vmax=np.max(np.abs(phi_data[0,:-1,:-1]))
quad1.set_clim(-vmax,vmax)
cbar1=fig.colorbar(quad1,shrink=1.0,aspect=10)
# - ax2: reconstructed phi -
ax2 = fig.add_subplot(122)
ax2.set_xlabel("x")
ax2.set_ylabel("y")
quad2=ax2.pcolormesh(x2, y2, phi_rec[0,:,:],shading="auto",cmap="jet")
title2=ax2.set_title("Reconstructed data")
quad2.set_clim(-vmax,vmax)
cbar2=fig.colorbar(quad2,shrink=1.0,aspect=10)
fig.tight_layout()

def update_quad(i):
    title.set_text(r"$\phi(x,y)$ Time = {:5.2f}".format(wk_t[i]))
    vmax=np.max(np.abs(phi_data[i,:,:]))
    quad1.set_array(phi_data[i,:,:].ravel())
    quad1.set_clim(-vmax,vmax)
    quad2.set_array(phi_rec[i,:,:].ravel())
    quad2.set_clim(-vmax,vmax)
    return quad1, quad2

ani = FuncAnimation(fig, update_quad,
                    frames=range(0,len(wk_t)//2,2), interval=100)
ani.save("compare_svd_compression.mp4", writer="ffmpeg", dpi=100)
plt.close()

# --- Display in notebook ---
from IPython.display import Video
Video("compare_svd_compression.mp4", embed=True)


# In[15]:




