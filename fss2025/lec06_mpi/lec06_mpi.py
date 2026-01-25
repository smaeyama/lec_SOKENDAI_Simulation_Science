#!/usr/bin/env python
# coding: utf-8

# In[1]:


# --- Install required packages ---

get_ipython().system('apt install -y libopenmpi-dev openmpi-bin')
get_ipython().system('pip install mpi4py')


# In[2]:


# --- Download lec06_mpi_grad_shafranov.py ---

get_ipython().system('wget https://raw.githubusercontent.com/smaeyama/lec_SOKENDAI_Simulation_Science/main/fss2025/lec06_mpi/lec06_mpi_grad_shafranov.py')


# In[3]:


# --- Run MPI application ---
# !mpirun -np 6 python3 "lec06_mpi_grad_shafranov.py"

# --- Options for Google Colab ---
#  Generally, the option --allow-run-as-root is not recommended. Root privileges have less impact in Google Colab, because it is a temporary workspace.
get_ipython().system('mpirun --allow-run-as-root --oversubscribe -np 6 python3 "lec06_mpi_grad_shafranov.py"')


# In[4]:


import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

file_list = sorted(glob.glob("psi.*.nc"))         # Collect all MPI rank output files

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()
cmap = plt.get_cmap("tab10")                      # Discrete colormap for rank coloring
legend_elements = []

for i, fname in enumerate(file_list):
    ds = xr.open_dataset(fname)                   # Load local grid data of each MPI rank
    r = ds["r"]                                   # Local R-coordinates
    z = ds["z"]                                   # Local Z-coordinates
    color = cmap(i % cmap.N)                      # Assign a unique color per rank
    ax.pcolormesh(r, z, np.zeros_like(ds["psi"])[:-1, :-1],  # Draw mesh edges
        shading="auto", edgecolor=color, facecolor='none', linewidth=0.2)
    legend_elements.append(Line2D([0], [0], color=color, lw=2, label=f"Rank {i}"))
                                                  # Legend entry for each rank
    ds.close()

ax.set_xlabel(r'Major radius $R$')
ax.set_ylabel(r'Height $Z$')
ax.set_aspect('equal', adjustable='box')
ax.set_title("MPI Rank Grid")
ax.legend(handles=legend_elements, title="MPI Rank", loc="upper right", fontsize="small", ncol=2)
fig.tight_layout()
plt.show()


# In[5]:


ds = xr.open_mfdataset("psi.*.nc")  # Open and merge all per-rank NetCDF files into one dataset
print(ds)                           # Display merged dataset structure and coordinates
r = ds["r"]                         # Global R-coordinate array
z = ds["z"]                         # Global Z-coordinate array
psi = ds["psi"]                     # Global psi array

fig = plt.figure(figsize=(8.2, 6))
ax = fig.add_subplot()
quad = ax.pcolormesh(r, z, psi, shading="auto")
ax.set_xlabel(r'Major radius $R$')
ax.set_ylabel(r'Height $Z$')
ax.set_aspect('equal', adjustable='box')
ax.set_title(r"$\psi(R,Z)$")
fig.colorbar(quad)
plt.show()

ds.close()


# In[5]:




