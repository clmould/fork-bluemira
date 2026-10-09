# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,title,-all
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% tags=["remove-cell"]
# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Example of using toroidal harmonics constraints in a
coil current optimisation.
"""

# %% [markdown]
# # Example of using Toroidal Harmonic Constraints in a Coil Current Optimisation
#
# This example illustrates the usage of the bluemira
# toroidal_harmonic_approximation function to create
# Toroidal Harmonic (TH) constraints to be used in a
# coil current optimisation problem for a single null
# DEMO-like equilibrium.

# %%
# Imports

from dataclasses import dataclass
from pathlib import Path
from time import time

import matplotlib.pyplot as plt

from bluemira.base.file import get_bluemira_path
from bluemira.display.plotter import Zorder
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.optimisation.harmonics.harmonics_approx_functions import (
    PointType,
    collocation_points,
)
from bluemira.equilibria.optimisation.harmonics.toroidal_harmonics_approx_functions import (  # noqa: E501
    TauLimit,
    plot_toroidal_harmonic_approximation,
    toroidal_harmonic_approximation,
    toroidal_harmonic_grid_and_coil_setup,
)

# %%
# Get equilibrium data from EQDSK file and plot
file_path = Path(get_bluemira_path("equilibria/data", subfolder="examples"), "SOF.json")

ref_eq = Equilibrium.from_eqdsk(file_path, from_cocos=3, qpsi_positive=False)

# f, ax = plt.subplots()
# ref_eq.plot(ax)
# ref_eq.coilset.plot(ax)

# %%
n_points = 8
psi_norm_inner = 0.65
collocation = collocation_points(
    ref_eq.get_LCFS(),
    PointType.INNER_FS,
    n_points=n_points,
    psi_norm_inner=psi_norm_inner,
    eq=ref_eq,
)

# inner FS for plotting
inner_fs = ref_eq.get_flux_surface(psi_norm_inner)
inner_fs_2 = ref_eq.get_flux_surface(psi_norm_inner / 2)


f, ax = plt.subplots()
ref_eq.plot(ax)
ax.scatter(collocation.x, collocation.z, zorder=10)
ax.plot(inner_fs.x, inner_fs.z, zorder=9, color="r")
# ax.plot(inner_fs_2.x, inner_fs_2.z, zorder=9, color="b")
plt.show()

# %%

n_points = 4
psi_norm_inner = 0.65
collocation = collocation_points(
    ref_eq.get_LCFS(),
    PointType.DOUBLE_INNER_FS,
    n_points=n_points,
    psi_norm_inner=psi_norm_inner,
    eq=ref_eq,
)

# inner FS for plotting
inner_fs = ref_eq.get_flux_surface(psi_norm_inner)
inner_fs_2 = ref_eq.get_flux_surface(psi_norm_inner / 2)


f, ax = plt.subplots()
ref_eq.plot(ax)
ax.scatter(collocation.x, collocation.z, zorder=10)
ax.plot(inner_fs.x, inner_fs.z, zorder=9, color="r")
ax.plot(inner_fs_2.x, inner_fs_2.z, zorder=9, color="b")
plt.show()
# %%
# try approx with these colloc pts, as a start

psi_norm = 0.95
R_0, Z_0 = ref_eq.effective_centre()
th_params = toroidal_harmonic_grid_and_coil_setup(
    eq=ref_eq, R_0=R_0, Z_0=Z_0, tau_limit=TauLimit.COIL
)

# TH approximation
th_result = toroidal_harmonic_approximation(
    eq=ref_eq,
    th_params=th_params,
    psi_norm=psi_norm,
    n_degrees_of_freedom=6,
    max_harmonic_mode=5,
    plasma_mask=True,
    collocation_pts=collocation,
)
f, ax = plot_toroidal_harmonic_approximation(
    eq=ref_eq, th_params=th_params, result=th_result, psi_norm=psi_norm
)
ax.set_title("Comparison of bluemira coilset psi to TH approx.")
ref_eq.coilset.plot(ax)
plt.show()


print(f"Cos modes used = {th_result.cos_m}")
print(f"Sin modes used = {th_result.sin_m}")
print(f"Error in approx = {th_result.error}")

# %%
n_points = 2
collocation = collocation_points(
    ref_eq.get_LCFS(),
    PointType.WONKY_PLUS,
    n_points=n_points,
    # psi_norm_inner=psi_norm_inner,
    eq=ref_eq,
)


f, ax = plt.subplots()
ref_eq.plot(ax)
ax.scatter(collocation.x, collocation.z, zorder=10)
plt.show()


psi_norm = 0.95
R_0, Z_0 = ref_eq.effective_centre()
th_params = toroidal_harmonic_grid_and_coil_setup(
    eq=ref_eq, R_0=R_0, Z_0=Z_0, tau_limit=TauLimit.COIL
)

# TH approximation
th_result = toroidal_harmonic_approximation(
    eq=ref_eq,
    th_params=th_params,
    psi_norm=psi_norm,
    n_degrees_of_freedom=6,
    max_harmonic_mode=5,
    plasma_mask=True,
    collocation_pts=collocation,
)
f, ax = plot_toroidal_harmonic_approximation(
    eq=ref_eq, th_params=th_params, result=th_result, psi_norm=psi_norm
)
ax.set_title("Comparison of bluemira coilset psi to TH approx.")
ref_eq.coilset.plot(ax)
plt.show()


print(f"Cos modes used = {th_result.cos_m}")
print(f"Sin modes used = {th_result.sin_m}")
print(f"Error in approx = {th_result.error}")
