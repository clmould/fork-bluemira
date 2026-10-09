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

import matplotlib.pyplot as plt

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.optimisation.harmonics.toroidal_harmonics_approx_functions import (  # noqa: E501
    TauLimit,
    plot_toroidal_harmonic_approximation,
    toroidal_harmonic_approximation,
    toroidal_harmonic_grid_and_coil_setup,
)
from time import time
import numpy as np

# %%
# Get equilibrium data from EQDSK file and plot
file_path = Path(get_bluemira_path("equilibria/data", subfolder="examples"), "SOF.json")

ref_eq = Equilibrium.from_eqdsk(file_path, from_cocos=3, qpsi_positive=False)

f, ax = plt.subplots()
ref_eq.plot(ax)
ref_eq.coilset.plot(ax)

# %%

# TH setup
psi_norm = 0.95
R_0, Z_0 = ref_eq.effective_centre()
th_params = toroidal_harmonic_grid_and_coil_setup(
    eq=ref_eq, R_0=R_0, Z_0=Z_0, tau_limit=TauLimit.COIL
)


# %%

# setup collocation investigation parameters

collocation_results = {}
all_th_results = {}


# %%
@dataclass
class CollocationResult:
    dof: int
    value: int
    n_points: int
    error: float
    residual: float
    condition_number: float


value_range = 37
dof_values = [3, 4, 5, 6]


# %%
# NOTE - this runs everything, but have saved all results in a dict - read in from next cell!

# t = time()
# test_dict = {}
# # for dof in [3]:  # due to failure when first running
# for dof in dof_values:
#     # want 36 extra points
#     for val in range(value_range):
#         # plot and save plots for certain values
#         # some certain ones, some relative, eg 2 * DOF, 3 * DOF, DOF**2
#         # some values plus 2DOF, 3DOF, 4DOF, DOF**2
#         # plot = (
#         #     True
#         #     if val in [0, 18, 36, 37, dof, 2 * dof, 3 * dof, dof**2 - dof]
#         #     else False
#         # )  # noqa: E501
#         try:
#             th_result = toroidal_harmonic_approximation(
#                 eq=ref_eq,
#                 th_params=th_params,
#                 psi_norm=psi_norm,
#                 n_degrees_of_freedom=dof,
#                 max_harmonic_mode=5,
#                 n_points=dof + val,  # use DOF + value as n_points
#                 plasma_mask=True,
#                 value=val,
#                 plot=True,
#             )
#             colloc_res = CollocationResult(
#                 dof=dof,
#                 value=val,
#                 n_points=th_result.n_points,
#                 error=th_result.error,
#                 residual=th_result.residual,
#                 condition_number=th_result.condition_number,
#             )
#         except EquilibriaError:
#             th_result = "N/A"
#             colloc_res = "N/A"
#         # save th for completeness and possible plotting later
#         # all_th_results[dof] = th_result
#         # collocation_results[dof, val] = colloc_res
#         test_dict[dof, val] = colloc_res

# # for reference of how long this takes to run
# coll_invest = time() - t
# print(f"time = {coll_invest}")


# %%

# want to iterate over different DOFs and values (to change # of collco pts)
# want t save to csv: input DOF , number of colloc pts (TODO return in th_res), error,
# residual and condition number
# for loop in for loop
# then plot

# can look at difference in DOF against res and cond number and see if
# trend between runs with different DOF
# want to find where we have enough points where we have enough info
# but not too many that we overfit


# %%
# dict of form {(DOF, value): CollocationResult(error, residual, condition number)}
# plot each DOF [3,4,5,6] separately for all values vs cond, res
# plot all DOF for same value vs cond, res (eg value 0 + all DOF, value 1 + all DOF)

# matrix condition number ideally want to be 1
# -> an ill-conditioned matrix (large condition number) means that a
# small change in the input results in a large change in the output
# eg small noise would mean huge amplitude changes for ill conditioned matrix
# residual as close to 0 as possible (depending on scale )
# want error as small as we can get
# %%
# Create some plots of DOF vs condition number/residual
# 4 different values of DOF, 37 of value, 3 to 42 points sampled

# - plots comparing number of points sampled vs condition and residual
#   (will be a lot of points on this graph, but is there a trend between
#   n_points and condition+residual?)
# - plots for same n_points, diff DOF vs cond + res - could help find
#   relation between DOF and n_points to use
# - plots for same degree but different number of points - could help
#   find relation between DOF and n_points to use


# %%
# read in dict from file to not have to run again
import numpy as np
from numpy import array, float64

# import np.float64
with open("collocation_results.txt", "r") as f:
    collocation_results = f.read()

collocation_results = eval(collocation_results)


# %%
# plot all
# colour code for diff DOF
dof_colours = {3: "red", 4: "cyan", 5: "magenta", 6: "orange"}

f, ax = plt.subplots(1, 2)
ax[0].set_title("DOF vs condition number")
ax[1].set_title("DOF vs residual")
for key in collocation_results.keys():
    dof, _val = key
    if collocation_results[key] == "N/A":
        continue
    cond = collocation_results[key].condition_number
    ax[0].scatter(dof, cond, color=dof_colours[dof])

    res = collocation_results[key].residual
    # print(f"{key=}")
    # print(f"{res=}")

    if isinstance(res, np.ndarray):
        if res.size > 0:
            ax[1].scatter(dof, res, color=dof_colours[dof])

    else:
        ax[1].scatter(dof, res, color=dof_colours[dof])


for a in ax:
    a.set_yscale("log")
    a.set_xticks(dof_values)

# %%
# # for the same value, plot diff dof vs cond + res
# f, ax = plt.subplots(value_range, 2)


# for i in range(value_range):
#     ax[i, 0].set_title(f"DOF vs condition number for value {i}")
#     ax[i, 1].set_title(f"DOF vs residual for value {i}")
#     for dof in dof_values:
#         if collocation_results[dof, i] == "N/A":
#             print(f"skipping {dof=} value={i}")
#             continue
#         cond = collocation_results[dof, i].condition_number
#         res = collocation_results[dof, i].residual

#         ax[i, 0].scatter(dof, cond)
#         ax[i, 1].scatter(dof, res)
#         for a in [ax[i, 0], ax[i, 1]]:
#             a.set_yscale("log")
#             a.set_xticks(dof_values)
#     # plt.show()
# f.tight_layout()
# plt.show()


# # %%
# # same DOF, different values:

f, ax = plt.subplots(len(dof_values), 3)


for i, dof in enumerate(dof_values):
    ax[i, 0].set_title(f"number of points vs condition number for DOF {dof}")
    ax[i, 1].set_title(f"number of points vs residual for DOF {dof}")
    ax[i, 2].set_title(f"number of points vs th approx error for DOF {dof}")
    for value in range(value_range):
        if collocation_results[dof, value] == "N/A":
            print(f"skipping {dof=} {value=}")
            continue
        cond = collocation_results[dof, value].condition_number
        res = collocation_results[dof, value].residual
        err = collocation_results[dof, value].error

        ax[i, 0].scatter(value, cond)
        try:
            ax[i, 1].scatter(value, res)
        except ValueError:  # when res is []
            print(f"{dof=}")
            print(f"{res=}")

        ax[i, 2].scatter(value, err)

        for a in [ax[i, 0], ax[i, 1], ax[i, 2]]:
            a.set_yscale("log")
            a.set_xticks(
                range(value_range)[::2],
            )


ax[2][0].set_ylim(10**1, 10**4)
ax[1][0].set_ylim(10**1, 10**4)
ax[0][0].set_ylim(10**1, 10**4)
ax[3][0].set_ylim(10**1, 10**4)

ax[0][1].set_ylim(10**-7, 10**0)
ax[1][1].set_ylim(10**-7, 10**0)
ax[2][1].set_ylim(10**-7, 10**0)
ax[3][1].set_ylim(10**-7, 10**0)

ax[0][2].set_ylim(10**0, 10**2)
ax[1][2].set_ylim(10**0, 10**2)
ax[2][2].set_ylim(10**0, 10**2)
ax[3][2].set_ylim(10**0, 10**2)
# from matplotlib.ticker import MaxNLocator
# plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=6))


f.tight_layout()
plt.show()

# %%
# # plot DOF vs cond and res for all values in same plot
# colors = [""] * value_range
# colors = ["r", "b", "g", "c", "m"]

# f, ax = plt.subplots(3, 2)
# # plot number of points vs cond and res, colour code for diff DOF
# for p, points in enumerate(range(3, 6)):
#     for i, dof in enumerate(range(dof_values)):
#         ax[i, p].set_title(f"{p=}")


# %%
th_result = toroidal_harmonic_approximation(
    eq=ref_eq,
    th_params=th_params,
    psi_norm=psi_norm,
    n_degrees_of_freedom=5,
    max_harmonic_mode=5,
    n_points=6,
    plasma_mask=True,
    plot=True,
)
#
f, ax = plot_toroidal_harmonic_approximation(
    eq=ref_eq, th_params=th_params, result=th_result, psi_norm=psi_norm
)
ref_eq.coilset.plot(ax)
plt.show()
