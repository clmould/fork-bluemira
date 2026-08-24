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
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimisation.constraints import (
    IsofluxConstraint,
    PsiBoundaryConstraint,
    RadialFieldConstraint,
    VerticalFieldConstraint,
)
from bluemira.equilibria.optimisation.harmonics.harmonics_constraints import (
    ToroidalHarmonicConstraint,
)
from bluemira.equilibria.optimisation.harmonics.toroidal_harmonics_approx_functions import (  # noqa: E501
    TauLimit,
    plot_toroidal_harmonic_approximation,
    toroidal_harmonic_approximation,
    toroidal_harmonic_grid_and_coil_setup,
)
from bluemira.equilibria.optimisation.problem._minimal_current import MinimalCurrentCOP
from bluemira.equilibria.solve import (
    DudsonConvergence,
    PicardIterator,
)

# %% TODOs


# TODO save all the plots using a save fn from bluemira,
# then put all these in a sharepoint folder
# save all 20 plots, then save with dudson convergence 1e-3 for both
# 4 folders: TH 1e-3, TH 5e-3 (same for core con)
# put conv limit as arg, and have save as a bool
# can use save_figure() , just create a folder
# then put on sharepoint and share with georgie

# once saved speak to georgie

# TODO want converged counter , and want converged status in the graph -
# (title: iter x, converged yes/no)


# TODO for each opt:
# - rename TH approx result to th_result DONE
# - compare the value of delta_psi for end of opt DONE
# - figure of merit at end of each opt from Picard result.f_x DONE
# - number of iterations of G-S to get to that point (printed in window) DONE
# (want to save value printed from solve line ~185) - DONE
# - for each eq, want starting eq, converged status and whether or not the
# constraints have all been satisfied (put info on each plot) DONE
# - add diff plot to compare the entire psi field (just the 1, not split for plasma and
# coilset) DONE
# ^good to get them all on the same figure - 3 plots and extra window for printing
# the relevant info -> do in 2x2 grid DONE
# - add the point coordinates to the info window DONE
# - round it all to 3dp !!! DONE
# - add to title 'core con' or 'th' so we know which we are doing DONE


# - how well did we achieve our goal, how quickly (many iterations), how stable (eg
# smaller delta_psi means more stable solution)
# can also look at figure of merit (how well we moved the leg)

# TODO with Georgie
# - save info in table as well as in plots

# first:
# - also investigate why was core converging with 1e-3? run with 1e-3 then
# look at the actual convergence values there -> looked like the core
# constraints were converging at a higher dpsi_rel than expected -> just double
# check we weren't misreading something here!


# %%


# Minimal current

# Need to use the unconstrained from eudemo example for this!
file_path = Path(
    get_bluemira_path("equilibria/data", subfolder="examples"),
    "unconstrained_reference_eq.json",
)

unconstrained_eq = Equilibrium.from_eqdsk(file_path, from_cocos=3, qpsi_positive=False)
unconstrained_eq.plot()
plt.show()


psi_norm = 0.95
R_0, Z_0 = unconstrained_eq.effective_centre()
uncon_th_params = toroidal_harmonic_grid_and_coil_setup(
    eq=unconstrained_eq, R_0=R_0, Z_0=Z_0, tau_limit=TauLimit.COIL
)

# TH approximation
uncon_th_result = toroidal_harmonic_approximation(
    eq=unconstrained_eq,
    th_params=uncon_th_params,
    psi_norm=psi_norm,
    n_degrees_of_freedom=6,
    max_harmonic_mode=5,
    plasma_mask=True,
)

print(f"Cos modes used = {uncon_th_result.cos_m}")
print(f"Sin modes used = {uncon_th_result.sin_m}")
print(f"Error in approx = {uncon_th_result.error}")

# Plot to compare th approx psi to bm psi
f, ax = plot_toroidal_harmonic_approximation(
    eq=unconstrained_eq,
    th_params=uncon_th_params,
    result=uncon_th_result,
    psi_norm=psi_norm,
)
ax.set_title("Comparison of bluemira coilset psi to TH approx.")
unconstrained_eq.coilset.plot(ax)
plt.show()

# %%
# new constraints for the unconstrained eq
# TH

uncon_th_con = ToroidalHarmonicConstraint(
    th_result=uncon_th_result,
    constraint_type="equality",
)

# 6 core cons
un_lcfs = unconstrained_eq.get_LCFS()
un_x_bdry, un_z_bdry = un_lcfs.x, un_lcfs.z
arg_inner = np.argmin(un_x_bdry)

un_x_extrema = un_lcfs.x[
    np.array([
        np.argmin(un_lcfs.z),
        np.argmax(un_lcfs.z),
        np.argmin(un_lcfs.x),
        np.argmax(un_lcfs.x),
    ])
]
un_z_extrema = un_lcfs.z[
    np.array([
        np.argmin(un_lcfs.z),
        np.argmax(un_lcfs.z),
        np.argmin(un_lcfs.x),
        np.argmax(un_lcfs.x),
    ])
]

un_extrema = IsofluxConstraint(
    un_x_extrema,
    un_z_extrema,
    ref_x=un_x_extrema[2],
    ref_z=un_z_extrema[2],
)

un_radial_constraint = RadialFieldConstraint(
    x=un_x_extrema[2],
    z=un_z_extrema[2],
    target_value=unconstrained_eq.Bx(un_x_extrema[2], un_z_extrema[2]),
)

un_vertical_constraint = VerticalFieldConstraint(
    x=un_x_extrema[1],
    z=un_z_extrema[1],
    target_value=unconstrained_eq.Bz(un_x_extrema[1], un_z_extrema[1]),
)

f, ax = plt.subplots()
un_extrema.plot(ax)
un_radial_constraint.plot(ax)
un_vertical_constraint.plot(ax)
unconstrained_eq.plot(ax)


# psi bdry using 4 extrema
un_psi_bdry = PsiBoundaryConstraint(
    x=un_x_extrema,
    z=un_z_extrema,
    target_value=unconstrained_eq.psi(un_x_extrema[2], un_z_extrema[2]),
)

# %%
f, ax = plt.subplots(2, 4)
for axs in ax[:][1]:
    axs.axis("off")

unconstrained_eq.plot(ax[0][0])
ax[0][0].set_title("Starting unconstrained eq")
f.suptitle("Minimal current opt")

# TH con
minimal_current_eq_th = deepcopy(unconstrained_eq)
minimal_current_opt_problem_th = MinimalCurrentCOP(
    minimal_current_eq_th,
    opt_algorithm="SLSQP",
    constraints=[uncon_th_con],
)
program = PicardIterator(
    minimal_current_opt_problem_th,
    fixed_coils=True,
    convergence=DudsonConvergence(5e-3),
    relaxation=0.1,
)
result = program()

minimal_current_eq_th.plot(ax[0][1])
ax[0][1].set_title("TH constraints")
ax[1][1].annotate(f"Converged: {program.check_converged()}", (0.1, 0.9))
ax[1][1].annotate(f"FOM: {result.f_x:.3f}", (0.1, 0.8))
ax[1][1].annotate(f"Iterations: {result.n_evals}", (0.1, 0.7))
ax[1][1].annotate(
    f"delta_psi: {100 * program.convergence.progress[-1]:.3f}%", (0.1, 0.6)
)
ax[1][1].annotate(f"Constraints satisfied: \n{result.constraints_satisfied}", (0.1, 0.5))


minimal_current_eq_core = deepcopy(unconstrained_eq)
minimal_current_opt_problem_core = MinimalCurrentCOP(
    minimal_current_eq_core,
    opt_algorithm="SLSQP",
    constraints=[un_extrema, un_radial_constraint, un_vertical_constraint],
)
program = PicardIterator(
    minimal_current_opt_problem_core,
    fixed_coils=True,
    convergence=DudsonConvergence(5e-3),
    relaxation=0.1,
)
result = program()


minimal_current_eq_core.plot(ax[0][2])

ax[0][2].set_title("Original 6 core constraints")
ax[1][2].annotate(f"Converged: {program.check_converged()}", (0.1, 0.9))
ax[1][2].annotate(f"FOM: {result.f_x:.3f}", (0.1, 0.8))
ax[1][2].annotate(f"Iterations: {result.n_evals}", (0.1, 0.7))
ax[1][2].annotate(
    f"delta_psi: {100 * program.convergence.progress[-1]:.3f}%", (0.1, 0.6)
)
ax[1][2].annotate(f"Constraints satisfied: \n{result.constraints_satisfied}", (0.1, 0.5))


# Min current with new core con:
minimal_current_eq_new_core = deepcopy(unconstrained_eq)
minimal_current_opt_problem_new_core = MinimalCurrentCOP(
    minimal_current_eq_new_core,
    opt_algorithm="SLSQP",
    constraints=[un_psi_bdry],
)
program = PicardIterator(
    minimal_current_opt_problem_new_core,
    fixed_coils=True,
    convergence=DudsonConvergence(5e-3),
    relaxation=0.1,
)
result = program()
minimal_current_eq_new_core.plot(ax[0][3])
ax[0][3].set_title("New core constraints\nw psi bdry")
ax[1][3].annotate(f"Converged: {program.check_converged()}", (0.1, 0.9))
ax[1][3].annotate(f"FOM: {result.f_x:.3f}", (0.1, 0.8))
ax[1][3].annotate(f"Iterations: {result.n_evals}", (0.1, 0.7))
ax[1][3].annotate(
    f"delta_psi: {100 * program.convergence.progress[-1]:.3f}%", (0.1, 0.6)
)
ax[1][3].annotate(f"Constraints satisfied: \n{result.constraints_satisfied}", (0.1, 0.5))
plt.show()


# TODO - try with removing the cos 0
