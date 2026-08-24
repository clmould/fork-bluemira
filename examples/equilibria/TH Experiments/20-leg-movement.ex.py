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
from bluemira.equilibria.analysis import EqAnalysis
from bluemira.equilibria.diagnostics import (
    EqDiagnosticOptions,
    EqPlotMask,
    EqSubplots,
    PsiPlotType,
)
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimisation.constraints import (
    IsofluxConstraint,
    MagneticConstraintSet,
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
from bluemira.equilibria.optimisation.problem._tikhonov import (  # noqa: PLC2701
    TikhonovCurrentCOP,
)
from bluemira.equilibria.solve import (
    DudsonConvergence,
    PicardIterator,
)
from bluemira.utilities.plot_tools import save_figure

# %%
# Get equilibrium data from EQDSK file and plot
file_path = Path(get_bluemira_path("equilibria/data", subfolder="examples"), "SOF.json")

ref_eq = Equilibrium.from_eqdsk(file_path, from_cocos=3, qpsi_positive=False)

f, ax = plt.subplots()
ref_eq.plot(ax)
ref_eq.coilset.plot(ax)
# %% [markdown]
# Find the TH approximation of the coilset contribution to the
# core plasma region

# %%
# Setup grid for TH approximation
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
)

# %% [markdown]
# We can see the TH modes selected by our approximation function
# and plot to compare the TH approx. coilset psi to the
# bluemira coilset psi
# %%
# Info and plot
print(f"Cos modes used = {th_result.cos_m}")
print(f"Sin modes used = {th_result.sin_m}")
print(f"Error in approx = {th_result.error}")

# Plot to compare th approx psi to bm psi
f, ax = plot_toroidal_harmonic_approximation(
    eq=ref_eq, th_params=th_params, result=th_result, psi_norm=psi_norm
)
ax.set_title("Comparison of bluemira coilset psi to TH approx.")
ref_eq.coilset.plot(ax)
plt.show()

# %% [markdown]
# We can now make a TH constraint that we will use to hold the
# coilset contribution to the plasma region fixed in
# a coil current optimisation problem. This constraint uses
# the TH amplitudes from our approximation.
# %%
th_constraint = ToroidalHarmonicConstraint(
    th_result=th_result,
    constraint_type="equality",
)
# Ensure the control coils are set appropriately
# - could within the TH approx. region must have
# their currents held fixed
ref_eq.coilset.control = list(th_params.th_coil_names)

# Plot the constraint region
f, ax = plt.subplots()
th_constraint.plot(ax=ax)
ref_eq.coilset.plot(ax=ax)
ref_eq.plot(ax=ax)


# %%
lcfs = ref_eq.get_LCFS()
arg_inner = np.argmin(lcfs.x)


ref_lcfs = ref_eq.get_LCFS()
arg_inner = np.argmin(ref_lcfs.x)
# TODO are these argmins the same?

# %%
# fair core constraints:
x_extrema = lcfs.x[
    np.array([
        np.argmin(lcfs.z),
        np.argmax(lcfs.z),
        np.argmin(lcfs.x),
        np.argmax(lcfs.x),
    ])
]
z_extrema = lcfs.z[
    np.array([
        np.argmin(lcfs.z),
        np.argmax(lcfs.z),
        np.argmin(lcfs.x),
        np.argmax(lcfs.x),
    ])
]

extrema = IsofluxConstraint(
    x_extrema,
    z_extrema,
    ref_x=x_extrema[2],
    ref_z=z_extrema[2],
)

radial_constraint = RadialFieldConstraint(
    x=x_extrema[2],
    z=z_extrema[2],
    target_value=ref_eq.Bx(x_extrema[2], z_extrema[2]),
)

vertical_constraint = VerticalFieldConstraint(
    x=x_extrema[1],
    z=z_extrema[1],
    target_value=ref_eq.Bz(x_extrema[1], z_extrema[1]),
)

f, ax = plt.subplots()
# extrema.plot(ax)
radial_constraint.plot(ax)
vertical_constraint.plot(ax)
ref_eq.plot(ax)


# %%
# psi boundary constraint for core instead of radial+vertical field
psi_bdry = PsiBoundaryConstraint(
    x=x_extrema,
    z=z_extrema,
    target_value=ref_eq.psi(x_extrema[2], z_extrema[2]),
)
# %%


def plot_compare(th_move_legs_eq, leg_points, pt_data, con_type, result, program, conv):  # noqa: D103
    # ref eq w/ leg pts
    f, axs = plt.subplots(2, 2)
    ref_eq.plot(axs[0][0])
    leg_points.plot(axs[0][0])
    axs[0][0].set_title("Starting equilibrium")

    # opt eq w leg points
    th_move_legs_eq.plot(axs[0][1])
    leg_points.plot(axs[0][1])
    axs[0][1].set_title("Equilibrium after optimisation")

    # diff plot
    diag_ops = EqDiagnosticOptions(
        psi_diff=PsiPlotType.PSI_ABS_DIFF,
        plot_mask=EqPlotMask.OUT_COMBO_LCFS,
    )
    eq_analysis = EqAnalysis(
        input_eq=th_move_legs_eq, diag_ops=diag_ops, reference_eq=ref_eq
    )
    eq_analysis.plot_compare_psi(ax=axs[1][0])

    # final axis for text about the run
    x_coord_1, x_coord_2 = pt_data[1]
    z_coord_1, z_coord_2 = pt_data[2]
    axs[1][1].axis("off")
    axs[1][1].annotate(
        f"Leg target coords: ({x_coord_1:.3f}, {z_coord_1:.3f}), ({x_coord_2:.3f}, {z_coord_2:.3f})",
        (0.1, 1.0),
    )
    axs[1][1].annotate(f"Converged: {program.check_converged()}", (0.1, 0.9))
    axs[1][1].annotate(f"FOM: {result.f_x:.3f}", (0.1, 0.8))
    axs[1][1].annotate(f"Iterations: {result.n_evals}", (0.1, 0.7))
    axs[1][1].annotate(
        f"delta_psi: {100 * program.convergence.progress[-1]:.3f}%", (0.1, 0.6)
    )
    axs[1][1].annotate(
        f"Constraints satisfied: {result.constraints_satisfied}", (0.1, 0.5)
    )
    axs[1][1].annotate(f"Convergence limit: {conv * 100}%", (0.1, 0.4))
    # TODO
    # constraints satisfied?
    color = "green" if program.check_converged() else "red"
    f.suptitle(
        f"Point placement #{pt_data[0]}, using {con_type} constraints",
        y=1.05,
        color=color,
    )
    # save plot
    # eg TH_conv_lim_1e-3, Core_conv_lim_1e-3
    folder = con_type + "_conv_lim_" + str(conv)
    if not Path(folder).exists():
        Path.mkdir(folder)
    name = "point_placement_" + str(pt_data[0]) + "_multi"
    save_figure(fig=f, name=name, save=True, folder=folder)

    # split diff plot
    diag_ops = EqDiagnosticOptions(
        psi_diff=PsiPlotType.PSI_ABS_DIFF,
        split_psi_plots=EqSubplots.XZ_COMPONENT_PSI,
        plot_mask=EqPlotMask.OUT_COMBO_LCFS,
    )
    eq_analysis = EqAnalysis(
        input_eq=th_move_legs_eq, diag_ops=diag_ops, reference_eq=ref_eq
    )

    f2, _ax2 = eq_analysis.plot_compare_psi()
    plt.suptitle(
        f"Point placement #{pt_data[0]}, using {con_type} constraints",
        y=0.85,
        color=color,
    )
    name = "point_placement_" + str(pt_data[0]) + "_split_diff"
    save_figure(fig=f2, name=name, save=True, folder=folder)


def run_th_con_optimisation(leg_points, pt_data, conv):  # noqa: D103
    th_move_legs_eq = deepcopy(ref_eq)

    th_move_legs = TikhonovCurrentCOP(
        eq=th_move_legs_eq,
        targets=MagneticConstraintSet([
            leg_points,
        ]),
        constraints=[th_constraint],
        gamma=1e-8,
    )

    program = PicardIterator(
        th_move_legs,
        fixed_coils=True,
        convergence=DudsonConvergence(limit=conv),
        relaxation=0.2,
        maxiter=30,
        check_constraints=True,
        # diagnostic_plotting=PicardDiagnosticOptions(plot=PicardDiagnostic.EQ),
    )
    # result = program() # use this to get the convergence status
    result = program()
    # print(f"convergence: {result.f_x}")
    # print(result.constraints_satisfied)
    plot_compare(
        th_move_legs_eq=th_move_legs_eq,
        leg_points=leg_points,
        pt_data=pt_data,
        con_type="TH",
        result=result,
        program=program,
        conv=conv,
    )


def run_core_con_optimisation(leg_points, pt_data, conv):  # noqa: D103
    th_move_legs_eq = deepcopy(ref_eq)

    th_move_legs = TikhonovCurrentCOP(
        eq=th_move_legs_eq,
        targets=MagneticConstraintSet([
            leg_points,
        ]),
        constraints=[extrema, radial_constraint, vertical_constraint],
        gamma=1e-8,
    )

    program = PicardIterator(
        th_move_legs,
        fixed_coils=True,
        convergence=DudsonConvergence(limit=conv),
        relaxation=0.2,
        maxiter=30,
        check_constraints=True,
        # diagnostic_plotting=PicardDiagnosticOptions(plot=PicardDiagnostic.EQ),
    )

    result = program()
    print(f"converged in {result.n_evals} evals")
    # print(f"convergence: {result.f_x}")
    # print(result.constraints_satisfied)
    plot_compare(
        th_move_legs_eq=th_move_legs_eq,
        leg_points=leg_points,
        pt_data=pt_data,
        con_type="Core",
        result=result,
        program=program,
        conv=conv,
    )


def run_new_core_con_optimisation(leg_points, pt_data, conv):  # noqa: D103
    th_move_legs_eq = deepcopy(ref_eq)

    th_move_legs = TikhonovCurrentCOP(
        eq=th_move_legs_eq,
        targets=MagneticConstraintSet([
            leg_points,
        ]),
        constraints=[psi_bdry],
        gamma=1e-8,
    )

    program = PicardIterator(
        th_move_legs,
        fixed_coils=True,
        convergence=DudsonConvergence(limit=conv),
        relaxation=0.2,
        maxiter=30,
        check_constraints=True,
        # diagnostic_plotting=PicardDiagnosticOptions(plot=PicardDiagnostic.EQ),
    )

    result = program()
    # print(f"converged in {result.n_evals} evals")
    # print(f"convergence: {result.f_x}")
    # print(result.constraints_satisfied)
    plot_compare(
        th_move_legs_eq=th_move_legs_eq,
        leg_points=leg_points,
        pt_data=pt_data,
        con_type="New_core",
        result=result,
        program=program,
        conv=conv,
    )


# %%
# want arrays of leg points to try
# keep at 20 points for now, then decide if we want to do more
theta = np.linspace(-5 * np.pi / 8, -np.pi, 20)  # 10 divisions to start
# theta = np.linspace(-5 * np.pi / 8, -np.pi, 2)  # for testing of graphs

# anchor pt is the x point
_, xs = ref_eq.get_OX_points()
anchor_x, anchor_z = xs[0].x, xs[0].z

# new point at (x + r cos(theta), z + r sin(theta))
# calculated based on the points that currently work

# 1 point to start

dist_1 = 2.4  # values calculated from the 2 points that work above
dist_2 = 3.5  # values calculated from the 2 points that work above

# Try a new distance
# dist_1 = 2.6
# dist_2 = 4.0

coords_x = anchor_x + dist_1 * np.cos(theta)
coords_z = anchor_z + dist_1 * np.sin(theta)

coords_x_2 = anchor_x + dist_2 * np.cos(theta)
coords_z_2 = anchor_z + dist_2 * np.sin(theta)
from matplotlib import cm

colors = cm.rainbow(np.linspace(0, 1, len(coords_x)))
f, ax = plt.subplots()

for i in range(len(coords_x)):
    ax.scatter(coords_x[i], coords_z[i], zorder=10, color=colors[i])
    ax.scatter(coords_x_2[i], coords_z_2[i], zorder=10, color=colors[i])
ref_eq.plot(ax)

# %%

for i in range(len(theta)):
    print(f"Point placement #{i}")
    x_coord = np.array([coords_x[i], coords_x_2[i]])
    z_coord = np.array([coords_z[i], coords_z_2[i]])
    leg_constraint = IsofluxConstraint(
        x_coord,
        z_coord,
        ref_lcfs.x[arg_inner],
        ref_lcfs.z[arg_inner],
        tolerance=1e-3,
    )
    conv = 5e-3
    # array of point data for plot info
    pt_data = [i, x_coord, z_coord]
    run_th_con_optimisation(leg_constraint, pt_data=pt_data, conv=conv)
    # run_core_con_optimisation(leg_constraint, pt_data=pt_data, conv=conv)
    # run_new_core_con_optimisation(leg_constraint, pt_data=pt_data, conv=conv)


# %%
# TODO
# for ones that have converged + constraints satisfied:
# FOM - how well
# n_evals - how fast
# delta psi - how stable

# of the 20 we look at, how many successfully converged for each experiment
# maybe more successfully converge with one version, but lower FOM for another (things
# like this)
