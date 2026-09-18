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
from dataclasses import asdict
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.analysis import EqAnalysis
from bluemira.equilibria.diagnostics import (
    EqDiagnosticOptions,
    EqPlotMask,
    EqSubplots,
    PsiPlotType,
)
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.error import FluxSurfaceError
from bluemira.equilibria.find import _in_plasma
from bluemira.equilibria.optimisation.constraints import (
    IsofluxConstraint,
    MagneticConstraintSet,
    PsiBoundaryConstraint,
    RadialFieldConstraint,
    VerticalFieldConstraint,
)
from bluemira.equilibria.optimisation.harmonics.harmonics_approx_functions import (
    fs_fit_metric,
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


# %%


def from_optimisation(
    ref_eq, move_legs_eq, pt_data, program, result, conv_lim, opt_time, psi_norm
):
    """Create opt dict"""
    try:
        eq_summary = move_legs_eq.analyse_plasma()
        physics_dict = asdict(eq_summary)
        for key in physics_dict:
            physics_dict[key] = physics_dict[key].value
    except FluxSurfaceError:
        physics_dict = {}

    _os, xs = move_legs_eq.get_OX_points()
    x_pt_x = xs[0].x
    x_pt_z = xs[0].z

    original_FS = ref_eq.get_flux_surface(psi_norm)

    opt_FS = move_legs_eq.get_flux_surface(psi_norm)  # TODO is this okay?
    fs_fit_metric_95 = fs_fit_metric(original_FS, opt_FS)

    # in original FS
    mask_matrix = np.zeros_like(ref_eq.x)
    mask = _in_plasma(
        ref_eq.x, ref_eq.z, mask_matrix, original_FS.xz.T, include_edges=True
    )
    l2_error_95 = np.linalg.norm(mask * (ref_eq.psi() - move_legs_eq.psi()))

    results_dict = {
        "pt_placement": pt_data[0] + 1,
        "converged": program.check_converged(),
        "fom": result.f_x,
        "iters": result.n_evals,
        "opt_time": opt_time,
        "rel_delta_psi": 100 * program.convergence.progress[-1],
        "cons_satisfied": result.constraints_satisfied,
        "conv_lim": conv_lim * 100,
        "x_pt_x": x_pt_x,
        "x_pt_z": x_pt_z,
        "fs_fit_metric_95": fs_fit_metric_95,
        "l2_error_95": l2_error_95,
    }
    results_dict.update(physics_dict)
    return results_dict


# %%
# FULL SETUP OF CONSTRAINTS TIMED
# keep separate to results dataclass in case of changes with refactor?
# %% [markdown]
# Find the TH approximation of the coilset contribution to the
# core plasma region
# %%

t = time()

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

# set up constraint
th_constraint = ToroidalHarmonicConstraint(
    th_result=th_result,
    constraint_type="equality",
)
# default tolerance is 1e-3 * amps

# time taken
th_time = time() - t

print(f"{th_time=}")

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
# %%


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
# 6 cons setup

t = time()
lcfs = ref_eq.get_LCFS()
arg_inner = np.argmin(lcfs.x)

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

# tolerance is
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
    tolerance=1e-3 * np.abs(ref_eq.Bx(x_extrema[2], z_extrema[2])),
)

vertical_constraint = VerticalFieldConstraint(
    x=x_extrema[1],
    z=z_extrema[1],
    target_value=ref_eq.Bz(x_extrema[1], z_extrema[1]),
    tolerance=1e-3
    * np.abs(
        ref_eq.Bz(x_extrema[1], z_extrema[1]),
    ),
)

six_con_time = time() - t

print(f"{six_con_time=}")

f, ax = plt.subplots()
# extrema.plot(ax)
radial_constraint.plot(ax)
vertical_constraint.plot(ax)
ref_eq.plot(ax)

# %%

# psi boundary constraint for core instead of radial+vertical field
t = time()
lcfs = ref_eq.get_LCFS()
arg_inner = np.argmin(lcfs.x)

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

psi_bdry = PsiBoundaryConstraint(
    x=x_extrema,
    z=z_extrema,
    target_value=ref_eq.psi(x_extrema[2], z_extrema[2]),
)

psi_bdry_time = time() - t
print(f"{psi_bdry_time=}")
# %%


def plot_compare(
    move_legs_eq, leg_points, pt_data, con_type, result, program, conv, opt_time
):
    # ref eq w/ leg pts
    f, axs = plt.subplots(2, 2)
    ref_eq.plot(axs[0][0])
    leg_points.plot(axs[0][0])
    axs[0][0].set_title("Starting equilibrium")

    # opt eq w leg points
    move_legs_eq.plot(axs[0][1])
    leg_points.plot(axs[0][1])
    axs[0][1].set_title("Equilibrium after optimisation")

    # diff plot
    diag_ops = EqDiagnosticOptions(
        psi_diff=PsiPlotType.PSI_ABS_DIFF,
        plot_mask=EqPlotMask.OUT_COMBO_LCFS,
    )
    eq_analysis = EqAnalysis(
        input_eq=move_legs_eq, diag_ops=diag_ops, reference_eq=ref_eq
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
    axs[1][1].annotate(f"Time [s]: {opt_time:.3f}", (0.1, 0.6))
    axs[1][1].annotate(
        f"delta_psi: {100 * program.convergence.progress[-1]:.3f}%", (0.1, 0.5)
    )
    axs[1][1].annotate(
        f"Constraints satisfied: {result.constraints_satisfied}", (0.1, 0.4)
    )
    axs[1][1].annotate(f"Convergence limit: {conv * 100}%", (0.1, 0.3))

    color = "green" if program.check_converged() else "red"
    f.suptitle(
        f"Point placement #{pt_data[0] + 1}, using {con_type} constraints",
        y=1.05,
        color=color,
    )
    # save plot
    # eg TH_conv_lim_1e-3, Core_conv_lim_1e-3 etc
    folder = con_type + "_conv_lim_" + str(conv)
    if not Path(folder).exists():
        Path.mkdir(folder)
    name = "point_placement_" + str(pt_data[0] + 1) + "_multi"
    save_figure(fig=f, name=name, save=True, folder=folder)

    # split diff plot
    diag_ops = EqDiagnosticOptions(
        psi_diff=PsiPlotType.PSI_ABS_DIFF,
        split_psi_plots=EqSubplots.XZ_COMPONENT_PSI,
        plot_mask=EqPlotMask.OUT_COMBO_LCFS,
    )
    eq_analysis = EqAnalysis(
        input_eq=move_legs_eq, diag_ops=diag_ops, reference_eq=ref_eq
    )

    f2, _ax2 = eq_analysis.plot_compare_psi()
    plt.suptitle(
        f"Point placement #{pt_data[0] + 1}, using {con_type} constraints",
        y=0.85,
        color=color,
    )
    name = "point_placement_" + str(pt_data[0] + 1) + "_split_diff"
    save_figure(fig=f2, name=name, save=True, folder=folder)

    # plot_compare_profiles
    ref_eq.label = "Ref eq"
    move_legs_eq.label = "Opt eq"
    eq_analysis = EqAnalysis(input_eq=move_legs_eq, reference_eq=ref_eq)
    f3, _ax = eq_analysis.plot_compare_profiles()
    plt.suptitle(
        f"Point placement #{pt_data[0] + 1}, using {con_type} constraints",
        y=0.95,
        color=color,
    )
    name = "point_placement_" + str(pt_data[0] + 1) + "_plot_compare_psi"
    save_figure(fig=f3, name=name, save=True, folder=folder)

    return from_optimisation(
        ref_eq=ref_eq,
        move_legs_eq=move_legs_eq,
        pt_data=pt_data,
        program=program,
        result=result,
        conv_lim=conv,
        opt_time=opt_time,
        psi_norm=0.95,  # compare for 95th
    )


def th_cons_opt(leg_points, pt_data, conv):  # noqa: D103
    move_legs_eq = deepcopy(ref_eq)
    t = time()
    th_move_legs = TikhonovCurrentCOP(
        eq=move_legs_eq,
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
    )
    result = program()

    opt_time = time() - t

    return (
        move_legs_eq.coilset.current,
        plot_compare(
            move_legs_eq=move_legs_eq,
            leg_points=leg_points,
            pt_data=pt_data,
            con_type="TH",
            result=result,
            program=program,
            conv=conv,
            opt_time=opt_time,
        ),
    )


def core_cons_opt(leg_points, pt_data, conv):  # noqa: D103
    move_legs_eq = deepcopy(ref_eq)
    t = time()
    th_move_legs = TikhonovCurrentCOP(
        eq=move_legs_eq,
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
    )

    result = program()

    opt_time = time() - t

    return (
        move_legs_eq.coilset.current,
        plot_compare(
            move_legs_eq=move_legs_eq,
            leg_points=leg_points,
            pt_data=pt_data,
            con_type="Bdry",
            result=result,
            program=program,
            conv=conv,
            opt_time=opt_time,
        ),
    )


def psi_bdry_cons_opt(leg_points, pt_data, conv):  # noqa: D103
    move_legs_eq = deepcopy(ref_eq)
    t = time()
    th_move_legs = TikhonovCurrentCOP(
        eq=move_legs_eq,
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
    )

    result = program()

    opt_time = time() - t

    return (
        move_legs_eq.coilset.current,
        plot_compare(
            move_legs_eq=move_legs_eq,
            leg_points=leg_points,
            pt_data=pt_data,
            con_type="Psi_bdry_core",
            result=result,
            program=program,
            conv=conv,
            opt_time=opt_time,
        ),
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

dist_1 = 2.4  # values calculated from the original 2 points that worked
dist_2 = 3.5  # values calculated from the original 2 points that worked


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


def leg_opt(opt_func, conv):
    # Ref eq physics params for comparison
    # all other values set to None
    reference_results = {}
    reference_results["pt_placement"] = 0
    _, xs = ref_eq.get_OX_points()
    x_pt_x, x_pt_z = xs[0].x, xs[0].z
    reference_results["x_pt_x"] = x_pt_x
    reference_results["x_pt_z"] = x_pt_z
    vars = [
        "converged",
        "fom",
        "iters",
        "opt_time",
        "rel_delta_psi",
        "cons_satisfied",
        "conv_lim",
        "fs_fit_metric_95",
        "l2_error_95",
    ]

    for var in vars:
        reference_results[var] = None

    phys_results = asdict(ref_eq.analyse_plasma())
    for key in phys_results:
        phys_results[key] = phys_results[key].value

    reference_results.update(phys_results)

    results_matrix = [reference_results]

    # Original coil currents for comparison
    coilset_matrix = [ref_eq.coilset.current]
    for i in range(len(theta)):
        print(f"Point placement #{i}")
        x_coord = np.array([coords_x[i], coords_x_2[i]])
        z_coord = np.array([coords_z[i], coords_z_2[i]])
        leg_constraint = IsofluxConstraint(
            x_coord,
            z_coord,
            lcfs.x[arg_inner],
            lcfs.z[arg_inner],
            tolerance=1e-3,
        )

        # array of point data for plot info
        pt_data = [i, x_coord, z_coord]
        coilset_currents, opt_results = opt_func(
            leg_constraint, pt_data=pt_data, conv=conv
        )
        results_matrix.append(opt_results)
        coilset_matrix.append(coilset_currents)

    # put to table or to csv

    df = pd.DataFrame(results_matrix)

    name = opt_func.__name__ + "_" + str(conv) + "_eq_params.xlsx"

    df.to_excel(name, index=False)
    name = opt_func.__name__ + "_" + str(conv) + "_eq_params.csv"
    df.to_csv(name, index=False)

    df

    point_placement = list(range(len(theta) + 1))
    coil_names = [name + " [A]" for name in ref_eq.coilset.name]
    current_dict = dict(zip(point_placement, coilset_matrix, strict=True))

    name = opt_func.__name__ + "_" + str(conv) + "_coil_currents.xlsx"

    df2 = pd.DataFrame(current_dict, index=coil_names)
    df2.to_excel(name)

    name = opt_func.__name__ + "_" + str(conv) + "_coil_currents.csv"
    df2.to_csv(name)


# %%

opt_func_list = [
    th_cons_opt,
    th_cons_opt,
    core_cons_opt,
    core_cons_opt,
    psi_bdry_cons_opt,
    psi_bdry_cons_opt,
]

convs = [1e-3, 5e-3] * 3


for opt_func, conv in zip(opt_func_list, convs, strict=True):
    leg_opt(opt_func=opt_func, conv=conv)
# %%
# coilset currents need coil names and point placement

# df.to_excel("TH_1e-3.xlsx", header=labels)
# appended_data = pd.concat(results_matrix)

# appended_data.to_excel("TH_results.xlsx")
# %%
# TODO
# for ones that have converged + constraints satisfied:
# FOM - how well
# n_evals - how fast
# delta psi - how stable

# of the 20 we look at, how many successfully converged for each experiment
# maybe more successfully converge with one version, but lower FOM for another (things
# like this)


# %%
# TODO for paper
# - run opts with time so save the time for each opt - get for each opt for each set of
#   leg pts
# - time for running cells: isoflux (+ getting lcfs), toroidal, psi bdry
#   ^put all in one cell for each type and time (all in isolation)
# - for each eq produced from each opt get in one big table:
#       - the eq.analyse_plasma()
#       - could create modified version of physics_info_table to contain all the info eg
#         phys and coilset I and lcfs fit metric and x pt locs - could use to save the
#         info as a json or csv or something (csv easier to move to excel)
#         ^ create results dataclass with the data and then whack in table then to csv
#         and have column headers as point placement 1, 2, 3 etc, then excel with sheet
#         for each -> put phys params at bottom: exp results, stuff written here, then
#         phys params + coil currents
#       - coil currents (save currents and positions)
# - plot_compare_profiles() - one for each point placement for each type of cons
# - LCFS fit metric for each one (compares the area of the 2 LCFSs)
# - x point location (just first one - if it flips it is reflected in z value)
# - keep to 20 points (dont want finer than the eq grid)
# - calculate the error when comparing ref to the opt eq using same calc as in TH that we
#  use to decide on appropriate harmonics to use


# results dataclass inherit from EqSummary and then add the params we need
# might need to allow stuff to be None so can use from_equilibrium method
