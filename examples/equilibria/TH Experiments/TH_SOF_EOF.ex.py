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
# %%
import json
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimisation.constraints import (
    FieldNullConstraint,
    PsiBoundaryConstraint,
)
from bluemira.equilibria.optimisation.harmonics.harmonics_constraints import (
    ToroidalHarmonicConstraint,
)
from bluemira.equilibria.optimisation.harmonics.toroidal_harmonics_approx_functions import (
    TauLimit,
    plot_toroidal_harmonic_approximation,
    toroidal_harmonic_approximation,
    toroidal_harmonic_grid_and_coil_setup,
    toroidal_harmonics_to_positions,
)
from bluemira.equilibria.optimisation.problem._minimal_current import MinimalCurrentCOP
from bluemira.equilibria.plotting import PLOT_DEFAULTS
from bluemira.equilibria.solve import DudsonConvergence, PicardIterator

# %%
# CREATE reference data (see eudemo_2017.ex.py)
path = get_bluemira_path("equilibria", subfolder="examples")
name = "EUDEMO_2017_CREATE_SOF_separatrix.json"
with open(Path(path, name)) as file:
    data = json.load(file)

sof_xbdry = data["xbdry"]
sof_zbdry = data["zbdry"]
# %%

# Calculated boundary psi for SOF and EOF (see eudemo_2017.ex.py)
psi_sof = 22.81
psi_eof = -25.54
# %%
# Reference equilibrium (see eudemo_2017.ex.py)
# Equilbria obtained from initial unconstrained optimisation for CREATE boundary
# with EU DEMO machine params (we used the CustomProfile option).
file_path = Path(
    get_bluemira_path("equilibria/data", subfolder="examples"),
    "unconstrained_reference_eq.json",
)

ref_eq = Equilibrium.from_eqdsk(file_path, from_cocos=3, qpsi_positive=False)
ref_eq.plot()
plt.show()


# %%
def th_approx_and_info(eq, setup):
    """Get the TH approx, plot and return the result"""
    R_0, Z_0 = eq.effective_centre()
    th_params_ref = toroidal_harmonic_grid_and_coil_setup(
        eq=eq, R_0=R_0, Z_0=Z_0, tau_limit=setup["tau_limit"]
    )

    result = toroidal_harmonic_approximation(
        eq=eq,
        th_params=th_params_ref,
        psi_norm=setup["psi_norm"],
        n_degrees_of_freedom=setup["n_degrees_of_freedom"],
        max_harmonic_mode=setup["max_harmonic_mode"],
        plasma_mask=setup["plasma_mask"],
    )

    _f, ax = plot_toroidal_harmonic_approximation(
        eq=eq, th_params=th_params_ref, result=result, psi_norm=setup["psi_norm"]
    )
    eq.coilset.plot(ax=ax)
    ax.set_title("Comparison of bluemira coilset psi to TH approx.")
    plt.show()

    print("Modes")
    print("------")
    print(f"sin m = {result.sin_m}")
    print(f"cos m = {result.cos_m}")
    print(" ")
    print("Mode Amplitudes")
    print("------")
    print(f"sin amps  = {result.sin_amplitudes}")
    print(f"cos amps = {result.cos_amplitudes}")

    return result, th_params_ref


# %%
# Input settings
setup = {
    "tau_limit": TauLimit.COIL,
    "psi_norm": 0.95,
    "n_degrees_of_freedom": 6,
    "max_harmonic_mode": 5,
    "plasma_mask": True,
}

# %%
ref_result, ref_params = th_approx_and_info(ref_eq, setup)

# %%
# Optimised results for comparison
# SOF from eudemo_2017 example
file_path = Path(get_bluemira_path("equilibria/data", subfolder="examples"), "SOF.json")
sof = Equilibrium.from_eqdsk(file_path, from_cocos=3, qpsi_positive=False)
# EOF from eudemo_2017 example
file_path = Path(get_bluemira_path("equilibria/data", subfolder="examples"), "EOF.json")
eof = Equilibrium.from_eqdsk(file_path, from_cocos=3, qpsi_positive=False)

# %%
_ = sof.get_OX_points()
_ = eof.get_OX_points()


# %%
def psi_info(eq):
    coil_bdry_psi = eq.coilset.psi(*eq._x_points[0][:2])
    plasma_bdry_psi = eq.plasma.psi(*eq._x_points[0][:2])
    total_bdry_psi = eq.psi(*eq._x_points[0][:2])
    coil_cntr_psi = eq.coilset.psi(*eq._o_points[0][:2])
    plasma_cntr_psi = eq.plasma.psi(*eq._o_points[0][:2])
    total_cntr_psi = eq.psi(*eq._o_points[0][:2])
    coil_diff = coil_cntr_psi - coil_bdry_psi
    plasma_diff = plasma_cntr_psi - plasma_bdry_psi
    total_diff = total_cntr_psi - total_bdry_psi
    bdry_diff = coil_bdry_psi - plasma_bdry_psi
    print(f"{coil_bdry_psi=}")
    print(f"{plasma_bdry_psi=}")
    print(f"{total_bdry_psi=}")
    print(f"{coil_cntr_psi=}")
    print(f"{plasma_cntr_psi=}")
    print(f"{total_cntr_psi=}")
    print(f"{coil_diff=}")
    print(f"{plasma_diff=}")
    print(f"{total_diff=}")
    print(f"{bdry_diff=}")


psi_info(ref_eq)
psi_info(sof)
psi_info(eof)
for name in sof.coilset.name:
    print(name + f": {sof.coilset[name].current / 1e6}")
for name in eof.coilset.name:
    print(name + f": {eof.coilset[name].current / 1e6}")

# %%
# Find desired coilset psi for the centre of the plasma
# Use to calulate the factor needed to modify the constant term
# in our harminic equations

# TODO Clair - run this nb, add my 4 way plot at the end
#
# sof psi bdry value is a calculated value in the eudemo nb
# psi_sof/psi_eof is target psi value we want to achieve
# want to find min currents to achieve these psi bdry values

psi_bdry_coil_ref = ref_eq.coilset.psi(*ref_eq._x_points[0][:2])
psi_cntr_coil_ref = ref_eq.coilset.psi(*ref_eq._o_points[0][:2])
psi_bdry_plasma_ref = ref_eq.plasma.psi(*ref_eq._x_points[0][:2])

diff = psi_bdry_coil_ref - psi_cntr_coil_ref
sof_coil_cntr_psi = (
    psi_sof - psi_bdry_plasma_ref
) - diff  # psi_sof is a calculated value
eof_coil_cntr_psi = (psi_eof - psi_bdry_plasma_ref) - diff

sof_factor = sof_coil_cntr_psi / psi_cntr_coil_ref
eof_factor = eof_coil_cntr_psi / psi_cntr_coil_ref

print(f"{psi_cntr_coil_ref=}")
print(f"{sof_coil_cntr_psi=}")
print(f"{eof_coil_cntr_psi=}")
print(f"{sof_factor=}")
print(f"{eof_factor=}")

# %%
# Construct SOF and EOF from modified harmonics equtions obtained
# from TH approx of reference eq

harm_cos_term, harm_sin_term = toroidal_harmonics_to_positions(
    th_params=ref_result.th_params, n_allowed=setup["n_degrees_of_freedom"]
)
harm_cos_term = harm_cos_term[ref_result.cos_m, :]
harm_sin_term = harm_sin_term[ref_result.sin_m, :]

psi_calc_ref = harm_cos_term.T @ (ref_result.cos_amplitudes) + harm_sin_term.T @ (
    ref_result.sin_amplitudes
)

# TODO Clair modify zeroth harm amp by the factor from cell above

# Modify zeroth cos harmonic as this reprisents a constant
# We with to shift the psi value but keep shaping the same
sof_cos_amps, eof_cos_amps = (
    deepcopy(ref_result.cos_amplitudes),
    deepcopy(ref_result.cos_amplitudes),
)
sof_cos_amps[0] = sof_cos_amps[0] * sof_factor
eof_cos_amps[0] = eof_cos_amps[0] * eof_factor

psi_calc_sof = (
    harm_cos_term.T @ sof_cos_amps + harm_sin_term.T @ ref_result.sin_amplitudes
)
psi_calc_eof = (
    harm_cos_term.T @ eof_cos_amps + harm_sin_term.T @ ref_result.sin_amplitudes
)

# %%
from bluemira.equilibria.find import in_zone
from bluemira.equilibria.optimisation.harmonics.toroidal_harmonics_approx_functions import (
    _get_plasma_mask,
)

th_mask = _get_plasma_mask(ref_eq, ref_params, psi_norm=1.0, plasma_mask=True)
ref_mask = in_zone(
    ref_eq.grid.x, ref_eq.grid.z, ref_eq.get_LCFS().xz.T, include_edges=True
)

# %%
vmax = np.max([
    np.max(sof.coilset.psi(sof.x, sof.z) * ref_mask),
    np.max(eof.coilset.psi(eof.x, eof.z) * ref_mask),
    np.max(psi_calc_sof * th_mask.T),
    np.max(psi_calc_eof * th_mask.T),
])
vmin = np.min([
    np.min(sof.coilset.psi(sof.x, sof.z) * ref_mask),
    np.min(eof.coilset.psi(eof.x, eof.z) * ref_mask),
    np.min(psi_calc_sof * th_mask.T),
    np.min(psi_calc_eof * th_mask.T),
])
print(vmin, vmax)

# %%
sof_vmax = np.max([
    np.max(sof.coilset.psi(sof.x, sof.z) * ref_mask),
    np.max(psi_calc_sof * th_mask.T),
])
sof_vmin = np.min([
    np.min(sof.coilset.psi(sof.x, sof.z) * ref_mask),
    np.min(psi_calc_sof * th_mask.T),
])
eof_vmax = np.max([
    np.max(eof.coilset.psi(eof.x, eof.z) * ref_mask),
    np.max(psi_calc_eof * th_mask.T),
])
eof_vmin = np.min([
    np.min(eof.coilset.psi(eof.x, eof.z) * ref_mask),
    np.min(psi_calc_eof * th_mask.T),
])
print(sof_vmax, sof_vmin)
print(eof_vmax, eof_vmin)

# %%
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

nlevels = 40
cmap = PLOT_DEFAULTS["psi"]["cmap"]
f, axs = plt.subplots(1, 4)

im = axs[0].contourf(
    sof.x,
    sof.z,
    sof.coilset.psi(sof.x, sof.z) * ref_mask,
    levels=nlevels,
    cmap=cmap,
    vmin=sof_vmin,
    vmax=sof_vmax,
)
axs[0].set_title("SOF reference")
cax = make_axes_locatable(axs[0]).append_axes("right", size="5%", pad="2%")
cbar = plt.colorbar(mappable=im, cax=cax, ticks=np.linspace(vmin, vmax, 10))

axs[1].contourf(
    ref_result.th_params.R,
    ref_result.th_params.Z,
    psi_calc_sof.T * th_mask,
    levels=nlevels,
    cmap=cmap,
    vmin=sof_vmin,
    vmax=sof_vmax,
)
axs[1].set_title("SOF reconstructed")
cax = make_axes_locatable(axs[1]).append_axes("right", size="5%", pad="2%")
cbar = plt.colorbar(mappable=im, cax=cax, ticks=np.linspace(vmin, vmax, 10))

axs[2].contourf(
    eof.x,
    eof.z,
    eof.coilset.psi(eof.x, eof.z) * ref_mask,
    levels=nlevels,
    cmap=cmap,
    vmin=eof_vmin,
    vmax=eof_vmax,
)
axs[2].set_title("EOF reference")
cax = make_axes_locatable(axs[2]).append_axes("right", size="5%", pad="2%")
cbar = plt.colorbar(mappable=im, cax=cax, ticks=np.linspace(vmin, vmax, 10))

axs[3].contourf(
    ref_result.th_params.R,
    ref_result.th_params.Z,
    psi_calc_eof.T * th_mask,
    levels=nlevels,
    cmap=cmap,
    vmin=eof_vmin,
    vmax=eof_vmax,
)
axs[3].set_title("EOF reconstructed")
cax = make_axes_locatable(axs[3]).append_axes("right", size="5%", pad="2%")
cbar = plt.colorbar(mappable=im, cax=cax, ticks=np.linspace(vmin, vmax, 10))

axs[0].set_aspect("equal")
axs[1].set_aspect("equal")
axs[2].set_aspect("equal")
axs[3].set_aspect("equal")

# %%

sof_result = deepcopy(ref_result)
sof_result.cos_amplitudes[0] = ref_result.cos_amplitudes[0] * sof_factor
eof_result = deepcopy(ref_result)
eof_result.cos_amplitudes[0] = ref_result.cos_amplitudes[0] * eof_factor

sof_th_constraint = ToroidalHarmonicConstraint(
    th_result=sof_result,
    constraint_type="equality",
    relative_tolerance_cos=2e-3,
    relative_tolerance_sin=2e-3,
)
eof_th_constraint = ToroidalHarmonicConstraint(
    th_result=eof_result,
    constraint_type="equality",
    relative_tolerance_cos=2e-3,
    relative_tolerance_sin=2e-3,
)


# %%
# TODO Add psi bdry cons, and remove the field null cons
lcfs = ref_eq.get_LCFS()
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
psi_bdry_sof = PsiBoundaryConstraint(
    x=x_extrema, z=z_extrema, target_value=psi_sof, tolerance=0.5
)

psi_bdry_eof = PsiBoundaryConstraint(
    x=x_extrema, z=z_extrema, target_value=psi_eof, tolerance=0.5
)
# 1e-3 * np.abs(psi_eof)

# %%

_os, xs = ref_eq.get_OX_points()
x_point_constraint = FieldNullConstraint(
    xs[0].x,
    xs[0].z,
    tolerance=1e-3,
)
f, ax = plt.subplots()
x_point_constraint.plot(ax)
ref_eq.plot(ax)
# use an x point constraint
# %%
# f, ax = plt.subplots(5,2)
# for axs in ax[:][1]:
#     axs.axis("off")

# plot separately otherwise figures are too small

# uncon, sof th, eof th
sof_th_opt_eq = deepcopy(ref_eq)
sof_th_opt_eq.coilset.control = ref_result.th_params.th_coil_names

eof_th_opt_eq = deepcopy(ref_eq)
eof_th_opt_eq.coilset.control = ref_result.th_params.th_coil_names

f, axs = plt.subplots(2, 3)

for ax in axs[1][:]:
    ax.axis("off")

ref_eq.plot(axs[0][0])

axs[0][0].set_title("Starting unconstrained eq")

# TODO make sure PicardIterator is set up in the same way as in my other nb
# SOF + TH
sof_th_opt_problem = MinimalCurrentCOP(
    sof_th_opt_eq,
    opt_algorithm="SLSQP",
    constraints=[sof_th_constraint, x_point_constraint],
    opt_conditions={"max_eval": 1000, "ftol_rel": 1e-6},
    max_currents=3e10,
)

program = PicardIterator(
    sof_th_opt_problem,
    fixed_coils=True,
    convergence=DudsonConvergence(5e-3),
    relaxation=0.2,
    maxiter=50,
)
result = program()

sof_th_opt_eq.plot(axs[0][1])
axs[0][1].set_title("SOF + TH constraints")
axs[1][1].annotate(f"Converged: {program.check_converged()}", (0.1, 0.8))
axs[1][1].annotate(f"FOM: {result.f_x:.3f}", (0.1, 0.7))
axs[1][1].annotate(f"Iterations: {result.n_evals}", (0.1, 0.6))
axs[1][1].annotate(
    f"delta_psi: {100 * program.convergence.progress[-1]:.3f}%", (0.1, 0.5)
)
axs[1][1].annotate(f"Cons satisfied: {result.constraints_satisfied}", (0.1, 0.4))


# EOF + TH
eof_th_opt_problem = MinimalCurrentCOP(
    eof_th_opt_eq,
    opt_algorithm="SLSQP",
    constraints=[sof_th_constraint, x_point_constraint],
    opt_conditions={"max_eval": 1000, "ftol_rel": 1e-6},
    max_currents=3e10,
)

program = PicardIterator(
    eof_th_opt_problem,
    fixed_coils=True,
    convergence=DudsonConvergence(5e-3),
    relaxation=0.2,
    maxiter=50,
)
result = program()

eof_th_opt_eq.plot(axs[0][2])
axs[0][2].set_title("EOF + TH constraints")
axs[1][2].annotate(f"Converged: {program.check_converged()}", (0.1, 0.8))
axs[1][2].annotate(f"FOM: {result.f_x:.3f}", (0.1, 0.7))
axs[1][2].annotate(f"Iterations: {result.n_evals}", (0.1, 0.6))
axs[1][2].annotate(
    f"delta_psi: {100 * program.convergence.progress[-1]:.3f}%", (0.1, 0.5)
)
axs[1][2].annotate(f"Cons satisfied: {result.constraints_satisfied}", (0.1, 0.4))

# %%
# uncon, sof + core, eof + core

sof_core_opt_eq = deepcopy(ref_eq)
sof_core_opt_eq.coilset.control = ref_result.th_params.th_coil_names

eof_core_opt_eq = deepcopy(ref_eq)
eof_core_opt_eq.coilset.control = ref_result.th_params.th_coil_names

f, axs = plt.subplots(2, 3)
for ax in axs[1][:]:
    ax.axis("off")

ref_eq.plot(axs[0][0])
axs[0][0].set_title("Starting unconstrained eq")


# SOF + psi bdry
sof_core_opt_problem = MinimalCurrentCOP(
    sof_core_opt_eq,
    opt_algorithm="SLSQP",
    constraints=[psi_bdry_sof, x_point_constraint],
    opt_conditions={"max_eval": 1000, "ftol_rel": 1e-6},
    max_currents=3e10,
)

program = PicardIterator(
    sof_core_opt_problem,
    fixed_coils=True,
    convergence=DudsonConvergence(5e-3),
    relaxation=0.2,
    maxiter=50,
)
result = program()

sof_core_opt_eq.plot(axs[0][1])
axs[0][1].set_title("SOF + psi boundary constraint")
axs[1][1].annotate(f"Converged: {program.check_converged()}", (0.1, 0.8))
axs[1][1].annotate(f"FOM: {result.f_x:.3f}", (0.1, 0.7))
axs[1][1].annotate(f"Iterations: {result.n_evals}", (0.1, 0.6))
axs[1][1].annotate(
    f"delta_psi: {100 * program.convergence.progress[-1]:.3f}%", (0.1, 0.5)
)
axs[1][1].annotate(f"Cons satisfied: {result.constraints_satisfied}", (0.1, 0.4))


# EOF + psi bdry
eof_core_opt_problem = MinimalCurrentCOP(
    eof_core_opt_eq,
    opt_algorithm="SLSQP",
    constraints=[psi_bdry_eof, x_point_constraint],
    opt_conditions={"max_eval": 1000, "ftol_rel": 1e-6},
    max_currents=3e10,
)

program = PicardIterator(
    eof_core_opt_problem,
    fixed_coils=True,
    convergence=DudsonConvergence(5e-3),
    relaxation=0.2,
    maxiter=50,
)
result = program()

eof_core_opt_eq.plot(axs[0][2])
axs[0][2].set_title("EOF + psi boundary constraint")
axs[1][2].annotate(f"Converged: {program.check_converged()}", (0.1, 0.8))
axs[1][2].annotate(f"FOM: {result.f_x:.3f}", (0.1, 0.7))
axs[1][2].annotate(f"Iterations: {result.n_evals}", (0.1, 0.6))
axs[1][2].annotate(
    f"delta_psi: {100 * program.convergence.progress[-1]:.3f}%", (0.1, 0.5)
)
axs[1][2].annotate(f"Cons satisfied: {result.constraints_satisfied}", (0.1, 0.4))


# TODO do for EOF

# original eq, sof th, eof th, sof psi bdry, eof psi bdry",
