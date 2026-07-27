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
# %%
from bluemira.equilibria.find_legs import LegFlux
from copy import deepcopy
from dataclasses import Field
from pathlib import Path
from weakref import ref

from bluemira.equilibria.optimisation.constraints import (
    FieldNullConstraint,
    IsofluxConstraint,
    MagneticConstraint,
    MagneticConstraintSet,
)
from bluemira.equilibria.optimisation.problem._tikhonov import TikhonovCurrentCOP
from bluemira.equilibria.plotting import PLOT_DEFAULTS
from bluemira.equilibria.solve import (
    ConvergenceCriterion,
    DudsonConvergence,
    PicardIterator,
)
import numpy as np
import matplotlib.pyplot as plt

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimisation.harmonics.harmonics_constraints import (
    ToroidalHarmonicConstraint,
)
from bluemira.equilibria.optimisation.harmonics.toroidal_harmonics_approx_functions import (  # noqa: E501
    TauLimit,
    plot_toroidal_harmonic_approximation,
    toroidal_harmonic_approximation,
    toroidal_harmonic_grid_and_coil_setup,
)

# %%
ref_eq = Equilibrium.from_eqdsk(
    "REF_EUDEMO.json", from_cocos=3, qpsi_positive=False, force_symmetry=True
)
file_path = Path(get_bluemira_path("equilibria/data", subfolder="examples"), "SOF.json")

ref_eq = Equilibrium.from_eqdsk(file_path, from_cocos=3, qpsi_positive=False)


ref_eq.plot()
# %%
# TH approx
# these first 3 lines not needed as theyre the defaults in the approx fn
# but keeping here so can edit if needed
# TODO
psi_norm = 0.95
R_0, Z_0 = ref_eq.effective_centre()
th_params = toroidal_harmonic_grid_and_coil_setup(
    eq=ref_eq, R_0=R_0, Z_0=Z_0, tau_limit=TauLimit.COIL
)

# toroidal harmonic approximation
result = toroidal_harmonic_approximation(
    eq=ref_eq,
    th_params=th_params,
    psi_norm=psi_norm,
    n_degrees_of_freedom=6,
    max_harmonic_mode=5,
    plasma_mask=True,
)

# print info and plot
print(f"Cos modes used = {result.cos_m}")
print(f"Sin modes used = {result.sin_m}")
# plot to compare th approx psi to bm psi
f, ax = plot_toroidal_harmonic_approximation(
    eq=ref_eq, th_params=th_params, result=result, psi_norm=psi_norm
)
ax.set_title("COIL")
ref_eq.coilset.plot(ax)
plt.show()


# %%
# make constraint

th_constraint = ToroidalHarmonicConstraint(
    th_result=result,
    constraint_type="equality",
    relative_tolerance_cos=1e-6,
    relative_tolerance_sin=1e-6,
)
ref_eq.coilset.control = list(th_params.th_coil_names)

print(f"control coils = {ref_eq.coilset.control}")
# Show the constraint region
f, ax = plt.subplots()
th_constraint.plot(ax=ax)
ref_eq.coilset.plot(ax=ax)
ref_eq.plot(ax=ax)


# %%
# try to move leg

lcfs = ref_eq.get_LCFS()
x_bdry, z_bdry = lcfs.x, lcfs.z
arg_inner = np.argmin(x_bdry)


# x points
os, xs = ref_eq.get_OX_points()
x_point = FieldNullConstraint(xs[0].x, xs[0].z, tolerance=1e-3)
x_point_1 = FieldNullConstraint(xs[1].x, xs[1].z, tolerance=1e-3)


def get_leg_for_iso(ref_eq, leg_choice="both"):
    legs = LegFlux(ref_eq).get_legs()
    if leg_choice == "outer":
        leg_x = legs["lower_outer"][0].x
        leg_z = legs["lower_outer"][0].z
    elif leg_choice == "inner":
        leg_x = legs["lower_inner"][0].x
        leg_z = legs["lower_inner"][0].z
    elif leg_choice == "both":
        leg_x = np.append(legs["lower_inner"][0].x, legs["lower_outer"][0].x, axis=0)
        leg_z = np.append(legs["lower_inner"][0].z, legs["lower_outer"][0].z, axis=0)

    return leg_x[0::5], leg_z[0::5]


in_leg_x, in_leg_z = get_leg_for_iso(ref_eq, "inner")
out_leg_x, out_leg_z = get_leg_for_iso(ref_eq, "outer")

in_leg_x = in_leg_x[5:]
in_leg_z = in_leg_z[5:]

# new inner leg - first try
# in_leg_z[0] += 0.1
# in_leg_z[1] += 0.3
# in_leg_z[2] += 0.5
# in_leg_z[3] += 0.8
# in_leg_z[4] += 1.2

# more movement
in_leg_z[0] += 0.1 * 2 + 0.3
in_leg_z[1] += 0.3 * 2 + 0.3
in_leg_z[2] += 0.5 * 2 + 0.3
in_leg_z[3] += 0.8 * 2 + 0.3
# in_leg_z[4] += 1.2 * 2


out_leg_x = out_leg_x[4:]
out_leg_z = out_leg_z[4:]


ref_lcfs = ref_eq.get_LCFS()
arg_inner = np.argmin(ref_lcfs.x)
inner_leg = IsofluxConstraint(
    in_leg_x, in_leg_z, ref_lcfs.x[arg_inner], ref_lcfs.z[arg_inner], tolerance=1e-3
)
outer_leg = IsofluxConstraint(
    out_leg_x, out_leg_z, ref_lcfs.x[arg_inner], ref_lcfs.z[arg_inner], tolerance=1e-3
)
f, ax = plt.subplots()
ref_eq.plot(ax)
inner_leg.plot(ax)
outer_leg.plot(ax)

# leg_x[-1] = leg_x[-1] + 1.0
# leg_x[-2] = leg_x[-2] + 0.5
# leg_x[-3] = leg_x[-3] + 0.3

# %%
# TODO
# Modify inner leg abd remove points isoflux from near x-point

# inner move left-most 5, remove middle, then right-most 4 keep in
# TODO try with isoflux pts first, then try with th

# legs_mod = IsofluxConstraint(
#     leg_x,
#     leg_z,
#     ref_lcfs.x[arg_inner],
#     ref_lcfs.z[arg_inner],
#     constraint_value=0.0,
# )
# f, ax = plt.subplots()
# legs_mod.plot(ax)
# ref_eq.plot(ax)
# ref_eq.coilset.plot(ax)

# %%
# # new legs

# # unmoved inner:
# inner_xs = np.array([5.0, 5.1, 5.8, 6.2, 6.6, 7.0])
# inner_zs = np.array([-8.5, -7.8, -6.7, -6.4, -6.0, -5.8])

# # unmoved outer:
# outer_xs = np.array([8.0, 8.4, 8.8, 9.2, 9.6])
# outer_zs = np.array([-6.0, -6.6, -7.3, -8.0, -8.9])

# # moved outer:
# outer_xs = np.array([8.2, 8.8, 9.4, 10.2, 10.8])
# outer_zs = np.array([-6.0, -6.6, -7.3, -8.0, -8.9])

# inner_leg = IsofluxConstraint(
#     inner_xs,
#     inner_zs,
#     ref_x=x_bdry[arg_inner],
#     ref_z=z_bdry[arg_inner],
#     tolerance=1e-3,
# )

# outer_leg = IsofluxConstraint(
#     outer_xs,
#     outer_zs,
#     ref_x=x_bdry[arg_inner],
#     ref_z=z_bdry[arg_inner],
#     tolerance=1e-3,
# )

f, ax = plt.subplots()
ref_eq.plot(ax)
# inner_leg.plot(ax)
# outer_leg.plot(ax)
x_point.plot(ax)
x_point_1.plot(ax)

x_xbdry = np.array(x_bdry)[::15]
z_zbdry = np.array(z_bdry)[::15]

isoflux = IsofluxConstraint(
    x_xbdry,
    z_zbdry,
    x_xbdry[0],
    z_zbdry[0],
    tolerance=1e-3,
    # constraint_value=0.25,  # Difficult to choose...
)


# %%
# %%
th_move_legs_eq = deepcopy(ref_eq)

th_move_legs = TikhonovCurrentCOP(
    eq=th_move_legs_eq,
    targets=MagneticConstraintSet([
        inner_leg,
        outer_leg,
    ]),
    constraints=[
        th_constraint,
        # isoflux,
        # x_point,
    ],
    gamma=1e-8,
)

program = PicardIterator(
    th_move_legs,
    fixed_coils=True,
    convergence=DudsonConvergence(limit=1e-3),
    relaxation=0.2,
    maxiter=100,
    check_constraints=True,
)

program()

f, ax = plt.subplots()
th_move_legs_eq.plot(ax)
th_move_legs_eq.coilset.plot(ax)
inner_leg.plot(ax)
outer_leg.plot(ax)
# outer_leg.plot(ax)
ax.set_aspect("equal")


f, axs = plt.subplots(1, 2)
ref_eq.plot(axs[0])
inner_leg.plot(axs[0])
outer_leg.plot(axs[0])
x_point.plot(axs[0])
axs[0].set_title("Starting eq")

th_move_legs_eq.plot(axs[1])
inner_leg.plot(axs[1])
outer_leg.plot(axs[1])
x_point.plot(axs[1])
axs[1].set_title("Eq after optimisation")


# %%
# Plot coilset psi before and after:
nlevels = PLOT_DEFAULTS["psi"]["nlevels"]
cmap = PLOT_DEFAULTS["psi"]["cmap"]

f, axs = plt.subplots(1, 2)
og_coil_psi = ref_eq.coilset.psi(th_params.R, th_params.Z)
axs[0].contourf(th_params.R, th_params.Z, og_coil_psi, cmap=cmap, levels=nlevels)
axs[0].set_title("Starting eq coilset psi")
axs[0].set_aspect("equal")

new_coil_psi = th_move_legs_eq.coilset.psi(th_params.R, th_params.Z)
axs[1].contourf(th_params.R, th_params.Z, new_coil_psi, cmap=cmap, levels=nlevels)
axs[1].set_title("Coilset psi after optimisation")
axs[1].set_aspect("equal")

f.suptitle("Coilset psi before and after", y=0.8)

# Plot coilset psi before and after:
nlevels = PLOT_DEFAULTS["psi"]["nlevels"]
cmap = PLOT_DEFAULTS["psi"]["cmap"]

f, axs = plt.subplots(1, 2)
og_plasma_psi = ref_eq.plasma.psi(th_params.R, th_params.Z)
axs[0].contourf(th_params.R, th_params.Z, og_plasma_psi, cmap=cmap, levels=nlevels)
axs[0].set_title("Starting eq plasma psi")
axs[0].set_aspect("equal")

new_plasma_psi = th_move_legs_eq.plasma.psi(th_params.R, th_params.Z)
axs[1].contourf(th_params.R, th_params.Z, new_plasma_psi, cmap=cmap, levels=nlevels)
axs[1].set_title("Plasma psi after optimisation")
axs[1].set_aspect("equal")
f.suptitle("Plasma psi before and after", y=0.8)


# %%
