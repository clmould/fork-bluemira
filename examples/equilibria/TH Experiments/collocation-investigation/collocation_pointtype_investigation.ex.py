# %%
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from bluemira.base.file import get_bluemira_path
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

# investigating the different collocation options

collocation_options = [
    PointType.ARC,
    PointType.ARC_PLUS_EXTREMA,
    PointType.RANDOM,
    PointType.RANDOM_PLUS_EXTREMA,
    # PointType.GRID_POINTS,
    PointType.INNER_FS,
    PointType.WONKY_PLUS,
    PointType.DOUBLE_INNER_FS,
]

# %%

# how many default points do the different options have?
# test with n_points=4
n_points = 6

psi_norm_inner = 0.65
f, axs = plt.subplots(4, 2)

ref_lcfs = ref_eq.get_LCFS()

for i, collocation_option in enumerate(collocation_options):
    # probably a nicer way of doing this ! just wanted some quick plots tho
    # if collocation_option == PointType.DOUBLE_INNER_FS:
    #     collocation = collocation_points(
    #         ref_eq.get_LCFS(),
    #         collocation_option,
    #         n_points=n_points,
    #         psi_norm_inner=psi_norm_inner,
    #         eq=ref_eq,
    #         second_inner=True,
    #     )
    # else:
    collocation = collocation_points(
        ref_eq.get_LCFS(),
        collocation_option,
        n_points=n_points,
        psi_norm_inner=psi_norm_inner,
        eq=ref_eq,
    )
    j = 0 if i <= 3 else 1
    axs[i % 4][j].scatter(collocation.x, collocation.z, zorder=10)
    # ref_eq.plot(axs[i % 4][j])
    axs[i % 4][j].plot(ref_lcfs.x, ref_lcfs.z, color="r")
    axs[i % 4][j].set_title(f"{collocation_option}")
    axs[i % 4][j].set_aspect("equal")
# for ax in axs:
#     ref_eq.plot(ax)
f.suptitle(f"{n_points=} for all PointType options")
f.tight_layout()

plt.show()


# %%
# setup for running - just a test atm, it fails for GRID_POINTS :/

n_colloc_pts = 9

# TODO do with 13 points total (like in other nb)
# TODO save err, res, matr cond number
n_colloc_pts = 13

# OR can create a grid then set the others to match up with # of pts
# we get for the grid

# has changing pt distrib actually done anything? and what specifically?
# do we need to sample all the different aspect, eg some in middle + some
# on outside some on bdry etc want to know gradient and shaping too of FSs?
#

# need to get same total number of points for each PointType
# NOTE: will have to do GRID_POINTS manually as there is no
# clear relation between points selected and n_points, eg
# n_points = 4 has 4 points chosen, n_points=5 has 9, n_points=6 has 15

# ARC chooses n_pts
# ARC_PLUS_EXTREMA chooses n_pts + 4
# RANDOM chooses n_pts
# RANDOM_PLUS_EXTREMA chooses n_pts + 4
# GRID doesn't have a clear relation, need to sort manually
# INNER_FS chooses n_pts + 5
# WONKY_PLUS chooses 4 * n_pts + 5
# DOUBLE_INNER_FS chooses 2 * n_pts + 5

grid_points = 6  # TODO must manually choose this, can't see a
# clear way to know how many points it will select?

# TODO 2 sets of comparisons
# 1 where we compare all apart from grid with 13 pts
# 1 where compare all apart from wonky plus with 15 pts
# cant get grid with 13 or wonky plus with 15 :'(

n_pts_dict = {
    PointType.ARC: n_colloc_pts,
    PointType.ARC_PLUS_EXTREMA: n_colloc_pts - 4,
    PointType.RANDOM: n_colloc_pts,
    PointType.RANDOM_PLUS_EXTREMA: n_colloc_pts - 4,
    # PointType.GRID_POINTS: grid_points,
    PointType.INNER_FS: n_colloc_pts - 5,
    PointType.WONKY_PLUS: (n_colloc_pts - 5) / 4,
    PointType.DOUBLE_INNER_FS: (n_colloc_pts - 5) / 2,
}

results_dict = {}

for collocation_option in collocation_options:
    collocation = collocation_points(
        ref_eq.get_LCFS(),
        collocation_option,
        n_points=n_pts_dict[collocation_option],
        psi_norm_inner=psi_norm_inner,
        eq=ref_eq,
    )
    f, ax = plt.subplots()
    ax.scatter(collocation.x, collocation.z, zorder=10)
    ax.plot(ref_lcfs.x, ref_lcfs.z, color="r")
    ax.set_aspect("equal")

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
    results_dict[str(collocation_option)] = th_result
    f, ax = plot_toroidal_harmonic_approximation(
        eq=ref_eq, th_params=th_params, result=th_result, psi_norm=psi_norm
    )
    ax.set_title(f"using {collocation_option}")
    ref_eq.coilset.plot(ax)
    ax.set_aspect("equal")
    plt.show()
