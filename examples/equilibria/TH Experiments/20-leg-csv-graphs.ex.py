# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

from bluemira.utilities.plot_tools import save_figure

# %%
# comparing 0.005
th_df = pd.read_csv("th_cons_opt_0.005_eq_params.csv")
th_df

core_con_df = pd.read_csv("core_cons_opt_0.005_eq_params.csv")
core_con_df

psi_bdry_df = pd.read_csv("psi_bdry_cons_opt_0.005_eq_params.csv")
psi_bdry_df

# %%
# save as dict of arrays
th_dict = {}
for col in th_df:
    th_dict[col] = np.asarray(th_df[col])

core_con_dict = {}
for col in core_con_df:
    core_con_dict[col] = np.asarray(core_con_df[col])

psi_bdry_dict = {}
for col in psi_bdry_df:
    psi_bdry_dict[col] = np.asarray(psi_bdry_df[col])


# %%
# pt placement on x, params on y

# for category, plot for each dict
folder = "pt_placement_vs_parameters"
if not Path(folder).exists():
    Path.mkdir(folder)


for category in th_dict:
    if category == "pt_placement":
        continue
    # ignore 1st entry as this is reference
    th_cat_to_plot = th_dict[category][1:]
    core_cat_to_plot = core_con_dict[category][1:]
    psi_bdry_cat_to_plot = psi_bdry_dict[category][1:]

    f, ax = plt.subplots()
    ax.set_xticks(th_dict["pt_placement"][1:])
    ax.scatter(th_dict["pt_placement"][1:], th_cat_to_plot, color="red", label="TH")
    ax.scatter(
        core_con_dict["pt_placement"][1:],
        core_cat_to_plot,
        color="blue",
        label="core con",
    )
    ax.scatter(
        psi_bdry_dict["pt_placement"][1:],
        psi_bdry_cat_to_plot,
        color="green",
        label="psi bdry",
    )
    ax.set_xlabel("point placement")
    ax.set_ylabel(category)
    f.suptitle(f"{category} vs point placement")
    f.legend(loc="outside right")

    name = "pt_placement_vs_" + category
    save_figure(fig=f, name=name, save=True, folder=folder)


# %%
for data, title in zip(
    [th_dict, core_con_dict, psi_bdry_dict],
    ["th_dict", "core_con_dict", "psi_bdry_dict"],
    strict=True,
):
    x_pts_x = data["x_pt_x"]

    og_x_pt_x = x_pts_x[0]

    delta_x_pts = [og_x_pt_x - pt for pt in x_pts_x[1:]]

    fom = data["fom"]
    fom = np.array(fom[1:])

    colors = cm.rainbow(np.linspace(0, 1, len(fom)))
    f, ax = plt.subplots()
    for i in range(len(fom)):
        ax.scatter(delta_x_pts[i], fom[i], color=colors[i], label=i + 1)
    ax.set_xlabel("change in x point x coord")
    ax.set_ylabel("FOM")
    f.legend(loc="outside right")

    f.suptitle(f"FOM vs change in x coord of x point {title}")

    name = "delta_x_vs_FOM_" + title
    save_figure(fig=f, name=name, save=True, folder=folder)
