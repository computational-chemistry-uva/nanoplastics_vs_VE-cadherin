import numpy as np
import matplotlib.pyplot as plt
from buq import BQConfig, BayesianQuadratureRunner
from cadherin_setup import Cadherin
import os
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 12,          # base font size
    "axes.titlesize": 14,     # axes titles
    "axes.labelsize": 13,     # x/y labels
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
})
# ----------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------
# chains excluded from training, used only for blind tests
blind_chains = [10.0, 20.0]
# chains whose FES define the objective
val_chains   = [0.0, 40.0]

path = rf"figures/"

os.makedirs(path + "derslices", exist_ok=True)
os.makedirs(path + "fesslices", exist_ok=True)
os.makedirs(path + "fits", exist_ok=True)
os.makedirs(path + "ders", exist_ok=True)

results_file = os.path.join(path, "optuna_trial_results.csv")
with open(results_file, "w") as f:
    f.write(
        "trial,kernel,lengthscale1,lengthscale2,noise,"
        "rmsd_0,rmsd_40,totalrmsd,blind_fes_rmsd,blind_grad_rmsd\n"
    )

# CV2 (com) range for FES RMSD
com_min, com_max = 2.1, 6.6

# ----------------------------------------------------------------------
# PLOTTING HELPERS
# ----------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_fes_slices_at_chain_values(
    runner,
    chain_values,
    fes_refs=None,
    val_chains=None,
    blind_chains=None,
    com_min=None,
    com_max=None,
    savepath=None,
    show=True,
):
    """
    For each fixed chain value (CV1 = x), plot:
      - GP FES slice A(com) along CV2
      - OPTIONAL: reference FES curves (dashed) for chains
        that appear in fes_refs. Chains in val_chains and blind_chains
        are part of the objective / blind test.
    """

    # fixed color mapping to match your legend
    chain_color_map = {
        0.0:  "green",      # "No PS"
        10.0: "coral",      # "PS10"
        20.0: "red",        # "PS20"
        40.0: "brown",      # "PS40"
        # if you also plot "No PS, no Ca2+" here, you can add e.g.:
        # -1.0: "0.5",      # gray
    }
    tol_chain_color = 0.6  # tolerance to match float CVs to keys

    X_grid = runner.X_grid_2d   # (nx, ny)
    Y_grid = runner.Y_grid_2d   # (nx, ny)
    fes_2d = runner.current_fes_2d  # (nx, ny)

    chain_grid = X_grid[:, 0]   # (nx,)
    com_grid   = Y_grid[0, :]   # (ny,)

    fig, ax = plt.subplots(figsize=(9, 5))

    for k, cv1_val in enumerate(chain_values):
        # Snap to nearest grid index along CV1
        i = int(np.argmin(np.abs(chain_grid - cv1_val)))
        actual_cv1 = chain_grid[i]

        # choose color based on chain value
        # (fall back to a default if not in map)
        chain_key = min(chain_color_map.keys(), key=lambda c: abs(c - actual_cv1))
        if abs(chain_key - actual_cv1) < tol_chain_color:
            color = chain_color_map[chain_key]
        else:
            color = "black"  # or any default

        # FES slice (GP)
        fes_slice = fes_2d[i, :]
        fes_slice_shifted = fes_slice - np.min(fes_slice)

        # labels
        if np.isclose(actual_cv1, 0.0):
            base_label = "No PS"
        else:
            base_label = f"PS{actual_cv1:.0f}"

        label_gp = base_label
        if blind_chains is not None and any(np.isclose(actual_cv1, bc) for bc in blind_chains):
            label_gp += " (test)"

        ax.plot(com_grid, fes_slice_shifted, color=color, label=label_gp)

        # Overlay reference FES, if available (same color, dashed, no legend entry)
        if fes_refs is not None and len(fes_refs) > 0:
            ref_chain = min(fes_refs.keys(), key=lambda c: abs(c - actual_cv1))
            if abs(ref_chain - actual_cv1) < 0.6:  # tolerance
                com_ref, A_ref = fes_refs[ref_chain]
                if com_min is not None and com_max is not None:
                    mask = (com_ref >= com_min) & (com_ref <= com_max)
                    com_ref_plot = com_ref[mask]
                    A_ref_plot   = A_ref[mask]
                else:
                    com_ref_plot = com_ref
                    A_ref_plot   = A_ref

                A_ref_plot = A_ref_plot - np.min(A_ref_plot)

                ax.plot(
                    com_ref_plot,
                    A_ref_plot,
                    linestyle="--",
                    color=color,
                    linewidth=2,
                    label=None,   # no extra legend entry
                )
                print(ref_chain, fes_slice[-1]-A_ref[-1])

    ax.set_xlabel("COM distance (nm)")
    ax.set_ylabel("Dissociation Free Energy (kcal/mol)")
    ax.set_xlim(com_min, com_max)
    ax.set_ylim(0, 70)
    ax.grid(True)
    plt.tight_layout()

    # legend from GP curves only
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncol=1)

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

# ----------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------
data = np.loadtxt("all_points_except_1.6.csv", delimiter=",", skiprows=1)
chain = data[:, 1].astype(float)   # CV1 (chain length)
com   = data[:, 2].astype(float)   # CV2
fy    = data[:, 3].astype(float)   # negative force
std   = data[:, 4].astype(float)   # std_force
fx = np.linspace(0, 1, len(fy))
grid_size_2d = (41, 10)
X_all = np.column_stack([chain, com])
Y_all = np.column_stack([fx, fy])
N = X_all.shape[0]

# --- define train / blind masks and indices ---
mask_blind = np.isin(chain, blind_chains)
train_mask = ~mask_blind

train_idx_fixed  = np.where(train_mask)[0]
blind_idx_fixed  = np.where(mask_blind)[0]

print(f"Train points: {len(train_idx_fixed)}, "
      f"blind points (chains {blind_chains}): {len(blind_idx_fixed)}")

bounds = (chain.min(), chain.max(), com.min(), com.max())
system = Cadherin(bounds)

# ----------------------------------------------------------------------
# FES REFERENCE CURVES FROM .dat FILES
# ----------------------------------------------------------------------
# Chains for which we expect reference FES; includes all val + blind chains
all_fes_chains = sorted(set(val_chains + blind_chains))

fes_refs = {}  # mapping: chain_value -> (com_ref, A_ref)
for ch in all_fes_chains:
    fname = f"chain_length_{int(ch)}.dat"
    if not os.path.isfile(fname):
        print(f"WARNING: reference FES file not found: {fname}")
        continue

    arr = np.loadtxt(fname, comments="#")
    com_ref = arr[:, 0]    # CV_Value(nm)
    A_ref   = arr[:, 3]    # Mean_Free_Energy(kcal/mol)

    mask_range = (com_ref >= com_min) & (com_ref <= com_max)
    fes_refs[ch] = (com_ref[mask_range], A_ref[mask_range])


kernel= "RBF"
lengthscale1=19.983680929020696
lengthscale2=0.48678238249941896
noise=0.2815430467860205





# ----------------------------------------------------------------------
# FINAL MODELS
# ----------------------------------------------------------------------
config_best = BQConfig(
    kernel_type=kernel,
    lengthscale=[lengthscale1, lengthscale2],
    noise=noise,
    variance=1.0,
    n_queries=0,
    grid_size_2d=grid_size_2d,
    use_mini=True,
    fast_mini=True,
    acq_function="IVR",
    gradient_components=[1],
)

# Model A: training only (no blind chains)
runner_train = BayesianQuadratureRunner(system=system, config=config_best)
runner_train.initialize_from_data(X_all[train_mask], Y_all[train_mask])


chain_values = [0,10,20,40]

is_train_train = train_mask



plot_fes_slices_at_chain_values(
    runner_train,
    chain_values=chain_values,
    fes_refs=fes_refs,
    val_chains=val_chains,
    blind_chains=blind_chains,
    com_min=com_min,
    com_max=com_max,
    savepath=(
        f"{path}final_fes_slices_train_only_"
        f"{kernel}_{lengthscale1}_"
        f"{lengthscale2}_{noise}.pdf"
    ),
    show=False,
)

# -------- Final plots: train + all data model --------
is_train_all = np.ones_like(chain, dtype=bool)





def plot_final_fes_and_slice_diff(
    runner,
    fes_refs,
    chain_values,
    com_min=None,
    com_max=None,
    com_tol=1e-3,
    X_obs=None,
    is_train=None,      # boolean mask over X_obs (True = training point)
    left_cbar_label=r"Dissociation Free Energy (kcal/mol)",
    right_cbar_label=r"ΔFree Energy between GP and Reference (kcal/mol)",
    savepath=None,
    show=True,
):
    """
    Two-panel figure for the paper:

      Left:  full 2D FES from `runner` + training points (from X_obs[is_train])
      Right: ΔF = GP - reference, only at selected chains.

    chain_values: list of chain lengths (e.g. [0.0, 10.0, 20.0, 40.0])
    fes_refs:     dict {chain_value -> (com_ref, A_ref)}
    X_obs:        array (N, 2) of (chain, com) observation locations
    is_train:     boolean array (N,) marking which X_obs are training points
    """

    # --- ensure FES is up to date ---
    if runner.current_fes_2d is None:
        runner._update_fes()

    X_grid = runner.X_grid_2d   # (nx, ny)
    Y_grid = runner.Y_grid_2d   # (nx, ny)
    fes_2d = runner.current_fes_2d

    chain_grid = X_grid[:, 0]   # (nx,)
    com_grid   = Y_grid[0, :]   # (ny,)

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    ax_left, ax_right = axes
    cmap_fes  = plt.cm.get_cmap("magma", 14)    # 10 discrete levels
    # ------------------------------------------------------------------
    # LEFT PANEL: full FES + training points
    # ------------------------------------------------------------------
    im_left = ax_left.pcolormesh(
        X_grid,
        Y_grid,
        fes_2d,
        shading="auto",
        vmin=0,
        vmax = 70,
        cmap=cmap_fes,
    )

    # overlay observation locations if provided
    if X_obs is not None:
        if is_train is None:
            # if no mask, plot all points the same
            ax_left.scatter(
                X_obs[:, 0], X_obs[:, 1],
                s=10, edgecolor="k", facecolor="none", linewidth=0.7,
                label="observations",
            )
        else:
            X_obs = np.asarray(X_obs)
            is_train = np.asarray(is_train, dtype=bool)
            X_train = X_obs[is_train]
            X_blind = X_obs[~is_train]

            if X_train.size > 0:
                ax_left.scatter(
                    X_train[:, 0], X_train[:, 1],
                    s=14, edgecolor="white", facecolor="none", linewidth=0.8,
                    label="training data",
                )
            # if X_blind.size > 0:
            #     ax_left.scatter(
            #         X_blind[:, 0], X_blind[:, 1],
            #         s=12, edgecolor="red", facecolor="none", linewidth=0.7,
            #         label="blind points",
            #     )

            ax_left.legend(loc= "upper center", frameon=True, fontsize=10)

    ax_left.set_xlabel("PS length")
    ax_left.set_ylabel("COM distance (nm)")
    cbar_left = fig.colorbar(im_left, ax=ax_left)
    cbar_left.set_label(left_cbar_label)

    # ------------------------------------------------------------------
    # RIGHT PANEL: ΔF = GP - reference on selected chains
    # ------------------------------------------------------------------
    # restrict to COM range for the diff plot
    if com_min is not None and com_max is not None:
        com_mask = (com_grid >= com_min) & (com_grid <= com_max)
    else:
        com_mask = np.ones_like(com_grid, dtype=bool)

    com_sel = com_grid[com_mask]

    X_sel    = X_grid[:, com_mask]
    Y_sel    = Y_grid[:, com_mask]
    Z_gp_full = fes_2d[:, com_mask]

    nx_sel, ny_sel = Z_gp_full.shape

    Z_ref_unshifted = np.full_like(Z_gp_full, np.nan, dtype=float)

    for ch in chain_values:
        if ch not in fes_refs:
            continue
        com_ref, A_ref = fes_refs[ch]

        # restrict reference to same com range
        if com_min is not None and com_max is not None:
            mask = (com_ref >= com_min) & (com_ref <= com_max)
            com_ref_plot = com_ref[mask]
            A_ref_plot   = A_ref[mask]
        else:
            com_ref_plot = com_ref
            A_ref_plot   = A_ref

        # nearest chain index in GP grid
        i_chain = int(np.argmin(np.abs(chain_grid - ch)))

        # align reference COM to com_sel
        for col, c in enumerate(com_sel):
            j = int(np.argmin(np.abs(com_ref_plot - c)))
            if np.abs(com_ref_plot[j] - c) > com_tol:
                continue
            Z_ref_unshifted[i_chain, col] = A_ref_plot[j]

    Z_diff = Z_gp_full - Z_ref_unshifted
    Z_diff_masked = np.ma.array(Z_diff, mask=np.isnan(Z_ref_unshifted))
    cmap_diff = plt.cm.get_cmap("bwr", 20)    # 9 discrete levels
    im_right = ax_right.pcolormesh(
        X_sel,
        Y_sel,
        Z_diff_masked,
        shading="auto",
        cmap=cmap_diff,
        vmin=-20,
        vmax=20
    )

    ax_right.set_xlabel("PS length")
    ax_right.set_ylabel("COM distance (nm)")
    cbar_right = fig.colorbar(im_right, ax=ax_right)
    cbar_right.set_label(right_cbar_label)

    # consistent x-limits
    x_min = -0.5
    x_max = 40.5
    for ax in axes:
        ax.set_xlim(x_min, x_max)

    fig.tight_layout()

    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


slice_chains = [0.0, 10.0, 20.0, 40.0]

savepath_paper = (
    f"{path}final_fes_and_diff_"
    f"{kernel}_{lengthscale1}_{lengthscale2}_{noise}.pdf"
)

plot_final_fes_and_slice_diff(
    runner_train,
    fes_refs=fes_refs,
    chain_values=slice_chains,
    com_min=com_min,
    com_max=com_max,
    X_obs=X_all,          # all (chain, com) points
    is_train=train_mask,  # True for training (canonical), False for blind
    savepath=savepath_paper,
    show=False,
)