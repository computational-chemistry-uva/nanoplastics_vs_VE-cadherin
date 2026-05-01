import numpy as np
import optuna
import matplotlib.pyplot as plt
import pickle
from buq import BQConfig, BayesianQuadratureRunner
from cadherin_setup import Cadherin
import os

# ----------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------
# chains excluded from training, used only for blind tests
blind_chains = [10.0, 20.0]
# chains whose FES define the objective
val_chains   = [0.0, 40.0]
kernel       = "RBF"


path = rf"results_optunasearch_RBF_all_points_blind{'_'.join(str(int(c)) for c in blind_chains)}_fes_obj_{'_'.join(str(int(c)) for c in val_chains)}/"
number_of_trials = 500  # number of optuna loops

os.makedirs(path + "derslices", exist_ok=True)
os.makedirs(path + "fesslices", exist_ok=True)
os.makedirs(path + "fes", exist_ok=True)
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
def plot_derivative_slices_at_chain_values(
    runner,
    chain_values,
    X_obs,
    Y_obs,
    std_obs,
    savepath=None,
    show=True,
    n_sigma=2,
    is_train=None,
):
    Y_obs   = np.asarray(Y_obs).ravel()
    std_obs = np.asarray(std_obs).ravel()

    chain_grid = runner.X_grid_2d[:, 0]   # (nx,)
    com_grid   = runner.Y_grid_2d[0, :]   # (ny,)

    cmap = plt.get_cmap("tab10")

    for k, cv1_val in enumerate(chain_values):
        color = cmap(k % 10)

        # --- snap to nearest grid index along chain ---
        i = int(np.argmin(np.abs(chain_grid - cv1_val)))
        actual_cv1 = chain_grid[i]

        # --- GP prediction along com at this chain value ---
        X_pred = np.column_stack([
            np.full_like(com_grid, actual_cv1),
            com_grid,
        ])                                          # (ny, 2)
        mean, var = runner.emukit_method.predict(X_pred)
        mean = np.asarray(mean).ravel()
        std  = np.sqrt(np.asarray(var).ravel())

        # --- observations at this chain value ---
        tol = 0.01
        obs_mask  = np.abs(X_obs[:, 0] - actual_cv1) <= tol
        com_obs   = X_obs[obs_mask, 1]
        fy_obs    = Y_obs[obs_mask]
        std_obs_s = std_obs[obs_mask]

        if is_train is not None:
            train_mask = is_train[obs_mask]
            com_train  = com_obs[train_mask]
            fy_train_s = fy_obs[train_mask]
            std_train_s= std_obs_s[train_mask]

            com_test   = com_obs[~train_mask]
            fy_test_s  = fy_obs[~train_mask]
            std_test_s = std_obs_s[~train_mask]
        else:
            com_train, fy_train_s, std_train_s = com_obs, fy_obs, std_obs_s
            com_test, fy_test_s, std_test_s    = [], [], []

        # --- plot ---
        fig, ax = plt.subplots(figsize=(9, 5))

        # GP uncertainty band
        ax.fill_between(
            com_grid,
            mean - n_sigma * std,
            mean + n_sigma * std,
            alpha=0.25,
            color=color,
            label=f"GP ±{n_sigma}σ",
        )

        # GP mean
        ax.plot(com_grid, mean, color=color, linewidth=2,
                label="GP mean (dA/dy)")

        # observations with error bars from CSV std
        ax.errorbar(
            com_train, fy_train_s,
            yerr=std_train_s,
            fmt="o",
            color="black",
            ecolor="gray",
            capsize=4,
            label="Train obs ± std",
        )

        # test / blind points
        if len(com_test) > 0:
            ax.errorbar(
                com_test, fy_test_s,
                yerr=std_test_s,
                fmt="s",
                color="red",
                ecolor="pink",
                capsize=4,
                label="Test/Blind obs ± std",
            )

        ax.set_xlabel("com / nm (CV2)")
        ax.set_ylabel("dA/dy  (negative force)")
        ax.set_title(f"Derivative slice at chain length = {actual_cv1:.3g}")
        ax.legend()
        ax.grid(True, alpha=0.4)
        plt.tight_layout()

        if savepath is not None:
            base, ext = savepath.rsplit(".", 1) if "." in savepath else (savepath, "png")
            plt.savefig(f"{base}_chain{actual_cv1:.3g}.{ext}", dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()


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
      - OPTIONAL: reference FES curves (same colour, dashed) for chains
        that appear in fes_refs. Chains in val_chains and blind_chains
        are part of the objective / blind test.
    """
    X_grid = runner.X_grid_2d   # (nx, ny)
    Y_grid = runner.Y_grid_2d   # (nx, ny)
    fes_2d = runner.current_fes_2d  # (nx, ny)

    chain_grid = X_grid[:, 0]   # (nx,)
    com_grid   = Y_grid[0, :]   # (ny,)

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")

    for k, cv1_val in enumerate(chain_values):
        color = cmap(k % 10)

        # Snap to nearest grid index along CV1
        i = int(np.argmin(np.abs(chain_grid - cv1_val)))
        actual_cv1 = chain_grid[i]

        # FES slice (GP)
        fes_slice = fes_2d[i, :]
        fes_slice_shifted = fes_slice - np.min(fes_slice)

        label_gp = f"GP chain ≈ {actual_cv1:.3f}"
        if val_chains is not None and any(np.isclose(actual_cv1, c) for c in val_chains):
            label_gp += " (objective)"
        if blind_chains is not None:
            if any(np.isclose(actual_cv1, bc) for bc in blind_chains):
                label_gp += " (blind)"

        ax.plot(com_grid, fes_slice_shifted, color=color, label=label_gp)

        # Overlay reference FES, if available
        if fes_refs is not None and len(fes_refs) > 0:
            # find closest key in fes_refs
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

                label_ref = f"ref {ref_chain:.0f}"
                if val_chains is not None and any(np.isclose(ref_chain, c) for c in val_chains):
                    label_ref += " (objective)"
                if blind_chains is not None:
                    if any(np.isclose(ref_chain, bc) for bc in blind_chains):
                        label_ref += " (blind)"

                ax.plot(
                    com_ref_plot,
                    A_ref_plot,
                    linestyle="--",
                    color=color,
                    linewidth=2,
                    label=label_ref,
                )

    ax.set_xlabel("com (CV2)")
    ax.set_ylabel("A(com | chain) — shifted to min=0")
    ax.set_title("FES slices at fixed chain values")
    ax.legend(ncol=2)
    ax.grid(True)
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

# ----------------------------------------------------------------------
# LOADING IN DATA
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

# ----------------------------------------------------------------------
# HELPER: FES RMSD FOR ONE CHAIN
# ----------------------------------------------------------------------
def fes_rmsd_for_chain(runner, chain_value, com_ref, A_ref, com_tol=1e-3):
    """
    Compute RMSD between runner's FES slice at fixed chain_value and
    a reference FES curve (com_ref, A_ref). Only points where com_ref
    matches the FES grid (within com_tol) are used.
    """
    chain_grid = runner.X_grid_2d[:, 0]   # (nx,)
    com_grid   = runner.Y_grid_2d[0, :]   # (ny,)
    fes_2d     = runner.current_fes_2d    # (nx, ny)

    # nearest chain index in the grid
    i_chain = int(np.argmin(np.abs(chain_grid - chain_value)))
    fes_slice = fes_2d[i_chain, :]        # (ny,)

    A_pred = []
    A_true = []

    for c, a in zip(com_ref, A_ref):
        j = int(np.argmin(np.abs(com_grid - c)))
        if np.abs(com_grid[j] - c) > com_tol:
            continue
        A_pred.append(fes_slice[j])
        A_true.append(a)

    A_pred = np.asarray(A_pred)
    A_true = np.asarray(A_true)

    if A_pred.size == 0:
        raise RuntimeError(
            f"No matching CV2 points between FES grid and reference for chain={chain_value}"
        )

    mse = np.mean((A_pred - A_true) ** 2)
    return np.sqrt(mse)

# ----------------------------------------------------------------------
# OPTUNA OBJECTIVE
# ----------------------------------------------------------------------
def objective(trial):
    lengthscale1 = trial.suggest_float("lengthscale1", 1, 20)
    lengthscale2 = trial.suggest_float("lengthscale2", 0.1, 5)
    noise        = trial.suggest_float("noise", 0.2, 1)

    train_idx = train_idx_fixed
    blind_idx = blind_idx_fixed

    X_train = X_all[train_idx]
    Y_train = Y_all[train_idx]

    config = BQConfig(
        kernel_type=kernel,
        lengthscale=[lengthscale1, lengthscale2],
        noise=noise,
        gradient_components=[1],
        variance=1.0,
        n_queries=0,
        grid_size_2d=grid_size_2d,
        use_mini=True,
        fast_mini=True,
        acq_function="IVR",
    )

    runner = BayesianQuadratureRunner(system=system, config=config)
    runner.initialize_from_data(X_train, Y_train)

    # ---------- FES RMSD on val_chains (objective) ----------
    rmsd_per_chain = {}
    for ch in val_chains:
        if ch not in fes_refs:
            raise RuntimeError(f"No reference FES loaded for chain={ch}")
        com_ref, A_ref = fes_refs[ch]
        rmsd_ch  = fes_rmsd_for_chain(runner, ch, com_ref, A_ref)
        rmsd_per_chain[ch] = rmsd_ch

    # here: total objective = sum of per-chain RMSDs
    val_fes_rmsd = sum(rmsd_per_chain.values())
    rmsd_0  = rmsd_per_chain.get(0.0, np.nan)
    rmsd_40 = rmsd_per_chain.get(40.0, np.nan)

    # ---------- Blind test: FES + gradient over all blind_chains ----------
    if blind_idx.size > 0:
        # gradient blind RMSD (over all blind points)
        X_blind  = X_all[blind_idx]
        Y_blind  = Y_all[blind_idx]
        fy_true_blind = Y_blind[:, 1]

        grad_pred_blind, _ = runner.predict_grad_at(X_blind)
        fy_pred_blind = grad_pred_blind[:, 0]
        blind_grad_rmsd = np.sqrt(np.mean((fy_pred_blind - fy_true_blind) ** 2))

        # FES blind RMSD: sum over blind chains
        blind_fes_rmsd = 0.0
        for ch in blind_chains:
            if ch in fes_refs:
                com_blind_ref, A_blind_ref = fes_refs[ch]
                blind_fes_rmsd += fes_rmsd_for_chain(
                    runner, ch, com_blind_ref, A_blind_ref
                )
            else:
                print(f"WARNING: no reference FES for blind chain {ch}")
    else:
        blind_grad_rmsd = np.nan
        blind_fes_rmsd  = np.nan

    # ---------- derivative / FES plots (train-only model) ----------
    runner.plot_derivatives(
        show=False,
        savepath=f"{path}ders/derivatives_train_only_{kernel}_{lengthscale1}_{lengthscale2}_{noise}.png"
    )
    runner.plot_fes(
        show=False,
        savepath=f"{path}fes/fes_train_only_{kernel}_{lengthscale1}_{lengthscale2}_{noise}.png"
    )
    chain_values = [0, 10, 20, 40]
    is_train = np.zeros_like(chain, dtype=bool)
    is_train[train_idx] = True

    plot_derivative_slices_at_chain_values(
        runner,
        chain_values=chain_values,
        X_obs=X_all,
        Y_obs=fy,
        std_obs=std,
        is_train=is_train,
        savepath=(
            f"{path}derslices/der_slices_train_only_"
            f"{kernel}_{lengthscale1}_{lengthscale2}_{noise}.png"
        ),
        show=False,
        n_sigma=2,
    )

    plot_fes_slices_at_chain_values(
        runner,
        chain_values=chain_values,
        fes_refs=fes_refs,
        val_chains=val_chains,
        blind_chains=blind_chains,
        com_min=com_min,
        com_max=com_max,
        savepath=(
            f"{path}fesslices/fes_slices_train_only_"
            f"{kernel}_{lengthscale1}_{lengthscale2}_{noise}.png"
        ),
        show=False,
    )

    # ---------- log parameters + scores to CSV ----------
    with open(results_file, "a") as f:
        f.write(
            f"{trial.number},{kernel},{lengthscale1},{lengthscale2},"
            f"{noise},{rmsd_0},{rmsd_40},{val_fes_rmsd},{blind_fes_rmsd},{blind_grad_rmsd}\n"
        )

    # Objective: ONLY validation FES RMSD (sum over val_chains)
    return val_fes_rmsd

# ----------------------------------------------------------------------
# RUN OPTUNA
# ----------------------------------------------------------------------
sampler = optuna.samplers.TPESampler(
    n_startup_trials=50,
    multivariate=True,
    group=True,
    seed=24,
)

study = optuna.create_study(
    direction="minimize",
    sampler=sampler,
)

study.optimize(objective, n_trials=number_of_trials)

best_params = study.best_params
pickle.dump(study, open(f"{path}optuna_study.pkl", "wb"))

print("Best parameters:", best_params)

# ----------------------------------------------------------------------
# FINAL MODELS
# ----------------------------------------------------------------------
config_best = BQConfig(
    kernel_type="RBF",
    lengthscale=[best_params["lengthscale1"], best_params["lengthscale2"]],
    noise=best_params["noise"],
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

# Model B: all data (train + blind)
runner_all = BayesianQuadratureRunner(system=system, config=config_best)
runner_all.initialize_from_data(X_all, Y_all)

chain_values = [0, 10, 20, 40]

# -------- Final plots: training-only model --------
runner_train.plot_derivatives(
    show=False,
    savepath=(
        f"{path}ders/final_derivatives_train_only_"
        f"RBF_{best_params['lengthscale1']}_"
        f"{best_params['lengthscale2']}_{best_params['noise']}.png"
    ),
)

runner_train.plot_fes(
    show=False,
    savepath=(
        f"{path}final_fes_train_only_"
        f"RBF_{best_params['lengthscale1']}_"
        f"{best_params['lengthscale2']}_{best_params['noise']}.png"
    ),
)

is_train_train = train_mask

plot_derivative_slices_at_chain_values(
    runner_train,
    chain_values=chain_values,
    X_obs=X_all,
    Y_obs=fy,
    std_obs=std,
    is_train=is_train_train,
    savepath=(
        f"{path}final_der_slices_train_only_"
        f"RBF_{best_params['lengthscale1']}_"
        f"{best_params['lengthscale2']}_{best_params['noise']}.png"
    ),
    show=False,
    n_sigma=2,
)

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
        f"RBF_{best_params['lengthscale1']}_"
        f"{best_params['lengthscale2']}_{best_params['noise']}.png"
    ),
    show=False,
)

# -------- Final plots: train + all data model --------
is_train_all = np.ones_like(chain, dtype=bool)

runner_all.plot_derivatives(
    show=False,
    savepath=(
        f"{path}ders/final_derivatives_train_plus_all_"
        f"RBF_{best_params['lengthscale1']}_"
        f"{best_params['lengthscale2']}_{best_params['noise']}.png"
    ),
)

runner_all.plot_fes(
    show=False,
    savepath=(
        f"{path}final_fes_train_plus_all_"
        f"RBF_{best_params['lengthscale1']}_"
        f"{best_params['lengthscale2']}_{best_params['noise']}.png"
    ),
)

plot_derivative_slices_at_chain_values(
    runner_all,
    chain_values=chain_values,
    X_obs=X_all,
    Y_obs=fy,
    std_obs=std,
    is_train=is_train_all,
    savepath=(
        f"{path}final_der_slices_train_plus_all_"
        f"RBF_{best_params['lengthscale1']}_"
        f"{best_params['lengthscale2']}_{best_params['noise']}.png"
    ),
    show=False,
    n_sigma=2,
)

plot_fes_slices_at_chain_values(
    runner_all,
    chain_values=chain_values,
    fes_refs=fes_refs,
    val_chains=val_chains,
    blind_chains=blind_chains,
    com_min=com_min,
    com_max=com_max,
    savepath=(
        f"{path}final_fes_slices_train_plus_all_"
        f"RBF_{best_params['lengthscale1']}_"
        f"{best_params['lengthscale2']}_{best_params['noise']}.png"
    ),
    show=False,
)


def plot_fes_comparison(
    runner1,
    runner2,
    label1="FES (model 1)",
    label2="FES (model 2)",
    diff_label="FES1 - FES2",
    savepath=None,
    show=True,
):
    """
    Plot FES of two BayesianQuadratureRunner instances side-by-side
    and their difference (FES1 - FES2).

    Works for dim=1 and dim=2.
    """

    # Make sure FES are up to date
    if runner1.current_fes_1d is None and runner1.current_fes_2d is None:
        runner1._update_fes()
    if runner2.current_fes_1d is None and runner2.current_fes_2d is None:
        runner2._update_fes()

    if runner1.dim != runner2.dim:
        raise ValueError("runner1 and runner2 must have the same dim.")

    dim = runner1.dim

    if dim == 1:
        # 1D case: A(x) as lines
        x1 = runner1.x_grid_1d
        x2 = runner2.x_grid_1d
        if not np.allclose(x1, x2):
            raise ValueError("1D grids do not match; cannot subtract FES directly.")

        fes1 = runner1.current_fes_1d
        fes2 = runner2.current_fes_1d
        fes_diff = fes1 - fes2

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

        axes[0].plot(x1, fes1, color="C0")
        axes[0].set_title(label1)
        axes[0].set_xlabel("CV")
        axes[0].set_ylabel("Free energy")

        axes[1].plot(x1, fes2, color="C1")
        axes[1].set_title(label2)
        axes[1].set_xlabel("CV")

        axes[2].plot(x1, fes_diff, color="C2")
        axes[2].set_title(diff_label)
        axes[2].set_xlabel("CV")

        fig.tight_layout()

    else:
        # 2D case: A(x, y) as pcolormesh
        X1, Y1 = runner1.X_grid_2d, runner1.Y_grid_2d
        X2, Y2 = runner2.X_grid_2d, runner2.Y_grid_2d

        if not (np.allclose(X1, X2) and np.allclose(Y1, Y2)):
            raise ValueError("2D grids do not match; cannot subtract FES directly.")

        fes1 = runner1.current_fes_2d
        fes2 = runner2.current_fes_2d
        fes_diff = fes1 - fes2

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # common color range for the first two
        vmin = min(fes1.min(), fes2.min())
        vmax = max(fes1.max(), fes2.max())

        im0 = axes[0].pcolormesh(X1, Y1, fes1, shading="auto", vmin=vmin, vmax=vmax, cmap="viridis")
        axes[0].set_title(label1)
        axes[0].set_xlabel("CV1")
        axes[0].set_ylabel("CV2")
        fig.colorbar(im0, ax=axes[0], label="Free energy")

        im1 = axes[1].pcolormesh(X1, Y1, fes2, shading="auto", vmin=vmin, vmax=vmax, cmap="viridis")
        axes[1].set_title(label2)
        axes[1].set_xlabel("CV1")
        axes[1].set_ylabel("CV2")
        fig.colorbar(im1, ax=axes[1], label="Free energy")

        # symmetric color scale around 0 for the difference
        absmax = np.max(np.abs(fes_diff))
        im2 = axes[2].pcolormesh(
            X1, Y1, fes_diff, shading="auto",
            vmin=-absmax, vmax=absmax, cmap="coolwarm"
        )
        axes[2].set_title(diff_label)
        axes[2].set_xlabel("CV1")
        axes[2].set_ylabel("CV2")
        fig.colorbar(im2, ax=axes[2], label="ΔF")

        fig.tight_layout()

    # Save / show
    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


savepath_cmp = (
    f"{path}final_fes_comparison_"
    f"RBF_{best_params['lengthscale1']}_"
    f"{best_params['lengthscale2']}_{best_params['noise']}.png"
)

plot_fes_comparison(
    runner_train,
    runner_all,
    label1="FES (train only)",
    label2="FES (train + all)",
    diff_label="FES(train) - FES(train+all)",
    savepath=savepath_cmp,
    show=False,
)