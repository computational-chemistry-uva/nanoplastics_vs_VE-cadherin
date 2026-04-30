from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union, List
import numpy as np
import GPy
import matplotlib.pyplot as plt
from emukit.quadrature.acquisitions import IntegralVarianceReduction, UncertaintySampling, MutualInformation
from emukit.quadrature.methods import VanillaBayesianQuadrature
from emukit.model_wrappers.gpy_quadrature_wrappers import  BaseGaussianProcessGPy
from emukit.quadrature.measures import LebesgueMeasure
from buq.systems import CollectiveVariableSystem
from buq.integration import integration_1D_trapz, integration_2D_rgrid
from buq.kernels import (
    SumRBFWhiteGPy,
    SumMaternWhiteGPy) 
from emukit.quadrature.kernels import (
    QuadratureRBFLebesgueMeasure,   
    QuadratureProductMatern52LebesgueMeasure,
    QuadratureProductMatern12LebesgueMeasure,
    QuadratureProductMatern32LebesgueMeasure)



ArrayLike = Union[np.ndarray, List[float]]


@dataclass
class BQConfig:
    kernel_type: str              # "RBF", "Matern52", "Matern32","Matern12"
    lengthscale: Union[float, np.ndarray]
    noise: float
    variance: float = 1.0
    n_queries: int = 0

    grid_size_1d: int = 100
    grid_size_2d: Tuple[int, int] = (50, 50)

    use_mini: bool = True
    fast_mini: bool = False

    optimize_hyperparams: bool = False

    acq_function: str = "IVR"     # "IVR", "US" or "MI"

    #   None  -> all components (original behaviour, Y has shape (n, dim))
    #   [0]   -> only dA/dx  (CV1 direction)
    #   [1]   -> only dA/dy  (CV2 / "fy" direction)
    #   [0,1] -> both (same as None)
    # For a 1D system this is always [0] and you can leave it as None.
    gradient_components: Optional[List[int]] = None

 
class BayesianQuadratureRunner:
    def __init__(
        self,
        system: CollectiveVariableSystem,
        config: BQConfig,
        bounds: Optional[Tuple[float, ...]] = None,
    ):
        self.emukit_method = None
        self.acq_function = None
        self.system = system
        self.dim = system.dim
        self.config = config

        # ---------------------------------------------------------------- CHANGE 2
        # Resolve which gradient components are active.
        # For a 1D system it is always [0].
        # For a 2D system: default to all components unless the user restricted them.
        if self.dim == 1:
            self._grad_components = [0]
        else:
            if config.gradient_components is None:
                self._grad_components = list(range(self.dim))   # [0, 1] — original behaviour
            else:
                self._grad_components = list(config.gradient_components)
        # Number of GP outputs = number of provided gradient components
        self._n_grad = len(self._grad_components)
        # ----------------------------------------------------------------

        if bounds is None:
            if not hasattr(system, "bounds"):
                raise ValueError("System must define a 'bounds' attribute or pass bounds explicitly.")
            bounds = system.bounds

        if self.dim == 1:
            if len(bounds) != 2:
                raise ValueError("For 1D systems, bounds must be (x_min, x_max).")
            self.bounds_1d = (bounds[0], bounds[1])
            self.bounds_2d = None
        elif self.dim == 2:
            if len(bounds) != 4:
                raise ValueError("For 2D systems, bounds must be (x_min, x_max, y_min, y_max).")
            self.bounds_2d = (bounds[0], bounds[1], bounds[2], bounds[3])
            self.bounds_1d = None
        else:
            raise ValueError("BayesianQuadratureRunner currently supports only dim = 1 or 2.")

        self.X_data: Optional[np.ndarray] = None
        self.Y_data: Optional[np.ndarray] = None

        self.x_grid_1d: Optional[np.ndarray] = None
        self.X_grid_2d: Optional[np.ndarray] = None
        self.Y_grid_2d: Optional[np.ndarray] = None
        self.grid_flat: Optional[np.ndarray] = None

        self.current_fes_1d: Optional[np.ndarray] = None
        self.current_fes_2d: Optional[np.ndarray] = None
    
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def initialize(self, initial_points: np.ndarray) -> None:
        initial_points = np.atleast_2d(initial_points)
        if initial_points.shape[1] != self.dim:
            raise ValueError(f"initial_points must have shape (n, {self.dim}).")

        forces = []
        for x in initial_points:
            print(f"Running initial simulation at x = {x}")
            try:
                f = self.system.get_force(x)
            except Exception:
                self.system.run_simulation(x)
            f = self.system.get_force(x)
            f = np.asarray(f).reshape(self.dim)
            # ------------------------------------------------------------ CHANGE 3
            # Keep only the requested gradient components.
            f = f[self._grad_components]
            # ------------------------------------------------------------
            forces.append(f)

        self.X_data = initial_points
        self.Y_data = np.vstack(forces)   # (n, n_grad)

        self._build_grid()
        self._build_gp_and_emukit()
        self._update_fes()

    def initialize_from_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Initialize directly from precomputed data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, dim)
        Y : ndarray, shape (n_samples, n_grad)
            n_grad == len(config.gradient_components).
            For the single-fy case this is (n_samples, 1).
        """
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)

        if X.shape[1] != self.dim:
            raise ValueError(f"X must have shape (n, {self.dim}); got {X.shape}.")

        # ---------------------------------------------------------------- CHANGE 4
        # Accept Y with either n_grad columns (new) or dim columns (legacy full gradient).
        # If the user passes the full gradient matrix we silently slice the right columns.
        if Y.shape[1] == self.dim and self._n_grad < self.dim:
            # Legacy call: full gradient provided — extract the requested components.
            Y = Y[:, self._grad_components]
        elif Y.shape[1] != self._n_grad:
            raise ValueError(
                f"Y must have {self._n_grad} column(s) (gradient_components={self._grad_components}); "
                f"got {Y.shape}."
            )
        # ----------------------------------------------------------------

        self.X_data = X
        self.Y_data = Y

        self._build_grid()
        self._build_gp_and_emukit()
        self._update_fes()

    # ------------------------------------------------------------------ CHANGE 5
    # Extracted _build_gp_and_emukit() so initialize() and initialize_from_data()
    # share a single code path instead of duplicating ~40 lines.
    def _build_gp_and_emukit(self) -> None:
        """Build GPy model + Emukit wrapper from self.X_data / self.Y_data."""

        # GP input dimension is always self.dim (2D spatial inputs).
        # GP output dimension is self._n_grad (1 if only fy is available).
        if self.config.kernel_type == "RBF":
            base = GPy.kern.RBF(self.dim, lengthscale=self.config.lengthscale,
                                variance=self.config.variance, ARD=True)
        elif self.config.kernel_type == "Matern52":
            base = GPy.kern.Matern52(self.dim, lengthscale=self.config.lengthscale,
                                     variance=self.config.variance, ARD=True)
        elif self.config.kernel_type == "Matern12":
            base = GPy.kern.Exponential(self.dim, lengthscale=self.config.lengthscale,
                                        variance=self.config.variance, ARD=True)
        elif self.config.kernel_type == "Matern32":
            base = GPy.kern.Matern32(self.dim, lengthscale=self.config.lengthscale,
                                     variance=self.config.variance, ARD=True)
        else:
            raise ValueError(f"Unknown kernel_type '{self.config.kernel_type}'.")

        white = GPy.kern.White(self.dim, variance=self.config.noise)
        kernel = base + white

        # Y_data has shape (n, n_grad); GPy handles multi-output via ICM if n_grad > 1,
        # but for n_grad == 1 a standard GPRegression is fine.
        gpy_model = GPy.models.GPRegression(X=self.X_data, Y=self.Y_data, kernel=kernel)
        if self.config.optimize_hyperparams:
            gpy_model.optimize(messages=False, max_iters=100)

        if self.dim == 1:
            bounds_list = [(self.bounds_1d[0], self.bounds_1d[1])]
        else:
            x_min, x_max, y_min, y_max = self.bounds_2d
            bounds_list = [(x_min, x_max), (y_min, y_max)]
        measure = LebesgueMeasure.from_bounds(bounds=bounds_list)

        if self.config.kernel_type == "RBF":
            emukit_kernel = SumRBFWhiteGPy(gpy_model.kern)
            quad_kernel = QuadratureRBFLebesgueMeasure(emukit_kernel, measure)
        elif self.config.kernel_type == "Matern52":
            emukit_kernel = SumMaternWhiteGPy(gpy_model.kern)
            quad_kernel = QuadratureProductMatern52LebesgueMeasure(emukit_kernel, measure)
        elif self.config.kernel_type == "Matern12":
            emukit_kernel = SumMaternWhiteGPy(gpy_model.kern)
            quad_kernel = QuadratureProductMatern12LebesgueMeasure(emukit_kernel, measure)
        elif self.config.kernel_type == "Matern32":
            emukit_kernel = SumMaternWhiteGPy(gpy_model.kern)
            quad_kernel = QuadratureProductMatern32LebesgueMeasure(emukit_kernel, measure)

        emu_base_gp = BaseGaussianProcessGPy(kern=quad_kernel, gpy_model=gpy_model)
        self.emukit_method = VanillaBayesianQuadrature(
            base_gp=emu_base_gp, X=self.X_data, Y=self.Y_data
        )
        self.acq_function = self.config.acq_function

    def run(
        self,
        n_queries: Optional[int] = None,
        weight_var: float = 1.0,
        weight_fes: float = 0.0,
        weight_path: float = 0.0,
        sampling_grid: Optional[np.ndarray] = None,
    ) -> None:
        """
        Run the adaptive loop for n_queries steps.

        For each iteration:
          1. Select next point via a variance-based acquisition (optionally combined
             with the current FES and a user-specified path weighting).
          2. Run simulation and read force.
          3. Update GP models.
          4. Recompute the FES.

        Parameters
        ----------
        n_queries : int or None
            Number of additional queries. If None, uses config.n_queries.
        weight_var : float
            Weight on predictive variance in the acquisition (analogous to IVR weight).
        weight_fes : float
            Weight on (scaled) FES (subtracted in acquisition, i.e. prefer low FES if > 0).
        weight_path : float
            Weight on an optional user-supplied sampling_grid (e.g. path bias).
        sampling_grid : ndarray or None
            - 1D: shape (n_grid,)
            - 2D: shape (nx, ny)
            Used only in acquisition (not required).
        """
        if n_queries is None:
            n_queries = self.config.n_queries

        if n_queries <= 0:
            print("No adaptive queries requested (n_queries <= 0).")
            return

        for q in range(n_queries):
            print(f"\n=== Adaptive iteration {q + 1} / {n_queries} ===")
            self.run_one_query(
                weight_var=weight_var,
                weight_fes=weight_fes,
                weight_path=weight_path,
                sampling_grid=sampling_grid,
                compute_fes=False,  # recompute FES after each query by default
            )
        self._update_fes() #always compute FES after all queries are done





    def run_one_query(
        self,
        weight_var: float = 1.0,
        weight_fes: float = 0.0,
        weight_path: float = 0.0,
        sampling_grid: Optional[np.ndarray] = None,
        compute_fes: bool = True,
    ) -> None:
        """
        Perform a single adaptive query:
          - choose next point via acquisition,
          - run simulation,
          - update GPs and FES.

        This allows patterns like:

        >>> for i in range(25):
        ...     runner.run_one_query()
        ...     runner.plot_fes()
        """
        x_next = self._select_next_point(
            weight_var=weight_var,
            weight_fes=weight_fes,
            weight_path=weight_path,
            sampling_grid=sampling_grid,
        )
        print(f"Selected next point: {x_next}")

        # Run simulation at x_next
        try:
            f_next = self.system.get_force(x_next)  # check if already run (for example, in testing)
        except Exception:   
            self.system.run_simulation(x_next)
            f_next = self.system.get_force(x_next)
        
        f_next = np.asarray(f_next).ravel()[self._grad_components].reshape(1, self._n_grad)
    
        # Append data
        self.X_data = np.vstack([self.X_data, x_next.reshape(1, self.dim)])
        self.Y_data = np.vstack([self.Y_data, f_next])

        # Sync Emukit and recompute
        self.emukit_method.set_data(self.X_data, self.Y_data)
        if compute_fes:
            self._update_fes()



    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    
    def plot_fes(
        self,
        savepath: Optional[str] = None,
        show: bool = True,
        true_fes_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        align_min: bool = True,
        true_label: str = "Analytical FES",
    ) -> None:
        """
        Plot the current free energy estimate.

        - 1D: line plot A(x) with training locations; optionally overlays a
            reference FES (true_fes_func or system.true_fes if available).
        - 2D: contour plot A(x, y) with training locations.

        Parameters
        ----------
        true_fes_func : callable or None
            If provided in 1D, a function f(x: ndarray)->ndarray returning
            the analytical/reference FES on x-grid. If None, and the system
            defines `true_fes(x)`, that will be used automatically.
        align_min : bool
            If True, shift the reference FES by its minimum to zero to match
            the runner’s FES (which is shifted to min=0).
        true_label : str
            Legend label for the reference FES curve.
        """
        if self.dim == 1:
            if self.current_fes_1d is None or self.x_grid_1d is None:
                raise RuntimeError("FES not available; call initialize() first.")

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(self.x_grid_1d, self.current_fes_1d, label="FES (BQ)", color="C0")
            if self.X_data is not None:
                xq = self.X_data[:, 0]
                xq_clipped = np.clip(xq, self.x_grid_1d[0], self.x_grid_1d[-1])
                yq = np.interp(xq_clipped, self.x_grid_1d, self.current_fes_1d)

                ax.scatter(xq_clipped, yq, color="red", marker="x", label="Queries")
                

            # Optional analytical/reference curve
            ref_func = true_fes_func
            if ref_func is None and hasattr(self.system, "true_fes"):
                ref_func = getattr(self.system, "true_fes")

            if callable(ref_func):
                y_true = np.asarray(ref_func(self.x_grid_1d)).ravel()
                if align_min:
                    y_true = y_true - np.min(y_true)
                ax.plot(self.x_grid_1d, y_true, label=true_label, color="C1", linestyle="--")

            ax.set_xlabel("CV")
            ax.set_ylabel("Free energy (arb. units)")
            ax.set_title("Bayesian Quadrature FES (1D)")
            ax.legend()

        else:
            if self.current_fes_2d is None or self.X_grid_2d is None or self.Y_grid_2d is None:
                raise RuntimeError("FES not available; call initialize() first.")

            fig, ax = plt.subplots(figsize=(7, 6))
            contour = ax.pcolormesh(self.X_grid_2d, self.Y_grid_2d, self.current_fes_2d,
                                cmap="viridis")
            cbar = fig.colorbar(contour, ax=ax)
            cbar.set_label("Free energy (arb. units)")
            if self.X_data is not None:
                ax.scatter(self.X_data[:, 0], self.X_data[:, 1],
                        color="white", edgecolor="black", s=20, label="Queries")
                ax.legend()
            ax.set_xlabel("CV1")
            ax.set_ylabel("CV2")
            ax.set_title("Bayesian Quadrature FES (2D)")

        if savepath is not None:
            plt.savefig(savepath, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_acq(
        self,
        weight_var: float = 1.0,
        weight_fes: float = 0.0,
        weight_path: float = 0.0,
        sampling_grid: Optional[np.ndarray] = None,
        savepath: Optional[str] = None,
        show: bool = True,
        full: bool = True,
    ) -> None:
        """
        Plot a variance-based "IVR-like" acquisition:

        - 2D: three panels (variance, scaled FES, combined acquisition) as in your
        original code; this mirrors your old IVR/FES/combined plots.
        - 1D: lines for variance, FES, and combined acquisition.

        The acquisition used here is:

            acq = weight_var * (var_norm)
                - weight_fes * (fes_norm)
                + weight_path * sampling_grid

        where "var_norm" is predictive variance normalized by its max,
        and "fes_norm" is the FES normalized by its max. This can be
        replaced by true IVR later if you wire in Emukit.
        """
        var_norm, fes_norm, acq = self._compute_acquisition_grid(
            weight_var=weight_var,
            weight_fes=weight_fes,
            weight_path=weight_path,
            sampling_grid=sampling_grid,
        )

        if self.dim == 1:
            fig, ax = plt.subplots(1, 1, figsize=(7, 4))
            x = self.x_grid_1d
            label = self.acq_function + " acquisition"
            ax.plot(x, var_norm, label=label, color="C0")
            ax.plot(x, fes_norm, label="Normalized FES", color="C1")
            ax.plot(x, acq, label="Combined acquisition", color="C3")
            ax.set_xlabel("CV")
            ax.set_ylabel("Value (normalized)")
            ax.set_title("IVR / FES / acquisition (1D)")
            ax.legend()

        else:
            X = self.X_grid_2d
            Y = self.Y_grid_2d
            var_grid = var_norm
            fes_grid = fes_norm
            acq_grid = acq

            if full:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                # Plot 1: variance
                label = self.acq_function + " variance"
                contour1 = axes[0].contourf(X, Y, var_grid, levels=100, cmap="viridis")
                cbar1 = fig.colorbar(contour1, ax=axes[0], shrink=0.8, aspect=30, pad=0.02)
                cbar1.set_label(label)
                axes[0].set_title(f"{label} Contour")

                # Plot 2: scaled FES
                contour2 = axes[1].contourf(X, Y, fes_grid, levels=100, cmap="plasma")
                cbar2 = fig.colorbar(contour2, ax=axes[1], shrink=0.8, aspect=30, pad=0.02)
                cbar2.set_label("Scaled FES")
                axes[1].set_title("Scaled FES")

                # Plot 3: combined
                contour3 = axes[2].contourf(X, Y, acq_grid, levels=100, cmap="cividis")
                cbar3 = fig.colorbar(contour3, ax=axes[2], shrink=0.8, aspect=30, pad=0.02)
                cbar3.set_label("Combined acquisition")
                axes[2].set_title("Combined acquisition")

            else:
                fig, ax = plt.subplots(figsize=(7, 6))
                contour = ax.contourf(X, Y, acq_grid, levels=100, cmap="viridis")
                cbar = fig.colorbar(contour, ax=ax)
                cbar.set_label("Combined acquisition")
                ax.set_title("Acquisition")

        if savepath is not None:
            plt.savefig(savepath, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()
    

    def plot_derivatives(
        self,
        true_1d: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        true_2d: Optional[Callable[[np.ndarray, np.ndarray], tuple]] = None,
        savepath: Optional[str] = None,
        show: bool = True,
    ) -> None:
        if self.grid_flat is None:
            self._build_grid()
        if self.emukit_method is None:
            raise RuntimeError("Emukit not initialized. Call initialize() first.")

        grad = self._predict_grad_on_grid()  # (n,1) in 1D; (nx,ny,dim) in 2D (zeros for missing)

        if self.dim == 1:
            x = self.x_grid_1d
            dA_dx = grad[:, 0]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(x, dA_dx, label="Predicted dA/dx", color="C0")
            if self.X_data is not None and self.Y_data is not None:
                ax.scatter(self.X_data[:, 0], self.Y_data[:, 0],
                        color="red", marker="x", label="Observed dA/dx")
            ref_fn = true_1d
            if ref_fn is None:
                ref_fn = getattr(self.system, "true_grad", None) or getattr(self.system, "true_force", None)
            if callable(ref_fn):
                ax.plot(x, np.asarray(ref_fn(x)).ravel(), label="Analytical dA/dx",
                        color="C1", linestyle="--")
            ax.set_xlabel("CV"); ax.set_ylabel("dA/dx")
            ax.set_title("Derivative (1D)"); ax.legend()
            if savepath:
                plt.savefig(savepath, dpi=300, bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close()
            return

        # -------------------------- 2D case --------------------------
        # Only plot components actually predicted by the GP
        component_labels = {
            0: ("dA/dx", "CV1", "viridis"),
            1: ("dA/dy", "CV2", "plasma"),
        }
        panels = self._grad_components  # e.g. [1] for fy-only, [0,1] for both

        fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 6))
        if len(panels) == 1:
            axes = [axes]  # make iterable

        X = self.X_grid_2d
        Y = self.Y_grid_2d

        for ax, comp in zip(axes, panels):
            label, cv_label, cmap = component_labels[comp]
            Z = grad[:, :, comp]

            # GP field
            pcm = ax.pcolormesh(X, Y, Z, cmap=cmap)
            cb = fig.colorbar(pcm, ax=ax)
            cb.set_label(label)
            ax.set_title(f"Predicted {label} (2D)")
            ax.set_xlabel("CV1")
            ax.set_ylabel("CV2")

            # Optional analytical / reference field
            if callable(true_2d):
                Zx_true, Zy_true = true_2d(X, Y)
                Z_true = Zx_true if comp == 0 else Zy_true
                ax.contour(X, Y, Z_true, levels=10, colors="k", linewidths=0.8)

            # Overlay data points colored by observed value for this component
            if self.X_data is not None and self.Y_data is not None:
                Xd = self.X_data
                Yd = np.asarray(self.Y_data)
                if Yd.ndim == 1:
                    # single-output case
                    y_vals = Yd
                else:
                    # Try to map component index to column index
                    if Yd.shape[1] == len(self._grad_components):
                        # columns correspond to self._grad_components order
                        col_idx = self._grad_components.index(comp)
                    elif Yd.shape[1] == grad.shape[2]:
                        # columns aligned with full dim (0:dA/dx, 1:dA/dy)
                        col_idx = comp
                    else:
                        # fallback: first column
                        col_idx = 0
                    y_vals = Yd[:, col_idx]

                ax.scatter(
                    Xd[:, 0],
                    Xd[:, 1],
                    c=y_vals,
                    cmap=cmap,
                    norm=pcm.norm,   # same colorscale as the pcolormesh
                    edgecolor="black",
                    s=25,
                    label="Data (obs)",
                )
                ax.legend(loc="upper right")

        fig.suptitle("Derivative contours (2D)")
        fig.tight_layout()
        if savepath:
            plt.savefig(savepath, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()



    def predict_grad(self):
        if self.grid_flat is None:
            self._build_grid()
        if self.emukit_method is None:
            raise RuntimeError("Emukit not initialized.")
        grad = self._predict_grad_on_grid()
        if self.dim == 1:
            return self.x_grid_1d, None, grad[:, 0]
        return self.X_grid_2d, self.Y_grid_2d, grad

    def predict_grad_at(self, X_query: np.ndarray):
        """
        Predict gradient at arbitrary CV locations.

        Returns
        -------
        grad_pred : ndarray, shape (n_points, n_grad)
            Only the components listed in gradient_components.
        std : ndarray
        """
        if self.emukit_method is None:
            raise RuntimeError("Emukit not initialized.")
        X_query = np.atleast_2d(X_query)
        mean, std = self.emukit_method.predict(X_query)
        # ---------------------------------------------------------------- CHANGE 8
        # mean has shape (n, n_grad) — already correct because the GP was trained
        # on n_grad outputs. No reshape to self.dim needed.
        grad_pred = np.asarray(mean)
        if grad_pred.ndim == 1:
            grad_pred = grad_pred.reshape(-1, self._n_grad)
        # ----------------------------------------------------------------
        return grad_pred, std
    

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_grid(self) -> None:
        if self.dim == 1:
            x_min, x_max = self.bounds_1d
            self.x_grid_1d = np.linspace(x_min, x_max, num=self.config.grid_size_1d)
            self.grid_flat = self.x_grid_1d.reshape(-1, 1)
        else:
            x_min, x_max, y_min, y_max = self.bounds_2d
            nx, ny = self.config.grid_size_2d
            x_grid = np.linspace(x_min, x_max, num=nx)
            y_grid = np.linspace(y_min, y_max, num=ny)
            self.X_grid_2d, self.Y_grid_2d = np.meshgrid(x_grid, y_grid, indexing="ij")
            self.grid_flat = np.vstack(
                [self.X_grid_2d.ravel(), self.Y_grid_2d.ravel()]
            ).T

    def _predict_grad_on_grid(self) -> np.ndarray:
        """
        Returns full gradient array of shape (n, 1) in 1D or (nx, ny, dim) in 2D.
        Components not modelled by the GP are set to zero.
        """
        if self.grid_flat is None:
            self._build_grid()
        if self.emukit_method is None:
            raise RuntimeError("Emukit not initialized.")

        mean, _ = self.emukit_method.predict(self.grid_flat)   # (N, n_grad)
        mean = np.asarray(mean)
        if mean.ndim == 1:
            mean = mean.reshape(-1, 1)

        if self.dim == 1:
            return mean.reshape(-1, 1)

        # ---------------------------------------------------------------- CHANGE 9
        # Build a full (nx, ny, dim) array; fill only the predicted components.
        nx, ny = self.config.grid_size_2d
        grad_full = np.zeros((nx, ny, self.dim))
        for out_idx, comp in enumerate(self._grad_components):
            grad_full[:, :, comp] = mean[:, out_idx].reshape(nx, ny)
        return grad_full
        # ----------------------------------------------------------------

    def _update_fes(self) -> None:
        grad = self._predict_grad_on_grid()

        if self.dim == 1:
            dA_dx = grad[:, 0]
            self.current_fes_1d = integration_1D_trapz(self.x_grid_1d, dA_dx)
            return

        # ---------------------------------------------------------------- CHANGE 10
        # 2D FES integration with partial gradient.
        #
        # Case A — both components available: use integration_2D_rgrid as before.
        # Case B — only dA/dy (component 1): integrate along y at each x column.
        # Case C — only dA/dx (component 0): integrate along x at each y row.
        #
        # Cases B and C give a FES surface that is exact along the integrated axis
        # and constant (zero-gradient) along the other axis — the best we can do
        # with a single gradient component.

        if set(self._grad_components) == {0, 1}:
            # Original full-gradient path
            XY_combined = np.stack((self.Y_grid_2d, self.X_grid_2d), axis=-1)
            derivative_xy_combined = grad[:, :, [1, 0]]
            self.current_fes_2d = integration_2D_rgrid(
                XY_combined,
                derivative_xy_combined,
                integrator="simpson+mini" if self.config.use_mini else "simpson",
                fast=self.config.fast_mini,
            )

        elif self._grad_components == [1]:
            # Only dA/dy: cumulative trapz along the y-axis (axis=1 in (nx, ny) grid)
            dA_dy = grad[:, :, 1]                          # (nx, ny)
            y_1d = self.Y_grid_2d[0, :]                   # (ny,) — same for every row
            fes = np.zeros_like(dA_dy)
            for i in range(dA_dy.shape[0]):               # loop over x-slices
                fes[i, :] = np.concatenate(
                    [[0.0], np.cumsum(
                        0.5 * (dA_dy[i, :-1] + dA_dy[i, 1:]) * np.diff(y_1d)
                    )]
                )
            self.current_fes_2d = fes

        elif self._grad_components == [0]:
            # Only dA/dx: cumulative trapz along the x-axis (axis=0 in (nx, ny) grid)
            dA_dx = grad[:, :, 0]                          # (nx, ny)
            x_1d = self.X_grid_2d[:, 0]                   # (nx,)
            fes = np.zeros_like(dA_dx)
            for j in range(dA_dx.shape[1]):               # loop over y-slices
                fes[:, j] = np.concatenate(
                    [[0.0], np.cumsum(
                        0.5 * (dA_dx[:-1, j] + dA_dx[1:, j]) * np.diff(x_1d)
                    )]
                )
            self.current_fes_2d = fes

        else:
            raise ValueError(
                f"Unsupported gradient_components={self._grad_components}. "
                "Use [0], [1], or [0, 1] / None."
            )
        # ----------------------------------------------------------------

    def _compute_acquisition_grid(self, weight_var, weight_fes, weight_path, sampling_grid):
        if self.grid_flat is None:
            self._build_grid()
        if self.emukit_method is None:
            raise RuntimeError("Emukit not initialized.")

        if self.acq_function == "IVR":
            acquisition = IntegralVarianceReduction(self.emukit_method)
        elif self.acq_function == "MI":
            acquisition = MutualInformation(self.emukit_method)
        elif self.acq_function == "US":
            acquisition = UncertaintySampling(self.emukit_method)
        else:
            raise RuntimeError("acq_function must be 'IVR', 'US', or 'MI'.")

        acq_flat = np.asarray(acquisition.evaluate(self.grid_flat)).ravel()
        amax = np.max(acq_flat)
        acq_norm_flat = acq_flat / amax if amax > 0 else np.zeros_like(acq_flat)

        if weight_fes != 0.0:
            self._update_fes()
        fes = self.current_fes_1d if self.dim == 1 else self.current_fes_2d
        fmax = np.max(fes)
        fes_norm = fes / fmax if fmax > 0 else np.zeros_like(fes)

        if self.dim == 1:
            acq_norm = acq_norm_flat
            sampling_term = 0.0 if sampling_grid is None else np.asarray(sampling_grid)
        else:
            nx, ny = self.config.grid_size_2d
            acq_norm = acq_norm_flat.reshape(nx, ny)
            sampling_term = 0.0 if sampling_grid is None else np.asarray(sampling_grid)
            if sampling_grid is not None and sampling_term.shape != acq_norm.shape:
                raise ValueError(f"sampling_grid shape {sampling_term.shape} must be {(nx, ny)}")

        combined = weight_var * acq_norm - weight_fes * fes_norm + weight_path * sampling_term
        return acq_norm, fes_norm, combined

    def _select_next_point(self, weight_var, weight_fes, weight_path, sampling_grid) -> np.ndarray:
        _, _, acq = self._compute_acquisition_grid(weight_var, weight_fes, weight_path, sampling_grid)
        if self.dim == 1:
            idx = int(np.argmax(acq))
            return np.array([self.x_grid_1d[idx]])
        else:
            nx, ny = self.config.grid_size_2d
            i, j = np.unravel_index(int(np.argmax(acq)), (nx, ny))
            return np.array([self.X_grid_2d[i, j], self.Y_grid_2d[i, j]])

