import numpy as np
import time as tm
import pandas as pd
from hanan.optimization.Unit import Unit
from hanan.optimization.Step_Control import Step_Control
from hanan.optimization.QuadFairness import QuadFairness
from hanan.optimization.Lap_Fairness import Lap_Fairness
from hanan.optimization.Edge_Length import Edge_Length
from hanan.optimization.TargetValue import TargetValue
from hanan.optimization.FixValueVariable import FixValueVariable
from hanan.optimization.proximityRef import proximity_reference
from hanan.optimization.glideRef import glideReference
from hanan.optimization.prevIterationReg import prevRegularizer

from hanan.optimization.Corner import Corner
from hanan.geometry.utils import unit
from scipy.sparse import diags, vstack, eye, issparse
from scipy.sparse.linalg import spsolve, gmres, lsqr
import matplotlib.pyplot as plt



class Optimizer():
    def __init__(self) -> None:
        """
        Optimizer for solving non-linear least squares problems using either
        Levenberg–Marquardt (LM) or Projected Gauss–Newton (PG) methods.
        
        The optimization problem is formulated as:
            J^T J dx = - J^T r (for LM)
        or, in the case of PG (to be implemented),
            J^T J (X + λ I) = J^T (X) - r dx.
        
        Attributes:
            X (np.ndarray): The current variable vector.
            X0 (np.ndarray): The initial variable vector.
            var_idx (dict): A dictionary mapping variable names to their indices.
            var (int): Total number of variables.
            bestX (np.ndarray): The best solution found.
            bestit (int): The iteration number of the best solution.
            H (sparse matrix): The Hessian matrix (J^T * J).
            b (np.ndarray): The right-hand side vector (J^T * r).
            it (int): Current iteration count.
            step (float): Step size used in variable updates.
            prevdx (float): Norm of the previous update step.
            method (str): Optimization method ("LM" or "PG").
            energy (list): List of energy (cost) values per iteration.
            energy_dic (dict): Energy contribution per objective term.
            norm_energy_dic (dict): Normalized energy per objective term.
            objective terms (dict): Dictionary of objective term objects.
            verbose (bool): Flag to print progress messages.
            stop (bool): Flag to indicate early termination.
        """
        # Variables and their indices
        self.X = None
        self.X0 = None
        self.var_idx = {}  # Dictionary mapping variable names to indices (e.g. "v1": np.arange(...))
        self.var = 0       # Total number of variables
        
        # Best solution found during optimization
        self.bestX = None
        self.bestit = None
        
        # Hessian and gradient-related objects
        self.H = None
        self.b = None
        self.mu = None  # Damping factor for LM
        self.adaptive_mu = True  # Flag to enable/disable adaptive damping
        self._total_energy = 0.0  # Total energy across ALL terms (computed in get_gradients)
        
        # Iteration details and step size
        self.it = None
        self.step = None
        self.prevdx = None
        self.iter_time = []
        self.dataFrameOptimization = pd.DataFrame()
        
        # Optimization method ("LM" for Levenberg–Marquardt or "PG" for Projected Gauss–Newton)
        self.method = None
        
        # Energy history and objective term energy breakdown
        self.energy = []
        self.energy_dic = {}
        self.norm_energy_dic = {}

        # Objective terms registered with the optimizer
        self.objective_terms = {}
        self.constraints = self.objective_terms  # Backward compatibility alias
        
        # Flags for verbosity and stopping the optimization
        self.verbose = False
        self.stop = False
        
        # Log string for recording optimization progress
        self.log = ""

    def clear_energy(self) -> None:
        """
        Clears the energy history and resets iteration bookkeeping.
        """
        self.energy = []
        self.it = 0
        self.bestit = None
        self.bestX = None
        self.prevdx = None

    def clear_objective_terms(self) -> None:
        """
        Clears the stored objective terms.
        """
        self.objective_terms = {}
        self.constraints = self.objective_terms  # Keep alias in sync

    def clear_constraints(self) -> None:
        """
        DEPRECATED: Use clear_objective_terms() instead.
        Clears the stored objective terms.
        """
        self.clear_objective_terms()

    def add_variable(self, var_name: str, vals: np.ndarray) -> None:
        """
        Declares a named variable block and sets its initial values.
        The size of the block is inferred from len(vals).

        Args:
            var_name (str): The name of the variable.
            vals (np.ndarray): Initial values; their length defines the block size.
        """
        vals = np.asarray(vals, dtype=float).ravel()
        dim  = len(vals)

        old_var = self.var
        self.var_idx[var_name] = np.arange(old_var, old_var + dim)
        self.var += dim

        # Grow X while preserving previously set values.
        new_X  = np.zeros(self.var)
        new_X0 = np.zeros(self.var)
        if self.X is not None and old_var > 0:
            new_X[:old_var]  = self.X[:old_var]
            new_X0[:old_var] = self.X0[:old_var]
        self.X  = new_X
        self.X0 = new_X0
        self.X[self.var_idx[var_name]]  = vals
        self.X0[self.var_idx[var_name]] = vals

    def set_variable(self, name: str, vals: np.ndarray) -> None:
        """
        Sets the values of a variable block in X (current state only).

        Args:
            name (str): The variable name.
            vals (np.ndarray): The new values for the variable.
        """
        self.X[self.var_idx[name]] = vals

    def init_variable(self, name: str, vals: np.ndarray) -> None:
        """
        Sets the initial values of a variable block (both X and X0).
        Kept for backward compatibility — prefer passing vals to add_variable.
        """
        self.X[self.var_idx[name]]  = vals
        self.X0[self.var_idx[name]] = vals

    def init_variables(self, X: np.ndarray) -> None:
        """
        Replaces the entire variable vector at once.

        Args:
            X (np.ndarray): Full variable vector (must match total declared size).
        """
        assert len(X) == self.var, f"Expected {self.var} variables, got {len(X)}"
        self.X = X.copy()
        self.X0 = X.copy()

    def add_objective_term(self, term, args, w: float = 1, ce: bool = False) -> None:
        """
        Adds an objective term to the optimizer.

        Args:
            term: An instance of an ObjectiveTerm subclass.
            args: Additional arguments for the term's initialization.
            w (float): Weight factor for the term.
            ce (bool): If True, the energy from this term is recorded.
        """
        term.consider_energy = ce
        # Initialize term with the current variables and indices.
        term._initialize_objective(self.X, self.var_idx, *args)
        term.set_weigth(w)
        self.objective_terms[term.name] = term
        self.constraints = self.objective_terms  # Keep alias in sync

    def add_constraint(self, constraint, args, w: float = 1, ce: bool = False) -> None:
        """
        DEPRECATED: Use add_objective_term() instead.
        Adds an objective term to the optimizer.

        Args:
            constraint: An instance of an ObjectiveTerm subclass.
            args: Additional arguments for the term's initialization.
            w (float): Weight factor for the term.
            ce (bool): If True, the energy from this term is recorded.
        """
        self.add_objective_term(constraint, args, w, ce)

    def get_term_weights_dict(self) -> dict:
        return dict([(key, term.w) for key, term in self.objective_terms.items()])


    def set_term_weights(self, weights_dict: dict) -> None:
        """
        Sets the weights of objective terms using a dictionary.

        Args:
            weights_dict (dict): Dictionary mapping term names to their new weights.
        """
        changed = False
        for name, w in weights_dict.items():
            if name in self.objective_terms and self.objective_terms[name].w != w:
                changed = True
                self.objective_terms[name].set_weigth(w)

        # Keep track of updated weights
        if changed:
            self.log += f"Updated weights at it {self.it}:\n"
            for name, w in weights_dict.items():
                self.log += f"  {name}: {w}\n" 
        

    def set_term_weight(self, name: str, w: float) -> None:
        """
        Sets the weight of a specific objective term.

        Args:
            name (str): The name of the objective term.
            w (float): The new weight for the term.
        """
        if name in self.objective_terms:
            self.objective_terms[name].set_weigth(w)

    def set_constraints_weights_dic(self, dic_e_w: dict) -> None:
        """
        DEPRECATED: Use set_term_weights() instead.
        Sets the weights of objective terms using a dictionary.

        Args:
            dic_e_w (dict): Dictionary mapping term names to their new weights.
        """
        self.set_term_weights(dic_e_w)

    def get_weights(self) -> dict:
        """
        Returns a dictionary of current weights for all objective terms.

        Returns:
            dict: A dictionary mapping term names to their weights.
        """
        return {name: term.w for name, term in self.objective_terms.items()}

    def initialize_optimizer(self, verbose: bool = False, adaptive_mu: bool = True) -> None:
        """
        Resets iteration bookkeeping before starting (or restarting) the optimization loop.

        Call this after all variables have been declared with add_variable() and
        their initial values have been set with init_variable(). It does NOT
        touch X so variable values are preserved.

        Typical setup order:
            1. opt.add_variable("v", n*3)       # declare all variable blocks
            2. opt.init_variable("v", v0)        # set initial values
            3. opt.add_objective_term(...)        # register energy terms
            4. opt.initialize_optimizer(...)      # reset iteration state — call last
            5. opt.optimize(...)

        Args:
            verbose (bool): Print per-iteration energy if True.
            adaptive_mu (bool): Use gain-ratio adaptive damping (recommended).
                                Set False for the simpler fixed-damping LM variant.
        """
        self.it = 0
        self.verbose = verbose
        self.adaptive_mu = adaptive_mu
        self.stop = False
        self.energy = []
        self.energy_dic = {}
        self.norm_energy_dic = {}
        self.bestX = None
        self.bestit = None
        self.prevdx = None
        self.mu = None
        self._total_energy = 0.0
        self.eye = eye(n=self.var, m=self.var, format='csr')

    # ------------------------ Optimization Methods ------------------------

    def get_gradients(self) -> None:
        """
        Computes and accumulates the gradients (Hessian and residual vector) from all objective terms.
        
        The method loops through all objective terms, calls their compute routines,
        and accumulates the resulting Hessian blocks and residual contributions.
        """

        self.H = None
        self.b = None
        self._total_energy = 0.0  # Reset — will sum ALL terms (free, res already computed)

        for term in self.objective_terms.values():
            if not np.isclose(term.w, 0):

                # Compute the constraint-specific Jacobian and residual vector
                H, b, res, meanRes = term._computeGrad(self.X)

                if self.H is None and self.b is None:
                    self.H = H
                    self.b = b
                else:
                    # Accumulate Hessian and residual vector
                    self.H += H
                    self.b += b

                # Accumulate total energy across ALL terms (no extra cost)
                self._total_energy += res

                # Record per-term energy contributions if enabled
                if term.name is not None and term.consider_energy:
                    self.energy_dic[term.name] = res
                    self.norm_energy_dic[term.name] = meanRes

        # grad_time = tm.time() - init_time

        # print(f"Gradient computation time: {grad_time:.4f} seconds")
    

    # def update_variables(self, dx: np.ndarray) -> None:
    #     """
    #     Updates the variable vector using the computed update step.
        
    #     Args:
    #         dx (np.ndarray): The update step computed from the linear system.
    #     """
    #     if self.method == "LM":
    #         self.X += self.step * dx
    #     elif self.method == "PG":
    #         # For PG, one could add a projection step here.
    #         # (This is left as a placeholder for future implementation.)
    #         pass

    def evaluate_energy(self, X) -> float:
        """
        Evaluates the total energy of the current variable vector by summing contributions from all objective terms.

        Args:
            X (np.ndarray): Variable vector to evaluate energy at.

        Returns:
            float: The total energy.
        """
        total_energy = 0.0
        for term in self.objective_terms.values():
            if not np.isclose(term.w, 0):
                energy = term.compute_energy(X)
                total_energy += energy
        return total_energy

    def evaluate_energy_fast(self, X, dx) -> float:
        """
        Fast energy evaluation using residuals computed at X+dx.
        Avoids recomputing constraint Jacobians.

        Args:
            X (np.ndarray): Current variable vector.
            dx (np.ndarray): Step direction.

        Returns:
            float: The total energy at X+dx.
        """
        total_energy = 0.0
        X_trial = X + dx
        for term in self.objective_terms.values():
            if not np.isclose(term.w, 0):
                r = term._res_dispatch(X_trial)
                total_energy += term.w * np.sum(r**2)
        return total_energy
    
    def _init_mu(self) -> None:
        """
        Initializes the damping parameter mu based on the Hessian diagonal.

        Uses tau * max(diag(H)) where tau = 1e-6 as initial damping.
        This follows the standard LM initialization strategy.
        """
        if self.mu is None:
            if issparse(self.H):
                diag_vals = np.abs(self.H.diagonal())
                diag_max = diag_vals.max() if diag_vals.size > 0 else 1.0
            else:
                diag_max = np.abs(np.diag(self.H)).max()
            # Ensure mu is never zero or negative
            self.mu = 1e-6 * max(diag_max, 1.0)

    def _solve_damped_system(self) -> np.ndarray:
        """
        Solves the damped linear system: (H + μI) dx = -b

        Returns:
            np.ndarray: The computed step direction dx.
        """
        if issparse(self.H):
            H_damped = self.H.copy()
            H_damped.setdiag(H_damped.diagonal() + self.mu)
        else:
            H_damped = self.H.copy()
            H_damped.flat[::self.var + 1] += self.mu
        return spsolve(H_damped, -self.b)

    def _LM_simple(self) -> tuple:
        """
        Simple LM step without adaptive damping (always accepts step).

        Records pre-step energy (from _total_energy computed in get_gradients,
        zero extra cost). bestX is saved before stepping so it matches the
        recorded energy — energy[i] = E(X_i) and bestX achieves energy[bestit].

        Returns:
            tuple: (dx, energy, rho) where rho is None for simple mode.
        """
        dx = self._solve_damped_system()
        energy = self._total_energy  # E(X) before step — already computed, free

        # Track best solution at X (before step), so bestX matches its energy
        if self.it == 0 or energy < (self.energy[self.bestit] if self.bestit is not None else np.inf):
            self.bestX = self.X.copy()
            self.bestit = self.it

        self.X += dx
        self.energy.append(energy)
        self.prevdx = np.linalg.norm(dx)

        return dx, energy, None

    def _LM_adaptive(self, energy_old: float, max_trials: int = 10) -> tuple:
        """
        Adaptive LM step with step acceptance/rejection based on gain ratio.

        Args:
            energy_old: Energy before the step.
            max_trials: Maximum number of trials with increased damping.

        Returns:
            tuple: (dx, energy_new, rho) where rho is the gain ratio.
        """
        rho = None
        dx = None
        energy_new = energy_old

        for trial in range(max_trials):
            dx = self._solve_damped_system()

            # Compute predicted reduction from quadratic model
            predicted_reduction = 0.5 * (self.mu * np.dot(dx, dx) - np.dot(self.b, dx))

            # Evaluate energy at trial position
            energy_new = self.evaluate_energy_fast(self.X, dx)

            # Compute actual reduction and gain ratio
            actual_reduction = energy_old - energy_new

            if predicted_reduction > 1e-15:
                rho = actual_reduction / predicted_reduction
            else:
                rho = 1.0  # Accept if predicted reduction is negligible

            # Decide to accept or reject step
            if rho > 0.0:
                # Adjust damping based on gain ratio
                if rho > 0.75:
                    self.mu = max(self.mu / 3.0, 1e-12)
                elif rho < 0.25:
                    self.mu = min(self.mu * 2.0, 1e12)

                # Accept step
                self.X += dx
                self.energy.append(energy_new)
                self.prevdx = np.linalg.norm(dx)

                if self.it == 0 or energy_new < self.energy[self.bestit]:
                    self.bestX = self.X.copy()
                    self.bestit = self.it

                if self.verbose and trial > 0:
                    print(f"      Step accepted after {trial+1} trial(s) (ρ={rho:.3f}, μ={self.mu:.3e})")

                return dx, energy_new, rho

            else:
                # Reject step and increase damping
                self.mu = min(self.mu * 4.0, 1e12)
                if self.verbose:
                    print(f"      Trial {trial+1}: rejected (ρ={rho:.3f}), increasing μ to {self.mu:.3e}")

        # Max trials exceeded - accept step anyway
        if self.verbose:
            print(f"      WARNING: Max trials ({max_trials}) reached, accepting step")

        self.X += dx
        self.energy.append(energy_new)
        self.prevdx = np.linalg.norm(dx)

        if self.it == 0 or energy_new < self.energy[self.bestit]:
            self.bestX = self.X.copy()
            self.bestit = self.it

        return dx, energy_new, rho


    def optimize_step(self) -> None:
        """
        Executes one iteration of the Levenberg–Marquardt method.

        If adaptive_mu is True, uses adaptive damping with step acceptance/rejection.
        If adaptive_mu is False, uses simple LM that always accepts steps.
        """      
        init_time = tm.time()
        self._init_mu()
        energy_old = self._total_energy  # All terms — consistent with evaluate_energy_fast

        if self.adaptive_mu:
            dx, energy_new, rho = self._LM_adaptive(energy_old)
        else:
            dx, energy_new, rho = self._LM_simple()

        lm_time = tm.time() - init_time

        # Build header string on first iteration
        if self.it == 0:
            mode_str = "Adaptive" if self.adaptive_mu else "Simple"
            header = "=" * 90 + "\n"
            header += f"Levenberg-Marquardt Optimization ({mode_str})\n"
            header += f"{'Iter':<6} | {'Energy':<13} | {'Norm Energy':<13} | {'||dx||':<13} | {'μ':<13} | {'ρ':<8} | {'Time':<8}\n"
            header += "-" * 90
            self.log += header + "\n"
            if self.verbose:
                print(header)

        self.iter_time.append(lm_time)

        # Store iteration data in a DataFrame (optional)
        new_row = {
            'Iteration': self.it + 1,
            'Energy': self.energy[-1],
            'Norm_Energy': sum(self.norm_energy_dic.values()),
            'Step_Norm': self.prevdx,
            'Mu': self.mu,
            'Rho': rho,
            'Time': lm_time
        }
        self.dataFrameOptimization = pd.concat([self.dataFrameOptimization, pd.DataFrame([new_row])], ignore_index=True)
    

        # Build iteration summary string
        energy_norm = sum(self.norm_energy_dic.values())
        rho_str = f"{rho:.4f}" if rho is not None else "N/A"
        iter_line = f"{self.it+1:<6} | {self.energy[-1]:<13.6e} | {energy_norm:<13.6e} | {self.prevdx:<13.6e} | {self.mu:<13.6e} | {rho_str:<8} | {lm_time:<8.4f}"
        self.log += iter_line + "\n"
        if self.verbose and ((self.it + 1) % 10 == 0 or self.it == 0):
            print(iter_line)

        self.it += 1
 

    def optimize(self, max_iter: int = 100, tol: float = 1e-8) -> bool:
        """
        Runs the optimization loop until convergence or until the maximum number of iterations is reached.
        
        Args:
            max_iter (int): Maximum number of iterations.
            tol (float): Convergence tolerance for the change in energy.
        
        Returns:
            bool: True if the optimization converged, False if it was stopped early.
        
        Raises:
            RuntimeError: If the optimizer is not properly initialized.
        """
        if self.X is None:
            raise RuntimeError("Optimizer not initialized. Call initialize_optimizer() first.")
            
        if self.verbose:
            print(f"Starting {self.method} optimization (max_iter={max_iter}, tol={tol})")
            
        while self.it < max_iter and not self.stop:
            try:
                # Check convergence before computing new gradients
                if self._check_convergence(tol):
                    if self.verbose:
                        print(f"Converged at iteration {self.it} with energy {self.energy[-1]:.6e}")
                    self.it += 1  # Final bookkeeping iteration
                    return True
                
                # Compute gradients and accumulate energies
                self.get_gradients()
                # Perform an optimization step (LM or PG)
                self.optimize_step()

            except Exception as e:
                print(f"Optimization failed at iteration {self.it}: {str(e)}")
                self.stop = True
                return False

        if self.verbose:
            if self.stop:
                print(f"Optimization stopped at iteration {self.it}")
            else:
                print(f"Reached maximum iterations ({max_iter})")
        return self.stop

    def _check_convergence(self, tol: float) -> bool:
        """
        Checks whether the optimization has converged based on the change in energy or the step size.
        
        Args:
            tol (float): Tolerance value for convergence.
        
        Returns:
            bool: True if the change in energy or step norm is below tolerance, else False.
        """
        if len(self.energy) < 2:
            return False

        energy_change = abs(self.energy[-1] - self.energy[-2])
        if energy_change < tol or self.prevdx is not None and self.prevdx < tol:
            return True

        # if self.prevdx is not None and self.prevdx < tol:
        #     return True

        return False

    def get_variables(self) -> np.ndarray:
        """
        Prints and returns the best variable vector found.
        
        Returns:
            np.ndarray: The best variable vector.
        """
        print(f"Best iteration: {self.bestit + 1}\tBest energy: {self.energy[self.bestit]}")
        return self.bestX

    def get_data_frame_optimization(self) -> pd.DataFrame:
        """
        Returns the DataFrame containing optimization iteration data.

        Returns:
            pd.DataFrame: DataFrame with iteration details.
        """
        return self.dataFrameOptimization
    
    def print_report(self) -> None:
        """
        Print weights, iterations and energy per constraint in a formatted table.
        """
        report = self.get_report()
        print(report)

    def get_report(self) -> str:
        """
        Returns weights, iterations and energy per constraint as a formatted string.

        Returns:
            str: The formatted report.
        """
        lines = []
        sep = "=" * 80

        # Weights section
        lines.append(f"\n{sep}")
        lines.append(" WEIGHTS REPORT")
        lines.append(sep)
        if self.objective_terms:
            max_name_len = max(len(name) for name in self.objective_terms)
            lines.append(f"{'Term':<{max_name_len}}  {'Weight':>12}")
            lines.append("-" * (max_name_len + 14))
            for name, term in self.objective_terms.items():
                lines.append(f"{name:<{max_name_len}}  {term.w:>12.4e}")
        else:
            lines.append("No objective terms registered.")

        # Iterations
        lines.append(sep)
        lines.append(f" Total Iterations: {self.it}")
        lines.append(f" Time per Iteration: {np.mean(self.iter_time) if self.iter_time else 0.0:.4f} seconds")

        # Energy section
        lines.append(sep)
        lines.append(" ENERGY REPORT")
        lines.append(sep)
        if self.energy_dic:
            total_energy = sum(energy if self.objective_terms[name].consider_energy and self.objective_terms[name].w > 1e-12 else 0.0 for name, energy in self.energy_dic.items())
            max_name_len = max(len(name) for name in self.energy_dic)
            lines.append(f"{'Term':<{max_name_len}}  {'Energy':>14}  {'Normalized':>14}  {'Contrib.':>8}")
            lines.append("-" * (max_name_len + 42))
            for name, energy in self.energy_dic.items():
                if self.objective_terms[name].consider_energy and self.objective_terms[name].w > 1e-12:
                    norm_energy = self.norm_energy_dic.get(name, 0.0)
                    pct = 100.0 * energy / total_energy if total_energy > 0 else 0.0
                    lines.append(f"{name:<{max_name_len}}  {energy:>14.6e}  {norm_energy:>14.6e}  {pct:>7.1f}%")
        else:
            lines.append("No energy data recorded.")

        # Final energy summary
        lines.append(sep)
        lines.append(" FINAL ENERGY SUMMARY")
        lines.append(sep)
        if self.energy:
            lines.append(f"  Final Energy:    {self.energy[-1]:.6e}")
            if self.bestit is not None:
                lines.append(f"  Best Iteration:  {self.bestit + 1}")
                lines.append(f"  Best Energy:     {self.energy[self.bestit]:.6e}")
        else:
            lines.append("  No optimization performed yet.")
        lines.append(sep + "\n")

        return "\n".join(lines)

    def print_energy_report(self) -> None:
        """
        Prints the energy contributions per constraint.
        """
        print("ENERGY REPORT")
        print("===========================================")
        for name, energy in self.energy_dic.items():
            print(f"{name}: {energy}")
        print("===========================================")
        print("=============Final Energy ==================")
        print(f"Final Energy: {self.energy[-1]}")
        print(f"Best iteration: {self.bestit + 1}\nBest energy: {self.energy[self.bestit]}\n")

    def get_norm_energy_per_constraint(self) -> None:
        """
        Prints the normalized energy contributions per constraint.
        """
        print("Normalized ENERGY REPORT")
        print("===========================================")
        for name, energy in self.norm_energy_dic.items():
            print(f"{name}: {energy}")
        print("===========================================")
        print("=============Final Energy ==================")
        total_norm = sum(e for e in self.norm_energy_dic.values())
        print(f"Final Iteration: {self.it} Energy: {total_norm}\n")

    def log_simple(self, filename: str) -> None:
        """
        Creates a simple log file containing the total energy per iteration.
        
        Args:
            filename (str): The name of the log file.
        """
        with open(filename, "w") as log_file:
            for i, energy in enumerate(self.energy):
                log_file.write(f"{i} {energy}\n")

    def print_log(self) -> None:
        """
        Prints the energy log as a Pandas DataFrame.
        """
        df = pd.DataFrame(self.energy, columns=["Energy"])
        print(df)

    def unpack(self, *v_idx) -> np.ndarray:
        """
        Extracts a sub-vector from the variable vector based on the provided variable name(s).
        
        Args:
            *v_idx: One or more variable names.
        
        Returns:
            np.ndarray: The extracted variable sub-vector(s).
        """
        if len(v_idx) == 1:
            return self.X[self.var_idx[v_idx[0]]]
        else:
            return [self.X[self.var_idx[k]] for k in v_idx]

    def reset(self) -> None:
        """
        Resets the optimizer state to its initial configuration.
        """
        self.X = self.X0.copy()
        self.prevdx = None
        self.energy = []
        self.it = 0
        self.bestX = None
        self.bestit = None
        self.clear_constraints()

    def force_unit_variable(self, v_name: str, dim: int) -> None:
        """
        Forces the variable corresponding to v_name to be a unit vector.
        
        Args:
            v_name (str): The variable name.
            dim (int): The dimension of the variable.
        """
        self.X[self.var_idx[v_name]] = unit(
            self.X[self.var_idx[v_name]].reshape(-1, dim)
        ).flatten()

    # ------------------------ Built-in Soft Constraints ------------------------

    def set_corner_term(self, var_name: str, corners, corner_adjlst, target_angle, w: float) -> None:
        """
        Adds a sharp corner constraint to the optimizer.
        
        Args:
            var_name (str): The variable name for which the constraint is applied.
            corners: List of corner vertex indices.
            corner_adjlst: List of adjacent vertex indices for each corner.
            target_angle: Optional target angle for the corner (in degrees).
            w (float): Weight of the constraint.
        """
        corner = Corner()
        corner._initialize_objective(self.X, self.var_idx, var_name, corners, corner_adjlst, target_angle)
        corner.consider_energy = False
        corner.w = w
        self.objective_terms[corner.name] = corner

    def prev_iteration_regularizer(self, w: float, ce=False) -> None:
        """
        Adds a regularization constraint to keep the solution close to the previous iteration.
        
        Args:
            w (float): The weight for this constraint.
        """
        reg = prevRegularizer()
        reg._initialize_objective(self.X, self.var_idx)
        reg.consider_energy = ce
        reg.w = w
        reg.name = "previous_iteration_regularizer"
        self.objective_terms[reg.name] = reg

    def control_variable(self, var_name: str, w: float) -> None:
        """
        Adds a step control constraint that limits the change (step size) of a variable.
        
        Args:
            var_name (str): The variable name.
            w (float): The weight for this constraint.
        """
        sc = Step_Control()
        sc._initialize_objective(self.X, self.var_idx, var_name)
        sc.consider_energy = True
        sc.w = w
        sc.name = var_name + "_step_control"
        self.objective_terms[sc.name] = sc

    def proximity_reference(self, var_name: str, vertices_Reference, faces_Reference, w_proximity: float, w_gliding: float) -> None:
        """
        Adds a proximity constraint to keep vertices close to a reference mesh. 
        Args:
            var_name (str): The variable name representing vertex positions.
            vertices_Reference: Reference mesh vertices.
            faces_Reference: Reference mesh faces.
            w_proximity (float): Weight for the proximity term.
            w_gliding (float): Weight for the gliding term.
        """

        proximity_term = proximity_reference()
        proximity_term.name = var_name + "_proximity"
        proximity_term._initialize_objective(self.X, self.var_idx, var_name, vertices_Reference, faces_Reference)
        proximity_term.consider_energy = True
        proximity_term.w = w_proximity
        self.objective_terms[proximity_term.name] = proximity_term

        gliding_term = glideReference()
        gliding_term.name = var_name + "_gliding"
        gliding_term._initialize_objective(self.X, self.var_idx, var_name, vertices_Reference, faces_Reference)
        gliding_term.consider_energy = True
        gliding_term.w = w_gliding
        self.objective_terms[gliding_term.name] = gliding_term


    def set_lap_smooth(self, var_name: str, indices: np.ndarray, adj_list: list, dim: int, damp_factor=0.9, damp_iteration=10, w=0.001, ce=False) -> None:
        """
        Adds a fairness constraint to smooth out variations among adjacent vertices.
        
        Args:
            var_name (str): The variable name for which fairness is enforced.
            adj_list: List of adjacent vertices indices.
            dec_fac (float): Decrease factor for fairness weight.
            dec_step (float): Step decrement for fairness weight.
            dim (int): Dimension of the variable.
            w (float): Weight of the constraint.
        """
        fairness = Lap_Fairness()
        fairness.consider_energy = ce
        fairness._initialize_objective(self.X, self.var_idx, var_name, indices, adj_list, dim, damp_factor, damp_iteration)
        fairness.w = w
        self.objective_terms[fairness.name] = fairness

    def set_fairness(self, var_name: str, adj_list,  dim: int, damp_factor=0.9, damp_iteration=10, w=0.001, ce=False) -> None:
        """
        Adds a fairness constraint to smooth out variations among adjacent vertices.
        Args:
            var_name (str): The variable name for which fairness is enforced.
            adj_list: List of adjacent vertices indices.
            damp_factor (float): Decrease factor for fairness weight.
            damp_iteration (float): Number of iterations after which the weight is set 0    .
            dim (int): Dimension of the variable.
            w (float): Weight of the constraint.
        """

        variable4_idx = []
        variable4_adj_list = []

        lapVar_idx = []
        lapVar_adj_list = []
        # Separate adjacent terms for Laplacian and quadratic fairness
        for i, neighbors in enumerate(adj_list):
            if len(neighbors) == 4:
                variable4_idx.append(i)
                variable4_adj_list.append(neighbors)
            else: 
                lapVar_idx.append(i)
                
                lapVar_adj_list.append(neighbors)


        if len(variable4_idx) > 0:
            print(f"Adding quadratic fairness for {len(variable4_idx)} vertices with 4 neighbors")
            quadFariness = QuadFairness()
            quadFariness.name = "quad_fairness_"+var_name
            quadFariness.consider_energy = ce
            quadFariness._initialize_objective(self.X, self.var_idx, var_name, variable4_idx, variable4_adj_list, dim, damp_factor, damp_iteration)
            quadFariness.w = w
            self.objective_terms[quadFariness.name] = quadFariness

        if len(lapVar_idx) > 0:
            print(f"Adding Laplacian fairness for {len(lapVar_idx)} ")
            lapFairness = Lap_Fairness()
            lapFairness.name = "lap_fairness_"+var_name
            lapFairness.consider_energy = ce
            lapFairness._initialize_objective(self.X, self.var_idx, var_name, lapVar_idx, lapVar_adj_list, dim, damp_factor, damp_iteration)
            lapFairness.w = w
            self.objective_terms[lapFairness.name] = lapFairness

    def unitize_variable(self, var_name: str, dim: int, w: float) -> None:
        """
        Adds a unit vector constraint to enforce that a variable has unit norm.
        
        Args:
            var_name (str): The variable name.
            dim (int): The dimension of the variable.
            w (float): Weight of the unit objective term.
        """
        unit_term = Unit()
        unit_term._initialize_objective(self.X, self.var_idx, var_name, dim)
        unit_term.consider_energy = False
        unit_term.w = w
        unit_term.name = var_name + "_unit"
        self.objective_terms[unit_term.name] = unit_term

    def fix_values_variables(self, var_name: str, indices: np.ndarray, fixed_values: np.ndarray, w: float)-> None:
        """
        Adds a fixed value constraint to enforce specific values for certain variable indices.
        """
        fix_term = FixValueVariable()

        fix_term._initialize_objective(self.X, self.var_idx, var_name, indices, fixed_values)
        fix_term.consider_energy = True
        fix_term.w = w
        self.objective_terms[fix_term.name] = fix_term

    def set_target_value(self, var_name: str, target_value: np.ndarray, w: float, ce=False) -> None:
        """
        Adds a target value constraint to enforce that a variable approaches a specified target.
        
        Args:
            var_name (str): The variable name.
            target_value (np.ndarray): The target value for the variable.
            w (float): Weight of the target value constraint.
        """
        target_term = TargetValue()
        target_term.consider_energy = ce
        target_term._initialize_objective(self.X, self.var_idx, var_name, target_value)
        target_term.consider_energy = True
        target_term.w = w
        self.objective_terms[target_term.name] = target_term
    
    def quad_fairness(self, var_name: str, verticesList, adj_list, dim: int, dampFactor, dampIt,  w: float) -> None:
        """
        Adds a quadratic fairness constraint to smooth out variations among adjacent vertices.
        
        Args:
            var_name (str): The variable name for which fairness is enforced.
            adj_list: List of adjacent vertices indices.
            dim (int): Dimension of the variable.
            w (float): Weight of the constraint.
        """
        fairness = QuadFairness()
        fairness.consider_energy=True
        fairness._initialize_objective(self.X, self.var_idx, var_name, verticesList, adj_list, dim, dampFactor, dampIt)
        fairness.w = w
        self.objective_terms[fairness.name] = fairness

    def edge_length(self, var_name: str, edges, target_length:None, w: float) -> None:
        """
        Adds an edge length constraint to enforce specific lengths for edges.
        
        Args:
            var_name (str): The variable name representing vertex positions.
            edges: List of edge vertex index pairs.
            target_lengths: Target lengths for each edge.
            w (float): Weight of the edge length constraint.
        """
        edge_length_term = Edge_Length()
        edge_length_term._initialize_objective(self.X, self.var_idx, var_name, edges, target_length)
        edge_length_term.consider_energy = True
        edge_length_term.w = w
        self.objective_terms[edge_length_term.name] = edge_length_term
