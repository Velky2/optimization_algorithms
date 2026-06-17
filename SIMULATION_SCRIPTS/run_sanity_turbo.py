# ============================================================
# Pipeline: Lagrangiano Aumentado + FEM Barra (Tração) + TuRBO
# ============================================================

import math
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from botorch.exceptions import BadInitialCandidatesWarning
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# -- Problema físico (Nova Barra 3D com Ótimo Conhecido) ------
from sanity import f, g, h, params_to_x, x_to_params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.double

solver_int = 'turbo'
problema = 'sanity'
x0 = params_to_x(20.0, 80.0) # Ponto inicial idêntico para comparação justa

# ============================================================
# TuRBO: Dataclass de controle e atualização de estado
# ============================================================
@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float           = 0.8
    length_min: float       = 0.5 ** 7
    length_max: float       = 1.6
    failure_counter: int    = 0
    failure_tolerance: int  = float("nan")
    success_counter: int    = 0
    success_tolerance: int  = 10
    best_value: float       = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max(4.0 / self.batch_size, float(self.dim) / self.batch_size)
        )

def update_state(state: TurboState, Y_next: torch.Tensor) -> TurboState:
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter  = 0
    else:
        state.success_counter  = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state

# ============================================================
# TuRBO: Geração de candidatos via Thompson Sampling
# ============================================================
def generate_batch(
    state: TurboState,
    model: SingleTaskGP,
    X: torch.Tensor,
    Y: torch.Tensor,
    batch_size: int,
    n_candidates: Optional[int] = None,
    max_cholesky_size: int = 2000,
) -> torch.Tensor:
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    dim = X.shape[-1]
    x_center = X[Y.argmax(), :].clone()

    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    if weights.ndim == 0:
        weights = weights.unsqueeze(0)
        
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / dim))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    sobol = SobolEngine(dim, scramble=True)
    pert  = sobol.draw(n_candidates).to(dtype=dtype, device=device)
    pert  = tr_lb + (tr_ub - tr_lb) * pert

    prob_perturb = min(20.0 / dim, 1.0)
    mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
    ind  = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

    X_cand          = x_center.expand(n_candidates, dim).clone()
    X_cand[mask]    = pert[mask]

    thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
        with torch.no_grad():
            X_next = thompson_sampling(X_cand, num_samples=batch_size)

    return X_next

# ============================================================
# Fábrica do solver TuRBO (Dimensão ajustada para 2D)
# ============================================================
def solver_turbo(
    dim: int   = 2,        # Alterado de 3 para 2 (Variáveis: A1 e A2)
    n_init: int = 5,       
    batch_size: int = 4,       
    max_evals: int  = 500,     
    seed: int = 42,
    verbose: bool = False,
    max_cholesky_size: int = 2000,
):
    def solver(func, x_init: np.ndarray) -> np.ndarray:
        torch.manual_seed(seed)

        sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
        X_sobol = sobol.draw(n=max(n_init - 1, 1)).to(dtype=dtype, device=device)

        x_init_t = torch.tensor(np.clip(x_init, 0.0, 1.0), dtype=dtype, device=device).unsqueeze(0)
        X_turbo = torch.cat([x_init_t, X_sobol], dim=0)

        Y_turbo = torch.tensor(
            [-func(x.cpu().numpy()) for x in X_turbo],
            dtype=dtype, device=device,
        ).unsqueeze(-1)

        state = TurboState(dim=dim, batch_size=batch_size, best_value=Y_turbo.max().item())
        evals = len(X_turbo)

        while not state.restart_triggered and evals < max_evals:
            Y_mean = Y_turbo.mean()
            Y_std  = Y_turbo.std().clamp(min=1e-6)
            train_Y = (Y_turbo - Y_mean) / Y_std

            # Aumentamos levemente o piso do ruído (de 1e-8 para 1e-5) para evitar matrizes singulares
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-5, 1e-2))

            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.01, 4.0)))
            model = SingleTaskGP(X_turbo, train_Y, covar_module=covar_module, likelihood=likelihood)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)

            with gpytorch.settings.max_cholesky_size(max_cholesky_size):
                try:
                    # Tenta o ajuste padrão
                    fit_gpytorch_mll(mll)
                except Exception:
                    # Se a otimização estrita falhar devido ao super-afunilamento das amostras,
                    # injetamos um ruído de estabilização diretamente na diagonal para salvar o passo do TuRBO
                    model.likelihood.noise = torch.tensor([1e-4], dtype=dtype, device=device)

            actual_batch = min(batch_size, max_evals - evals)
            X_next = generate_batch(state=state, model=model, X=X_turbo, Y=train_Y, batch_size=actual_batch, max_cholesky_size=max_cholesky_size)

            Y_next = torch.tensor([-func(x.cpu().numpy()) for x in X_next], dtype=dtype, device=device).unsqueeze(-1)
            evals += actual_batch

            state = update_state(state=state, Y_next=Y_next)
            X_turbo = torch.cat([X_turbo, X_next], dim=0)
            Y_turbo = torch.cat([Y_turbo, Y_next], dim=0)

            if verbose:
                print(f"  [TuRBO] evals={evals:4d} | best f_aug={-state.best_value:.4e} | TR length={state.length:.3f}")

        best_idx = Y_turbo.argmax().item()
        return np.clip(X_turbo[best_idx].cpu().numpy(), 0.0, 1.0)

    return solver

# ============================================================
# Pipeline principal
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("Viga - TURBO")
    print("=" * 60)

    # 1. Ponto inicial
    print("\n[Passo 1] Ponto inicial do chute:")
    A1_0, A2_0 = x_to_params(x0)
    vol0 = f(x0)
    print(f"  A1 = {A1_0:.1f} mm²,  A2 = {A2_0:.1f} mm²")
    print(f"  Volume Inicial = {vol0:.1f} mm³")

    # 2. Otimização
    print("\n[Passo 2] Iniciando Lagrangiano Aumentado + TuRBO...")
    solver = solver_turbo(dim=2, max_evals=500, verbose=True)

    from lagrangian_aug_for_simulations import alg_lagran_aug
    x_opt, historico = alg_lagran_aug(
        f=f,
        g=g,           # Vazio
        h=h,           # Restrição de igualdade do deslocamento da ponta
        x0=x0.copy(),
        solver=solver,
        max_iter=50,   
    )

    # 3. Resultados e Verificação com Gabarito
    print("\n[Passo 3] Resultados da otimização:")
    A1_opt, A2_opt = x_to_params(x_opt)
    vol_opt = f(x_opt)

    print(f"  A1* = {A1_opt:.4f} mm²  (Teórico: 50.0000)")
    print(f"  A2* = {A2_opt:.4f} mm²  (Teórico: 50.0000)")
    print(f"  Volume Mínimo* = {vol_opt:.2f} mm³")
    
    erro_abs = np.linalg.norm(np.array([A1_opt, A2_opt]) - np.array([50.0, 50.0]))
    print(f"  Distância Euclidiana até o Ótimo Teórico: {erro_abs:.6f}")
    print("=" * 60)

    # 4. Exportação
    print("\n[Passo 4] Exportando resultados...")
    from sanity import exportar_resultados_barra
    exportar_resultados_barra(x0,    solver=solver_int, label="chute_inicial")
    exportar_resultados_barra(x_opt, solver=solver_int, label="barra_otima")

    from utils import salvar_historico_csv
    salvar_historico_csv(historico, label=f'{problema}_{solver_int}', pasta='results')

    return x_opt

if __name__ == "__main__":
    main()