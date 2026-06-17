# run_optimization_viga_cma.py
# ============================================================
# Pipeline: Lagrangiano Aumentado + FEM (Timoshenko) + CMA-ES
#
# Variáveis de projeto: E, b, h  (3 variáveis)
# Objetivo: massa = rho * L * b * h   (minimizar)
# Restrição: delta_FEM / DELTA_MAX - 1 <= 0
# ============================================================

import numpy as np
import cma

from viga import f, g, h_eq as h, x0, x_to_params, verify_fem
from lagrangian_aug_for_simulations import alg_lagran_aug
from utils import salvar_historico_csv

solver_int = 'cma'
problema   = 'viga'


# ============================================================
# Solver interno
# ============================================================

def solver_cma(x_min=0.0, x_max=1.0, max_evals=500, sigma=0.3, seed=367):
    def solver(func, x0):
        dim     = len(x0)
        popsize = 4 + int(3 * np.log(dim))
        optimizer = cma.CMAEvolutionStrategy(
            x0, sigma * (x_max - x_min),
            {'popsize': popsize, 'verb_disp': 0,
             'seed': int(seed) if seed is not None else 0}
        )
        evals = 0
        while evals + popsize <= max_evals:
            X     = optimizer.ask()
            fvals = [func(np.clip(x, 0.0, 1.0)) for x in X]
            optimizer.tell(X, fvals)
            evals += popsize
        return np.clip(optimizer.result.xbest, 0.0, 1.0)
    return solver


# ============================================================
# Pipeline principal
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("  Viga de Timoshenko + CMA-ES")
    print("  Objetivo: minimizar massa   Restrição: rigidez")
    print("=" * 60)

    # 0. Verificação FEM
    print("\n[Passo 0] Verificando FEM vs. solução analítica...")
    erro_rel = verify_fem(verbose=True)
    if erro_rel > 5.0:
        print(f"  AVISO: erro relativo {erro_rel:.1f}% > 5% — refine a malha")

    # 1. Ponto inicial
    print("\n[Passo 1] Ponto inicial:")
    E0, b0, h0 = x_to_params(x0)
    f0 = f(x0)
    g0 = g(x0)
    print(f"  E = {E0/1e9:.1f} GPa  b = {b0*1e3:.0f} mm  h = {h0*1e3:.0f} mm")
    print(f"  massa inicial    = {f0:.3f} kg")
    print(f"  g[0] rigidez     = {g0[0]:+.4f}  ({'VIOLADA' if g0[0]>0 else 'ok'})")

    # 2. Otimização
    print("\n[Passo 2] Iniciando otimização...")
    solver = solver_cma(x_min=0.0, x_max=1.0)

    x_opt, historico = alg_lagran_aug(
        f=f, g=g, h=h,
        x0=x0.copy(),
        solver=solver,
        max_iter=50,
    )

    # 3. Resultados
    print("\n[Passo 3] Resultados:")
    E_opt, b_opt, h_opt = x_to_params(x_opt)
    f_opt = f(x_opt)
    g_opt = g(x_opt)

    print(f"  E* = {E_opt/1e9:.2f} GPa")
    print(f"  b* = {b_opt*1e3:.1f} mm")
    print(f"  h* = {h_opt*1e3:.1f} mm")
    print(f"  massa* = {f_opt:.3f} kg")
    print(f"  g[0] rigidez = {g_opt[0]:+.4f}  ({'VIOLADA ⚠' if g_opt[0]>1e-4 else 'satisfeita ✓'})")
    print(f"\n  Redução de massa: {(f0-f_opt)/f0*100:.1f}%  ({f0:.3f} → {f_opt:.3f} kg)")
    print("=" * 60)

    # 4. Exportação
    print("\n[Passo 4] Exportando resultados...")
    from viga import exportar_resultados
    exportar_resultados(x0,    solver=solver_int, label="ponto_inicial", pasta="results")
    exportar_resultados(x_opt, solver=solver_int, label="ponto_otimo",   pasta="results")

    salvar_historico_csv(historico, label=f'{problema}_{solver_int}', pasta='results')

    return x_opt, E_opt, b_opt, h_opt


if __name__ == "__main__":
    main()