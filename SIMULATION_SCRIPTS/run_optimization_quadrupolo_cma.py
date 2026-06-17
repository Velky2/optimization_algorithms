import numpy as np
from scipy.optimize import minimize
import cma

# --------------------------------------------------
# Problema físico
# --------------------------------------------------
from quadrupolo import (
    f,
    g,
    h,
    x0,
    x_to_params,
    verify_fem,
    exportar_resultados,
)

# --------------------------------------------------
# Otimizador
# --------------------------------------------------
from lagrangian_aug_for_simulations import alg_lagran_aug

solver_int = 'cma'
problema = 'quadrupolo'
# ==================================================
# Solver CMA-ES
# ==================================================

def solver_cma(
    x_min=0.0,
    x_max=1.0,
    max_evals=500,
    sigma=0.30,
    seed=367,
):

    def solver(func, x0):

        dim = len(x0)

        popsize = 4 + int(3 * np.log(dim))

        optimizer = cma.CMAEvolutionStrategy(
            x0,
            sigma * (x_max - x_min),
            {
                "popsize": popsize,
                "verb_disp": 0,
                "seed": seed,
            },
        )

        evals = 0

        while evals + popsize <= max_evals:

            X = optimizer.ask()

            fvals = [
                func(np.clip(x, 0.0, 1.0))
                for x in X
            ]

            optimizer.tell(X, fvals)

            evals += popsize

        return np.clip(
            optimizer.result.xbest,
            0.0,
            1.0,
        )

    return solver


# ==================================================
# Pipeline principal
# ==================================================

def main():

    print()
    print("=" * 60)
    print("Quadrupolo - CMA")
    print("=" * 60)

    # --------------------------------------------------
    # Verificação FEM
    # --------------------------------------------------

    print("\n[Passo 0] Verificação FEM")

    erro_rel = verify_fem(verbose=True)

    if erro_rel > 5.0:
        print(
            "\nAVISO: erro acima de 5%"
        )

    # --------------------------------------------------
    # Ponto inicial
    # --------------------------------------------------

    print("\n[Passo 1] Ponto inicial")

    d0, r0, J0 = x_to_params(x0)

    f0 = f(x0)
    g0 = g(x0)

    print(f"d_wire = {d0:.4f} m")
    print(f"r_wire = {r0:.4f} m")
    print(f"J0     = {J0:.3e} A/m²")

    print(f"f(x0) = {f0:.6f}")

    print(
        f"g(x0) = [{g0[0]:.6f}, {g0[1]:.6f}]"
    )

    # --------------------------------------------------
    # Otimização
    # --------------------------------------------------

    print("\n[Passo 2] Otimização")

    solver = solver_cma(
        x_min=0.0,
        x_max=1.0,
        max_evals=500,
    )

    x_opt, historico = alg_lagran_aug(
        f=f,
        g=g,
        h=h,
        x0=x0.copy(),
        solver=solver,
        max_iter=50,
    )

    # --------------------------------------------------
    # Resultado
    # --------------------------------------------------

    print("\n[Passo 3] Resultado")

    d_opt, r_opt, J_opt = x_to_params(x_opt)

    f_opt = f(x_opt)
    g_opt = g(x_opt)

    G0 = -f0
    Gopt = -f_opt

    print(f"d*  = {d_opt:.5f} m")
    print(f"r*  = {r_opt:.5f} m")
    print(f"J*  = {J_opt:.3e} A/m²")

    print()
    print(f"G inicial = {G0:.6f} T/m")
    print(f"G ótimo   = {Gopt:.6f} T/m")

    print()
    print(f"f* = {f_opt:.6f}")

    print(
        f"g* = [{g_opt[0]:.6f}, {g_opt[1]:.6f}]"
    )

    print(
        f"g[0] sobreposição "
        f"{'violada' if g_opt[0] > 1e-4 else 'ok'}"
    )

    print(
        f"g[1] domínio "
        f"{'violada' if g_opt[1] > 1e-4 else 'ok'}"
    )

    melhora = (
        (Gopt - G0)
        / max(abs(G0), 1e-12)
        * 100
    )

    print()
    print(
        f"Melhora do gradiente = "
        f"{melhora:.2f}%"
    )

    # --------------------------------------------------
    # Exportação
    # --------------------------------------------------

    print("\n[Passo 4] Exportação")

    exportar_resultados(
        x0,
        solver=solver_int,
        label="quadrupolo_inicial",
        pasta="results",
    )

    exportar_resultados(
        x_opt,
        solver=solver_int,
        label="quadrupolo_otimo",
        pasta="results",
    )

    from utils import salvar_historico_csv
    salvar_historico_csv(historico, label=f'{problema}_{solver_int}', pasta='results')

    print("\nArquivos gerados em results/")

    return x_opt


if __name__ == "__main__":
    main()