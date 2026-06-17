# ============================================================
# Script de integração: Lagrangiano Aumentado + FEM Barra (Tração)
# Otimizador Interno: CMA-ES
# ============================================================

import numpy as np
import cma

# -- Problema físico (Nova Barra 3D com Ótimo Conhecido) ------
from sanity import f, g, h, params_to_x, x_to_params, L_bloco

solver_int = 'cma'
problema = 'sanity'
# Ponto inicial arbitrário de teste (ex: Áreas [20.0, 80.0])
x0 = params_to_x(20.0, 80.0)

# ============================================================
# Solver interno do subproblema irrestrito
# ============================================================
def solver_cma(x_min, x_max, max_evals=500, sigma=0.2, seed=367):
    def solver(func, x0):
        dim = len(x0)
        popsize = 4 + int(3 * np.log(dim))

        optimizer = cma.CMAEvolutionStrategy(
            x0, sigma * (x_max - x_min),
            {'popsize': popsize, 'verb_disp': 0, 
             'seed': int(seed) if seed is not None else 0}
        )

        evals = 0
        while evals + popsize <= max_evals:
            X      = optimizer.ask()
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
    print("Viga - CMA")
    print("=" * 60)

    # 1. Avaliação no ponto inicial
    print("\n[Passo 1] Ponto inicial do chute:")
    A1_0, A2_0 = x_to_params(x0)
    vol0 = f(x0)
    print(f"  A1 = {A1_0:.1f} mm²,  A2 = {A2_0:.1f} mm²")
    print(f"  Volume Inicial = {vol0:.1f} mm³")

    # 2. Otimização
    print("\n[Passo 2] Iniciando otimização com CMA...")
    solver = solver_cma(x_min=0.0, x_max=1.0)

    # Framework externo do Lagrangiano Aumentado
    from lagrangian_aug_for_simulations import alg_lagran_aug
    
    x_opt, historico = alg_lagran_aug(
        f=f,
        h=h,           # Agora passamos a função h (igualdade de deslocamento)
        g=g,           # g está vazio (retorna np.array([]))
        x0=x0.copy(),
        solver=solver,
        max_iter=50,   # iterações externas do Lagrangiano Aumentado
    )

    # 3. Resultados e Validação com o Gabarito Físico
    print("\n[Passo 3] Resultados da otimização:")
    A1_opt, A2_opt = x_to_params(x_opt)
    vol_opt = f(x_opt)

    print(f"  A1* = {A1_opt:.4f} mm²  (Teórico: 50.0000)")
    print(f"  A2* = {A2_opt:.4f} mm²  (Teórico: 50.0000)")
    print(f"  Volume Mínimo* = {vol_opt:.2f} mm³")
    
    # Validação absoluta do erro em relação à física teórica
    erro_abs = np.linalg.norm(np.array([A1_opt, A2_opt]) - np.array([50.0, 50.0]))
    print(f"  Distância Euclidiana até o Ótimo Teórico: {erro_abs:.6f}")
    print("=" * 60)

    # 4. Exportação dos resultados
    print("\n[Passo 4] Exportando resultados para ParaView e gráficos...")
    from sanity import exportar_resultados_barra
    exportar_resultados_barra(x0,    solver=solver_int, label="chute_inicial")
    exportar_resultados_barra(x_opt, solver=solver_int, label="barra_otima")

    from utils import salvar_historico_csv
    salvar_historico_csv(historico, label=f'{problema}_{solver_int}', pasta='results')

    return x_opt

if __name__ == "__main__":
    main()