# barra_otimizacao.py
# ============================================================
# BASELINE TRACIONADO — Barra 3D segmentada com ótimo analítico conhecido
#
# Variáveis de projeto (normalizadas em [0,1]^2):
#   x[0] -> A1 (Área da seção transversal do Bloco 1) -> escala linear
#   x[1] -> A2 (Área da seção transversal do Bloco 2) -> escala linear
#
# Ótimo Teórico Absoluto Alvo:
#   A1* = 50.0 mm²
#   A2* = 50.0 mm²
# ============================================================

from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
import ufl
import numpy as np
from basix.ufl import element
import os

# ------------------------------------------------------------
# Geometria e Física Fixa (Para garantir o ótimo em 50.0 mm²)
# ------------------------------------------------------------
L_bloco = 100.0       # Comprimento de cada bloco [mm]
L_total = 200.0       # Comprimento total [mm]
E_aco   = 200000.0    # Módulo de Elasticidade [MPa] (N/mm²)
nu_aco  = 0.3         # Coeficiente de Poisson
F_ponta = 100.0       # Força de tração na extremidade [N]
rho_aco = 7.85e-3     # Densidade do aço (g/mm^3)
U_ALVO  = 0.002       # Deslocamento alvo na ponta [mm]

# Limites de busca para as áreas (mm²)
BOUNDS = {
    'A_min': 1.0,
    'A_max': 100.0
}

# ============================================================
# Escala de variáveis [0,1]^2 
# ============================================================

def x_to_params(x_norm):
    """[0,1]^2 -> (A1 [mm²], A2 [mm²])"""
    A1 = BOUNDS['A_min'] + x_norm[0] * (BOUNDS['A_max'] - BOUNDS['A_min'])
    A2 = BOUNDS['A_min'] + x_norm[1] * (BOUNDS['A_max'] - BOUNDS['A_min'])
    return A1, A2

def params_to_x(A1, A2):
    """(A1, A2) -> [0,1]^2"""
    x = np.array([
        (A1 - BOUNDS['A_min']) / (BOUNDS['A_max'] - BOUNDS['A_min']),
        (A2 - BOUNDS['A_min']) / (BOUNDS['A_max'] - BOUNDS['A_min'])
    ])
    return np.clip(x, 0.0, 1.0)


# ============================================================
# Núcleo FEM (FEniCSx)
# ============================================================

def rodar_fem_barra(A1, A2):
    """
    Simula uma barra 3D onde a primeira metade tem área A1
    e a segunda metade tem área A2 através de amostragem DG0.
    """
    # Malha base unificada para evitar descontinuidades geométricas nos solvers black-box
    dominio = mesh.create_box(
        MPI.COMM_WORLD, 
        [[0.0, -1.0, -1.0], [L_total, 1.0, 1.0]], 
        [20, 2, 2],
        cell_type=mesh.CellType.hexahedron
    )
    
    elem = element("Lagrange", dominio.basix_cell(), degree=1, shape=(dominio.geometry.dim,))
    V = fem.functionspace(dominio, elem)
    
    # Parâmetros de Lamé base teóricos
    mu_1    = E_aco / (2 * (1 + nu_aco))
    lmbda_1 = E_aco * nu_aco / ((1 + nu_aco) * (1 - 2 * nu_aco))
    
    # Espaço DG0 para aplicar propriedades físicas diferentes por zona (x <= 100 e x > 100)
    W_sc = fem.functionspace(dominio, ("DG", 0))
    mu_f = fem.Function(W_sc)
    lmbda_f = fem.Function(W_sc)
    
    # Pondera as propriedades mecânicas/rigidez baseadas na área simulada vs área da malha base (4.0 mm²)
    fatores = lambda x: np.where(x[0] <= L_bloco, A1 / 4.0, A2 / 4.0)
    mu_f.interpolate(lambda x: mu_1 * fatores(x))
    lmbda_f.interpolate(lambda x: lmbda_1 * fatores(x))

    def epsilon(u): 
        return ufl.sym(ufl.grad(u))
        
    def sigma(u):   
        return lmbda_f * ufl.tr(epsilon(u)) * ufl.Identity(3) + 2 * mu_f * epsilon(u)

    # Condição de contorno: Engastado em x = 0
    dofs_engaste = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
    bc = fem.dirichletbc(np.zeros(3, dtype=float), dofs_engaste, V)

    # Carga pontual aplicada na face livre extrema x = L_total (dimensão - 1 = 2)
    facetas_livres = mesh.locate_entities_boundary(dominio, dominio.topology.dim - 1, lambda x: np.isclose(x[0], L_total))
    tag_facetas = mesh.meshtags(dominio, dominio.topology.dim - 1, facetas_livres, np.ones(len(facetas_livres), dtype=np.int32))
    ds = ufl.Measure("ds", domain=dominio, subdomain_data=tag_facetas)
    
    # Força distribuída equivalente na face final
    t_vec = fem.Constant(dominio, np.array([F_ponta / 4.0, 0.0, 0.0])) 

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.inner(t_vec, v) * ds(1)

    problema = LinearProblem(
        a, L, bcs=[bc], 
        petsc_options_prefix="barra_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    u_h = problema.solve()

    # Deslocamento máximo axial (X) na ponta da barra
    u_array = u_h.x.array.reshape(-1, 3)
    u_x_max = float(u_array[:, 0].max())
    
    # Volume total real da barra (Função Objetivo)
    volume_total = (A1 * L_bloco) + (A2 * L_bloco)
    
    return u_x_max, volume_total, u_h, dominio, V


# ============================================================
# Interface Blackbox exigida pelo lagrangian_aug_for_simulations
# ============================================================

def f(x_norm):
    """Função Objetivo: Minimizar o Volume total da barra."""
    A1, A2 = x_to_params(x_norm)
    _, volume, _, _, _ = rodar_fem_barra(A1, A2)
    return rho_aco * volume


def g(x_norm):
    """Restrições de desigualdade (Vazio para este problema)."""
    return np.array([])


def h(x_norm):
    """
    Restrição de igualdade purificada: h(x) = u_ponta - U_ALVO = 0
    """
    if any(val < 0.0 or val > 1.0 for val in x_norm):
        return np.array([1e6])  # Penalização externa extrema fora de bounds
        
    A1, A2 = x_to_params(x_norm)
    u_ponta, _, _, _, _ = rodar_fem_barra(A1, A2)
    
    return np.array([u_ponta - U_ALVO])


# ============================================================
# Exportação de Resultados (Padrão Dokken VTX/PyVista)
# ============================================================

def exportar_resultados_barra(x_norm, solver, label="barra_resultado", pasta="results"):
    import dolfinx.io as dio
    from dolfinx.plot import vtk_mesh
    import pyvista
    from pathlib import Path
    
    A1, A2 = x_to_params(x_norm)
    Path(pasta).mkdir(exist_ok=True, parents=True)
    base = str(Path(pasta) / f"{label}_{solver}")
    
    # Executa o solver para extrair os objetos de malha atualizados
    _, _, u_h, dominio, V = rodar_fem_barra(A1, A2)
    u_h.name = "Deslocamento"
    
    # Salva VTXWriter para visualização no ParaView (Padrão BP4)
    path_bp = f"{base}_desl_{solver}.bp"
    with dio.VTXWriter(MPI.COMM_WORLD, path_bp, [u_h], engine="BP4") as vtx:
        vtx.write(0.0)
    print(f"  [VTX Exportado] {path_bp}")

    # Captura de Tela PNG via PyVista Offscreen
    pyvista.OFF_SCREEN = True
    topology, cell_types, coords = vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, coords)
    
    u_arr = u_h.x.array.reshape(-1, 3)
    grid.point_data["Deslocamento_X"] = u_arr[:, 0]
    grid.point_data["Deslocamento_Vec"] = u_arr
    
    # Deforma a viga visualmente para o relatório
    warped = grid.warp_by_vector("Deslocamento_Vec", factor=5000.0)
    warped.set_active_scalars("Deslocamento_X")
    
    pl = pyvista.Plotter()
    pl.add_mesh(warped, scalars="Deslocamento_X", cmap="viridis", show_edges=True,
                scalar_bar_args={"title": "U_x [mm]"})
    pl.add_title(f"Barra {solver.upper()} | A1={A1:.2f}mm² | A2={A2:.2f}mm²", font_size=10)
    pl.view_isometric()
    
    path_png = f"{base}_deformada.png"
    pl.screenshot(path_png)
    pl.close()
    print(f"  [PyVista Imagem] {path_png}")


if __name__ == "__main__":
    # Teste de sanidade rápido para checar se a física e a API do FEniCSx estão OK
    print("Executando teste de sanidade direta do FEniCSx...")
    x_teste = params_to_x(50.0, 50.0)
    u, vol, _, _, _ = rodar_fem_barra(50.0, 50.0)
    print(f"  No ótimo teórico (50, 50):")
    print(f"  Deslocamento calculado: {u:.6f} mm (Esperado: {U_ALVO} mm)")
    print(f"  Volume calculado:       {vol:.1f} mm³")