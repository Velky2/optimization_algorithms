# viga_timoshenko.py
# ============================================================
# FEM — Viga de Timoshenko engastada, carga pontual na ponta
#
# Variáveis de projeto (normalizadas em [0,1]^3):
#   x[0] -> E  (módulo de Young)  — escala logarítmica
#   x[1] -> b  (largura da seção) — escala linear
#   x[2] -> h  (altura da seção)  — escala linear
#
# Parâmetros fixos do problema (não variáveis de otimização):
#   nu  = 0.30   — Poisson, fixo (varia <10% entre metais estruturais)
#   rho = 7850   — densidade [kg/m³], aço estrutural genérico
#   P   = 1000   — carga pontual [N]
#   L   = 1.0    — comprimento da viga [m]
#
# Objetivo:
#   f(x) = massa = rho * L * b * h   [kg]   <- minimizar
#
# Restrição (g <= 0 -> viável):
#   g[0] = delta_FEM / DELTA_MAX - 1         <- rigidez
#
# Verificação analítica (Timoshenko):
#   delta = PL^3/(3EI) + PL/(kappa*A*G)
#   I = b*h^3/12,  A = b*h,  G = E/(2*(1+nu))
#   kappa = 10*(1+nu)/(12+11*nu)   [Cowper 1966]
#
# Referências:
#   Cowper (1966), J. Appl. Mech. 33(2):335-340
#   Bathe (1996), Finite Element Procedures, p.401
# ============================================================

from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
import ufl
import numpy as np
from basix.ufl import element
import os

os.makedirs("results", exist_ok=True)


# ============================================================
# Parâmetros fixos do problema
# ============================================================
L_VIGA    = 1.0       # comprimento [m]
NU        = 0.30      # coeficiente de Poisson (fixo)
RHO       = 7850.0    # densidade [kg/m³] (aço estrutural, fixo)
P_CARGA   = 1000.0    # carga pontual [N]
DELTA_MAX = 5e-3      # deflexão máxima admissível [m]
NX, NY, NZ = 20, 4, 2


# ============================================================
# Espaço de busca (escala física)
# ============================================================
BOUNDS = {
    'E_min': 50e9,    'E_max': 250e9,   # [Pa] — Al a aço carbono
    'b_min': 0.02,    'b_max': 0.30,    # largura [m]
    'h_min': 0.02,    'h_max': 0.50,    # altura  [m]
}


# ============================================================
# Normalização / desnormalização
# ============================================================

def x_to_params(x_norm):
    """[0,1]^3 -> (E [Pa], b [m], h [m])"""
    E = 10 ** (
        np.log10(BOUNDS['E_min'])
        + x_norm[0] * (np.log10(BOUNDS['E_max']) - np.log10(BOUNDS['E_min']))
    )
    b = BOUNDS['b_min'] + x_norm[1] * (BOUNDS['b_max'] - BOUNDS['b_min'])
    h = BOUNDS['h_min'] + x_norm[2] * (BOUNDS['h_max'] - BOUNDS['h_min'])
    return E, b, h


def params_to_x(E, b, h):
    """(E, b, h) -> [0,1]^3"""
    x = np.array([
        (np.log10(E) - np.log10(BOUNDS['E_min']))
        / (np.log10(BOUNDS['E_max']) - np.log10(BOUNDS['E_min'])),
        (b - BOUNDS['b_min']) / (BOUNDS['b_max'] - BOUNDS['b_min']),
        (h - BOUNDS['h_min']) / (BOUNDS['h_max'] - BOUNDS['h_min']),
    ])
    return np.clip(x, 0.0, 1.0)


# Ponto inicial — aço carbono ASTM A36, seção 100mm x 200mm
x0 = params_to_x(E=200e9, b=0.10, h=0.20)


# ============================================================
# Solução analítica de referência (Timoshenko)
# ============================================================

def solucao_analitica(E, b, h, P=P_CARGA, L=L_VIGA, nu=NU):
    """
    delta = PL^3/(3EI) + PL/(kappa*A*G)

    Fator de cisalhamento de Cowper (1966) para seção retangular:
        kappa = 10*(1+nu) / (12 + 11*nu)
    """
    I     = b * h**3 / 12
    A     = b * h
    G     = E / (2 * (1 + nu))
    kappa = 10 * (1 + nu) / (12 + 11 * nu)
    return P * L**3 / (3 * E * I) + P * L / (kappa * A * G)


# ============================================================
# Núcleo FEM — elasticidade linear 3D
# ============================================================

def rodar_fem(E, b, h):
    """
    Viga engastada em x=0, tração distribuída em x=L equivalente a P.
    Seção transversal b x h (largura x altura).

    Retorna delta_max [m].
    """
    mu_val    = E / (2 * (1 + NU))
    lmbda_val = E * NU / ((1 + NU) * (1 - 2 * NU))

    dominio = mesh.create_box(
        MPI.COMM_WORLD,
        points=[[0.0, 0.0, 0.0], [L_VIGA, h, b]],
        n=[NX, NY, NZ],
        cell_type=mesh.CellType.hexahedron,
    )

    elem = element("Lagrange", dominio.basix_cell(), degree=1,
                   shape=(dominio.geometry.dim,))
    V = fem.functionspace(dominio, elem)

    mu    = fem.Constant(dominio, float(mu_val))
    lmbda = fem.Constant(dominio, float(lmbda_val))

    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return lmbda * ufl.tr(epsilon(u)) * ufl.Identity(3) + 2 * mu * epsilon(u)

    # Engaste em x=0
    dofs_engaste = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
    bc = fem.dirichletbc(np.zeros(3, dtype=float), dofs_engaste, V)

    # Tração distribuída em x=L equivalente a carga pontual P
    area_face = h * b
    facetas_livre = mesh.locate_entities_boundary(
        dominio, dominio.topology.dim - 1,
        lambda x: np.isclose(x[0], L_VIGA)
    )
    tag_facetas = mesh.meshtags(
        dominio, dominio.topology.dim - 1,
        facetas_livre, np.ones(len(facetas_livre), dtype=np.int32)
    )
    ds    = ufl.Measure("ds", domain=dominio, subdomain_data=tag_facetas)
    t_vec = fem.Constant(dominio, np.array([0.0, -P_CARGA / area_face, 0.0]))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a      = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L_form = ufl.inner(t_vec, v) * ds(1)

    problema = LinearProblem(
        a, L_form, bcs=[bc],
        petsc_options_prefix="viga_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    u_h = problema.solve()

    u_array   = u_h.x.array.reshape(-1, 3)
    delta_max = float(np.linalg.norm(u_array, axis=1).max())

    return delta_max, u_h, dominio, V


# ============================================================
# Interface blackbox
# ============================================================

def f(x_norm):
    """
    Objetivo: minimizar massa [kg].

    massa = rho * L * b * h
    rho e L são constantes do problema — não variáveis de otimização.
    E afeta a massa *indiretamente* via restrição de rigidez:
    E maior -> delta menor -> permite b,h menores -> massa menor.
    """
    E, b, h = x_to_params(x_norm)
    return RHO * L_VIGA * b * h


def g(x_norm):
    """
    Restrição de rigidez (g <= 0 -> viável):
        g[0] = delta_FEM / DELTA_MAX - 1
    """
    E, b, h = x_to_params(x_norm)
    delta_fem, _, _, _ = rodar_fem(E, b, h)
    return np.array([delta_fem / DELTA_MAX - 1.0])


h_eq = None  # sem restrições de igualdade


# ============================================================
# Verificação: FEM vs. analítico
# ============================================================

def verify_fem(verbose=True):
    E0, b0, h0 = x_to_params(x0)

    delta_fem, _, _, _ = rodar_fem(E0, b0, h0)
    delta_ref          = solucao_analitica(E0, b0, h0)
    erro               = abs(delta_fem - delta_ref) / delta_ref * 100

    massa = RHO * L_VIGA * b0 * h0
    g0    = g(x0)

    if verbose:
        print("=" * 55)
        print("  Verificação FEM x Analítico (Timoshenko)")
        print("=" * 55)
        print(f"  E   = {E0/1e9:.1f} GPa  (variável de projeto)")
        print(f"  b   = {b0*1e3:.0f} mm        (variável de projeto)")
        print(f"  h   = {h0*1e3:.0f} mm        (variável de projeto)")
        print(f"  nu  = {NU:.2f}           (fixo)")
        print(f"  rho = {RHO:.0f} kg/m³    (fixo)")
        print("-" * 55)
        print(f"  delta_FEM       = {delta_fem:.6e} m")
        print(f"  delta_analitico = {delta_ref:.6e} m")
        print(f"  Erro relativo   = {erro:.2f} %")
        print(f"  Massa           = {massa:.3f} kg")
        print(f"  DELTA_MAX       = {DELTA_MAX*1e3:.1f} mm")
        print("-" * 55)
        status = "VIOLADA" if g0[0] > 0 else "ok"
        print(f"  g[0] rigidez = {g0[0]:+.4f}  [{status}]")
        print("=" * 55)

    return erro


# ============================================================
# Exportação de resultados
#
# Padrão Dokken (2026) §"Saving functions to file" e
#                       §"Plotting the solution":
#   - VTXWriter (.bp / BP4)  para ParaView
#   - vtk_mesh + pyvista     para screenshots OFF_SCREEN
#
# Arquivos gerados em `pasta/`:
#   <label>_desl_<solver>.bp       — deslocamento vetorial (BP4)
#   <label>_pv_deformada_<solver>.png — warp_by_vector colorido por |u|
# ============================================================

def exportar_resultados(x_norm, solver, label="resultado", pasta="results"):
    import dolfinx.io as dio
    from dolfinx.plot import vtk_mesh
    from pathlib import Path
    import pyvista

    Path(pasta).mkdir(exist_ok=True, parents=True)
    base = Path(pasta) / label

    E, b, h = x_to_params(x_norm)
    massa   = RHO * L_VIGA * b * h

    print(f"\n[exportar_resultados] label='{label}'")
    print(f"  E={E/1e9:.2f} GPa  b={b*1e3:.1f} mm  h={h*1e3:.1f} mm  massa={massa:.3f} kg")

    # FEM completo (reutiliza rodar_fem que já devolve u_h, dominio, V)
    _, u_h, dominio, V = rodar_fem(E, b, h)
    u_h.name = "Deslocamento"

    # -- VTXWriter / BP4 -----------------------------------------
    path_bp = str(base) + f"_desl_{solver}.bp"
    with dio.VTXWriter(MPI.COMM_WORLD, path_bp, [u_h], engine="BP4") as vtx:
        vtx.write(0.0)
    print(f"  [VTX] {path_bp}")

    # -- PyVista OFF_SCREEN: viga deformada colorida por |u| -----
    pyvista.OFF_SCREEN = True

    topology, cell_types, coords = vtk_mesh(V)
    grid  = pyvista.UnstructuredGrid(topology, cell_types, coords)
    u_arr = u_h.x.array.reshape(-1, 3)
    grid.point_data["Deslocamento"] = u_arr
    grid.point_data["u_mag"]        = np.linalg.norm(u_arr, axis=1)

    warped = grid.warp_by_vector("Deslocamento", factor=200.0)
    warped.set_active_scalars("u_mag")

    pl = pyvista.Plotter()
    pl.add_mesh(warped, scalars="u_mag", cmap="coolwarm", show_edges=True,
                scalar_bar_args={"title": "|u| [m]"})
    pl.add_title(
        f"Deformada x200  |  E={E/1e9:.1f} GPa  b={b*1e3:.0f} mm  h={h*1e3:.0f} mm",
        font_size=9
    )
    pl.view_isometric()
    png = str(base) + f"_pv_deformada_{solver}.png"
    pl.screenshot(png)
    pl.close()
    print(f"  [PV]  {png}")

    print(f"  Exportação concluída → {pasta}/")


# ============================================================
# Execução direta
# ============================================================
if __name__ == "__main__":
    verify_fem()
    exportar_resultados(x0, solver="direto", label="ponto_inicial")