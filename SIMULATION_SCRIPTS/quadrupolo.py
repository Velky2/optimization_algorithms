# quadrupolo.py
# ============================================================
# FEM — Campo Magnético de um Quadrupolo com 4 Fios de Corrente
#
#   Referência FEM:
#     https://jsdokken.com/dolfinx-tutorial/chapter3/em.html
#   Referência exportação:
#     Dokken, J.S. (2026). FEniCSx Tutorial — Membrane implementation.
#     https://jsdokken.com/dolfinx-tutorial/chapter1/membrane_code.html
#
# Física:
#   -∇·(μ⁻¹ ∇A_z) = J_z   (Poisson magnético 2D)
#   B = (∂A_z/∂y, −∂A_z/∂x)
#   G = dB_y/dx|_(0,0)     (gradiente de campo — figura de mérito)
#
# Variáveis de projeto (normalizadas em [0,1]³):
#   x[0] -> d_wire  (distância centro→fio)   escala linear
#   x[1] -> r_wire  (raio do fio)            escala linear
#   x[2] -> J0      (densidade de corrente)  escala log
#
# Exporta:
#   f(x_norm)      -> escalar    (−G [T/m], minimizar ↔ maximizar G)
#   g(x_norm)      -> array(2,)  (restrições de desigualdade)
#   h              -> None
#   x0             -> array(3,)  (ponto de referência normalizado)
#   verify_fem()                 (compara FEM com estimativa analítica)
#   exportar_resultados()        (salva .bp + PNGs Matplotlib)
# ============================================================

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.tri as tri
from pathlib import Path
from scipy.spatial import cKDTree

from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import compute_midpoints, meshtags, locate_entities_boundary
from dolfinx.plot import vtk_mesh
from dolfinx import io as dio
import ufl
from ufl import TestFunction, TrialFunction, dot, grad, dx, as_vector

os.makedirs("results", exist_ok=True)

# ── Geometria fixa ────────────────────────────────────────────
R   = 0.1         # [m] metade do lado do domínio quadrado (abertura)
mu0 = 4 * np.pi * 1e-7  # [H/m]
N   = 200            # divisões da malha

# Configuração angular dos 4 fios (quadrupolo simétrico)
ANGLES = [np.radians(a) for a in [45, 135, 225, 315]]
SIGNS  = [+1, -1, +1, -1]

# ── Espaço de busca (escala física) ───────────────────────────
BOUNDS = {
    'd_min': 0.015,  'd_max': 0.070,   # distância centro→fio [m]
    'r_min': 0.003,  'r_max': 0.020,   # raio do fio [m]
    'J_min': 1e5,    'J_max': 1e7,     # densidade de corrente [A/m²]
}

# ── Restrições ────────────────────────────────────────────────
DOMAIN_TOL  = 0.002  # margem mínima até a borda: d + r < R − tol  [m]


# ============================================================
# Escala de variáveis
# ============================================================

def x_to_params(x_norm):
    """[0,1]³ → (d_wire [m], r_wire [m], J0 [A/m²])"""
    d = BOUNDS['d_min'] + x_norm[0] * (BOUNDS['d_max'] - BOUNDS['d_min'])
    r = BOUNDS['r_min'] + x_norm[1] * (BOUNDS['r_max'] - BOUNDS['r_min'])
    J = 10 ** (np.log10(BOUNDS['J_min']) +
               x_norm[2] * (np.log10(BOUNDS['J_max']) - np.log10(BOUNDS['J_min'])))
    return d, r, J


def params_to_x(d_wire, r_wire, J0):
    """(d_wire, r_wire, J0) → [0,1]³"""
    x = np.array([
        (d_wire - BOUNDS['d_min']) / (BOUNDS['d_max'] - BOUNDS['d_min']),
        (r_wire - BOUNDS['r_min']) / (BOUNDS['r_max'] - BOUNDS['r_min']),
        (np.log10(J0)      - np.log10(BOUNDS['J_min'])) /
        (np.log10(BOUNDS['J_max']) - np.log10(BOUNDS['J_min'])),
    ])
    return np.clip(x, 0.0, 1.0)


# ============================================================
# Estimativa analítica do gradiente (fio infinito)
# ============================================================

def gradiente_analitico(d_wire, r_wire, J0):
    """
    Gradiente de campo no centro de um quadrupolo de 4 fios circulares.

    Cada fio de raio r transporta corrente I = J0 · π r².
    Somando as contribuições dos 4 fios em configuração quadrupolar:
        G ≈ 4 · μ0 I / (2π d²) = 2 μ0 J0 r² / d²

    Referência: Wiedemann, H. (2015). Particle Accelerator Physics, §10.2.
    """
    I = J0 * np.pi * r_wire**2
    return 4 * mu0 * I / (2 * np.pi * d_wire**2)


# ============================================================
# Núcleo FEM
# ============================================================

def rodar_fem(d_wire, r_wire, J0):
    """
    Resolve -∇·(μ⁻¹ ∇A_z) = J_z no domínio [-R,R]².

    Retorna
    -------
    G      : float     Gradiente dB_y/dx no centro [T/m]
    A_z    : Function  Potencial vetorial (Lagrange P1)
    B_fem  : Function  Campo B (DG0 vetorial)
    dominio: Mesh
    """
    wire_centers = [(d_wire * np.cos(a), d_wire * np.sin(a)) for a in ANGLES]

    # Malha
    dominio = mesh.create_rectangle(
        MPI.COMM_WORLD,
        points=[[-R, -R], [R, R]],
        n=[N, N],
        cell_type=mesh.CellType.triangle,
    )

    # Subdomínios dos fios
    tdim      = dominio.topology.dim
    num_cells = dominio.topology.index_map(tdim).size_local
    dominio.topology.create_connectivity(tdim, tdim)
    all_cells = np.arange(num_cells, dtype=np.int32)
    midpts    = compute_midpoints(dominio, tdim, all_cells)

    cell_tags_array = np.zeros(num_cells, dtype=np.int32)
    for i, (cx, cy) in enumerate(wire_centers):
        dist = np.hypot(midpts[:, 0] - cx, midpts[:, 1] - cy)
        cell_tags_array[dist < r_wire] = i + 1
    ct = meshtags(dominio, tdim, all_cells, cell_tags_array)

    # Parâmetros físicos por subdomínio
    Q  = fem.functionspace(dominio, ("DG", 0))
    mu = fem.Function(Q); mu.x.array[:] = mu0
    Jf = fem.Function(Q); Jf.x.array[:] = 0.0
    for i in range(4):
        cells = ct.find(i + 1)
        if len(cells) > 0:
            Jf.x.array[cells] = SIGNS[i] * J0
    print("\n=== Diagnóstico J ===")

    for i in range(4):
        cells = ct.find(i + 1)
        print(
            f"fio {i+1}: "
            f"{len(cells)} células, "
            f"J = {SIGNS[i]*J0:.3e} A/m²"
        )

    print("J min =", Jf.x.array.min())
    print("J max =", Jf.x.array.max())
    print("células não nulas =", np.count_nonzero(Jf.x.array))

    # Formulação fraca: (1/μ) ∇A_z · ∇v = J v
    V      = fem.functionspace(dominio, ("Lagrange", 1))
    facets = locate_entities_boundary(dominio, tdim - 1,
                                      lambda x: np.full(x.shape[1], True))
    dofs   = fem.locate_dofs_topological(V, tdim - 1, facets)
    bc     = fem.dirichletbc(default_scalar_type(0), dofs, V)

    u, v = TrialFunction(V), TestFunction(V)
    a    = (1 / mu) * dot(grad(u), grad(v)) * dx
    L    = Jf * v * dx

    A_z = fem.Function(V, name="A_z")
    LinearProblem(
        a, L, u=A_z, bcs=[bc],
        petsc_options_prefix="quad_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    ).solve()

    print("\n=== Diagnóstico A_z ===")
    print("max |A_z| =", np.max(np.abs(A_z.x.array)))
    print("min A_z   =", np.min(A_z.x.array))
    print("max A_z   =", np.max(A_z.x.array))

    # Campo B = (∂A_z/∂y, −∂A_z/∂x)
    W     = fem.functionspace(dominio, ("DG", 0, (dominio.geometry.dim,)))
    B_fem = fem.Function(W, name="B")
    B_fem.interpolate(
        fem.Expression(as_vector((A_z.dx(1), -A_z.dx(0))),
                       W.element.interpolation_points)
    )

    # Gradiente G = dB_y/dx por regressão linear perto do centro
    # Usa componente y do campo (índice 1) ao longo do eixo x,
    # conforme definição G = dB_y/dx|_(0,0).
    num_cells_total = (dominio.topology.index_map(tdim).size_local +
                       dominio.topology.index_map(tdim).num_ghosts)
    midpts_all = compute_midpoints(
        dominio, tdim, np.arange(num_cells_total, dtype=np.int32))
    W = B_fem.function_space

    num_dofs = (
        W.dofmap.index_map.size_local +
        W.dofmap.index_map.num_ghosts
    )
    B_vals = B_fem.x.array.real.reshape(num_dofs, 2)

    print("\n=== Diagnóstico B ===")
    print("shape(B_fem.x.array) =", B_fem.x.array.shape)
    print("num_dofs             =", num_dofs)
    print("len(midpts_all)      =", len(midpts_all))
    print("max |Bx| =", np.max(np.abs(B_vals[:, 0])))
    print("max |By| =", np.max(np.abs(B_vals[:, 1])))

    mesh_h = 2 * R / N

    # Seleciona células próximas ao eixo x (|y| < 2h) e perto do centro
    mask_x = (
        (np.abs(midpts_all[:, 1]) < 2 * mesh_h) &
        (np.abs(midpts_all[:, 0]) < d_wire * 0.3)
    )

    x_fit  = midpts_all[mask_x, 0]
    By_fit = B_vals[mask_x, 0]    

    coef = np.polyfit(x_fit, By_fit, 1)

    G = float(coef[0])

    return G, A_z, B_fem, dominio




# ============================================================
# Interface blackbox
# ============================================================

def f(x_norm):
    """Figura de mérito: −G [T/m]. Minimizar ↔ maximizar gradiente."""
    d, r, J = x_to_params(x_norm)
    G, *_ = rodar_fem(d, r, J)
    return -G


def g(x_norm):
    """
    Restrições de desigualdade g(x) ≤ 0.

    g[0] = r_wire − d_wire/√2       : fios adjacentes não se sobrepõem.
                                       A distância entre dois fios vizinhos
                                       (separados por 90°) é d·√2, portanto
                                       a condição de não-sobreposição é
                                       2r ≤ d·√2  →  r ≤ d/√2.
                                       (mais restritiva que r ≤ d)
    g[1] = (d_wire + r_wire) − (R − DOMAIN_TOL) : fios dentro do domínio
    """
    d, r, _ = x_to_params(x_norm)
    return np.array([
        r - d / np.sqrt(2),
        (d + r) - (R - DOMAIN_TOL),
    ])


# Sem restrições de igualdade
h = None

# Ponto de referência: quadrupolo típico de laboratório
x0 = params_to_x(d_wire=0.04, r_wire=0.008, J0=1e6)


# ============================================================
# Verificação: FEM vs. Analítico
# ============================================================

def verify_fem(verbose=True):
    """
    Compara o gradiente FEM com a estimativa analítica no ponto x0.
    Erro relativo esperado < 10 % (fio finito vs. fio infinito).

    Retorna o erro relativo percentual.
    """
    d, r, J = x_to_params(x0)
    G_fem, *_ = rodar_fem(d, r, J)
    G_ref     = gradiente_analitico(d, r, J)
    erro      = abs(G_fem - G_ref) / abs(G_ref) * 100
    g0        = g(x0)

    if verbose:
        print("=" * 52)
        print("  Verificação FEM × Analítico (Quadrupolo)")
        print("=" * 52)
        print(f"  d_wire = {d*100:.2f} cm,  r_wire = {r*100:.2f} cm,"
              f"  J0 = {J:.2e} A/m²")
        print(f"  G_FEM       = {G_fem:.4f} T/m")
        print(f"  G_analítico = {G_ref:.4f} T/m")
        print(f"  Erro relativo = {erro:.2f} %")
        print(f"  f(x0) = {-G_fem:.4f}  (−G)")
        print(f"  g[0] (r − d/√2)              = {g0[0]:.4f}"
              f"  ({'violada' if g0[0]>0 else 'ok'})")
        print(f"  g[1] (d+r − (R−tol))        = {g0[1]:.4f}"
              f"  ({'violada' if g0[1]>0 else 'ok'})")
        print("=" * 52)

    return erro


# ============================================================
# Exportação de resultados
#
# Padrão Dokken (2026) §"Saving functions to file" e
#                       §"Plotting the solution":
#   - VTXWriter(.bp / BP4)   para ParaView
#   - Matplotlib             para análise (contornos, vetores, gradiente)
# ============================================================

def exportar_resultados(x_norm, solver, label="resultado", pasta="results"):
    """
    Executa o FEM para x_norm e salva os resultados.

    Arquivos gerados em `pasta/`:
      <label>_Az.bp          — potencial A_z  (VTXWriter/BP4)
      <label>_B.bp           — campo B        (VTXWriter/BP4)
      <label>_Az.png         — contorno de A_z + linhas de nível
      <label>_B.png          — campo vetorial B
      <label>_grad.png       — perfis de B e ajuste linear (G)
    """
    Path(pasta).mkdir(exist_ok=True, parents=True)
    base = Path(pasta) / label

    d, r, J = x_to_params(x_norm)
    print(f"\n[exportar_resultados] label='{label}'")
    print(f"  d={d*100:.2f} cm  r={r*100:.2f} cm  J0={J:.2e} A/m²")

    G, A_z, B_fem, dominio = rodar_fem(d, r, J)
    print(f"  G = {G:.4f} T/m")

    wire_centers = [(d * np.cos(a), d * np.sin(a)) for a in ANGLES]

    # -- VTXWriter / BP4 (Dokken 2026 §"Saving functions to file") --
    for path, fields in [
        (str(base) + f"_Az_{solver}.bp", [A_z]),
        (str(base) + f"_B_{solver}.bp",  [B_fem]),
    ]:
        with dio.VTXWriter(MPI.COMM_WORLD, path, fields, engine="BP4") as vtx:
            vtx.write(0.0)
        print(f"  [VTX] {path}")

    # -- Dados auxiliares para Matplotlib --
    topology, _, coords = vtk_mesh(dominio, dominio.topology.dim)
    cells_tri = topology.reshape(-1, 4)[:, 1:]
    triang    = tri.Triangulation(coords[:, 0], coords[:, 1], cells_tri)
    Az_nodes  = A_z.x.array.real

    tdim = dominio.topology.dim
    num_cells_total = (dominio.topology.index_map(tdim).size_local +
                       dominio.topology.index_map(tdim).num_ghosts)
    midpts_all = compute_midpoints(
        dominio, tdim, np.arange(num_cells_total, dtype=np.int32))
    W = B_fem.function_space

    num_dofs = (
        W.dofmap.index_map.size_local +
        W.dofmap.index_map.num_ghosts
    )
    B_vals = B_fem.x.array.real.reshape(num_dofs, 2)
    Bmag   = np.hypot(B_vals[:, 0], B_vals[:, 1])
    xm, ym = midpts_all[:, 0], midpts_all[:, 1]

    def _wire_markers(ax):
        for i, (cx, cy) in enumerate(wire_centers):
            color = "red" if SIGNS[i] > 0 else "blue"
            ax.add_patch(plt.Circle((cx, cy), r, color=color, alpha=0.85, zorder=5))
            ax.text(cx, cy, "⊙" if SIGNS[i] > 0 else "⊗",
                    ha="center", va="center", fontsize=10,
                    color="white", fontweight="bold", zorder=6)

    # -- Matplotlib 1: Potencial A_z --
    fig, ax = plt.subplots(figsize=(7, 7))
    img = ax.tricontourf(triang, Az_nodes, levels=50, cmap="RdBu_r")
    ax.tricontour(triang, Az_nodes, levels=25, colors="k", linewidths=0.4, alpha=0.5)
    plt.colorbar(img, ax=ax, label=r"$A_z$ [T·m]")
    _wire_markers(ax)
    ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_aspect("equal")
    ax.set_title(r"Potencial Vetorial Magnético $A_z$", fontsize=13)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.legend(handles=[
        mpatches.Patch(facecolor="red",  alpha=0.8, label="+J (saindo)"),
        mpatches.Patch(facecolor="blue", alpha=0.8, label="−J (entrando)"),
    ], loc="upper right", fontsize=9)
    plt.tight_layout()
    png = str(base) + f"_Az_{solver}.png"
    plt.savefig(png, dpi=150); plt.close()
    print(f"  [MPL] {png}")

    # -- Matplotlib 2: Campo B vetorial --
    r_mid = np.hypot(xm, ym)
    mask  = (Bmag > 0) & (r_mid < R * 0.95)
    step  = max(1, mask.sum() // 700)
    sel   = np.where(mask)[0][::step]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.tricontourf(triang, Az_nodes, levels=50, cmap="RdBu_r", alpha=0.5)
    ax.quiver(xm[sel], ym[sel],
              B_vals[sel, 0] / (Bmag[sel] + 1e-30),
              B_vals[sel, 1] / (Bmag[sel] + 1e-30),
              Bmag[sel], cmap="plasma", scale=60, width=0.003, alpha=0.85)
    _wire_markers(ax)
    ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_aspect("equal")
    ax.set_title(r"Campo Magnético $\vec{B}$ — Quadrupolo de 4 Fios", fontsize=13)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    plt.tight_layout()
    png = str(base) + f"_B_{solver}.png"
    plt.savefig(png, dpi=150); plt.close()
    print(f"  [MPL] {png}")

    # -- Matplotlib 3: Perfis de B e gradiente --
    tree   = cKDTree(midpts_all[:, :2])
    x_line = np.linspace(-d * 0.8, d * 0.8, 400)
    y_line = np.linspace(-d * 0.8, d * 0.8, 400)
    _, ix  = tree.query(np.column_stack([x_line, np.zeros_like(x_line)]))
    _, iy  = tree.query(np.column_stack([np.zeros_like(y_line), y_line]))

    mc_x  = np.abs(x_line) < d * 0.35
    mc_y  = np.abs(y_line) < d * 0.35
    G_x   = np.polyfit(x_line[mc_x], B_vals[ix, 1][mc_x], 1)[0]
    G_y   = np.polyfit(y_line[mc_y], B_vals[iy, 0][mc_y], 1)[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(x_line * 100, B_vals[ix, 1], "b-",  lw=2, label=r"$B_y(x,\,0)$")
    ax.plot(x_line * 100, B_vals[ix, 0], "r--", lw=2, label=r"$B_x(x,\,0)$")
    fit_x = np.polyval(np.polyfit(x_line[mc_x], B_vals[ix, 1][mc_x], 1), x_line)
    ax.plot(x_line[mc_x] * 100, fit_x[mc_x], "k:", lw=2,
            label=f"Ajuste G = {G_x:.2f} T/m")
    ax.axhline(0, color="gray", lw=0.8); ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("x [cm]"); ax.set_ylabel("B [T]")
    ax.set_title("Perfil — eixo x"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(y_line * 100, B_vals[iy, 0], "r-",  lw=2, label=r"$B_x(0,\,y)$")
    ax.plot(y_line * 100, B_vals[iy, 1], "b--", lw=2, label=r"$B_y(0,\,y)$")
    fit_y = np.polyval(np.polyfit(y_line[mc_y], B_vals[iy, 0][mc_y], 1), y_line)
    ax.plot(y_line[mc_y] * 100, fit_y[mc_y], "k:", lw=2,
            label=f"Ajuste G = {G_y:.2f} T/m")
    ax.axhline(0, color="gray", lw=0.8); ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("y [cm]"); ax.set_ylabel("B [T]")
    ax.set_title("Perfil — eixo y"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle(f"Gradiente — Quadrupolo Magnético (FEM)  G={G:.2f} T/m ({solver})",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    png = str(base) + f"_grad_{solver}.png"
    plt.savefig(png, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [MPL] {png}")

    print(f"  Exportação concluída → {pasta}/")


# ============================================================
# Execução direta
# ============================================================
if __name__ == "__main__":
    verify_fem()
    exportar_resultados(x0, solver="lu", label="ref_inicial")