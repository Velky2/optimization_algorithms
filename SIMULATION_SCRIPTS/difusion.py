# difusao_otimizacao.py  (v2 — figura de mérito localizada)
# ============================================================
# FEM — Difusão Transiente de um Pulso Gaussiano com Controle
#
#   Referência FEM:
#     https://jsdokken.com/dolfinx-tutorial/chapter2/heat_equation.html
#
# Física:
#   ∂u/∂t − α ∇²u = 0    no domínio [-L, L]²
#   u(x, 0)       = exp(−a (x² + y²))   (pulso gaussiano)
#   u = 0 em ∂Ω  (Dirichlet homogêneo)
#
# Figura de mérito (v2 — LOCALIZADA):
#   J(x_norm) = ∫_{Ω_alvo} u(x, T; α, a, T) dx
#
#   Ω_alvo = disco de raio R_alvo centrado na origem.
#   Maximizar J ↔ minimizar f = −J
#
#   Aplicação prática: dado um pulso gaussiano de concentração
#   (fármaco, dopante, poluente), encontrar a combinação de
#   difusividade α, concentração inicial a e tempo de processo T
#   que maximiza a substância retida na região de interesse
#   Ω_alvo ao final do processo.
#
#   Contraste com v1 (massa total): aqui o ótimo NÃO é
#   simplesmente "minimize α" — um pulso muito concentrado
#   (a grande) pode vazar para fora de Ω_alvo mesmo com α
#   pequeno se a gaussiana inicial tiver cauda fora do alvo.
#   O otimizador precisa equilibrar α, a e T simultaneamente.
#
# Variáveis de projeto (normalizadas em [0,1]³):
#   x_norm[0] → α   (difusividade)       escala log  [α_min, α_max]
#   x_norm[1] → a   (concentração gauss) escala linear [a_min, a_max]
#   x_norm[2] → T   (tempo final)        escala linear [T_min, T_max]
#
# Restrições de desigualdade g(x) ≤ 0:
#   g[0] = α · T − DIFUSAO_MAX  : produto α·T limitado
#   g[1] = DIFUSAO_MIN − α · T  : difusão mínima não trivial
#
# Restrição de igualdade h(x) = 0:
#   h[0] = ∫_Ω u(x, 0) dx − π/a  (conservação da massa inicial)
#   (satisfeita analiticamente; incluída para validação)
#
# Exporta:
#   f(x_norm)          → escalar     (−J_alvo, minimizar)
#   g(x_norm)          → array(2,)   (restrições de desigualdade)
#   h(x_norm)          → array(1,)   (restrição de igualdade)
#   x0                 → array(3,)   (ponto de referência)
#   verify_fem()                     (FEM vs. analítico global)
#   exportar_resultados()            (BP4 estado inicial/final + PNGs 2D/3D)
# ============================================================

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh, io as dio, plot
from dolfinx.fem.petsc import (
    assemble_vector,
    assemble_matrix,
    create_vector,
    apply_lifting,
    set_bc,
)
import ufl

os.makedirs("results", exist_ok=True)

# ── Geometria fixa ────────────────────────────────────────────
L      = 2.0      # [m] metade do lado do domínio
NX, NY = 50, 50   # divisões da malha
NSTEPS = 50       # número de passos de tempo

# ── Região de interesse Ω_alvo ────────────────────────────────
# Disco de raio R_alvo centrado na origem.
# Fisicamente: região onde queremos reter a substância.
# Escolha R_alvo = 0.5 m ≈ σ_max do pulso mais largo (a=1 →
# σ = 1/√2 ≈ 0.71 m), garantindo que o alvo é menor que o
# pulso — o problema de retenção é não-trivial.
R_ALVO = 0.5      # [m]

# ── Espaço de busca (escala física) ───────────────────────────
BOUNDS = {
    'alpha_min': 1e-3, 'alpha_max': 0.4,    # difusividade [m²/s]
    'a_min':     1.0,  'a_max':    10.0,    # concentração gaussiana [1/m²]
    'T_min':     0.1,  'T_max':    2.0,     # tempo final [s]
}

# ── Restrições de desigualdade ────────────────────────────────
DIFUSAO_MAX = 0.8    # α·T máximo
DIFUSAO_MIN = 0.05   # α·T mínimo


# ── Massa inicial analítica ────────────────────────────────────
def massa_inicial_analitica(a):
    """∫_{R²} exp(−a(x²+y²)) dx dy = π/a"""
    return np.pi / a


# ── Massa no alvo analítica (t=0) ─────────────────────────────
def massa_alvo_analitica_t0(a, R=R_ALVO):
    """
    ∫_{disco R} exp(−a(x²+y²)) dx dy = π/a · (1 − exp(−a R²))

    Derivação: integração em coordenadas polares.
    """
    return (np.pi / a) * (1.0 - np.exp(-a * R**2))


# ============================================================
# Escala de variáveis
# ============================================================

def x_to_params(x_norm):
    """[0,1]³ → (alpha [m²/s], a [1/m²], T [s])"""
    alpha = 10 ** (
        np.log10(BOUNDS['alpha_min']) +
        x_norm[0] * (np.log10(BOUNDS['alpha_max']) -
                     np.log10(BOUNDS['alpha_min']))
    )
    a = BOUNDS['a_min'] + x_norm[1] * (BOUNDS['a_max'] - BOUNDS['a_min'])
    T = BOUNDS['T_min'] + x_norm[2] * (BOUNDS['T_max'] - BOUNDS['T_min'])
    return alpha, a, T


def params_to_x(alpha, a, T):
    """(alpha, a, T) → [0,1]³"""
    x = np.array([
        (np.log10(alpha) - np.log10(BOUNDS['alpha_min'])) /
        (np.log10(BOUNDS['alpha_max']) - np.log10(BOUNDS['alpha_min'])),
        (a - BOUNDS['a_min']) / (BOUNDS['a_max'] - BOUNDS['a_min']),
        (T - BOUNDS['T_min']) / (BOUNDS['T_max'] - BOUNDS['T_min']),
    ])
    return np.clip(x, 0.0, 1.0)


# ============================================================
# Solução analítica global (domínio infinito, referência)
# ============================================================

def massa_analitica(alpha, a, T):
    """
    Massa total retida em t=T para u₀ = exp(−a(x²+y²)) em R²:
        J(T) = π / (a (1 + 4αaT))

    Usada apenas para verificação FEM vs analítico.
    """
    return np.pi / (a * (1.0 + 4.0 * alpha * a * T))


# ============================================================
# Núcleo FEM
# ============================================================

def rodar_fem(alpha, a, T, salvar_bp=False, pasta="results", label="sim", solver_label="ref"):
    """
    Resolve ∂u/∂t − α ∇²u = 0 no domínio [-L,L]².

    Método: Euler implícito (incondicionalmente estável).

    Parâmetros
    ----------
    salvar_bp   : bool   Se True, salva BP4 do estado t=0 e t=T via VTXWriter.
                         Usado apenas em exportar_resultados() para evitar
                         I/O desnecessário durante a otimização.

    Retorna
    -------
    J_alvo  : float      ∫_{Ω_alvo} u(·,T) dx  (figura de mérito)
    J_total : float      ∫_Ω u(·,T) dx          (massa total, para referência)
    uh      : Function   Solução em t=T
    u0_arr  : ndarray    Valores nodais em t=0  (para plots)
    dominio : Mesh
    """
    dt = T / NSTEPS

    dominio = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([-L, -L]), np.array([L, L])],
        [NX, NY],
        mesh.CellType.triangle,
    )
    V = fem.functionspace(dominio, ("Lagrange", 1))

    # Condição inicial u₀ = exp(−a(x²+y²))
    def ci(x):
        return np.exp(-a * (x[0]**2 + x[1]**2))

    u_n = fem.Function(V, name="u_n")
    u_n.interpolate(ci)
    uh  = fem.Function(V, name="u")
    uh.interpolate(ci)

    # Guarda valores nodais de t=0 para plots
    u0_arr = uh.x.array.copy()

    # Condição de contorno Dirichlet homogênea
    fdim = dominio.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        dominio, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
    )
    bc = fem.dirichletbc(
        PETSc.ScalarType(0),
        fem.locate_dofs_topological(V, fdim, boundary_facets),
        V,
    )

    # Salva estado inicial (t=0) em BP4
    if salvar_bp:
        Path(pasta).mkdir(exist_ok=True, parents=True)
        u_init = fem.Function(V, name="u")
        u_init.x.array[:] = u0_arr
        path_t0 = str(Path(pasta) / f"{label}_t0_{solver_label}.bp")
        with dio.VTXWriter(MPI.COMM_WORLD, path_t0, [u_init], engine="BP4") as vtx:
            vtx.write(0.0)
        print(f"  [VTX] {path_t0}  (t=0)")

    # Formulação fraca: (u − u_n)/dt · v + α ∇u · ∇v = 0
    u, v   = ufl.TrialFunction(V), ufl.TestFunction(V)
    a_form = (u * v * ufl.dx
              + alpha * dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    L_form = u_n * v * ufl.dx

    bilinear_form = fem.form(a_form)
    linear_form   = fem.form(L_form)

    A = assemble_matrix(bilinear_form, bcs=[bc])
    A.assemble()
    b = create_vector(fem.extract_function_spaces(linear_form))

    ksp = PETSc.KSP().create(dominio.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.getPC().setType(PETSc.PC.Type.LU)

    # Marcha temporal
    for _ in range(NSTEPS):
        with b.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b, linear_form)
        apply_lifting(b, [bilinear_form], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                      mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, [bc])
        ksp.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        u_n.x.array[:] = uh.x.array

    # Figura de mérito: ∫_{Ω_alvo} u(·,T) dx
    # Ω_alvo = {(x,y) : x²+y² ≤ R_ALVO²}
    # Implementado via função indicadora interpolada em DG0
    Q       = fem.functionspace(dominio, ("DG", 0))
    ind_fn  = fem.Function(Q)
    tdim    = dominio.topology.dim
    num_c   = dominio.topology.index_map(tdim).size_local
    from dolfinx.mesh import compute_midpoints
    midpts  = compute_midpoints(dominio, tdim,
                                np.arange(num_c, dtype=np.int32))
    ind_fn.x.array[:num_c] = (
        midpts[:, 0]**2 + midpts[:, 1]**2 <= R_ALVO**2
    ).astype(float)

    J_alvo_form = fem.form(uh * ind_fn * ufl.dx)
    J_alvo      = float(fem.assemble_scalar(J_alvo_form))

    J_total_form = fem.form(uh * ufl.dx)
    J_total      = float(fem.assemble_scalar(J_total_form))

    # Salva estado final (t=T) em BP4
    if salvar_bp:
        path_tT = str(Path(pasta) / f"{label}_tT_{solver_label}.bp")
        with dio.VTXWriter(MPI.COMM_WORLD, path_tT, [uh], engine="BP4") as vtx:
            vtx.write(T)
        print(f"  [VTX] {path_tT}  (t=T={T:.2f}s)")

    A.destroy(); b.destroy(); ksp.destroy()

    return J_alvo, J_total, uh, u0_arr, dominio


# ============================================================
# Interface blackbox
# ============================================================

def f(x_norm):
    """
    Figura de mérito: −J_alvo  [m²].
    Minimizar f ↔ maximizar a massa retida em Ω_alvo ao tempo T.
    """
    alpha, a, T = x_to_params(x_norm)
    J_alvo, *_ = rodar_fem(alpha, a, T)
    return -J_alvo


def g(x_norm):
    """
    Restrições de desigualdade  g(x) ≤ 0.

    g[0] = α·T − DIFUSAO_MAX  : limita difusão excessiva
    g[1] = DIFUSAO_MIN − α·T  : garante difusão mínima não trivial
    """
    alpha, a, T = x_to_params(x_norm)
    produto = alpha * T
    return np.array([
        produto - DIFUSAO_MAX,
        DIFUSAO_MIN - produto,
    ])


def h(x_norm):
    """
    Restrição de igualdade  h(x) = 0.

    h[0] = massa_inicial_FEM − π/a
    Verifica que a condição inicial está corretamente normalizada.
    (Na prática sempre satisfeita por construção; inclusa para
    conformidade com a interface do Lagrangiano Aumentado.)
    """
    alpha, a, T = x_to_params(x_norm)
    dominio = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([-L, -L]), np.array([L, L])],
        [NX, NY],
        mesh.CellType.triangle,
    )
    V  = fem.functionspace(dominio, ("Lagrange", 1))
    u0 = fem.Function(V)
    u0.interpolate(lambda x: np.exp(-a * (x[0]**2 + x[1]**2)))
    J0_fem = float(fem.assemble_scalar(fem.form(u0 * ufl.dx)))
    J0_ref = massa_inicial_analitica(a)
    return np.array([J0_fem - J0_ref])


# Ponto de referência: difusividade moderada, gaussiana típica
x0 = params_to_x(alpha=0.05, a=5.0, T=1.0)


# ============================================================
# Verificação: FEM vs. Analítico
# ============================================================

def verify_fem(verbose=True):
    """
    Compara a massa total FEM com a solução analítica global
    (domínio infinito). Erro relativo esperado < 5 %.

    Também reporta J_alvo e a fração de massa retida no alvo.
    """
    alpha, a, T = x_to_params(x0)
    J_alvo, J_total, *_ = rodar_fem(alpha, a, T)
    J_ref  = massa_analitica(alpha, a, T)
    J_t0   = massa_alvo_analitica_t0(a)
    erro   = abs(J_total - J_ref) / abs(J_ref) * 100
    g0     = g(x0)

    if verbose:
        print("=" * 60)
        print("  Verificação FEM × Analítico (Difusão Gaussiana v2)")
        print("=" * 60)
        print(f"  α = {alpha:.4f} m²/s,  a = {a:.2f} 1/m²,  T = {T:.2f} s")
        print(f"  R_alvo = {R_ALVO} m")
        print()
        print(f"  J_total FEM       = {J_total:.6f} m²")
        print(f"  J_total analítico = {J_ref:.6f} m²  (domínio infinito)")
        print(f"  Erro relativo     = {erro:.2f} %")
        print()
        print(f"  J_alvo  (t=T)     = {J_alvo:.6f} m²")
        print(f"  J_alvo  (t=0)     = {J_t0:.6f} m²  (analítico)")
        print(f"  Retenção no alvo  = {J_alvo/J_t0*100:.1f} % da massa inicial")
        print()
        print(f"  f(x0) = {-J_alvo:.6f}  (−J_alvo)")
        print(f"  g[0] (α·T − {DIFUSAO_MAX}) = {g0[0]:.4f}"
              f"  ({'violada' if g0[0] > 0 else 'ok'})")
        print(f"  g[1] ({DIFUSAO_MIN} − α·T) = {g0[1]:.4f}"
              f"  ({'violada' if g0[1] > 0 else 'ok'})")
        print("=" * 60)

    return erro


# ============================================================
# Exportação de resultados
# ============================================================

def exportar_resultados(x_norm, solver="ref", label="resultado",
                        pasta="results"):
    """
    Executa o FEM para x_norm e salva os resultados.

    Arquivos gerados em `pasta/`:
      <label>_t0_<solver>.bp     — estado inicial t=0   (VTXWriter/BP4)
      <label>_tT_<solver>.bp     — estado final   t=T   (VTXWriter/BP4)
      <label>_2d_<solver>.png    — mapas de calor 2D (t=0 | t=T) + contorno Ω_alvo
      <label>_3d_<solver>.png    — superfícies 3D  (t=0 | t=T)
      <label>_perfis_<solver>.png— perfis u(x,0) comparando t=0 e t=T

    Os arquivos .bp contêm a solução completa e podem ser
    visualizados no ParaView com representação 3D (Warp by Scalar).
    """
    Path(pasta).mkdir(exist_ok=True, parents=True)
    base = Path(pasta) / label

    alpha, a, T = x_to_params(x_norm)
    print(f"\n[exportar_resultados] label='{label}'")
    print(f"  α={alpha:.4e} m²/s  a={a:.2f} 1/m²  T={T:.2f} s")
    print(f"  R_alvo = {R_ALVO} m")

    # Roda FEM com salvamento BP4 de t=0 e t=T
    J_alvo, J_total, uh, u0_arr, dominio = rodar_fem(
        alpha, a, T,
        salvar_bp=True,
        pasta=pasta,
        label=label,
        solver_label=solver,
    )
    J_t0 = massa_alvo_analitica_t0(a)
    print(f"  J_alvo(t=T) = {J_alvo:.6f} m²")
    print(f"  J_alvo(t=0) = {J_t0:.6f} m²  (analítico)")
    print(f"  Retenção    = {J_alvo/J_t0*100:.1f}%")

    # ── Dados auxiliares para Matplotlib ──────────────────────
    topology, _, coords = plot.vtk_mesh(dominio, dominio.topology.dim)
    cells_tri = topology.reshape(-1, 4)[:, 1:]
    triang    = tri.Triangulation(coords[:, 0], coords[:, 1], cells_tri)
    uT_vals   = uh.x.array.real
    u0_vals   = u0_arr.real

    vmax = u0_vals.max()

    # Círculo da região Ω_alvo para overlay
    theta_circ = np.linspace(0, 2 * np.pi, 200)
    xc = R_ALVO * np.cos(theta_circ)
    yc = R_ALVO * np.sin(theta_circ)

    # ── PNG 1: mapas de calor 2D (t=0 | t=T) ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, vals, titulo, tempo in [
        (axes[0], u0_vals, f"t = 0 s", 0.0),
        (axes[1], uT_vals, f"t = T = {T:.2f} s", T),
    ]:
        img = ax.tricontourf(triang, vals, levels=50,
                             cmap="viridis", vmin=0, vmax=vmax)
        ax.tricontour(triang, vals, levels=10,
                      colors="w", linewidths=0.4, alpha=0.4)
        plt.colorbar(img, ax=ax, label="u")
        # Contorno do alvo
        ax.plot(xc, yc, "r--", lw=1.5, label=f"Ω_alvo (R={R_ALVO} m)")
        ax.set_xlim(-L, L); ax.set_ylim(-L, L)
        ax.set_aspect("equal")
        ax.set_title(titulo, fontsize=12)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.legend(fontsize=9, loc="upper right")

    J_label = f"J_alvo(T) = {J_alvo:.4f} m²  ({J_alvo/J_t0*100:.1f}% retido)"
    fig.suptitle(
        f"Difusão Gaussiana — α={alpha:.3e}  a={a:.1f}  T={T:.2f} s\n{J_label}",
        fontsize=12,
    )
    plt.tight_layout()
    png_2d = str(base) + f"_2d_{solver}.png"
    plt.savefig(png_2d, dpi=150); plt.close()
    print(f"  [MPL] {png_2d}")

    # ── PNG 2: superfícies 3D (t=0 | t=T) ────────────────────
    # Interpolação em grade regular para plot_surface limpo
    Ng     = 80
    xi     = np.linspace(-L, L, Ng)
    yi     = np.linspace(-L, L, Ng)
    Xi, Yi = np.meshgrid(xi, yi)

    import matplotlib.tri as mtri
    interp0 = mtri.LinearTriInterpolator(triang, u0_vals)
    interpT = mtri.LinearTriInterpolator(triang, uT_vals)
    Z0 = np.clip(interp0(Xi, Yi).filled(0), 0, None)
    ZT = np.clip(interpT(Xi, Yi).filled(0), 0, None)

    fig = plt.figure(figsize=(14, 6))

    for idx, (Z, titulo, tempo) in enumerate([
        (Z0, f"t = 0 s",           0.0),
        (ZT, f"t = T = {T:.2f} s", T),
    ]):
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
        surf = ax.plot_surface(Xi, Yi, Z, cmap="viridis",
                               vmin=0, vmax=vmax,
                               linewidth=0, antialiased=True, alpha=0.92)
        # Projeção do contorno Ω_alvo no plano z=0
        ax.plot(xc, yc, np.zeros_like(xc),
                "r--", lw=1.5, label=f"Ω_alvo")
        fig.colorbar(surf, ax=ax, shrink=0.5, label="u")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("u")
        ax.set_title(titulo, fontsize=11)
        ax.set_zlim(0, vmax * 1.05)
        ax.view_init(elev=30, azim=-60)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        f"Superfície 3D — α={alpha:.3e}  a={a:.1f}  T={T:.2f} s",
        fontsize=12,
    )
    plt.tight_layout()
    png_3d = str(base) + f"_3d_{solver}.png"
    plt.savefig(png_3d, dpi=150); plt.close()
    print(f"  [MPL] {png_3d}")

    # ── PNG 3: perfis u(x,0) — comparação t=0 vs t=T ─────────
    tol    = (2 * L / NX) * 0.6
    on_x   = np.abs(coords[:, 1]) < tol
    ix     = np.where(on_x)[0]
    ix     = ix[np.argsort(coords[ix, 0])]
    x_lin  = coords[ix, 0]
    fator  = 1.0 / (1.0 + 4.0 * alpha * a * T)
    u_ax   = fator * np.exp(-a * fator * x_lin**2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_lin, u0_vals[ix], "k-",  lw=2,   label="t = 0")
    ax.plot(x_lin, uT_vals[ix], "b-",  lw=2,   label=f"t = T = {T:.2f} s  (FEM)")
    ax.plot(x_lin, u_ax,        "r--", lw=1.5, label=f"t = T  (analítico ∞)")
    ax.axvspan(-R_ALVO, R_ALVO, alpha=0.08, color="red",
               label=f"Ω_alvo  |x| ≤ {R_ALVO} m")
    ax.axvline(-R_ALVO, color="red", lw=0.8, ls=":")
    ax.axvline( R_ALVO, color="red", lw=0.8, ls=":")
    ax.set_xlabel("x [m]"); ax.set_ylabel("u(x, 0, t)")
    ax.set_title(
        f"Perfil ao longo de y=0\n"
        f"α={alpha:.3e}  a={a:.1f}  T={T:.2f} s"
    )
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    png_perf = str(base) + f"_perfis_{solver}.png"
    plt.savefig(png_perf, dpi=150); plt.close()
    print(f"  [MPL] {png_perf}")

    print(f"  Exportação concluída → {pasta}/")


# ============================================================
# Execução direta
# ============================================================
if __name__ == "__main__":
    verify_fem()
    exportar_resultados(x0, solver="ref", label="ref_inicial")