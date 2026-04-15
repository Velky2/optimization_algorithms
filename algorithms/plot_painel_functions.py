import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker


# ── Estilo global (aplica antes de chamar plot_painel) ───────────────────────
def aplicar_estilo_artigo():
    """
    Configura rcParams para estilo de publicação científica.
    Chame uma vez no início do notebook.
    """
    rcParams.update({
        # Fonte — serifada para corpo, monospace para números
        "font.family":        "serif",
        "font.serif":         ["Georgia", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset":   "dejavuserif",

        # Tamanhos
        "font.size":          9,
        "axes.titlesize":     9,
        "axes.labelsize":     8,
        "xtick.labelsize":    7.5,
        "ytick.labelsize":    7.5,
        "legend.fontsize":    7.5,

        # Eixos
        "axes.linewidth":     0.6,
        "axes.edgecolor":     "#2a2a2a",
        "axes.facecolor":     "#fafaf8",
        "axes.grid":          True,
        "axes.axisbelow":     True,

        # Grid
        "grid.color":         "#d8d5ce",
        "grid.linewidth":     0.4,
        "grid.linestyle":     "--",
        "grid.alpha":         0.7,

        # Ticks
        "xtick.direction":    "in",
        "ytick.direction":    "in",
        "xtick.major.width":  0.5,
        "ytick.major.width":  0.5,
        "xtick.major.size":   3,
        "ytick.major.size":   3,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.size":   1.5,
        "ytick.minor.size":   1.5,
        "xtick.minor.width":  0.4,
        "ytick.minor.width":  0.4,

        # Linhas e marcadores
        "lines.linewidth":    1.2,
        "lines.markersize":   3.5,

        # Legenda
        "legend.frameon":         True,
        "legend.framealpha":      0.92,
        "legend.edgecolor":       "#c0bcb4",
        "legend.fancybox":        False,

        # Figura
        "figure.facecolor":   "white",
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.facecolor":  "white",
    })


# ── Paleta ───────────────────────────────────────────────────────────────────
_AZUL    = "#1a3a5c"   # principal — azul naval profundo
_AZUL_L  = "#c8d8e8"   # faixa de incerteza
_CINZA   = "#5a5a5a"   # melhor execução / secundário
_VERMELHO= "#8b2020"   # destaque pontual (boxplot flier)


# ── Estimativa de convergência ───────────────────────────────────────────────
def _estimar_convergencia(hist, x, popsize, epsilon=1e-6):
    conv_evals = []
    for run in hist:
        conv_gen = None
        for i in range(1, len(run)):
            ant = run[i - 1]
            if ant == 0:
                continue
            if abs(ant - run[i]) / abs(ant) < epsilon:
                ref = run[i]
                if ref == 0:
                    continue
                if all(abs(ref - v) / abs(ref) < epsilon for v in run[i:]):
                    conv_gen = i
                    break
        conv_evals.append(conv_gen * popsize if conv_gen is not None else x[-1])
    return np.array(conv_evals)


# ── Formatador de eixo log legível ───────────────────────────────────────────
def _fmt_log(val, pos):
    exp = int(np.log10(val)) if val > 0 else 0
    return rf"$10^{{{exp}}}$"


# ── Função principal ─────────────────────────────────────────────────────────
def plot_painel(hist, func, dim, salvar_como=None):
    """
    Parâmetros
    ----------
    hist         : list[list[float]] — histórico de melhores valores por execução
    func         : str  — nome da função (usado no título e na caixa de stats)
    dim          : int  — dimensionalidade do problema
    salvar_como  : str | None — caminho para salvar (ex: 'levy_10d.pdf').
                   Se None, exibe com plt.show().
    """
    aplicar_estilo_artigo()

    hist = np.asarray(hist, dtype=float)
    popsize = 4 + int(3 * np.log(dim))
    x = np.arange(1, hist.shape[1] + 1) * popsize

    # estatísticas da curva
    mean_   = np.mean(hist,   axis=0)
    std_    = np.std(hist,    axis=0)
    median_ = np.median(hist, axis=0)
    q1_     = np.percentile(hist, 25, axis=0)
    q3_     = np.percentile(hist, 75, axis=0)

    best_final = hist[:, -1]
    best_idx   = np.argmin(best_final)
    best_run   = hist[best_idx]

    conv_evals  = _estimar_convergencia(hist, x, popsize)
    n_conv      = int(np.sum(conv_evals < x[-1]))
    n_runs      = len(best_final)

    # ── Layout ───────────────────────────────────────────────────────────────
    # Largura de coluna dupla típica de journal: ~17 cm → 6.7 in
    fig = plt.figure(figsize=(7.2, 8.4))

    # Título principal — discreto, estilo caption de figura
    fig.text(
        0.5, 0.992,
        rf"CMA-ES — {func} function, $d={dim}$",
        ha="center", va="top",
        fontsize=10, fontstyle="italic",
        color="#1a1a1a",
    )

    # Linha horizontal decorativa sob o título
    fig.add_artist(
        plt.Line2D([0.08, 0.92], [0.972, 0.972],
                   transform=fig.transFigure,
                   color="#2a2a2a", linewidth=0.7)
    )

    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        top=0.950, bottom=0.065,
        left=0.09, right=0.97,
        hspace=0.52, wspace=0.38,
        height_ratios=[1, 1, 0.45],
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])

    # ── Utilitário: formatar eixos log ────────────────────────────────────────
    def _preparar_log_ax(ax, axis="y"):
        fmt = ticker.FuncFormatter(_fmt_log)
        if axis in ("y", "both"):
            ax.yaxis.set_major_formatter(fmt)
        if axis in ("x", "both"):
            ax.xaxis.set_major_formatter(fmt)

    # ── ax1: mediana + IQR + melhor execução ─────────────────────────────────
    ax1.fill_between(x, q1_, q3_, color=_AZUL_L, alpha=0.7, label="IQR (25–75%)", zorder=2)
    ax1.plot(x, median_, color=_AZUL, linewidth=1.4, label="Median", zorder=3)
    ax1.plot(x, best_run, color=_CINZA, linewidth=0.9,
             linestyle=(0, (4, 2)), label="Best run", zorder=4)

    ax1.set_yscale("log")
    _preparar_log_ax(ax1, "y")
    ax1.set_title("(a) Median convergence + IQR", loc="left", pad=4)
    ax1.set_xlabel("Function evaluations")
    ax1.set_ylabel("Best $f(x)$")
    leg1 = ax1.legend(loc="upper right", handlelength=1.6)
    leg1.get_frame().set_linewidth(0.5)

    # ── ax2: média ± desvio padrão ────────────────────────────────────────────
    ax2.fill_between(x, mean_ - std_, mean_ + std_,
                     color=_AZUL_L, alpha=0.7, label=r"Mean $\pm$ SD", zorder=2)
    ax2.plot(x, mean_, color=_AZUL, linewidth=1.4, label="Mean", zorder=3)

    ax2.set_yscale("log")
    _preparar_log_ax(ax2, "y")
    ax2.set_title(r"(b) Mean convergence $\pm$ SD", loc="left", pad=4)
    ax2.set_xlabel("Function evaluations")
    ax2.set_ylabel("Best $f(x)$")
    leg2 = ax2.legend(loc="upper right", handlelength=1.6)
    leg2.get_frame().set_linewidth(0.5)

    # ── ax3: boxplot ──────────────────────────────────────────────────────────
    bp = ax3.boxplot(
        [best_final],
        tick_labels=[func],
        patch_artist=True,
        widths=0.35,
        medianprops=dict(color=_AZUL, linewidth=1.4),
        whiskerprops=dict(color="#2a2a2a", linewidth=0.7, linestyle="--"),
        capprops=dict(color="#2a2a2a", linewidth=0.7),
        boxprops=dict(linewidth=0.7),
        flierprops=dict(marker="o", markersize=2.5,
                        markerfacecolor=_VERMELHO,
                        markeredgecolor=_VERMELHO, alpha=0.5),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(_AZUL_L)
        patch.set_alpha(0.8)

    ax3.set_yscale("log")
    _preparar_log_ax(ax3, "y")
    ax3.set_title("(c) Final value distribution", loc="left", pad=4)
    ax3.set_ylabel("Best $f(x)$")
    ax3.set_xticklabels([func], style="italic")

    # ── ax4: ECDF ─────────────────────────────────────────────────────────────
    xs_ecdf = np.sort(best_final)
    ys_ecdf = np.arange(1, n_runs + 1) / n_runs

    ax4.step(xs_ecdf, ys_ecdf, color=_AZUL, linewidth=1.3, where="post")
    ax4.fill_between(xs_ecdf, 0, ys_ecdf,
                     step="post", color=_AZUL, alpha=0.08)

    # linha de referência em 0.5
    ax4.axhline(0.5, color=_CINZA, linewidth=0.6,
                linestyle=":", alpha=0.8, label="$p = 0.5$")

    ax4.set_xscale("log")
    _preparar_log_ax(ax4, "x")
    ax4.set_ylim(-0.02, 1.06)
    ax4.set_title("(d) ECDF of final best values", loc="left", pad=4)
    ax4.set_xlabel("Best $f(x)$")
    ax4.set_ylabel("Cumulative proportion")
    leg4 = ax4.legend(loc="upper left", handlelength=1.2)
    leg4.get_frame().set_linewidth(0.5)

    # ── ax5: caixa de estatísticas ────────────────────────────────────────────
    ax5.set_axis_off()

    # fundo estilizado
    bg = FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0,
        boxstyle="square,pad=0",
        facecolor="#f3f1ec",
        edgecolor="#2a2a2a",
        linewidth=0.7,
        transform=ax5.transAxes,
        clip_on=False,
        zorder=0,
    )
    ax5.add_patch(bg)

    # Usar coordenadas de figura para posicionar o texto com precisão em pontos,
    # independentemente da altura do axes comprimido
    ax5_bbox   = ax5.get_position()           # fração da figura
    fig_w_in, fig_h_in = fig.get_size_inches()

    # altura disponível na caixa em pontos
    box_h_pt = ax5_bbox.height * fig_h_in * 72

    # linha decorativa superior (deixa 18% do topo para o cabeçalho)
    sep_y = ax5_bbox.y0 + ax5_bbox.height * 0.80
    fig.add_artist(
        plt.Line2D(
            [ax5_bbox.x0 + 0.005, ax5_bbox.x1 - 0.005],
            [sep_y, sep_y],
            transform=fig.transFigure,
            color="#2a2a2a", linewidth=0.5, zorder=5,
        )
    )

    # cabeçalho — centralizado no topo da caixa
    fig.text(
        ax5_bbox.x0 + ax5_bbox.width / 2,
        ax5_bbox.y0 + ax5_bbox.height * 0.905,
        "Statistical summary",
        ha="center", va="center",
        fontsize=8, fontstyle="italic",
        color="#1a1a1a",
        transform=fig.transFigure,
    )

    # conteúdo — 4 colunas × 2 linhas
    if n_conv > 0:
        conv_med = np.median(conv_evals[conv_evals < x[-1]])
        conv_txt = rf"{conv_med:.0f} evals"
    else:
        conv_txt = "none within budget"

    colunas = [
        [("Runs",      f"{n_runs}"),                        ("Budget",    f"{int(x[-1])} evals/run")],
        [("Best",      f"{np.min(best_final):.3e}"),        ("Worst",     f"{np.max(best_final):.3e}")],
        [("Median",    f"{np.median(best_final):.3e}"),     ("Mean",      f"{np.mean(best_final):.3e}")],
        [("Std. dev.", f"{np.std(best_final):.3e}"),        ("Conv.",     f"{n_conv}/{n_runs} — {conv_txt}")],
    ]

    n_cols   = len(colunas)
    col_w    = ax5_bbox.width / n_cols
    pad_left = 0.008   # margem interna esquerda por coluna (fração da figura)

    # duas linhas de conteúdo: y1 (linha 0) e y2 (linha 1)
    # distribuídas na área abaixo do separador com margem superior e inferior
    content_top = ax5_bbox.y0 + ax5_bbox.height * 0.74
    content_bot = ax5_bbox.y0 + ax5_bbox.height * 0.08
    row_lbl_offset = ax5_bbox.height * 0.14   # espaço entre label e valor dentro da linha
    row_gap = (content_top - content_bot) / 2  # gap entre linha 0 e linha 1

    for ci, col in enumerate(colunas):
        x_col = ax5_bbox.x0 + ci * col_w + pad_left
        for ri, (lbl, val) in enumerate(col):
            y_lbl = content_top - ri * row_gap
            y_val = y_lbl - row_lbl_offset

            fig.text(
                x_col, y_lbl, f"{lbl}:",
                ha="left", va="center",
                fontsize=7.5, fontweight="bold",
                color="#1a1a1a",
                transform=fig.transFigure,
            )
            fig.text(
                x_col, y_val, val,
                ha="left", va="center",
                fontsize=7.5,
                color="#2e2e2e",
                transform=fig.transFigure,
            )

    # ── Salvar ou exibir ──────────────────────────────────────────────────────
    if salvar_como:
        fig.savefig(salvar_como, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Figura salva em: {salvar_como}")
    else:
        plt.show()

# ── Painel dimensional ───────────────────────────────────────────────────────
def plot_painel_dimensional(hist_dict, func="None", salvar_como=None):
    """
    Parâmetros
    ----------
    hist_dict    : dict[int, list[list[float]]] — {dim: histórico de runs}
    func         : str  — nome da função (usado no título)
    salvar_como  : str | None — caminho para salvar. Se None, exibe com plt.show().

    Subplots
    --------
    (a) Curvas de convergência mediana por dimensão
    (b) IQR normalizado (dispersão relativa) ao longo das avaliações
    (c) Boxplot dos valores finais por dimensão
    (d) Avaliações medianas até convergência vs. dimensão (scaling)
    """
    aplicar_estilo_artigo()

    dims = sorted(hist_dict.keys())

    # paleta sequencial: uma cor por dimensão
    cmap   = plt.get_cmap("Blues")
    colors = [cmap(0.35 + 0.55 * i / max(len(dims) - 1, 1)) for i in range(len(dims))]

    # ── Layout — espelha o primeiro painel ───────────────────────────────────
    fig = plt.figure(figsize=(7.2, 8.4))

    fig.text(
        0.5, 0.992,
        rf"CMA-ES — scaling with dimension ({func})",
        ha="center", va="top",
        fontsize=10, fontstyle="italic",
        color="#1a1a1a",
    )

    fig.add_artist(
        plt.Line2D([0.08, 0.92], [0.972, 0.972],
                   transform=fig.transFigure,
                   color="#2a2a2a", linewidth=0.7)
    )

    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        top=0.950, bottom=0.065,
        left=0.09, right=0.97,
        hspace=0.52, wspace=0.38,
        height_ratios=[1, 1, 0.45],
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])

    def _preparar_log_ax(ax, axis="y"):
        fmt = ticker.FuncFormatter(_fmt_log)
        if axis in ("y", "both"):
            ax.yaxis.set_major_formatter(fmt)
        if axis in ("x", "both"):
            ax.xaxis.set_major_formatter(fmt)

    # ── (a) Convergência mediana por dimensão ────────────────────────────────
    for dim, color in zip(dims, colors):
        hist    = np.asarray(hist_dict[dim])
        popsize = 4 + int(3 * np.log(dim))
        x       = np.arange(1, hist.shape[1] + 1) * popsize
        median_ = np.median(hist, axis=0)
        ax1.plot(x, median_, color=color, linewidth=1.3, label=f"$d={dim}$")

    ax1.set_yscale("log")
    _preparar_log_ax(ax1, "y")
    ax1.set_title("(a) Median convergence by dimension", loc="left", pad=4)
    ax1.set_xlabel("Function evaluations")
    ax1.set_ylabel("Best $f(x)$")
    leg1 = ax1.legend(loc="upper right", handlelength=1.4, ncol=2)
    leg1.get_frame().set_linewidth(0.5)

    # ── (b) IQR relativo (Q3-Q1)/median — dispersão normalizada ─────────────
    # Mostra como a variabilidade entre runs cresce com a dimensão;
    # normalizar pela mediana torna as dimensões comparáveis na mesma escala.
    for dim, color in zip(dims, colors):
        hist    = np.asarray(hist_dict[dim])
        popsize = 4 + int(3 * np.log(dim))
        x       = np.arange(1, hist.shape[1] + 1) * popsize
        q1_     = np.percentile(hist, 25, axis=0)
        q3_     = np.percentile(hist, 75, axis=0)
        median_ = np.median(hist, axis=0)
        # evita divisão por zero
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_iqr = np.where(median_ > 0, (q3_ - q1_) / median_, np.nan)
        ax2.plot(x, rel_iqr, color=color, linewidth=1.3, label=f"$d={dim}$")

    ax2.set_yscale("log")
    _preparar_log_ax(ax2, "y")
    ax2.set_title("(b) Relative IQR  $(Q_3-Q_1)/\\tilde{f}$", loc="left", pad=4)
    ax2.set_xlabel("Function evaluations")
    ax2.set_ylabel("Relative spread")
    leg2 = ax2.legend(loc="upper right", handlelength=1.4, ncol=2)
    leg2.get_frame().set_linewidth(0.5)

    # ── (c) Boxplot dos valores finais por dimensão ──────────────────────────
    data = [np.asarray(hist_dict[d])[:, -1] for d in dims]

    bp = ax3.boxplot(
        data,
        tick_labels=[str(d) for d in dims],
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color=_AZUL, linewidth=1.4),
        whiskerprops=dict(color="#2a2a2a", linewidth=0.7, linestyle="--"),
        capprops=dict(color="#2a2a2a", linewidth=0.7),
        boxprops=dict(linewidth=0.7),
        flierprops=dict(marker="o", markersize=2.5,
                        markerfacecolor=_VERMELHO,
                        markeredgecolor=_VERMELHO, alpha=0.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax3.set_yscale("log")
    _preparar_log_ax(ax3, "y")
    ax3.set_title("(c) Final value distribution", loc="left", pad=4)
    ax3.set_xlabel("Dimension $d$")
    ax3.set_ylabel("Best $f(x)$")

    # ── (d) Scaling: avaliações medianas até convergência vs. dimensão ───────
    # Usa _estimar_convergencia (mesma lógica do painel 1) — robusto e consistente.
    conv_med_list  = []
    conv_q1_list   = []
    conv_q3_list   = []

    for dim in dims:
        hist    = np.asarray(hist_dict[dim])
        popsize = 4 + int(3 * np.log(dim))
        x       = np.arange(1, hist.shape[1] + 1) * popsize
        ce      = _estimar_convergencia(hist, x, popsize)
        conv_med_list.append(np.median(ce))
        conv_q1_list.append(np.percentile(ce, 25))
        conv_q3_list.append(np.percentile(ce, 75))

    dims_arr = np.array(dims, dtype=float)
    cm = np.array(conv_med_list)
    cq1 = np.array(conv_q1_list)
    cq3 = np.array(conv_q3_list)

    ax4.fill_between(dims_arr, cq1, cq3, color=_AZUL_L, alpha=0.7, label="IQR (25–75%)")
    ax4.plot(dims_arr, cm, color=_AZUL, linewidth=1.3,
             marker="o", markersize=3.5, label="Median")

    ax4.set_xscale("log")
    ax4.set_yscale("log")
    _preparar_log_ax(ax4, "both")
    ax4.set_title("(d) Evaluations to convergence", loc="left", pad=4)
    ax4.set_xlabel("Dimension $d$")
    ax4.set_ylabel("Evaluations")
    leg4 = ax4.legend(loc="upper left", handlelength=1.4)
    leg4.get_frame().set_linewidth(0.5)

    # ── ax5: caixa de estatísticas resumidas ─────────────────────────────────
    ax5.set_axis_off()

    bg = FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0,
        boxstyle="square,pad=0",
        facecolor="#f3f1ec",
        edgecolor="#2a2a2a",
        linewidth=0.7,
        transform=ax5.transAxes,
        clip_on=False,
        zorder=0,
    )
    ax5.add_patch(bg)

    ax5_bbox = ax5.get_position()

    sep_y = ax5_bbox.y0 + ax5_bbox.height * 0.80
    fig.add_artist(
        plt.Line2D(
            [ax5_bbox.x0 + 0.005, ax5_bbox.x1 - 0.005],
            [sep_y, sep_y],
            transform=fig.transFigure,
            color="#2a2a2a", linewidth=0.5, zorder=5,
        )
    )

    fig.text(
        ax5_bbox.x0 + ax5_bbox.width / 2,
        ax5_bbox.y0 + ax5_bbox.height * 0.905,
        "Dimensional summary",
        ha="center", va="center",
        fontsize=8, fontstyle="italic",
        color="#1a1a1a",
        transform=fig.transFigure,
    )

    # Métricas resumidas: uma coluna por dimensão
    n_cols  = len(dims)
    col_w   = ax5_bbox.width / n_cols
    pad_left = 0.008

    content_top      = ax5_bbox.y0 + ax5_bbox.height * 0.74
    content_bot      = ax5_bbox.y0 + ax5_bbox.height * 0.08
    row_lbl_offset   = ax5_bbox.height * 0.14
    row_gap          = (content_top - content_bot) / 2

    for ci, dim in enumerate(dims):
        best_final = np.asarray(hist_dict[dim])[:, -1]
        col = [
            (f"d={dim}",  f"med={np.median(best_final):.2e}"),
            ("best",      f"{np.min(best_final):.2e}"),
        ]
        x_col = ax5_bbox.x0 + ci * col_w + pad_left
        for ri, (lbl, val) in enumerate(col):
            y_lbl = content_top - ri * row_gap
            y_val = y_lbl - row_lbl_offset
            fig.text(x_col, y_lbl, f"{lbl}:",
                     ha="left", va="center", fontsize=7.5, fontweight="bold",
                     color="#1a1a1a", transform=fig.transFigure)
            fig.text(x_col, y_val, val,
                     ha="left", va="center", fontsize=7.5,
                     color="#2e2e2e", transform=fig.transFigure)

    # ── saída ─────────────────────────────────────────────────────────────────
    if salvar_como:
        fig.savefig(salvar_como, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Figura salva em: {salvar_como}")
    else:
        plt.show()