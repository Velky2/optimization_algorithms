# plot_historico.py
# ============================================================
# Script independente de plot do histórico do LA.
# Lê um CSV gerado por utils.salvar_historico_csv e gera
# um PNG com três blocos:
#   1. f(x) por iteração
#   2. restrições g e h por iteração
#   3. parâmetros físicos por iteração (via x_to_params)
#
# Uso:
#   python plot_historico.py <csv> <problema>
#
# Exemplos:
#   python plot_historico.py results/viga_cma_historico_LA.csv viga
#   python plot_historico.py results/sanity_turbo_historico_LA.csv sanity
#   python plot_historico.py results/difusao_cma_historico_LA.csv difusao
#   python plot_historico.py results/quadrupolo_turbo_historico_LA.csv quadrupolo
# ============================================================

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


# ============================================================
# Configuração por problema:
#   x_to_params : função de conversão (importada do módulo físico)
#   param_names : nomes dos parâmetros físicos para exibição
#   param_scales: fator de escala aplicado antes de plotar
#   g_names     : nomes das restrições de desigualdade
#   h_names     : nomes das restrições de igualdade (ou [])
# ============================================================

PROBLEMAS = {
    'sanity': {
        'modulo':       'sanity',
        'param_names':  ['A1 (mm²)', 'A2 (mm²)'],
        'param_scales': [1.0, 1.0],
        'g_names':      [],
        'h_names':      ['u_ponta - U_ALVO (mm)'],
    },
    'viga': {
        'modulo':       'viga',
        'param_names':  ['E (GPa)', 'nu', 'rho (kg/m³)'],
        'param_scales': [1e-9, 1.0, 1.0],
        'g_names':      ['FS_MIN - FS', 'm - M_MAX (kg)'],
        'h_names':      [],
    },
    'difusao': {
        'modulo':       'difusion',
        'param_names':  ['alpha (m²/s)', 'a (1/m²)', 'T (s)'],
        'param_scales': [1.0, 1.0, 1.0],
        'g_names':      ['alpha·T - DIFUSAO_MAX', 'DIFUSAO_MIN - alpha·T'],
        'h_names':      ['J0_fem - J0_ref'],
    },
    'quadrupolo': {
        'modulo':       'quadrupolo',
        'param_names':  ['d_wire (m)', 'r_wire (m)', 'J (A/m²)'],
        'param_scales': [1.0, 1.0, 1.0],
        'g_names':      ['sobreposição', 'domínio'],
        'h_names':      [],
    },
}


# ============================================================
# Leitura do CSV
# ============================================================

def ler_csv(path_csv: str) -> dict:
    """
    Lê o CSV e retorna um dict com arrays numpy para cada coluna.
    """
    with open(path_csv, newline='') as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV vazio: {path_csv}")

    # Infere colunas x, g, h pelo prefixo
    header = list(rows[0].keys())
    x_cols = sorted([c for c in header if c.startswith('x_')])
    g_cols = sorted([c for c in header if c.startswith('g_')])
    h_cols = sorted([c for c in header if c.startswith('h_')])

    data = {
        'k':    np.array([int(r['k'])   for r in rows]),
        'f':    np.array([float(r['f']) for r in rows]),
        'viab': np.array([float(r['viab']) for r in rows]),
        'rho':  np.array([float(r['rho'])  for r in rows]),
        'x':    np.array([[float(r[c]) for c in x_cols] for r in rows]),
        'g':    np.array([[float(r[c]) for c in g_cols] for r in rows]) if g_cols else np.empty((len(rows), 0)),
        'h':    np.array([[float(r[c]) for c in h_cols] for r in rows]) if h_cols else np.empty((len(rows), 0)),
    }
    return data


# ============================================================
# Estilo IEEE/Nature — legível em impressão A4, painel ou tela
# ============================================================

# Paleta monocromática com distinção por estilo de linha/marcador.
# Funciona tanto em colorido quanto em P&B impresso.
_LINE_STYLES  = ['-',  '--', '-.',  ':',  '-',  '--']
_MARKERS      = ['o',  's',  '^',   'D',  'v',  'P' ]
_GRAYS        = ['#111111', '#444444', '#777777', '#999999', '#bbbbbb', '#dddddd']

def _apply_ieee_style():
    """
    Aplica rcParams globais no estilo IEEE Transactions / Nature.
    Fonte base 9 pt (equivalente ao corpo de texto de duas colunas A4),
    linha 1.2 pt, grid sutil, sem borda direita/superior (spine).
    """
    plt.rcParams.update({
        # Fonte
        'font.family':       'serif',
        'font.serif':        ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size':         9,
        'axes.titlesize':    9,
        'axes.labelsize':    9,
        'xtick.labelsize':   8,
        'ytick.labelsize':   8,
        'legend.fontsize':   8,
        'figure.titlesize':  10,

        # Linhas
        'lines.linewidth':   1.4,
        'lines.markersize':  5,
        'axes.linewidth':    0.8,
        'grid.linewidth':    0.5,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,

        # Grid discreto
        'axes.grid':         True,
        'grid.color':        '#cccccc',
        'grid.linestyle':    ':',
        'grid.alpha':        1.0,

        # Spines — remove superior e direita (estilo Nature/ACS)
        'axes.spines.top':   False,
        'axes.spines.right': False,

        # Fundo branco limpo
        'axes.facecolor':    'white',
        'figure.facecolor':  'white',

        # Ticks internos
        'xtick.direction':   'in',
        'ytick.direction':   'in',

        # Layout apertado — crucial para encaixar em painel A4
        'figure.constrained_layout.use': True,
        'figure.constrained_layout.h_pad':  0.06,
        'figure.constrained_layout.w_pad':  0.06,
        'figure.constrained_layout.hspace': 0.05,
    })


def _style(idx: int) -> dict:
    """Retorna kwargs de estilo para o idx-ésimo série."""
    return dict(
        color=_GRAYS[idx % len(_GRAYS)],
        linestyle=_LINE_STYLES[idx % len(_LINE_STYLES)],
        marker=_MARKERS[idx % len(_MARKERS)],
        markevery=max(1, 1),   # todos os pontos; ajuste se houver muitas iterações
        markerfacecolor='white',
        markeredgewidth=1.2,
    )


def _add_refline(ax, y=0, label='', linestyle='--'):
    """Linha de referência horizontal discreta."""
    ax.axhline(y, color='#888888', linewidth=0.8, linestyle=linestyle,
                label=label, zorder=1)


def _finish_ax(ax, xlabel='Iteração LA', ylabel='', title='', legend=False):
    """Aplica formatação final em um eixo."""
    ax.set_xlabel(xlabel, labelpad=3)
    ax.set_ylabel(ylabel, labelpad=3)
    ax.set_title(title, pad=4, fontweight='bold')
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=6))
    if legend:
        ax.legend(frameon=False, handlelength=2)


# ============================================================
# Plot
# ============================================================

def plotar(data: dict, cfg: dict, x_to_params, label: str, pasta: str):

    _apply_ieee_style()

    iters  = data['k']
    f_vals = data['f']
    x_vals = data['x']
    g_vals = data['g']
    h_vals = data['h']
    n_g    = g_vals.shape[1]
    n_h    = h_vals.shape[1]

    # Converte x normalizado → parâmetros físicos escalados
    params_raw = np.array([x_to_params(x_vals[i]) for i in range(len(iters))])
    scales     = np.array(cfg['param_scales'])
    params_viz = params_raw * scales

    n_constraint_rows = max(n_g + n_h, 1)
    n_param_rows      = params_viz.shape[1]
    n_rows = 1 + n_constraint_rows + n_param_rows

    # Largura: ~8.5 cm (coluna simples A4) ou ~17.5 cm (coluna dupla/página inteira).
    # Altura: ~4.5 cm por subplot — suficiente para leitura sem desperdiçar espaço.
    FIG_WIDTH  = 8.6 / 2.54          # 8.6 cm → inches  (coluna IEEE simples)
    ROW_HEIGHT = 3.8 / 2.54          # 3.8 cm por subplot
    fig_height = n_rows * ROW_HEIGHT

    fig = plt.figure(figsize=(FIG_WIDTH, fig_height))
    gs  = gridspec.GridSpec(n_rows, 1, figure=fig)
    row = 0

    # ---- f(x) -----------------------------------------------
    ax = fig.add_subplot(gs[row]); row += 1
    ax.plot(iters, f_vals, **_style(0), label='f(x)', zorder=2)
    _add_refline(ax, y=0)
    _finish_ax(ax, ylabel='f(x)', title='Função objetivo')

    # ---- restrições g ---------------------------------------
    for i in range(n_g):
        ax = fig.add_subplot(gs[row]); row += 1
        name = cfg['g_names'][i] if i < len(cfg['g_names']) else f'g[{i}]'
        ax.plot(iters, g_vals[:, i], **_style(i), label=name, zorder=2)
        _add_refline(ax, y=0, label='g ≤ 0')
        _finish_ax(ax, ylabel=name, title=f'Restrição g: {name}', legend=True)

    # ---- restrições h ---------------------------------------
    for i in range(n_h):
        ax = fig.add_subplot(gs[row]); row += 1
        name = cfg['h_names'][i] if i < len(cfg['h_names']) else f'h[{i}]'
        ax.plot(iters, h_vals[:, i], **_style(n_g + i), label=name, zorder=2)
        _add_refline(ax, y=0, label='h = 0')
        _finish_ax(ax, ylabel=name, title=f'Restrição h: {name}', legend=True)

    # Fallback: sem restrições → plota viabilidade
    if n_g == 0 and n_h == 0:
        ax = fig.add_subplot(gs[row]); row += 1
        ax.semilogy(iters, data['viab'], **_style(0), zorder=2)
        _finish_ax(ax, ylabel='Viabilidade', title='Viabilidade (log)')

    # ---- parâmetros físicos ---------------------------------
    for i in range(n_param_rows):
        ax = fig.add_subplot(gs[row]); row += 1
        name = cfg['param_names'][i] if i < len(cfg['param_names']) else f'param[{i}]'
        ax.plot(iters, params_viz[:, i], **_style(i), zorder=2)
        _finish_ax(ax, ylabel=name, title=f'Parâmetro: {name}')

    # ---- salva — PDF vetorial (preferido para relatório) + PNG de alta resolução
    Path(pasta).mkdir(exist_ok=True, parents=True)
    stem     = Path(pasta) / f"{label}_historico_LA"
    path_pdf = str(stem) + '.pdf'
    path_png = str(stem) + '.png'

    fig.savefig(path_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(path_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot salvo em:\n  {path_pdf}\n  {path_png}")


# ============================================================
# Entrypoint
# ============================================================

def main():
    if len(sys.argv) < 3:
        print("Uso: python plot_historico.py <csv> <problema>")
        print("Problemas disponíveis:", list(PROBLEMAS.keys()))
        sys.exit(1)

    path_csv = sys.argv[1]
    problema = sys.argv[2].lower()

    if problema not in PROBLEMAS:
        print(f"Problema '{problema}' desconhecido. Opções: {list(PROBLEMAS.keys())}")
        sys.exit(1)

    cfg = PROBLEMAS[problema]

    # Importa x_to_params do módulo físico correspondente
    import importlib
    modulo = importlib.import_module(cfg['modulo'])
    x_to_params = modulo.x_to_params

    data  = ler_csv(path_csv)
    pasta = str(Path(path_csv).parent)
    label = Path(path_csv).stem.replace('_historico_LA', '')

    plotar(data, cfg, x_to_params, label, pasta)


if __name__ == "__main__":
    main()