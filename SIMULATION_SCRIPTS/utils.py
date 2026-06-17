# utils.py
# ============================================================
# Utilitários compartilhados: salvamento do histórico do
# Lagrangiano Aumentado em CSV.
#
# Uso em qualquer run_*.py:
#
#   from utils import salvar_historico_csv
#
#   x_opt, historico = alg_lagran_aug(...)
#
#   salvar_historico_csv(
#       historico,
#       label  = 'viga_cma',
#       pasta  = 'results',
#   )
# ============================================================

import csv
import numpy as np
from pathlib import Path


def salvar_historico_csv(
    historico: list,
    label:     str = "resultado",
    pasta:     str = "results",
) -> str:
    """
    Salva o histórico de iterações externas do LA em um arquivo CSV.

    Colunas geradas automaticamente a partir do primeiro dict:
      k, f, viab, rho, x_0, x_1, ..., g_0, g_1, ..., h_0, h_1, ...

    Parâmetros
    ----------
    historico : lista de dicts retornada por alg_lagran_aug.
                Cada dict deve ter as chaves:
                  'k', 'f', 'viab', 'rho', 'x', 'g', 'h'
    label     : prefixo do arquivo de saída (ex: 'viga_cma')
    pasta     : diretório de saída

    Retorna
    -------
    path_csv : caminho do arquivo gerado
    """
    Path(pasta).mkdir(exist_ok=True, parents=True)
    path_csv = str(Path(pasta) / f"{label}_historico_LA.csv")

    if not historico:
        print("  [utils] Histórico vazio — nenhum CSV gerado.")
        return path_csv

    # Infere dimensões a partir do primeiro registro
    primeiro = historico[0]
    dim_x = len(primeiro['x'])
    dim_g = len(primeiro['g'])
    dim_h = len(primeiro['h'])

    # Monta cabeçalho
    header  = ['k', 'f', 'viab', 'rho']
    header += [f'x_{i}' for i in range(dim_x)]
    header += [f'g_{i}' for i in range(dim_g)]
    header += [f'h_{i}' for i in range(dim_h)]

    with open(path_csv, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(header)

        for d in historico:
            row  = [d['k'], d['f'], d['viab'], d['rho']]
            row += list(d['x'])
            row += list(d['g']) if len(d['g']) > 0 else []
            row += list(d['h']) if len(d['h']) > 0 else []
            writer.writerow(row)

    print(f"  [utils] Histórico salvo em {path_csv}")
    return path_csv
