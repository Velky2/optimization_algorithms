''' Este Script tem como objetivo implementar o framework do método de otmização do Algoritmo de Lagrangiano Aumentado em linguagem python.

Referências:

MARTÍNEZ, José Mario. Otimizaçao prática usando o lagrangiano aumentado. Tech. rep. https://www.ime.unicamp.br/martinez/opusculo.html, 2006.

Autor: Edélio Gabriel Magalhães de Jesus

'''

# Módulos necessários
import numpy as np

def param_init(f, h, g, x0):
    '''
    Inicializa os parâmetros do Lagrangiano Aumentado.

    Inputs:
        f:  função objetivo
        h:  restrições de igualdade (ou None)
        g:  restrições de desigualdade (ou None)
        x0: ponto inicial

    Output:
        dicionário com os parâmetros iniciais
    '''
    h0 = h(x0) if h is not None else np.array([])
    g0 = np.maximum(0, g(x0)) if g is not None else np.array([])

    f0    = f(x0)
    denom = np.linalg.norm(h0)**2 + np.linalg.norm(g0)**2
    rho_candidate = 2 * abs(f0) / denom if denom > 0 else 10.0

    return {
        'tau':        0.5,
        'gamma':      10,
        'lambda_min': -1e20,
        'lambda_max':  1e20,
        'mu_max':      1e20,
        'epsilon':     1e-4,
        'lambda_bar':  np.zeros(len(h0)),
        'mu_bar':      np.zeros(len(g0)),
        'rho':         np.maximum(1e-6, np.minimum(10, rho_candidate)),
        'm':           len(h0),
        'p':           len(g0),
    }


def lagran_aug(f, h, g, x, params):
    '''
    Avalia o Lagrangiano Aumentado em x.

    Inputs:
        f, h, g: funções objetivo e restrições
        x:       ponto de avaliação
        params:  dicionário de parâmetros

    Output:
        valor escalar do Lagrangiano Aumentado em x
    '''
    rho        = params['rho']
    lambda_bar = params['lambda_bar']
    mu_bar     = params['mu_bar']

    h0 = h(x) if h is not None else np.array([])
    g0 = g(x) if g is not None else np.array([])

    penalty_eq   = np.sum((h0 + lambda_bar / rho) ** 2)
    penalty_ineq = np.sum(np.maximum(0, g0 + mu_bar / rho) ** 2)

    return f(x) + (rho / 2) * (penalty_eq + penalty_ineq)


def step1_solve_subproblem(f, h, g, x0, params, solver):
    '''
    Passo 1 — resolve o subproblema minimizando o Lagrangiano Aumentado.

    Inputs:
        f, h, g: funções objetivo e restrições
        x0:      ponto inicial da iteração atual
        params:  dicionário de parâmetros
        solver:  função com assinatura solver(func, x0) -> x_opt

    Output:
        x_new: minimizador aproximado do subproblema
    '''
    def subproblem(x):
        return lagran_aug(f, h, g, x, params)

    return solver(subproblem, x0)


def step2_lagran_multi_aprox(x_k, h, g, params):
    '''
    Passo 2 — atualiza os multiplicadores de Lagrange.

    Inputs:
        x_k:    ponto atual
        h, g:   restrições
        params: dicionário de parâmetros

    Output:
        lambda_new: multiplicadores de igualdade atualizados (livres em R)
        mu_new:     multiplicadores de desigualdade atualizados (projetados em >= 0)
    '''
    lambda_bar = params['lambda_bar']
    mu_bar     = params['mu_bar']
    rho        = params['rho']

    h0 = h(x_k) if h is not None else np.array([])
    g0 = g(x_k) if g is not None else np.array([])

    lambda_new = lambda_bar + rho * h0
    mu_new     = np.maximum(0, mu_bar + rho * g0)

    return lambda_new, mu_new


def step3_new_rho(x_k, x_ant, h, g, params, k, V_ant=None):
    '''
    Passo 3 — decide se aumenta o parâmetro de penalidade rho.

    Inputs:
        x_k:    ponto atual
        x_ant:  ponto da iteração anterior
        h, g:   restrições
        params: dicionário de parâmetros
        k:      índice da iteração atual
        V_ant:  vetor V da iteração anterior (None se k==1)

    Output:
        rho_new: novo valor do parâmetro de penalidade
    '''
    mu_bar = params['mu_bar']
    tau    = params['tau']
    rho    = params['rho']
    gamma  = params['gamma']

    h_k = h(x_k) if h is not None else np.array([])
    g_k = g(x_k) if g is not None else np.array([])
    V_k = np.minimum(-g_k, mu_bar / rho)

    if k == 1:
        return rho, V_k   # sem referência anterior, mantém rho e devolve V_k

    term_k   = max(np.linalg.norm(h_k),   np.linalg.norm(V_k))
    term_ant = max(np.linalg.norm(h(x_ant) if h is not None else np.array([])),
                   np.linalg.norm(V_ant))

    if term_k <= tau * term_ant:
        return rho, V_k           # progresso suficiente: mantém rho
    else:
        return gamma * rho, V_k   # sem progresso: aumenta rho


def step4_safeguard(lambda_new, mu_new, params):
    '''
    Passo 4 — aplica safeguard nos multiplicadores, truncando nos limites.

    Inputs:
        lambda_new: multiplicadores de igualdade do passo 2
        mu_new:     multiplicadores de desigualdade do passo 2
        params:     dicionário de parâmetros

    Output:
        lambda_bar_new: lambda truncado em [lambda_min, lambda_max]
        mu_bar_new:     mu truncado em [0, mu_max]
    '''
    lambda_min = params['lambda_min']
    lambda_max = params['lambda_max']
    mu_max     = params['mu_max']

    lambda_bar_new = np.maximum(lambda_min, np.minimum(lambda_new, lambda_max))
    mu_bar_new     = np.maximum(0,          np.minimum(mu_new,     mu_max))

    return lambda_bar_new, mu_bar_new


def step5_update(params, rho_new, lambda_bar_new, mu_bar_new):
    '''
    Passo 5 — atualiza o dicionário de parâmetros para a próxima iteração.

    Inputs:
        params:         dicionário de parâmetros atual
        rho_new:        novo rho vindo do passo 3
        lambda_bar_new: novos multiplicadores de igualdade do passo 4
        mu_bar_new:     novos multiplicadores de desigualdade do passo 4

    Output:
        dicionário de parâmetros atualizado
    '''
    params['rho']        = rho_new
    params['lambda_bar'] = lambda_bar_new
    params['mu_bar']     = mu_bar_new

    return params


def alg_lagran_aug(f, h, g, x0, solver, max_iter=100, verbose=True):
    '''
    Loop principal do Algoritmo de Lagrangiano Aumentado.

    Inputs:
        f, h, g:  funções objetivo e restrições
        x0:       ponto inicial
        solver:   função com assinatura solver(func, x0) -> x_opt
        max_iter: número máximo de iterações externas
        verbose:  imprime progresso a cada iteração (default: True)

    Output:
        x: solução aproximada
    '''
    params = param_init(f, h, g, x0)
    x      = x0
    x_ant  = None
    V_ant  = None
    historico = []
    for k in range(1, max_iter + 1):

        # Passo 1: resolve subproblema
        x_ant = x
        x     = step1_solve_subproblem(f, h, g, x, params, solver)

        # Passo 2: atualiza multiplicadores
        lambda_new, mu_new = step2_lagran_multi_aprox(x, h, g, params)

        # Passo 3: decide novo rho
        rho_new, V_ant = step3_new_rho(x, x_ant, h, g, params, k, V_ant)

        # Passo 4: safeguard nos multiplicadores
        lambda_bar_new, mu_bar_new = step4_safeguard(lambda_new, mu_new, params)

        # Passo 5: atualiza parâmetros para próxima iteração
        params = step5_update(params, rho_new, lambda_bar_new, mu_bar_new)

        # Critério de convergência
        h_k = h(x) if h is not None else np.array([])
        g_k = g(x) if g is not None else np.array([])
        V_k = np.minimum(-g_k, params['mu_bar'] / params['rho']) if len(g_k) > 0 else np.array([0.0])

        viabilidade = max(np.linalg.norm(h_k), np.linalg.norm(V_k))

        historico.append({
            'k':    k,
            'x':    x.copy(),
            'f':    f(x),
            'g':    g_k.copy() if len(g_k) > 0 else np.array([]),
            'h':    h_k.copy() if len(h_k) > 0 else np.array([]),
            'viab': viabilidade,
            'rho':  params['rho'],
        })
        
        if verbose:
            f_k = f(x)
            g_str = ', '.join(f'{gi:.3f}' for gi in g_k) if len(g_k) > 0 else '—'
            print(f'  iter {k:3d} | f={f_k:.4e} | g=[{g_str}] | '
                  f'rho={params["rho"]:.2e} | viab={viabilidade:.2e}')

        if k > 1 and viabilidade <= params['epsilon']:
            print(f'Convergiu na iteração {k}')
            break

    return x, historico