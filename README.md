# optimization_algorithms

Implementação e estudo de **métodos de otimização para problemas com restrições**, com foco no acoplamento do **Lagrangiano Aumentado (LA)** a solvers internos sem derivadas — **CMA-ES** e **TuRBO** — aplicados a problemas de engenharia governados por **simulações de elementos finitos (FEM)**.

O repositório reúne três eixos complementares:

1. **Notebooks de estudo** de cada algoritmo de otimização (CMA-ES, TuRBO, FuRBO, PSO, COBYLA, Trust-Region, Lagrangiano Aumentado), usados para entender e validar cada método isoladamente.
2. **Benchmarking padronizado** (`COCO/`), em que as variantes com Lagrangiano Aumentado (AL-TuRBO, AL-CMA-ES) são comparadas a baselines puros (TuRBO, CMA-ES) sobre o conjunto **bbob-constrained** da plataforma COCO, avaliadas por perfis de desempenho.
3. **Pipeline aplicado** (`SIMULATION_SCRIPTS/`), em que o Lagrangiano Aumentado conduz o laço externo e um solver interno (CMA-ES ou TuRBO) resolve cada subproblema sobre quatro problemas-teste resolvidos por FEM (FEniCSx).

---

## Visão geral do método

O laço externo segue a formulação **PHR (Powell–Hestenes–Rockafellar)** com salvaguardas no estilo Birgin–Martínez. Dado o problema

$$
\min_{x \in [0,1]^d} f(x) \quad \text{sujeito a} \quad h(x) = 0, \quad g(x) \le 0,
$$

o subproblema interno minimiza o Lagrangiano Aumentado

$$
L_\rho(x, \lambda, \mu) = f(x) + \frac{\rho}{2}\left[ \sum_i \left(h_i(x) + \frac{\lambda_i}{\rho}\right)^2 + \sum_j \max\left(0,\; g_j(x) + \frac{\mu_j}{\rho}\right)^2 \right].
$$

A cada iteração externa $k$:

1. **Subproblema** — minimiza $L_\rho$ a partir de $x_{k-1}$ usando o solver interno (CMA-ES ou TuRBO).
2. **Multiplicadores** — atualiza $\lambda \leftarrow \lambda + \rho\, h(x_k)$ e $\mu \leftarrow \max(0,\; \mu + \rho\, g(x_k))$.
3. **Penalidade** — aumenta $\rho \leftarrow \gamma \rho$ apenas quando a medida de inviabilidade não decresce o suficiente (fator $\tau$); caso contrário mantém $\rho$.
4. **Salvaguarda** — trunca os multiplicadores em $[\lambda_{\min}, \lambda_{\max}]$ e $[0, \mu_{\max}]$.
5. **Convergência** — para quando a inviabilidade $\max\big(\|h(x_k)\|,\; \|V_k\|\big) \le \varepsilon$, com $V_k = \min(-g(x_k),\; \mu/\rho)$.

Os solvers internos operam sempre no domínio normalizado $[0,1]^d$; cada problema converte essas variáveis para a escala física.

---

## Estrutura do repositório

```
optimization_algorithms/
│
├── algorithms/                      # Notebooks de estudo por método
│   ├── cma_introduction.ipynb       # Introdução ao CMA-ES
│   ├── cma_functions.ipynb          # CMA-ES em funções de teste
│   ├── turbo_tuto.ipynb             # TuRBO (Trust Region Bayesian Optimization)
│   ├── furbo_tuto.ipynb             # FuRBO (variante com região de confiança por amostragem)
│   ├── pso_functions.ipynb          # PSO (Particle Swarm Optimization)
│   ├── cobyla.ipynb                 # COBYLA (otimização restrita por interpolação linear)
│   ├── trust_region.ipynb           # Método de Trust-Region
│   └── plot_painel_functions.py     # Auxiliar de visualização de funções-teste
│
├── augmented_lagrangian/
│   └── augmented_lagrangian.ipynb   # Estudo do Lagrangiano Aumentado (numpy/scipy)
│
├── COCO/                            # Benchmarking no testbed bbob-constrained
│   └── augmented_lagrangian_coco.ipynb                      # Comparação AL-TuRBO / AL-CMA-ES / TuRBO / CMA-ES
│
├── SIMULATION_SCRIPTS/              # Pipeline aplicado: LA + solver interno + FEM
│   ├── lagrangian_aug_for_simulations.py   # Núcleo do Lagrangiano Aumentado (PHR)
│   │
│   ├── viga.py                      # Problema 1: viga de Timoshenko (FEM)
│   ├── quadrupolo.py                # Problema 2: quadrupolo magnético (FEM)
│   ├── difusion.py                  # Problema 3: difusão transiente (FEM)
│   ├── sanity.py                    # Problema 4: barra tracionada (ótimo analítico)
│   │
│   ├── run_optimization_viga_cma.py
│   ├── run_optimization_viga_turbo.py
│   ├── run_optimization_quadrupolo_cma.py
│   ├── run_optimization_quadrupolo_turbo.py
│   ├── run_sanity_cma.py
│   ├── run_sanity_turbo.py
│   │
│   ├── utils.py                     # Salvamento do histórico do LA em CSV
│   └── plot_historico.py            # Plot do histórico (f, restrições, parâmetros físicos)
│
├── cma_functions.ipynb              # (cópia do notebook na raiz)
├── LICENSE                          # MIT
└── README.md
```

---

## Problemas-teste

Cada problema expõe a mesma interface (`f`, `g`, `h`, `x0`, `x_to_params`, `verify_fem`, `exportar_resultados`), o que permite trocar o solver interno sem alterar a definição do problema. Todos têm 2–3 variáveis de projeto normalizadas em $[0,1]^d$.

| Problema | Arquivo | Variáveis | Objetivo | Restrições |
|---|---|---|---|---|
| **Viga de Timoshenko** | `viga.py` | $E, b, h$ (3) | minimizar massa | $g$: deflexão FEM $\le$ deflexão máxima |
| **Quadrupolo magnético** | `quadrupolo.py` | $d_\text{fio}, r_\text{fio}, J_0$ (3) | maximizar gradiente de campo $G$ | 2 desigualdades geométricas/físicas |
| **Difusão transiente** | `difusion.py` | $\alpha, a, T$ (3) | maximizar massa retida em região-alvo | 2 desigualdades + 1 igualdade (conservação) |
| **Barra tracionada (sanity)** | `sanity.py` | $A_1, A_2$ (2) | atingir deslocamento-alvo | restrição de rigidez; ótimo analítico $A^*=50\,\text{mm}^2$ |

O problema **barra tracionada** funciona como teste de sanidade: tem ótimo analítico conhecido, servindo para validar o acoplamento LA + solver antes de aplicá-lo aos problemas mais caros.


---

## Solvers internos

**CMA-ES** (via [`pycma`](https://github.com/CMA-ES/pycma)) — estratégia evolutiva com adaptação da matriz de covariância.
**TuRBO** (via [BoTorch](https://botorch.org/)/[GPyTorch](https://gpytorch.ai/)) — otimização Bayesiana com região de confiança. GP `SingleTaskGP` com kernel Matérn-5/2 ARD, candidatos por Thompson Sampling (`MaxPosteriorSampling`), inicialização Sobol, região de confiança adaptada por contadores de sucesso/falha.

---

## Benchmarking na suite bbob-constrained (`COCO/`)

Além da aplicação aos problemas FEM, o repositório inclui um estudo de **benchmarking padronizado** na plataforma [COCO](https://github.com/numbbo/coco) (Comparing Continuous Optimizers), usando o conjunto **bbob-constrained**. O objetivo é avaliar, de forma reprodutível e em larga escala, as vantagens de cada backbone para o Lagrangiano Aumentado.

Duas configurações são comparadas sob paridade de orçamento:

- **AL-TuRBO** e **AL-CMA-ES** — Lagrangiano Aumentado no laço externo, com o respectivo solver resolvendo cada subproblema (estado do solver reaproveitado entre iterações externas: comprimento da região de confiança no TuRBO, objeto da estratégia evolutiva no CMA-ES).

A avaliação usa **perfis de desempenho** (Dolan & Moré). O `bbob-constrained` contém restrições de desigualdade.

---

## Instalação


| Componente | Pacotes |
|---|---|
| Núcleo numérico | `numpy`, `scipy`, `matplotlib` |
| Otimização Bayesiana / TuRBO | `torch`, `botorch`, `gpytorch` |
| Estratégia evolutiva | `cma`, `cmaes` |
| PSO (notebook) | `pyswarms` |
| Benchmarking COCO | `cocoex`, `cocopp` |
| FEM (FEniCSx) | `dolfinx`, `ufl`, `basix`, `mpi4py`, `petsc4py` |

---

## Execução

Os scripts de pipeline devem ser executados a partir de `SIMULATION_SCRIPTS/` (os imports são relativos a essa pasta):

```bash
cd SIMULATION_SCRIPTS

# Viga de Timoshenko
python run_optimization_viga_cma.py
python run_optimization_viga_turbo.py

# Quadrupolo magnético
python run_optimization_quadrupolo_cma.py
python run_optimization_quadrupolo_turbo.py

# Teste de sanidade (ótimo analítico conhecido)
python run_sanity_cma.py
python run_sanity_turbo.py
```

Cada execução: (0) verifica o FEM contra a solução analítica de referência, (1) reporta o ponto inicial, (2) roda o Lagrangiano Aumentado com o solver interno escolhido, (3) imprime a solução ótima e (4) exporta resultados.

### Visualização do histórico

```bash
python plot_historico.py <caminho_do_csv> <problema>

# exemplos
python plot_historico.py results/viga_cma_historico_LA.csv viga
python plot_historico.py results/quadrupolo_turbo_historico_LA.csv quadrupolo
```

---

## Saídas

Os runners criam a pasta `results/` com:

- **`*_historico_LA.csv`** — histórico das iterações externas do LA (iteração `k`, objetivo `f`, inviabilidade `viab`, penalidade `rho`, variáveis `x`, restrições `g`/`h`), gerado por `utils.salvar_historico_csv`.
- **Arquivos `.bp` (ADIOS2/VTX)** e **PNGs** dos campos FEM (estado inicial e ótimo), gerados por `exportar_resultados` de cada problema.
- **PNG do histórico** com três painéis (objetivo, restrições e parâmetros físicos por iteração), via `plot_historico.py`.

---

## Notebooks de estudo (`algorithms/`)

Material exploratório usado para entender cada método isoladamente antes da integração: introdução e aplicação do CMA-ES, tutoriais de TuRBO e FuRBO, PSO sobre funções de teste, COBYLA e Trust-Region. O notebook `augmented_lagrangian/augmented_lagrangian.ipynb` apresenta o método do Lagrangiano Aumentado de forma autocontida (numpy/scipy), servindo de base para a implementação.

---

## Referências

- J. M. Martínez. *Otimização prática usando o Lagrangiano Aumentado*. Notas, IMECC/UNICAMP, 2006. — formulação PHR e laço externo com salvaguardas.
- N. Hansen. *The CMA Evolution Strategy: A Comparing Review*. 2006. — CMA-ES.
- D. Eriksson, M. Pearce, J. Gardner, R. Turner, M. Poloczek. *Scalable Global Optimization via Local Bayesian Optimization* (TuRBO). NeurIPS, 2019.
- M. Balandat et al. *BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization*. NeurIPS, 2020.
- E. D. Dolan, J. J. Moré. *Benchmarking Optimization Software with Performance Profiles*. Mathematical Programming, 2002.
- N. Hansen, A. Auger, R. Ros, O. Mersmann, T. Tušar, D. Brockhoff. *COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting*. Optimization Methods and Software, 2021.
- P. Dufossé, N. Hansen, D. Brockhoff, P. R. Sampaio, A. Atamna, A. Auger. *The bbob-constrained COCO Test Suite*. 2022.
- G. R. Cowper. *The Shear Coefficient in Timoshenko's Beam Theory*. J. Appl. Mech., 1966. — referência da viga.

---

## Licença

Distribuído sob a licença **MIT**. Ver [`LICENSE`](LICENSE).

## Autoria

- Edélio Gabriel Magalhães de Jesus (Ilum)

- Matheus Pereira Velloso da Silveira (Ilum)
