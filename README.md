# BareNN — Otimização de Rede Neural Feedforward: CPU e CUDA

**Disciplina:** Supercomputação 2026.2 — Profa. Licia Sales (Insper)
**Autor:** Vinicius Dib
**Rede:** Feedforward 64→32→16→10 (ReLU, Online SGD, lr=0.02, 51 épocas, seed 1337)
**Dataset:** optdigits (3823 treino, 1797 teste)

---

## Tabela Consolidada de Resultados

| Versão | Arquivo | Tempo (mean ± σ) | Speedup vs Baseline | Acurácia | Tag |
|:---|:---|:---|:---|:---|:---|
| **Baseline (legado)** | `barenn.cpp` | 1.343 s ± 0.096 s | 1.00x | 95.5% | [`v0.1-baseline`](../../releases/tag/v0.1-baseline) |
| **CPU Otimizado** | `barenn_cpu.cpp` | 0.563 s ± 0.037 s | **2.39x** | 95.6% | [`v0.2-cpu-opt`](../../releases/tag/v0.2-cpu-opt) |
| **CUDA Básico** | `barenn.cu` | 27.123 s ± 1.260 s | 0.05x | 95.4% | [`v0.3-cuda-basic`](../../releases/tag/v0.3-cuda-basic) |
| **CUDA Otimizado** | `barenn_cuda_opt.cu` | 16.735 s ± 0.039 s | 0.08x | 95.4% | [`v0.4-cuda-opt`](../../releases/tag/v0.4-cuda-opt) |

**Ambiente local:** WSL2 Ubuntu 24.04, i9-11900H @ 2.50GHz, RTX 3060 Laptop GPU (30 SMs, Compute 8.6)
**Compilação CPU:** `g++ -O3 -fopenmp -march=native -std=c++17` | **CUDA:** `nvcc -arch=sm_86 -O3 -std=c++17`

### Comparativo: Local vs Cluster Franky (Insper HPC)

| Versão | Local (i9 + RTX 3060 Laptop) | Franky (Xeon + RTX A1000) | Razão Franky/Local |
|:---|:---|:---|:---|
| **Baseline (legado)** | 1.343 s | **0.471 s** | 2.85x mais rápido |
| **CPU Otimizado (16T)** | 7.650 s | 9.867 s | 0.78x (overhead OpenMP similar) |
| **CUDA Básico** | 27.123 s | **7.073 s** | 3.83x mais rápido |
| **CUDA Otimizado** | 16.735 s | **4.253 s** | 3.94x mais rápido |

**Cluster Franky:** Intel Xeon (nó de computação), NVIDIA RTX A1000 (18 SMs, Compute 8.6, 7838 MB VRAM)

**Análise cross-environment:**
- O Xeon do Franky é ~2.85x mais rápido que o i9 mobile no baseline (clock + cache superiores em servidor)
- A RTX A1000 (18 SMs) é ~3.9x mais rápida que a RTX 3060 Laptop (30 SMs) nos kernels CUDA — apesar de ter menos SMs, o ambiente de servidor tem menor overhead de driver/OS e melhor throughput PCIe
- No Franky, o **CUDA otimizado (4.25s)** é mais rápido que o **CUDA básico (7.07s)** — as otimizações de streams/pinned memory/fused kernels dão **1.66x speedup**, consistente com o 1.62x observado localmente
- O OpenMP permanece ineficaz em ambos ambientes para camadas pequenas (Lei de Amdahl)
- Mesmo no Franky, CUDA opt (4.25s) é 9x mais lento que o baseline CPU (0.47s) — confirmando que redes pequenas não beneficiam de GPU

Dados completos: [`profiling/franky_all_output.txt`](profiling/franky_all_output.txt)

---

## Índice de Issues (Relatório Técnico)

| # | Issue | Foco | Resultado Principal |
|:---|:---|:---|:---|
| 1 | [Profiling do baseline — callgrind, perf stat, hyperfine](../../issues/1) | Identificação de gargalos | `Backward()` = 68.23% das instruções, 442K cache misses |
| 2 | [Otimização CPU — flatten, const ref, pré-alocação, OpenMP](../../issues/2) | Otimização de memória + paralelismo | **2.39x speedup**, -54% instruções; OpenMP ineficaz para camadas pequenas |
| 3 | [Migração CUDA básica — kernels forward/backward/loss](../../issues/3) | Porting para GPU | 48x mais lento que CPU; overhead de launch/memcpy domina |
| 4 | [Otimizações CUDA avançadas — shared mem, fused kernels, streams](../../issues/4) | Redução de overhead CUDA | **1.62x** vs CUDA básico; memcpy -94%, launches -42% |

---

## Estrutura do Repositório

```
├── barenn.cpp              # Código legado (original do fork)
├── barenn_cpu.cpp           # CPU otimizado (flatten, const ref, pré-alocação)
├── barenn.cu                # CUDA básico (5 kernels, online SGD)
├── barenn_cuda_opt.cu       # CUDA otimizado (shared mem, streams, fused kernels)
├── profiling/               # Dados de profiling (callgrind, perf, nsys, hyperfine)
├── slurm/                   # Scripts SLURM para cluster Franky (Insper HPC)
└── README.md
```

## Compilação e Execução

```bash
# Baseline (legado)
g++ -O2 -std=c++17 barenn.cpp -o barenn && ./barenn

# CPU otimizado
g++ -O3 -fopenmp -march=native -std=c++17 barenn_cpu.cpp -o barenn_cpu && ./barenn_cpu

# CUDA básico
nvcc -arch=sm_86 -O2 -std=c++17 barenn.cu -o barenn_cuda && ./barenn_cuda

# CUDA otimizado
nvcc -arch=sm_86 -O3 -std=c++17 barenn_cuda_opt.cu -o barenn_cuda_opt && ./barenn_cuda_opt
```

## Discussão Geral

A otimização CPU demonstrou que **layout de memória é o fator dominante** para redes pequenas: flatten de `vector<vector<float>>` para memória contígua, eliminação de cópias por valor e pré-alocação de buffers reduziram instruções em 54% e cache misses em 56%. OpenMP foi testado (1-16 threads) e demonstrado como contraproducente para camadas de 10-32 neurônios — um caso clássico de granularidade insuficiente para paralelismo (Lei de Amdahl).

A migração CUDA revelou que **GPUs não são panaceia**: para redes com camadas pequenas, o overhead de kernel launch (~2.9 μs × 2.3M lançamentos) e memcpy síncrono (~35 μs × 390K chamadas) supera o tempo de computação efetiva. As otimizações avançadas (shared memory, fused kernels, CUDA streams, pinned memory) reduziram o overhead em 38%, mas a GPU permanece mais lenta que a CPU. Isso é esperado e demonstra compreensão crítica do trade-off overhead vs computação em GPGPU.

---

# Projeto Barenn: O Legado, a Sobrevivência e o Salto para a GPU

Era uma segunda-feira aparentemente tranquila quando a demanda chegou à sua mesa. A liderança técnica se aproximou com um sorriso contido e soltou aquela frase clássica que faz qualquer engenheiro gelar por dentro:

> "Temos um componente crítico rodando em produção há anos. O desenvolvedor original já saiu da empresa, mas agora o projeto é oficialmente seu."
    
Você procurou documentação, manuais ou notas de arquitetura e não encontrou rigorosamente nada. Restou apenas um código hermético a correr em produção.

Depois de analisar o código entendeu que é a implementação de uma Feedforward Neural Network, mas por que razão reinventar a roda? No passado, exigências estritas de conformidade e restrições de licenciamento impediram o recurso a bibliotecas de mercado. Sem alternativa, a empresa optou por criar a sua própria implementação de raiz. O resultado funcionou durante anos, mas tornou-se um labirinto técnico esquecido.

Com o aumento exponencial do volume de dados, a máquina atual começou a falhar nos prazos operacionais. Antes de qualquer migração complexa, **o seu primeiro desafio é estabilizar o código e espremer todo o desempenho possível da CPU — atacando gargalos, repensando o acesso à memória e explorando o paralelismo nativo.**

Para resolver o problema a longo prazo, a empresa adquiriu servidores blade equipados com GPUs corporativas NVIDIA. O hardware de alto desempenho já está montado no datacenter: **a sua missão final é reescrever os pontos críticos do algoritmo em CUDA e levar o processamento ao nível do paralelismo massivo.**

---

## Diretrizes de Conduta e Governança no Repositório

Neste projeto, o repositório não é apenas um local de entrega de arquivos, mas a **comprovação do seu processo investigativo de engenharia**. Espera-se uma postura estritamente empírica: nenhuma otimização deve ser feita no escuro e nenhuma melhoria de velocidade tem valor se corromper o resultado final do programa.

---

### O Princípio do Bloco Lógico de Trabalho

Você tem autonomia para decidir **o que** otimizar e **em qual ordem**, mas cada entrega deve ser estruturada como um **bloco lógico fechado**.

> **O que é um bloco lógico?**
> É uma unidade completa de intervenção técnica que engloba: **um problema identificado por profiling $\rightarrow$ uma hipótese de melhoria $\rightarrow$ a alteração no código $\rightarrow$ a validação da corretude $\rightarrow$ a medição do ganho real.**

* **Não crie tags por tempo decorrido:** Evite tags como `versao-dia-2` ou commits de "fim de expediente".
* **Não crie tags para alterações inconclusivas:** Se a alteração piorou o desempenho ou quebrou os testes, documente na Issue, reverta ou ajuste antes de formalizar o marco.
* **Toda tag deve ser reprodutível:** O professor ou qualquer colega deve conseguir executar `git checkout <tag>`, compilar o código e obter exatamente o mesmo comportamento e ordem de grandeza de desempenho documentados.

---

### O Fluxo Esperado no GitHub

Para cada intervenção técnica planejada, seu ciclo de trabalho no repositório deve seguir três pilares:

#### 1. Abertura de Issue (A Hipótese e o Ponto de Partida)

Antes de alterar uma única linha de código, abra uma Issue explicando a intenção do bloco lógico:

* **Diagnóstico:** Qual gargalo você encontrou usando ferramentas de análise (profilers)?
* **Hipótese:** Qual conceito técnico você pretende aplicar para resolver esse gargalo?
* **Métrica Base:** Qual é o tempo ou custo de processamento atual dessa rotina?

#### 2. Desenvolvimento Isolado e Commits Semânticos

* Trabalhe em uma *branch* dedicada àquela hipótese.
* Faça commits descritivos focados na causa da mudança (ex: `perf: ...`, `refactor: ...`, `fix: ...`).
* Certifique-se de que a saída do modelo/algoritmo continua correta. Código rápido que calcula resultados errados é considerado código quebrado.

#### 3. Formalização com Tag Anotada e Fechamento da Issue

Quando o bloco estiver concluído e validado, crie uma **Git Tag anotada**:

* A tag marca o estado exato da base de código naquele degrau de maturidade.
* A mensagem da tag deve resumir sucintamente o resultado alcançado.
* Feche a Issue anexando as evidências (dados de tempo, prints do profiler ou tabelas comparando o "antes e depois").

---

### Exemplo Ilustrativo de um Bloco Lógico

*(Este é apenas um exemplo de como organizar a narrativa técnica e as ferramentas; não representa um passo obrigatório ou a solução do seu problema).*

---

#### 1. Cenário: O aluno percebeu lentidão na etapa de processamento sequencial

* **Abertura da Issue #04:**
> **Título:** *Redução de tempo de computação na rotina de cálculo vetorial*
> **Contexto:** Ao inspecionar o código com o profiler, identificou-se que 65% do tempo de execução é consumido em um laço interno que realiza operações matemáticas repetitivas.
> **Métricas atuais:** Tempo total de execução: `14.2s`.
> **Hipótese:** Reestruturar a ordem dos dados para favorecer o uso de registradores e paralelizar o laço com múltiplos núcleos deve reduzir o tempo dessa rotina.


* **Implementação na branch:** `feature/otimizacao-laco-vetorial`
* Commits atômicos ajustando o laço e aplicando a estratégia escolhida.
* Verificação de que o resultado final do programa bate com o original.


* **Criação da Tag:**
```bash
git tag -a v0.3-laco-paralelo -m "Otimizacao do calculo vetorial: tempo reduzido de 14.2s para 4.8s (Speedup 2.95x). Corretude validada."
git push origin v0.3-laco-paralelo

```


* **Fechamento da Issue:**
* O aluno responde na Issue #04 anexando o comparativo de tempo (antes: `14.2s` / depois: `4.8s`), faz o merge da branch e referencia a tag criada:


> *"Otimização implementada e validada na tag `v0.3-laco-paralelo`. Issue fechada pelo PR #05."*


---

# Rubrica de Avaliação (Total: 10,0 Pontos)

<div style="background:rgba(255, 0, 0, 0.65)"> 

> ### ⚠️ CLÁUSULA ELIMINATÓRIA GERAL (CONDIÇÃO DE NOTA ZERO)
> 
> 
> A avaliação adota uma política estrita de controle de qualidade de engenharia:
> **Se em QUALQUER um dos critérios ou subcritérios de avaliação for observado o indicativo de `0,0 ponto`, a ATIVIDADE SERÁ INTEGRALMENTE ZERADA (Nota Final = 0,0).**
> Não é tolerada a ausência completa de qualquer pilar fundamental da atividade (seja por código inoperante/incorreto, falta de governança no Git ou ausência de embasamento empírico nas Issues).

* **Implementação e Desempenho Técnico:** 4,0 pontos
* **Governança, Reprodutibilidade e Tags:** 2,0 pontos
* **Relatório Técnico nas Issues (Análise e Discussão):** 4,0 pontos

</div>

---

### Critério 1: Implementação e Desempenho Técnico (4,0 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — O código não compila, aborta durante a execução, não é funcional ou gera saídas matematicamente corrompidas.
* **+0,5 ponto** — Executou com sucesso o código legado, mediu o baseline de referência de forma reproduzível e identificou os gargalos iniciais.
* **+1,0 ponto** — Aplicou otimizações eficazes na CPU com ganhos consistentes de velocidade e paralelismo multicore preservando a integridade numérica.
* **+1,0 ponto** — **Migração Funcional para CUDA:** Porting inicial para GPU completo e correto (kernel funcional, alocação e liberação de memória no Device e transferências Host/Device íntegras).
* **+0,75 ponto** — Otimizou a execução em GPU.
* **+0,75 ponto** — Implementou técnicas avançadas em GPU, comprovando ganhos.

---

### Critério 2: Governança, Reprodutibilidade e Tags (2,0 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — Ausência de tags, ou repositório com commit único ("tudo pronto") / commits lineares sem separação lógica, inviabilizando a reprodutibilidade dos marcos intermediários.
* **+0,5 ponto** — Utilizou tags, mas elas marcam datas/horários arbitrários em vez de blocos lógicos fechados de trabalho, ou as tags não compilam de forma independente.
* **+1,2 ponto** — Trabalho estruturado em **blocos lógicos**: cada intervenção possui uma Git Tag anotada associada a uma Issue, permitindo compilar e checar a métrica de cada versão via `git checkout <tag>`.
* **+2,0 pontos** — Reprodutibilidade e histórico impecáveis: commits atômicos com mensagens semânticas (`perf:`, `refactor:`, `test:`), branches limpas integradas via Pull Requests documentados e rastreabilidade total de cada degrau de performance.

---

### Critério 3: Relatório Técnico nas Issues (Análise e Discussão) (4,0 pontos)

#### 3.1 Profiling e Evidências Empíricas (1,5 ponto)

* **0,0 ponto [ZERA A ATIVIDADE]** — Não apresenta dados de medição nas Issues ou baseia-se unicamente em impressões subjetivas sem qualquer dado empírico.
* **+0,5 ponto** — Apresenta apenas tempos brutos de relógio (*wall-clock*) colados em texto corrido, sem dados de profilers ou repetições estatísticas.
* **+1,0 ponto** — Apresenta dados e capturas das ferramentas oficiais (**perf/callgrind** e **Nsight Systems/Compute**), mas faltam informações cruciais de ambiente (flags, parâmetros de teste ou hardware).
* **+1,5 ponto** — Issues contêm relatórios completos de profiling (CPU e GPU), métricas de hardware detalhadas (tempo de kernel vs transferência PCIe, cache misses, ocupação teórica vs real de SMs) e metodologia experimental rigorosa.

#### 3.2 Qualidade da Discussão Técnica e Argumentação de Engenharia (2,5 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — Discussão inexistente, Issues fechadas apenas com "ok/concluído" ou explicações inconsistentes/conflitantes com os dados coletados.
* **+1,0 ponto** — Discussão puramente descritiva (lista o que foi alterado nas linhas de código, mas não analisa *por que* houve ou não ganho com base em hardware).
* **+1,8 ponto** — Discute tecnicamente a relação entre o código e o comportamento de hardware (ex: impacto do layout de memória nos caches da CPU, balanceamento de carga entre threads).
* **+2,5 pontos** — Análise técnica aprofundada: correlaciona os resultados com conceitos avançados de arquitetura (roofline model, limites da Lei de Amdahl, latência do PCIe vs throughput do kernel CUDA) e debate criticamente as decisões de engenharia tomadas.

---

## Tabela de Cenários e Balizadores

| Cenário | Situação Técnica no GitHub | Resultado / Nota |
| --- | --- | --- |
| **Desclassificado** | Tirou `0,0` em qualquer critério (ex: código não compila, sem tags, sem dados de profiling nas issues ou issues vazias). | **0,0 (Zerada)** |
| **Abaixo do Esperado** | Código compila e funciona, mas há otimizações superficiais, profiling fraco e tags incompletas. | **0,1 a 5,9** |
| **Linha de Base (Alvo: 6,0)** | CPU razoavelmente otimizada + **Migração simples para CUDA funcional** + Issues estruturadas com diagnóstico e tags atreladas. | **6,0 a 6,5** |
| **Avançado / Destaque** | CPU altamente eficiente + **CUDA com otimizações avançadas (Shared Memory/Streams)** + discussões ricas nas Issues com Nsight/Perf + governança impecável. | **8,5 a 10,0** |

---

## Formato de Entrega

A entrega consiste exclusivamente no **link do repositório no GitHub**.

1. O `README.md` principal deve atuar como o **sumário executivo do relatório**, sendo adiciocionado **NO TOPO DO README**:
* A tabela consolidada final comparando todas as versões (Baseline Legado, CPU Otimizada, CUDA Inicial, CUDA Otimizado);
* Um **índice navegável com links diretos** para cada uma das Issues/Tags que contêm a discussão detalhada;

2. Todas as branches de trabalho devem estar integradas à `main`;
3. Todas as Issues de experimentos devem estar fechadas e associadas à sua respectiva Git Tag anotada.
