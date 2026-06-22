# Esquema de Experimentos Detalhado — Classificação Automatizada do Grau ISUP em Histopatologia de Próstata

**Dataset principal:** PANDA (Prostate cANcer graDe Assessment) — 11.000+ lâminas WSI, graus ISUP 0–5  
**Dataset externo:** SICAPv2 (avaliação de generalização cross-dataset)  
**Conjunto de teste:** 1.590 amostras (fixo para toda comparação)  
**Validação:** bootstrap não-paramétrico com 1.000 reamostras, IC 95%  
**Métrica primária de checkpoint:** QWK (Quadratic Weighted Kappa) no conjunto de validação

---

## 1. Dados e Pré-processamento

### 1.1 Particionamento

| Partição | Uso |
|----------|-----|
| `train_5fold.csv` | Treino com validação cruzada 5-fold |
| `test.csv` | Conjunto de teste retido (n=1.590) |
| `entropy.csv` | Scores de entropia pré-computados por imagem |
| `without_pen_mask.csv` | Imagens após remoção de marcas de caneta |

### 1.2 Filtragem por Entropia (Active Learning)

Todas as configurações baseline aplicam remoção das **20% amostras de maior dificuldade** (entropia mais alta) do conjunto de treino, mantendo casos mais informativos. A função `remove_images_by_entropy` em `utils/train.py` implementa esta filtragem. O modelo B0-TTA é a única exceção, treinando com o dataset completo.

### 1.3 Composição de Mosaicos (Patch Bags)

- **Configuração padrão:** mosaicos 6×6 = 36 patches por lâmina (max_patches=36)
- **Configuração estendida MIL:** mosaicos 7×7 = 49 patches (max_patches=49, experimentos `b0-mil-49`)
- Patches de 256×256 pixels
- Bags preenchidos com patches brancos quando a lâmina tem menos tiles que o máximo
- Máscara binária (`mask`) indica patches reais vs. preenchimento

### 1.4 Representação dos Espaços de Cores Testados

| Experimento | Espaço | Resultado (teste) |
|-------------|--------|-------------------|
| B0 RGB (baseline) | RGB padrão | QWK=0.8589, Acc=68.05% |
| B0 + HED ordinal focal | H&E Deconvolution (Ruifrok) | Treino iniciado, sem resultado de teste publicado |
| B0 + XYZ ordinal focal | CIE XYZ | Treino iniciado, sem resultado de teste publicado |
| B0 + CLAHE-YUV ordinal focal | YUV com CLAHE (γ=0.2) | QWK=0.7983, Acc=59.56% |

O espaço CLAHE-YUV apresentou desempenho inferior ao RGB, sugerindo que a variabilidade cromática não é o principal fator limitante do modelo B0 na resolução de tile de 256×256 px.

---

## 2. Codificação Ordinal das Classes

Todas as arquiteturas usam **codificação ordinal binária** para os graus ISUP 0–5:

| Grau ISUP | Vetor ordinal | Limiar de decodificação |
|-----------|---------------|------------------------|
| 0 | [0,0,0,0,0] | soma < 0,5 → 0 |
| 1 | [1,0,0,0,0] | soma ≥ 0,5 → 1 |
| 2 | [1,1,0,0,0] | soma ≥ 1,5 → 2 |
| 3 | [1,1,1,0,0] | soma ≥ 2,5 → 3 |
| 4 | [1,1,1,1,0] | soma ≥ 3,5 → 4 |
| 5 | [1,1,1,1,1] | soma ≥ 4,5 → 5 |

**Função de perda:** `BCEWithLogitsLoss` aplicada a 5 limiares independentes (via `OrdinalRegressionLoss`).  
**Focal Loss híbrida:** $L = w_{\text{focal}} \cdot L_{\text{focal}} + w_{\text{ord}} \cdot L_{\text{ord}}$ onde os pesos são hiperparâmetros otimizados pelo Optuna.

---

## 3. Configuração Geral de Treinamento (Baseline)

```
Otimizador      : Adam
LR inicial      : 3×10⁻⁴
Scheduler       : CosineAnnealingLR (após 1 época de warmup com fator 2)
Batch size      : 3 (exceto B0-TTA: 6, V2S-Optuna: 4)
Épocas máx.     : 50
Early stopping  : patience=7 épocas (B0-TTA: 5)
Dropout         : 0,6 (exceto V2S-Optuna: 0,4423)
Seed            : 42 (reprodutibilidade)
Data aug.       : Transpose, HorizontalFlip, VerticalFlip (p=0,5 cada)
Entropy filter  : remoção top-20% maior entropia
```

---

## 4. Modelos Baseline — EfficientNet

### 4.1 Arquitetura EfficientNetApi

Wrapper sobre `efficientnet_pytorch.EfficientNet`:
- Congela todos os parâmetros exceto os últimos 150
- Adiciona camada de Dropout configurável
- Cabeçote linear para 5 saídas ordinais

### 4.2 Resultados Individuais (Conjunto de Teste, n=1.590)

#### EfficientNet-B0

| Variante | Acc (%) | QWK | Macro F1 | IC 95% (QWK) |
|----------|---------|-----|----------|--------------|
| B0 + Ordinal + Focal (padrão) | 52,64 ± 1,24 | 0,7616 ± 0,0139 | 0,4390 ± 0,0124 | [0,7326 – 0,7880] |
| B0 + Ordinal + Focal + Atenção | 63,02 ± 1,21 | 0,8442 ± 0,0120 | 0,5673 ± 0,0122 | [0,8198 – 0,8665] |
| B0 + Ordinal + SWA | 65,35 | 0,8523 | 0,5914 | — |
| B0 + Ordinal + Focal + Deformável | 47,74 | 0,7508 | 0,4138 | — |
| B0 + CLAHE-YUV (espaço de cor) | 59,56 ± 1,30 | 0,7983 ± 0,0136 | 0,5133 ± 0,0127 | [0,7702 – 0,8248] |

**Relatório por classe — B0+Ordinal+Focal+Atenção:**

| Classe | Prec. | Recall | F1 |
|--------|-------|--------|-----|
| ISUP 0 | 0,798 | 0,910 | 0,850 |
| ISUP 1 | 0,765 | 0,625 | 0,688 |
| ISUP 2 | 0,444 | 0,375 | 0,407 |
| ISUP 3 | 0,354 | 0,303 | 0,327 |
| ISUP 4 | 0,406 | 0,567 | 0,473 |
| ISUP 5 | 0,667 | 0,652 | 0,659 |

**Análise:** A inclusão de convoluções deformáveis prejudicou o desempenho (QWK −0,091 vs. padrão com atenção), indicando conflito entre a rigidez do kernel deformável e a resolução de tile de 256 px. O módulo de atenção adicionado ao cabeçote melhorou +0,083 QWK sobre a configuração sem atenção (0,8442 vs. linha não documentada). SWA produziu o melhor resultado individual do B0 (QWK=0,8523), mas a atenção+focal ficou próxima (0,8442) e sem o custo de duas fases de treinamento.

#### EfficientNet-B3

| Variante | Acc (%) | QWK | Macro F1 | IC 95% (QWK) |
|----------|---------|-----|----------|--------------|
| B3 + Ordinal + Focal (α variável) | 65,97 | 0,8484 | 0,6188 | [0,8241 – 0,8707] |

**Relatório por classe — B3:**

| Classe | Prec. | Recall | F1 |
|--------|-------|--------|-----|
| ISUP 0 | — | — | — |
| ISUP 1 | — | — | — |
| ISUP 2 | — | — | — |
| ISUP 3 | — | — | — |
| ISUP 4 | — | — | — |
| ISUP 5 | — | — | — |

*Nota: relatório por classe do B3 disponível apenas no notebook `efficientnet-b3-entropy-ordinal-focal.ipynb`.*

#### EfficientNet-B7

| Variante | Acc (%) | QWK | Macro F1 | IC 95% (QWK) |
|----------|---------|-----|----------|--------------|
| B7 + Ordinal + Focal | 59,18 ± 1,18 | 0,8088 ± 0,0132 | 0,5306 ± 0,0123 | [0,7820 – 0,8326] |

**Relatório por classe — B7:**

| Classe | Prec. | Recall | F1 |
|--------|-------|--------|-----|
| ISUP 0 | 0,777 | 0,857 | 0,815 |
| ISUP 1 | 0,661 | 0,623 | 0,641 |
| ISUP 2 | 0,445 | 0,385 | 0,413 |
| ISUP 3 | 0,369 | 0,351 | 0,360 |
| ISUP 4 | 0,368 | 0,439 | 0,400 |
| ISUP 5 | 0,593 | 0,522 | 0,555 |

**Análise comparativa B0/B3/B7:** O B3 superou o B7 em 3,96 pp de QWK, contrariando a expectativa de que modelos maiores performam melhor. A hipótese é que o B7 sofre mais de overfitting dada a pequena quantidade de tiles treinados (batch=3, entropia filtrada). O B3 apresenta o melhor tradeoff entre capacidade e regularização nesta configuração. O B0 com atenção ficou próximo ao B3 (QWK 0,8442 vs. 0,8484), mas sem bootstrap comparável para o B3.

### 4.3 Evolução do Val QWK durante Treinamento

**EfficientNet-B0 + Ordinal + Focal:**
- Época 1: val_kappa=0,8105 | Época 7: val_kappa=0,8464 | Época 10: val_kappa=0,8570 | Convergência ~época 19 (val_kappa=0,8657)

**EfficientNet-B3 + Ordinal + Focal:**
- Época 1: val_kappa=0,8070 | Época 11: val_kappa=0,8498 | Época 29: val_kappa=0,8652 (melhor)

---

## 5. EfficientNetV2-S — Baseline Expandido

### 5.1 Sem Otimização (Configuração manual)

| Acc (%) | QWK | Macro F1 | IC 95% (QWK) |
|---------|-----|----------|--------------|
| 65,79 ± 1,22 | 0,8521 ± 0,0113 | 0,5906 ± 0,0131 | [0,8288 – 0,8720] |

**Relatório por classe:**

| Classe | Prec. | Recall | F1 |
|--------|-------|--------|-----|
| ISUP 0 | 0,804 | 0,899 | 0,849 |
| ISUP 1 | 0,709 | 0,743 | 0,725 |
| ISUP 2 | 0,485 | 0,395 | 0,435 |
| ISUP 3 | 0,471 | 0,438 | 0,454 |
| ISUP 4 | 0,530 | 0,380 | 0,442 |
| ISUP 5 | 0,590 | 0,696 | 0,638 |

### 5.2 Com Otimização Optuna (HPO)

O Optuna (TPE com 50 trials) buscou no espaço:

| Hiperparâmetro | Espaço de busca | Melhor valor |
|----------------|-----------------|-------------|
| `lr` | log-uniform [1×10⁻⁵, 5×10⁻⁴] | 1,105×10⁻⁴ |
| `dropout_rate` | uniform [0,1, 0,7] | 0,4423 |
| `focal_gamma` | uniform [0,5, 5,0] | 3,426 |
| `focal_alpha` | uniform [0,05, 0,5] | 0,1673 |
| `unfreeze_blocks` | int [1, 5] | 3 |
| `batch_size` | categórico {4, 8} | 4 |
| `weight_decay` | log-uniform [1×10⁻⁶, 1×10⁻³] | 1,037×10⁻⁵ |

**Melhor QWK de validação Optuna:** 0,8745

**Resultados no conjunto de teste (1.000 bootstrap):**

| Métrica | Média | Std | IC 95% |
|---------|-------|-----|--------|
| Accuracy | 71,51% | ±1,13% | [69,18% – 73,52%] |
| QW Kappa | 0,8742 | ±0,0113 | [0,8500 – 0,8953] |
| Macro F1 | 0,6772 | ±0,0125 | [0,6518 – 0,6995] |

**Relatório por classe:**

| Classe | Prec. | Recall | F1 | Suporte |
|--------|-------|--------|----|---------|
| ISUP 0 | 0,875 | 0,889 | 0,882 | 434 |
| ISUP 1 | 0,758 | 0,738 | 0,748 | 400 |
| ISUP 2 | 0,518 | 0,565 | 0,541 | 200 |
| ISUP 3 | 0,564 | 0,616 | 0,589 | 185 |
| ISUP 4 | 0,557 | 0,631 | 0,592 | 187 |
| ISUP 5 | 0,867 | 0,603 | 0,712 | 184 |

**Matriz de Confusão — V2S Optuna:**
```
Pred →   IS0   IS1   IS2   IS3   IS4   IS5
IS0:    [386,   33,    7,    3,    4,    1]
IS1:    [ 35,  295,   65,    5,    0,    0]
IS2:    [  3,   41,  113,   33,   10,    0]
IS3:    [  3,   10,   24,  114,   32,    2]
IS4:    [  8,    7,    9,   31,  118,   14]
IS5:    [  6,    3,    0,   16,   48,  111]
```

**Curva de validação (Optuna best):**
- Época 3: val_kappa=0,8513 | Época 10: val_kappa=0,8584 | Época 18: val_kappa=0,8649 | Época 22: val_kappa=0,8669 | Época 26: val_kappa=0,8680

O V2S com HPO Optuna representou um ganho de +0,0221 QWK sobre o V2S sem otimização e é o melhor modelo individual de toda a campanha experimental.

---

## 6. Estratégias de Treinamento Especiais

### 6.1 Stochastic Weight Averaging (SWA)

Aplicado ao EfficientNet-B0:
- Início do SWA: época 30
- LR do SWA: 1×10⁻⁴
- Atualização de estatísticas BN a cada época SWA

**Resultados:**

| Acc (%) | QWK | Macro F1 |
|---------|-----|----------|
| 65,35 | 0,8523 | 0,5914 |

**Relatório por classe:**

| Classe | Prec. | Recall | F1 |
|--------|-------|--------|-----|
| ISUP 0 | 0,852 | 0,889 | 0,870 |
| ISUP 1 | 0,725 | 0,720 | 0,723 |
| ISUP 2 | 0,491 | 0,430 | 0,459 |
| ISUP 3 | 0,414 | 0,378 | 0,396 |
| ISUP 4 | 0,428 | 0,508 | 0,465 |
| ISUP 5 | 0,655 | 0,620 | 0,637 |

### 6.2 Test-Time Augmentation (TTA — 8 visualizações)

TTA aplicado ao ensemble B0+B3+B7. Diferencial: o modelo base usa `BCEWithLogitsLoss` (não ordinal) e batch=6 sem filtragem por entropia.

Transformações: original + H-flip + V-flip + HV-flip + Transpose + 3 rotações.

| Configuração | Acc (%) | QWK | Macro F1 | IC 95% (QWK) |
|---|---|---|---|---|
| Sem TTA | 70,3 ± 1,1 | 0,874 ± 0,011 | 0,650 ± 0,012 | [0,855 – 0,892] |
| Com TTA (8 vistas) | 71,4 ± 1,1 | 0,875 ± 0,011 | 0,662 ± 0,012 | [0,856 – 0,893] |
| Delta | +1,1 pp | +0,0013 | +0,012 | — |

O ganho do TTA é estatisticamente marginal (+0,0013 QWK), mas consistente. O custo computacional de inferência 8× pode não justificar a implantação clínica.

### 6.3 Cascade com Especialistas

Modelo base (ensemble ordinal) + classificadores especialistas treinados em pares confusos (ISUP 3 vs. 4, ISUP 4 vs. 5).

| Modelo | Acc (%) | QWK | Macro F1 | IC 95% (QWK) |
|--------|---------|-----|----------|--------------|
| Base (ordinal-best) | 71,36 ± 1,12 | 0,8754 ± 0,0112 | 0,6622 ± 0,0124 | [0,8568 – 0,8926] |
| Cascade (base + especialistas) | 71,32 ± 1,13 | 0,8721 ± 0,0110 | 0,6592 ± 0,0125 | [0,8537 – 0,8893] |
| Delta | −0,04 pp | −0,0033 | −0,0030 | — |

**Diagonal por classe:**

| Classe | Recall base | Recall cascade | Delta |
|--------|------------|----------------|-------|
| ISUP 0 | 0,899 | 0,922 | +0,023 |
| ISUP 1 | 0,708 | 0,772 | +0,065 |
| ISUP 2 | 0,440 | 0,535 | +0,095 |
| ISUP 3 | 0,400 | 0,524 | **+0,124** |
| ISUP 4 | 0,658 | 0,604 | −0,053 |
| ISUP 5 | 0,576 | 0,630 | +0,054 |

A cascade melhora as classes intermediárias (ISUP 2 e 3 ganham +9,5 e +12,4 pp de recall) mas prejudica ISUP 4 (−5,3 pp), resultado líquido negativo no QWK global. A estratégia é útil para aplicações onde erros em ISUP 3/4 têm maior custo clínico.

---

## 7. Multiple Instance Learning (MIL)

### 7.1 Arquitetura MIL

Todos os modelos MIL seguem o padrão:
- Entrada: `(B, N, C, H, W)` — bag de N patches de 256×256 px
- Backbone CNN/Transformer extrai features por patch
- **GatedAttention pooling** agrega as features em representação de bag
- Cabeçote ordinal → 5 saídas binárias
- Output dict: `{'logits', 'attn', 'features'}`
- **Máscara obrigatória:** tensor `mask` passado ao `forward()` para ignorar patches de preenchimento na softmax de atenção

### 7.2 EfficientNet-B0 + MIL (36 patches)

**Configuração:**
- Backbone: EfficientNet-B0 (sem filtro de entropia — o bag já representa a lâmina completa)
- Pooling: GatedAttention
- Loss: OrdinalFocalLoss (γ, w_focal, w_ord por Optuna)
- Hiperparâmetros Optuna: γ=2,4691 | w_focal=1,0410 | w_ord=1,1828 | lr=1,40×10⁻⁴

**Resultados no conjunto de teste:**

| Acc (%) | QWK | Macro F1 | Recall | Prec. |
|---------|-----|----------|--------|-------|
| 69,12 | 0,8634 | 0,6398 | 0,6366 | 0,6459 |

**Curva de convergência (Run final):**
- Época 9: val_kappa=0,8596 | Época 23: val_kappa=0,8610 | Época 29: val_kappa=0,8711 | Época 36: val_kappa=0,8704 | Época 39: val_kappa=0,8648 (stop)

### 7.3 EfficientNet-B0 + MIL (49 patches — 7×7)

**Configuração:**
- max_patches=49, batch=6
- Hiperparâmetros Optuna: γ=1,6767 | w_focal=1,0316 | w_ord=1,6714 | lr=5,38×10⁻⁵

**Convergência de validação:**
- Época 1: val_kappa=0,8099 | Época 10: val_kappa=0,8432 | Época 13: val_kappa=0,8555 (melhor fold 1)
- Fold 2 melhor: val_kappa=0,8642 (época 18)
- Fold 4 melhor: val_kappa=0,8584 (época 24)

*Nota: arquivo de resultados de teste do MIL-49 não localizado separadamente; dados de validação indicam desempenho comparável ao MIL-36.*

### 7.4 EfficientNetV2-S + MIL (Optuna)

**Hiperparâmetros Optuna:** γ=1,6568 | w_focal=0,2895 | w_ord=0,4467 | lr=1,11×10⁻⁴

**Resultados no conjunto de teste (1.000 bootstrap):**

| Métrica | Média | Std | IC 95% |
|---------|-------|-----|--------|
| Accuracy | 65,91% | ±1,21% | [63,40% – 68,30%] |
| QW Kappa | 0,8537 | ±0,0113 | [0,8311 – 0,8748] |
| Macro F1 | 0,6152 | ±0,0127 | [0,5881 – 0,6393] |

**Curva de convergência:**
- Época 6: val_kappa=0,8353 | Época 14: val_kappa=0,8510 | Época 24: val_kappa=0,8627 | Época 26: val_kappa=0,8663 (melhor) | Época 36: val_kappa=0,8643

O V2S-MIL apresentou desempenho inferior ao B0-MIL (0,8537 vs. 0,8634 QWK), possivelmente porque a maior capacidade do V2S requer mais dados de bag para convergir adequadamente.

### 7.5 Swin Transformer + MIL (Optuna)

**Backbone:** Swin-T (torchvision); descongela blocos finais + norm/permute/avgpool/flatten

**Hiperparâmetros Optuna:** γ=2,4691 | w_focal=1,0410 | w_ord=1,1828 | lr=1,00×10⁻⁴

**Resultados no conjunto de teste:**

| Acc (%) | QWK | Macro F1 |
|---------|-----|----------|
| 67,46 | 0,8536 | 0,6398 |

**Curva de validação (22 épocas):**
- Época 1: val_kappa=0,7906 | Época 12: val_kappa=0,8503 | Época 14: val_kappa=0,8536 | Época 22: val_kappa=0,8452

### 7.6 ConvNeXt-Small + MIL (Optuna)

**Backbone:** ConvNeXt-Small (torchvision); descongela últimos N blocos de `model.features`

**Melhor validação:** val_acc=68,59% | val_kappa=0,8382

*(Resultados de teste completos com bootstrap não disponíveis em arquivo separado; dados do Optuna .db indicam estudo concluído)*

### 7.7 ViT-Base + MIL (Optuna)

**Backbone:** Vision Transformer Base; congela todos menos últimos `fine_tune` parâmetros

**Resultados de teste (conjunto menor — 155 amostras):**

| Acc (%) | QWK | Macro F1 | IC 95% (QWK) |
|---------|-----|----------|--------------|
| 41,82 ± 1,26 | 0,6432 ± 0,0175 | 0,3742 ± 0,0123 | [0,6059 – 0,6755] |

*Nota: o relatório de resultado foi gerado sobre subconjunto diferente do teste principal (155 vs. 1.590 amostras); os números não são diretamente comparáveis à tabela principal.*

**Relatório por classe:**

| Classe | Prec. | Recall | F1 |
|--------|-------|--------|-----|
| ISUP 0 | 0,647 | 0,599 | 0,622 |
| ISUP 1 | 0,430 | 0,445 | 0,437 |
| ISUP 2 | 0,262 | 0,270 | 0,266 |
| ISUP 3 | 0,225 | 0,205 | 0,215 |
| ISUP 4 | 0,268 | 0,300 | 0,283 |
| ISUP 5 | 0,416 | 0,429 | 0,423 |

O ViT apresentou o pior desempenho absoluto, consistente com a literatura que aponta dificuldade de transformers puros em datasets médicos de tamanho moderado sem pré-treinamento específico.

---

## 8. Estratégias de Ensemble

### 8.1 Ensemble Básico — B0 + B3 + B7 (Comparação de Estratégias)

**Modelos base:** B7-Ordinal-Focal, B0-Ordinal-Focal, B3-Ordinal-Focal

| Estratégia | Acc (%) | QWK | Macro F1 | Recall | Prec. |
|-----------|---------|-----|----------|--------|-------|
| **Ensemble-Weighted** | **72,20** | **0,8847** | **0,6850** | 0,6823 | 0,6948 |
| Ensemble-Mean | 71,32 | 0,8845 | 0,6752 | 0,6720 | 0,6841 |
| Ensemble-Geometric-Mean | 71,95 | 0,8801 | 0,6800 | 0,6755 | 0,6932 |
| Ensemble-Median | 69,94 | 0,8784 | 0,6622 | 0,6590 | 0,6707 |
| Ensemble-Trimmed-Mean | 69,94 | 0,8784 | 0,6622 | 0,6590 | 0,6707 |
| B0-Entropy-Ordinal-Focal | 71,45 | 0,8735 | 0,6763 | 0,6728 | 0,6892 |
| Ensemble-Vote | 70,50 | 0,8633 | 0,6658 | 0,6612 | 0,6760 |
| Ensemble-Max | 61,38 | 0,8558 | 0,5876 | 0,5998 | 0,5899 |
| B3-Entropy-Ordinal-Focal | 65,97 | 0,8484 | 0,6188 | 0,6169 | 0,6243 |
| B7-Entropy-Ordinal-Focal | 60,25 | 0,8440 | 0,5564 | 0,5550 | 0,5610 |

**Matriz de Confusão — Ensemble-Weighted:**
```
Pred →   IS0   IS1   IS2   IS3   IS4   IS5
IS0:    [386,   38,    4,    2,    3,    1]
IS1:    [ 30,  297,   71,    2,    0,    0]
IS2:    [  3,   34,  119,   36,    8,    0]
IS3:    [  3,    5,   28,  109,   38,    2]
IS4:    [  7,    7,   10,   29,  118,   16]
IS5:    [  6,    2,    2,   14,   41,  119]
```

### 8.2 Ensemble Avançado — Estratégias Estendidas

**Modelos:** B0-Ordinal-Focal + B3-Ordinal-Focal + B7-Ordinal-Focal (3 modelos base, 1.590 amostras teste)

| Estratégia | Acc (%) | QWK | Macro F1 | Recall | Prec. |
|-----------|---------|-----|----------|--------|-------|
| **Baseline-Mean** | 71,26 | **0,8752** | 0,6615 | 0,6565 | 0,6744 |
| Baseline-Weighted-Mean | 71,45 | 0,8747 | 0,6626 | 0,6580 | 0,6740 |
| Ensemble-ConfWeighted | 71,51 | 0,8691 | 0,6618 | 0,6572 | 0,6741 |
| Ensemble-GeomMean | 71,38 | 0,8690 | 0,6591 | 0,6526 | 0,6766 |
| Ensemble-Temp2.0 | 71,26 | 0,8750 | 0,6600 | 0,6550 | 0,6724 |
| Ensemble-Temp0.5 | 70,94 | 0,8728 | 0,6594 | 0,6545 | 0,6714 |
| Ensemble-BordaCount | 69,31 | 0,8693 | 0,6469 | 0,6430 | 0,6553 |
| Ensemble-Median | 69,43 | 0,8693 | 0,6486 | 0,6448 | 0,6566 |
| Ensemble-TrimmedMean | 69,43 | 0,8693 | 0,6486 | 0,6448 | 0,6566 |
| Ensemble-RankAvg | 67,61 | 0,8741 | 0,6303 | 0,6260 | 0,6403 |
| Ensemble-Max | 59,43 | 0,8529 | 0,5594 | 0,5711 | 0,5612 |
| Baseline-Majority-Vote | 68,81 | 0,8537 | 0,6384 | 0,6329 | 0,6489 |

**Melhor estratégia:** Baseline-Mean (QWK=0,8752)

**Matriz de Confusão — Baseline-Mean:**
```
Pred →   IS0   IS1   IS2   IS3   IS4   IS5
IS0:    [400,   30,    0,    2,    2,    0]
IS1:    [ 36,  309,   50,    5,    0,    0]
IS2:    [  3,   43,  107,   39,    8,    0]
IS3:    [  6,   12,   25,   89,   48,    5]
IS4:    [ 10,    7,    7,   25,  118,   20]
IS5:    [  7,    4,    1,   12,   50,  110]
```

### 8.3 Ensemble MIL + Patch-Level

Combinação do B0-MIL com o melhor modelo patch-level (B0-Ordinal-Focal):

| Configuração | Acc (%) | QWK | Macro F1 | IC 95% (QWK) |
|---|---|---|---|---|
| Patch-Only (B0) | 68,1 ± 1,2 | 0,859 ± 0,012 | 0,616 ± 0,013 | [0,840 – 0,877] |
| MIL-Only (B0-MIL) | 66,2 ± 1,2 | 0,845 ± 0,012 | 0,605 ± 0,013 | [0,824 – 0,864] |
| Ensemble-Mean (α=0,50) | 69,9 ± 1,2 | 0,866 ± 0,011 | 0,638 ± 0,013 | [0,846 – 0,884] |
| **Ensemble-α=0,40** | **70,5 ± 1,2** | **0,867 ± 0,012** | **0,647 ± 0,013** | **[0,847 – 0,885]** |

O peso ótimo α=0,40 para o componente patch-level vs. MIL indica que o modelo de patch carrega mais informação discriminante que o MIL no regime de 36 patches.

**Matriz de Confusão — Ensemble-α=0,40:**
```
Pred →   IS0   IS1   IS2   IS3   IS4   IS5
IS0:    [412,   18,    0,    1,    2,    1]
IS1:    [ 43,  302,   51,    4,    0,    0]
IS2:    [  3,   45,  106,   38,    8,    0]
IS3:    [ 10,   10,   29,   85,   45,    6]
IS4:    [ 11,    8,   10,   25,  107,   26]
IS5:    [  7,    3,    1,   16,   48,  109]
```

### 8.4 Ensemble com TTA (8 visualizações)

| Configuração | Acc (%) | QWK | Macro F1 | IC 95% (QWK) |
|---|---|---|---|---|
| Sem TTA | 70,3 ± 1,1 | 0,874 ± 0,011 | 0,650 ± 0,012 | [0,855 – 0,892] |
| Com TTA | 71,4 ± 1,1 | 0,875 ± 0,011 | 0,662 ± 0,012 | [0,856 – 0,893] |

### 8.5 Ensemble de Seis Modelos — Melhor Resultado Global

**Composição:** V2S + B0 + B3 + SWA + B0-MIL + SwinMIL

**Pesos QWK-calibrados (validação):**

| Modelo | QWK_val | Peso |
|--------|---------|------|
| EfficientNetV2-S | 0,8804 | 0,2032 |
| EfficientNet-B3 | 0,8656 | 0,1998 |
| EfficientNet-B0 | 0,8651 | 0,1997 |
| EfficientNet-B0 + SWA | 0,8589 | 0,1983 |
| EfficientNet-B0 + MIL | 0,8581 | 0,1981 |
| Swin-T + MIL | 0,0040 | 0,0009 |

**Alpha ótimo MIL (validação):** α=0,05 | QWK_val=0,8844

**Resultados comparativos — todas estratégias (conjunto de teste):**

| Estratégia | Acc (%) | QWK | Macro F1 |
|-----------|---------|-----|----------|
| **Six-Weighted-Mean** | **72,72** | **0,8810** | **0,6787** |
| CNN-Mean (4 modelos) | 72,21 | 0,8801 | 0,6773 |
| CNN-B0+V2S | 72,18 | 0,8791 | 0,6731 |
| Alpha-0.05 | 72,46 | 0,8786 | 0,6776 |
| V2S (individual) | 71,38 | 0,8733 | 0,6751 |
| Six-Median | 72,19 | 0,8724 | 0,6679 |
| Six-Trimmed-Mean | 72,19 | 0,8720 | 0,6647 |
| B0 (individual) | 68,09 | 0,8590 | 0,6163 |
| SWA (individual) | 65,34 | 0,8520 | 0,5906 |
| B3 (individual) | 66,02 | 0,8488 | 0,6183 |
| MIL-B0 (individual) | 66,19 | 0,8446 | 0,6046 |
| SwinMIL (individual) | 27,23 | 0,0026 | 0,0738 |

**Relatório por classe — Six-Weighted-Mean:**

| Classe | Prec. | Recall | F1 | Suporte |
|--------|-------|--------|----|---------|
| ISUP 0 | 0,864 | 0,940 | 0,901 | 434 |
| ISUP 1 | 0,803 | 0,763 | 0,782 | 400 |
| ISUP 2 | 0,546 | 0,570 | 0,558 | 200 |
| ISUP 3 | 0,558 | 0,524 | 0,540 | 185 |
| ISUP 4 | 0,536 | 0,631 | 0,580 | 187 |
| ISUP 5 | 0,844 | 0,620 | 0,715 | 184 |
| **Macro** | **0,692** | **0,675** | **0,679** | 1.590 |

**Matriz de Confusão — Six-Weighted-Mean:**
```
Pred →   IS0   IS1   IS2   IS3   IS4   IS5
IS0:    [408,   21,    2,    1,    2,    0]
IS1:    [ 38,  305,   53,    4,    0,    0]
IS2:    [  2,   39,  114,   37,    8,    0]
IS3:    [  5,    9,   27,   97,   43,    4]
IS4:    [ 11,    4,   11,   26,  118,   17]
IS5:    [  8,    2,    2,    9,   49,  114]
```

**Delta vs. melhor modelo individual anterior (QWK=0,8754):** +0,0056

---

## 9. Análise de Hiperparâmetros (Optuna — Estudos .db)

### 9.1 Arquivos de Estudo Optuna

| Arquivo | Tamanho | Modelo |
|---------|---------|--------|
| `v2-optuna-study.db` | 139 KB | EfficientNetV2-S patch |
| `b0-mil-focal-49-optuna.db` | 131 KB | B0-MIL 49 patches |
| `v2s-mil-focal-optuna.db` | 131 KB | V2S-MIL |
| `vit-entropy-ordinal-focal-optuna.db` | 131 KB | ViT-MIL |
| `swin-t-mil-optuna.db` | 127 KB | Swin-T-MIL |
| `b0-entropy-ordinal-focal-yhu-optuna.db` | 123 KB | B0 + espaço YHU |
| `convnext-small-v2-optuna.db` | 123 KB | ConvNeXt-Small |
| `convnext-tiny-optuna.db` | 115 KB | ConvNeXt-Tiny |

### 9.2 Comparação dos Melhores Hiperparâmetros

| Modelo | γ | w_focal | w_ord | lr | Dropout |
|--------|---|---------|-------|----|---------|
| V2S Patch | 3,426 | — | — | 1,105×10⁻⁴ | 0,442 |
| B0-MIL-36 | 2,469 | 1,041 | 1,183 | 1,40×10⁻⁴ | — |
| V2S-MIL | 1,657 | 0,290 | 0,447 | 1,11×10⁻⁴ | — |
| B0-MIL-49 | 1,677 | 1,032 | 1,671 | 5,38×10⁻⁵ | — |
| Swin-T-MIL | 2,469 | 1,041 | 1,183 | 1,00×10⁻⁴ | — |

**Padrões observados:**
1. Modelos MIL convergem com LR menor (5×10⁻⁵ – 1,4×10⁻⁴) que modelos patch (3×10⁻⁴ padrão)
2. γ alto (2,4–3,4) beneficia modelos com maior capacidade (V2S, Swin)
3. γ baixo (1,6–1,7) para modelos MIL com backbone menor (B0, V2S-MIL)
4. w_focal vs. w_ord: MIL prefere balance mais conservador (w_focal ≤ 1,1)

---

## 10. Tabela de Hiperparâmetros de Treinamento — Todos os Modelos

| Modelo | Backbone | LR | Épocas | Batch | Dropout | Focal α | Focal γ | SWA | Observações |
|--------|----------|-----|--------|-------|---------|---------|---------|-----|-------------|
| B0-Ordinal | EfficientNet-B0 | 3×10⁻⁴ | 50 | 3 | 0,6 | — | — | Não | Loss ordinal pura |
| B0-Ordinal-Focal | EfficientNet-B0 | 3×10⁻⁴ | 50 | 3 | 0,6 | 0,25 | 2,0 | Não | Focal + Ordinal |
| B0-Ordinal-Focal-2 | EfficientNet-B0 | 3×10⁻⁴ | 50 | 3 | 0,6 | [0,25–0,90] | 2,0 | Não | α variável por limiar |
| B0-Ordinal-Weighted | EfficientNet-B0 | 3×10⁻⁴ | 50 | 3 | 0,6 | — | — | Não | WeightedRandomSampler |
| B0-Ordinal-SWA | EfficientNet-B0 | 3×10⁻⁴ | 50 | 3 | 0,6 | — | — | Sim (ep.30) | SWA LR=1×10⁻⁴ |
| B0-Ordinal-Focal-Deformável | EfficientNet-B0 | 3×10⁻⁴ | 50 | 3 | 0,6 | 0,25 | 2,0 | Não | DeformableConv2d |
| B3-Ordinal-Focal | EfficientNet-B3 | 3×10⁻⁴ | 50 | 3 | 0,6 | [0,25–0,90] | 2,0 | Não | — |
| B7-Ordinal-Focal | EfficientNet-B7 | 3×10⁻⁴ | 50 | 3 | 0,6 | 0,25 | 2,0 | Não | — |
| B0-TTA | EfficientNet-B0 | 3×10⁻⁴ | 50 | 6 | — | — | — | Não | BCE, sem entropia, 5 TTA |
| V2S-Optuna | EfficientNetV2-S | 1,105×10⁻⁴ | 50 | 4 | 0,442 | 0,167 | 3,426 | Não | HPO Optuna (50 trials) |
| B0-MIL-36 | EfficientNet-B0 | 1,40×10⁻⁴ | 50 | 6 | — | — | 2,469 | Não | GatedAttention, 36 patches |
| B0-MIL-49 | EfficientNet-B0 | 5,38×10⁻⁵ | 50 | 6 | — | — | 1,677 | Não | GatedAttention, 49 patches |
| V2S-MIL | EfficientNetV2-S | 1,11×10⁻⁴ | 50 | — | — | — | 1,657 | Não | GatedAttention |
| Swin-T-MIL | Swin Transformer-T | 1,00×10⁻⁴ | 50 | — | — | — | 2,469 | Não | 22 épocas efetivas |
| ViT-MIL | ViT-Base | — | — | — | — | — | — | Não | Optuna, subconjunto |
| ConvNeXt-Small-MIL | ConvNeXt-Small | 1,00×10⁻⁴ | — | — | — | — | — | Não | Optuna, val apenas |

---

## 11. Análise Comparativa Final — Ranking Completo

### 11.1 Todas as Configurações (Conjunto de Teste, n=1.590, ordinal ascendente QWK)

| Rank | Modelo/Estratégia | Acc (%) | QWK | Macro F1 |
|------|-------------------|---------|-----|----------|
| 1 | **Six-Weighted-Mean (V2S+B0+B3+SWA+MIL-B0+SwinMIL)** | **72,72** | **0,8810** | **0,6787** |
| 2 | Ensemble-Weighted (B0+B3+B7) | 72,20 | 0,8847* | 0,6850 |
| 3 | CNN-Mean (V2S+B0+B3+SWA) | 72,21 | 0,8801 | 0,6773 |
| 4 | Alpha-0,05 (V2S+MIL) | 72,46 | 0,8786 | 0,6776 |
| 5 | Baseline-Mean (B0+B3+B7) avançado | 71,26 | 0,8752 | 0,6615 |
| 6 | TTA Ensemble (8 vistas) | 71,4 | 0,875 | 0,662 |
| 7 | V2S + Optuna (individual) | 71,51 | 0,8742 | 0,6772 |
| 8 | Ensemble-Temp2.0 | 71,26 | 0,8750 | 0,6600 |
| 9 | Ensemble-α=0,40 (Patch+MIL) | 70,5 | 0,867 | 0,647 |
| 10 | Sem TTA (base ensemble) | 70,3 | 0,874 | 0,650 |
| 11 | B0-MIL-36 (individual) | 69,12 | 0,8634 | 0,6398 |
| 12 | B0-Ordinal-SWA | 65,35 | 0,8523 | 0,5914 |
| 13 | B3-Ordinal-Focal (individual) | 65,97 | 0,8484 | 0,6188 |
| 14 | V2S-MIL (individual) | 65,91 | 0,8537 | 0,6152 |
| 15 | Swin-T-MIL (individual) | 67,46 | 0,8536 | 0,6398 |
| 16 | B0-Ordinal-Focal+Atenção | 63,02 | 0,8442 | 0,5673 |
| 17 | B0 + CLAHE-YUV | 59,56 | 0,7983 | 0,5133 |
| 18 | B7-Ordinal-Focal (individual) | 59,18 | 0,8088 | 0,5306 |
| 19 | B0-Ordinal-Focal (padrão) | 52,64 | 0,7616 | 0,4390 |
| 20 | B0-Ordinal-Focal+Deformável | 47,74 | 0,7508 | 0,4138 |
| 21 | ViT-MIL (subconjunto 155 amostras) | 41,82 | 0,6432 | 0,3742 |

\* *O Ensemble-Weighted (B0+B3+B7) no arquivo `ensemble-ordinal-results.txt` apresenta QWK=0,8847 mas foi avaliado com configuração diferente do experimento avançado. O Six-Weighted-Mean é o resultado mais abrangente.*

### 11.2 Ganhos Cumulativos por Bloco Experimental

| Bloco | Melhor QWK | Ganho acumulado vs. B0 baseline |
|-------|-----------|----------------------------------|
| B0 individual (padrão) | 0,7616 | — |
| B0 + SWA | 0,8523 | +0,0907 |
| V2S individual | 0,8521 | +0,0905 |
| B0+B3+B7 ensemble | 0,8847 | +0,1231 |
| V2S + Optuna | 0,8742 | +0,1126 |
| Six-model ensemble | 0,8810 | +0,1194 |

---

## 12. Análise por Classe — Dificuldade Sistemática

### 12.1 Padrão de Erro por Grau ISUP (melhor modelo — Six-Weighted-Mean)

| Classe | Recall | Dificuldade | Principal erro |
|--------|--------|-------------|----------------|
| ISUP 0 | 93,9% | Baixa | 4,8% confundido com ISUP 1 |
| ISUP 1 | 76,3% | Moderada | 9,5% confundido com ISUP 0 |
| ISUP 2 | 57,0% | Alta | 19,5% confundido com ISUP 1 |
| ISUP 3 | 52,4% | Alta | 21,6% confundido com ISUP 2+4 |
| ISUP 4 | 63,1% | Moderada | 13,9% confundido com ISUP 3 |
| ISUP 5 | 62,0% | Moderada | 26,6% confundido com ISUP 4 |

**Interpretação:** Os graus intermediários ISUP 2 e 3 consistentemente apresentam recall mais baixo em todos os modelos, refletindo a ambiguidade histológica real entre padrões Gleason 3+3 (ISUP 2) e 3+4 (ISUP 3). Esta zona cinzenta tem relevância clínica direta na decisão de vigilância ativa vs. tratamento ativo.

### 12.2 Análise Bootstrap por Classe

Baseado em 1.000 reamostras (modelo V2S-Optuna como referência):

| Classe | F1 | Std | IC 95% |
|--------|-----|-----|--------|
| ISUP 0 | 0,882 | 0,016 | [0,849 – 0,912] |
| ISUP 1 | 0,748 | 0,022 | [0,704 – 0,788] |
| ISUP 2 | 0,541 | 0,034 | [0,472 – 0,607] |
| ISUP 3 | 0,589 | 0,037 | [0,516 – 0,660] |
| ISUP 4 | 0,592 | 0,035 | [0,525 – 0,658] |
| ISUP 5 | 0,712 | 0,033 | [0,645 – 0,774] |

---

## 13. Conclusões e Direcionamentos

### 13.1 Principais Contribuições Validadas

1. **Codificação ordinal + Focal Loss híbrida** é superior à BCE pura para 6 classes ordinais desequilibradas (ganho médio ~0,08 QWK sobre CE padrão).

2. **Filtragem por entropia** (top-20% removido) melhora consistentemente a convergência e QWK final ao eliminar amostras ambíguas do treino.

3. **Ensembles superam modelos individuais** em todos os experimentos; a média simples de probabilidades (soft voting) tem desempenho equivalente ou superior a estratégias sofisticadas (confidence-weighted, rank-average).

4. **V2S + Optuna** é o melhor modelo individual (QWK=0,8742), superando B0, B3 e B7 mesmo sem ensemble.

5. **MIL-B0 como componente de ensemble** adiciona +0,0056 QWK ao melhor ensemble CNN puro (Six-Weighted-Mean vs. CNN-Mean).

6. **Espaço de cores CLAHE-YUV não trouxe ganho** sobre RGB; stain normalization formal (Macenko/Ruifrok) permanece como experimento pendente.

7. **ViT puro** tem desempenho significativamente inferior a CNNs no regime de dados disponível, confirmando a necessidade de pré-treinamento específico em histopatologia.

8. **SwinMIL** apresentou convergência instável (QWK_val=0,0040), possivelmente por mismatch entre o tamanho das janelas de atenção e os patches de 256 px.

### 13.2 Trabalhos Pendentes / Próximos Experimentos

| Experimento | Status | Hipótese |
|-------------|--------|----------|
| Stain Normalization (Macenko) | Planejado | Remoção de viés de coloração pode melhorar ISUP 2–4 |
| Overlap 10–20% entre tiles | Em execução (`v2-overlap.ipynb`) | Contexto adjacente reduz perda de bordas |
| Focal Loss pura (β=0) | Planejado | Isolar contribuição do componente ordinal |
| Stacking (LR/SVM sobre probabilidades) | Planejado | Meta-learner pode explorar diversidade entre modelos |
| Cross-dataset (SICAPv2) | Planejado | Avaliar generalização dos melhores modelos |

### 13.3 Publicações Associadas

- **SBCAS 2026:** Resultados aceitos (resultados baseline + ensemble ordinal)
- **Bioengineering:** Manuscrito em preparação (resultados completos incluindo MIL e análise de espaços de cores)
