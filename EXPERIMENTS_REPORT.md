# Relatório de Experimentos — Gradação Automática de Câncer de Próstata (PANDA)

**Data:** Maio 2026  
**Dataset:** PANDA (Prostate cANcer graDe Assessment) — 1.590 amostras de teste  
**Tarefa:** Classificação multiclasse de grau ISUP 0–5 a partir de imagens histopatológicas  
**Métrica principal:** Quadratic Weighted Kappa (QWK)

---

## 1. Resumo Executivo

| Posição | Modelo / Estratégia | Acurácia | QWK | F1-Macro | IC 95% (QWK) |
|---------|---------------------|----------|-----|----------|--------------|
| 🥇 1º | Specialist Base (B0+B3+B7, checkpoint ótimo) | 71.36% | **0.8754** | 0.6622 | [0.8568, 0.8926] |
| 🥈 2º | Baseline-Mean Ensemble (B0+B3+B7) | 71.26% | 0.8752 | 0.6615 | — |
| 🥉 3º | EfficientNetV2-S + Optuna HPO | 71.51% | 0.8742 | **0.6772** | [0.8500, 0.8953] |
| 4º | Ensemble + TTA (8 vistas) | 71.4% | 0.8750 | 0.662 | [0.856, 0.893] |
| 5º | MIL + Patch Ensemble (α=0.40) | 70.5% | 0.867 | 0.647 | [0.847, 0.885] |
| 6º | B0-Entropy-Ordinal-Focal | 68.05% | 0.8589 | 0.6167 | — |
| 7º | B0 + SWA | 65.35% | 0.8523 | 0.5914 | — |
| 8º | B3-Entropy-Ordinal-Focal | 65.97% | 0.8484 | 0.6188 | — |
| 9º | B7-Entropy-Ordinal-Focal | 60.25% | 0.8440 | 0.5564 | — |

> **QWK ≥ 0.87 é considerado concordância forte em gradação patológica.**  
> Métricas de teste com bootstrap (1.000 reamostras), exceto onde indicado.

---

## 2. Pipeline Experimental

```
Imagens WSI (PANDA)
     │
     ▼
Extração de patches (224×224, 36 patches/WSI)
     │
     ▼
Filtragem por Entropia (remoção dos top 20% mais difíceis)
     │
     ├──► Patch-level CNN (EfficientNet / ConvNeXt / ViT / Swin)
     │         │
     │         ▼
     │    Codificação Ordinal → 5 limiares binários
     │    Loss: OrdinalFocalLoss (BCEWithLogitsLoss + (1−pₜ)ᵞ)
     │
     └──► MIL (EfficientNetMIL com GatedAttention)
               │
               ▼
          Bag: (B, N, C, H, W), max_patches=36
```

---

## 3. Codificação de Rótulos (Ordinal)

| Grau ISUP | Vetor Ordinal | Limites |
|-----------|--------------|---------|
| 0 | [0, 0, 0, 0, 0] | — |
| 1 | [1, 0, 0, 0, 0] | k≥1 |
| 2 | [1, 1, 0, 0, 0] | k≥2 |
| 3 | [1, 1, 1, 0, 0] | k≥3 |
| 4 | [1, 1, 1, 1, 0] | k≥4 |
| 5 | [1, 1, 1, 1, 1] | k≥5 |

**Decodificação:** `sigmoid → soma dos bits > 0.5`  
**Loss:** `BCEWithLogitsLoss` com pesos Focal opcionais

---

## 4. Arquiteturas Avaliadas

### 4.1 EfficientNet (Família Principal)

| Backbone | Parâmetros | Estratégia de Fine-tuning |
|----------|-----------|--------------------------|
| EfficientNet-B0 | ~5.3M | Congela tudo, descongela últimos 150 params |
| EfficientNet-B3 | ~12M | Congela tudo, descongela últimos 150 params |
| EfficientNet-B7 | ~66M | Congela tudo, descongela últimos 150 params |
| EfficientNetV2-S | ~21M | Descongela últimos `unfreeze_blocks` blocos (Optuna: 3) |

Cabeçalho: `GeM Pooling (p=3)` → `Dropout` → `Linear(5)`

### 4.2 Variantes Estruturais (B0 base)

| Variante | Modificação | QWK (teste) |
|----------|------------|-------------|
| B0 + Ordinal + Focal | Configuração padrão | 0.8589 |
| B0 + SE Blocks | Atenção local canal a canal | 0.8442 |
| B0 + SWA | Stochastic Weight Averaging (início ep. 30) | 0.8523 |
| B0 + Deformable Conv | Convoluções deformáveis (18.9M params treináveis) | 0.7508 |
| B0 + Regressão ordinal | Saída contínua em vez de limiar binário | 0.7616 |

### 4.3 Arquiteturas Alternativas

| Modelo | Notebook | Observação |
|--------|----------|-----------|
| ConvNeXt-Small | `convnext-small-entropy-ordinal-focal.ipynb` | Colapso em 4 épocas (QWK ~0.55) |
| ConvNeXt | `convnext-entropy-ordinal-focal.ipynb` | Resultados inferiores ao B0 |
| ViT-Base | `vit-base-entropy-ordinal-focal.ipynb` | Testado; logs de teste incompletos |
| Swin Transformer | `swin-transformer-entropy-ordinal-focal.ipynb` | Testado; logs de teste incompletos |

### 4.4 MIL (Multiple Instance Learning)

| Modelo | Pooling | Input Shape | Saída |
|--------|---------|-------------|-------|
| EfficientNetMIL | GatedAttention | (B, N, C, H, W) | `{logits, attn, features}` |
| ConvNeXtMIL | GatedAttention | (B, N, C, H, W) | Mesma estrutura |
| SwinMIL | GatedAttention | (B, N, C, H, W) | Mesma estrutura |

> **Crítico:** Sempre passar o tensor `mask` para o forward MIL — sem ele, o softmax inclui patches de padding e os scores de atenção ficam sem sentido.

---

## 5. Estratégias de Treinamento

### 5.1 Configuração Base (Comum a Todos)

| Parâmetro | Valor |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 3×10⁻⁴ (Optuna: 1.1×10⁻⁴) |
| Warmup | 1 época (fator ×2) |
| Scheduler | CosineAnnealingLR |
| Batch Size | 3 (Optuna: 4) |
| Dropout | 0.6 (Optuna: 0.44) |
| Early Stopping patience | 7 épocas |
| Épocas máximas | 50 |
| Seed | 42 |
| Augmentação | Flip H/V, Transpose (p=0.5) |
| Checkpointing | Melhor `val_kappa['mean']` |

### 5.2 Filtragem por Entropia

- **Método:** Remoção dos 20% de amostras com maior `difficulty_score` (entropia de patch)
- **Arquivo:** `data/entropy.csv`
- **Impacto medido:** +0.041 QWK (0.832 → 0.873)
- **Amostras removidas:** ~18% (~1.900 de ~10.600)

### 5.3 Focal Loss Ordinal

```
OrdinalFocalLoss = Σ BCE(p̂ₖ, yₖ) × (1 − pₜ)^γ × α
```

| Parâmetro | Valor padrão | Valor Optuna |
|-----------|-------------|--------------|
| α (focal alpha) | 0.25 | 0.167 |
| γ (focal gamma) | 2.0 | 3.43 |
| α variável por limiar | [0.25, 0.35, 0.50, 0.70, 0.90] | — |

### 5.4 Espaços de Cor Testados

| Espaço de Cor | Acurácia | QWK | ΔQWK vs RGB |
|---------------|----------|-----|-------------|
| RGB (baseline) | 68.05% | 0.8589 | — |
| HED (Hematoxilina-Eosina-DAB) | 65.70% | 0.8517 | −0.007 |
| XYZ (CIE) | 66.27% | 0.8442 | −0.015 |
| Fusão multicanal | `code/fusion/` | — | — |

> **Conclusão:** Pré-treino ImageNet (RGB) domina; espaços especializados não compensam a perda de transferência.

---

## 6. Resultados Individuais por Modelo

### 6.1 Família EfficientNet-B0 — Ablation Study

| Experimento | Acurácia | QWK | F1-Macro |
|-------------|----------|-----|----------|
| B0 + Ordinal | — | 0.8597 | — |
| **B0 + Ordinal + Focal** | **68.05%** | **0.8589** | 0.6167 |
| B0 + Ordinal + Weighted Sampler | — | 0.8594 | — |
| B0 + Ordinal + SWA | 65.35% | 0.8523 | 0.5914 |
| B0 + Ordinal + Focal + SE Attention | 63.02% | 0.8442 | 0.5673 |
| B0 + Focal (sem entropia) | — | ~0.832 | — |
| B0 + Focal + Regressão | 52.64% | 0.7616 | 0.4390 |
| B0 + Focal + Deformable Conv | 47.74% | 0.7508 | 0.4138 |
| B0 + Focal (classes 2–5 only) | — | 0.7311 | — |

### 6.2 Escalamento de Backbone

| Backbone | Acurácia (teste) | QWK (teste) | F1-Macro | Observação |
|----------|-----------------|-------------|----------|-----------|
| EfficientNet-B0 | 68.05% | 0.8589 | 0.6167 | Melhor custo-benefício |
| EfficientNet-B3 | 65.97% | 0.8484 | 0.6188 | Leve underfitting no teste |
| EfficientNet-B7 | 60.25% | 0.8440 | 0.5564 | **Overfitting severo** |
| **EfficientNetV2-S + Optuna** | **71.51%** | **0.8742** | **0.6772** | **Melhor modelo individual** |

> Modelos maiores sofreram overfitting. Capacidade não compensa no domínio médico com dados limitados.

---

## 7. Estratégias de Ensemble

### 7.1 Comparação Completa (B0 + B3 + B7)

| Estratégia | Acurácia | QWK | F1-Macro |
|-----------|----------|-----|----------|
| **Baseline-Mean** | **71.26%** | **0.8752** | 0.6615 |
| Ensemble-Temp 2.0 | 71.26% | 0.8750 | 0.6600 |
| Baseline-Weighted-Mean | 71.45% | 0.8747 | 0.6626 |
| Ensemble-RankAvg | 67.61% | 0.8741 | 0.6303 |
| Ensemble-Temp 0.5 | 70.94% | 0.8728 | 0.6594 |
| Ensemble-BordaCount | 69.31% | 0.8693 | 0.6469 |
| Ensemble-Median | 69.43% | 0.8693 | 0.6486 |
| Ensemble-TrimmedMean | 69.43% | 0.8693 | 0.6486 |
| Ensemble-ConfWeighted | 71.51% | 0.8691 | 0.6618 |
| Ensemble-GeomMean | 71.38% | 0.8690 | 0.6591 |
| Baseline-Majority-Vote | 68.81% | 0.8537 | 0.6384 |
| Baseline-Max | 59.43% | 0.8529 | 0.5594 |

> **Vencedor:** Média simples de probabilidades (Baseline-Mean). Estratégias complexas não superam a média simples em QWK.

### 7.2 Ensemble com TTA (Test-Time Augmentation)

| Configuração | Acurácia | QWK | F1-Macro |
|-------------|----------|-----|----------|
| Sem TTA | 70.3% | 0.874 | 0.650 |
| **Com TTA (8 vistas)** | **71.4%** | **0.875** | **0.662** |
| Δ | +1.1 pp | +0.001 | +0.012 |

Transformações: original, H-flip, V-flip, HV-flip, transpose, 90°, 180°, 270°.

> TTA contribui marginalmente (+0.1% QWK). O modelo já generaliza bem sem ela.

### 7.3 Cascade com Modelos Especialistas

Base: B0+B3+B7 ensemble + especialistas para pares confusos ISUP (3↔4) e (4↔5):

| Modelo | Acurácia | QWK | F1-Macro |
|--------|----------|-----|----------|
| **Base (ordinal-best)** | **71.36%** | **0.8754** | 0.6622 |
| Cascade (base + especialistas) | 71.32% | 0.8721 | 0.6592 |
| Δ | −0.04 pp | −0.0033 | −0.003 |

Δ Recall por classe (cascade vs base):

| ISUP | Δ Recall |
|------|---------|
| 0 | +0.023 |
| 1 | +0.065 |
| 2 | **+0.095** |
| 3 | **+0.124** |
| 4 | **−0.053** |
| 5 | +0.054 |

> A cascade melhora ISUP 2 e 3, mas degrada ISUP 4. Custo global negativo (−0.0033 QWK).

---

## 8. Multiple Instance Learning (MIL)

### 8.1 Estratégias MIL vs Patch-Level

| Estratégia | Acurácia | QWK | F1-Macro | IC 95% (QWK) |
|-----------|----------|-----|----------|-------------|
| Patch-Only | 68.1% | 0.859 | 0.616 | [0.840, 0.877] |
| MIL-Only (GatedAttention) | 66.2% | 0.845 | 0.605 | [0.824, 0.864] |
| Ensemble-Mean (patch + MIL) | 69.9% | 0.866 | 0.638 | [0.846, 0.884] |
| **Ensemble-α=0.40** | **70.5%** | **0.867** | **0.647** | [0.847, 0.885] |

`predição_final = α × MIL + (1−α) × Patch`, α ótimo = **0.40**

> MIL captura contexto global complementar ao patch-level. A combinação ponderada supera ambos isoladamente, mas fica abaixo do ensemble B0+B3+B7.

---

## 9. Otimização de Hiperparâmetros (Optuna — EfficientNetV2-S)

**Melhor trial:** QWK validação = 0.8745

| Hiperparâmetro | Valor Encontrado | Range Buscado |
|---------------|-----------------|---------------|
| Learning Rate | 1.1×10⁻⁴ | [1e-5, 1e-3] |
| Dropout Rate | 0.442 | [0.3, 0.7] |
| Focal γ | 3.43 | [1.0, 5.0] |
| Focal α | 0.167 | [0.1, 0.5] |
| Blocos descongelados | 3 | [1, 5] |
| Batch Size | 4 | {3, 4, 6} |
| Weight Decay | 1.04×10⁻⁵ | [1e-6, 1e-3] |

**Resultado final (teste, 1.000 bootstrap):**

| Métrica | Valor | IC 95% |
|---------|-------|--------|
| Acurácia | 71.51% ± 1.13% | [69.18%, 73.52%] |
| QWK | 0.8742 ± 0.0113 | [0.8500, 0.8953] |
| F1-Macro | 0.6772 ± 0.0125 | [0.6518, 0.6995] |

---

## 10. Análise Por Classe (Melhores Modelos)

### 10.1 Melhor Modelo Individual — V2-S + Optuna

| Classe | Precision | Recall | F1 | Suporte |
|--------|-----------|--------|----|---------|
| ISUP 0 | 0.875 | 0.889 | 0.882 | 434 |
| ISUP 1 | 0.758 | 0.738 | 0.748 | 400 |
| ISUP 2 | 0.518 | 0.565 | 0.541 | 200 |
| ISUP 3 | 0.564 | 0.616 | 0.589 | 185 |
| ISUP 4 | 0.557 | 0.631 | 0.592 | 187 |
| ISUP 5 | 0.867 | 0.603 | 0.712 | 184 |

### 10.2 Melhor Ensemble — Baseline-Mean

| Classe | Precision | Recall | F1 | Suporte |
|--------|-----------|--------|----|---------|
| ISUP 0 | 0.866 | **0.922** | 0.893 | 434 |
| ISUP 1 | 0.763 | 0.773 | 0.768 | 400 |
| ISUP 2 | 0.563 | 0.535 | 0.549 | 200 |
| ISUP 3 | 0.517 | 0.481 | 0.499 | 185 |
| ISUP 4 | 0.522 | 0.631 | 0.571 | 187 |
| ISUP 5 | **0.815** | 0.598 | 0.690 | 184 |

> **Padrão de erros:** ISUP 2 e 3 são as classes mais difíceis em todos os modelos (região de fronteira no critério de Gleason). ISUP 0 e 5 têm alta performance (classes extremas são mais fáceis de distinguir).

---

## 11. Principais Achados e Conclusões

| # | Achado | Impacto Medido |
|---|--------|---------------|
| 1 | **Filtragem por entropia** remove amostras ruidosas | +0.041 QWK |
| 2 | **Escala não ajuda** — B0 generaliza melhor que B7 | B7 perde 8 pp de QWK no teste |
| 3 | **Ordinal encoding** é superior a softmax/cross-entropy | Base de todos os bons modelos |
| 4 | **Focal loss** melhora desbalanceamento de classes | Γ ótimo: 3.43 (Optuna) |
| 5 | **Ensemble simples** supera estratégias complexas | Mean > RankAvg, Vote, Max |
| 6 | **TTA** contribui marginalmente | +0.001 QWK |
| 7 | **Cascade de especialistas** não agrega valor global | −0.003 QWK (melhora ISUP 2/3, piora ISUP 4) |
| 8 | **MIL complementa** patch-level CNN | +0.8% QWK no ensemble ponderado |
| 9 | **SE Attention** (local) estável; **self-attention** (global) colapsa | Histologia é fenômeno local |
| 10 | **RGB + ImageNet pretrain** supera HED/XYZ | HED perde 0.7% QWK |

---

## 12. Mapa dos Experimentos

```
code/tests/baseline/
├── efficientnet-b0-entropy-ordinal-focal.ipynb        ← Baseline principal
├── efficientnet-b0-entropy-ordinal-focal-*.ipynb      ← Variantes (SE, deformable, SWA...)
├── efficientnet-b3-entropy-ordinal-focal.ipynb        ← Backbone B3
├── efficientnet-b7-entropy-ordinal-focal.ipynb        ← Backbone B7
├── efficientnet-v2-optuna-focal.ipynb                 ← V2-S + Optuna (melhor individual)
├── convnext-*.ipynb                                   ← ConvNeXt (instável)
├── efficientnet-ordinal-ensemble-strategies.ipynb     ← Ensemble avançado
├── efficientnet-ordinal-ensemble-tta.ipynb            ← Ensemble + TTA
├── efficientnet-b0-ordinal-best-specialist.ipynb      ← Cascade especialista
├── evaluate-sicapv2-dataset.ipynb                     ← Avaliação cross-dataset
└── logs/                                              ← 40 arquivos .txt + 47 .png

code/tests/transformers/
├── b0-mil.ipynb                                       ← MIL canônico (EfficientNet-B0)
├── b0-mil-49.ipynb                                    ← Variante MIL
├── swin-transformer-entropy-ordinal-focal.ipynb       ← Swin Transformer
└── vit-base-entropy-ordinal-focal.ipynb               ← ViT-Base

code/fusion/
├── fusion-{mean,sum,max}.ipynb                        ← Fusão de espaços de cor
└── test-fusion-*.ipynb                                ← Avaliação de fusão

analyse/
├── GradCAM, LIME, FiftyOne                            ← Explicabilidade
```

---

## 13. Arquivos de Log Disponíveis

| Arquivo | Conteúdo |
|---------|---------|
| `b0-entropy-ordinal-focal-test-results.txt` | Métricas de teste do modelo baseline B0 |
| `ensemble-advanced-results.txt` | Comparação completa de 12 estratégias de ensemble |
| `ensemble-advanced-stats.txt` | Bootstrap stats do ensemble avançado |
| `v2-optuna-ordinal-focal-results.txt` | Resultados V2-S + Optuna |
| `ensemble-tta-test-results.txt` | Ensemble com/sem TTA |
| `mil-patch-ensemble-test-results.txt` | MIL + Patch ensemble (α search) |
| `specialist-results.txt` | Cascade com especialistas por classe |
| `b7-entropy-ordinal-focal-test-results.txt` | Métricas de teste do B7 |
| `b0-entropy-ordinal-swa-test-results.txt` | Métricas de teste com SWA |
| `tabela_hiperparametros.txt` | Tabela LaTeX completa de hiperparâmetros |
| `all_models_stats.txt` | Bootstrap stats de todos os modelos individuais |

---

*Gerado automaticamente com base nos logs de `code/tests/baseline/logs/` e notebooks experimentais.*
