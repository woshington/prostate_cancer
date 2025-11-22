# Projeto de Classificação com EfficientNet e Ensemble

## 📋 Visão Geral

Este projeto implementa um sistema de classificação de imagens utilizando múltiplas variantes da arquitetura **EfficientNet** (B0, B1, B2 e B3) e técnicas de **ensemble learning** para melhorar a performance final.

## 🏗️ Arquitetura

### Modelos Treinados

Foram treinados quatro modelos distintos da família EfficientNet:

| Modelo | Parâmetros | Arquivo de Treinamento | Arquivo de Pesos |
|--------|-----------|------------------------|------------------|
| **EfficientNet-B0** | ~5.3M | `code/tests/efficientnet-b0.ipynb` | `code/tests/models/b0.pth` |
| **EfficientNet-B1** | ~7.8M | `code/tests/efficientnet-b1.ipynb` | `code/tests/models/b1.pth` |
| **EfficientNet-B2** | ~9.2M | `code/tests/efficientnet-b2.ipynb` | `code/tests/models/b2.pth` |
| **EfficientNet-B3** | ~12M | `code/tests/efficientnet-b3.ipynb` | `code/tests/models/b3.pth` |

### Características dos Modelos

Todos os modelos utilizam a classe `EfficientNetApi` (`utils/models.py:41-66`) com as seguintes características:

- **Transfer Learning**: Pesos pré-treinados do PyTorch
- **Fine-tuning seletivo**: Últimos 150 parâmetros treináveis
- **Dropout**: Taxa de 0.6 para regularização
- **Classificação**: 5 classes de saída
- **Loss Function**: BCEWithLogitsLoss

## 🎯 Processo de Treinamento

### Configuração de Treinamento

```python
- Seed: 42 (reprodutibilidade)
- Batch Size: 6
- Learning Rate inicial: 3e-4
- Épocas máximas: 50
- Early Stopping: 5 épocas sem melhoria
- Otimizador: Adam
- Scheduler: CosineAnnealingLR com Warmup (2x, 1 época)
```

### Data Augmentation

Augmentações aplicadas apenas no conjunto de treino:

- Transpose (p=0.5)
- Vertical Flip (p=0.5)
- Horizontal Flip (p=0.5)

### Validação Cruzada

- **Estratégia**: 5-fold cross-validation
- **Fold de validação**: Fold 3
- **Conjunto de treino**: 7,219 imagens
- **Conjunto de validação**: 1,805 imagens
- **Conjunto de teste**: 1,592 imagens

## 📊 Resultados dos Modelos Individuais

### EfficientNet-B0

**Melhor época**: 10/50

| Métrica | Train | Validation | Test |
|---------|-------|------------|------|
| **Loss** | 0.039 | 0.428 | - |
| **Accuracy** | - | 60.53% ± 1.16 | 59.28% ± 1.21 |
| **Kappa** | - | **0.832 ± 0.011** | **0.826 ± 0.012** |
| **F1-Score** | - | 0.552 ± 0.012 | 0.537 ± 0.013 |
| **Recall** | - | 0.555 ± 0.012 | 0.541 ± 0.013 |
| **Precision** | - | 0.564 ± 0.012 | 0.551 ± 0.013 |

**Observações**:
- Early stopping acionado na época 15
- Melhor QWK (Quadratic Weighted Kappa): 0.832
- Convergência estável com generalização consistente

### EfficientNet-B1

**Melhor época**: 9/50

| Métrica | Train | Validation | Test |
|---------|-------|------------|------|
| **Loss** | 0.040 | 0.403 | - |
| **Accuracy** | - | 63.38% ± 1.14 | 63.80% ± 1.16 |
| **Kappa** | - | **0.839 ± 0.011** | **0.834 ± 0.013** |
| **F1-Score** | - | 0.575 ± 0.012 | 0.572 ± 0.013 |
| **Recall** | - | 0.572 ± 0.012 | 0.569 ± 0.013 |
| **Precision** | - | 0.586 ± 0.013 | 0.584 ± 0.013 |

**Observações**:
- Early stopping acionado na época 14
- Melhor QWK (Quadratic Weighted Kappa): 0.839
- **Melhor modelo individual do projeto** com performance superior ao B0
- Excelente generalização: teste (0.834) próximo da validação (0.839)

### EfficientNet-B2

**Melhor época**: 2/50

| Métrica | Train | Validation | Test |
|---------|-------|------------|------|
| **Loss** | 0.294 | 0.283 | - |
| **Accuracy** | - | 54.04% ± 1.17 | 52.45% ± 1.27 |
| **Kappa** | - | **0.775 ± 0.012** | **0.779 ± 0.013** |
| **F1-Score** | - | 0.453 ± 0.012 | 0.434 ± 0.012 |
| **Recall** | - | 0.457 ± 0.011 | 0.451 ± 0.012 |
| **Precision** | - | 0.547 ± 0.013 | 0.558 ± 0.016 |

**Observações**:
- Early stopping acionado na época 6 (após apenas 4 épocas sem melhoria)
- Convergência muito rápida, mas performance inferior aos modelos B0 e B1
- Possível underfitting devido ao early stopping prematuro
- Boa generalização: teste (0.779) ligeiramente superior à validação (0.775)

### EfficientNet-B3

**Melhor época**: 4/50

| Métrica | Train | Validation |
|---------|-------|------------|
| **Loss** | 0.118 | 0.361 |
| **Accuracy** | - | 58.68% ± N/A |
| **Kappa** | - | **0.801 ± N/A** |

**Observações**:
- **Treino ainda em andamento** (4 épocas completadas)
- Performance promissora já na época 4 (Kappa: 0.801)
- Sem resultados de teste ainda disponíveis
- Tendência de melhoria: Kappa evoluiu de 0.772 (época 1) para 0.801 (época 4)

## 🔗 Ensemble Learning

### Classe `EnsembleEfficientNet`

Implementada em `utils/models.py:189-317`, esta classe combina múltiplos modelos EfficientNet usando diferentes estratégias de agregação.

#### Métodos de Agregação Suportados

1. **Mean (Média Simples)**
   ```python
   result = torch.mean(outputs, dim=0)
   ```
   - Calcula a média aritmética das predições de todos os modelos
   - Assume igual importância para todos os modelos

2. **Weighted Mean (Média Ponderada)**
   ```python
   result = torch.sum(outputs * weights, dim=0)
   ```
   - Permite atribuir pesos diferentes para cada modelo
   - Útil quando alguns modelos têm melhor performance

3. **Max (Máximo)**
   ```python
   result, _ = torch.max(outputs, dim=0)
   ```
   - Seleciona o valor máximo entre todos os modelos
   - Mais confiante nas predições altas

4. **Majority Vote (Voto Majoritário)**
   - Hard voting: cada modelo vota na classe com maior probabilidade
   - A classe com mais votos é selecionada
   - Robusto a outliers

5. **Weighted Vote (Voto Ponderado)**
   - Soft voting com pesos personalizados
   - Combina probabilidades ponderadas antes da decisão final

#### Parâmetros do Ensemble

- **models**: Lista de modelos PyTorch treinados
- **method**: Método de agregação (`'mean'`, `'weighted_mean'`, `'max'`, `'majority_vote'`, `'weighted_vote'`)
- **weights**: Pesos opcionais para cada modelo (devem somar 1.0)
- **temperature**: Temperatura para suavização de probabilidades (padrão: 1.0)

### Exemplo de Uso

```python
from utils.models import EnsembleEfficientNet
import torch

# Carregar modelos treinados
model_b0 = EfficientNetApi(...)
model_b0.load_state_dict(torch.load('code/tests/common/models/b0.pth'))

model_b1 = EfficientNetApi(...)
model_b1.load_state_dict(torch.load('code/tests/common/models/b1.pth'))

model_b2 = EfficientNetApi(...)
model_b2.load_state_dict(torch.load('code/tests/common/models/b2.pth'))

model_b3 = EfficientNetApi(...)
model_b3.load_state_dict(torch.load('code/tests/common/models/b3.pth'))

# Criar ensemble com média ponderada baseada na performance de validação
ensemble = EnsembleEfficientNet(
    models=[model_b0, model_b1, model_b2, model_b3],
    method='weighted_mean',
    weights=[0.35, 0.45, 0.15, 0.05]  # B1 tem maior peso (melhor Kappa: 0.839)
)

# Fazer predições
predictions = ensemble.predict(input_images)
probabilities = ensemble.predict_proba(input_images)
```

### Vantagens do Ensemble

1. **Redução de Variância**: Combina predições de múltiplos modelos para reduzir erros aleatórios
2. **Melhor Generalização**: Cada modelo pode capturar padrões diferentes nos dados
3. **Robustez**: Menos sensível a peculiaridades de um único modelo
4. **Flexibilidade**: Permite ajustar estratégias de agregação dinamicamente

### Pesos Sugeridos para Ensemble

Baseado na performance de validação (Kappa):

```python
# Opção 1: Pesos proporcionais ao desempenho
weights = [
    0.35,  # B0: Kappa 0.832 (validation)
    0.45,  # B1: Kappa 0.839 (melhor modelo - validation)
    0.15,  # B2: Kappa 0.775 (validation)
    0.05   # B3: Kappa 0.801 (treino incompleto)
]

# Opção 2: Pesos uniformes (baseline)
weights = [0.25, 0.25, 0.25, 0.25]

# Opção 3: Apenas os 2 melhores modelos
weights = [0.48, 0.52, 0.00, 0.00]  # Apenas B0 e B1
```

**Recomendação**: Utilizar a Opção 1 para balancear performance individual com diversidade do ensemble.

## 📁 Estrutura do Projeto

```
repo/
├── code/
│   └── tests/
│       ├── efficientnet-b0.ipynb       # Treinamento B0
│       ├── efficientnet-b1.ipynb       # Treinamento B1
│       ├── efficientnet-b2.ipynb       # Treinamento B2
│       ├── efficientnet-b3.ipynb       # Treinamento B3
│       ├── logs/
│       │   ├── b0.txt                  # Métricas por época B0
│       │   ├── b1.txt                  # Métricas por época B1
│       │   ├── b2.txt                  # Métricas por época B2
│       │   └── b3.txt                  # Métricas por época B3
│       └── models/
│           ├── b0.pth                  # Pesos treinados B0
│           ├── b1.pth                  # Pesos treinados B1
│           ├── b2.pth                  # Pesos treinados B2
│           └── b3.pth                  # Pesos treinados B3
├── utils/
│   ├── models.py                       # Arquiteturas e Ensemble
│   ├── train.py                        # Loop de treinamento
│   ├── metrics.py                      # Cálculo de métricas
│   ├── dataset.py                      # Dataset loader
│   └── layer.py                        # Camadas customizadas
├── data/
│   ├── train_5fold.csv                 # Dados de treino com folds
│   └── test.csv                        # Dados de teste
└── README.md
```

## 🚀 Como Executar

### 1. Treinar Modelos Individuais

```bash
# Abrir e executar cada notebook
jupyter notebook code/tests/efficientnet-b0.ipynb
jupyter notebook code/tests/efficientnet-b1.ipynb
jupyter notebook code/tests/efficientnet-b2.ipynb
jupyter notebook code/tests/efficientnet-b3.ipynb
```

### 2. Usar Ensemble para Predição

```python
# Ver exemplo de uso na seção "Exemplo de Uso" acima
```

## 📈 Métricas de Avaliação

- **Accuracy**: Acurácia geral do modelo
- **Kappa (QWK)**: Quadratic Weighted Kappa - métrica principal para ordenação
- **F1-Score**: Média harmônica entre precisão e recall (macro)
- **Recall**: Taxa de verdadeiros positivos (macro)
- **Precision**: Proporção de predições corretas (macro)
- **Intervalos de Confiança**: 95% CI usando bootstrap

## 🔧 Dependências

```
torch
torchvision
efficientnet_pytorch
albumentations
pandas
numpy
tqdm
warmup_scheduler
```

## 📝 Notas Técnicas

### Early Stopping

Implementado com paciência de 5 épocas. O treinamento para se não houver melhoria no QWK de validação por 5 épocas consecutivas.

### Checkpointing

O melhor modelo é salvo automaticamente quando há melhoria no QWK de validação:

```python
if metrics["val_kappa"]['mean'] >= best_metric_criteria:
    checkpoint(model, best_metric_criteria, metrics["val_kappa"]['mean'], path_to_save_model)
```

### Reprodutibilidade

Seeds fixadas para garantir reprodutibilidade:
- PyTorch: `torch.manual_seed(42)`
- NumPy: `np.random.seed(42)`
- Random: `random.seed(42)`

## 📊 Visualizações e Próximos Passos

Para melhor compreensão dos resultados, considere:

1. **Gráficos de Treinamento**:
   - Curvas de loss (train vs validation)
   - Evolução das métricas por época
   - Comparação entre modelos

2. **Análise de Erro**:
   - Matriz de confusão para cada modelo
   - Análise de casos difíceis
   - Visualização de predições incorretas

3. **Ablation Studies**:
   - Impacto de diferentes métodos de ensemble
   - Sensibilidade aos pesos do ensemble
   - Análise da temperatura na suavização

4. **Otimizações Futuras**:
   - Testar outros backbones (EfficientNet-B4, B5)
   - Experimentar diferentes taxas de dropout
   - Test-Time Augmentation (TTA)
   - Stacking com meta-learner

## 👥 Autor

Projeto desenvolvido como parte de pesquisa de Doutorado.

## 📄 Licença

[Especificar licença do projeto]

---

**Status**: ✅ Modelos B0, B1, B2 treinados e testados | ⚠️ B3 treino em andamento | ✅ Ensemble implementado

### 🏆 Rankings

**Por Performance no Validation Set:**
1. 🥇 EfficientNet-B1: Kappa **0.839** ± 0.011
2. 🥈 EfficientNet-B0: Kappa **0.832** ± 0.011
3. 🥉 EfficientNet-B3: Kappa **0.801** ± N/A (incompleto)
4. EfficientNet-B2: Kappa **0.775** ± 0.012

**Por Performance no Test Set:**
1. 🥇 EfficientNet-B1: Kappa **0.834** ± 0.013
2. 🥈 EfficientNet-B0: Kappa **0.826** ± 0.012
3. 🥉 EfficientNet-B2: Kappa **0.779** ± 0.013
4. EfficientNet-B3: Aguardando conclusão do treino

**Performance Esperada do Ensemble**: > 0.845 Kappa (estimativa baseada na complementaridade dos modelos)
