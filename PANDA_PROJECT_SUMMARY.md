# Relatório Técnico: Classificação de Câncer de Próstata (Dataset PANDA)

Este documento resume a metodologia, os experimentos e os resultados obtidos no projeto de classificação automática de grau ISUP em biópsias de próstata utilizando o dataset **PANDA (Prostate cANcer graDe Assessment)**.

---

## 1. Visão Geral do Dataset
- **Dataset:** PANDA Challenge (Kaggle).
- **Total de Imagens:** 10.616 slides inteiros (WSI).
- **Classes (ISUP Grade):** 0 (Benigno) a 5 (Mais agressivo).
- **Desafios:**
  - Imagens de altíssima resolução (gigapixel).
  - Grande quantidade de ruído (áreas sem tecido, artefatos de coloração).
  - Subjetividade na anotação (variabilidade entre patologistas).

---

## 2. Metodologia

### 2.1 Pré-processamento e Augmentação
- **Tiling:** Divisão dos slides em patches de $224 \times 224$ pixels.
- **Augmentação (Treino):** Flips (H/V), Transposição, Rotação e Jittering de cor para simular variações de laboratório.

### 2.2 Arquitetura do Modelo
- **Backbone Principal:** EfficientNet-B0 (escolhido pelo excelente balanço entre eficiência e performance).
- **Pooling:** Generalized Mean (GeM) pooling ($p=3$).
- **Atenção:** Squeeze-and-Excitation (SE) blocks integrados.
- **Ordinal Encoding:** Implementação de encoding cumulativo para preservar a ordem das classes (Grade 3 = [1, 1, 1, 0, 0]).

### 2.3 Filtragem por Entropia (Contribuição Principal)
- **Técnica:** Remoção de patches com alta entropia de Shannon ($H > 7.5$).
- **Objetivo:** Eliminar áreas ruidosas (bolhas de ar, tecido conjuntivo excessivo, artefatos).
- **Impacto:** Redução de ~18% do dataset, mas com melhora significativa na qualidade dos dados.

---

## 3. Experimentos Realizados

### 3.1 Escalonamento de Modelos (EfficientNet B0 a B7)
- Testamos diferentes escalas da EfficientNet para avaliar se modelos maiores trariam melhores resultados.
- **Conclusão:** Modelos menores (B0, B1) generalizaram melhor. O modelo B7 sofreu de *overfitting* severo devido ao tamanho limitado do dataset médico.

### 3.2 Espaços de Cores Alternativos
- Avaliamos CIE XYZ, HED (deconvolução de cor para patologia) e CIELUV.
- **Conclusão:** O espaço RGB padrão superou todos os outros, validando a robustez das features pré-treinadas no ImageNet.

### 3.3 Mecanismos de Atenção
- Comparamos SE blocks (locais) com Self-Attention global.
- **Conclusão:** Self-Attention global foi instável (colapso de gradiente), enquanto a atenção local (SE) foi fundamental para capturar padrões celulares.

### 3.4 Técnicas Avançadas
- **SWA (Stochastic Weight Averaging):** Média de pesos no fim do treino para encontrar mínimos mais estáveis.
- **TTA (Test-Time Augmentation):** Média de predições de 8 versões aumentadas da imagem de teste.
- **Ensemble:** Combinação dos modelos B0-Entropy, B1 e B3.

---

## 4. Resultados Obtidos

| Configuração | Quadratic Weighted Kappa (QWK) | Accuracy (%) |
| :--- | :---: | :---: |
| **EfficientNet-B0 Baseline** | 0.8321 | 60.53 |
| **EfficientNet-B0 + Entropy Filtering** | **0.8730** | **66.59** |
| **EfficientNet-B1** | 0.8390 | 63.38 |
| **EfficientNet-B7** | 0.8034 | 59.36 |
| **Ensemble (B0-Entropy, B1, B3)** | **0.8810** | **67.20** |

### Significância Estatística
- **Bootstrap (n=1000):** O modelo com Filtragem por Entropia obteve QWK de 0.8730 [IC 95%: 0.852, 0.894], superando o baseline de forma estatisticamente relevante.

---

## 5. Principais Conclusões
1. **Qualidade > Quantidade:** A remoção de 18% dos dados ruidosos (via entropia) trouxe o maior ganho de performance individual (+4% no QWK).
2. **Eficiência Arquitetural:** Em datasets médicos limitados, modelos menores e eficientes (B0) superam modelos gigantes (B7).
3. **Localização é Chave:** Padrões histológicos são locais; mecanismos de atenção local (SE blocks) são mais estáveis e eficazes que atenção global.
4. **Encoding Ordinal:** Essencial para problemas de graduação (ISUP), garantindo que erros entre classes próximas sejam menos penalizados que erros grosseiros.

---
**Data:** 15 de Maio de 2026
**Projeto:** Doutorado - Classificação de Câncer de Próstata
