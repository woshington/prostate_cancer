import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { Download, FileText, BarChart3, TrendingUp } from 'lucide-react';

const ExperimentPipelineDoc = () => {
  const [activeSection, setActiveSection] = useState('overview');

  const colorSpaceData = [
    { name: 'RGB', accuracy: 59.282, f1: 53.7, kappa: 82.6 },
    { name: 'XYZ', accuracy: 61.859, f1: 55.3, kappa: 81.9 },
    { name: 'LAB', accuracy: 56.106, f1: 50.0, kappa: 80.2 },
    { name: 'LUV', accuracy: 49.512, f1: 44.8, kappa: 78.1 }
  ];

  const ensembleData = [
    { name: 'RGB', accuracy: 59.282, f1: 53.7 },
    { name: 'FUSION', accuracy: 62.17, f1: 55.0 },
    { name: 'ENTROPY', accuracy: 62.929, f1: 57.4 },
    { name: 'SWA', accuracy: 61.781, f1: 55.5 },
    { name: 'GEM+SA', accuracy: 62.51, f1: 57.0 },
    { name: 'TTA', accuracy: 47.515, f1: 42.8 }
  ];

  const architectureData = [
    { name: 'B0', accuracy: 62.929, f1: 57.4, kappa: 83.3, precision: 57.3, recall: 58.0 },
    { name: 'B3', accuracy: 52.448, f1: 43.4, kappa: 77.9, precision: 55.8, recall: 45.1 },
    { name: 'B7', accuracy: 56.753, f1: 49.9, kappa: 80.4, precision: 50.4, recall: 49.9 },
    { name: 'ResNet', accuracy: 52.856, f1: 44.0, kappa: 75.9, precision: 46.0, recall: 44.0 },
    { name: 'Ensemble', accuracy: 64.935, f1: 58.6, kappa: 84.3, precision: 59.1, recall: 58.5 }
  ];

  const radarData = [
    { metric: 'Acurácia', B0: 62.929, Ensemble: 64.935, B7: 56.753 },
    { metric: 'F1', B0: 57.4, Ensemble: 58.6, B7: 49.9 },
    { metric: 'Kappa', B0: 83.3, Ensemble: 84.3, B7: 80.4 },
    { metric: 'Precision', B0: 57.3, Ensemble: 59.1, B7: 50.4 },
    { metric: 'Recall', B0: 58.0, Ensemble: 58.5, B7: 49.9 }
  ];

  const rankingData = [
    { rank: 1, model: 'Ensemble B0-B3-B7', accuracy: 64.935, kappa: 84.3, f1: 58.6 },
    { rank: 2, model: 'SWA (B0)', accuracy: 61.781, kappa: 83.7, f1: 55.5 },
    { rank: 3, model: 'ENTROPY (B0)', accuracy: 62.929, kappa: 83.3, f1: 57.4 },
    { rank: 4, model: 'FUSION (B0)', accuracy: 62.17, kappa: 83.0, f1: 55.0 },
    { rank: 5, model: 'GEM+SA (B0)', accuracy: 62.51, kappa: 83.0, f1: 57.0 },
    { rank: 6, model: 'RGB (B0)', accuracy: 59.282, kappa: 82.6, f1: 53.7 },
    { rank: 7, model: 'XYZ (B0)', accuracy: 61.859, kappa: 81.9, f1: 55.3 },
    { rank: 8, model: 'B7', accuracy: 56.753, kappa: 80.4, f1: 49.9 }
  ];

  const downloadAsWord = () => {
    const content = `
DOCUMENTAÇÃO DO PIPELINE DE EXPERIMENTOS
CLASSIFICAÇÃO DE IMAGENS COM DEEP LEARNING

═══════════════════════════════════════════════════════════════════

1. VISÃO GERAL DO PROJETO

Este documento apresenta a documentação completa do pipeline de experimentos 
desenvolvido para classificação de imagens utilizando diferentes arquiteturas 
de redes neurais convolucionais (CNN) e técnicas avançadas de ensemble learning.

Objetivo: Maximizar a acurácia de classificação em um problema multi-classe (5 classes)
Abordagem: Transfer Learning com fine-tuning e ensemble de modelos
Metodologia: Avaliação sistemática com intervalos de confiança de 95%

═══════════════════════════════════════════════════════════════════

2. CONFIGURAÇÕES EXPERIMENTAIS

2.1 HIPERPARÂMETROS BASE

Parâmetro                Valor      Descrição
Seed                     42         Reprodutibilidade
Batch Size               6          Tamanho do lote
Num Workers              4          Threads de carregamento
Output Classes           5          Número de classes
Learning Rate Inicial    3e-4       Taxa de aprendizado
Warmup Factor            2          Fator de aquecimento
Warmup Epochs            1          Épocas de warmup
Total Epochs             50         Épocas totais
Pre-trained              True       Pesos pré-treinados ImageNet
Dropout                  0.3        Taxa de regularização
Fine Tune Layers         150        Camadas para ajuste fino

2.2 ESTRATÉGIA DE TREINAMENTO

✓ Transfer Learning: Aproveitamento de conhecimento do ImageNet
✓ Fine-tuning Parcial: Ajuste das últimas 150 camadas
✓ Warm-up Learning Rate: Aquecimento gradual na primeira época
✓ Dropout Regularization: 30% para prevenção de overfitting
✓ Data Shuffling: Embaralhamento dos dados a cada época

═══════════════════════════════════════════════════════════════════

3. EXPERIMENTOS COM EFFICIENTNET-B0

3.1 ESPAÇOS DE COR

Espaço   Acurácia   F1       Kappa    IC 95%
RGB      59.28%     53.7%    82.6%    [57.29% - 61.24%]
XYZ      61.86%     55.3%    81.9%    [59.92% - 63.88%]
LAB      56.11%     50.0%    80.2%    [54.08% - 58.23%]
LUV      49.51%     44.8%    78.1%    [47.36% - 51.57%]

Análise:
• XYZ superou RGB em +2.58 pontos percentuais
• LAB teve desempenho intermediário
• LUV apresentou resultados significativamente inferiores

3.2 TÉCNICAS DE ENSEMBLE E OTIMIZAÇÃO

Técnica     Acurácia   F1       Kappa    IC 95%
FUSION      62.17%     55.0%    83.0%    [60.24% - 64.26%]
ENTROPY     62.93%     57.4%    83.3%    [60.93% - 64.82%]
SWA         61.78%     55.5%    83.7%    [59.86% - 63.69%]
GEM+SA      62.51%     57.0%    83.0%    [60.49% - 64.45%]
TTA         47.52%     42.8%    78.1%    [45.54% - 49.56%]

Destaques:
★ ENTROPY: Melhor resultado individual com B0 (62.93%)
★ SWA: Maior valor de Kappa (83.7%)
★ FUSION: Boa combinação de múltiplos espaços de cor

═══════════════════════════════════════════════════════════════════

4. COMPARAÇÃO ENTRE ARQUITETURAS

Arquitetura   Acurácia   F1      Kappa    Precision   Recall
B0 (ENTROPY)  62.93%     57.4%   83.3%    57.3%       58.0%
B3            52.45%     43.4%   77.9%    55.8%       45.1%
B7            56.75%     49.9%   80.4%    50.4%       49.9%
ResNet        52.86%     44.0%   75.9%    46.0%       44.0%

Observações:
• B0 superou consistentemente B3 e B7
• Modelos maiores provavelmente sofrem de overfitting
• Dataset atual não justifica arquiteturas mais complexas

═══════════════════════════════════════════════════════════════════

5. ENSEMBLE MULTI-ARQUITETURA

ENSEMBLE B0 + B3 + B7 (MELHOR RESULTADO GERAL)

Acurácia:  64.94% ± 1.23%  [IC 95%: 62.81% - 66.90%]
Kappa:     84.3% ± 1.2%    [IC 95%: 82.2% - 86.3%]
F1-Score:  58.6% ± 1.3%
Recall:    58.5% ± 1.3%
Precision: 59.1% ± 1.4%

Ganho sobre melhor modelo individual: +2.01 pontos percentuais

═══════════════════════════════════════════════════════════════════

6. RANKING COMPLETO DOS MODELOS (ORDENADO POR KAPPA)

Pos  Modelo                 Kappa   Acurácia   F1      Concordância
1    Ensemble B0-B3-B7      84.3%   64.94%     58.6%   Excelente
2    SWA (B0)               83.7%   61.78%     55.5%   Excelente
3    ENTROPY (B0)           83.3%   62.93%     57.4%   Excelente
4    FUSION (B0)            83.0%   62.17%     55.0%   Excelente
5    GEM+SA (B0)            83.0%   62.51%     57.0%   Excelente
6    RGB (B0)               82.6%   59.28%     53.7%   Excelente
7    XYZ (B0)               81.9%   61.86%     55.3%   Excelente
8    B7                     80.4%   56.75%     49.9%   Substancial

Interpretação do Kappa (Cohen's Kappa):
• 0.81 - 1.00: Concordância Excelente
• 0.61 - 0.80: Concordância Substancial
• 0.41 - 0.60: Concordância Moderada

Destaque Especial:
★ SWA conquistou o melhor Kappa entre modelos individuais (83.7%)
★ Ensemble B0-B3-B7 lidera tanto em Kappa quanto em Acurácia
★ Todos os modelos baseados em B0 atingiram concordância "Excelente"

═══════════════════════════════════════════════════════════════════

7. RECOMENDAÇÕES

PARA PRODUÇÃO:

Opção 1: ENTROPY com B0 (62.93%)
- Menor latência e uso de recursos
- Recomendado para APIs em tempo real

Opção 2: Ensemble B0-B3-B7 (64.94%)
- Máxima acurácia e robustez
- Recomendado quando acurácia é crítica

PRÓXIMOS PASSOS:

Prioridade ALTA:
□ Investigar baixo desempenho do TTA
□ Avaliar Vision Transformer (ViT)
□ Testar HSL e HSV
□ Aumentar batch size

Prioridade MÉDIA:
□ Testar voting strategies
□ Implementar stacking ensemble
□ Explorar mixup/cutmix
□ Ajustar hiperparâmetros B3 e B7

═══════════════════════════════════════════════════════════════════

Documento gerado automaticamente
Data: ${new Date().toLocaleDateString('pt-BR')}
    `;

    const blob = new Blob([content], { type: 'application/msword' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'Pipeline_Experimentos_Completo.doc';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white rounded-lg shadow-xl p-8 mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-gray-800 mb-2">
                Pipeline de Experimentos
              </h1>
              <p className="text-xl text-gray-600">
                Classificação de Imagens com Deep Learning
              </p>
              <div className="mt-4 flex gap-4 text-sm text-gray-500">
                <span className="flex items-center gap-2">
                  <BarChart3 className="w-4 h-4" />
                  13 Configurações Testadas
                </span>
                <span className="flex items-center gap-2">
                  <TrendingUp className="w-4 h-4" />
                  Melhor Acurácia: 64.94%
                </span>
              </div>
            </div>
            <button
              onClick={downloadAsWord}
              className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-semibold transition-colors shadow-lg"
            >
              <Download className="w-5 h-5" />
              Baixar Documento Word
            </button>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-4 mb-8">
          <div className="flex gap-2 overflow-x-auto">
            {[
              { id: 'overview', label: 'Visão Geral', icon: FileText },
              { id: 'colorspace', label: 'Espaços de Cor', icon: BarChart3 },
              { id: 'ensemble', label: 'Ensemble B0', icon: TrendingUp },
              { id: 'architectures', label: 'Arquiteturas', icon: BarChart3 },
              { id: 'ranking', label: 'Ranking Final', icon: TrendingUp }
            ].map(section => {
              const IconComponent = section.icon;
              return (
                <button
                  key={section.id}
                  onClick={() => setActiveSection(section.id)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                    activeSection === section.id
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  <IconComponent className="w-4 h-4" />
                  {section.label}
                </button>
              );
            })}
          </div>
        </div>

        {activeSection === 'overview' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-lg p-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">Configurações Experimentais</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                  { label: 'Seed', value: '42' },
                  { label: 'Batch Size', value: '6' },
                  { label: 'Epochs', value: '50' },
                  { label: 'Learning Rate', value: '3e-4' },
                  { label: 'Classes', value: '5' },
                  { label: 'Dropout', value: '0.3' },
                  { label: 'Workers', value: '4' },
                  { label: 'Fine-tune', value: '150 camadas' }
                ].map((item, idx) => (
                  <div key={idx} className="bg-gradient-to-br from-blue-50 to-indigo-50 p-4 rounded-lg">
                    <p className="text-sm text-gray-600 mb-1">{item.label}</p>
                    <p className="text-xl font-bold text-gray-800">{item.value}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-white rounded-lg shadow-lg p-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">Destaques do Pipeline</h2>
              <div className="grid md:grid-cols-3 gap-6">
                <div className="border-l-4 border-green-500 pl-4">
                  <h3 className="font-bold text-lg text-gray-800 mb-2">Melhor Modelo Individual</h3>
                  <p className="text-3xl font-bold text-green-600">62.93%</p>
                  <p className="text-gray-600">ENTROPY com B0</p>
                </div>
                <div className="border-l-4 border-blue-500 pl-4">
                  <h3 className="font-bold text-lg text-gray-800 mb-2">Melhor Ensemble</h3>
                  <p className="text-3xl font-bold text-blue-600">64.94%</p>
                  <p className="text-gray-600">B0 + B3 + B7</p>
                </div>
                <div className="border-l-4 border-purple-500 pl-4">
                  <h3 className="font-bold text-lg text-gray-800 mb-2">Ganho Total</h3>
                  <p className="text-3xl font-bold text-purple-600">+5.66%</p>
                  <p className="text-gray-600">vs RGB Baseline</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeSection === 'colorspace' && (
          <div className="bg-white rounded-lg shadow-lg p-8">
            <h2 className="text-2xl font-bold text-gray-800 mb-6">Comparação de Espaços de Cor (EfficientNet-B0)</h2>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={colorSpaceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis yAxisId="left" domain={[40, 70]} />
                <YAxis yAxisId="right" orientation="right" domain={[40, 90]} />
                <Tooltip />
                <Legend />
                <Bar yAxisId="left" dataKey="accuracy" fill="#3B82F6" name="Acurácia (%)" />
                <Bar yAxisId="right" dataKey="f1" fill="#10B981" name="F1-Score (%)" />
                <Bar yAxisId="right" dataKey="kappa" fill="#8B5CF6" name="Kappa (%)" />
              </BarChart>
            </ResponsiveContainer>
            <div className="mt-6 grid md:grid-cols-2 gap-4">
              <div className="bg-green-50 p-4 rounded-lg">
                <h3 className="font-bold text-green-800 mb-2">✓ Melhor Desempenho: XYZ</h3>
                <p className="text-gray-700">Acurácia de 61.86%, F1 de 55.3%, Kappa de 81.9%</p>
              </div>
              <div className="bg-red-50 p-4 rounded-lg">
                <h3 className="font-bold text-red-800 mb-2">✗ Pior Desempenho: LUV</h3>
                <p className="text-gray-700">Acurácia de 49.51%, F1 de 44.8%, Kappa de 78.1%</p>
              </div>
            </div>
          </div>
        )}

        {activeSection === 'ensemble' && (
          <div className="bg-white rounded-lg shadow-lg p-8">
            <h2 className="text-2xl font-bold text-gray-800 mb-6">Técnicas de Ensemble com EfficientNet-B0</h2>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={ensembleData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis yAxisId="left" domain={[40, 70]} />
                <YAxis yAxisId="right" orientation="right" domain={[40, 60]} />
                <Tooltip />
                <Legend />
                <Line yAxisId="left" type="monotone" dataKey="accuracy" stroke="#3B82F6" strokeWidth={3} name="Acurácia (%)" />
                <Line yAxisId="right" type="monotone" dataKey="f1" stroke="#10B981" strokeWidth={3} name="F1-Score (%)" />
              </LineChart>
            </ResponsiveContainer>
            <div className="mt-6 space-y-3">
              {[
                { name: 'ENTROPY', acc: '62.93%', desc: 'Melhor técnica individual - Ensemble baseado em entropia', bg: 'bg-yellow-50', border: 'border-yellow-500' },
                { name: 'GEM + Self-Attention', acc: '62.51%', desc: 'Combinação de pooling global e atenção', bg: 'bg-blue-50', border: 'border-blue-500' },
                { name: 'FUSION', acc: '62.17%', desc: 'Fusão de múltiplos espaços de cor', bg: 'bg-purple-50', border: 'border-purple-500' },
                { name: 'SWA', acc: '61.78%', desc: 'Stochastic Weight Averaging - Melhor Kappa', bg: 'bg-green-50', border: 'border-green-500' }
              ].map((item, idx) => (
                <div key={idx} className={`${item.bg} border-l-4 ${item.border} p-4 rounded`}>
                  <div className="flex justify-between items-center">
                    <h3 className="font-bold text-gray-800">{item.name}</h3>
                    <span className="text-2xl font-bold text-gray-700">{item.acc}</span>
                  </div>
                  <p className="text-gray-600 mt-1">{item.desc}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeSection === 'architectures' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-lg p-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-6">Comparação entre Arquiteturas</h2>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={architectureData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis domain={[0, 70]} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="accuracy" fill="#3B82F6" name="Acurácia (%)" />
                  <Bar dataKey="f1" fill="#10B981" name="F1-Score (%)" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-white rounded-lg shadow-lg p-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-6">Radar Chart - Métricas Múltiplas</h2>
              <ResponsiveContainer width="100%" height={400}>
                <RadarChart data={radarData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="metric" />
                  <PolarRadiusAxis domain={[0, 100]} />
                  <Radar name="B0 ENTROPY" dataKey="B0" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.3} />
                  <Radar name="Ensemble" dataKey="Ensemble" stroke="#10B981" fill="#10B981" fillOpacity={0.3} />
                  <Radar name="B7" dataKey="B7" stroke="#EF4444" fill="#EF4444" fillOpacity={0.3} />
                  <Legend />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {activeSection === 'ranking' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-lg p-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-6">Ranking Final dos Modelos</h2>
              <div className="space-y-3">
                {rankingData.map((item, idx) => (
                  <div 
                    key={idx} 
                    className={`p-4 rounded-lg border-2 ${
                      idx === 0 ? 'bg-yellow-50 border-yellow-400' : 
                      idx === 1 ? 'bg-gray-50 border-gray-400' : 
                      idx === 2 ? 'bg-orange-50 border-orange-400' : 
                      'bg-white border-gray-200'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-4">
                        <span className={`text-3xl font-bold ${
                          idx === 0 ? 'text-yellow-600' : 
                          idx === 1 ? 'text-gray-600' : 
                          idx === 2 ? 'text-orange-600' : 
                          'text-gray-400'
                        }`}>
                          #{item.rank}
                        </span>
                        <div>
                          <h3 className="font-bold text-lg text-gray-800">{item.model}</h3>
                          <p className="text-sm text-gray-600">
                            {idx === 0 && '🏆 Melhor Kappa Geral'}
                            {idx === 1 && '🥈 Melhor Kappa Individual'}
                            {idx === 2 && '🥉 Excelente Concordância'}
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="flex gap-6">
                          <div>
                            <p className="text-2xl font-bold text-blue-600">{item.accuracy.toFixed(2)}%</p>
                            <p className="text-xs text-gray-500">Acurácia</p>
                          </div>
                          <div>
                            <p className="text-2xl font-bold text-purple-600">{item.kappa.toFixed(1)}%</p>
                            <p className="text-xs text-gray-500">Kappa</p>
                          </div>
                          <div>
                            <p className="text-2xl font-bold text-green-600">{item.f1.toFixed(1)}%</p>
                            <p className="text-xs text-gray-500">F1-Score</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-white rounded-lg shadow-lg p-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-6">Comparação Visual - Top 8 Modelos (Ordenado por Kappa)</h2>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={rankingData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="model" angle={-45} textAnchor="end" height={120} />
                  <YAxis domain={[0, 90]} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="kappa" fill="#8B5CF6" name="Kappa (%) ⭐" />
                  <Bar dataKey="accuracy" fill="#3B82F6" name="Acurácia (%)" />
                  <Bar dataKey="f1" fill="#10B981" name="F1-Score (%)" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 rounded-lg">
              <h3 className="font-bold text-xl text-gray-800 mb-4">Principais Conclusões (Ordenado por Kappa)</h3>
              <div className="space-y-2 text-gray-700">
                <p>✓ O Ensemble B0-B3-B7 alcançou o melhor Kappa (84.3%) e melhor acurácia (64.94%)</p>
                <p>✓ SWA obteve o 2º melhor Kappa (83.7%), superando ENTROPY em concordância</p>
                <p>✓ ENTROPY ficou em 3º lugar com Kappa de 83.3%, mas tem a melhor acurácia individual (62.93%)</p>
                <p>✓ FUSION e GEM+SA empataram em Kappa (83.0%), ambos com excelente desempenho</p>
                <p>✓ O Kappa mede concordância ajustada pelo acaso - valores acima de 80% são considerados excelentes</p>
                <p>✓ Todos os modelos B0 com técnicas de ensemble superaram 82% de Kappa</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ExperimentPipelineDoc;