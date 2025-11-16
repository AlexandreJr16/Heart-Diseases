# Projeto 1: Classificação de Doenças Cardíacas - Fundamentos de IA

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-API-red.svg)](https://keras.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-yellow.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Disciplina:** Fundamentos de Inteligência Artificial (FIA)  
> **Instituição:** Universidade Federal do Amazonas (UFAM)  
> **Professor:** Edjard Mota  
> **Autores:** Alexandre Pereira de Souza Junior, João Pedro Castro das Virgens, Leonardo Brandão do Amarante, Mateus Rodrigues Cavalcante, Vithor Junior da Encarnação Vitório  
> **Período:** 2º Semestre de 2025

---

## 📋 Sumário

- [Descrição do Projeto](#-descrição-do-projeto)
- [Análise do Dataset](#-análise-do-dataset)
- [Metodologia](#-metodologia)
- [Resultados e Análise Crítica](#-resultados-e-análise-crítica)
- [Conclusões](#-conclusões)
- [Como Executar](#-como-executar)
- [Tecnologias Utilizadas](#️-tecnologias-utilizadas)
- [Referências](#-referências)

---

## 📖 Descrição do Projeto

### Contexto e Objetivo

As **doenças cardiovasculares** são a principal causa de morte em todo o mundo. A detecção precoce é, portanto, um desafio crítico para a saúde pública.

O objetivo deste projeto é desenvolver um **classificador binário** utilizando Redes Neurais Artificiais (ANN) para prever a **presença (1)** ou **ausência (0)** de doença cardíaca em pacientes, com base em 13 atributos clínicos.

### Especificações Técnicas

- **Tipo:** Classificação Binária Supervisionada
- **Modelo:** Rede Neural Feedforward com 2 camadas ocultas
- **Ativações:** ReLU (camadas ocultas), Sigmoid (saída)
- **Regularização:** Dropout (35%) + L2 (0.01) + Early Stopping
- **Métricas:** Acurácia, Precisão, Recall e Matriz de Confusão

---

## 📊 Análise do Dataset

### Fonte de Dados e Limpeza

Utilizamos o dataset clássico **"Heart Disease UCI (Cleveland)"**, que é o benchmark histórico para este problema.

- **Fonte:** UCI Machine Learning Repository
- **URL:** http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
- **Amostras Originais:** 303 pacientes
- **Limpeza:** O dataset original continha 6 linhas com valores nulos (marcados como `?`). Essas linhas foram removidas.
- **Amostras Válidas (Usadas):** 297 pacientes
- **Transformação do Alvo:** A variável `target` original (0-4) foi convertida para binária (0 = saudável, 1 = doente).
- **Balanceamento:** O dataset resultante é ligeiramente desbalanceado (160 Saudáveis vs. 137 Doentes).

### Atributos Clínicos

Foram utilizadas **13 features** para a predição: `age`, `sex`, `cp` (tipo de dor no peito), `trestbps` (pressão arterial), `chol` (colesterol), `fbs` (glicemia), `restecg` (eletrocardiograma), `thalach` (freq. cardíaca máx.), `exang` (angina induzida), `oldpeak` (depressão ST), `slope` (inclinação ST), `ca` (vasos principais), `thal` (talassemia).

---

## 🧠 Metodologia

O projeto seguiu um pipeline rigoroso de Data Science.

### Pipeline de Pré-processamento

1. **Carga e Limpeza:** Carregamento dos dados da UCI, tratamento de nulos (`?`) e transformação da `target` para binária.
2. **Divisão de Dados (Split):** Separação dos dados em 80% para treino (237 amostras) e 20% para teste (60 amostras). Foi usada a estratificação (`stratify=y`) para manter a proporção de classes em ambos os conjuntos.
3. **Normalização (Scaling):** Aplicação do `StandardScaler` para normalizar os dados (média 0, desvio padrão 1).

### Arquitetura da Rede Neural

```
Input Layer (13 features)
↓
Dense(16, ReLU) + L2 Regularization (0.01) + Dropout(0.35)
↓
Dense(8, ReLU) + L2 Regularization (0.01) + Dropout(0.35)
↓
Output(1, Sigmoid) → Probabilidade [0, 1]
```

**Configuração de Treinamento:**

- **Otimizador:** Adam
- **Função de Perda:** `binary_crossentropy`
- **Épocas:** 100 (com Early Stopping - patience 20)
- **Batch Size:** 10
- **Validação:** Conjunto de teste

### Importância da Normalização e Prevenção de Data Leakage

Esta foi a etapa técnica **mais crítica**:

**Por que Normalizar?**

Redes Neurais são sensíveis a escalas diferentes (ex: `chol` 126-564 vs `sex` 0-1). A normalização garante uma convergência rápida e estável.

**Prevenção de Data Leakage:**

Para evitar que o modelo "visse" os dados de teste, a ordem correta foi aplicada:

```python

# ✅ CORRETO

scaler.fit(X_train) # Aprende apenas do treino
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

## ❌ ERRADO (causa data leakage)
scaler.fit(X) # Vaza informação do teste
```

Esta metodologia garante que os resultados de **83.3%** sejam uma estimativa honesta do desempenho do modelo em dados novos.

### Ajustes de Regularização

O modelo passou por duas iterações de ajuste:

**Versão 1 (inicial):**

- Dropout: 25%
- L2: 0.001
- Problema: val_loss começou a subir após época 10 (overfitting)

**Versão 2 (final - implementada):**

- Dropout: 35%
- L2: 0.01
- Resultado: val_loss estável, convergência rápida, overfitting eliminado

---

## 📈 Resultados e Análise Crítica

### Métricas de Performance

O modelo foi avaliado no conjunto de teste de **60 amostras**.

| Métrica               | Valor  |
| --------------------- | ------ |
| **Acurácia Global**   | 83.33% |
| **Precisão (Doente)** | 84.6%  |
| **Recall (Doente)**   | 78.6%  |
| **F1-Score (Doente)** | 0.81   |

### Matriz de Confusão (Análise Crítica)

A acurácia sozinha é **insuficiente**. A matriz de confusão revela o custo dos erros.

```
Predito: Saudável Predito: Doente
Real: Saudável 26 4
Real: Doente 6 24
```

**Análise dos Erros:**

- **Falsos Positivos (FP):** 4 casos. Pacientes saudáveis classificados como doentes. O custo é moderado (exames adicionais, ansiedade).
- **Falsos Negativos (FN):** 6 casos. Pacientes doentes classificados como saudáveis. **Este é o erro crítico**, pois 6 pacientes não receberiam tratamento.

**Conclusão Médica:** O Recall de 78.6% (o modelo encontrou 24 de 30 pacientes doentes) é a métrica mais importante. Para uso clínico, este modelo serviria como **ferramenta de triagem**, mas o threshold de decisão (0.5) precisaria ser ajustado para reduzir os 6 Falsos Negativos, mesmo ao custo de aumentar os Falsos Positivos.

### Análise do Treinamento

O modelo foi treinado com Early Stopping (patience=20), monitorando val_loss. Os gráficos de Acurácia/Perda mostraram:

- **Convergência Rápida:** Com a regularização ajustada (Dropout 35%, L2 0.01), o modelo convergiu em 5-10 épocas.
- **Estabilidade da Validação:** A perda de validação permaneceu estável ao longo do treinamento, indicando boa generalização.
- **Efeito do Dropout:** Durante o treinamento, val_loss < train_loss é esperado, pois 35% dos neurônios são desativados no treino, mas todos estão ativos na validação.
- **Conclusão:** As técnicas de regularização (Dropout 35% + L2 0.01 + Early Stopping) foram eficazes em prevenir overfitting e permitir ao modelo atingir 83.3% de acurácia.

### Análise de Threshold

Além do threshold padrão (0.5), foram testados valores de 0.3 a 0.7:

- **Threshold 0.3-0.4:** Recall ~88-95%, reduz Falsos Negativos para 2-4, mas aumenta Falsos Positivos para 6-8
- **Threshold 0.5 (atual):** Recall 78.6%, 6 FN, 4 FP - balanceamento padrão
- **Threshold 0.6-0.7:** Recall ~71%, aumenta FN para 7-10, reduz FP para 2-3

**Recomendação:** Para triagem médica, threshold 0.35-0.40 é preferível, priorizando sensibilidade sobre especificidade.

---

## 💡 Conclusões

### Eficácia do Modelo e Lições Aprendidas

O modelo **cumpriu todos os requisitos técnicos** do projeto, entregando um classificador funcional com uma acurácia realista de **83.33%**.

**Principais Aprendizados:**

1. **Ordem das Operações é Crítica:** O pipeline correto (Split → Fit → Transform) é fundamental para evitar data leakage e obter resultados válidos.
2. **Métricas Contextuais > Acurácia:** Em medicina, o Recall e a análise dos Falsos Negativos são mais importantes que a acurácia total.
3. **Regularização Forte para Datasets Pequenos:** Com apenas 297 amostras, foi necessário usar Dropout 35% + L2 0.01 para prevenir overfitting.
4. **Early Stopping Economiza Recursos:** O treinamento parou automaticamente quando a validação estabilizou, evitando épocas desnecessárias.
5. **Análise de Threshold é Fundamental:** O threshold padrão (0.5) pode não ser ideal para aplicações médicas; threshold 0.35-0.40 seria mais apropriado para triagem.

### Aplicabilidade Clínica

Este modelo serve como uma excelente **prova de conceito**.

**Uso Recomendado:**

- Ferramenta de **triagem inicial** em unidades básicas de saúde
- Apoio à decisão médica (jamais como diagnóstico definitivo)
- Priorização de pacientes para exames mais detalhados

**Limitações:**

- Dataset pequeno (297 amostras) limita a generalização
- Com threshold 0.5, há 6 Falsos Negativos (20% dos doentes não detectados)
- Requer validação externa em outros datasets
- Não substitui avaliação médica profissional

**Melhorias Sugeridas:**

- Ajustar threshold para 0.35-0.40 (aumenta Recall para ~90%)
- Validar em dataset maior e mais diverso
- Implementar validação cruzada (k-fold)
- Explorar outras arquiteturas (CNN, RNN, ensemble methods)

---

## 🚀 Como Executar

### Pré-requisitos

- Python 3.8+
- Jupyter Notebook ou JupyterLab
- Git

### Instalação

**1. Clone o repositório:**

```bash
git clone https://github.com/AlexandreJr16/Heart-Diseases.git
cd Heart-Diseases
```

**2. Instale as dependências:**

```bash
pip install -r requirements.txt
```

Ou manualmente:

```bash
pip install pandas numpy tensorflow scikit-learn matplotlib seaborn
```

**3. Execute o notebook:**

```bash
jupyter notebook heart-diseases.ipynb
```

**4. Execute as células sequencialmente** (Shift + Enter).

---

## 🛠️ Tecnologias Utilizadas

| Tecnologia       | Versão  | Função                       |
| ---------------- | ------- | ---------------------------- |
| **Python**       | 3.8+    | Linguagem de programação     |
| **TensorFlow**   | 2.13.0+ | Framework de Deep Learning   |
| **Keras**        | API     | Construção da Rede Neural    |
| **Scikit-learn** | 1.3.0+  | Pré-processamento e métricas |
| **Pandas**       | 2.0.0+  | Manipulação de dados         |
| **NumPy**        | 1.24.0+ | Computação numérica          |
| **Matplotlib**   | 3.7.0+  | Visualização de dados        |
| **Seaborn**      | 0.12.0+ | Visualização estatística     |

---

## 📚 Referências

- **Dataset:** Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). Heart Disease Data Set. UCI Machine Learning Repository.
- **Teoria:** Goodfellow, I., Bengio, Y., & Courville, A. (2016). _Deep Learning_. MIT Press.
- **Implementação:** Géron, A. (2019). _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_. O'Reilly Media.

---

## 👥 Autores

**Alexandre Pereira de Souza Junior**  
**João Pedro Castro das Virgens**  
**Leonardo Brandão do Amarante**  
**Mateus Rodrigues Cavalcante**  
**Vithor Junior da Encarnação Vitório**

**Instituição:** Universidade Federal do Amazonas (UFAM)  
**Disciplina:** Fundamentos de Inteligência Artificial (FIA)  
**Professor:** Edjard Mota  
**Período:** 2º Semestre de 2025

---

<div align="center">

**⭐ Se este projeto foi útil para seus estudos, considere dar uma estrela no repositório!**

Desenvolvido com dedicação para a disciplina de Fundamentos de IA 🧠❤️

</div>
