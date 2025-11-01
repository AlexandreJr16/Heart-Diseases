# 🫀 Classificador de Doenças Cardíacas com Redes Neurais

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-API-red.svg)](https://keras.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-yellow.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Disciplina:** Fundamentos de Inteligência Artificial (FIA)  
> **Autor:** Alexandre Pereira de Souza Junior

---

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Contexto e Objetivo](#-contexto-e-objetivo)
- [Dataset](#-dataset)
- [Metodologia](#-metodologia)
- [Resultados](#-resultados)
- [Como Executar](#-como-executar)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Conclusões](#-conclusões)

---

## 🎯 Visão Geral

Este repositório contém o desenvolvimento de um **classificador binário de alta performance** para a detecção de doenças cardíacas, utilizando **Redes Neurais Artificiais (ANN)** implementadas com TensorFlow/Keras.

O projeto demonstra a aplicação de técnicas avançadas de Machine Learning e Deep Learning para resolver um problema crítico de saúde pública, alcançando **92.68% de acurácia** no conjunto de teste.

---

## 🏥 Contexto e Objetivo

As **doenças cardiovasculares** são a principal causa de morte no mundo, tornando a detecção precoce um desafio crítico para a saúde pública.

### Objetivo Principal

Construir e avaliar um modelo de classificação binária capaz de prever com alta precisão a **presença (1)** ou **ausência (0)** de doença cardíaca em um paciente, com base em um conjunto de atributos clínicos.

### Entregável

Jupyter Notebook contendo todo o processo de:

- ✅ Análise Exploratória de Dados (EDA)
- ✅ Pré-processamento e Feature Engineering
- ✅ Modelagem e Treinamento
- ✅ Avaliação Crítica dos Resultados

---

## 📊 Dataset

### Heart Disease UCI

Para este projeto, utilizamos o renomado **Heart Disease UCI Dataset**, disponibilizado via Kaggle.

- **Fonte:** [Heart Disease Dataset (Kaggle)](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Amostras:** 1.025 pacientes (após limpeza)
- **Atributos:** 13 características clínicas

#### Principais Features

| Feature    | Descrição                              |
| ---------- | -------------------------------------- |
| `age`      | Idade do paciente                      |
| `sex`      | Sexo (1 = masculino, 0 = feminino)     |
| `cp`       | Tipo de dor no peito (0-3)             |
| `trestbps` | Pressão arterial em repouso (mm Hg)    |
| `chol`     | Colesterol sérico (mg/dl)              |
| `fbs`      | Glicemia em jejum > 120 mg/dl          |
| `restecg`  | Resultados eletrocardiográficos        |
| `thalach`  | Frequência cardíaca máxima alcançada   |
| `exang`    | Angina induzida por exercício          |
| `oldpeak`  | Depressão de ST induzida por exercício |
| `slope`    | Inclinação do segmento ST de pico      |
| `ca`       | Número de vasos principais (0-3)       |
| `thal`     | Talassemia (1-3)                       |

#### Variável-Alvo (Target)

- **0:** Ausência de doença cardíaca
- **1:** Presença de doença cardíaca

---

## 🔬 Metodologia

O projeto foi estruturado em **quatro fases principais**, seguindo um pipeline rigoroso de Data Science.

### Fase 1️⃣: Análise Exploratória de Dados (EDA)

Antes de qualquer modelagem, uma análise detalhada foi conduzida para entender a natureza dos dados.

#### Principais Descobertas

✅ **Balanceamento Perfeito**

- 526 instâncias da classe '1' (doente)
- 499 instâncias da classe '0' (saudável)
- Validação da **Acurácia** como métrica confiável

✅ **Qualidade dos Dados**

- Dataset completo, **sem valores nulos**
- Não exigiu etapas de imputation
- Pronto para modelagem após scaling

### Fase 2️⃣: Pré-Processamento e Prevenção de Data Leakage

Esta foi a etapa técnica mais crítica do projeto.

#### Divisão de Dados

```python
Train: 80% | Test: 20%
Stratified Split (mantém proporção das classes)
```

#### Normalização (StandardScaler)

**Por que é crucial?**  
Redes Neurais são altamente sensíveis a características em escalas diferentes:

- `age`: 29-77
- `chol`: 126-564

**Metodologia Rigorosa para Prevenir Data Leakage:**

```python
# ✅ CORRETO: Fit apenas no treino
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ❌ ERRADO: Fit em todos os dados (causa data leakage)
scaler.fit(X)  # NÃO FAZER ISSO!
```

### Fase 3️⃣: Arquitetura e Treinamento do Modelo

#### Arquitetura da Rede Neural

```
Input Layer (13 features)
        ↓
Dense(16, ReLU) + L2 Regularization
        ↓
Dropout(0.25)
        ↓
Dense(8, ReLU) + L2 Regularization
        ↓
Dropout(0.25)
        ↓
Output(1, Sigmoid) → Probabilidade [0, 1]
```

#### Configuração de Treinamento

| Parâmetro          | Valor                       |
| ------------------ | --------------------------- |
| **Optimizer**      | Adam                        |
| **Loss Function**  | Binary Crossentropy         |
| **Epochs**         | 100                         |
| **Batch Size**     | 10                          |
| **Regularization** | L2 (0.001) + Dropout (0.25) |

#### Técnicas de Regularização

- **Dropout:** Previne overfitting desativando aleatoriamente 25% dos neurônios
- **L2 Regularization:** Penaliza pesos grandes, promovendo generalização
- **Validation Split:** Monitoramento contínuo da performance no teste

---

## 📈 Resultados

### 🎯 Performance Geral

```
Acurácia no Conjunto de Teste: 92.68%
```

Isso significa que o modelo classificou corretamente **quase 93 de cada 100 pacientes** no conjunto de teste.

### 🏥 Análise da Matriz de Confusão

> **Importante:** Em problemas médicos, a acurácia por si só não é suficiente.  
> O custo de um **Falso Negativo** (paciente doente diagnosticado como saudável) é muito maior que o de um **Falso Positivo**.

#### Matriz de Confusão

|                        | **Previsto: Saudável (0)** | **Previsto: Doente (1)** |
| ---------------------- | -------------------------- | ------------------------ |
| **Real: Saudável (0)** | 93 (TN) ✅                 | 7 (FP) ⚠️                |
| **Real: Doente (1)**   | 8 (FN) ❌                  | 97 (TP) ✅               |

#### Métricas Detalhadas por Classe

| Classe           | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| **Saudável (0)** | 92%       | 93%    | 93%      | 100     |
| **Doente (1)**   | 93%       | 92%    | 93%      | 105     |

### 🔍 Análise Crítica

#### ✅ Pontos Fortes

1. **Recall (Sensibilidade) - Classe Doente: 92%**

   - O modelo identificou corretamente **97 dos 105 pacientes doentes**
   - Métrica crucial para aplicações médicas

2. **Equilíbrio entre Precision e Recall**

   - Ambas as métricas > 92% para as duas classes
   - Modelo balanceado e confiável

3. **Baixa Taxa de Falsos Positivos**
   - Apenas 7 pacientes saudáveis classificados como doentes
   - Evita exames desnecessários

#### ⚠️ Pontos de Atenção

1. **Falsos Negativos: 8 casos**
   - Este é o erro mais crítico
   - 8 pacientes doentes foram classificados como saudáveis
   - Em produção, seria necessário um segundo nível de validação

### 📊 Curvas de Aprendizado

O treinamento por 100 épocas mostrou:

- ✅ Excelente convergência
- ✅ Sem sinais de overfitting
- ✅ Acurácia de validação acompanhando (e até superando) a de treino

---

## 🚀 Como Executar

### Pré-requisitos

- Python 3.8 ou superior
- pip instalado

### 1️⃣ Clone o Repositório

```bash
git clone https://github.com/AlexandreJr16/Heart-Diseases.git
cd Heart-Diseases
```

### 2️⃣ Instale as Dependências

```bash
pip install pandas numpy tensorflow scikit-learn matplotlib seaborn
```

Ou usando um arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3️⃣ Execute o Notebook

Abra o Jupyter Notebook em um ambiente de sua escolha:

**VS Code:**

```bash
code heart-diseases.ipynb
```

**Jupyter Lab:**

```bash
jupyter lab heart-diseases.ipynb
```

**Google Colab:**

- Faça upload do arquivo `.ipynb` e `heart.csv`

---

## 🛠️ Tecnologias Utilizadas

### Core Libraries

| Biblioteca       | Versão | Propósito                    |
| ---------------- | ------ | ---------------------------- |
| **Python**       | 3.8+   | Linguagem base               |
| **TensorFlow**   | 2.0+   | Framework de Deep Learning   |
| **Keras**        | API    | Construção da Rede Neural    |
| **Scikit-learn** | Latest | Pré-processamento e métricas |
| **Pandas**       | Latest | Manipulação de dados         |
| **NumPy**        | Latest | Computação numérica          |
| **Matplotlib**   | Latest | Visualização de dados        |
| **Seaborn**      | Latest | Visualização estatística     |

---

## 💡 Conclusões

### Principais Aprendizados

1. **Performance Excepcional**

   - O modelo de Rede Neural Artificial alcançou **92.68% de acurácia**
   - Superou as expectativas iniciais do projeto

2. **Importância da Normalização**

   - Sem StandardScaler, características com escalas maiores (como `chol`) teriam dominado o aprendizado
   - Padronização foi crucial para treinamento estável e eficiente

3. **Prevenção de Data Leakage**

   - A metodologia rigorosa de fit/transform garantiu a integridade do modelo
   - Sem data leakage, os resultados refletem a verdadeira capacidade de generalização

4. **Métricas Além da Acurácia**
   - A análise da matriz de confusão revelou insights críticos
   - **Recall de 92%** para pacientes doentes é o resultado mais importante

### Aplicabilidade Clínica

Este modelo poderia ser usado como:

- 🏥 **Ferramenta de triagem inicial** em unidades de saúde
- 🔍 **Sistema de apoio à decisão** para médicos
- 📊 **Identificador de pacientes de risco** para exames complementares

### Próximos Passos

- [ ] Implementar validação cruzada (K-Fold)
- [ ] Testar arquiteturas mais profundas
- [ ] Aplicar técnicas de ensemble (Random Forest, XGBoost)
- [ ] Analisar feature importance com SHAP values
- [ ] Desenvolver API REST para deploy do modelo

---

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## 👤 Autor

**Alexandre Pereira de Souza Junior**

- GitHub: [@AlexandreJr16](https://github.com/AlexandreJr16)
- Projeto: [Heart-Diseases](https://github.com/AlexandreJr16/Heart-Diseases)

---

## 🙏 Agradecimentos

- Dataset fornecido pela comunidade UCI Machine Learning Repository
- Disponibilizado via Kaggle
- Disciplina de Fundamentos de Inteligência Artificial (FIA)

---

<div align="center">

**⭐ Se este projeto foi útil, considere dar uma estrela!**

Desenvolvido com ❤️ e ☕ por Alexandre Jr.

</div>
