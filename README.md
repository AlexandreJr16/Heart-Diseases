# Projeto 1: Classificação de Doenças Cardíacas# Projeto 1: Classificação de Doenças Cardíacas# Projeto 1: Classificação de Doenças Cardíacas - Fundamentos de IA# Projeto 1: Classificação de Doenças Cardíacas - Fundamentos de IA

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)

[![Keras](https://img.shields.io/badge/Keras-API-red.svg)](https://keras.io/)[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)

[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-yellow.svg)](https://scikit-learn.org/)

[![Keras](https://img.shields.io/badge/Keras-API-red.svg)](https://keras.io/)[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)

> **Disciplina:** Fundamentos de Inteligência Artificial (FIA)

> **Autores:** Alexandre Pereira de Souza Junior, Leonardo Brandão, Vithor Vitório [![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-yellow.svg)](https://scikit-learn.org/)

> **Instituição:** Universidade Federal de Alagoas (UFAL)

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)[![Keras](https://img.shields.io/badge/Keras-API-red.svg)](https://keras.io/)[![Keras](https://img.shields.io/badge/Keras-API-red.svg)](https://keras.io/)

---

> **Disciplina:** Fundamentos de Inteligência Artificial (FIA) [![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-yellow.svg)](https://scikit-learn.org/)[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-yellow.svg)](https://scikit-learn.org/)

## 📋 Sumário

> **Autores:** Alexandre Pereira de Souza Junior, Leonardo Brandão, Vithor Vitório

- [Descrição do Projeto](#-descrição-do-projeto)

- [Análise do Dataset](#-análise-do-dataset)> **Instituição:** Universidade Federal de Alagoas (UFAL)[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

- [Metodologia](#-metodologia)

- [Resultados Obtidos](#-resultados-obtidos)---> **Disciplina:** Fundamentos de Inteligência Artificial (FIA) > **Disciplina:** Fundamentos de Inteligência Artificial (FIA)

- [Conclusões](#-conclusões)

- [Como Executar](#-como-executar)## 📋 Sumário> **Autor:** Alexandre Pereira de Souza Junior, Leonardo Brandão, Vithor Vitorio. > **Autor:** Alexandre Pereira de Souza Junior, Leonardo Brandão, Vithor Vitorio.

- [Tecnologias Utilizadas](#-tecnologias-utilizadas)

- [Contexto e Objetivo](#-contexto-e-objetivo)---

---

- [Análise do Dataset](#-análise-do-dataset)

## 📖 Descrição do Projeto

- [Metodologia](#-metodologia)## 📋 Índice## 📋 Índice

### Contexto

- [Resultados](#-resultados)

As **doenças cardiovasculares** são a principal causa de morte em todo o mundo, tornando a detecção precoce fundamental para salvar vidas. Este projeto aplica técnicas de **Deep Learning** para auxiliar na identificação de pacientes com risco de doença cardíaca.

- [Conclusões](#-conclusões)- [Contexto do Problema](#-contexto-do-problema)- [Contexto do Problema](#-contexto-do-problema)

### Objetivo

- [Como Executar](#-como-executar)

Desenvolver um **classificador binário** utilizando Redes Neurais Artificiais (ANN) para prever a **presença (1)** ou **ausência (0)** de doença cardíaca em pacientes, com base em 13 atributos clínicos.

- [Tecnologias Utilizadas](#️-tecnologias-utilizadas)- [Dataset: Origem, Estrutura e Limpeza](#-dataset-origem-estrutura-e-limpeza)- [Dataset: Origem, Estrutura e Limpeza](#-dataset-origem-estrutura-e-limpeza)

### Especificações Técnicas

- [Referências](#-referências)

- **Tipo:** Classificação Binária Supervisionada

- **Modelo:** Rede Neural Feedforward com 2 camadas ocultas- [Metodologia](#️-metodologia)- [Metodologia](#️-metodologia)

- **Ativações:** ReLU (camadas ocultas), Sigmoid (saída)

- **Regularização:** Dropout (25%) + L2 (0.001)---

- **Métricas:** Acurácia, Precisão, Recall e Matriz de Confusão

- [Resultados e Análise Crítica](#-resultados-e-análise-crítica)- [Resultados e Análise Crítica](#-resultados-e-análise-crítica)

---

## 🎯 Contexto e Objetivo

## 📊 Análise do Dataset

- [Conclusão](#-conclusão)- [Conclusão](#-conclusão)

### Fonte de Dados

As **doenças cardiovasculares** são a principal causa de morte em todo o mundo, tornando a detecção precoce um desafio crítico para a saúde pública. A identificação precoce de pacientes em risco pode salvar vidas através de intervenções preventivas e tratamentos adequados.

**Dataset:** Heart Disease UCI (Cleveland Heart Disease Database)

**Repositório:** UCI Machine Learning Repository - [Instruções de Execução](#-instruções-de-execução)- [Instruções de Execução](#-instruções-de-execução)

**URL:** http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data

### Objetivo do Projeto

### Características

- [Referências](#-referências)- [Referências](#-referências)

- **Total de Amostras:** 303 pacientes

- **Amostras Válidas:** 297 (após limpeza de 6 linhas com valores nulos)Desenvolver um **modelo de classificação binária** baseado em Redes Neurais Artificiais (ANN) capaz de prever a **presença (1)** ou **ausência (0)** de doença cardíaca em um paciente, utilizando 13 atributos clínicos como entrada.

- **Features:** 13 atributos clínicos

- **Target:** Binário (0=saudável, 1=doente)---

- **Divisão:** 80% treino (237 amostras) | 20% teste (60 amostras)

### Especificações Técnicas

### Atributos Clínicos

## 📋 Contexto do Problema## 📋 Contexto do Problema

| Feature | Descrição |

|------------|------------------------------------------------|- **Tipo de Problema:** Classificação Binária Supervisionada

| `age` | Idade do paciente |

| `sex` | Sexo (1=masculino, 0=feminino) |- **Modelo:** Rede Neural Feedforward (2-3 camadas ocultas)Este projeto acadêmico foi desenvolvido como parte da disciplina de Fundamentos de Inteligência Artificial e tem como objetivo construir um **classificador binário** para predição de doenças cardíacas. O modelo desenvolvido classifica pacientes em duas categorias:Este projeto acadêmico foi desenvolvido como parte da disciplina de Fundamentos de Inteligência Artificial e tem como objetivo construir um **classificador binário** para predição de doenças cardíacas. O modelo desenvolvido classifica pacientes em duas categorias:

| `cp` | Tipo de dor no peito (0-3) |

| `trestbps` | Pressão arterial em repouso (mm Hg) |- **Ativação:** ReLU nas camadas ocultas, Sigmoid na saída

| `chol` | Colesterol sérico (mg/dl) |

| `fbs` | Glicemia em jejum > 120 mg/dl |- **Regularização:** Dropout para prevenção de overfitting- **0 (Saudável)**: Ausência de doença cardíaca- **0:** Ausência de doença cardíaca (Saudável)

| `restecg` | Resultados eletrocardiográficos |

| `thalach` | Frequência cardíaca máxima alcançada |- **Métricas de Avaliação:** Acurácia, Precisão, Recall e Matriz de Confusão

| `exang` | Angina induzida por exercício |

| `oldpeak` | Depressão de ST induzida por exercício |- **1 (Doente)**: Presença de doença cardíaca- **1:** Presença de doença cardíaca (Doente)

| `slope` | Inclinação do segmento ST |

| `ca` | Número de vasos principais (0-3) |---

| `thal` | Talassemia (defeito cardíaco) |

A abordagem utiliza técnicas de **Deep Learning** para analisar 13 atributos clínicos e fisiológicos de pacientes, construindo uma Rede Neural Artificial (ANN) feedforward capaz de realizar predições com base em dados históricos.---

### Pré-processamento

## 📊 Análise do Dataset

1. **Limpeza:** Remoção de 6 linhas com valores nulos (303 → 297 amostras)

2. **Transformação:** Conversão do target multi-classe (0-4) para binário (0-1)---## 🛠️ Metodologia

3. **Divisão Estratificada:** 80/20 com manutenção da proporção de classes

4. **Normalização:** StandardScaler aplicado após divisão treino/teste### Fonte de Dados

---## 🔬 Dataset: Origem, Estrutura e LimpezaO projeto foi estruturado em **cinco fases principais**, seguindo um pipeline rigoroso de Data Science para garantir a validade e a replicabilidade dos resultados.

## 🧠 Metodologia**Dataset:** Heart Disease UCI (Cleveland Heart Disease Database)

### Arquitetura da Rede Neural**Repositório:** UCI Machine Learning Repository ### Fonte de Dados### Fase 1️⃣: Análise Exploratória de Dados (EDA)

`````**URL Original:** http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data

Input Layer (13 features)

    ↓**Disponível também em:** [Kaggle - Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)O dataset utilizado é o clássico **Cleveland Heart Disease Database** do repositório UCI Machine Learning, acessível via:Antes de qualquer modelagem, uma análise detalhada foi conduzida para entender a natureza dos dados:

Dense(16 neurônios) + ReLU + L2(0.001) + Dropout(25%)

    ↓### Características do Dataset````- **Balanceamento de Classes**: Verificação da distribuição entre pacientes saudáveis e doentes

Dense(8 neurônios) + ReLU + L2(0.001) + Dropout(25%)

    ↓- **Amostras Totais:** 303 pacienteshttp://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data- **Matriz de Correlação**: Identificação de relações lineares entre as features

Output(1 neurônio) + Sigmoid → Probabilidade [0, 1]

```- **Amostras Utilizadas:** 297 (após limpeza de valores nulos)



### Configuração de Treinamento- **Atributos:** 13 features clínicas + 1 variável target```- **Estatísticas Descritivas**: Compreensão da distribuição de cada atributo clínico



| Parâmetro          | Valor                      |- **Divisão:** 80% treino (237 amostras) | 20% teste (60 amostras)

|--------------------|----------------------------|

| **Optimizer**      | Adam                       |### Atributos Clínicos (Features)

| **Loss Function**  | Binary Crossentropy        |

| **Epochs**         | 100                        |**Nota Importante sobre a Escolha do Dataset**: Durante a fase inicial do projeto, identificamos uma discrepância entre o dataset sugerido no material de apoio (Kaggle, 1025 amostras) e o dataset utilizado no notebook de referência do professor. Após análise crítica, confirmamos que o dataset correto para este projeto é o **UCI Cleveland original (303 amostras)**, que representa o benchmark histórico para pesquisas em classificação de doenças cardíacas.### Fase 2️⃣: Pré-processamento e Prevenção de Data Leakage

| **Batch Size**     | 10                         |

| **Regularization** | L2 (0.001) + Dropout (25%) || Feature | Descrição | Escala |



### Importância da Normalização|------------|------------------------------------------------|-------------|



A normalização dos dados é **crítica** para o sucesso de Redes Neurais:| `age` | Idade do paciente | 29-77 |



**Por que normalizar?**| `sex` | Sexo (1=masculino, 0=feminino) | 0-1 |### Estrutura do DatasetEsta foi a etapa técnica **mais crítica** do projeto, onde seguimos rigorosamente as melhores práticas de Machine Learning.

- Features possuem escalas muito diferentes (ex: `chol`: 126-564 vs `sex`: 0-1)

- Sem normalização, atributos de maior magnitude dominam o gradiente| `cp` | Tipo de dor no peito (0-3) | 0-3 |

- Dados normalizados permitem convergência mais rápida e estável

| `trestbps` | Pressão arterial em repouso (mm Hg) | ~94-200 |

**Prevenção de Data Leakage:**

- **Ordem CORRETA:** Dividir dados → Fit no treino → Transform em treino e teste| `chol` | Colesterol sérico (mg/dl) | ~126-564 |

- **Ordem ERRADA:** Normalizar tudo → Dividir (causa vazamento de informação do teste)

| `fbs` | Glicemia em jejum > 120 mg/dl (1=sim, 0=não) | 0-1 |- **Amostras Originais**: 303 pacientes#### Pipeline de Pré-processamento

```python

# ✅ CORRETO| `restecg` | Resultados eletrocardiográficos em repouso | 0-2 |

X_train, X_test = train_test_split(X, y)

scaler.fit(X_train)                    # Aprende apenas do treino| `thalach` | Frequência cardíaca máxima alcançada | ~71-202 |- **Atributos**: 13 features clínicas + 1 variável target

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)| `exang` | Angina induzida por exercício (1=sim, 0=não) | 0-1 |



# ❌ ERRADO| `oldpeak` | Depressão de ST induzida por exercício | 0-6.2 |- **Features Incluem**: Idade, sexo, tipo de dor no peito (cp), pressão arterial em repouso (trestbps), colesterol sérico (chol), glicemia em jejum (fbs), resultados de ECG em repouso (restecg), frequência cardíaca máxima (thalach), angina induzida por exercício (exang), depressão ST (oldpeak), inclinação do segmento ST (slope), número de vasos principais (ca), e talassemia (thal).1. **Separação de Features e Target**:

scaler.fit(X)                          # Vaza informação do teste

X_scaled = scaler.transform(X)| `slope` | Inclinação do segmento ST no pico de exercício | 0-2 |

X_train, X_test = train_test_split(X_scaled)

```| `ca` | Número de vasos principais coloridos (0-3) | 0-3 | ```python



---| `thal` | Talassemia (1=normal, 2=defeito fixo, 3=reversível) | 1-3 |



## 📈 Resultados Obtidos#### Principais Features X = data.drop('target', axis=1) # 13 features



### Métricas de Performance### Variável-Alvo (Target)



| Métrica                | Valor   |y = data['target'] # variável binária

|------------------------|---------|

| **Acurácia Global**    | 83.33%  |- **0:** Ausência de doença cardíaca (Saudável)

| **Precisão (Doente)**  | 84.6%   |

| **Recall (Doente)**    | 78.6%   |- **1:** Presença de doença cardíaca (Doente)| Feature | Descrição | ```

| **F1-Score**           | 0.81    |

_Nota: O dataset original possui target multi-classe (0-4). Foi realizada conversão para binário: 0 permanece 0, valores >0 foram convertidos para 1._| ---------- | -------------------------------------- |

### Matriz de Confusão

### Pré-processamento Aplicado| `age` | Idade do paciente |2. **Divisão Estratificada Train/Test**:

`````

                    Predito: Saudável    Predito: Doente1. **Tratamento de Valores Ausentes**| `sex` | Sexo (1 = masculino, 0 = feminino) | ```python

Real: Saudável 26 4

Real: Doente 6 24 - Identificação de valores nulos marcados como `'?'` no dataset original

````

   - Remoção de 6 linhas com dados faltantes (303 → 297 amostras)| `cp` | Tipo de dor no peito (0-3) | train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

**Interpretação:**

- **26 Verdadeiros Negativos:** Pacientes saudáveis corretamente identificados2. **Transformação do Target**| `trestbps` | Pressão arterial em repouso (mm Hg) | ```

- **24 Verdadeiros Positivos:** Pacientes doentes corretamente identificados

- **4 Falsos Positivos:** Saudáveis classificados como doentes (exames extras)   - Conversão de multi-classe (0, 1, 2, 3, 4) para binário (0, 1)

- **6 Falsos Negativos:** Doentes classificados como saudáveis ⚠️ (mais crítico)

   - Aplicação: `lambda x: 1 if x > 0 else 0`| `chol` | Colesterol sérico (mg/dl) | - **Conjunto de Treino**: 237 amostras (80%)

### Análise no Contexto Médico

3. **Divisão Estratificada**| `fbs` | Glicemia em jejum > 120 mg/dl | - **Conjunto de Teste**: 60 amostras (20%)

**Recall de 78.6%:** O modelo detecta aproximadamente 4 em cada 5 pacientes doentes.

   - Split 80/20 com estratificação para manter proporção de classes

**Falsos Negativos (6 casos):** Representa o erro mais crítico em medicina, pois pacientes doentes não receberiam tratamento. Para uso clínico real, seria necessário:

- Ajustar o threshold de decisão (reduzir de 0.5 para ~0.3-0.4)   - `stratify=y` no `train_test_split()`| `restecg` | Resultados eletrocardiográficos | - **Estratificação**: Mantém a proporção de classes em ambos os conjuntos

- Combinar com avaliação médica profissional

- Usar como ferramenta de triagem, não diagnóstico definitivo4. **Normalização (StandardScaler)**| `thalach` | Frequência cardíaca máxima alcançada |



**Falsos Positivos (4 casos):** Erro menos perigoso, resultando em exames complementares desnecessários, mas sem risco à vida.   - **CRÍTICO:** Aplicado **APÓS** a divisão treino/teste para evitar data leakage



### Comportamento do Treinamento   - Fit realizado exclusivamente nos dados de treino| `exang` | Angina induzida por exercício |3. **Normalização (StandardScaler)** - **PONTO CRÍTICO**:



- **Convergência:** Estável nas primeiras 30-40 épocas   - Transform aplicado independentemente em treino e teste

- **Overfitting:** Detectado após ~40 épocas (esperado para dataset pequeno)

- **Regularização:** Dropout + L2 limitaram efetivamente o overfitting| `oldpeak` | Depressão de ST induzida por exercício | ```python



------



## 💡 Conclusões| `slope` | Inclinação do segmento ST de pico | scaler = StandardScaler()



### Eficácia do Modelo## 🧠 Metodologia



O modelo **cumpriu todos os requisitos** estabelecidos:| `ca` | Número de vasos principais (0-3) | X_train_scaled = scaler.fit_transform(X_train) # Fit APENAS no treino



✅ Rede Neural com 2 camadas ocultas (ReLU) e regularização Dropout  ### Arquitetura da Rede Neural

✅ Classificação binária com saída Sigmoid

✅ Acurácia de 83.33% (realista para 297 amostras)  | `thal` | Talassemia (1-3) | X_test_scaled = scaler.transform(X_test) # Transform no teste

✅ Métricas completas: Precisão, Recall e Matriz de Confusão

O modelo implementado segue uma arquitetura feedforward com as seguintes especificações:

### Importância da Normalização dos Dados

````

A normalização foi **essencial** em três aspectos:

````

1. **Convergência:** Permitiu treinamento eficiente em ~30 épocas. Sem normalização, features de alta magnitude (`chol`, `trestbps`) dominariam o gradiente, impedindo convergência.

Input Layer (13 neurônios - features de entrada)### Etapas de Limpeza

2. **Prevenção de Data Leakage:** A ordem correta (Split → Fit → Transform) garantiu que estatísticas do teste não influenciassem o treino, simulando corretamente um cenário de produção.

    ↓

3. **Contribuição Balanceada:** Todas as 13 features contribuíram equilibradamente. Sem normalização, features binárias (`sex`, `fbs`) seriam ignoradas.

Dense Layer 1: 16 neurônios#### ⚠️ Importância da Normalização dos Dados

### Performance Contextualizada

    - Ativação: ReLU

- **83.33% de acurácia** é apropriado para um dataset de 297 amostras

- Performance competitiva com estudos acadêmicos usando o mesmo dataset UCI    - Regularização: L2 (λ=0.001)1. **Tratamento de Valores Ausentes**: O dataset original continha valores nulos representados pelo caractere `'?'`. Esses valores foram identificados durante a carga dos dados utilizando o parâmetro `na_values='?'` do pandas.

- O overfitting observado (~40 épocas) é esperado e foi adequadamente controlado

    - Dropout: 25%

### Aplicabilidade Clínica

    ↓A normalização dos dados revelou-se **absolutamente essencial** para o sucesso do projeto:

**Uso Recomendado:**

- 🏥 Ferramenta de triagem inicial em unidades de saúdeDense Layer 2: 8 neurônios

- 🔍 Sistema de apoio à decisão para médicos

- 📊 Identificação de pacientes de risco para exames complementares    - Ativação: ReLU2. **Remoção de Amostras Incompletas**: Aplicamos `dropna()` para remover todas as linhas com valores ausentes, resultando em **297 amostras válidas** para análise.



**Limitações:**    - Regularização: L2 (λ=0.001)

- Não substitui diagnóstico médico profissional

- Requer validação em datasets externos maiores    - Dropout: 25%**Por que normalizar?**

- Threshold de decisão deve ser ajustado para maximizar Recall

    ↓

---

Output Layer: 1 neurônio3. **Transformação da Variável Target**: A variável-alvo original era multi-classe (0, 1, 2, 3, 4), representando diferentes níveis de severidade da doença. Convertemos para um problema binário aplicando a transformação:- Redes Neurais são altamente sensíveis a características em escalas diferentes

## 🚀 Como Executar

    - Ativação: Sigmoid

### Pré-requisitos

    - Saída: Probabilidade [0, 1]   ```python- Features como `chol`(126-564) dominariam features como`sex` (0-1) sem normalização

- Python 3.8+

- Jupyter Notebook ou JupyterLab```



### Instalação   target_binário = 1 if target_original > 0 else 0- A convergência do gradiente descendente é muito mais eficiente com dados normalizados



**1. Clone o repositório:**### Configuração de Treinamento

```bash

git clone https://github.com/AlexandreJr16/Heart-Diseases.git   ```

cd Heart-Diseases

```| Hiperparâmetro        | Valor                      | Justificativa                                          |



**2. Instale as dependências:**|-----------------------|----------------------------|--------------------------------------------------------|   ```

```bash

pip install -r requirements.txt| **Optimizer**         | Adam                       | Convergência adaptativa, eficiente para ANNs           |

````

| **Loss Function** | Binary Crossentropy | Padrão para classificação binária |**Por que esta ordem é crucial?**

**3. Execute o notebook:**

`````bash| **Epochs**            | 100                        | Permite observação de overfitting                      |

jupyter notebook heart-diseases.ipynb

```| **Batch Size**        | 10                         | Balanceamento entre estabilidade e ruído do gradiente  |#### Variável-Alvo (Target)- Realizar o scaling **antes** da divisão train/test causaria **data leakage**



**4. Execute as células sequencialmente** (Shift + Enter) ou todas de uma vez (Cell → Run All)| **L2 Regularization** | 0.001                      | Penaliza pesos grandes, promove generalização          |



---| **Dropout Rate**      | 0.25 (25%)                 | Desativa neurônios aleatoriamente, previne overfitting |- Informações estatísticas do conjunto de teste (média e desvio padrão) "vazariam" para o conjunto de treino



## 🛠 Tecnologias Utilizadas| **Validation Data**   | Test set (60 amostras)     | Monitoramento contínuo durante treinamento             |



| Tecnologia       | Versão  | Função                          |- **0:** Ausência de doença cardíaca (Saudável)- O scaler deve aprender os parâmetros **exclusivamente** dos dados de treino

|------------------|---------|----------------------------------|

| **Python**       | 3.8+    | Linguagem de programação         |### Justificativas Técnicas

| **TensorFlow**   | 2.13.0+ | Framework de Deep Learning       |

| **Keras**        | API     | Construção da Rede Neural        |- **1:** Presença de doença cardíaca (Doente)- Esta prática simula o cenário real de produção, onde novos dados nunca foram vistos durante o treinamento

| **Scikit-learn** | 1.3.0+  | Pré-processamento e métricas     |

| **Pandas**       | 2.0.0+  | Manipulação de dados             |**1. Função de Ativação ReLU**

| **NumPy**        | 1.24.0+ | Computação numérica              |

| **Matplotlib**   | 3.7.0+  | Visualização de dados            |- Evita vanishing gradient---### Fase 3️⃣: Construção do Modelo (ANN)

| **Seaborn**      | 0.12.0+ | Visualização estatística         |

- Computacionalmente eficiente

---

- Boa performance em problemas de classificação## 🛠️ MetodologiaDesenvolvemos uma Rede Neural Artificial feedforward com a seguinte arquitetura:

## 📚 Referências



- **Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988).** Heart Disease Data Set. UCI Machine Learning Repository.

- **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** Deep Learning. MIT Press.**2. Regularização Combinada (L2 + Dropout)**O projeto foi estruturado em **cinco fases principais**, seguindo um pipeline rigoroso de Data Science para garantir a validade e a replicabilidade dos resultados.```

- **Géron, A. (2019).** Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. O'Reilly Media.

- **L2:** Penaliza pesos elevados, força distribuição mais suave

---

- **Dropout:** Desativa 25% dos neurônios a cada iteração, reduz co-adaptaçãoCamada de Entrada: 13 neurônios (features)

## 👥 Autores



**Alexandre Pereira de Souza Junior**

**Leonardo Brandão**  **3. Sigmoid na Saída**### Fase 1️⃣: Análise Exploratória de Dados (EDA) ↓

**Vithor Vitório**

- Comprime saída para intervalo [0, 1]

**Instituição:** Universidade Federal de Alagoas (UFAL)

**Disciplina:** Fundamentos de Inteligência Artificial (FIA)  - Interpretável como probabilidade de doençaCamada Oculta 1: 16 neurônios

**Professor:** Edjard Mota

- Threshold de decisão em 0.5 (pode ser ajustado)

---

Antes de qualquer modelagem, uma análise detalhada foi conduzida para entender a natureza dos dados: - Ativação: ReLU

<div align="center">

### Importância da Normalização dos Dados

**Desenvolvido para a disciplina de Fundamentos de IA - 2025** 🧠❤️

    - Regularização: L2 (lambda=0.001)

</div>

A normalização é **absolutamente crítica** para o sucesso de Redes Neurais:

- **Balanceamento de Classes**: Verificação da distribuição entre pacientes saudáveis e doentes - Dropout: 25%

#### Por que Normalizar?

- **Matriz de Correlação**: Identificação de relações lineares entre as features ↓

- **Escalas Divergentes:** Features possuem magnitudes muito diferentes (`chol`: 126-564 vs `sex`: 0-1)

- **Dominância de Features:** Sem normalização, atributos com valores maiores dominam o cálculo do gradiente- **Estatísticas Descritivas**: Compreensão da distribuição de cada atributo clínicoCamada Oculta 2: 8 neurônios

- **Convergência:** Dados normalizados permitem convergência muito mais rápida e estável

  - Ativação: ReLU

#### Por que a Ordem Importa?

### Fase 2️⃣: Pré-processamento e Prevenção de Data Leakage - Regularização: L2 (lambda=0.001)

**⚠️ PREVENÇÃO DE DATA LEAKAGE**

    - Dropout: 25%

**Ordem CORRETA:**

```pythonEsta foi a etapa técnica **mais crítica** do projeto, onde seguimos rigorosamente as melhores práticas de Machine Learning. ↓

1. Split treino/teste

2. scaler.fit_transform(X_train)    # Aprende média/std do TREINOCamada de Saída: 1 neurônio

3. scaler.transform(X_test)          # Aplica parâmetros do TREINO no TESTE

```#### Pipeline de Pré-processamento - Ativação: Sigmoid (probabilidade de doença)



**Ordem ERRADA (causa data leakage):**````

```python

1. scaler.fit_transform(X_completo)  # ❌ Informação do teste vaza para treino1. **Separação de Features e Target**:

2. Split treino/teste

```   ```python#### Configuração de Treinamento



**Consequência do Data Leakage:**   X = data.drop('target', axis=1)  # 13 features

- Estatísticas do teste (média, desvio padrão) influenciam a normalização do treino

- Modelo tem acesso indireto a informações que só veria na produção   y = data['target']               # variável binária| Parâmetro          | Valor                       |

- Resultados são otimistas e não generalizam para dados reais

   ```| ------------------ | --------------------------- |

---

| **Optimizer**      | Adam                        |

## 📈 Resultados

2. **Divisão Estratificada Train/Test**:| **Loss Function**  | Binary Crossentropy         |

### Métricas de Performance

   ```python| **Epochs**         | 100                         |

| Métrica                  | Valor   | Interpretação                                        |

|--------------------------|---------|------------------------------------------------------|   train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)| **Batch Size**     | 10                          |

| **Acurácia Global**      | 83.33%  | 50 de 60 pacientes corretamente classificados        |

| **Precisão (Doente)**    | 84.6%   | Quando prevê doença, está correto em 84.6% dos casos|   ```| **Regularization** | L2 (0.001) + Dropout (0.25) |

| **Recall (Doente)**      | 78.6%   | Detecta 78.6% dos pacientes realmente doentes        |

| **F1-Score (Doente)**    | 0.81    | Média harmônica entre Precisão e Recall              |   - **Conjunto de Treino**: 237 amostras (80%)| **Validação**      | Conjunto de teste           |



### Matriz de Confusão   - **Conjunto de Teste**: 60 amostras (20%)



```   - **Estratificação**: Mantém a proporção de classes em ambos os conjuntos#### Estratégia de Regularização

┌─────────────────────┬──────────────────┬──────────────────┐

│                     │ Previsto: Saudável│ Previsto: Doente│

├─────────────────────┼──────────────────┼──────────────────┤

│ Real: Saudável (30) │     26 (TN) ✅   │      4 (FP) ⚠️   │3. **Normalização (StandardScaler)** - **PONTO CRÍTICO**:- **L2 Regularization**: Penaliza pesos muito altos, promovendo uma distribuição mais suave dos pesos

│ Real: Doente (30)   │      6 (FN) ❌   │     24 (TP) ✅   │

└─────────────────────┴──────────────────┴──────────────────┘   ```python- **Dropout (25%)**: Durante o treino, desativa aleatoriamente 25% dos neurônios em cada camada oculta, forçando a rede a aprender representações mais robustas e reduzindo a dependência de neurônios específicos

`````

scaler = StandardScaler()

**Legenda:**

- **TN (True Negative):** 26 pacientes saudáveis corretamente identificados X_train_scaled = scaler.fit_transform(X_train) # Fit APENAS no treino### Fase 4️⃣: Treinamento e Análise de Overfitting

- **TP (True Positive):** 24 pacientes doentes corretamente identificados

- **FP (False Positive):** 4 pacientes saudáveis incorretamente classificados como doentes X_test_scaled = scaler.transform(X_test) # Transform no teste

- **FN (False Negative):** 6 pacientes doentes incorretamente classificados como saudáveis

  ```O modelo foi treinado por 100 épocas com monitoramento contínuo das métricas de treino e validação. Os gráficos de histórico revelaram um padrão clássico de **overfitting** após aproximadamente 30-40 épocas:

  ```

### Análise Crítica no Contexto Médico

#### ⚠️ Falsos Negativos: O Erro Mais Crítico

#### ⚠️ Importância da Normalização dos Dados- **Acurácia de Treino**: Continuou aumentando até ~90%

Em aplicações médicas, os **Falsos Negativos** (FN) representam o maior risco:

- **Acurácia de Validação**: Estagnou em ~83% e apresentou flutuações

- **6 pacientes doentes** foram classificados como saudáveis

- **Consequência:** Esses pacientes não receberiam tratamento adequadoA normalização dos dados revelou-se **absolutamente essencial** para o sucesso do projeto:- **Perda de Validação**: Começou a aumentar enquanto a perda de treino diminuía

- **Custo:** Potencialmente fatal - progressão da doença sem intervenção

#### ✅ Falsos Positivos: Erro Menos Perigoso

**Por que normalizar?\*\***Interpretação**: Este comportamento é **esperado e normal\*\* para um dataset pequeno (237 amostras de treino). As técnicas de regularização (Dropout + L2) foram eficazes em limitar o overfitting, mas não em eliminá-lo completamente.

- **4 pacientes saudáveis** foram classificados como doentes

- **Consequência:** Encaminhamento para exames complementares desnecessários- Redes Neurais são altamente sensíveis a características em escalas diferentes

- **Custo:** Financeiro e emocional, mas não fatal

- Features como `chol` (126-564) dominariam features como `sex` (0-1) sem normalização### Fase 5️⃣: Avaliação Final e Análise Crítica

#### Recall (Sensibilidade): Métrica Prioritária

- A convergência do gradiente descendente é muito mais eficiente com dados normalizados

- **Recall de 78.6%** significa que o modelo detecta aproximadamente 4 em cada 5 pacientes doentes

- Em triagem médica, é preferível maximizar o Recall (detectar o máximo de doentes)A avaliação final utilizou múltiplas métricas para fornecer uma visão completa da performance do modelo, com ênfase especial nas métricas mais relevantes para o contexto médico.

- **Trade-off:** Aumentar Recall geralmente reduz Precisão (mais falsos positivos)

**Por que esta ordem é crucial?**

#### Estratégias de Melhoria

- Realizar o scaling **antes** da divisão train/test causaria **data leakage**A abordagem utiliza técnicas de **Deep Learning** para analisar 13 atributos clínicos e fisiológicos de pacientes, construindo uma Rede Neural Artificial (ANN) feedforward capaz de realizar predições com base em dados históricos.

1. **Ajuste de Threshold:** Reduzir de 0.5 para ~0.3-0.4 aumentaria Recall

2. **Class Weights:** Penalizar mais erros na classe "doente"- Informações estatísticas do conjunto de teste (média e desvio padrão) "vazariam" para o conjunto de treino

3. **SMOTE:** Balanceamento sintético se houvesse desbalanceamento

4. **Ensemble Methods:** Combinar múltiplos modelos para decisão final- O scaler deve aprender os parâmetros **exclusivamente** dos dados de treino---

### Análise do Treinamento- Esta prática simula o cenário real de produção, onde novos dados nunca foram vistos durante o treinamento

**Observações dos Gráficos de Aprendizado:**## 🔬 Dataset: Origem, Estrutura e Limpeza

- **Convergência:** Estável nas primeiras 30-40 épocas### Fase 3️⃣: Construção do Modelo (ANN)

- **Overfitting Detectado:** Após ~40 épocas

  - Acurácia de treino continua subindo (~90%)### Fonte de Dados

  - Acurácia de validação estagna (~83%)

  - Perda de validação começa a aumentarDesenvolvemos uma Rede Neural Artificial feedforward com a seguinte arquitetura:

- **Eficácia da Regularização:** Dropout + L2 limitaram, mas não eliminaram o overfitting

O dataset utilizado é o clássico **Cleveland Heart Disease Database** do repositório UCI Machine Learning, acessível via:

**Interpretação:**

Este padrão é **esperado e normal** para datasets pequenos (237 amostras de treino). A regularização funcionou adequadamente, mas datasets maiores seriam necessários para eliminar completamente o overfitting.````

---Camada de Entrada: 13 neurônios (features)```

## 💡 Conclusões ↓http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data

### Eficácia do ModeloCamada Oculta 1: 16 neurônios```

O modelo desenvolvido **atendeu plenamente aos requisitos estabelecidos**: - Ativação: ReLU

✅ **Arquitetura Implementada:** Rede Neural Feedforward com 2 camadas ocultas (16-8 neurônios) - Regularização: L2 (lambda=0.001)**Nota Importante sobre a Escolha do Dataset**: Durante a fase inicial do projeto, identificamos uma discrepância entre o dataset sugerido no material de apoio (Kaggle, 1025 amostras) e o dataset utilizado no notebook de referência do professor. Após análise crítica, confirmamos que o dataset correto para este projeto é o **UCI Cleveland original (303 amostras)**, que representa o benchmark histórico para pesquisas em classificação de doenças cardíacas.

✅ **Ativações Corretas:** ReLU nas camadas ocultas, Sigmoid na saída

✅ **Regularização:** Dropout (25%) aplicado após cada camada oculta - Dropout: 25%

✅ **Classificação Binária:** Target convertido corretamente (0/1)

✅ **Métricas Completas:** Acurácia (83.3%), Precisão (84.6%), Recall (78.6%), Matriz de Confusão ↓### Estrutura do Dataset

### Performance no Contexto RealCamada Oculta 2: 8 neurônios

**Acurácia de 83.3%:** Resultado realista e apropriado para: - Ativação: ReLU- **Amostras Originais**: 303 pacientes

- Dataset pequeno (297 amostras)

- Problema complexo (diagnóstico médico) - Regularização: L2 (lambda=0.001)- **Atributos**: 13 features clínicas + 1 variável target

- Benchmark competitivo com estudos acadêmicos usando o mesmo dataset UCI

  - Dropout: 25%- **Features Incluem**: Idade, sexo, tipo de dor no peito (cp), pressão arterial em repouso (trestbps), colesterol sérico (chol), glicemia em jejum (fbs), resultados de ECG em repouso (restecg), frequência cardíaca máxima (thalach), angina induzida por exercício (exang), depressão ST (oldpeak), inclinação do segmento ST (slope), número de vasos principais (ca), e talassemia (thal).

**Comparação com Literatura:**

- Estudos similares no dataset UCI Cleveland reportam acurácias entre 80-85% ↓

- Datasets maiores (Kaggle, 1025 amostras) alcançam ~92-93%

Camada de Saída: 1 neurônio#### Principais Features

### Importância da Normalização dos Dados

    - Ativação: Sigmoid (probabilidade de doença)

A normalização revelou-se **essencial** em três dimensões:

`````| Feature    | Descrição                              |

**1. Convergência do Treinamento**

- Sem normalização: features de alta magnitude dominam o gradiente| ---------- | -------------------------------------- |

- Com normalização: convergência estável em ~30 épocas

- **Resultado:** Treinamento viável e eficiente#### Configuração de Treinamento| `age`      | Idade do paciente                      |



**2. Prevenção de Data Leakage**| `sex`      | Sexo (1 = masculino, 0 = feminino)     |

- Ordem correta: Split → Fit (treino) → Transform (treino e teste)

- Garante que estatísticas do teste não influenciem o treino| Parâmetro          | Valor                       || `cp`       | Tipo de dor no peito (0-3)             |

- **Resultado:** Modelo validado corretamente

| ------------------ | --------------------------- || `trestbps` | Pressão arterial em repouso (mm Hg)    |

**3. Contribuição Balanceada de Features**

- Todas as 13 features contribuem de forma equilibrada| **Optimizer**      | Adam                        || `chol`     | Colesterol sérico (mg/dl)              |

- Sem normalização: `chol`, `trestbps` dominariam completamente

- **Resultado:** Acurácia de 83.3% reflete aprendizado real| **Loss Function**  | Binary Crossentropy         || `fbs`      | Glicemia em jejum > 120 mg/dl          |



### Lições Aprendidas| **Epochs**         | 100                         || `restecg`  | Resultados eletrocardiográficos        |



1. **Datasets Pequenos Exigem Regularização Agressiva**| **Batch Size**     | 10                          || `thalach`  | Frequência cardíaca máxima alcançada   |

   - Dropout (25%) + L2 (0.001) foram eficazes mas não eliminaram overfitting

   - Early Stopping em ~40 épocas seria benéfico| **Regularization** | L2 (0.001) + Dropout (0.25) || `exang`    | Angina induzida por exercício          |



2. **Ordem das Operações é Crítica**| **Validação**      | Conjunto de teste           || `oldpeak`  | Depressão de ST induzida por exercício |

   - Data leakage invalida completamente os resultados

   - Pipeline correto: Split → Fit → Transform| `slope`    | Inclinação do segmento ST de pico      |



3. **Métricas Contextuais > Acurácia Global**#### Estratégia de Regularização| `ca`       | Número de vasos principais (0-3)       |

   - Em medicina, Recall é mais importante que Acurácia

   - Matriz de Confusão revela insights que métrica única não mostra| `thal`     | Talassemia (1-3)                       |



4. **Overfitting é Esperado, Não um Fracasso**- **L2 Regularization**: Penaliza pesos muito altos, promovendo uma distribuição mais suave dos pesos

   - Com 237 amostras de treino, overfitting após 40 épocas é inevitável

   - O importante é limitá-lo via regularização- **Dropout (25%)**: Durante o treino, desativa aleatoriamente 25% dos neurônios em cada camada oculta, forçando a rede a aprender representações mais robustas e reduzindo a dependência de neurônios específicos### Etapas de Limpeza



### Aplicabilidade Clínica



**Uso Recomendado:**### Fase 4️⃣: Treinamento e Análise de Overfitting1. **Tratamento de Valores Ausentes**: O dataset original continha valores nulos representados pelo caractere `'?'`. Esses valores foram identificados durante a carga dos dados utilizando o parâmetro `na_values='?'` do pandas.

- 🏥 Ferramenta de **triagem inicial** em unidades de saúde

- 🔍 **Sistema de apoio à decisão** para médicos (não diagnóstico definitivo)

- 📊 **Identificação de pacientes de risco** para exames complementares

O modelo foi treinado por 100 épocas com monitoramento contínuo das métricas de treino e validação. Os gráficos de histórico revelaram um padrão clássico de **overfitting** após aproximadamente 30-40 épocas:2. **Remoção de Amostras Incompletas**: Aplicamos `dropna()` para remover todas as linhas com valores ausentes, resultando em **297 amostras válidas** para análise.

**Limitações para Uso Clínico:**

- Requer validação em datasets externos maiores

- 6 Falsos Negativos (21% dos doentes) é alto para uso autônomo

- Deve ser combinado com avaliação médica profissional- **Acurácia de Treino**: Continuou aumentando até ~90%3. **Transformação da Variável Target**: A variável-alvo original era multi-classe (0, 1, 2, 3, 4), representando diferentes níveis de severidade da doença. Convertemos para um problema binário aplicando a transformação:



### Próximos Passos- **Acurácia de Validação**: Estagnou em ~83% e apresentou flutuações   ```python



- [ ] Implementar K-Fold Cross-Validation para resultados mais robustos- **Perda de Validação**: Começou a aumentar enquanto a perda de treino diminuía   target_binário = 1 if target_original > 0 else 0

- [ ] Testar arquiteturas alternativas (3 camadas, diferentes configurações)

- [ ] Ajustar threshold de decisão para maximizar Recall````

- [ ] Comparar com modelos baseline (Random Forest, SVM, XGBoost)

- [ ] Análise de importância de features (SHAP values)**Interpretação**: Este comportamento é **esperado e normal** para um dataset pequeno (237 amostras de treino). As técnicas de regularização (Dropout + L2) foram eficazes em limitar o overfitting, mas não em eliminá-lo completamente.

- [ ] Coletar mais dados para reduzir overfitting

#### Variável-Alvo (Target)

---

### Fase 5️⃣: Avaliação Final e Análise Crítica

## 🚀 Como Executar

- **0:** Ausência de doença cardíaca (Saudável)

### Pré-requisitos

A avaliação final utilizou múltiplas métricas para fornecer uma visão completa da performance do modelo, com ênfase especial nas métricas mais relevantes para o contexto médico.- **1:** Presença de doença cardíaca (Doente)

- **Python:** 3.8 ou superior

- **Jupyter Notebook ou JupyterLab**---

- **Git** (para clonar o repositório)

## 📊 Resultados e Análise Crítica## 🔬 Metodologia

### Passo a Passo

### Métricas de PerformanceO projeto foi estruturado em **quatro fases principais**, seguindo um pipeline rigoroso de Data Science.

**1. Clone o Repositório**

| Métrica | Valor |### Fase 1️⃣: Análise Exploratória de Dados (EDA)

```bash

git clone https://github.com/AlexandreJr16/Heart-Diseases.git|---------|-------|

cd Heart-Diseases

```| **Acurácia Global** | 83.3% |Antes de qualquer modelagem, uma análise detalhada foi conduzida para entender a natureza dos dados.



**2. Instale as Dependências**| **Precisão (Doente)** | 84.6% |



```bash| **Recall (Doente)** | 78.6% |#### Principais Descobertas

pip install -r requirements.txt

```| **F1-Score (Doente)** | 0.81 |



Ou instale manualmente:✅ **Balanceamento Perfeito**



```bash### Matriz de Confusão

pip install pandas numpy tensorflow scikit-learn matplotlib seaborn

```- 526 instâncias da classe '1' (doente)



**3. Execute o Notebook**```- 499 instâncias da classe '0' (saudável)



Opção A - VS Code:                 Predito: Saudável    Predito: Doente- Validação da **Acurácia** como métrica confiável

```bash

code heart-diseases.ipynbReal: Saudável          26                   4

```

Real: Doente             6                  24✅ **Qualidade dos Dados**

Opção B - Jupyter Notebook:

```bash```

jupyter notebook heart-diseases.ipynb

```- Dataset completo, **sem valores nulos**



Opção C - Jupyter Lab:**Interpretação Detalhada**:- Não exigiu etapas de imputation

```bash

jupyter lab heart-diseases.ipynb- Pronto para modelagem após scaling

```

1. **Verdadeiros Negativos (26)**: Pacientes saudáveis corretamente classificados como saudáveis

**4. Execute as Células**

2. **Falsos Positivos (4)**: Pacientes saudáveis incorretamente classificados como doentes### Fase 2️⃣: Pré-Processamento e Prevenção de Data Leakage

- Execute todas de uma vez: `Cell → Run All`

- Execute célula por célula: `Shift + Enter`3. **Falsos Negativos (6)**: Pacientes doentes incorretamente classificados como saudáveis ⚠️



### Estrutura do Projeto4. **Verdadeiros Positivos (24)**: Pacientes doentes corretamente classificados como doentesEsta foi a etapa técnica mais crítica do projeto.



```### 🏥 Análise Crítica no Contexto Médico#### Divisão de Dados

Heart-Diseases/

│#### Importância do Recall (78.6%)```python

├── heart-diseases.ipynb    # Notebook principal com análise completa

├── heart.csv               # Dataset local (backup)Train: 80% | Test: 20%

├── requirements.txt        # Dependências Python

├── README.md              # Este arquivoEm aplicações médicas de diagnóstico, o **Recall** (sensibilidade) é frequentemente mais crítico que a precisão:Stratified Split (mantém proporção das classes)

└── .github/

    └── copilot-instructions.md  # Instruções do projeto````

```

- Um Recall de 78.6% significa que o modelo detectou corretamente **78.6% dos casos reais de doença cardíaca**

---

- Os **6 Falsos Negativos** representam o maior risco: pacientes doentes que não receberiam o tratamento adequado se confiássemos apenas no modelo#### Normalização (StandardScaler)

## 🛠️ Tecnologias Utilizadas



### Bibliotecas Principais

#### Falsos Positivos vs. Falsos Negativos**Por que é crucial?**

| Biblioteca         | Versão  | Função                                              |

|--------------------|---------|-----------------------------------------------------|Redes Neurais são altamente sensíveis a características em escalas diferentes:

| **Python**         | 3.8+    | Linguagem de programação base                       |

| **TensorFlow**     | 2.13.0+ | Framework de Deep Learning                          |- **Falsos Positivos (4)**: Pacientes saudáveis que seriam encaminhados para exames adicionais. Embora cause custos e ansiedade, é o "erro menos perigoso"

| **Keras**          | (API)   | API de alto nível para construção de redes neurais |

| **Scikit-learn**   | 1.3.0+  | Pré-processamento, métricas e validação            |- **Falsos Negativos (6)**: Pacientes doentes que receberiam alta médica. Este é o erro crítico que pode ter consequências fatais- `age`: 29-77

| **Pandas**         | 2.0.0+  | Manipulação e análise de dados                     |

| **NumPy**          | 1.24.0+ | Computação numérica e arrays                       |- `chol`: 126-564

| **Matplotlib**     | 3.7.0+  | Visualização de dados (gráficos)                   |

| **Seaborn**        | 0.12.0+ | Visualização estatística avançada                  |#### Conclusão sobre Performance



### Ferramentas de Desenvolvimento**Metodologia Rigorosa para Prevenir Data Leakage:**



- **Jupyter Notebook:** Ambiente interativo de desenvolvimento- Uma acurácia de **83.3%** é **realista e apropriada** para um dataset de 297 amostras

- **Git/GitHub:** Controle de versão e colaboração

- **VS Code:** Editor de código (opcional)- O desempenho é competitivo com estudos acadêmicos similares usando o mesmo dataset UCI```python



---- Para uso clínico real, o modelo precisaria de:# ✅ CORRETO: Fit apenas no treino



## 📚 Referências  - Ajuste do threshold de decisão (reduzir de 0.5 para ~0.3) para aumentar o Recallscaler.fit(X_train)



### Dataset  - Validação em datasets externos maioresX_train_scaled = scaler.transform(X_train)



- **Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988).** Heart Disease Data Set. UCI Machine Learning Repository. Disponível em: http://archive.ics.uci.edu/ml/datasets/Heart+Disease  - Integração como ferramenta de triagem, não diagnóstico definitivoX_test_scaled = scaler.transform(X_test)



### Fundamentação Teórica



- **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*. MIT Press.### 📈 Análise do Treinamento# ❌ ERRADO: Fit em todos os dados (causa data leakage)

- **Géron, A. (2019).** *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (2nd ed.). O'Reilly Media.

- **Chollet, F. (2017).** *Deep Learning with Python*. Manning Publications.scaler.fit(X)  # NÃO FAZER ISSO!



### Documentação TécnicaOs gráficos de histórico de treinamento revelaram:```



- TensorFlow Documentation: https://www.tensorflow.org/api_docs

- Keras API Reference: https://keras.io/api/

- Scikit-learn Documentation: https://scikit-learn.org/stable/documentation.html- **Convergência**: O modelo convergiu de forma estável nas primeiras 40 épocas### Fase 3️⃣: Arquitetura e Treinamento do Modelo



---- **Overfitting**: Detectado após ~40 épocas, com divergência entre treino e validação



## 👥 Autores- **Eficácia da Regularização**: Dropout e L2 limitaram o overfitting, mas não o eliminaram completamente#### Arquitetura da Rede Neural



**Alexandre Pereira de Souza Junior**

**Leonardo Brandão**

**Vithor Vitório****Contexto**: Este padrão é esperado e normal para datasets pequenos (237 amostras de treino).```



**Instituição:** Universidade Federal de Alagoas (UFAL)  Input Layer (13 features)

**Disciplina:** Fundamentos de Inteligência Artificial (FIA)

**Professor:** Edjard Mota  ---        ↓

**Período:** 2º Semestre de 2025

Dense(16, ReLU) + L2 Regularization

---

## 🎯 Conclusão        ↓

## 📄 Licença

Dropout(0.25)

Este projeto está sob a licença MIT. Consulte o arquivo `LICENSE` para mais detalhes.

### Eficácia do Modelo        ↓

---

Dense(8, ReLU) + L2 Regularization

## 📧 Contato

O modelo desenvolvido **atendeu plenamente aos requisitos do projeto**:        ↓

Para dúvidas, sugestões ou colaborações:

Dropout(0.25)

- **GitHub:** [@AlexandreJr16](https://github.com/AlexandreJr16)

- **Repositório:** [Heart-Diseases](https://github.com/AlexandreJr16/Heart-Diseases)✅ Construção de uma ANN feedforward com 2 camadas ocultas (ReLU) e regularização Dropout          ↓



---✅ Camada de saída com ativação sigmoid para classificação binária  Output(1, Sigmoid) → Probabilidade [0, 1]



<div align="center">✅ Avaliação utilizando Acurácia (83.3%), Precisão (84.6%), Recall (78.6%) e Matriz de Confusão  ```



**⭐ Se este projeto foi útil para seus estudos, considere dar uma estrela no repositório!**✅ Entrega de um classificador funcional com análise realista de desempenho



Desenvolvido com dedicação para a disciplina de Fundamentos de IA 🧠❤️#### Configuração de Treinamento



</div>### Importância da Normalização dos Dados


| Parâmetro          | Valor                       |

A normalização dos dados revelou-se **absolutamente essencial** para o sucesso do projeto:| ------------------ | --------------------------- |

| **Optimizer**      | Adam                        |

1. **Convergência do Treinamento**: Sem normalização, as features com escalas maiores (ex: colesterol ~200-300) dominariam o gradiente, dificultando ou impedindo a convergência da rede neural.| **Loss Function**  | Binary Crossentropy         |

| **Epochs**         | 100                         |

2. **Prevenção de Data Leakage**: A aplicação correta do StandardScaler (fit no treino, transform no teste) garantiu que o modelo não tivesse acesso a informações futuras, simulando adequadamente um cenário de produção.| **Batch Size**     | 10                          |

| **Regularization** | L2 (0.001) + Dropout (0.25) |

3. **Performance**: A normalização permitiu que todas as 13 features contribuíssem de forma balanceada para as predições, resultando na acurácia de 83.3%.

#### Técnicas de Regularização

### Lições Aprendidas

- **Dropout:** Previne overfitting desativando aleatoriamente 25% dos neurônios

- Datasets pequenos requerem técnicas agressivas de regularização- **L2 Regularization:** Penaliza pesos grandes, promovendo generalização

- A ordem das operações no pipeline de pré-processamento é crítica para a validade do modelo- **Validation Split:** Monitoramento contínuo da performance no teste

- Métricas contextuais (Recall em medicina) são mais importantes que acurácia global

- Overfitting é um fenômeno esperado e deve ser monitorado, não necessariamente eliminado---



---## 📈 Resultados



## 🚀 Instruções de Execução### 🎯 Performance Geral



### Pré-requisitos```

Acurácia no Conjunto de Teste: 92.68%

- Python 3.8+```

- Jupyter Notebook ou JupyterLab

Isso significa que o modelo classificou corretamente **quase 93 de cada 100 pacientes** no conjunto de teste.

### Passos para Execução

### 🏥 Análise da Matriz de Confusão

1. **Clone o repositório**:

   ```bash> **Importante:** Em problemas médicos, a acurácia por si só não é suficiente.

   git clone https://github.com/AlexandreJr16/Heart-Diseases.git> O custo de um **Falso Negativo** (paciente doente diagnosticado como saudável) é muito maior que o de um **Falso Positivo**.

   cd Heart-Diseases

   ```#### Matriz de Confusão



2. **Instale as dependências**:|                        | **Previsto: Saudável (0)** | **Previsto: Doente (1)** |

   ```bash| ---------------------- | -------------------------- | ------------------------ |

   pip install -r requirements.txt| **Real: Saudável (0)** | 93 (TN) ✅                 | 7 (FP) ⚠️                |

   ```| **Real: Doente (1)**   | 8 (FN) ❌                  | 97 (TP) ✅               |



3. **Execute o notebook**:#### Métricas Detalhadas por Classe

   ```bash

   jupyter notebook heart-diseases.ipynb| Classe           | Precision | Recall | F1-Score | Support |

   ```| ---------------- | --------- | ------ | -------- | ------- |

| **Saudável (0)** | 92%       | 93%    | 93%      | 100     |

4. **Execute todas as células** sequencialmente (Cell → Run All) ou execute célula por célula para acompanhar a narrativa completa da análise.| **Doente (1)**   | 93%       | 92%    | 93%      | 105     |



### Dependências Principais### 🔍 Análise Crítica



- TensorFlow 2.13.0+ (inclui Keras)#### ✅ Pontos Fortes

- scikit-learn 1.3.0+

- pandas 2.0.0+1. **Recall (Sensibilidade) - Classe Doente: 92%**

- numpy 1.24.0+

- matplotlib 3.7.0+   - O modelo identificou corretamente **97 dos 105 pacientes doentes**

- seaborn 0.12.0+   - Métrica crucial para aplicações médicas



---2. **Equilíbrio entre Precision e Recall**



## 📚 Referências   - Ambas as métricas > 92% para as duas classes

   - Modelo balanceado e confiável

- UCI Machine Learning Repository: [Heart Disease Dataset](http://archive.ics.uci.edu/ml/datasets/Heart+Disease)

- Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). Heart Disease Data Set. UCI Machine Learning Repository.3. **Baixa Taxa de Falsos Positivos**

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.   - Apenas 7 pacientes saudáveis classificados como doentes

   - Evita exames desnecessários

---

#### ⚠️ Pontos de Atenção

## 👤 Autor

1. **Falsos Negativos: 8 casos**

**Alexandre Pereira de Souza Junior**     - Este é o erro mais crítico

Projeto desenvolvido para a disciplina de Fundamentos de Inteligência Artificial   - 8 pacientes doentes foram classificados como saudáveis

   - Em produção, seria necessário um segundo nível de validação

---

### 📊 Curvas de Aprendizado

**Licença**: MIT

**Última atualização**: Novembro 2025O treinamento por 100 épocas mostrou:


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
`````

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
