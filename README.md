# Projeto 1: Classificação de Doenças Cardíacas - Fundamentos de IA# Projeto 1: Classificação de Doenças Cardíacas - Fundamentos de IA

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)

[![Keras](https://img.shields.io/badge/Keras-API-red.svg)](https://keras.io/)[![Keras](https://img.shields.io/badge/Keras-API-red.svg)](https://keras.io/)

[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-yellow.svg)](https://scikit-learn.org/)[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-yellow.svg)](https://scikit-learn.org/)

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Disciplina:** Fundamentos de Inteligência Artificial (FIA) > **Disciplina:** Fundamentos de Inteligência Artificial (FIA)

> **Autor:** Alexandre Pereira de Souza Junior, Leonardo Brandão, Vithor Vitorio. > **Autor:** Alexandre Pereira de Souza Junior, Leonardo Brandão, Vithor Vitorio.

---

## 📋 Índice## 📋 Índice

- [Contexto do Problema](#-contexto-do-problema)- [Contexto do Problema](#-contexto-do-problema)

- [Dataset: Origem, Estrutura e Limpeza](#-dataset-origem-estrutura-e-limpeza)- [Dataset: Origem, Estrutura e Limpeza](#-dataset-origem-estrutura-e-limpeza)

- [Metodologia](#️-metodologia)- [Metodologia](#️-metodologia)

- [Resultados e Análise Crítica](#-resultados-e-análise-crítica)- [Resultados e Análise Crítica](#-resultados-e-análise-crítica)

- [Conclusão](#-conclusão)- [Conclusão](#-conclusão)

- [Instruções de Execução](#-instruções-de-execução)- [Instruções de Execução](#-instruções-de-execução)

- [Referências](#-referências)- [Referências](#-referências)

---

## 📋 Contexto do Problema## 📋 Contexto do Problema

Este projeto acadêmico foi desenvolvido como parte da disciplina de Fundamentos de Inteligência Artificial e tem como objetivo construir um **classificador binário** para predição de doenças cardíacas. O modelo desenvolvido classifica pacientes em duas categorias:Este projeto acadêmico foi desenvolvido como parte da disciplina de Fundamentos de Inteligência Artificial e tem como objetivo construir um **classificador binário** para predição de doenças cardíacas. O modelo desenvolvido classifica pacientes em duas categorias:

- **0 (Saudável)**: Ausência de doença cardíaca- **0:** Ausência de doença cardíaca (Saudável)

- **1 (Doente)**: Presença de doença cardíaca- **1:** Presença de doença cardíaca (Doente)

A abordagem utiliza técnicas de **Deep Learning** para analisar 13 atributos clínicos e fisiológicos de pacientes, construindo uma Rede Neural Artificial (ANN) feedforward capaz de realizar predições com base em dados históricos.---

---## 🛠️ Metodologia

## 🔬 Dataset: Origem, Estrutura e LimpezaO projeto foi estruturado em **cinco fases principais**, seguindo um pipeline rigoroso de Data Science para garantir a validade e a replicabilidade dos resultados.

### Fonte de Dados### Fase 1️⃣: Análise Exploratória de Dados (EDA)

O dataset utilizado é o clássico **Cleveland Heart Disease Database** do repositório UCI Machine Learning, acessível via:Antes de qualquer modelagem, uma análise detalhada foi conduzida para entender a natureza dos dados:

````- **Balanceamento de Classes**: Verificação da distribuição entre pacientes saudáveis e doentes

http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data- **Matriz de Correlação**: Identificação de relações lineares entre as features

```- **Estatísticas Descritivas**: Compreensão da distribuição de cada atributo clínico



**Nota Importante sobre a Escolha do Dataset**: Durante a fase inicial do projeto, identificamos uma discrepância entre o dataset sugerido no material de apoio (Kaggle, 1025 amostras) e o dataset utilizado no notebook de referência do professor. Após análise crítica, confirmamos que o dataset correto para este projeto é o **UCI Cleveland original (303 amostras)**, que representa o benchmark histórico para pesquisas em classificação de doenças cardíacas.### Fase 2️⃣: Pré-processamento e Prevenção de Data Leakage



### Estrutura do DatasetEsta foi a etapa técnica **mais crítica** do projeto, onde seguimos rigorosamente as melhores práticas de Machine Learning.



- **Amostras Originais**: 303 pacientes#### Pipeline de Pré-processamento

- **Atributos**: 13 features clínicas + 1 variável target

- **Features Incluem**: Idade, sexo, tipo de dor no peito (cp), pressão arterial em repouso (trestbps), colesterol sérico (chol), glicemia em jejum (fbs), resultados de ECG em repouso (restecg), frequência cardíaca máxima (thalach), angina induzida por exercício (exang), depressão ST (oldpeak), inclinação do segmento ST (slope), número de vasos principais (ca), e talassemia (thal).1. **Separação de Features e Target**:

   ```python

#### Principais Features   X = data.drop('target', axis=1)  # 13 features

   y = data['target']               # variável binária

| Feature    | Descrição                              |   ```

| ---------- | -------------------------------------- |

| `age`      | Idade do paciente                      |2. **Divisão Estratificada Train/Test**:

| `sex`      | Sexo (1 = masculino, 0 = feminino)     |   ```python

| `cp`       | Tipo de dor no peito (0-3)             |   train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

| `trestbps` | Pressão arterial em repouso (mm Hg)    |   ```

| `chol`     | Colesterol sérico (mg/dl)              |   - **Conjunto de Treino**: 237 amostras (80%)

| `fbs`      | Glicemia em jejum > 120 mg/dl          |   - **Conjunto de Teste**: 60 amostras (20%)

| `restecg`  | Resultados eletrocardiográficos        |   - **Estratificação**: Mantém a proporção de classes em ambos os conjuntos

| `thalach`  | Frequência cardíaca máxima alcançada   |

| `exang`    | Angina induzida por exercício          |3. **Normalização (StandardScaler)** - **PONTO CRÍTICO**:

| `oldpeak`  | Depressão de ST induzida por exercício |   ```python

| `slope`    | Inclinação do segmento ST de pico      |   scaler = StandardScaler()

| `ca`       | Número de vasos principais (0-3)       |   X_train_scaled = scaler.fit_transform(X_train)  # Fit APENAS no treino

| `thal`     | Talassemia (1-3)                       |   X_test_scaled = scaler.transform(X_test)        # Transform no teste

````

### Etapas de Limpeza

#### ⚠️ Importância da Normalização dos Dados

1. **Tratamento de Valores Ausentes**: O dataset original continha valores nulos representados pelo caractere `'?'`. Esses valores foram identificados durante a carga dos dados utilizando o parâmetro `na_values='?'` do pandas.

A normalização dos dados revelou-se **absolutamente essencial** para o sucesso do projeto:

2. **Remoção de Amostras Incompletas**: Aplicamos `dropna()` para remover todas as linhas com valores ausentes, resultando em **297 amostras válidas** para análise.

**Por que normalizar?**

3. **Transformação da Variável Target**: A variável-alvo original era multi-classe (0, 1, 2, 3, 4), representando diferentes níveis de severidade da doença. Convertemos para um problema binário aplicando a transformação:- Redes Neurais são altamente sensíveis a características em escalas diferentes

   ```python- Features como `chol`(126-564) dominariam features como`sex` (0-1) sem normalização

   target_binário = 1 if target_original > 0 else 0- A convergência do gradiente descendente é muito mais eficiente com dados normalizados

   ```

   ```

**Por que esta ordem é crucial?**

#### Variável-Alvo (Target)- Realizar o scaling **antes** da divisão train/test causaria **data leakage**

- Informações estatísticas do conjunto de teste (média e desvio padrão) "vazariam" para o conjunto de treino

- **0:** Ausência de doença cardíaca (Saudável)- O scaler deve aprender os parâmetros **exclusivamente** dos dados de treino

- **1:** Presença de doença cardíaca (Doente)- Esta prática simula o cenário real de produção, onde novos dados nunca foram vistos durante o treinamento

---### Fase 3️⃣: Construção do Modelo (ANN)

## 🛠️ MetodologiaDesenvolvemos uma Rede Neural Artificial feedforward com a seguinte arquitetura:

O projeto foi estruturado em **cinco fases principais**, seguindo um pipeline rigoroso de Data Science para garantir a validade e a replicabilidade dos resultados.```

Camada de Entrada: 13 neurônios (features)

### Fase 1️⃣: Análise Exploratória de Dados (EDA) ↓

Camada Oculta 1: 16 neurônios

Antes de qualquer modelagem, uma análise detalhada foi conduzida para entender a natureza dos dados: - Ativação: ReLU

    - Regularização: L2 (lambda=0.001)

- **Balanceamento de Classes**: Verificação da distribuição entre pacientes saudáveis e doentes - Dropout: 25%

- **Matriz de Correlação**: Identificação de relações lineares entre as features ↓

- **Estatísticas Descritivas**: Compreensão da distribuição de cada atributo clínicoCamada Oculta 2: 8 neurônios

  - Ativação: ReLU

### Fase 2️⃣: Pré-processamento e Prevenção de Data Leakage - Regularização: L2 (lambda=0.001)

    - Dropout: 25%

Esta foi a etapa técnica **mais crítica** do projeto, onde seguimos rigorosamente as melhores práticas de Machine Learning. ↓

Camada de Saída: 1 neurônio

#### Pipeline de Pré-processamento - Ativação: Sigmoid (probabilidade de doença)

````

1. **Separação de Features e Target**:

   ```python#### Configuração de Treinamento

   X = data.drop('target', axis=1)  # 13 features

   y = data['target']               # variável binária| Parâmetro          | Valor                       |

   ```| ------------------ | --------------------------- |

| **Optimizer**      | Adam                        |

2. **Divisão Estratificada Train/Test**:| **Loss Function**  | Binary Crossentropy         |

   ```python| **Epochs**         | 100                         |

   train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)| **Batch Size**     | 10                          |

   ```| **Regularization** | L2 (0.001) + Dropout (0.25) |

   - **Conjunto de Treino**: 237 amostras (80%)| **Validação**      | Conjunto de teste           |

   - **Conjunto de Teste**: 60 amostras (20%)

   - **Estratificação**: Mantém a proporção de classes em ambos os conjuntos#### Estratégia de Regularização



3. **Normalização (StandardScaler)** - **PONTO CRÍTICO**:- **L2 Regularization**: Penaliza pesos muito altos, promovendo uma distribuição mais suave dos pesos

   ```python- **Dropout (25%)**: Durante o treino, desativa aleatoriamente 25% dos neurônios em cada camada oculta, forçando a rede a aprender representações mais robustas e reduzindo a dependência de neurônios específicos

   scaler = StandardScaler()

   X_train_scaled = scaler.fit_transform(X_train)  # Fit APENAS no treino### Fase 4️⃣: Treinamento e Análise de Overfitting

   X_test_scaled = scaler.transform(X_test)        # Transform no teste

   ```O modelo foi treinado por 100 épocas com monitoramento contínuo das métricas de treino e validação. Os gráficos de histórico revelaram um padrão clássico de **overfitting** após aproximadamente 30-40 épocas:



#### ⚠️ Importância da Normalização dos Dados- **Acurácia de Treino**: Continuou aumentando até ~90%

- **Acurácia de Validação**: Estagnou em ~83% e apresentou flutuações

A normalização dos dados revelou-se **absolutamente essencial** para o sucesso do projeto:- **Perda de Validação**: Começou a aumentar enquanto a perda de treino diminuía



**Por que normalizar?****Interpretação**: Este comportamento é **esperado e normal** para um dataset pequeno (237 amostras de treino). As técnicas de regularização (Dropout + L2) foram eficazes em limitar o overfitting, mas não em eliminá-lo completamente.

- Redes Neurais são altamente sensíveis a características em escalas diferentes

- Features como `chol` (126-564) dominariam features como `sex` (0-1) sem normalização### Fase 5️⃣: Avaliação Final e Análise Crítica

- A convergência do gradiente descendente é muito mais eficiente com dados normalizados

A avaliação final utilizou múltiplas métricas para fornecer uma visão completa da performance do modelo, com ênfase especial nas métricas mais relevantes para o contexto médico.

**Por que esta ordem é crucial?**

- Realizar o scaling **antes** da divisão train/test causaria **data leakage**A abordagem utiliza técnicas de **Deep Learning** para analisar 13 atributos clínicos e fisiológicos de pacientes, construindo uma Rede Neural Artificial (ANN) feedforward capaz de realizar predições com base em dados históricos.

- Informações estatísticas do conjunto de teste (média e desvio padrão) "vazariam" para o conjunto de treino

- O scaler deve aprender os parâmetros **exclusivamente** dos dados de treino---

- Esta prática simula o cenário real de produção, onde novos dados nunca foram vistos durante o treinamento

## 🔬 Dataset: Origem, Estrutura e Limpeza

### Fase 3️⃣: Construção do Modelo (ANN)

### Fonte de Dados

Desenvolvemos uma Rede Neural Artificial feedforward com a seguinte arquitetura:

O dataset utilizado é o clássico **Cleveland Heart Disease Database** do repositório UCI Machine Learning, acessível via:

````

Camada de Entrada: 13 neurônios (features)```

    ↓http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data

Camada Oculta 1: 16 neurônios```

    - Ativação: ReLU

    - Regularização: L2 (lambda=0.001)**Nota Importante sobre a Escolha do Dataset**: Durante a fase inicial do projeto, identificamos uma discrepância entre o dataset sugerido no material de apoio (Kaggle, 1025 amostras) e o dataset utilizado no notebook de referência do professor. Após análise crítica, confirmamos que o dataset correto para este projeto é o **UCI Cleveland original (303 amostras)**, que representa o benchmark histórico para pesquisas em classificação de doenças cardíacas.

    - Dropout: 25%

    ↓### Estrutura do Dataset

Camada Oculta 2: 8 neurônios

    - Ativação: ReLU- **Amostras Originais**: 303 pacientes

    - Regularização: L2 (lambda=0.001)- **Atributos**: 13 features clínicas + 1 variável target

    - Dropout: 25%- **Features Incluem**: Idade, sexo, tipo de dor no peito (cp), pressão arterial em repouso (trestbps), colesterol sérico (chol), glicemia em jejum (fbs), resultados de ECG em repouso (restecg), frequência cardíaca máxima (thalach), angina induzida por exercício (exang), depressão ST (oldpeak), inclinação do segmento ST (slope), número de vasos principais (ca), e talassemia (thal).

    ↓

Camada de Saída: 1 neurônio#### Principais Features

    - Ativação: Sigmoid (probabilidade de doença)

````| Feature    | Descrição                              |

| ---------- | -------------------------------------- |

#### Configuração de Treinamento| `age`      | Idade do paciente                      |

| `sex`      | Sexo (1 = masculino, 0 = feminino)     |

| Parâmetro          | Valor                       || `cp`       | Tipo de dor no peito (0-3)             |

| ------------------ | --------------------------- || `trestbps` | Pressão arterial em repouso (mm Hg)    |

| **Optimizer**      | Adam                        || `chol`     | Colesterol sérico (mg/dl)              |

| **Loss Function**  | Binary Crossentropy         || `fbs`      | Glicemia em jejum > 120 mg/dl          |

| **Epochs**         | 100                         || `restecg`  | Resultados eletrocardiográficos        |

| **Batch Size**     | 10                          || `thalach`  | Frequência cardíaca máxima alcançada   |

| **Regularization** | L2 (0.001) + Dropout (0.25) || `exang`    | Angina induzida por exercício          |

| **Validação**      | Conjunto de teste           || `oldpeak`  | Depressão de ST induzida por exercício |

| `slope`    | Inclinação do segmento ST de pico      |

#### Estratégia de Regularização| `ca`       | Número de vasos principais (0-3)       |

| `thal`     | Talassemia (1-3)                       |

- **L2 Regularization**: Penaliza pesos muito altos, promovendo uma distribuição mais suave dos pesos

- **Dropout (25%)**: Durante o treino, desativa aleatoriamente 25% dos neurônios em cada camada oculta, forçando a rede a aprender representações mais robustas e reduzindo a dependência de neurônios específicos### Etapas de Limpeza



### Fase 4️⃣: Treinamento e Análise de Overfitting1. **Tratamento de Valores Ausentes**: O dataset original continha valores nulos representados pelo caractere `'?'`. Esses valores foram identificados durante a carga dos dados utilizando o parâmetro `na_values='?'` do pandas.



O modelo foi treinado por 100 épocas com monitoramento contínuo das métricas de treino e validação. Os gráficos de histórico revelaram um padrão clássico de **overfitting** após aproximadamente 30-40 épocas:2. **Remoção de Amostras Incompletas**: Aplicamos `dropna()` para remover todas as linhas com valores ausentes, resultando em **297 amostras válidas** para análise.



- **Acurácia de Treino**: Continuou aumentando até ~90%3. **Transformação da Variável Target**: A variável-alvo original era multi-classe (0, 1, 2, 3, 4), representando diferentes níveis de severidade da doença. Convertemos para um problema binário aplicando a transformação:

- **Acurácia de Validação**: Estagnou em ~83% e apresentou flutuações   ```python

- **Perda de Validação**: Começou a aumentar enquanto a perda de treino diminuía   target_binário = 1 if target_original > 0 else 0

````

**Interpretação**: Este comportamento é **esperado e normal** para um dataset pequeno (237 amostras de treino). As técnicas de regularização (Dropout + L2) foram eficazes em limitar o overfitting, mas não em eliminá-lo completamente.

#### Variável-Alvo (Target)

### Fase 5️⃣: Avaliação Final e Análise Crítica

- **0:** Ausência de doença cardíaca (Saudável)

A avaliação final utilizou múltiplas métricas para fornecer uma visão completa da performance do modelo, com ênfase especial nas métricas mais relevantes para o contexto médico.- **1:** Presença de doença cardíaca (Doente)

---

## 📊 Resultados e Análise Crítica## 🔬 Metodologia

### Métricas de PerformanceO projeto foi estruturado em **quatro fases principais**, seguindo um pipeline rigoroso de Data Science.

| Métrica | Valor |### Fase 1️⃣: Análise Exploratória de Dados (EDA)

|---------|-------|

| **Acurácia Global** | 83.3% |Antes de qualquer modelagem, uma análise detalhada foi conduzida para entender a natureza dos dados.

| **Precisão (Doente)** | 84.6% |

| **Recall (Doente)** | 78.6% |#### Principais Descobertas

| **F1-Score (Doente)** | 0.81 |

✅ **Balanceamento Perfeito**

### Matriz de Confusão

- 526 instâncias da classe '1' (doente)

```- 499 instâncias da classe '0' (saudável)

                 Predito: Saudável    Predito: Doente- Validação da **Acurácia** como métrica confiável

Real: Saudável          26                   4

Real: Doente             6                  24✅ **Qualidade dos Dados**

```

- Dataset completo, **sem valores nulos**

**Interpretação Detalhada**:- Não exigiu etapas de imputation

- Pronto para modelagem após scaling

1. **Verdadeiros Negativos (26)**: Pacientes saudáveis corretamente classificados como saudáveis

2. **Falsos Positivos (4)**: Pacientes saudáveis incorretamente classificados como doentes### Fase 2️⃣: Pré-Processamento e Prevenção de Data Leakage

3. **Falsos Negativos (6)**: Pacientes doentes incorretamente classificados como saudáveis ⚠️

4. **Verdadeiros Positivos (24)**: Pacientes doentes corretamente classificados como doentesEsta foi a etapa técnica mais crítica do projeto.

### 🏥 Análise Crítica no Contexto Médico#### Divisão de Dados

#### Importância do Recall (78.6%)```python

Train: 80% | Test: 20%

Em aplicações médicas de diagnóstico, o **Recall** (sensibilidade) é frequentemente mais crítico que a precisão:Stratified Split (mantém proporção das classes)

````

- Um Recall de 78.6% significa que o modelo detectou corretamente **78.6% dos casos reais de doença cardíaca**

- Os **6 Falsos Negativos** representam o maior risco: pacientes doentes que não receberiam o tratamento adequado se confiássemos apenas no modelo#### Normalização (StandardScaler)



#### Falsos Positivos vs. Falsos Negativos**Por que é crucial?**

Redes Neurais são altamente sensíveis a características em escalas diferentes:

- **Falsos Positivos (4)**: Pacientes saudáveis que seriam encaminhados para exames adicionais. Embora cause custos e ansiedade, é o "erro menos perigoso"

- **Falsos Negativos (6)**: Pacientes doentes que receberiam alta médica. Este é o erro crítico que pode ter consequências fatais- `age`: 29-77

- `chol`: 126-564

#### Conclusão sobre Performance

**Metodologia Rigorosa para Prevenir Data Leakage:**

- Uma acurácia de **83.3%** é **realista e apropriada** para um dataset de 297 amostras

- O desempenho é competitivo com estudos acadêmicos similares usando o mesmo dataset UCI```python

- Para uso clínico real, o modelo precisaria de:# ✅ CORRETO: Fit apenas no treino

  - Ajuste do threshold de decisão (reduzir de 0.5 para ~0.3) para aumentar o Recallscaler.fit(X_train)

  - Validação em datasets externos maioresX_train_scaled = scaler.transform(X_train)

  - Integração como ferramenta de triagem, não diagnóstico definitivoX_test_scaled = scaler.transform(X_test)



### 📈 Análise do Treinamento# ❌ ERRADO: Fit em todos os dados (causa data leakage)

scaler.fit(X)  # NÃO FAZER ISSO!

Os gráficos de histórico de treinamento revelaram:```



- **Convergência**: O modelo convergiu de forma estável nas primeiras 40 épocas### Fase 3️⃣: Arquitetura e Treinamento do Modelo

- **Overfitting**: Detectado após ~40 épocas, com divergência entre treino e validação

- **Eficácia da Regularização**: Dropout e L2 limitaram o overfitting, mas não o eliminaram completamente#### Arquitetura da Rede Neural



**Contexto**: Este padrão é esperado e normal para datasets pequenos (237 amostras de treino).```

Input Layer (13 features)

---        ↓

Dense(16, ReLU) + L2 Regularization

## 🎯 Conclusão        ↓

Dropout(0.25)

### Eficácia do Modelo        ↓

Dense(8, ReLU) + L2 Regularization

O modelo desenvolvido **atendeu plenamente aos requisitos do projeto**:        ↓

Dropout(0.25)

✅ Construção de uma ANN feedforward com 2 camadas ocultas (ReLU) e regularização Dropout          ↓

✅ Camada de saída com ativação sigmoid para classificação binária  Output(1, Sigmoid) → Probabilidade [0, 1]

✅ Avaliação utilizando Acurácia (83.3%), Precisão (84.6%), Recall (78.6%) e Matriz de Confusão  ```

✅ Entrega de um classificador funcional com análise realista de desempenho

#### Configuração de Treinamento

### Importância da Normalização dos Dados

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
````

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
