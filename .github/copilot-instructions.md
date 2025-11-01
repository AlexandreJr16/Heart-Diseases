# AI Coding Agent Instructions - Heart Disease Classifier

## Project Overview

Binary classification project using TensorFlow/Keras to predict heart disease presence (0=healthy, 1=diseased) from 13 clinical features. Single Jupyter notebook workflow with UCI Heart Disease dataset (297 patients after cleaning).

## Dataset & Data Handling

### Critical: Prevent Data Leakage

**ALWAYS** fit `StandardScaler` on training data only, then transform both train and test:

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit_transform on train
X_test_scaled = scaler.transform(X_test)        # transform only on test
```

### Dataset Source & Loading

- Primary source: UCI repository at `http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data`
- Local backup: `heart.csv` (303 patients, 14 columns including target)
- Missing values marked as `'?'` - **must** use `na_values='?'` when loading with pandas
- Expected final size: 297 patients after `dropna()`

### Target Variable Transformation

Original dataset has multi-class target (0-4). Convert to binary:

```python
data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)
```

### Train/Test Split

Use stratified split to maintain class balance:

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

## Model Architecture

### Standard Architecture (heart-diseases.ipynb)

```
Input(13) → Dense(16, relu) + L2(0.001) → Dropout(0.25)
         → Dense(8, relu) + L2(0.001) → Dropout(0.25)
         → Dense(1, sigmoid)
```

### Training Configuration

- Optimizer: `adam`
- Loss: `binary_crossentropy`
- Epochs: 100
- Batch size: 10
- Validation: Use test set during training via `validation_data` parameter

### Regularization Strategy

- **L2 regularization:** `kernel_regularizer=regularizers.l2(0.001)` on hidden layers
- **Dropout:** 0.25 (25%) after each hidden layer
- **Purpose:** Prevent overfitting on small dataset (237 training samples)

## Evaluation Methodology

### Primary Metrics (in order of importance for medical context)

1. **Recall (Sensitivity) for class 1 (diseased)** - minimize false negatives
2. **Confusion Matrix** - understand error types
3. **Precision** - assess false positive rate
4. **Overall Accuracy** - general performance

### Threshold Consideration

Default threshold is 0.5. When interpreting `model.predict()`:

```python
y_pred_probs = model.predict(X_test_scaled)
y_pred = (y_pred_probs > 0.5).astype(int)  # convert probabilities to classes
```

### False Negatives Are Critical

In medical context, classifying a diseased patient as healthy (false negative) is more dangerous than the reverse. Always analyze FN count separately.

## File Structure & Workflow

### Single Notebook Design

All analysis, modeling, and evaluation in `heart-diseases.ipynb`. Notebook follows this sequence:

1. Data loading & cleaning (remove `?` values)
2. EDA with correlation matrix and target distribution
3. Binary target conversion
4. Train/test split (stratified)
5. Scaling (fit on train only)
6. Model definition with `create_model()` function
7. Training with history tracking
8. Evaluation (metrics + visualizations)

### Dependencies

Managed in `requirements.txt` - all packages specify minimum versions with `>=`. Core dependencies:

- TensorFlow 2.13.0+ (Keras included)
- scikit-learn 1.3.0+ (StandardScaler, metrics)
- pandas 2.0.0+, numpy 1.24.0+
- matplotlib 3.7.0+, seaborn 0.12.0+ (visualizations)

## Language & Documentation

All markdown cells, variable names, and comments are in **Portuguese (Brazilian)**. Follow this convention:

- Class labels: "Saudável (0)" and "Doente (1)"
- Metrics: "Acurácia", "Precisão", "Recall", "Matriz de Confusão"
- Documentation: use formal academic Portuguese

## Code Patterns

### Feature/Target Separation

```python
X = data.drop('target', axis=1)  # 13 features
y = data['target']               # binary target
```

### Model Creation Pattern

Use a function wrapper for reusability:

```python
def create_model():
    model = Sequential()
    # architecture definition
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
```

### Visualization Standards

- Use `figsize=(12, 4)` for dual plots (train/val accuracy and loss)
- Color palette for target: `['#2ecc71', '#e74c3c']` (green=healthy, red=diseased)
- Always include titles in Portuguese with clear labels

## Performance Expectations

### UCI Dataset (297 patients)

Expected results with current architecture:

- Accuracy: ~83-85%
- Recall (diseased): ~79-81%
- False negatives: 5-7 cases
- Training should converge within 100 epochs without overfitting

### Dataset Comparison Context

Project documentation references a Kaggle dataset (1025 patients, 92% accuracy) for comparison. The UCI dataset is the **correct** dataset per project requirements - lower accuracy is expected and acceptable.

## Common Tasks

### Adding New Features

If engineering new features, add them **before** train/test split, then ensure scaler handles all numerical features.

### Adjusting Model Complexity

To reduce overfitting: increase dropout (try 0.3-0.4) or L2 penalty (try 0.01).
To increase capacity: add neurons to hidden layers or add another Dense layer.

### Changing Classification Threshold

To reduce false negatives (more conservative), lower threshold from 0.5:

```python
y_pred = (y_pred_probs > 0.3).astype(int)  # more sensitive to disease
```
