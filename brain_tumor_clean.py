# Brain Tumor Classification System

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import pickle
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import seaborn as sns
from pycaret.classification import *

warnings.filterwarnings('ignore')

# Configuration
train_path = '/kaggle/input/brain-tumor-classification-mri/Training'
test_path = '/kaggle/input/brain-tumor-classification-mri/Testing'
classes = {'no_tumor': 0, 'pituitary_tumor': 1, 'glioma_tumor': 2, 'meningioma_tumor': 3}
dec = {v: k for k, v in classes.items()}

# Data Loading
def load_data(data_path, classes):
    X, Y = [], []
    for cls in classes:
        class_path = os.path.join(data_path, cls)
        for img_name in os.listdir(class_path):
            img = cv2.imread(os.path.join(class_path, img_name), 0)
            img = cv2.resize(img, (200, 200))
            X.append(img)
            Y.append(classes[cls])
    return np.array(X), np.array(Y)

# Preprocessing
def preprocess_data(X):
    X_flat = X.reshape(len(X), -1)
    return X_flat / 255.0

# Model Training
def train_svm(X_train, y_train):
    model = SVC(C=10, gamma='scale', kernel='rbf', probability=True, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_logistic(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear')
    model.fit(X_train, y_train)
    return model

# Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc
    }

def plot_confusion_matrix(model, X_test, y_test, class_names):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def compare_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
    
    # Plot comparison
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, metrics) in enumerate(results.items()):
        values = [metrics[m] for m in metrics_names]
        ax.bar(x + i*width, values, width, label=name)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    plt.show()
    
    return results

# Testing
def test_data_loading():
    try:
        X, Y = load_data(train_path, classes)
        assert len(X) > 0 and len(Y) > 0
        print("✓ Data Loading: PASSED")
        return True
    except:
        print("✗ Data Loading: FAILED")
        return False

def test_preprocessing():
    try:
        test_data = np.random.randint(0, 255, (10, 200, 200))
        processed = preprocess_data(test_data)
        assert processed.max() <= 1.0 and processed.min() >= 0.0
        print("✓ Preprocessing: PASSED")
        return True
    except:
        print("✗ Preprocessing: FAILED")
        return False

def test_model_performance():
    try:
        dummy_X = np.random.rand(100, 1000)
        dummy_y = np.random.randint(0, 4, 100)
        X_train, X_test, y_train, y_test = train_test_split(dummy_X, dummy_y, test_size=0.2)
        
        model = train_svm(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        assert metrics['accuracy'] > 0
        print("✓ Model Performance: PASSED")
        return True
    except:
        print("✗ Model Performance: FAILED")
        return False

def run_tests():
    print("\nRunning Tests...")
    tests = [test_data_loading, test_preprocessing, test_model_performance]
    results = [test() for test in tests]
    print(f"Tests Passed: {sum(results)}/{len(results)}")

# Main Execution
if __name__ == "__main__":
    # Load and preprocess data
    X, Y = load_data(train_path, classes)
    X_processed = preprocess_data(X)
    
    print(f"Data shape: {X.shape}")
    print(f"Classes: {pd.Series(Y).value_counts()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_processed, Y, test_size=0.2, random_state=42)
    
    # Train models
    svm_model = train_svm(X_train, y_train)
    lr_model = train_logistic(X_train, y_train)
    
    # Evaluate models
    svm_metrics = evaluate_model(svm_model, X_test, y_test)
    lr_metrics = evaluate_model(lr_model, X_test, y_test)
    
    print("SVM Metrics:")
    for metric, value in svm_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nLogistic Regression Metrics:")
    for metric, value in lr_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Cross validation
    cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5)
    print(f"\nCross Validation - Mean: {cv_scores.mean():.4f}, Std: {cv_scores.std():.4f}")
    
    # Compare models
    models = {'SVM': svm_model, 'Logistic Regression': lr_model}
    comparison = compare_models(models, X_test, y_test)
    
    # Plot confusion matrices
    class_names = list(classes.keys())
    plot_confusion_matrix(svm_model, X_test, y_test, class_names)
    
    # Save models
    with open("svm_model.pkl", "wb") as f:
        pickle.dump(svm_model, f)
    with open("logistic_model.pkl", "wb") as f:
        pickle.dump(lr_model, f)
    
    # Run tests
    run_tests()
    
    # Classification report
    y_pred = svm_model.predict(X_test)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # PyCaret AutoML
    print("\nRunning AutoML...")
    exp = setup(X_processed, target=Y, session_id=123, verbose=False)
    best_automl = compare_models()
    tuned_model = tune_model(best_automl)
    evaluate_model_pycaret = evaluate_model(tuned_model)
    
    print("System Complete!")

