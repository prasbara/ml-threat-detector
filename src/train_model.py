import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import logging
import shap

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, precision_recall_curve, matthews_corrcoef)
from sklearn.impute import SimpleImputer

# --- 1. Konfigurasi Awal ---
print("--- ML Threat Detector Training Pipeline ---")

# Konfigurasi Path

MODEL_DIR = 'd:\\SOC PORTO\\ml-threat-detector\\models'
REPORTS_DIR = 'd:\\SOC PORTO\\ml-threat-detector\\reports'
PLOTS_DIR = os.path.join(REPORTS_DIR, 'plots')
MODEL_PATH = os.path.join(MODEL_DIR, 'threat_model.pkl')
LOG_FILE = os.path.join(REPORTS_DIR, 'training_log.txt')

# Buat direktori jika belum ada
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(LOG_FILE, mode='w'),
                        logging.StreamHandler()
                    ])

logging.info("--- Pipeline Training Dimulai ---")

logging.info("1. Memuat dan menggabungkan dataset...")
data_path1 = 'd:\\SOC PORTO\\ml-threat-detector\\data\\telesurgery_cybersecurity_dataset.csv'
data_path2 = 'd:\\SOC PORTO\\ml-threat-detector\\data\\robot_security_events.csv'

df1 = pd.read_csv(data_path1)
df2 = pd.read_csv(data_path2)

df = pd.concat([df1, df2], ignore_index=True)

logging.info(f"Total {len(df)} baris data dimuat dari dua dataset.")


# --- 2. Muat Data & Feature Engineering ---
logging.info("2. Melakukan feature engineering...")
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.sort_values('Timestamp', inplace=True)

# Konversi Threat Severity ke numerik sebelum feature engineering
# PASTIKAN TIPE DATA KONSISTEN SEBELUM ENCODING
df['Threat Severity'] = df['Threat Severity'].astype(str)
le_severity = LabelEncoder()
df['Threat Severity'] = le_severity.fit_transform(df['Threat Severity'])

# Buat fitur baru
df['Interaction_Latency_Response'] = df['Network Latency (ms)'] * df['Response Time (sec)']
df['Data_Volume'] = df['Data Transfer Rate (Mbps)'] * df['Gesture Duration (sec)']
df['Severity_to_Response'] = df['Threat Severity'] / (df['Response Time (sec)'] + 0.1)
df['Latency_to_Rate'] = df['Network Latency (ms)'] / (df['Data Transfer Rate (Mbps)'] + 1)

# --- 3. Temporal Split & Pra-pemrosesan ---
logging.info("2. Melakukan temporal split dan pra-pemrosesan...")
df.drop(columns=['Robot Gesture ID', 'Gesture Coordinates (x, y, z)', 'Message ID', 'Sender', 'Receiver'], inplace=True)

train_size = int(len(df) * 0.8)
train_df, test_df = df.iloc[:train_size], df.iloc[train_size:]

X_train = train_df.drop(['Threat Detected', 'Timestamp'], axis=1)
y_train = train_df['Threat Detected']
X_test = test_df.drop(['Threat Detected', 'Timestamp'], axis=1)
y_test = test_df['Threat Detected']

# Encoding
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
encoders = {'Threat Severity': le_severity} # Simpan encoder severity
for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = X_test[col].map(lambda s: '-1' if s not in le.classes_ else s)
    le.classes_ = np.append(le.classes_, '-1')
    X_test[col] = le.transform(X_test[col].astype(str))
    encoders[col] = le

# --- 4. Data Augmentation (Noise Injection) ---
logging.info("3. Melakukan augmentasi data (noise injection) pada data training...")
X_train_aug = X_train.copy()
numeric_cols_for_aug = ['Network Latency (ms)', 'Data Transfer Rate (Mbps)', 'Gesture Duration (sec)']
for col in numeric_cols_for_aug:
    noise = np.random.uniform(0.95, 1.05, len(X_train_aug))
    X_train_aug[col] *= noise

# Imputasi & Scaling
numeric_cols = X_train.select_dtypes(include=np.number).columns
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train_aug)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

feature_names = list(X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)


# --- 5. Hyperparameter Tuning ---
logging.info("4. Mencari hyperparameter terbaik dengan GridSearchCV...")
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='roc_auc')
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_
logging.info(f"Model Trained: {type(best_model).__name__}")
logging.info(f"Parameter terbaik: {grid_search.best_params_}")

# --- 6. Evaluasi & Analisis ---
logging.info("5. Mengevaluasi model pada data test...")
y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

# Metrik
logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
logging.info(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
logging.info(f"Matthews Correlation Coefficient: {matthews_corrcoef(y_test, y_pred):.4f}")
logging.info("\nClassification Report:\n" + classification_report(y_test, y_pred))

# Visualisasi
# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'))
plt.close()
logging.info(f"Plot Confusion Matrix disimpan di {os.path.join(PLOTS_DIR, 'confusion_matrix.png')}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_proba):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'roc_curve.png'))
plt.close()
logging.info(f"Plot ROC Curve disimpan di {os.path.join(PLOTS_DIR, 'roc_curve.png')}")

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'precision_recall_curve.png'))
plt.close()
logging.info(f"Plot Precision-Recall Curve disimpan di {os.path.join(PLOTS_DIR, 'precision_recall_curve.png')}")


# --- 7. Feature Importance & SHAP ---
logging.info("6. Menganalisis feature importance dan SHAP...")

# Feature Importance
importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
logging.info("\nTop 10 Feature Importances:\n" + str(feature_importance_df.head(10)))

# Simpan plot Top 10 Feature Importance
top_features = feature_importance_df.head(10)
plt.figure(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=top_features)
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "top10_feature_importance.png"))
plt.close()
logging.info(f"Grafik Top 10 Feature Importances disimpan di {os.path.join(PLOTS_DIR, 'top10_feature_importance.png')}")

# SHAP Analysis
logging.info("Menjelaskan model menggunakan SHAP...")
explainer = shap.TreeExplainer(best_model)

# Gunakan API objek Explanation yang lebih modern dan robust
explanation = explainer(X_test_scaled_df)

# Plot summary SHAP untuk kelas 'malicious' (1)
plt.figure()
# Gunakan slicing pada objek Explanation untuk mendapatkan nilai SHAP kelas 1
shap.summary_plot(explanation[:,:,1], X_test_scaled_df, show=False)
plt.title("SHAP Summary Plot (Malicious Class)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "shap_summary.png"))
plt.close()
logging.info(f"SHAP summary plot disimpan di {os.path.join(PLOTS_DIR, 'shap_summary.png')}")
plt.close('all')


# --- 8. Model Saving ---
logging.info("7. Menyimpan model final dan artefak...")
model_bundle = {
    "model": best_model,
    "encoders": encoders,
    "imputer": imputer,
    "scaler": scaler,
    "feature_names": feature_names
}
joblib.dump(model_bundle, MODEL_PATH)

logging.info(f"--- Pipeline Training Selesai. Model disimpan di {MODEL_PATH} ---")