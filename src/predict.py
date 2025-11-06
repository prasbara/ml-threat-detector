import joblib
import pandas as pd
import os
import numpy as np

# --- 1. Konfigurasi & Muat Artefak ---
print("--- ML Threat Detector Prediction Pipeline ---")
MODEL_PATH = 'd:\\SOC PORTO\\ml-threat-detector\\models\\threat_model.pkl'

print(f"1. Memuat bundel model dari {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"File model tidak ditemukan di {MODEL_PATH}. Jalankan train_model.py terlebih dahulu.")

model_bundle = joblib.load(MODEL_PATH)

# Ekstrak semua komponen dari bundel
model = model_bundle["model"]
encoders = model_bundle["encoders"]
imputer = model_bundle["imputer"]
scaler = model_bundle["scaler"]
feature_names = model_bundle["feature_names"]

print("   -> Model, encoders, imputer, scaler, dan feature_names berhasil dimuat.")

def predict_security_event(event_data):
    """
    Memprediksi apakah sebuah event keamanan bersifat 'Benign' atau 'Malicious'
    menggunakan pipeline yang sudah dilatih.
    """
    try:
        # --- 2. Pra-pemrosesan Input ---
        input_df = pd.DataFrame([event_data])

        # a. Encode 'Threat Severity'
        if 'Threat Severity' in input_df.columns:
            severity_encoder = encoders['Threat Severity']
            input_df['Threat Severity'] = input_df['Threat Severity'].apply(lambda x: severity_encoder.transform([x])[0] if x in severity_encoder.classes_ else -1)

        # b. Lakukan Feature Engineering
        input_df['Interaction_Latency_Response'] = input_df['Network Latency (ms)'] * input_df['Response Time (sec)']
        input_df['Data_Volume'] = input_df['Data Transfer Rate (Mbps)'] * input_df['Gesture Duration (sec)']
        input_df['Severity_to_Response'] = input_df['Threat Severity'] / (input_df['Response Time (sec)'] + 0.1)
        input_df['Latency_to_Rate'] = input_df['Network Latency (ms)'] / (input_df['Data Transfer Rate (Mbps)'] + 1)

        # c. Terapkan sisa encoder
        for col, encoder in encoders.items():
            if col != 'Threat Severity' and col in input_df.columns:
                input_df[col] = input_df[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)

        # --- 3. Samakan Urutan, Imputasi, & Scaling ---
        input_df = input_df.reindex(columns=feature_names, fill_value=0)
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)

        # --- 4. Lakukan Prediksi ---
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)

        # --- 5. Kembalikan Hasil ---
        result = 'Malicious' if prediction[0] == 1 else 'Benign'
        return result, probability[0][prediction[0]]

    except Exception as e:
        # Menggunakan print untuk error di sini agar pasti terlihat
        print(f"Error saat prediksi: {e}")
        return None, None

# --- Contoh Penggunaan ---
if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='ML Threat Detector Prediction Pipeline')
    parser.add_argument('file', nargs='?', default=None, help='Path to the CSV file to predict on. If not provided, runs a default prediction.')
    args = parser.parse_args()

    if args.file:
        print(f"--- Memulai Prediksi dari File: {args.file} ---")
        try:
            input_data = pd.read_csv(args.file)
            # Ganti nama kolom untuk konsistensi jika perlu
            input_data.columns = input_data.columns.str.strip()

            results = []
            for index, row in input_data.iterrows():
                event_data = row.to_dict()
                prediction, prob = predict_security_event(event_data)
                if prediction:
                    results.append((index, prediction, prob))
            
            # Tampilkan hasil dalam format tabel
            print("\n--- Hasil Prediksi ---")
            print(f"{'Index':<6} {'Prediksi':<12} {'Confidence':<12}")
            print("="*32)
            for index, prediction, prob in results:
                print(f"{index:<6} {prediction:<12} {prob:<12.2%}")

        except FileNotFoundError:
            print(f"Error: File tidak ditemukan di {args.file}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error saat memproses file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Contoh data input jika tidak ada file yang diberikan
        print("\n--- Menjalankan Prediksi Default ---")
        default_event = {
            'Gesture Type': 'Incision', 'Gesture Duration (sec)': 4.37,
            'Robot Status': 'Idle', 'Encryption Algorithm Used': 'Two Fish',
            'Encryption Status': 'Encrypted', 'Network Latency (ms)': 11,
            'Data Transfer Rate (Mbps)': 97, 'Threat Type': 'Man-in-the-Middle Attack',
            'Threat Severity': 'Low', 'Response Time (sec)': 4.63,
            'Response Action Taken': 'Reset Encryption'
        }
        
        prediction, prob = predict_security_event(default_event)
        if prediction:
            print(f"Hasil Prediksi Event: {prediction} (Confidence: {prob:.2%})")
