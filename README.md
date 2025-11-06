<p align="center"><img src="https://socialify.git.ci/prasbara/ml-threat-detector/image?language=1&amp;owner=1&amp;name=1&amp;stargazers=1&amp;theme=Light" alt="project-image"></p>

<p id="description">ML Threat Detector adalah proyek machine learning pipeline untuk mendeteksi aktivitas berbahaya (malicious) pada sistem robotik dan jaringan telesurgery. Model ini melakukan klasifikasi event keamanan menjadi dua kategori utama: - Benign (Normal) - Malicious (Ancaman)</p>

<p align="center"><img src="https://img.shields.io/badge/Python-3.13-blue?logo=python&amp;logoColor=white" alt="shields"><img src="https://img.shields.io/badge/Dataset-telesurgery__cybersecurity-lightgrey?logo=database" alt="shields"><img src="https://img.shields.io/badge/Threat%20Detected-Malicious-red?logo=shield" alt="shields"><img src="https://img.shields.io/badge/Build-passing-brightgreen?logo=githubactions" alt="shields"><img src="https://img.shields.io/badge/SOC_Engineer-sleep%20deprived-lightblue?logo=coffee" alt="shields"></p>

  
  
<h2>🧐 Features</h2>

Here're some of the project's best features:

*   Threat Classification
*   Model Optimization

<h2>🛠️ Installation Steps:</h2>

<p>1. Instal dependensi</p>

```
pip install -r requirements.txt
```

<p>2. Jalankan Training Model</p>

```
python src/train_model.py
```

<p>3. Jalankan Prediksi Default</p>

```
python src/predict.py
```

<p>4. Jalankan Prediksi dari File CSV</p>

```
python src/predict.py "data/robot_security_events.csv"
```

  
  
<h2>💻 Built with</h2>

Technologies used in the project:

*   Python 3.10+
*   scikit-learn
*   pandas / numpy
*   matplotlib / seaborn / shap
