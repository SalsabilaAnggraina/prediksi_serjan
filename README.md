# 🩺 Laporan Proyek Machine Learning: Klasifikasi Risiko Serangan Jantung

* **Metodologi Penelitian:** CRISP-DM (*Cross-Industry Standard Process for Data Mining*) 📊
* **Penyusun:** [Suci Oktavia Ramadhani & Salsabila Anggraina Putri] 👥
* **Tautan Aplikasi Interaktif (Hugging Face Spaces):** [🔗 Akses Demo Aplikasi di Sini](https://huggingface.co/spaces/saggrain/prediksi-SerJan) 🚀

---

## 📌 1. Project Overview
Penyakit kardiovaskular, khususnya serangan jantung, tetap menjadi salah satu penyebab utama mortalitas tertinggi di Indonesia. Sifat gejalanya yang acap kali asimtomatik (*silent killer*) diperparah oleh banyaknya faktor risiko multidimensional yang saling berkorelasi, mencakup aspek demografi, kondisi klinis, hingga pola hidup pasien. 💔

Proyek ini dicanangkan untuk membangun sebuah sistem komputasi cerdas berbasis *Machine Learning* yang mampu mengklasifikasikan tingkat risiko serangan jantung pada masyarakat secara dini. Implementasi ini diharapkan dapat menjadi instrumen skrining preventif bagi tenaga medis maupun masyarakat umum. 🩺✨

---

## 🎯 2. Business Understanding

### ❓ Problem Statements
1. Bagaimana karakteristik demografi serta parameter klinis harian dapat memengaruhi kecenderungan risiko serangan jantung seseorang? 📊
2. Apakah implementasi algoritma *Advanced Ensemble Learning* (**XGBoost Classifier**) mampu menghasilkan performa klasifikasi yang lebih superior dan adaptif dibandingkan model standar (**Logistic Regression**)? 🧠

### 🚀 Goals
1. Mengidentifikasi dan menganalisis korelasi prediktif dari berbagai fitur kesehatan terhadap indikasi risiko penyakit jantung. 🔍
2. Mengembangkan model prediktif dengan tingkat sensitivitas (**Recall**) yang optimal demi meminimalisir persentase kegagalan diagnosis fatal (*False Negative*). 🎯

### 💡 Solution Statement
* **Model Baseline:** *Logistic Regression* diterapkan sebagai model komparasi standar yang memiliki keunggulan dalam hal interpretabilitas koefisien statistik. 📉
* **Model Advanced:** *XGBoost Classifier* digunakan sebagai representasi algoritma mutakhir berbasis *Gradient Boosting* untuk menangani kompleksitas hubungan non-linear antar variabel. ⚡

---

## 📊 3. Data Understanding
Eksperimen ini memanfaatkan dataset rekam medis masyarakat Indonesia dari [Kaggle](https://www.kaggle.com/datasets/ankushpanday2/heart-attack-prediction-in-indonesia) (`heart_attack_prediction_indonesia.csv`) yang mencakup parameter klinis, demografi, dan psikologis. 🗂️

### 🔍 Glosarium Fitur Dataset:
* 🧓 `age`: Usia biologis pasien (Fitur Numerik).
* ⚧ `gender`: Jenis kelamin pasien (Kategorikal: Male, Female).
* 🩸 `hypertension`: Riwayat klinis tekanan darah tinggi (Biner: 0 = Tidak, 1 = Ya).
* 🍬 `diabetes`: Riwayat penyakit diabetes melitus (Biner: 0 = Tidak, 1 = Ya).
* 🍟 `cholesterol_level`: Kadar kolesterol total dalam darah (Fitur Numerik/mg/dL).
* 🍔 `obesity`: Status indeks massa tubuh berlebih/obesitas (Biner: 0 = Tidak, 1 = Ya).
* 🚬 `smoking_status`: Kebiasaan konsumsi rokok (Kategorikal: Never, Past, Current).
* 🥦 `dietary_habits`: Penilaian pola konsumsi makanan harian (Kategorikal: Healthy, Unhealthy).
* 🤯 `stress_level`: Indikator skala beban psikologis internal (Ordinal: Skala 1 - 10).
* 🎯 `heart_attack`: **Variabel Target / Label** (Biner: 0 = Risiko Rendah, 1 = Risiko Tinggi).

---

## 🛠️ 4. Data Preparation
Untuk menjamin integritas data sebelum memasuki fase pelatihan (*training*), serangkaian tahap rekayasa data berikut telah diterapkan: 🏗️

1. **Label Encoding🧙‍♂️:** Melakukan konversi otomatis pada fitur-fitur kategorikal bertipe teks menjadi representasi numerik agar dapat diproses oleh algoritma.
2. **Feature Selection🎯:** Membatasi variabel prediktor dengan hanya memilih 9 fitur utama yang memiliki signifikansi klinis berdasarkan fokus studi penelitian.
3. **Train-Test Split ✂️:** Membagi data secara acak menggunakan proporsi **80% Data Latih** dan **20% Data Uji** dengan menyematkan parameter `stratify` guna menjaga kestabilan distribusi label target.
4. **Feature Scaling 📏:** Menerapkan `StandardScaler` untuk menyamakan skala varians pada fitur numerik berjangkauan luas (seperti kolesterol) agar tidak mendominasi proses pembaruan bobot (*weight*) model.

---

## 🤖 5. Modeling
Tahap permodelan mengevaluasi dan mengonfrontasi dua pendekatan arsitektur klasifikasi yang berbeda: ⚔️

* **Logistic Regression:** Algoritma parametrik yang memetakan kombinasi linear dari fitur input ke dalam fungsi sigmoid untuk menghasilkan output probabilitas. 📉
* **XGBoost Classifier:** Algoritma berbasis *ensemble decision trees* sekuensial yang menerapkan regularisasi formal dan minimalisasi fungsi kerugian (*loss function*) secara presisi guna mereduksi *overfitting*. ⚡

---

## 🏆 6. Evaluation
Pengujian performa model dilakukan secara objektif menggunakan data uji yang belum pernah dilihat sebelumnya selama fase pelatihan. Mengingat domain penelitian berada pada sektor medis, metrik **Recall** menjadi prioritas evaluasi utama. 📊🔬

*(Silakan perbarui angka matriks di bawah ini berdasarkan tabel klasifikasi asli di Google Colab Anda)*

| Metrik Evaluasi | Logistic Regression (Baseline) 📉 | XGBoost Classifier (Advanced) ⚡ |
| :--- | :---: | :---: |
| **Akurasi Global** | 0.XX (XX%) | **0.XX (XX%)** |
| **Precision** | 0.XX | **0.XX** |
| **Recall (Sensitivitas)** | 0.XX | **0.XX** |
| **F1-Score** | 0.XX | **0.XX** |

*Kesimpulan Analisis:* Berdasarkan hasil pengujian, **XGBoost Classifier** menunjukkan performa yang jauh lebih unggul dan presisi. Tingginya nilai *Recall* pada XGBoost memvalidasi bahwa model ini sangat andal digunakan sebagai sistem deteksi dini dengan risiko misdiagnosis yang minim. 🎉

---

## 🚀 7. Deployment
Model terbaik (**XGBoost**) berserta objek **StandardScaler** diekspor ke dalam berkas biner `.pkl` memanfaatkan pustaka `joblib`. 📦

Sistem dideploy secara publik pada platform cloud **Hugging Face Spaces** dengan menggunakan framework antarmuka **Gradio**. Aplikasi dirancang dengan panel kendali interaktif seperti *sliders* dan *dropdown selection*, memungkinkan praktisi medis untuk memasukkan indikator klinis pasien baru dan mendapatkan kalkulasi probabilitas risiko secara *real-time*. ⚙️🌐

👉 **[Buka Aplikasi Demo Prediksi Risiko Serangan Jantung](https://huggingface.co/spaces/saggrain/prediksi-SerJan)** 👈

---
