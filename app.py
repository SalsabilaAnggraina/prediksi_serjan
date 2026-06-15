import gradio as gr
import numpy as np
import pandas as pd
import joblib

# =====================================================================
# 1. MEMUAT MODEL DAN SCALER (.PKL)
# =====================================================================
# Pastikan file model_SerJan_xgb.pkl dan scaler.pkl diunggah di folder yang sama di Hugging Face
model = joblib.load('model_SerJan_xgb.pkl')
scaler = joblib.load('scaler.pkl')

# =====================================================================
# 2. FUNGSI PREDIKSI UTAMA
# =====================================================================
def prediksi_serangan_jantung(age, gender, hypertension, diabetes, cholesterol, obesity, smoking, diet, stress):
    # Mapping data teks dari UI kembali ke format angka (Label Encoding sesuai data latihan)
    gender_val = 1 if gender == "Laki-laki" else 0
    hyper_val = 1 if hypertension == "Ya" else 0
    diabetes_val = 1 if diabetes == "Ya" else 0
    obesity_val = 1 if obesity == "Ya" else 0
    
    # Mapping smoking_status (0=Current, 1=Never, 2=Past)
    if smoking == "Perokok Aktif":
        smoking_val = 0
    elif smoking == "Tidak Pernah":
        smoking_val = 1
    else:
        smoking_val = 2
        
    # Mapping dietary_habits (0=Healthy, 1=Unhealthy)
    diet_val = 0 if diet == "Sehat" else 1

    # Gabungkan semua input menjadi array 2 dimensi sesuai urutan fitur latihan
    input_data = np.array([[
        age, gender_val, hyper_val, diabetes_val, 
        cholesterol, obesity_val, smoking_val, diet_val, stress
    ]])
    
    # WAJIB: Transformasikan data input baru menggunakan scaler yang sudah dilatih
    input_data_scaled = scaler.transform(input_data)
    
    # Melakukan prediksi dan menghitung probabilitas risiko
    hasil_prediksi = model.predict(input_data_scaled)[0]
    probabilitas = model.predict_proba(input_data_scaled)[0][1]
    
    # Menyusun output tampilan berdasarkan hasil diagnosis
    if hasil_prediksi == 1:
        status = "🚨 BERISIKO TINGGI SERANGAN JANTUNG"
        rekomendasi = "⚠️ Sangat disarankan untuk segera berkonsultasi dengan dokter spesialis jantung, membatasi konsumsi makanan berkolesterol tinggi, serta berolahraga secara teratur dengan pengawasan medis."
    else:
        status = "✅ RISIKO RENDAH / AMAN"
        rekomendasi = "🌿 Pertahankan pola hidup sehat Anda! Tetap konsumsi makanan bergizi seimbang, hindari stres berlebih, jauhi rokok, dan lakukan medical check-up secara berkala."
        
    persentase_risiko = f"{probabilitas * 100:.2f}%"
    
    return status, persentase_risiko, rekomendasi

# =====================================================================
# 3. ANTARMUKA PENGGUNA (GRADIO UI APP)
# =====================================================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🩺 Aplikasi Deteksi Dini Risiko Serangan Jantung
        Sistem ini dirancang menggunakan algoritma **XGBoost Classifier** berbasis metodologi **CRISP-DM**. Aplikasi ini mengklasifikasikan tingkat risiko serangan jantung seseorang berdasarkan profil demografi, indikator medis, dan kebiasaan gaya hidup.
        """
    )
    
    with gr.Row():
        # Kolom Kiri untuk Form Input Data Pasien
        with gr.Column():
            gr.Markdown("### 📋 Form Input Data Pasien")
            age = gr.Slider(minimum=1, maximum=100, value=45, step=1, label="Usia (Tahun)")
            gender = gr.Dropdown(choices=["Laki-laki", "Perempuan"], value="Laki-laki", label="Jenis Kelamin")
            hypertension = gr.Radio(choices=["Ya", "Tidak"], value="Tidak", label="Apakah memiliki riwayat Hipertensi?")
            diabetes = gr.Radio(choices=["Ya", "Tidak"], value="Tidak", label="Apakah memiliki riwayat Diabetes?")
            cholesterol = gr.Slider(minimum=100, maximum=400, value=210, step=5, label="Kadar Kolesterol Total (mg/dL)")
            obesity = gr.Radio(choices=["Ya", "Tidak"], value="Tidak", label="Apakah mengalami Obesitas?")
            smoking = gr.Dropdown(choices=["Tidak Pernah", "Mantan Perokok", "Perokok Aktif"], value="Tidak Pernah", label="Status Kebiasaan Merokok")
            diet = gr.Radio(choices=["Sehat", "Tidak Sehat"], value="Sehat", label="Pola Diet / Makan Sehari-hari")
            stress = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Skala Tingkat Stres Pribadi (1-10)")
            
            btn_submit = gr.Button("Mulai Analisis Medis", variant="primary")
            
        # Kolom Kanan untuk Menampilkan Hasil Prediksi
        with gr.Column():
            gr.Markdown("### 📊 Hasil Analisis Diagnosa Model")
            out_status = gr.Textbox(label="Status Hasil Klasifikasi", interactive=False)
            out_prob = gr.Textbox(label="Probabilitas Risiko Terkena", interactive=False)
            out_saran = gr.Textbox(label="Rekomendasi Tindakan Medis", interactive=False)
            
    # Menghubungkan tombol dengan fungsi prediksi
    btn_submit.click(
        fn=prediksi_serangan_jantung,
        inputs=[age, gender, hypertension, diabetes, cholesterol, obesity, smoking, diet, stress],
        outputs=[out_status, out_prob, out_saran]
    )

# Menjalankan aplikasi demo
if __name__ == "__main__":
    demo.launch()