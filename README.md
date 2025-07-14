# 🧠 Analisis Sentimen Ulasan Shopee & Tokopedia (Bahasa Indonesia)

Proyek ini membangun **model AI untuk mengklasifikasikan sentimen ulasan pengguna** dari dua platform e-commerce terbesar di Indonesia: **Shopee** dan **Tokopedia**, menggunakan pendekatan NLP (Natural Language Processing) berbasis TF-IDF dan Naive Bayes.

---

## 🚀 Tools & Teknologi
- Python (Pandas, NumPy)
- Scikit-learn (TF-IDF, Naive Bayes)
- Sastrawi (Stemming Bahasa Indonesia)
- Matplotlib, Seaborn, WordCloud
- Streamlit (Deploy UI model interaktif)

---

## 📦 Dataset
Dataset diambil dari **Kaggle** yang sudah dibalance (200 ulasan positif dan 200 negatif per platform), dengan struktur kolom:
- `rating`: 1–5 (digunakan untuk labeling)
- `clean`: ulasan yang sudah dibersihkan
- `likes`: jumlah orang yang menyukai ulasan
- `platform`: Shopee atau Tokopedia

---

## 📊 Proses Utama
1. Preprocessing teks (`clean`, case folding, stopword removal, stemming)
2. Labeling otomatis berdasarkan `rating`
   - Rating ≤ 2 → **Negatif**
   - Rating ≥ 4 → **Positif**
   - Rating = 3 → Dibuang
3. TF-IDF Vectorization
4. Model klasifikasi dengan Naive Bayes
5. Evaluasi: Akurasi, Confusion Matrix
6. Deploy ke Streamlit + Insight visual

---

## 🖥️ Aplikasi Demo – Streamlit

### 🎯 Fitur:
- Input teks ulasan → Prediksi sentimen otomatis
- WordCloud untuk ulasan **positif** dan **negatif**
- Visualisasi rata-rata **jumlah likes** per sentimen
- Analisis topik umum keluhan (Shopee vs Tokopedia)

### 📷 Screenshot Aplikasi
<img width="1919" height="775" alt="image" src="https://github.com/user-attachments/assets/b265f5d0-a084-4351-8eae-6922d298a973" />


---

## 📈 Insight Menarik
> *Note: Dataset bersifat seimbang, insight tetap relevan dalam konteks evaluasi model.*

- Ulasan **positif** memiliki rata-rata jumlah likes yang lebih tinggi dibanding positif
---


## 💡 Contoh Prediksi:
```text
Input: "Aplikasi lemot banget, sering error"
Output: Prediksi → NEGATIF

