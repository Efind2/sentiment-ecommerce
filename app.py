import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# ============================
# 1. Load Model & Dataset
# ============================
with open('model/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('model/naive_bayes_model.pkl', 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv('data/final_dataset.csv')
review_df = pd.read_csv('data/review_shopee_tokopedia_merged.csv')

# ============================
# 2. Lexicon Sentimen
# ============================
lexicon_positif = {
    'bagus', 'cepat', 'puas', 'murah', 'mantap', 'nyaman', 'suka', 'senang', 'oke', 'baik',
    'memuaskan', 'rekomendasi', 'top', 'mantul', 'keren', 'hebat', 'helpful'
}
lexicon_negatif = {
    'lama', 'buruk', 'kecewa', 'jelek', 'lambat', 'batal', 'hilang', 'parah', 'lelet',
    'ribet', 'susah', 'gagal', 'rusak', 'salah', 'tidak', 'mengecewakan', 'ngecewain'
}

# ============================
# 3. Fungsi Filter Lexicon
# ============================
def filter_lexicon(text, lexicon):
    return ' '.join([word for word in text.split() if word in lexicon])

# ============================
# 4. Streamlit UI & Prediksi
# ============================
st.title("💬 Klasifikasi Sentimen Ulasan (Indo)")

input_text = st.text_area("Masukkan ulasan:")

if st.button("Prediksi Sentimen"):
    if input_text.strip() == "":
        st.warning("Teks tidak boleh kosong.")
    else:
        clean_text = input_text.lower()
        X = vectorizer.transform([clean_text])
        pred = model.predict(X)[0]
        st.success(f"Prediksi Sentimen: **{pred.upper()}**")

# ============================
# 5. WordCloud Visualisasi
# ============================
st.subheader("☁️ WordCloud Sentimen (berbasis Lexicon)")

# Gabungkan teks dan filter
pos_raw = ' '.join(df[df['label'] == 'positif']['cleaned'].astype(str))
neg_raw = ' '.join(df[df['label'] == 'negatif']['cleaned'].astype(str))
pos_filtered = filter_lexicon(pos_raw, lexicon_positif)
neg_filtered = filter_lexicon(neg_raw, lexicon_negatif)

# WordCloud Positif
wc_pos = WordCloud(width=800, height=400, background_color='white').generate(pos_filtered)
fig_pos, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wc_pos, interpolation='bilinear')
ax.axis('off')
ax.set_title('Kata Populer Sentimen POSITIF (Lexicon)')
st.pyplot(fig_pos)

# WordCloud Negatif
wc_neg = WordCloud(width=800, height=400, background_color='white').generate(neg_filtered)
fig_neg, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wc_neg, interpolation='bilinear')
ax.axis('off')
ax.set_title('Kata Populer Sentimen NEGATIF (Lexicon)')
st.pyplot(fig_neg)

# ============================
# 6. Analisis Likes per Sentimen
# ============================
st.subheader("📊 Rata-rata Likes per Sentimen")

# Hitung rata-rata likes
avg_likes = review_df.groupby('label')['likes'].mean().round(2)

# Tampilkan sebagai tabel
st.write(avg_likes)

# Tampilkan sebagai bar chart
fig, ax = plt.subplots()
avg_likes.plot(kind='bar', color=['green', 'red'], ax=ax)
ax.set_title("Rata-rata Likes per Sentimen")
ax.set_ylabel("Likes")
ax.set_xlabel("Sentimen")
ax.tick_params(axis='x', rotation=0)
st.pyplot(fig)

# ============================
# 7. WordCloud Negatif berdasarkan Platform (warna lebih soft)
# ============================
st.subheader("💢 WordCloud Negatif berdasarkan Platform")

# Shopee
negatif_shopee = ' '.join(df[(df['label'] == 'negatif') & (df['platform'] == 'shopee')]['cleaned'].astype(str))
wc_shopee = WordCloud(width=800, height=400, background_color='white', colormap='Oranges').generate(negatif_shopee)
fig_shopee, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wc_shopee, interpolation='bilinear')
ax.axis('off')
ax.set_title('Kata Populer Ulasan Negatif – Shopee', fontsize=14)
st.pyplot(fig_shopee)

# Tokopedia
negatif_tokopedia = ' '.join(df[(df['label'] == 'negatif') & (df['platform'] == 'tokopedia')]['cleaned'].astype(str))
wc_tokopedia = WordCloud(width=800, height=400, background_color='white', colormap='PuBu').generate(negatif_tokopedia)
fig_tokopedia, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wc_tokopedia, interpolation='bilinear')
ax.axis('off')
ax.set_title('Kata Populer Ulasan Negatif – Tokopedia', fontsize=14)
st.pyplot(fig_tokopedia)
