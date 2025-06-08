from flask import Flask, render_template, redirect, url_for
from nlp.ekstraksi_entitas_lokasi_tanggal_dan_waktu import ner_with_chunking_and_cleaning, extract_ner_results
from scraper.live_scraping import scrape_all_sources
import pandas as pd
from geopy.geocoders import Nominatim
import folium
from folium.plugins import MarkerCluster
import time

app = Flask(__name__)

# Inisialisasi df_ner sebagai DataFrame kosong
df_ner = pd.DataFrame()



# Fungsi untuk mendapatkan koordinat lokasi dengan jeda waktu
def get_coordinates(location):
    geolocator = Nominatim(user_agent="disaster_locator")
    try:
        time.sleep(1)  # Jeda waktu 1 detik untuk menghindari rate limit Nominatim
        location_data = geolocator.geocode(location)
        if location_data:
            print(f"Geocoded {location}: {location_data.latitude}, {location_data.longitude}")
            return location_data.latitude, location_data.longitude
        else:
            print(f"Geocoding failed for: {location}")
            return None, None
    except Exception as e:
        print(f"Error geocoding {location}: {e}")
        return None, None

# Fungsi untuk membuat peta berdasarkan df_ner
def create_map(df_ner):
    print("Mencari koordinat lokasi...")
    df_ner[['Latitude', 'Longitude']] = df_ner['Location'].apply(lambda x: pd.Series(get_coordinates(x)))

    print("Membuat peta bencana...")
    m = folium.Map(location=[-2.5489, 118.0149], zoom_start=5)  # Fokus ke Indonesia
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in df_ner.iterrows():
        if pd.notnull(row['Latitude']) and pd.notnull(row['Longitude']):
            popup_text = f"""
            <b>Judul:</b> {row['Title']}<br>
            <b>Kategori:</b> {row['Category']}<br>
            <b>Lokasi:</b> {row['Location']}<br>
            <b>Tanggal:</b> {row['Date']}<br>
            <b>Waktu:</b> {row['Time']}<br>
            <a href='{row['URL']}' target='_blank'>Baca Selengkapnya</a>
            """
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color="red" if "Banjir" in row['Category'] else "blue")
            ).add_to(marker_cluster)

    map_path = "static/map_bencana.html"
    m.save(map_path)
    return map_path

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime

# Unduh stopwords untuk bahasa Indonesia
nltk.download('stopwords')
stop_words = stopwords.words('indonesian')

# Fungsi untuk membersihkan teks pada konten
def clean_text(text):
    """
    Membersihkan teks pada kolom Content:
    - Mengubah ke huruf kecil
    - Menghapus angka
    - Menghapus spasi berlebih
    """
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_text(text):
    """
    Membersihkan teks pada kolom Content:
    - Mengubah ke huruf kecil
    - Menghapus angka
    - Menghapus stopwords
    - Menghapus spasi berlebih
    """
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def preprocess_decimal_points(text):
    """
    Mengganti titik dalam angka desimal dengan placeholder <DECIMAL>.
    Contoh: '20.46' menjadi '20<DECIMAL>46'
    """
    text = re.sub(r'(\d)\.(\d)', r'\1<DECIMAL>\2', text)
    return text

def preprocess_quoted_dots(text):
    """
    Mengganti titik dalam kalimat kutipan dengan placeholder <QUOTE_DOT>.
    Contoh: '"Informasi ini penting."' menjadi '"Informasi ini penting<QUOTE_DOT>"'
    """
    text = re.sub(r'\.(?=\")', r'<QUOTE_DOT>', text)
    return text

def preprocess_special_cases(text):
    """
    Menangani kasus khusus lainnya, seperti titik setelah singkatan dalam tanda kurung.
    Contoh: 'Senin (1/1).' menjadi 'Senin (1/1)'
    """
    text = re.sub(r'\((\d+)/(\d+)\)\.', r'(\1/\2)', text)
    return text

def preprocess_text(text):
    """
    Melakukan semua langkah preprocessing pada teks:
    - Mengganti titik dalam angka desimal dengan placeholder <DECIMAL>.
    - Mengganti titik dalam kalimat kutipan dengan placeholder <QUOTE_DOT>.
    - Menangani kasus khusus lainnya seperti titik setelah singkatan dalam tanda kurung.
    """
    text = preprocess_decimal_points(text)
    text = preprocess_quoted_dots(text)
    text = preprocess_special_cases(text)
    return text

def postprocess_decimal_points(sentences):
    """
    Mengembalikan placeholder <DECIMAL> dan <QUOTE_DOT> menjadi titik.
    """
    if isinstance(sentences, list):
        sentences = [sentence.replace('<DECIMAL>', '.') for sentence in sentences]
        sentences = [sentence.replace('<QUOTE_DOT>', '.') for sentence in sentences]
    elif isinstance(sentences, str):
        sentences = sentences.replace('<DECIMAL>', '.').replace('<QUOTE_DOT>', '.')
    else:
        raise ValueError("Input harus berupa string atau list of strings.")
    return sentences

def sentence_tokenize(text):
    """
    Tokenize kalimat menggunakan NLTK, menangani titik dalam angka desimal dan kalimat kutipan:
    - Melakukan preprocessing untuk menangani titik dalam angka desimal dan tanda baca dalam kutipan.
    - Menggunakan NLTK untuk memisahkan teks menjadi kalimat.
    - Mengembalikan titik dalam angka desimal dan tanda baca dalam kutipan setelah tokenisasi.
    """
    text = preprocess_text(text)  # Preprocessing
    sentences = sent_tokenize(text)  # Tokenisasi menggunakan NLTK
    sentences = postprocess_decimal_points(sentences)  # Postprocessing
    return sentences




# Route Home
@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html", data=None)  # Default tanpa data

# Route untuk menampilkan peta
@app.route("/map")
def map():
    global df_ner  # Menggunakan variabel global
    if df_ner.empty:  # Pastikan df_ner sudah tersedia
        return "Data belum tersedia. Silakan proses data terlebih dahulu di halaman utama."
    map_file = create_map(df_ner)  # Membuat peta berdasarkan df_ner
    return render_template("map.html", map_file=map_file)

# Route untuk proses NER
@app.route("/process", methods=["POST"])
def process():
    global df_ner
    print("Scraping data...")
    scraped_articles = scrape_all_sources()

    all_data = pd.DataFrame(scraped_articles)
    print("Memproses hasil NER...")
    ner_results_list = []
    for _, row in all_data.iterrows():
        results = ner_with_chunking_and_cleaning(row["Content"])
        extracted = extract_ner_results(results)
        ner_results_list.append({
            "Title": row["Title"],
            "Category": row["Category"],
            "Location": extracted["Location"],
            "Date": extracted["Date"],
            "Time": extracted["Time"],
            "URL": row["URL"]
        })

    # Simpan hasil NER ke DataFrame global
    df_ner = pd.DataFrame(ner_results_list)
    print("NER Results:", df_ner)

    return render_template("index.html", data=ner_results_list)

if __name__ == "__main__":
    app.run(debug=True)
