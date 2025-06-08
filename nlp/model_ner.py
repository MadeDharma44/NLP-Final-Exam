from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import re
from datetime import datetime

# Load model NER
tokenizer = AutoTokenizer.from_pretrained("syafiqfaray/indobert-model-ner")
model = AutoModelForTokenClassification.from_pretrained("syafiqfaray/indobert-model-ner")
ner_pipe = pipeline("token-classification", model=model, tokenizer=tokenizer)

# Preprocessing teks
def preprocess_text(text):
    text = re.sub(r'\n+', ' ', text)  # Hapus newline
    text = re.sub(r'\s+', ' ', text).strip()  # Hapus spasi berlebih
    return text

# Fungsi NER dengan chunking dan cleaning
def ner_with_chunking_and_cleaning(text, max_length=512):
    def split_text_into_chunks(text, max_length):
        tokens = tokenizer.encode(text, truncation=False)
        chunks = []
        for i in range(0, len(tokens), max_length - 2):
            chunk = tokens[i:i + max_length - 2]
            chunk = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
            chunks.append(chunk)
        return chunks

    def clean_ner_results(results):
        cleaned_results = []
        for result in results:
            word = result['word']
            if word.startswith("##"):
                if cleaned_results:
                    cleaned_results[-1]['word'] += word[2:]
            else:
                cleaned_results.append(result)
        return cleaned_results

    chunks = split_text_into_chunks(text, max_length)
    all_results = []
    for chunk in chunks:
        decoded_chunk = tokenizer.decode(chunk, skip_special_tokens=True)
        ner_results = ner_pipe(decoded_chunk)
        cleaned_chunk_results = clean_ner_results(ner_results)
        all_results.extend(cleaned_chunk_results)

    return all_results

# Memproses hasil NER menjadi lokasi, tanggal, dan waktu
def insert_ner_results_to_df(results):
    location = []
    date = None
    time = None

    for result in results:
        entity = result.get("entity", "").upper()
        word = result['word'].title()

        if "LOC" in entity or "GPE" in entity:  # Lokasi
            location.append(word)
        elif "DATE" in entity or "DAT" in entity:  # Tanggal
            date = word
        elif "TIME" in entity or "TIM" in entity:  # Waktu
            time = word

    return {
        "Location": ", ".join(location) if location else "Tidak Ditemukan",
        "Date": date if date else "Tidak Ditemukan",
        "Time": time if time else "Tidak Ditemukan"
    }

# Integrasi dengan hasil scraping
def process_articles(articles):
    processed_data = []
    for article in articles:
        content = preprocess_text(article['Content'])  # Preprocessing
        ner_results = ner_with_chunking_and_cleaning(content)  # NER
        ner_data = insert_ner_results_to_df(ner_results)  # Hasil NER

        # Gabungkan data
        processed_data.append({
            "Title": article['Title'],
            "Category": article['Category'],
            "Location": ner_data['Location'],
            "Date": ner_data['Date'],
            "Time": ner_data['Time'],
            "URL": article['URL']
        })
    return processed_data
