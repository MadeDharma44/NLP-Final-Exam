# Import library
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import pandas as pd
import re
from datetime import datetime

# Load IndoBERT NER pipeline dan model
pipe = pipeline("token-classification", model="syafiqfaray/indobert-model-ner")
tokenizer = AutoTokenizer.from_pretrained("syafiqfaray/indobert-model-ner")
model = AutoModelForTokenClassification.from_pretrained("syafiqfaray/indobert-model-ner")

# Fungsi NER dengan handling chunking
def ner_with_chunking_and_cleaning(text, max_length=512):
    """
    Melakukan NER pada teks panjang dengan chunking dan pembersihan token ##.
    """
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

    pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    chunks = split_text_into_chunks(text, max_length)
    all_results = []
    for chunk in chunks:
        decoded_chunk = tokenizer.decode(chunk, skip_special_tokens=True)
        ner_results = pipe(decoded_chunk)
        cleaned_chunk_results = clean_ner_results(ner_results)
        all_results.extend(cleaned_chunk_results)
    return all_results

# Fungsi untuk mengekstrak hasil NER
def extract_ner_results(results):
    location = []
    date = None
    time = None

    gpe_sequence = []
    gpe_found = False
    for result in results:
        entity_group = result.get("entity_group", None)
        word = result.get("word", None)
        if entity_group == "GPE":
            if not gpe_found:
                gpe_found = True
            if len(gpe_sequence) < 4:
                gpe_sequence.append(word)
            else:
                break
        elif gpe_found:
            break

    location = ", ".join([loc.title() for loc in gpe_sequence])

    for result in results:
        if result.get("entity_group") == "DAT" and date is None:
            word = result.get("word", None)
            try:
                match = re.search(r"(\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{4})", word)
                if match:
                    cleaned_word = match.group(1).replace(" ", "")
                    date_obj = datetime.strptime(cleaned_word, "%d/%m/%Y")
                    date = date_obj.strftime("%d %B %Y")
                else:
                    match = re.search(r"(\d{1,2}\s*/\s*\d{1,2})", word)
                    if match:
                        cleaned_word = match.group(1).replace(" ", "")
                        current_year = datetime.now().year
                        cleaned_word += f"/{current_year}"
                        date_obj = datetime.strptime(cleaned_word, "%d/%m/%Y")
                        date = date_obj.strftime("%d %B")
            except ValueError:
                date = word

    for result in results:
        if result.get("entity_group") == "TIM" and time is None:
            word = result.get("word", None)
            match = re.search(r"(\d{1,2})\.\s*(\d{2})\s*(wib|wita|wit)?", word, re.IGNORECASE)
            if match:
                hours = match.group(1)
                minutes = match.group(2)
                timezone = match.group(3).upper() if match.group(3) else ""
                time = f"{hours}.{minutes} {timezone}".strip()
            else:
                time = word

    if not location:
        location = "Tidak Ditemukan"
    if not date:
        date = "Tidak Ditemukan"
    if not time:
        time = "Tidak ada dalam artikel"

    return {"Location": location, "Date": date, "Time": time}

# Simulasi DataFrame hasil scraping (contoh)
df_scraped = pd.DataFrame({
    "Title": ["Banjir Jakarta Hari Ini", "Gempa Guncang Yogyakarta"],
    "Category": ["Banjir", "Gempa Bumi"],
    "Content": [
        "Banjir besar terjadi di Jakarta Timur akibat hujan deras yang turun sejak dini hari tadi.",
        "Gempa berkekuatan 5.6 SR mengguncang daerah Yogyakarta, membuat warga panik dan berhamburan keluar rumah."
    ],
    "URL": ["https://example1.com", "https://example2.com"]
})

# DataFrame untuk hasil NER
df_ner = pd.DataFrame(columns=["Title", "Category", "Location", "Date", "Time", "URL"])

# Proses NER untuk setiap baris dalam DataFrame hasil scraping
for index, row in df_scraped.iterrows():
    content = row["Content"]
    results = ner_with_chunking_and_cleaning(content)
    ner_results = extract_ner_results(results)

    df_ner.loc[len(df_ner)] = {
        "Title": row["Title"],
        "Category": row["Category"],
        "Location": ner_results["Location"],
        "Date": ner_results["Date"],
        "Time": ner_results["Time"],
        "URL": row["URL"]
    }

# Tampilkan hasil akhir
print("Hasil NER dari DataFrame Hasil Scraping:")
print(df_ner)
