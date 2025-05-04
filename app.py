import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Modeli ve tokenizer'ı yükle
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("./emotion_model")
    model = AutoModelForSequenceClassification.from_pretrained("./emotion_model")
    return tokenizer, model

tokenizer, model = load_model()

turkish_labels = [
    'hayranlık', 'eğlence', 'öfke', 'rahatsızlık', 'onay',
    'şefkat', 'kafa karışıklığı', 'merak', 'arzu', 'hayal kırıklığı',
    'onaylamama', 'tiksinme', 'utanç', 'heyecan', 'korku',
    'minnettarlık', 'keder', 'neşe', 'aşk', 'gerginlik',
    'iyimserlik', 'gurur', 'farkındalık', 'rahatlama',
    'pişmanlık', 'üzüntü', 'şaşkınlık', 'nötr'
]

st.title("Türkçe Duygu Analizi")
text_input = st.text_area("Lütfen bir metin girin:")

if st.button("Tahmin Et"):
    if text_input.strip() == "":
        st.warning("Lütfen bir metin girin.")
    else:
        inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).squeeze().tolist()

        results = {label: prob for label, prob in zip(turkish_labels, probs)}
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

        st.subheader("Tahmin Sonuçları")
        for label, score in sorted_results[:5]:  # En yüksek 5 tahmini göster
            st.write(f"{label}: {score:.2f}")
