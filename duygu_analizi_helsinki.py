
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import torch


# Model ve tokenizer'ı yükle
model_name = "Helsinki-NLP/opus-mt-tc-big-en-tr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# GPU varsa kullan
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Model {device} üzerinde yüklendi")

# %%
def translate_with_model(text, model, tokenizer, device="cpu"):
    try:
        # Metni tokenize et ve modele uygun hale getir
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        
        # Çeviri yap
        translated = model.generate(**inputs)
        
        # Token'ları metne çevir
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    
    except Exception as e:
        print(f"Hata: {str(e)}")
        return f"[ÇEVİRİ HATASI] {text}"
    
# %%
# Veri setini yükle
from datasets import load_dataset
dataset = load_dataset("go_emotions")
df_train = pd.DataFrame(dataset['train'])

# Duygu eşleme sözlüğü
emotion_map_tr = {
    0: 'hayranlık', 1: 'eğlence', 2: 'öfke', 3: 'rahatsızlık', 4: 'onay',
    5: 'şefkat', 6: 'kafa karışıklığı', 7: 'merak', 8: 'arzu', 9: 'hayal kırıklığı',
    10: 'onaylamama', 11: 'tiksinme', 12: 'utanç', 13: 'heyecan', 14: 'korku',
    15: 'minnettarlık', 16: 'keder', 17: 'neşe', 18: 'aşk', 19: 'gerginlik',
    20: 'iyimserlik', 21: 'gurur', 22: 'farkındalık', 23: 'rahatlama',
    24: 'pişmanlık', 25: 'üzüntü', 26: 'şaşkınlık', 27: 'nötr'
}

# Batch çeviri fonksiyonu (daha hızlı)
def batch_translate(texts, model, tokenizer, device="cpu", batch_size=8):
    translations = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        translated = model.generate(**inputs)
        translations.extend([tokenizer.decode(t, skip_special_tokens=True) for t in translated])
    return translations

sample_data = df_train.copy()
translated_texts = batch_translate(sample_data['text'].tolist(), model, tokenizer, device)

# DataFrame oluştur
df_tr = pd.DataFrame({
    'original_text': sample_data['text'],
    'translated_text': translated_texts,
    'label_ids': sample_data['labels'].apply(lambda x: str(x)),
    'emotions': sample_data['labels'].apply(
        lambda x: str([emotion_map_tr[i] for i in (x if isinstance(x, list) else [x])]))
})

# Kaydet
df_tr.to_csv('goemotions_helsinki_tr.csv', index=False, encoding='utf-8-sig')
print("Çeviri tamamlandı ve kaydedildi!")

# %%
