# %%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score
import ast

# 1. Data Loading and Preprocessing
dataset = pd.read_csv("trans_final.csv")
dataset['label_ids'] = dataset['label_ids'].apply(ast.literal_eval)

# Turkish label mapping
turkish_labels = {
    0: 'hayranlık', 1: 'eğlence', 2: 'öfke', 3: 'rahatsızlık', 4: 'onay',
    5: 'şefkat', 6: 'kafa karışıklığı', 7: 'merak', 8: 'arzu', 9: 'hayal kırıklığı',
    10: 'onaylamama', 11: 'tiksinme', 12: 'utanç', 13: 'heyecan', 14: 'korku',
    15: 'minnettarlık', 16: 'keder', 17: 'neşe', 18: 'aşk', 19: 'gerginlik',
    20: 'iyimserlik', 21: 'gurur', 22: 'farkındalık', 23: 'rahatlama',
    24: 'pişmanlık', 25: 'üzüntü', 26: 'şaşkınlık', 27: 'nötr'
}

# 2. Data Splitting
train_df, temp_df = train_test_split(dataset, test_size=0.2, random_state=42)
validation_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# 3. Device Selection
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# 4. Class Weight Calculation
all_train_labels = [label for sublist in train_df['label_ids'] for label in sublist]
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(all_train_labels),
    y=all_train_labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# 5. Custom Dataset Class
class EmotionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['translated_text']
        labels = self.data.iloc[idx]['label_ids']

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        label_tensor = torch.zeros(len(turkish_labels), dtype=torch.float)
        label_tensor[labels] = 1

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_tensor
        }

# 6. Custom Loss Function
class WeightedBCELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, inputs, targets):
        loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        loss = (loss * self.weights).mean()
        return loss

# 7. Compute Metrics Function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()

    return {
        'f1': f1_score(labels, predictions, average='micro'),
        'accuracy': accuracy_score(labels, predictions)
    }

# 8. Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "dbmdz/bert-base-turkish-uncased",
    num_labels=len(turkish_labels),
    problem_type="multi_label_classification"
)
model.to(device)

# 9. Custom Trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").to(model.device)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        loss = WeightedBCELoss(class_weights.to(model.device))(logits, labels)
        return (loss, outputs) if return_outputs else loss

# 10. Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_dir='./logs',
    logging_steps=50,
    report_to="none",
    save_total_limit=2,
)

# 11. Create Datasets
train_dataset = EmotionDataset(train_df, tokenizer)
val_dataset = EmotionDataset(validation_df, tokenizer)
test_dataset = EmotionDataset(test_df, tokenizer)

# 12. Initialize Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# 13. Train the Model
trainer.train()

# 14. Save Model
model.save_pretrained("./emotion_model")
tokenizer.save_pretrained("./emotion_model")

# 15. Evaluate on Test Set
test_results = trainer.evaluate(test_dataset)
print("Test Results:", test_results)


"""
Test Results: {'eval_loss': 0.13599172234535217, 'eval_f1': 0.5178571428571429, 
'eval_accuracy': 0.3953006219765031, 'eval_runtime': 103.3961, 'eval_samples_per_second': 41.984, 
'eval_steps_per_second': 2.631, 'epoch': 5.0}
"""
# %%
