# Pytorch Deep Learning
import torch
# Pandas+Numpy
import numpy as np
import pandas as pd
# Sklearn metrics
from sklearn.metrics import balanced_accuracy_score,accuracy_score

# Hugging Face Transformer Libraries
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline,Trainer, TrainingArguments
# Hugging Face Datasets
from datasets import Dataset

if torch.cuda.is_available():
    device=0

df = pd.read_csv("data.csv")
df.rename({"Sentence":"text"},axis=1,inplace=True)
df['Sentiment'] = df['Sentiment'].str.capitalize()
# Model name from Model Hub
model_name = 'yiyanghkust/finbert-tone'
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.config.label2id

df['label']=df['Sentiment'].apply(lambda l:model.config.label2id[l])
print(df['label'].value_counts())

val_end_point = int(df.shape[0]*0.8)
df_test = df.iloc[val_end_point:,:]
print(df_test.shape)

# Converting pandas DataFrames into Hugging Face Dataset objects:
dataset_test = Dataset.from_pandas(df_test)

# Tokenizing the datasets:
dataset_test = dataset_test.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length' , max_length=128), batched=True)
dataset_test.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])

model_path="FinBERT"

trained_pipeline = pipeline("text-classification", model=model_path, tokenizer=model_path,device=device)

preds=trained_pipeline(df_test['text'].tolist())
df_test['prediction']=[pred['label'] for pred in preds]

# Calculate the balanced accuracy score
score = balanced_accuracy_score(df_test['Sentiment'], df_test['prediction'])
print(f"Balanced Accuracy Score: {score}")

# Calculate the balanced accuracy score
score = accuracy_score(df_test['Sentiment'], df_test['prediction'])
print(f"Accuracy Score: {score}")
