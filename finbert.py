# Pytorch Deep Learning
import torch
# Pandas+Numpy
import numpy as np
import pandas as pd
# Sklearn metrics
from sklearn.metrics import balanced_accuracy_score,accuracy_score

# Hugging Face Transformer Libraries
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline,Trainer, TrainingArguments
# Huggi§ng Face Datasets
from datasets import Dataset

if torch.cuda.is_available():
    device=0

df = pd.read_csv("data.csv")
df.rename({"Sentence":"text"},axis=1,inplace=True)
df['Sentiment'] = df['Sentiment'].str.capitalize()
# Model name from Model Hub
model_name = 'yiyanghkust/finbert-tone'

sentiment_pipeline = pipeline(task="sentiment-analysis", model=model_name,batch_size=128,device=device)
preds = sentiment_pipeline(df['text'].tolist())
df['prediction']=[pred['label'] for pred in preds]
print(df.groupby(['Sentiment','prediction']).size())
print(balanced_accuracy_score(df['Sentiment'],df['prediction']))
print(accuracy_score(df['Sentiment'],df['prediction']))