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

train_end_point = int(df.shape[0]*0.6)
val_end_point = int(df.shape[0]*0.8)
df_train = df.iloc[:train_end_point,:]
df_val = df.iloc[train_end_point:val_end_point,:]
df_test = df.iloc[val_end_point:,:]
print(df_train.shape, df_test.shape, df_val.shape)

# Converting pandas DataFrames into Hugging Face Dataset objects:
dataset_train = Dataset.from_pandas(df_train)
dataset_val = Dataset.from_pandas(df_val)
dataset_test = Dataset.from_pandas(df_test)

# Tokenizing the datasets:
dataset_train = dataset_train.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=128), batched=True)
dataset_val = dataset_val.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=128), batched=True)
dataset_test = dataset_test.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length' , max_length=128), batched=True)

# Setting the dataset format: (needed for Pytorch?)
dataset_train.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
dataset_val.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
dataset_test.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])


# Shuffle the training dataset
dataset_train_shuffled = dataset_train.shuffle(seed=42)  # Using a seed for reproducibility

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'balanced_accuracy' : balanced_accuracy_score(predictions, labels),'accuracy':accuracy_score(predictions,labels)}

args = TrainingArguments(
    output_dir='temp/',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_strategy="steps",  # Log every X steps
    logging_steps=50,  # Log every 50 steps
    learning_rate=2e-6,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.1,
    load_best_model_at_end=True,
    metric_for_best_model='balanced_accuracy',
)

trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=args,                  # training arguments, defined above
        train_dataset=dataset_train_shuffled,         # training dataset
        eval_dataset=dataset_val,            # evaluation dataset
        compute_metrics=compute_metrics
)

trainer.train()

predictions = trainer.predict(dataset_test)
print(predictions)

model_path = "FinBERT"


# Save the model
trainer.model.save_pretrained(model_path)

# Save the tokenizer associated with the model
# Save the tokenizer
tokenizer.save_pretrained(model_path)

