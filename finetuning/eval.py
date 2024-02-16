import os
from datasets import load_dataset
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from datasets import Dataset
import pandas as pd
from tqdm.auto import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from datasets import load_metric
from transformers import pipeline
from transformers import AutoTokenizer
from datasets import list_metrics
import numpy as np
import torch
from datasets import load_metric
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

#device = 'cpu' #torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

checkpoint = 'bert-base-cased'
finetuning_data = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/finetuning/'
path_train = os.path.join(finetuning_data, 'train_set.csv')
path_dev = os.path.join(finetuning_data, 'dev_set.csv')
path_test = os.path.join(finetuning_data, 'test_set.csv')

print(path_train)

df_train = pd.read_csv(path_train).drop(columns=['Unnamed: 0'])
df_train = pd.DataFrame(df_train)
df_dev = pd.read_csv(path_dev).drop(columns=['Unnamed: 0'])
df_dev = pd.DataFrame(df_dev)
df_test = pd.read_csv(path_test).drop(columns=['Unnamed: 0'])
df_test = pd.DataFrame(df_test)

train_ds = Dataset.from_pandas(df_train, split="train")
dev_ds = Dataset.from_pandas(df_dev, split="train")
test_ds = Dataset.from_pandas(df_test, split="test")

dataset = DatasetDict()

dataset['train'] = train_ds
dataset['dev'] = dev_ds
dataset['test'] = test_ds


#dataset = load_dataset('csv', data_files={"train": path_train, 'dev': path_dev, "test": path_test})
tokernizer = AutoTokenizer.from_pretrained(checkpoint)
#def tokenize_fn(batch):
  #return tokernizer(batch['sentence'], truncation = True)

def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True, padding=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, padding=True, max_length=512,
    return_tensors="pt")



def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


metrics_list = list_metrics()
#metric
print(metrics_list)
metric = load_metric("accuracy")

model_checkpoint = "bert-base-cased"
batch_size = 8
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

task = "addresso"
task_to_keys = { "addresso": ("sentences", None)}

sentence1_key, sentence2_key = task_to_keys[task]
if sentence2_key is None:
    print(f"Sentence: {dataset['train'][0][sentence1_key]}")
else:
    print(f"Sentence 1: {dataset['train'][0][sentence1_key]}")
    print(f"Sentence 2: {dataset['train'][0][sentence2_key]}")


encoded_dataset = dataset.map(preprocess_function, batched=True, load_from_cache_file=False)

num_labels = 2 # (cn or ad)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
model = model.to(device)

metric_name = "accuracy"
model_name = model_checkpoint.split("/")[-1]

model_path = "/export/b16/afavaro/TAUKADIAL-2024/finetuning/bert-base-cased-finetuned-addresso/checkpoint-100/"
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)

# Define test trainer
test_trainer = Trainer(model)
# ----- 3. Predict -----#
# Load test data
#test_data = pd.read_csv("test.csv")
X_test = list(df_test['sentences'])
y_test_true =  list(df_test['label'])
#X_test = list(test_data["review"])
X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)

# Create torch dataset
test_dataset = Dataset(X_test_tokenized)

# Make prediction
raw_pred, _, _ = test_trainer.predict(test_dataset)
# Preprocess raw predictions
y_pred = np.argmax(raw_pred, axis=1)
print(y_pred)