import os
from datasets import load_dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd
from datasets import list_metrics
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from datasets import load_metric
from transformers import AutoTokenizer
from datasets import list_metrics
import numpy as np
from datasets import load_metric

checkpoint = 'bert-base-cased'
finetuning_data = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/finetuning/'
path_train = os.path.join(finetuning_data, 'train_set.csv')
path_dev = os.path.join(finetuning_data, 'dev_set.csv')
path_test = os.path.join(finetuning_data, 'test_set.csv')

dataset = load_dataset('csv', data_files={"train": path_train, 'dev': path_dev, "test": path_test})
tokernizer = AutoTokenizer.from_pretrained(checkpoint)
print('done')

metrics_list = list_metrics()
#metric
print(metrics_list)
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


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



model_checkpoint = "bert-base-uncased"
batch_size = 8
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

task = "addresso"
task_to_keys = { "addresso": ("sentences", None)}

#%%

sentence1_key, sentence2_key = task_to_keys[task]
if sentence2_key is None:
    print(f"Sentence: {dataset['train'][0][sentence1_key]}")
else:
    print(f"Sentence 1: {dataset['train'][0][sentence1_key]}")
    print(f"Sentence 2: {dataset['train'][0][sentence2_key]}")

#%%

def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True, padding=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, padding=True, max_length=512,
    return_tensors="pt")

encoded_dataset = dataset.map(preprocess_function, batched=True,load_from_cache_file=False)

num_labels = 2 # (cn or ad)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)


metric_name = "accuracy"
model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    f"{model_name}-finetuned-{task}",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    fp16=True,
    logging_steps=1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=9,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    #push_to_hub=True,
    logging_dir='./logs'#exp-dir
   # output_dir='./test_dir'
)

