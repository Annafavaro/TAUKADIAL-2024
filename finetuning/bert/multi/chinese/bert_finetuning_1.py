cv_num = 1
out_scores = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/finetuning/results/multi/chinese/'
finetuning_data = f'/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/finetuning/data/multi/chinese/cv_{cv_num}/'

import os
from datasets import Dataset, DatasetDict
from datasets import Dataset
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from datasets import list_metrics
from transformers import AutoConfig, AutoModel
import numpy as np
from datasets import load_metric
import torch
torch.manual_seed(40)

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#checkpoint = 'distilbert-base-cased'
checkpoint='bert-base-chinese'
print(checkpoint)
#checkpoint = "bert-base-cased"

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

path_train = os.path.join(finetuning_data, 'train.csv')
path_dev = os.path.join(finetuning_data, 'dev.csv')
path_test = os.path.join(finetuning_data, 'test.csv')

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

tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
metrics_list = list_metrics()
metric = load_metric("accuracy")

task = f"taukadial_cv_{cv_num}"
task_to_keys = {task: ("sentences", None)}

sentence1_key, sentence2_key = task_to_keys[task]
if sentence2_key is None:
    print(f"Sentence: {dataset['train'][0][sentence1_key]}")
else:
    print(f"Sentence 1: {dataset['train'][0][sentence1_key]}")
    print(f"Sentence 2: {dataset['train'][0][sentence2_key]}")


encoded_dataset = dataset.map(preprocess_function, batched=True, load_from_cache_file=False)
num_labels = 2 # (cn or ad)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
model = model.to(device)
metric_name = "accuracy"
model_name = checkpoint.split("/")[-1]

args = TrainingArguments(
    f"{model_name}-finetuned-{task}",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-6,
    #learning_rate=2e-5,
    fp16=True,
    logging_steps=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=15,
    weight_decay=0.01,
    load_best_model_at_end=True,
    save_total_limit=1,
    metric_for_best_model=metric_name,
   # push_to_hub=True,
    logging_dir='./logs'#exp-dir
   # output_dir='./test_dir'
)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["dev"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics)

trainer.train()
model = AutoModelForSequenceClassification.from_pretrained(trainer.state.best_model_checkpoint, num_labels=2)
eval_trainer = Trainer(model, args, tokenizer=tokenizer, compute_metrics=compute_metrics)

print('best model loaded')
evaluation_results = eval_trainer.evaluate(eval_dataset=encoded_dataset["test"])
print('results test set')
print(evaluation_results)
acc = [evaluation_results['eval_accuracy']]
print('Accuracy-->')
print(acc)
predictions = eval_trainer.predict(encoded_dataset["test"])
print(predictions.predictions.shape, predictions.label_ids.shape)
preds = np.argmax(predictions.predictions, axis=-1)
print(f'predictions are {preds}')
sp_test = df_test['idx'].tolist()
y_test = df_test['label'].tolist()
dict = {'idx': sp_test, 'preds':preds, 'label': y_test, 'accuracy': acc*len(sp_test)}
df = pd.DataFrame(dict)
out_scores = os.path.join(out_scores, f'{cv_num}.csv')
df.to_csv(out_scores)
