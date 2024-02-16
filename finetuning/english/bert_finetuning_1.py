out_scores = '/export/b16/afavaro/TAUKADIAL-2024/finetuning/scores/english/'
import os
from datasets import Dataset, DatasetDict
from datasets import Dataset
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from datasets import list_metrics
import numpy as np
import torch
from datasets import load_metric
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

checkpoint = 'bert-base-cased'
model_checkpoint = "bert-base-cased"

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

finetuning_data = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/finetuning/english/cv_1/'
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

task = "taukadial_cv1"
task_to_keys = { "taukadial_cv1": ("sentences", None)}

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

args = TrainingArguments(
    f"{model_name}-finetuned-{task}",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate = 0.0001,
    #learning_rate=2e-5,
    fp16=True,
    logging_steps=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
   # push_to_hub=True,
    logging_dir='./logs'#exp-dir
   # output_dir='./test_dir'
)

#def compute_metrics(eval_pred):
#    logits, labels = eval_pred
#    predictions = np.argmax(logits, axis=-1)
#    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["dev"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics

)

trainer.train()

evaluation_results = trainer.evaluate(eval_dataset=encoded_dataset["test"])
print('RESULTS on the test set')
print(evaluation_results)
acc = evaluation_results['eval_accuracy']
print('ACCURACYYYYYYY')
print(acc)
predictions = trainer.predict(encoded_dataset["test"])
print(predictions.predictions.shape, predictions.label_ids.shape)
preds = np.argmax(predictions.predictions, axis=-1)
print(f'predictions are {preds}')
sp_test = df_test['idx'].tolist()
dict = {'idx': sp_test, 'preds':preds}
df = pd.DataFrame(dict)
out_scores = os.path.join(out_scores, 'cv1.csv')
df.to_csv(out_scores)
