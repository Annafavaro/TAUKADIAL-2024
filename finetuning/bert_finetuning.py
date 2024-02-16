import os
from datasets import load_dataset
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from datasets import Dataset
import pandas as pd
from datasets import list_metrics
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from datasets import load_metric
from transformers import pipeline
from transformers import AutoTokenizer
from datasets import list_metrics
import numpy as np
from datasets import load_metric

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
def tokenize_fn(batch):
  return tokernizer(batch['sentence'], truncation = True)

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
    print(f"Sentence: {train_ds[0][sentence1_key]}")
else:
    print(f"Sentence 1: {train_ds[0][sentence1_key]}")
    print(f"Sentence 2: {train_ds[sentence2_key]}")

#%%

#def preprocess_function(examples):
#    if sentence2_key is None:
#        return tokenizer(examples[sentence1_key], truncation=True, padding=True)
#    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, padding=True,
#    return_tensors="pt")

def tokenize_fn(batch):
  return tokernizer(batch['sentences'], truncation = True)

encoded_dataset = dataset.map(tokenize_fn, batched=True, load_from_cache_file=False)

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
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=9,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    #push_to_hub=True,
    logging_dir='./logs'#exp-dir
   # output_dir='./test_dir'
)



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["dev"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics

)

trainer.train()