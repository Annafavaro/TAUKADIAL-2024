import os
from datasets import load_dataset
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from datasets import Dataset
import pandas as pd
from datasets import ClassLabel
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
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

checkpoint = 'bert-base-cased'
#finetuning_data = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/finetuning/'
#path_train = os.path.join(finetuning_data, 'train_set.csv')
#path_dev = os.path.join(finetuning_data, 'dev_set.csv')
#path_test = os.path.join(finetuning_data, 'test_set.csv')
#
#print(path_train)
#
#df_train = pd.read_csv(path_train).drop(columns=['Unnamed: 0'])
#df_train = pd.DataFrame(df_train)
#df_dev = pd.read_csv(path_dev).drop(columns=['Unnamed: 0'])
#df_dev = pd.DataFrame(df_dev)
#df_test = pd.read_csv(path_test).drop(columns=['Unnamed: 0'])
#df_test = pd.DataFrame(df_test)
#
#train_ds = Dataset.from_pandas(df_train, split="train")
#dev_ds = Dataset.from_pandas(df_dev, split="train")
#test_ds = Dataset.from_pandas(df_test, split="test")
#
#dataset = DatasetDict()
#
#dataset['train'] = train_ds
#dataset['dev'] = dev_ds
#dataset['test'] = test_ds
#

data_dir_ = '/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_diagnosis_cv10_text_v5_Longformer_TrainDevTest/cv_1/test.tsv'
# data_dir_2 = '/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/data_ADReSSo_diagnosis_cv10_text_v5_Longformer_TrainDevTest/cv_2/test.tsv'

# %%

def path_to_csv_train(data_dir):
    cv_folder = os.path.basename(os.path.normpath(data_dir_))
    if os.path.exists(data_dir + '/train.tsv'):
        train_path = data_dir + '/train.tsv'
        train_csv = pd.read_csv(train_path, sep=',', header=None)
        speakers = train_csv[0].tolist()
        c2l = ClassLabel(num_classes=2, names=['cn', 'ad'])
        labels = train_csv[1].tolist()  # BERT only accepts the list of numerical values (float/int) as label.
        # numerical_labels = [c2l.str2int(label) for label in labels ]
        # ad == 1, cn ==0
        spk2lab = {sp: c2l.str2int(lab) for sp, lab in zip(speakers, labels)}
        # print(spk2lab)

    if os.path.exists(data_dir + '/utt2csvpath'):
        path_sentence_ = data_dir + '/utt2csvpath'
        paths_to_transcript = pd.read_csv(path_sentence_, sep=',', header=None)
        transcripts = paths_to_transcript[1].tolist()
        list_speaker = paths_to_transcript[0].tolist()

        sentences = []
        # for speaker in speakers:
        for spk, transcript in zip(list_speaker, transcripts):
            # if speaker in transcript:
            with open(transcript, 'r') as f:
                transcript_ = f.readlines()
                # print(transcript_)
                transcript_ = transcript_[0]
                sentences.append(transcript_)

        spk2sen = {sp: lab for sp, lab in zip(list_speaker, sentences)}

        data = np.array([[sp, spk2lab[sp], spk2sen[sp]] for sp in spk2lab]).T
        # print(data)
        # improved_list = [num for elem in sentences for num in elem]

        # dict = {'idx': speakers, 'label': numerical_labels, 'sentence': improved_list}
        dict = {'idx': data[0], 'label': data[1], 'sentence': data[2]}
        df = pd.DataFrame(dict)

        df.to_csv(f'/export/b14/afavaro/csv_addresso_2021/train/train_{cv_folder}.csv', index=False)  # Header?

        return f'/export/b14/afavaro/csv_addresso_2021/train/train_{cv_folder}.csv', df


# %%

# train_csv = path_to_csv_train(data_dir_)
# train_csv
# train_csv

# %%

def path_to_csv_dev(data_dir):
    cv_folder = os.path.basename(os.path.normpath(data_dir))
    if os.path.exists(data_dir + '/dev.tsv'):
        dev_path = data_dir + '/dev.tsv'
        dev_csv = pd.read_csv(dev_path, sep=',', header=None)
        speakers = dev_csv[0].tolist()
        c2l = ClassLabel(num_classes=2, names=['cn', 'ad'])
        labels = dev_csv[
            1].tolist()  # BERT only accepts the list of numerical values for labels (integer, float etc.) .
        # numerical_labels = [c2l.str2int(label) for label in labels ]
        # ad == 1, cn ==0
        spk2lab = {sp: c2l.str2int(lab) for sp, lab in zip(speakers, labels)}
        # print(spk2lab)

    if os.path.exists(data_dir + '/utt2csvpath'):
        path_sentence_ = data_dir + '/utt2csvpath'
        paths_to_transcript = pd.read_csv(path_sentence_, sep=',', header=None)
        transcripts = paths_to_transcript[1].tolist()
        list_speaker = paths_to_transcript[0].tolist()

        sentences = []
        # for speaker in speakers:
        for spk, transcript in zip(list_speaker, transcripts):
            # if speaker in transcript:
            with open(transcript, 'r') as f:
                transcript_ = f.readlines()
                # print(transcript_)
                transcript_ = transcript_[0]
                sentences.append(transcript_)

        spk2sen = {sp: lab for sp, lab in zip(list_speaker, sentences)}

        data = np.array([[sp, spk2lab[sp], spk2sen[sp]] for sp in spk2lab]).T
        # print(data)
        # improved_list = [num for elem in sentences for num in elem]

        # dict = {'idx': speakers, 'label': numerical_labels, 'sentence': improved_list}
        dict = {'idx': data[0], 'label': data[1], 'sentence': data[2]}
        df = pd.DataFrame(dict)

        df.to_csv(f'/export/b14/afavaro/csv_addresso_2021/dev/dev_{cv_folder}.csv', index=False)  # Header?

        return f'/export/b14/afavaro/csv_addresso_2021/dev/dev_{cv_folder}.csv', df


# %%

dev_csv = path_to_csv_dev(data_dir_)


def path_to_csv_test(data_dir):
    cv_folder = os.path.basename(os.path.normpath(data_dir))
    if os.path.exists(data_dir + '/test.tsv'):
        test_path = data_dir + '/test.tsv'
        test_csv = pd.read_csv(test_path, sep=',', header=None)
        speakers = test_csv[0].tolist()
        c2l = ClassLabel(num_classes=2, names=['cn', 'ad'])
        labels = test_csv[
            1].tolist()  # BERT only accepts the list of numerical values for labels (integer, float etc.) .
        # numerical_labels = [c2l.str2int(label) for label in labels ]
        # ad == 1, cn ==0
        spk2lab = {sp: c2l.str2int(lab) for sp, lab in zip(speakers, labels)}
        # print(spk2lab)

    if os.path.exists(data_dir + '/utt2csvpath'):
        path_sentence_ = data_dir + '/utt2csvpath'
        paths_to_transcript = pd.read_csv(path_sentence_, sep=',', header=None)
        transcripts = paths_to_transcript[1].tolist()
        list_speaker = paths_to_transcript[0].tolist()

        sentences = []
        # for speaker in speakers:
        for spk, transcript in zip(list_speaker, transcripts):
            # if speaker in transcript:
            with open(transcript, 'r') as f:
                transcript_ = f.readlines()
                # print(transcript_)
                transcript_ = transcript_[0]
                sentences.append(transcript_)

        spk2sen = {sp: lab for sp, lab in zip(list_speaker, sentences)}

        data = np.array([[sp, spk2lab[sp], spk2sen[sp]] for sp in spk2lab]).T
        # print(data)
        # improved_list = [num for elem in sentences for num in elem]

        # dict = {'idx': speakers, 'label': numerical_labels, 'sentence': improved_list}
        dict = {'idx': data[0], 'label': data[1], 'sentence': data[2]}
        df = pd.DataFrame(dict)

        df.to_csv(f'/export/b14/afavaro/csv_addresso_2021/test/test_{cv_folder}.csv', index=False)

        return f'/export/b14/afavaro/csv_addresso_2021/test/test_{cv_folder}.csv', df


# %%
train_csv = path_to_csv_train(data_dir_)
test_csv = path_to_csv_test(data_dir_)


dataset = load_dataset('csv', data_files={'train': train_csv, 'dev': dev_csv, 'test': test_csv})

#dataset = load_dataset('csv', data_files={"train": path_train, 'dev': path_dev, "test": path_test})
tokernizer = AutoTokenizer.from_pretrained(checkpoint)
#def tokenize_fn(batch):
  #return tokernizer(batch['sentence'], truncation = True)

def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True, padding=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, padding=True, max_length=512,
    return_tensors="pt")


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


metric_name = "accuracy"
model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    f"{model_name}-finetuned-{task}",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    fp16=True,
    logging_steps=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
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