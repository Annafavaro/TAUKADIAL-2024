import io
import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW,
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)


class MovieReviewsDataset2(Dataset):
    """
 a   PyTorch Dataset class for loading data from Pandas DataFrames.

    This class is designed to be used with dataframes where the text data is
    stored under the column 'sentences', and labels are binary (1s and 0s) stored
    under a 'label' column.

    Arguments:
        dataframe (:obj:`DataFrame`): Pandas DataFrame containing the reviews and labels.
        use_tokenizer (:obj:`Any`): Tokenizer instance to preprocess the text data.

    """

    def __init__(self, dataframe, use_tokenizer):
        # Assuming 'sentences' column contains the text and there's a 'label' column for labels
        self.texts = dataframe['sentences'].tolist()
        self.labels = dataframe['label'].tolist()  # Assuming binary labels are already 0s and 1s

        # Tokenization can be performed here if needed, for example:
        # self.tokenized_texts = [use_tokenizer(text) for text in self.texts]

        # Number of examples
        self.n_examples = len(self.labels)

    def __len__(self):
        """When used `len` return the number of examples."""
        return self.n_examples

    def __getitem__(self, item):
        """
        Given an index return an example from that position.

        Arguments:
            item (:obj:`int`): Index position to pick an example to return.

        Returns:
            A tuple containing the tokenized text and its associated label.
        """
        return {'text': self.texts[item],
                'label': self.labels[item]}


class Gpt2ClassificationCollator(object):
    r"""
    Data Collator used for GPT2 in a classificaiton rask.

    It uses a given tokenizer and label encoder to convert any text and labels to numbers that
    can go straight into a GPT2 model.

    This class is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed
    straight into the model - `model(**batch)`.

    Arguments:

      use_tokenizer (:obj:`transformers.tokenization_?`):
          Transformer type tokenizer used to process raw text into numbers.

      labels_ids (:obj:`dict`):
          Dictionary to encode any labels names into numbers. Keys map to
          labels names and Values map to number associated to those labels.

      max_sequence_len (:obj:`int`, `optional`)
          Value to indicate the maximum desired sequence to truncate or pad text
          sequences. If no value is passed it will used maximum sequence size
          supported by the tokenizer and model.

    """

    def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):
        # Tokenizer to be used inside the class.
        self.use_tokenizer = use_tokenizer
        # Check max sequence length.
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        # Label encoder used inside the class.
        self.labels_encoder = labels_encoder

        return

    def __call__(self, sequences):
        r"""
        This function allowes the class objesct to be used as a function call.
        Sine the PyTorch DataLoader needs a collator function, I can use this
        class as a function.

        Arguments:

          item (:obj:`list`):
              List of texts and labels.

        Returns:
          :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
          It holddes the statement `model(**Returned Dictionary)`.
        """

        # Get all texts from sequences list.
        texts = [sequence['text'] for sequence in sequences]
        # Get all labels from sequences list.
        labels = [sequence['label'] for sequence in sequences]
        # Encode all labels using label encoder.
        labels = [self.labels_encoder[label] for label in labels]
        # Call tokenizer on all texts to convert into tensors of numbers with
        # appropriate padding.
        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,
                                    max_length=self.max_sequence_len)
        # Update the inputs with the associated encoded labels as tensor.
        inputs.update({'labels': torch.tensor(labels)})

        return inputs


def train(dataloader, optimizer_, scheduler_, device_):
    r"""
    Train pytorch model on a single pass through the data loader.

    It will use the global variable `model` which is the transformer model
    loaded on `_device` that we want to train on.

    This function is built with reusability in mind: it can be used as is as long
      as the `dataloader` outputs a batch in dictionary format that can be passed
      straight into the model - `model(**batch)`.

    Arguments:

        dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
            Parsed data into batches of tensors.

        optimizer_ (:obj:`transformers.optimization.AdamW`):
            Optimizer used for training.

        scheduler_ (:obj:`torch.optim.lr_scheduler.LambdaLR`):
            PyTorch scheduler.

        device_ (:obj:`torch.device`):
            Device used to load tensors before feeding to model.

    Returns:

        :obj:`List[List[int], List[int], float]`: List of [True Labels, Predicted
          Labels, Train Average Loss].
    """

    # Use global variable for model.
    global model

    # Tracking variables.
    predictions_labels = []
    true_labels = []
    # Total loss for this epoch.
    total_loss = 0

    # Put the model into training mode.
    model.train()

    # For each batch of training data...
    for batch in tqdm(dataloader, total=len(dataloader)):
        # Add original labels - use later for evaluation.
        true_labels += batch['labels'].numpy().flatten().tolist()
        # print('true labels are', true_labels)
        # move batch to device
        batch = {k: v.type(torch.long).to(device_) for k, v in batch.items()}

        # Always clear any previously calculated gradients before performing a
        # backward pass.
        model.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        # The documentation for this a bert model function is here:
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(**batch)
        # The call to `model` always returns a tuple, so we need to pull the
        # loss value out of the tuple along with the logits. We will use logits
        # later to calculate training accuracy.
        loss, logits = outputs[:2]
        # print('logits are', logits)
        score_mci = torch.softmax(logits, dim=1)
        score_mci_sc = score_mci[:, 1]
        # print('scores', score_mci_sc)
        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value
        # from the tensor.
        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer_.step()

        # Update the learning rate.
        scheduler_.step()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()

        # Convert these logits to list of predicted labels values.
        predictions_labels += logits.argmax(axis=-1).flatten().tolist()
        # print('preds', predictions_labels)
    # Calculate the average loss over the training data.
    avg_epoch_loss = total_loss / len(dataloader)

    # Return all true labels and prediction for future evaluations.
    return true_labels, predictions_labels, avg_epoch_loss


def validation(dataloader, device_):
    r"""Validation function to evaluate model performance on a
    separate set of data.

    This function will return the true and predicted labels so we can use later
    to evaluate the model's performance.

    This function is built with reusability in mind: it can be used as is as long
      as the `dataloader` outputs a batch in dictionary format that can be passed
      straight into the model - `model(**batch)`.

    Arguments:

      dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
            Parsed data into batches of tensors.

      device_ (:obj:`torch.device`):
            Device used to load tensors before feeding to model.

    Returns:

      :obj:`List[List[int], List[int], float]`: List of [True Labels, Predicted
          Labels, Train Average Loss]
    """

    # Use global variable for model.
    global model

    # Tracking variables

    predictions_labels = []
    predictions_scores = []
    true_labels = []
    # total loss for this epoch.
    total_loss = 0

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Evaluate data for one epoch
    for batch in tqdm(dataloader, total=len(dataloader)):
        # add original labels
        true_labels += batch['labels'].numpy().flatten().tolist()

        # move batch to device
        batch = {k: v.type(torch.long).to(device_) for k, v in batch.items()}

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(**batch)

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple along with the logits. We will use logits
            # later to to calculate training accuracy.
            loss, logits = outputs[:2]
            scores = torch.softmax(logits, dim=1)
            scores_out = scores[:, 1].flatten().tolist()
            # print('scores validationnnnn', scores_out)
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_loss += loss.item()

            # get predicitons to list
            predict_content = logits.argmax(axis=-1).flatten().tolist()

            # update list
            predictions_labels += predict_content
            predictions_scores += scores_out

    # Calculate the average loss over the training data.
    avg_epoch_loss = total_loss / len(dataloader)
    #  print(predictions_scores)
    #  print(predictions_labels)
    # Return all true labels and prediciton for future evaluations.
    return true_labels, predictions_labels, predictions_scores, avg_epoch_loss


cv_range = range(1, 11)

for cv_num in cv_range:
    print(f'fold number {cv_num}')
    out_path = f'/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/finetuning/results/chatgpt_pytorch/mono/english/cv_{cv_num}.csv'

    cv_train1 = pd.read_csv(f'/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/finetuning/data/multi/english/cv_{cv_num}/train.csv')
    cv_train1 = cv_train1.drop(columns=['Unnamed: 0'])
    cv_train1['label'] = ['MCI' if elem == 0 else 'CN' for elem in list(cv_train1['label'])]

    cv_train2 = pd.read_csv(f'/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/finetuning/data/multi/english/cv_{cv_num}/dev.csv')
    cv_train2 = cv_train2.drop(columns=['Unnamed: 0'])
    cv_train2['label'] = ['MCI' if elem == 0 else 'CN' for elem in list(cv_train2['label'])]
    cv_train = pd.concat([cv_train1, cv_train2]).reset_index(drop=True)

    cv_test = pd.read_csv(f'/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/finetuning/data/multi/english/cv_{cv_num}/test.csv')
    cv_test = cv_test.drop(columns=['Unnamed: 0'])
    cv_test['label'] = ['MCI' if elem == 0 else 'CN' for elem in list(cv_test['label'])]


    set_seed(123)
    epochs = 1
    batch_size = 6
    max_length = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name_or_path = 'gpt2'
    labels_ids = {'MCI': 0, 'CN': 1}
    n_labels = len(labels_ids)
    print(n_labels)

    # Get model configuration.
    print('Loading configuration...')
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name_or_path, num_labels=n_labels)

    # Get model's tokenizer.
    print('Loading tokenizer...')
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
    # default to left padding
    tokenizer.padding_side = "left"
    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token

    # Get the actual model.
    print('Loading model...')
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=model_config)
    # resize model embedding to match new tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # fix model padding token id
    model.config.pad_token_id = model.config.eos_token_id

    # Load model to defined device.
    model.to(device)
    print('Model loaded to `%s`'%device)

    # Create data collator to encode text and labels into numbers.
    gpt2_classificaiton_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer,
                                                              labels_encoder=labels_ids,
                                                              max_sequence_len=max_length)

    print('Dealing with Train...')
    # Create pytorch dataset.
    train_dataset = MovieReviewsDataset2(cv_train, use_tokenizer=tokenizer)
    print('Created `train_dataset` with %d examples!'%len(train_dataset))

    # Move pytorch dataset into dataloader.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_classificaiton_collator)
    print('Created `train_dataloader` with %d batches!'%len(train_dataloader))

    print()

    print('Dealing with Validation...')
    # Create pytorch dataset. # test set
    valid_dataset =  MovieReviewsDataset2(cv_test,
                                   use_tokenizer=tokenizer)
    print('Created `valid_dataset` with %d examples!'%len(valid_dataset))
    # Move pytorch dataset into dataloader.
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=gpt2_classificaiton_collator)
    print('Created `eval_dataloader` with %d batches!'%len(valid_dataloader))

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr = 2e-5, # default is 5e-5, our notebook had 2e-5
                      eps = 1e-8 # default is 1e-8.
                      )

    total_steps = len(train_dataloader) * epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    # Store the average loss after each epoch so we can plot them.
    all_loss = {'train_loss':[], 'val_loss':[]}
    all_acc = {'train_acc':[], 'val_acc':[]}

    # Loop through each epoch.
    best_val_acc = 0.0  # Best validation accuracy seen so far
    patience = 1  # Number of epochs to wait after last time validation accuracy improved
    epochs_since_improvement = 0  # Counter for epochs since last improvement

    print('Epoch')

    for epoch in tqdm(range(epochs)):
      print()
      print('Training on batches...')
      # Perform one full pass over the training set.
      train_labels, train_predict, train_loss = train(train_dataloader, optimizer, scheduler, device)
      train_acc = accuracy_score(train_labels, train_predict)

      # Get prediction form model on validation data.
      print('Validation on batches...')
      valid_labels, valid_predict, val_scores, val_loss = validation(valid_dataloader, device)
      val_acc = accuracy_score(valid_labels, valid_predict)


      # Print loss and accuracy values to see how training evolves.
      print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f"%(train_loss, val_loss, train_acc, val_acc))
      print()

      # Store the loss value for plotting the learning curve.
      all_loss['train_loss'].append(train_loss)
      all_loss['val_loss'].append(val_loss)
      all_acc['train_acc'].append(train_acc)
      all_acc['val_acc'].append(val_acc)

      if val_acc > best_val_acc:
          best_val_acc = val_acc
          epochs_since_improvement = 0
          print(f"Validation accuracy improved to {val_acc:.5f}, saving model...")
          # Optionally, save the model here
      else:
          epochs_since_improvement += 1
          print(f"No improvement in validation accuracy for {epochs_since_improvement} epochs...")
          if epochs_since_improvement >= patience:
              print("Early stopping triggered.")
              break


    true_labels, predictions_labels, pred_scores, avg_epoch_loss = validation(valid_dataloader, device)

    # Create the evaluation report.
    evaluation_report = classification_report(true_labels, predictions_labels, labels=list(labels_ids.values()), target_names=list(labels_ids.keys()))
    # Show the evaluation report.
    print(evaluation_report)

    # Calculate accuacy
    accuracy = accuracy_score(true_labels, predictions_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    data = {
            'idx': cv_test['idx'].tolist(),
            'preds': predictions_labels,
            'score': pred_scores,
            'label':  true_labels,
            'accuracy': [accuracy] * len(cv_test['label'].tolist())
        }
    df = pd.DataFrame(data)
    df.to_csv(out_path, index=False)


