import pandas as pd
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2Model
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import math

# Define the range of cross-validation splits
cv_range = range(1, 11)

# Loop over cross-validation splits
for cv_num in cv_range:
    print(f'fold number {cv_num}')
    out_path = f'/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/finetuning/results/chatgpt/mono/chinese/cv_{cv_num}.csv'

    # Load data for the current cross-validation split
    cv_train1 = pd.read_csv(f'/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/finetuning/data/mono/chinese/cv_{cv_num}/train.csv')
    cv_train1 = cv_train1.drop(columns=['Unnamed: 0'])
    cv_train2 = pd.read_csv(f'/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/finetuning/data/mono/chinese/cv_{cv_num}/dev.csv')
    cv_train2 = cv_train2.drop(columns=['Unnamed: 0'])
    cv_train = pd.concat([cv_train1, cv_train2])
    cv_test = pd.read_csv(f'/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/finetuning/data/mono/chinese/cv_{cv_num}/test.csv')
    cv_test = cv_test.drop(columns=['Unnamed: 0'])
    X_train = cv_train['sentences']
    y_train = cv_train['label']
    X_test =  cv_test['sentences']
    y_test = cv_test['label']

    # Tokenization and preprocessing
    MAX_LENGTH = math.ceil((X_train.apply(lambda x: len(str(x).split())).mean())) + 2
    PAD_TOKEN = "<|pad|>"
    EOS_TOKEN = "<|endoftext|>"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large", pad_token=PAD_TOKEN, eos_token=EOS_TOKEN, max_length=MAX_LENGTH, is_split_into_words=True)
    X_train = [str(ex) + EOS_TOKEN for ex in X_train]
    X_test = [str(ex) + EOS_TOKEN for ex in X_test]
    X_train_ = [tokenizer(str(x), return_tensors='tf', max_length=MAX_LENGTH, truncation=True, pad_to_max_length=True, add_special_tokens=True)['input_ids'] for x in X_train]
    X_test_ = [tokenizer(str(x), return_tensors='tf', max_length=MAX_LENGTH, truncation=True, pad_to_max_length=True, add_special_tokens=True)['input_ids'] for x in X_test]
    X_train_in = tf.squeeze(tf.convert_to_tensor(X_train_), axis=1)
    X_test_in = tf.squeeze(tf.convert_to_tensor(X_test_), axis=1)
    X_train_mask_ = [tokenizer(str(x), return_tensors='tf', max_length=MAX_LENGTH, truncation=True, pad_to_max_length=True, add_special_tokens=True)["attention_mask"] for x in X_train]
    X_test_mask_ = [tokenizer(str(x), return_tensors='tf', max_length=MAX_LENGTH, truncation=True, pad_to_max_length=True, add_special_tokens=True)["attention_mask"] for x in X_test]
    X_train_mask = tf.squeeze(tf.convert_to_tensor(X_train_mask_), axis=1)
    X_test_mask = tf.squeeze(tf.convert_to_tensor(X_test_mask_), axis=1)

    # Model setup
    model = TFGPT2Model.from_pretrained("gpt2-large", use_cache=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    model.training = True
    model.resize_token_embeddings(len(tokenizer))

    for layer in model.layers:
        layer.trainable = False

    input_layer = tf.keras.layers.Input(shape=(None,), dtype='int32')
    mask_layer = tf.keras.layers.Input(shape=(None,), dtype='int32')
    x = model(input_layer, attention_mask=mask_layer)
    x_pool = tf.reduce_mean(x.last_hidden_state, axis=1)
    x = tf.keras.layers.Dense(256, activation='relu')(x_pool)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    clf = tf.keras.Model([input_layer, mask_layer], output_layer)

    # Model compilation and training
    base_learning_rate = 0.0001
    optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy()
    clf.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callbacks = tf.keras.callbacks.EarlyStopping(monitor="accuracy", verbose=1, patience=3, restore_best_weights=True)

    y_train_in = tf.constant(y_train, dtype=tf.int32)
    y_test_in = tf.constant(y_test, dtype=tf.int32)
    tf.config.experimental_run_functions_eagerly(True)
    history = clf.fit([X_train_in, X_train_mask], y_train_in, epochs=1, batch_size=32, validation_split=0.2, callbacks=callbacks)

    # Model evaluation
    clf.evaluate([X_test_in, X_test_mask], y_test_in)
    clf.training = False
    y_pred = clf.predict([X_test_in, X_test_mask])
    print('score',y_pred )
    y_pred_out = (y_pred[:, 0] >= 0.5).astype(int)
    # Print and save results
    print("Predictions:", y_pred_out)
    print(classification_report(y_test_in, y_pred_out))

    # Save predictions to CSV
    accuracy = accuracy_score(y_test_in, y_pred_out)
    score_list = [item[0] for item in y_pred.tolist()]
    data = {'idx': cv_test['idx'].tolist(), 'predictions': y_pred_out, 'score': score_list,
            'label': y_test_in.tolist(), 'accuracy': [accuracy] * len(y_pred_out)}
    df = pd.DataFrame(data)
    df.to_csv(out_path, index=False)
