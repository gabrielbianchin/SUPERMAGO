import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm
import click
import math

gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

@click.command()
@click.option('--ont', '-ont', help="Ontology (bp, cc or mf)")
@click.option('--layer', '-l', help="Layer (36, 35, 34, 33 or 32 for ESM2 T36) (24, 23, 22, 21, 20 for ProtT5)")

def main(ont, layer):
  y_train, pos_train = preprocess(df=pd.read_csv('../../base/{}_train.csv'.format(ont)))
  y_val, pos_val = preprocess(df=pd.read_csv('../../base/{}_val.csv'.format(ont)))
  y_test, pos_test = preprocess(df=pd.read_csv('../../base/{}_test.csv'.format(ont)))

  X_train = np.load('../../embs/{}-{}-train.npy'.format(layer, ont))
  X_val = np.load('../../embs/{}-{}-val.npy'.format(layer, ont))
  X_test = np.load('../../embs/{}-{}-test.npy'.format(layer, ont))

  X_train, y_train = protein_embedding(X_train, y_train, pos_train)
  X_val, y_val = protein_embedding(X_val, y_val, pos_val)
  X_test, y_test = protein_embedding(X_test, y_test, pos_test)

  model = tf.keras.models.Sequential([
    tf.keras.layers.Normalization(),
    tf.keras.layers.Dense(4096, activation='relu', input_shape=(X_train.shape[1], )),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dense(y_train.shape[1], activation='sigmoid')
  ])

  es = tf.keras.callbacks.EarlyStopping(patience=5, verbose=1, restore_best_weights=True)
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy')
  model.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[es], verbose=1, epochs=50)
  model.save('../../models/{}-{}.h5'.format(layer, ont))

  preds_val = model.predict(X_val)
  np.save('../../preds/{}-{}-val.npy'.format(layer, ont), preds_val)

  preds_test = model.predict(X_test)
  np.save('../../preds/{}-{}-test.npy'.format(layer, ont), preds_test)

  del model
  gc.collect()


def preprocess(df, subseq=1022):
  y = []
  positions = []
  sequences = df.iloc[:, 1].values
  target = df.iloc[:, 2:].values
  for i in tqdm(range(len(sequences))):
    len_seq = int(np.ceil(len(sequences[i]) / subseq))
    for idx in range(len_seq):
      y.append(target[i])
      positions.append(i)
  return np.array(y), np.array(positions)

def emb_method(matrix_embs, method):
  if method == 'mean':
    return np.mean(matrix_embs, axis=0)

def protein_embedding(X, y, pos, method='mean'):
  n_X = []
  last_pos = pos[0]
  cur_emb = []
  n_y = [y[0]]
  for i in range(len(X)):
    cur_pos = pos[i]
    if last_pos == cur_pos:
      cur_emb.append(X[i])
    else:
      n_X.append(emb_method(np.array(cur_emb), method))
      last_pos = cur_pos
      cur_emb = [X[i]]
      n_y.append(y[i])
  n_X.append(emb_method(np.array(cur_emb), method))

  return np.array(n_X), np.array(n_y)

if __name__ == '__main__':
  main()
