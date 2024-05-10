import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import autokeras as ak
import click

gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

@click.command()
@click.option('--ont', '-ont', help="Ontology")
@click.option('--backbone', '-b', help="esm2t36 for ESM2 T36 or prott5 for ProtT5")
@click.option('--layer', '-l', help="Layer (36, 35, or 34 for ESM2 T36 or 24, 23, or 22 for ProtT5)")

def main(ont, backbone, layer):

  window = 1024

  y_train, pos_train = preprocess(df=pd.read_csv('base/{}_train.csv'.format(ont)), subseq=window-2)
  y_val, pos_val = preprocess(df=pd.read_csv('base/{}_val.csv'.format(ont)), subseq=window-2)
  y_test, pos_test = preprocess(df=pd.read_csv('base/{}_test.csv'.format(ont)), subseq=window-2)

  X_train = np.load('embs/{}-{}-{}-train.npy'.format(backbone, layer, ont))
  X_val = np.load('embs/{}-{}-{}-val.npy'.format(backbone, layer, ont))
  X_test = np.load('embs/{}-{}-{}-test.npy'.format(backbone, layer, ont))

  X_train, y_train = protein_embedding(X_train, y_train, pos_train)
  X_val, y_val = protein_embedding(X_val, y_val, pos_val)
  X_test, y_test = protein_embedding(X_test, y_test, pos_test)

  model = tf.keras.models.load_model('models/best-model-{}-{}-{}.h5'.format(backbone, layer, ont), compile=False, custom_objects=ak.CUSTOM_OBJECTS)
  model.compile(loss='binary_crossentropy')
  model.optimizer.learning_rate.assign(model.optimizer.learning_rate * 0.2)
  es = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
  model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[es], verbose=1)
  model.save('models/best-model-{}-{}-{}-finetuned.h5'.format(backbone, layer, ont))

  preds = model.predict(X_test)
  np.save('predictions/{}-{}-{}.npy'.format(backbone, layer, ont), preds)

def preprocess(df, subseq):
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
