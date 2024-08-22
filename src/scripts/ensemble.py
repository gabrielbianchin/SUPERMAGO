import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
import click
from evaluate import *

gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

@click.command()
@click.option('--ont', '-ont', help="Ontology (bp, cc or mf)")

def main(ont):
  y_val = pd.read_csv('../../base/{}_val.csv'.format(ont)).iloc[:, 2:].values
  y_test = pd.read_csv('../../base/{}_test.csv'.format(ont)).iloc[:, 2:].values
  ontologies_names = pd.read_csv('../../base/{}_val.csv'.format(ont)).columns[2:].values
  ic_ont = pd.read_csv('../../base/{}_ic.csv'.format(ont))
  ic = ic_ont.set_index('terms')['IC'].to_dict()

  if ont == 'bp':
    ontology = generate_ontology('../../base/go.obo', specific_space=True, name_specific_space='biological_process')
    root = 'GO:0008150'
  elif ont == 'cc':
    ontology = generate_ontology('../../base/go.obo', specific_space=True, name_specific_space='cellular_component')
    root = 'GO:0005575'
  else:
    ontology = generate_ontology('../../base/go.obo', specific_space=True, name_specific_space='molecular_function')
    root = 'GO:0003674'

  preds_val, preds_test = [], []
  for layer in ['supermago', 'diamond']:
    preds_val.append(np.load('../../preds/{}-{}-val.npy'.format(layer, ont)))
    preds_test.append(np.load('../../preds/{}-{}-test.npy'.format(layer, ont)))

  mask_val = []
  for i in range(len(preds_val[0])):
    mask_val.append([(np.sum(preds_val[0][i]) > 0).astype(np.float32), (np.sum(preds_val[1][i]) > 0).astype(np.float32)])
  mask_val = np.array(mask_val)

  mask_test = []
  for i in range(len(preds_test[0])):
    mask_test.append([(np.sum(preds_test[0][i]) > 0).astype(np.float32), (np.sum(preds_test[1][i]) > 0).astype(np.float32)])
  mask_test = np.array(mask_test)

  input_data_val = np.hstack(preds_val)
  input_data_test = np.hstack(preds_test)

  model = create_custom_nn(preds_val[0].shape[1])
  es = tf.keras.callbacks.EarlyStopping(patience=5, verbose=1, restore_best_weights=True)
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy')
  model.fit([input_data_val, mask_val], y_val, epochs=50, batch_size=32, validation_split=0.2, callbacks=[es])
  model.save('../../models/ensemble-{}.h5'.format(ont))

  preds_supermago_val = model.predict([input_data_val, mask_val])
  np.save('../../preds/supermagoplus-{}-val.npy'.format(ont), preds_supermago_val)

  preds_supermago_test = model.predict([input_data_test, mask_test])
  np.save('../../preds/supermagoplus-{}-test.npy'.format(ont), preds_supermago_test)
  evaluate(preds_supermago_test, y_test, ontologies_names, ontology, ic, root)

class NormalizedWeightedSumWithMask(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(NormalizedWeightedSumWithMask, self).__init__(**kwargs)

  def build(self, input_shape):
    self._custom_weights = self.add_weight(name='custom_weights',
                     shape=(input_shape[1], 1),
                     initializer='uniform',
                     trainable=True)
    super(NormalizedWeightedSumWithMask, self).build(input_shape)

  def call(self, inputs, mask):
    expanded_mask = tf.expand_dims(mask, axis=-1)
    masked_weights = tf.where(expanded_mask > 0, self._custom_weights, tf.fill(tf.shape(self._custom_weights), -1e9))
    normalized_weights = tf.nn.softmax(masked_weights, axis=1)
    weighted_sum = tf.reduce_sum(inputs * tf.squeeze(normalized_weights, axis=-1), axis=1, keepdims=True)
    return weighted_sum

  def compute_output_shape(self, input_shape):
    return (input_shape[0], 1)

def create_custom_nn(num_outputs, num_models=2):
  input_layer = tf.keras.layers.Input(shape=(num_models * num_outputs,))
  mask_input = tf.keras.layers.Input(shape=(num_models, ), dtype=tf.float32)
  outputs = []
  for i in range(num_outputs):
    model_predictions = [tf.expand_dims(input_layer[:, j * num_outputs + i], axis=1) for j in range(num_models)]
    concatenated_predictions = tf.keras.layers.Concatenate(axis=1)(model_predictions)
    output = NormalizedWeightedSumWithMask()(concatenated_predictions, mask=mask_input)
    outputs.append(output)
  final_output = tf.keras.layers.Concatenate()(outputs)
  flatten_output = tf.keras.layers.Flatten()(final_output)
  model = tf.keras.models.Model(inputs=[input_layer, mask_input], outputs=flatten_output)
  return model

if __name__ == '__main__':
  main()
