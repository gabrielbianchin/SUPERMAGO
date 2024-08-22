import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import gc
import pandas as pd
from tqdm import tqdm
import click
from evaluate import *

gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

@click.command()
@click.option('--ont', '-ont', help="Ontology (bp, cc or mf)")

def main(ont):
  y_val = pd.read_csv('../base/{}_val.csv'.format(ont)).iloc[:, 2:].values
  y_test = pd.read_csv('../base/{}_test.csv'.format(ont)).iloc[:, 2:].values
  ontologies_names = pd.read_csv('../base/{}_val.csv'.format(ont)).columns[2:].values
  ic_ont = pd.read_csv('../base/{}_ic.csv'.format(ont))
  ic = ic_ont.set_index('terms')['IC'].to_dict()

  if ont == 'bp':
    ontology = generate_ontology('../base/go.obo', specific_space=True, name_specific_space='biological_process')
    root = 'GO:0008150'
  elif ont == 'cc':
    ontology = generate_ontology('../base/go.obo', specific_space=True, name_specific_space='cellular_component')
    root = 'GO:0005575'
  else:
    ontology = generate_ontology('../base/go.obo', specific_space=True, name_specific_space='molecular_function')
    root = 'GO:0003674'

  preds_val, preds_test = [], []
  for layer in ['36', '24', '35', '23', '34', '22', '33', '21', '32', '20']:
    preds_val.append(np.load('../preds/{}-{}-val.npy'.format(layer, ont)))
    preds_test.append(np.load('../preds/{}-{}-test.npy'.format(layer, ont)))

  input_data_val = np.hstack(preds_val)
  input_data_test = np.hstack(preds_test)

  model = create_custom_nn(preds_val[0].shape[1])
  es = tf.keras.callbacks.EarlyStopping(patience=5, verbose=1, restore_best_weights=True)
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy')
  model.fit(input_data_val, y_val, epochs=50, batch_size=32, validation_split=0.2, callbacks=[es])
  model.save('../models/stacking-{}.h5'.format(ont))

  preds_supermago_val = model.predict(input_data_val)
  np.save('../preds/supermago-{}-val.npy'.format(ont), preds_supermago_val)

  preds_supermago_test = model.predict(input_data_test)
  np.save('../preds/supermago-{}-test.npy'.format(ont), preds_supermago_test)
  evaluate(preds_supermago_test, y_test, ontologies_names, ontology, ic, root)

  del model
  gc.collect()

class NormalizedWeightedSum(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(NormalizedWeightedSum, self).__init__(**kwargs)

  def build(self, input_shape):
    self._custom_weights = self.add_weight(name='custom_weights',
                     shape=(input_shape[-1], 1),
                     initializer='uniform',
                     trainable=True)
    super(NormalizedWeightedSum, self).build(input_shape)

  def call(self, inputs):
    normalized_weights = tf.nn.softmax(self._custom_weights, axis=0)
    inputs = tf.expand_dims(inputs, axis=-1)
    weighted_sum = tf.reduce_sum(inputs * normalized_weights, axis=1, keepdims=True)
    return weighted_sum

  def compute_output_shape(self, input_shape):
    return (input_shape[0], 1)

def create_custom_nn(num_outputs, num_models=10):
  input_layer = tf.keras.layers.Input(shape=(num_models * num_outputs,))
  outputs = []
  for i in range(num_outputs):
    model_predictions = [tf.expand_dims(input_layer[:, j * num_outputs + i], axis=1) for j in range(num_models)]
    concatenated_predictions = tf.keras.layers.Concatenate(axis=1)(model_predictions)
    output = NormalizedWeightedSum()(concatenated_predictions)
    outputs.append(output)
  final_output = tf.keras.layers.Concatenate()(outputs)
  flatten_output = tf.keras.layers.Flatten()(final_output)
  model = tf.keras.models.Model(inputs=input_layer, outputs=flatten_output)
  return model

if __name__ == '__main__':
  main()
