import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import click
from evaluate import *

@click.command()
@click.option('--ont', '-ont', help="Ontology")

def main(ont):

  theta = {'bp': [0.1, 0.2, 0.2, 0.3, 0.1, 0.1], 'cc': [0.1, 0.4, 0.1, 0.1, 0.2, 0.1], 'mf': [0.2, 0.2, 0.1, 0.1, 0.3, 0.1]}

  train = pd.read_csv('base/{}_train.csv'.format(ont))
  val = pd.read_csv('base/{}_val.csv'.format(ont))
  test = pd.read_csv('base/{}_test.csv'.format(ont))

  y_train = train.iloc[:, 2:].values
  y_val = val.iloc[:, 2:].values
  y_test = test.iloc[:, 2:].values
  y = np.concatenate([y_train, y_val, y_test])
  ontologies_names = test.columns[2:].values

  esm2t36_layer36 = np.load('predictions/esm2t36-36-{}.npy'.format(ont))
  esm2t36_layer35 = np.load('predictions/esm2t36-35-{}.npy'.format(ont))
  esm2t36_layer34 = np.load('predictions/esm2t36-34-{}.npy'.format(ont))
  prott5_layer24 = np.load('predictions/prott5-24-{}.npy'.format(ont))
  prott5_layer23 = np.load('predictions/prott5-23-{}.npy'.format(ont))
  prott5_layer22 = np.load('predictions/prott5-22-{}.npy'.format(ont))

  if ont == 'bp':
    ontology = generate_ontology('base/go.obo', specific_space=True, name_specific_space='biological_process')
  elif ont == 'cc':
    ontology = generate_ontology('base/go.obo', specific_space=True, name_specific_space='cellular_component')
  else:
    ontology = generate_ontology('base/go.obo', specific_space=True, name_specific_space='molecular_function')

  preds = esm2t36_layer36 * theta[ont][0] + esm2t36_layer35 * theta[ont][1] + esm2t36_layer34 * theta[ont][2] + \
          prott5_layer24 * theta[ont][3] + prott5_layer23 * theta[ont][4] + prott5_layer22 * theta[ont][5]

  evaluate(preds, y_test, ontologies_names, ontology, y)
  np.save('predictions/supermago-{}.npy'.format(ont), preds)


if __name__ == '__main__':
  main()
