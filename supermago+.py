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

  alpha = {'bp': 0.13, 'cc': 0.51, 'mf': 0.57}

  train = pd.read_csv('base/{}_train.csv'.format(ont))
  val = pd.read_csv('base/{}_val.csv'.format(ont))
  test = pd.read_csv('base/{}_test.csv'.format(ont))

  y_train = train.iloc[:, 2:].values
  y_val = val.iloc[:, 2:].values
  y_test = test.iloc[:, 2:].values
  y = np.concatenate([y_train, y_val, y_test])
  ontologies_names = test.columns[2:].values

  preds_supermago = np.load('predictions/supermago-{}.npy'.format(ont))
  preds_diamond = np.load('predictions/diamond-{}.npy'.format(ont))

  if ont == 'bp':
    ontology = generate_ontology('base/go.obo', specific_space=True, name_specific_space='biological_process')
  elif ont == 'cc':
    ontology = generate_ontology('base/go.obo', specific_space=True, name_specific_space='cellular_component')
  else:
    ontology = generate_ontology('base/go.obo', specific_space=True, name_specific_space='molecular_function')

  preds = []
  for i, j in zip(preds_diamond, preds_supermago):
    if np.sum(i) != 0:
      preds.append(i * (1-alpha[ont]) + j * alpha[ont])
    else:
      preds.append(j)
  preds = np.array(preds)
  evaluate(preds, y_test, ontologies_names, ontology, y)
  np.save('predictions/supermago+-{}.npy'.format(ont), preds)


if __name__ == '__main__':
  main()
