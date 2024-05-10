import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys
import click
from evaluate import *

@click.command()
@click.option('--ont', '-ont', help="Ontology")

def main(ont):

  train = pd.read_csv('base/{}_train.csv'.format(ont))
  test = pd.read_csv('base/{}_test.csv'.format(ont))
  y_train = train.iloc[:, 2:].values
  y_test = test.iloc[:, 2:].values

  seq_train = preprocess(train, 'train')
  seq_test = preprocess(test, 'test')

  with open('base/reference-{}.fasta'.format(ont), 'w') as f:
    print(seq_train, file=f)

  with open('base/queries-{}.fasta'.format(ont), 'w') as f:
    print(seq_test, file=f)

  os.system('~/diamond makedb --in "base/reference-{}.fasta" -d base/reference-{}'.format(ont, ont))

  seq = {}
  s=''
  k=''
  with open('base/queries-{}.fasta'.format(ont), "r") as f:
    for lines in f:
      if lines[0]==">":
        if s!='':
          seq[k] = s
          s=''
        k = lines[1:].strip('\n')
      else:
        s+=lines.strip('\n')
  seq[k] = s

  output = os.popen("~/diamond blastp -d base/reference-{}.dmnd -q base/queries-{}.fasta --outfmt 6 qseqid sseqid bitscore -e 0.001".format(ont, ont)).readlines()

  with open('base/output-{}.txt'.format(ont), 'w') as f:
    print(''.join(output), file=f)

  with open('base/output-{}.txt'.format(ont)) as f:
    output = f.readlines()

  test_bits={}
  test_train={}
  for lines in output:
    line = lines.strip('\n').split()
    try:
      if float(line[2]) >= 300:
        if line[0] in test_bits:
          test_bits[line[0]].append(float(line[2]))
          test_train[line[0]].append(line[1])
        else:
          test_bits[line[0]] = [float(line[2])]
          test_train[line[0]] = [line[1]]
    except:
      pass

  preds_score=[]
  nlabels = len(y_test[0])

  for s in seq:
    probs = np.zeros(nlabels, dtype=np.float32)
    if s in test_bits:
      weights = np.array(test_bits[s])/np.sum(test_bits[s])

      for j in range(len(test_train[s])):
        id = int(test_train[s][j].split('_')[1])
        temp = y_train[id]
        probs+= weights[j] * temp

    preds_score.append(probs)

  preds_score = np.array(preds_score)
  np.save('predictions/diamond-{}.npy'.format(ont), preds_score)

def preprocess(df, mode):
  seq = df.sequence.values
  id = 0
  fasta = ''
  for i in tqdm(seq):
    fasta += '>' + str(mode) + '_' + str(id) + '\n'
    id += 1
    fasta += i
    fasta += '\n'
  return fasta


if __name__ == '__main__':
  main()


