import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformers import AutoModel, AutoTokenizer
import torch
import sys
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
import click

@click.command()
@click.option('--ont', '-ont', help="Ontology")
def main(ont):

  model_name = 'facebook/esm2_t36_3B_UR50D'
  window = 1024
  embed_size = 2560

  train = preprocess(df=pd.read_csv('base/{}_train.csv'.format(ont)), subseq=window-2)
  val = preprocess(df=pd.read_csv('base/{}_val.csv'.format(ont)), subseq=window-2)
  test = preprocess(df=pd.read_csv('base/{}_test.csv'.format(ont)), subseq=window-2)

  tokenizer = AutoTokenizer.from_pretrained(model_name)
  device = torch.device("cuda:0")
  model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
  model.eval()

  # training
  process_data(train, tokenizer, model, device, embed_size, ont, "train")

  # validation
  process_data(val, tokenizer, model, device, embed_size, ont, "val")

  # test
  process_data(test, tokenizer, model, device, embed_size, ont, "test")


def process_data(data, tokenizer, model, device, embed_size, ont, data_type):
  embeds_layer_36 = np.zeros((len(data), embed_size))
  embeds_layer_35 = np.zeros((len(data), embed_size))
  embeds_layer_34 = np.zeros((len(data), embed_size))

  for i, this_seq in enumerate(tqdm(data)):
    embeds_layer_36[i], embeds_layer_35[i], embeds_layer_34[i] = get_embeddings(this_seq, tokenizer, model, device)
    gc.collect()
    torch.cuda.empty_cache()
  np.save('embs/esm2t36-36-{}-{}.npy'.format(ont, data_type), embeds_layer_36)
  np.save('embs/esm2t36-35-{}-{}.npy'.format(ont, data_type), embeds_layer_35)
  np.save('embs/esm2t36-34-{}-{}.npy'.format(ont, data_type), embeds_layer_34)

def preprocess(df, subseq):
  prot_list = []
  sequences = df.iloc[:, 1].values

  for i in tqdm(range(len(sequences))):
    len_seq = int(np.ceil(len(sequences[i]) / subseq))
    for idx in range(len_seq):
      if idx != len_seq - 1:
        prot_list.append(sequences[i][idx * subseq : (idx + 1) * subseq])
      else:
        prot_list.append(sequences[i][idx * subseq :])

  return prot_list

def get_embeddings(seq, tokenizer, model, device):
  batch_seq = [" ".join(list(seq))]
  ids = tokenizer(batch_seq)
  input_ids = torch.tensor(ids['input_ids']).to(device)
  attention_mask = torch.tensor(ids['attention_mask']).to(device)

  with torch.no_grad():
    embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

  return embedding_repr.hidden_states[36][0].detach().cpu().numpy().mean(axis=0), \
         embedding_repr.hidden_states[35][0].detach().cpu().numpy().mean(axis=0), \
         embedding_repr.hidden_states[34][0].detach().cpu().numpy().mean(axis=0),


if __name__ == '__main__':
  main()
