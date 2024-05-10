import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformers import T5EncoderModel, T5Tokenizer
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

  model_name = 'Rostlab/prot_t5_xl_half_uniref50-enc'
  window = 1024
  embed_size = 1024

  train = preprocess(df=pd.read_csv('base/{}_train.csv'.format(ont)), subseq=window-2)
  val = preprocess(df=pd.read_csv('base/{}_val.csv'.format(ont)), subseq=window-2)
  test = preprocess(df=pd.read_csv('base/{}_test.csv'.format(ont)), subseq=window-2)

  tokenizer = T5Tokenizer.from_pretrained(model_name)
  device = torch.device("cuda:0")
  model = T5EncoderModel.from_pretrained(model_name, output_hidden_states=True).to(device)
  model.eval()

  # training
  process_data(train, tokenizer, model, device, embed_size, ont, "train")

  # validation
  process_data(val, tokenizer, model, device, embed_size, ont, "val")

  # test
  process_data(test, tokenizer, model, device, embed_size, ont, "test")


def process_data(data, tokenizer, model, device, embed_size, ont, data_type):
  embeds_layer_24 = np.zeros((len(data), embed_size))
  embeds_layer_23 = np.zeros((len(data), embed_size))
  embeds_layer_22 = np.zeros((len(data), embed_size))

  for i, this_seq in enumerate(tqdm(data)):
    embeds_layer_24[i], embeds_layer_23[i], embeds_layer_22[i] = get_embeddings(this_seq, tokenizer, model, device)
    gc.collect()
    torch.cuda.empty_cache()
  np.save('embs/prott5-24-{}-{}.npy'.format(ont, data_type), embeds_layer_24)
  np.save('embs/prott5-23-{}-{}.npy'.format(ont, data_type), embeds_layer_23)
  np.save('embs/prott5-22-{}-{}.npy'.format(ont, data_type), embeds_layer_22)

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

  return embedding_repr.hidden_states[24][0].detach().cpu().numpy().mean(axis=0), \
         embedding_repr.hidden_states[23][0].detach().cpu().numpy().mean(axis=0), \
         embedding_repr.hidden_states[22][0].detach().cpu().numpy().mean(axis=0),


if __name__ == '__main__':
  main()
