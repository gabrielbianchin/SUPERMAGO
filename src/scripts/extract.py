import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer
import torch
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
import click

@click.command()
@click.option('--model_name', '-m', help="Model (esm for ESM2 T36 or t5 for ProtT5)")
@click.option('--ont', '-ont', help="Ontology (bp, cc or mf)")

def main(model_name, ont):

  train = preprocess(df=pd.read_csv('../base/{}_train.csv'.format(ont)))
  val = preprocess(df=pd.read_csv('../base/{}_val.csv'.format(ont)))
  test = preprocess(df=pd.read_csv('../base/{}_test.csv'.format(ont)))
  device = torch.device("cuda:0")
  
  if model_name == 'esm':
    model_path = 'facebook/esm2_t36_3B_UR50D'
    embed_size = 2560
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, output_hidden_states=True).to(device)
  
  elif model_name == 't5':
    model_path = 'Rostlab/prot_t5_xl_half_uniref50-enc'
    embed_size = 1024
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5EncoderModel.from_pretrained(model_path, output_hidden_states=True).to(device)

  else:
    print('Model name error')
    return

  model.eval()
  process_data(train, tokenizer, model, device, embed_size, ont, "train", model_name)
  process_data(val, tokenizer, model, device, embed_size, ont, "val", model_name)
  process_data(test, tokenizer, model, device, embed_size, ont, "test", model_name)

  del model
  gc.collect()
  torch.cuda.empty_cache()

def process_data(data, tokenizer, model, device, embed_size, ont, data_type, model_name):
  layer_1 = np.zeros((len(data), embed_size))
  layer_2 = np.zeros((len(data), embed_size))
  layer_3 = np.zeros((len(data), embed_size))
  layer_4 = np.zeros((len(data), embed_size))
  layer_5 = np.zeros((len(data), embed_size))

  for i, this_seq in enumerate(tqdm(data)):
    layer_1[i], layer_2[i], layer_3[i], layer_4[i], layer_5[i] = get_embeddings(this_seq, tokenizer, model, device)
    gc.collect()
    torch.cuda.empty_cache()
    
  if model_name == 'esm':
    np.save('../embs/36-{}-{}.npy'.format(ont, data_type), layer_1)
    np.save('../embs/35-{}-{}.npy'.format(ont, data_type), layer_2)
    np.save('../embs/34-{}-{}.npy'.format(ont, data_type), layer_3)
    np.save('../embs/33-{}-{}.npy'.format(ont, data_type), layer_4)
    np.save('../embs/32-{}-{}.npy'.format(ont, data_type), layer_5)
  elif model_name == 't5':
    np.save('../embs/24-{}-{}.npy'.format(ont, data_type), layer_1)
    np.save('../embs/23-{}-{}.npy'.format(ont, data_type), layer_2)
    np.save('../embs/22-{}-{}.npy'.format(ont, data_type), layer_3)
    np.save('../embs/21-{}-{}.npy'.format(ont, data_type), layer_4)
    np.save('../embs/20-{}-{}.npy'.format(ont, data_type), layer_5)

def preprocess(df, subseq=1022):
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

  return embedding_repr.hidden_states[-1][0].detach().cpu().numpy().mean(axis=0), \
         embedding_repr.hidden_states[-2][0].detach().cpu().numpy().mean(axis=0), \
         embedding_repr.hidden_states[-3][0].detach().cpu().numpy().mean(axis=0), \
         embedding_repr.hidden_states[-4][0].detach().cpu().numpy().mean(axis=0), \
         embedding_repr.hidden_states[-5][0].detach().cpu().numpy().mean(axis=0),


if __name__ == '__main__':
  main()
