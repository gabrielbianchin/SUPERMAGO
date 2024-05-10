import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc
from collections import Counter
import math

def propagate_preds(predictions, ontologies_names, ontology):

  ont_n = ontologies_names.tolist()

  list_of_parents = []

  for idx_term in range(len(ont_n)):
    this_list_of_parents = []
    for parent in ontology[ont_n[idx_term]]['ancestors']:
      this_list_of_parents.append(ont_n.index(parent))
    list_of_parents.append(list(set(this_list_of_parents)))

  for idx_protein in tqdm(range(len(predictions))):
    for idx_term in range(len(ont_n)):
      for idx_parent in list_of_parents[idx_term]:
        predictions[idx_protein, idx_parent] = max(predictions[idx_protein, idx_parent], predictions[idx_protein, idx_term])
  return predictions

def evaluate(predictions, ground_truth, ontologies_names, ontology, y):

  predictions = propagate_preds(predictions, ontologies_names, ontology)

  annots = []
  for i in y:
    actual = []
    for j in range(len(i.tolist())):
      if i[j] == 1:
        actual.append(ontologies_names[j])
    annots.append(actual)

  cnt = Counter()
  for x in annots:
    cnt.update(x)

  ic = {}

  for go_id, n in cnt.items():
    if go_id in ontology:
      parents = ontology[go_id]['parents']
      if len(parents) == 0:
        min_n = n
      else:
        min_n = min([cnt[x] for x in parents])
      ic[go_id] = math.log(min_n / n, 2)

  precisions = []
  recalls = []
  fmax = -1
  smin = 1e100

  for i in tqdm(range(1, 101)):
    threshold = i/100
    p, r = 0, 0
    ru, mi = 0, 0
    number_of_proteins = 0

    for idx_protein in range(len(predictions)):
      protein_pred = set(ontologies_names[np.where(predictions[idx_protein, :] >= threshold)[0]].tolist())
      protein_gt = set(ontologies_names[np.where(ground_truth[idx_protein, :] == 1)[0]].tolist())

      if len(protein_pred) > 0:
        number_of_proteins += 1
        p += len(protein_pred.intersection(protein_gt)) / len(protein_pred)
      r += len(protein_pred.intersection(protein_gt)) / len(protein_gt)

      tp = protein_pred.intersection(protein_gt)
      fp = protein_pred - tp
      fn = protein_gt - tp
      for go_id in fp:
        mi += ic[go_id]
      for go_id in fn:
        ru += ic[go_id]

    if number_of_proteins > 0:
      threshold_p = p / number_of_proteins
    else:
      threshold_p = 0

    threshold_r = r / len(predictions)

    precisions.append(threshold_p)
    recalls.append(threshold_r)

    f1 = 0
    if threshold_p > 0 or threshold_r > 0:
      f1 = (2 * threshold_p * threshold_r) / (threshold_p + threshold_r)

    if f1 > fmax:
      fmax = f1

    ru = ru / len(predictions)
    mi = mi / len(predictions)

    smin_atual = math.sqrt((ru * ru) + (mi * mi))

    if smin_atual < smin:
      smin = smin_atual

  precisions = np.array(precisions)
  recalls = np.array(recalls)
  sorted_index = np.argsort(recalls)
  recalls = recalls[sorted_index]
  precisions = precisions[sorted_index]
  auprc = np.trapz(precisions, recalls)

  new_p = []
  new_r = []
  for i in range(1, 101):
    if len(np.where(recalls >= i/100)[0]) != 0:
      idx = np.where(recalls >= i/100)[0][0]
      new_r.append(i/100)
      new_p.append(max(precisions[idx:]))

  iauprc = np.trapz(new_p, new_r)

  print('Fmax:', fmax)
  print('Smin:', smin)
  print('AuPRC:', auprc)
  print('IAuPRC:', iauprc)


def get_ancestors(ontology, term):
  list_of_terms = []
  list_of_terms.append(term)
  data = []

  while len(list_of_terms) > 0:
    new_term = list_of_terms.pop(0)

    if new_term not in ontology:
      break
    data.append(new_term)
    for parent_term in ontology[new_term]['parents']:
      if parent_term in ontology:
        list_of_terms.append(parent_term)

  return data

def generate_ontology(file, specific_space=False, name_specific_space=''):
  ontology = {}
  gene = {}
  flag = False
  with open(file) as f:
    for line in f.readlines():
      line = line.replace('\n','')
      if line == '[Term]':
        if 'id' in gene:
          ontology[gene['id']] = gene
        gene = {}
        gene['parents'], gene['alt_ids'] = [], []
        flag = True

      elif line == '[Typedef]':
        flag = False

      else:
        if not flag:
          continue
        items = line.split(': ')
        if items[0] == 'id':
          gene['id'] = items[1]
        elif items[0] == 'alt_id':
          gene['alt_ids'].append(items[1])
        elif items[0] == 'namespace':
          if specific_space:
            if name_specific_space == items[1]:
              gene['namespace'] = items[1]
            else:
              gene = {}
              flag = False
          else:
            gene['namespace'] = items[1]
        elif items[0] == 'is_a':
          gene['parents'].append(items[1].split(' ! ')[0])
        elif items[0] == 'name':
          gene['name'] = items[1]
        elif items[0] == 'is_obsolete':
          gene = {}
          flag = False

    key_list = list(ontology.keys())
    for key in key_list:
      ontology[key]['ancestors'] = get_ancestors(ontology, key)
      for alt_ids in ontology[key]['alt_ids']:
        ontology[alt_ids] = ontology[key]

    for key, value in ontology.items():
      if 'children' not in value:
        value['children'] = []
      for p_id in value['parents']:
        if p_id in ontology:
          if 'children' not in ontology[p_id]:
            ontology[p_id]['children'] = []
          ontology[p_id]['children'].append(key)

  return ontology

