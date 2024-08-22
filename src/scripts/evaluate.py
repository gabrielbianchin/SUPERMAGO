from tqdm import tqdm
import numpy as np
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

def evaluate(preds, gt, ontologies_names, ontology, ic, root):
  preds = propagate_preds(preds, ontologies_names, ontology)
  wfmax = 0
  fmax = 0
  fmax_s = 0
  smin = 1e100
  pr_arr, rc_arr = [], []
  for tau in tqdm(np.linspace(0, 1, 101)):
    wpr, wrc, num_prot_w = 0, 0, 0
    pr_s, rc_s, num_prot_s = 0, 0, 0
    pr_n, rc_n, num_prot_n = 0, 0, 0
    ru, mi = 0, 0
    for i, pred in enumerate(preds):
      protein_pred = set(ontologies_names[pred >= tau].tolist())
      protein_gt = set(ontologies_names[gt[i] == 1].tolist())

      ic_pred = sum(ic[q] for q in protein_pred)
      ic_gt = sum(ic[q] for q in protein_gt)
      ic_intersect = sum(ic[q] for q in protein_pred.intersection(protein_gt))

      # wfmax
      if ic_pred > 0:
        num_prot_w += 1
        wpr += (ic_intersect / ic_pred)
      if ic_gt > 0:
        wrc += (ic_intersect / ic_gt)

      # fmax
      if len(protein_pred) > 0:
        num_prot_n += 1
        pr_n += len(protein_pred.intersection(protein_gt)) / len(protein_pred)
      rc_n += len(protein_pred.intersection(protein_gt)) / len(protein_gt)

      # smin
      tp = protein_pred.intersection(protein_gt)
      fp = protein_pred - tp
      fn = protein_gt - tp
      for go_id in fp:
        mi += ic[go_id]
      for go_id in fn:
        ru += ic[go_id]

      # fmax_s
      if root in protein_pred:
        protein_pred.remove(root)
      protein_gt.remove(root)
      if len(protein_pred) > 0:
        num_prot_s += 1
        pr_s += len(protein_pred.intersection(protein_gt)) / len(protein_pred)
      if len(protein_gt) > 0:
        rc_s += len(protein_pred.intersection(protein_gt)) / len(protein_gt)

    # wfmax
    if num_prot_w > 0:
      tau_wpr = wpr / num_prot_w
    else:
      tau_wpr = 0

    tau_wrc = wrc / len(preds)

    if tau_wrc + tau_wpr > 0:
      tau_wfmax = (2 * tau_wpr * tau_wrc) / (tau_wpr + tau_wrc)
      wfmax = max(wfmax, tau_wfmax)

    # fmax
    if num_prot_n > 0:
      tau_pr_n = pr_n / num_prot_n
    else:
      tau_pr_n = 0

    tau_rc_n = rc_n / len(preds)

    if tau_pr_n + tau_rc_n > 0:
      tau_fmax = (2 * tau_pr_n * tau_rc_n) / (tau_pr_n + tau_rc_n)
      fmax = max(fmax, tau_fmax)

    # AuPRC
    pr_arr.append(tau_pr_n)
    rc_arr.append(tau_rc_n)

    # smin
    ru = ru / len(preds)
    mi = mi / len(preds)
    smin = min(smin, math.sqrt((ru * ru) + (mi * mi)))

    # fmax_s
    if num_prot_s > 0:
      tau_pr_s = pr_s / num_prot_s
    else:
      tau_pr_s = 0

    tau_rc_s = rc_s / len(preds)

    if tau_pr_s + tau_rc_s > 0:
      tau_fmax_s = (2 * tau_pr_s * tau_rc_s) / (tau_pr_s + tau_rc_s)
      fmax_s = max(fmax_s, tau_fmax_s)

  # AuPRC
  pr_arr = np.array(pr_arr)
  rc_arr = np.array(rc_arr)
  sorted_index = np.argsort(rc_arr)
  rc_arr = rc_arr[sorted_index]
  pr_arr = pr_arr[sorted_index]
  auprc = np.trapz(pr_arr, rc_arr)

  # IAuPRC
  ipr_arr, irc_arr = [], []
  for tau in np.linspace(0, 1, 101):
    if len(np.where(rc_arr >= tau)[0]) != 0:
      idx = np.where(rc_arr >= tau)[0][0]
      irc_arr.append(tau)
      ipr_arr.append(max(pr_arr[idx:]))
  iauprc = np.trapz(ipr_arr, irc_arr)
  print('Fmax:', fmax)
  print('Fmax*:', fmax_s)
  print('wFmax:', wfmax)
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
