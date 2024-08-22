# SUPERMAGO, SUPERMAGO+ and SUPERMAGO+Web: Protein Function Prediction based on Transformers Embeddings

The paper is under review.

## Dataset
The dataset for this work is available [here](https://zenodo.org/records/10982903).
The IC values used in evaluation is available [here](https://zenodo.org/records/13362841).

## Models
Our layer-base models for each ontology are available [here](https://drive.google.com/drive/folders/1JITfN21nWZX3DYGbi0wDDPwUX4eUuEca?usp=sharing).

## Predictions
The predictions of SUPERMAGO and SUPERMAGO+ are available [here](https://drive.google.com/drive/folders/1ynayNr_HEC3tNC6sAtXplzpQ9s2xnpdG?usp=sharing).

## Reproducibility
* Create the folders ```embs```, ```models```, ```predictions```, and ```base```.
* Unzip the dataset into the ```base``` folder.
* For each ontology, run:
```
python extract_esm2t36.py --ont ontology
python extract_prott5.py --ont ontology
python automl.py --ont ontology --backbone backbone --layer layer
python finetuning.py --ont ontology --backbone backbone --layer layer
python ensemble.py --ont ontology
python diamond.py --ont ontology
python supermago+.py --ont ontology
```
The parameter **ontology** should be ```bp```, ```cc```, or ```mf``` for Biological Process (BP), Cellular Component (CC), or Molecular Function (MF), respectively.
The parameter **backbone** should be ```esm2t36``` or ```prott5``` for ESM2 T36 or ProtT5, respectively.
The parameter **layer** should be ```24```, ```23```, or ```22``` for ProtT5, or ```36```, ```35```, or ```34``` for ESM2 T36.

The content of each file is as follows:
* ```extract_esm2t36.py``` extracts the embeddings for a specific ontology from layers 36, 35, and 34 of ESM2 T36.
* ```extract_prott5.py``` extracts the embeddings for a specific ontology from layers 24, 23, and 22 of ProtT5.
* ```automl.py``` runs AutoML for a specific ontology, layer, and backbone.
* ```finetuning.py``` fine-tunes the best configuration discovered by AutoML for a specific ontology, layer, and backbone.
* ```ensemble.py``` generates the final prediction of SUPER-MAGO for a specific ontology.
* ```diamond.py``` runs DIAMOND with bitscore selection for a specific ontology.
* ```supermago+.py``` generates the final prediction of SUPER-MAGO+ for a specific ontology.
