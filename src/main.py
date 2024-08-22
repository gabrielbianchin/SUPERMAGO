import os
import click

@click.command()
@click.option('--ont', '-ont', help="Ontology (bp, cc or mf)")

def main(ont):

  # Extracting embeddings
  print('Extracting embeddings...')
  for model_name in ['esm', 't5']:
    print('model_name')
    os.system('python3 src/extract.py -m {} -ont {}'.format(model_name, ont))
  print('Extract - Done')

  # Training - ESM
  print('Layer Classification - ESM...')
  for layer in ['36', '35', '34', '33', '32']:
    os.system('python3 src/layer_classification.py -ont {} -l {}'.format(ont, layer))
  print('Layer Classification - ESM - Done')

  # Training - T5
  print('Layer Classification - T5...')
  for layer in ['24', '23', '22', '21', '20']:
    os.system('python3 src/layer_classification.py -ont {} -l {}'.format(ont, layer))
  print('Layer Classification - T5 - Done')

  # Stacking
  print('Stacking...')
  os.system('python3 src/stacking.py -ont {}'.format(ont))
  print('Stacking - Done')

  # DIAMOND
  print('DIAMOND...')
  os.system('python3 src/diamond.py -ont {}'.format(ont))
  print('DIAMOND - Done')

  # Ensemble
  print('Ensemble...')
  os.system('python3 src/ensemble.py -ont {}'.format(ont))
  print('Ensemble - Done')


if __name__ == '__main__':
  main()