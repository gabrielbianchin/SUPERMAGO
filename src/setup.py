import os

os.system('mkdir ../base')
os.system('mkdir ../embs')
os.system('mkdir ../models')
os.system('mkdir ../preds')
os.system('wget http://github.com/bbuchfink/diamond/releases/download/v2.0.14/diamond-linux64.tar.gz')
os.system('tar xzf diamond-linux64.tar.gz')
os.system('rm diamond-linux64.tar.gz')
os.system('mv diamond ..')