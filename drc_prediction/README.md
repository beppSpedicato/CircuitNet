# DRC Violation prediction task

### Preliminaries
Setup the repo .venv
- create venv
- install packages from requirements
- install torch
- install openmim
- mim install mmcv

### To start
aim init
aim up -p PORT # to see the run results

### Configure
under ./config there are two files:
- drc_test.yaml and drc_train.yaml # hydra configuration files

### download and manage dataset
- donwload routability_features dataset from [here](https://drive.google.com/drive/folders/1Xp2y29Le6Doo3meKhTZClVwxG_7z2QuF)
- Put it under ./routability_features folder
- Under preprocess_scripts, run: python decompress_routability.py
- then, run: python generate_training_set.py
- You will have the preprocessed dataset under ./training_set/DRC folder

### Start train
python train.py -m

### Start test
python test.py -m