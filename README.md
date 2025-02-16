# FusionAdapter


## Running the Experiments

### Requirements

+ Python 3.6.7
+ PyTorch 1.0.1
+ tensorboardX 1.8

You can also install dependencies by

```bash
pip install -r requirements.txt
```
### Dataset

We use WN9-IMG and FB-IMG to test our model. The orginal datasets and pretrain embeddings can be downloaded from [Lion ZS's repo]([https://github.com/Lion-ZS/OTKGE]). You can also download the zip files where we put the datasets and pretrain embeddings together in [Link](https://www.dropbox.com/scl/fo/0a6776fmf6j2ga8g97i83/AAkqh13l6H4p32clNiKAUh4?rlkey=v9obvygssg50wj25f7hkbw9pg&st=je2eg4n2&dl=0).

### Quick Start for Training & Testing

For training and testing on FusionAdapter, here is an example for quick start,

```bash
python main.py --seed 1 --few 5 --step train --eval_by_rel False --prefix WN9shot_pretrain  --device 0
python main.py --seed 1 --few 5 --step test  --eval_by_rel True --prefix WN9shot_pretrain  --device 0
```
