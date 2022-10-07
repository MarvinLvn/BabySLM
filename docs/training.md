# Text-based language models

All text-based language models are trained via [fairseq](https://github.com/facebookresearch/fairseq):

```bash
conda activate provi
git clone https://github.com/facebookresearch/fairseq.git
cd fairseq
# to make sure everything remains the same in the future
# we'll checkout to a specific commit
git checkout acd9a53607d1e5c64604e88fc9601d0ee56fd6f1
pip install -e .
# two additional dependencies for faster preprocessing
pip install fastBPE sacremoses
```

All commands will be given for the 128 hours training set blabla

### Train Ngram models
The bash script takes as input a folder organized as `hours/training_sets`. This script can be called with this command line:

```bash
sh scripts/models/ngram_model/train_all_ngram_models.sh -i <INPUT_FOLDER> -n <NGRAM_SIZE> -o <OUTPUT_FOLDER>

```
where:
- `-i   The folder storing all training files. Must be of the for hours/training_sets`
- `-n   The size of the ngrams.`
- `-o   The folder where all the trained models will be stored.`

### Train LSTM (phones)

1) Preprocess the data:

```bash
DATA=~/DATA/CPC_data/train/providence/cleaned/training_sets/128h/00/phones
MODEL=~/models/lstm_text/phones/128h/00
python scripts/text_lm/split_train_val_test_lm.py --input_path $DATA \
  --val_prop 0.1 --test_prop 0
fairseq-preprocess \
    --only-source \
    --trainpref $DATA/fairseq_train.txt \
    --validpref $DATA/fairseq_val.txt \
    --destdir $MODEL/data-bin \
    --workers 20
```

2) Train the model:

```bash
FAIRSEQ_SCRIPTS=/path/to/fairseq
MODEL=~/models/lstm_text/phones/128h/00
python ${FAIRSEQ_SCRIPTS}/train.py --fp16 $MODEL/data-bin \
      --task language_modeling \
      --save-dir ${MODEL} \
      --keep-last-epochs 2 \
      --tensorboard-logdir tensorboard \
      --arch lstm_lm \
      --decoder-embed-dim 200 --decoder-hidden-size 1024 --decoder-layers 3 \
      --decoder-out-embed-dim 200 \
      --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
      --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 1000 --warmup-init-lr 1e-07 \
      --dropout 0.1 --weight-decay 0.01 \
      --sample-break-mode none --tokens-per-sample 2048 \
      --max-tokens 163840 --update-freq 1 --max-update 100000 --patience 10
```

To train on phones, considering space as a separator token, you must train on 
`providence/cleaned/training_sets/128h/00/phones`.

### Train LSTM (words)

1) Preprocess the data:

```bash
DATA=~/DATA/CPC_data/train/providence/cleaned/training_sets/128h/00/sentences_bpe
MODEL=~/models/lstm_text/sentences_bpe/128h/00
python scripts/split_train_val_test_lm.py --input_path $DATA \
  --val_prop 0.1 --test_prop 0
fairseq-preprocess \
    --only-source \
    --trainpref $DATA/fairseq_train.txt \
    --validpref $DATA/fairseq_val.txt \
    --destdir $MODEL/data-bin \
    --workers 20
```

2) Train the model:

```bash
FAIRSEQ_SCRIPTS=/path/to/fairseq
MODEL=~/models/lstm_text/sentences_bpe/128h/00
python ${FAIRSEQ_SCRIPTS}/train.py --fp16 $MODEL/data-bin \
      --task language_modeling \
      --save-dir ${MODEL} \
      --keep-last-epochs 2 \
      --tensorboard-logdir tensorboard \
      --arch lstm_lm \
      --decoder-embed-dim 200 --decoder-hidden-size 1024 --decoder-layers 3 \
      --decoder-out-embed-dim 200 \
      --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
      --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 1000 --warmup-init-lr 1e-07 \
      --dropout 0.1 --weight-decay 0.01 \
      --sample-break-mode none --tokens-per-sample 2048 \
      --max-tokens 163840 --update-freq 1 --max-update 100000 --patience 10
```

# Speech-based language models

Audio-based models are trained using [this git repository](https://github.com/MarvinLvn/CPC2/tree/minibatch_building).
Please follow the installation instructions [there](https://github.com/MarvinLvn/CPC2/blob/minibatch_building/docs/installation.md).

### 1) Train CPC

### 2) Train K-means

### 3) Train LSTM