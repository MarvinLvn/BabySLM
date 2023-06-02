# Train CPC + K-means + LSTM (speech-based)

Audio-based models are trained using [this git repository](https://github.com/MarvinLvn/CPC2).
Please follow the installation instructions [there](https://github.com/MarvinLvn/CPC2/blob/minibatch_building/docs/installation.md).

### 1) Train CPC

```bash
conda activate cpc2
DATA=~/DATA/CPC_data/train/providence/cleaned/training_sets/128h/00/audio
MODEL=~/models/audio/128h/00/cpc_small
python CPC2/cpc/train.py --pathCheckpoint $MODEL \
                           --pathDB $DATA \
                           --file_extension .wav --nLevelsGRU 2 --save_step 2 --multihead_rnn \
                           --nEpoch ${NB_EPOCHS} --random_seed 42 --n_process_loader 1 --save_step 5 \
                          --max_size_loaded 200000000 --schedulerRamp 10 \
                           --dropout --augment_past --augment_type pitch artificial_reverb --samplingType=temporalsamespeaker \
                           --naming_convention=id_spkr_onset_offset
```

### 2) Train K-means

```bash
DATA=~/DATA/CPC_data/train/providence/cleaned/training_sets/128h/00/audio
KMEANS_MODEL=~/models/audio/128h/00/kmeans50
CPC_MODEL=~/models/audio/128h/00/cpc_small

# Retrieve CPC epoch that has the lowest validation loss
BEST_EPOCH=$(python scripts/best_val_epoch.py --model_path ${CPC_MODEL} | grep -oP "(?<=is : )([0-9]+)")
CPC_MODEL=${CPC_MODEL}/checkpoint_${BEST_EPOCH}.pt

# Train K-means
python CPC2/cpc/clustering/clustering_script.py $CPC_MODEL $KMEANS_MODEL $DATA \
                                                --recursionLevel 2 --extension wav --nClusters 50 --MAX_ITER 300 --save \
                                                --batchSizeGPU 200 --level_gru ${LEVEL_GRU} --perIterSize 1406 --save-last 5
```

### 3) Train LSTM

```bash
#  1) Discretize the training set using the K-means model
DATA=~/DATA/CPC_data/train/providence/cleaned/training_sets/128h/00/audio
OUTPUT=~/DATA/CPC_data/train/providence/tmp/quantized_train_128h_00_audio
PATH_KMEANS=~/models/audio/128h/00/kmeans50/checkpoint_last.pt
python scripts/audio_lm/quantize_audio.py ${PATH_KMEANS} ${DATA} ${OUTPUT} --file_extension .wav

# 2) Split train/val/test +  fairseq preprocess
python scripts/text_lm/split_train_val_test_lm.py --input_path ${OUTPUT}/quantized_outputs_2.txt
fairseq-preprocess --only-source \
      --trainpref $OUTPUT/quantized_train/fairseq_train.txt \
      --validpref $OUTPUT/quantized_train/fairseq_val.txt \
      --testpref $OUTPUT/quantized_train/fairseq_test.txt \
      --destdir $OUTPUT/fairseq_bin_data \
      --workers 20

# 3) Train LSTM on discretized units
OUTPUT=~/models/audio/128h/00/lstm
mkdir -p $OUTPUT
cp $OUTPUT/fairseq_bin_data/dict.txt $OUTPUT
cp -r $OUTPUT/fairseq_bin_data $OUTPUT
python fairseq/train.py --fp16 $OUTPUT/fairseq_bin_data \
      --task language_modeling \
      --save-dir ${OUTPUT} \
      --keep-last-epochs 2 \
      --tensorboard-logdir tensorboard \
      --arch lstm_lm \
      --decoder-embed-dim 200 --decoder-hidden-size 1024 --decoder-layers 3 \
      --decoder-out-embed-dim 200 \
      --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
      --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 1000 --warmup-init-lr 1e-07 \
      --dropout 0.1 --weight-decay 0.01 \
      --sample-break-mode none --tokens-per-sample 2048 \
      --max-tokens 163840 --update-freq 1 --max-update 100000
```