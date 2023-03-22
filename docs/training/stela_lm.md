# Train CPC + K-means + LSTM (speech-based)

Audio-based models are trained using [this git repository](https://github.com/MarvinLvn/CPC2/tree/minibatch_building).
Please follow the installation instructions [there](https://github.com/MarvinLvn/CPC2/blob/minibatch_building/docs/installation.md).

### 1) Train CPC

```bash
conda activate cpc2
pip install transformers
DATA=~/DATA/CPC_data/train/providence/cleaned/training_sets/128h/00/audio
MODEL=~/models/audio/128h/00/cpc_small
python /private/home/marvinlvn/BenchmarkLangAcq/CPC2/cpc/train.py --pathCheckpoint $MODEL \
                           --pathDB $DATA \
                           --file_extension .wav --nLevelsGRU 2 --save_step 2 --multihead_rnn \
                           --nEpoch ${NB_EPOCHS} --random_seed 42 --n_process_loader 1 --save_step 5 \
                          --max_size_loaded 200000000 --schedulerRamp 10 \
                           --dropout --augment_past --augment_type pitch artificial_reverb --samplingType=temporalsamespeaker \
                           --naming_convention=id_spkr_onset_offset
```

### 2) Train K-means

```bash
conda activate cpc2
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

```