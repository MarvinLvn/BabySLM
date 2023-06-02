### Installation instructions to train and evaluate STELA models (CPC+K-means+LSTM)

Assuming you're in the BabySLM folder. First, make sure [sox](http://sox.sourceforge.net/) is installed, and then run:

```bash
# Install CPC + K-means dependencies
git clone https://github.com/MarvinLvn/CPC2.git  && cd CPC2
conda env create -f environment.yml
conda activate cpc2
pip install -e .

git clone https://github.com/facebookresearch/WavAugment.git && cd WavAugment
git checkout 357b2f9f09832cbe64ff76633eea8dbd5f1e97d1
pip install -e .

# Install LSTM dependencies (fairseq)
cd ..
git clone https://github.com/facebookresearch/fairseq.git && cd fairseq
git checkout acd9a53607d1e5c64604e88fc9601d0ee56fd6f1
pip install -e .
```