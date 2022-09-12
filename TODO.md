- [x] Move to new git organization
- [ ] Rewrite preprocessing tools after deleting the option phonemic words.
- [ ] Run the models on the Marvin dataset.
- [ ] Notice to marvin that `output_path` option was added to the script `split_train_val_test_lm.py`
- [x] Make all training files for all hours
- [ ] Create a sh script for training all ngram models on all training sets
- [ ] Enable to import modules without sys.append. Correct the modules `scripts.models.ngram_model.compute_prob_ngram_lm.py` and `scripts.providence.phonemize`
- [ ]

Scripts to compute ngram_lm probabilites on gold.csv file:
`python scripts/models/ngram_model/compute_prob_ngram_lm.py --input_path data/model_evaluation/ --model_path trained_models/ngrams/unigrams/0.5h/00.pkl --model_type unigram --mode test --text --remove_word_spaces`

This will out a `.txt` file. Use this file as argument to this script for computing the accuracy on syntactic task:

`python scripts/metrics/compute_syntactic.py -o results -g data/model_evaluation/ -p trained_models/ngrams/unigrams/0.5h/data/tmp/model_evaluation/ -k test --is_text`
