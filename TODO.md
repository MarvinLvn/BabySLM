- [x] Move to new git organization
- [x] Train the models on the Marvin dataset.
- [ ] Notice to marvin that `output_path` option was added to the script `split_train_val_test_lm.py`
- [x] Make all training files for all hours
- [x] Create a sh script for training all ngram models on all training sets
- [ ] Notify that you have added the `output_path` in the scripts for computing the probabilities
- [ ] Enable to import modules without sys.append. Correct the modules `scripts.models.ngram_model.compute_prob_ngram_lm.py` and `scripts.providence.phonemize`
- [x] sh script for computting probabilities on the syntactic task for all the lms.


sh script for training the language models on all the datasets
`sh scripts/models/ngram_model/train_all_ngram_models.sh -i data/training_data -n 2 -o trained_models/ngrams/bigrams`

sh script for computing probabilities for all models:
`sh scripts/models/ngram_model/compute_prob_for_all_ngram_models.sh -i data/model_evaluation/lexical/ -o results/trigrams/ -m trained_models/ngrams/trigrams/ -t ngram -e dev`

sh script for computing all the scores for all the model on the syntactic task

`sh scripts/models/ngram_model/compute_scores_for_all_models.sh -g data/model_evaluation/ -p results/unigrams/ -k dev -t lexical`
