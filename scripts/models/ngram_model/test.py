from ngram_lm import NGramLM, UnigramLM
model_type = "unigram"
if model_type == "unigram":
    lm = UnigramLM()
else:
    lm = NGramLM()
lm.load("trained_models/ngrams/unigrams/0.5h/00.pkl")
print(lm.parameters)
print(lm.assign_logprob("ð ɪ s ɪ z"))
print(lm.assign_logprob("^ &é ! ) %"))