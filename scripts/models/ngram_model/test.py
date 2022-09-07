from ngram_lm import NGramLanguageModel

lm = NGramLanguageModel()
lm.load_model("trained_models/0.5h/00.pkl")
print(lm.ngram_counter)
print(lm.assign_logprob("ð ɪ s ɪ z"))
print(lm.assign_logprob("ð ɪ s z ɪ"))