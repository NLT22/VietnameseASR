import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("./data/lang_bpe_500/bpe.model")

text = "hôm nay tôi học nhận dạng tiếng nói"
print(sp.encode(text, out_type=str))
