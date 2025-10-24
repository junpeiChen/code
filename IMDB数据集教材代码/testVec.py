from gensim.models import Word2Vec

model = Word2Vec.load("300features")
result = model.wv.doesnt_match("man woman child kitchen".split())
print(f"不匹配的词: {result}")
result2 = model.wv.doesnt_match("france england germany berlin".split())
print(f"差异最大的是: {result2}")
result3 = model.wv.most_similar("man")
print(f"相似的词是: {result3}")