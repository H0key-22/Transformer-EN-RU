from spacy.lang.ru import Russian
text = "Не ветер, а какой-то ураган!"
nlp = Russian()
doc = nlp(text)
print(doc)
print([token.text for token in doc])
# ['Не', 'ветер', ',', 'а', 'какой', '-', 'то', 'ураган', '!']
# Notice that word "какой-то" is split into three tokens.