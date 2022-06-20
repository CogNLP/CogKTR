from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP("/data/mentianyi/code/CogKTR/test/stanford-corenlp-4.4.0")
sentence = "it still performs nicely with the exception of an occasional sound from the motor ."
print('Tokenize:', nlp.word_tokenize(sentence))
print('Part of Speech:', nlp.pos_tag(sentence))
print('Named Entities:', nlp.ner(sentence))
print('Constituency Parsing:', nlp.parse(sentence))
print('Dependency Parsing:', nlp.dependency_parse(sentence))
print(123)
