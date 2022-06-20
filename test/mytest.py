import cogie

# tokenize sentence into words
tokenize_toolkit = cogie.TokenizeToolkit(task='ws', language='english', corpus=None)
words = tokenize_toolkit.run('Ontario is the most populous province in Canada.')
# named entity recognition
ner_toolkit = cogie.NerToolkit(task='ner', language='english', corpus='trex')
ner_result = ner_toolkit.run(words)
# relation extraction
re_toolkit = cogie.ReToolkit(task='re', language='english', corpus='trex')
re_result = re_toolkit.run(words, ner_result)

token_toolkit = cogie.TokenizeToolkit(task='ws', language='english', corpus=None)
words = token_toolkit.run(
    'The true voodoo-worshipper attempts nothing of importance without certain sacrifices which are intended to propitiate his unclean gods.')
# frame identification
fn_toolkit = cogie.FnToolkit(task='fn', language='english', corpus=None)
fn_result = fn_toolkit.run(words)
# argument identification
argument_toolkit = cogie.ArgumentToolkit(task='fn', language='english', corpus='argument')
argument_result = argument_toolkit.run(words, fn_result)