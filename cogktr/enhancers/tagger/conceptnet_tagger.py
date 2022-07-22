from cogktr.enhancers.tagger import BaseTagger
from cogktr.utils.constant.conceptnet_constants.constants import *
from tqdm import tqdm
import json
import os
import networkx as nx
import spacy
import nltk
from cogktr.utils.log_utils import logger
from spacy.matcher import Matcher
from cogktr.utils.vocab_utils import Vocabulary
from cogktr.utils.io_utils import save_pickle,load_pickle


class ConcetNetTagger(BaseTagger):
    def __init__(self, path):
        super(ConcetNetTagger, self).__init__()
        if not os.path.exists(path):
            raise ValueError("Path {} does not exits!".format(path))
        self.pattern_path = os.path.join(path, "matcher_patterns.json")
        self.vocab_path = os.path.join(path, "concept.txt")
        self.conceptnet_path = os.path.join(path,"conceptnet-assertions-5.6.0.csv")
        self.concetnet_en_path = os.path.join(path,"conceptnet.en.csv")
        self.pruned_graph_path = os.path.join(path,"conceptnet.en.pruned.graph")
        self.unpruned_graph_path = os.path.join(path,"conceptnet.en.unpruned.graph")
        if not os.path.exists(self.conceptnet_path):
            raise FileNotFoundError("Original conceptnet file {} not found!".format(self.conceptnet_path))

        self.id2relation = MERGED_RELATIONS
        self.relation2id = {r: i for i, r in enumerate(self.id2relation)}

        if not os.path.exists(self.concetnet_en_path) or not os.path.exists(self.vocab_path):
            extract_english(
                conceptnet_path=self.conceptnet_path,
                output_csv_path=self.concetnet_en_path,
                output_vocab_path=self.vocab_path,
            )

        with open(self.vocab_path, "r", encoding="utf8") as fin:
            self.id2concept = [w.strip() for w in fin]
        self.concept2id = {w: i for i, w in enumerate(self.id2concept)}
        self.vocab = {
            "id2relation": self.id2relation,
            "relation2id": self.relation2id,
            "id2concept": self.id2concept,
            "concept2id": self.concept2id,
        }

        if not os.path.exists(self.pattern_path):
            create_matcher_patterns(
                cpnet_vocab=self.vocab,
                output_path=self.pattern_path,
            )

        self.nlp = spacy.load('en_core_web_sm', disable=['tokenizer', 'ner', 'parser', 'textcat'])
        try:
            self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
        except:
            self.nlp.add_pipe('sentencizer')

        self.matcher = load_matcher(self.nlp, self.pattern_path)


    def tag(self, sentence):
        concepts,docs = ground_mentioned_concepts(self.nlp, self.matcher, sentence)
        if len(concepts) == 0:
            concepts = hard_ground(self.nlp, sentence, self.vocab["id2concept"])
        concepts = prune(concepts, self.vocab["id2concept"])
        tag_results = [{
            "mention":sample["mention"],
            "start":sample["start"],
            "end":sample["end"],
        } for sample in concepts]
        result = {
            "words":[doc.text for doc in docs],
            "knowledge":tag_results,
        }
        return result


def load_matcher(nlp, pattern_path):
    logger.info("Creating matcher...")

    with open(pattern_path, "r", encoding="utf8") as fin:
        all_patterns = json.load(fin)
    matcher = Matcher(nlp.vocab)
    for concept, pattern in tqdm(all_patterns.items()):
        # matcher.add(concept, None, pattern) # old spacy versions less than v3.0
        matcher.add(concept, [pattern])

    return matcher

def prune(concepts, vocab):
    prune_concepts = []
    for result in concepts:
        prune_concepts_per_sample = []
        for concept in result["concepts"]:
            if concept[-2:] == 'er' and concept[:-2] in concepts:
                continue
            if concept[-1:] == 'e' and concept[:1] in concepts:
                continue
            stop = False
            for t in concept.split("_"):
                if t in NLTK_STOPWORDS:
                    stop = True
                    break
            if not stop and concept in vocab:
                prune_concepts_per_sample.append(concept)
        if len(prune_concepts_per_sample) != 0:
            prune_concepts.append(result)
    return prune_concepts


def extract_english(conceptnet_path, output_csv_path, output_vocab_path):
    """
    Reads original conceptnet csv file and extracts all English relations (head and tail are both English entities) into
    a new file, with the following format for each line: <relation> <head> <tail> <weight>.
    :return:
    """
    logger.info('extracting English concepts and relations from ConceptNet...')
    relation_mapping = load_merge_relation()
    num_lines = sum(1 for line in open(conceptnet_path, 'r', encoding='utf-8'))
    cpnet_vocab = []
    concepts_seen = set()
    with open(conceptnet_path, 'r', encoding="utf8") as fin, \
            open(output_csv_path, 'w', encoding="utf8") as fout:
        for line in tqdm(fin, total=num_lines):
            toks = line.strip().split('\t')
            if toks[2].startswith('/c/en/') and toks[3].startswith('/c/en/'):
                """
                Some preprocessing:
                    - Remove part-of-speech encoding.
                    - Split("/")[-1] to trim the "/c/en/" and just get the entity name, convert all to 
                    - Lowercase for uniformity.
                """
                rel = toks[1].split("/")[-1].lower()
                head = del_pos(toks[2]).split("/")[-1].lower()
                tail = del_pos(toks[3]).split("/")[-1].lower()

                if not head.replace("_", "").replace("-", "").isalpha():
                    continue
                if not tail.replace("_", "").replace("-", "").isalpha():
                    continue
                if rel not in relation_mapping:
                    continue

                rel = relation_mapping[rel]
                if rel.startswith("*"):
                    head, tail, rel = tail, head, rel[1:]

                data = json.loads(toks[4])

                fout.write('\t'.join([rel, head, tail, str(data["weight"])]) + '\n')

                for w in [head, tail]:
                    if w not in concepts_seen:
                        concepts_seen.add(w)
                        cpnet_vocab.append(w)

    with open(output_vocab_path, 'w') as fout:
        for word in cpnet_vocab:
            fout.write(word + '\n')

    logger.info(f'extracted ConceptNet csv file saved to {output_csv_path}')
    logger.info(f'extracted concept vocabulary saved to {output_vocab_path}')


def construct_graph(cpnet_csv_path, cpnet_vocab, output_path, prune=True):
    logger.info("Generating ConceptNet graph files with prune={}".format(str(prune)))

    # id2concept = cpnet_vocab["id2conceptnet"]
    id2relation = cpnet_vocab["id2relation"]
    concept2id = cpnet_vocab["concept2id"]
    relation2id = cpnet_vocab["relation2id"]

    graph = nx.MultiDiGraph()
    nrow = sum(1 for _ in open(cpnet_csv_path, 'r', encoding='utf-8'))
    with open(cpnet_csv_path, "r", encoding="utf8") as fin:

        def not_save(cpt):
            if cpt in BLACKLIST:
                return True
            return False

        attrs = set()

        for line in tqdm(fin, total=nrow):
            ls = line.strip().split('\t')
            rel = relation2id[ls[0]]
            subj = concept2id[ls[1]]
            obj = concept2id[ls[2]]
            weight = float(ls[3])
            if prune and (not_save(ls[1]) or not_save(ls[2]) or id2relation[rel] == "hascontext"):
                continue
            if subj == obj:  # delete loops
                continue
            if (subj, obj, rel) not in attrs:
                graph.add_edge(subj, obj, rel=rel, weight=weight)
                attrs.add((subj, obj, rel))
                graph.add_edge(obj, subj, rel=rel + len(relation2id), weight=weight)
                attrs.add((obj, subj, rel + len(relation2id)))

    nx.write_gpickle(graph, output_path)
    logger.info(f"graph file saved to {output_path}")


def create_matcher_patterns(cpnet_vocab, output_path, debug=False):
    id2concept = cpnet_vocab["concept2id"]
    cpnet_vocab = [c.replace("_", " ") for c in id2concept]

    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])
    docs = nlp.pipe(cpnet_vocab)
    all_patterns = {}

    if debug:
        f = open("filtered_concept.txt", "w")

    for doc in tqdm(docs, total=len(cpnet_vocab)):

        pattern = create_pattern(nlp, doc, debug)
        if debug:
            if not pattern[0]:
                f.write(pattern[1] + '\n')

        if pattern is None:
            continue
        all_patterns["_".join(doc.text.split(" "))] = pattern

    logger.info("Created " + str(len(all_patterns)) + " patterns.")
    with open(output_path, "w", encoding="utf8") as fout:
        json.dump(all_patterns, fout)
    if debug:
        f.close()


def ground_mentioned_concepts(nlp, matcher, s):
    blacklist = {"-PRON-", "actually", "likely", "possibly", "want", "make", "my", "someone", "sometimes_people",
                 "sometimes", "would", "want_to", "one", "something", "sometimes", "everybody", "somebody", "could",
                 "could_be"}

    if isinstance(s, str):
        s = s.lower()
        doc = nlp(s)
    elif isinstance(s, list):
        from spacy.tokens import Doc
        s = [c.lower() for c in s]
        input_doc = Doc(nlp.vocab, words=s)
        doc = nlp(input_doc)
    else:
        raise ValueError("Only str of list of str is supported but got {}!".format(type(s)))

    matches = matcher(doc)

    mentioned_concepts = set()
    span_to_concepts = {}

    span_to_location = {}
    for match_id, start, end in matches:
        span = doc[start:end].text  # the matched span
        span_to_location[span] = (start,end)
        original_concept = nlp.vocab.strings[match_id]
        original_concept_set = set()
        original_concept_set.add(original_concept)
        if len(original_concept.split("_")) == 1:
            original_concept_set.update(lemmatize(nlp, nlp.vocab.strings[match_id]))

        if span not in span_to_concepts:
            span_to_concepts[span] = set()

        span_to_concepts[span].update(original_concept_set)

    exact_match2span = {}
    for span, concepts in span_to_concepts.items():
        concepts_sorted = list(concepts)
        concepts_sorted.sort(key=len)
        shortest = concepts_sorted[0:3]

        for c in shortest:
            if c in blacklist:
                continue

            # a set with one string like: set("like_apples")
            lcs = lemmatize(nlp, c)
            intersect = lcs.intersection(shortest)
            if len(intersect) > 0:
                mentioned_concepts.add(list(intersect)[0])
                exact_match2span[list(intersect)[0]] = span
            else:
                mentioned_concepts.add(c)
                exact_match2span[c] = span

        # if a mention exactly matches with a concept

        exact_match = set([concept for concept in concepts_sorted if concept.replace("_", " ").lower() == span.lower()])

        assert len(exact_match) < 2
        mentioned_concepts.update(exact_match)
        exact_match2span[list(exact_match)[0]] = span

    results = []
    for span,concepts in span_to_concepts.items():
        mention = span
        start,end = span_to_location[span]
        concepts = [concept for concept in concepts if concept in mentioned_concepts]
        results.append({
            "mention":mention,
            "start":start,
            "end":end,
            "concepts":concepts,
        })

    return results,doc


def hard_ground(nlp, s, cpnet_vocab):
    if isinstance(s, str):
        s = s.lower()
        doc = nlp(s)
    elif isinstance(s, list):
        from spacy.tokens import Doc
        s = [c.lower() for c in s]
        input_doc = Doc(nlp.vocab, words=s)
        doc = nlp(input_doc)
    else:
        raise ValueError("Only str of list of str is supported but got {}!".format(type(s)))

    res = set()
    mention2location = {}
    concept2mention = {}
    for idx,t in enumerate(doc):
        if t.lemma_ in cpnet_vocab:
            res.add(t.lemma_)
            mention2location[t.text] = (idx,idx+1)
            concept2mention[t.lemma_] = t.text
    sent = "_".join([t.text for t in doc])
    if sent in cpnet_vocab:
        res.add(sent)
        mention2location[doc.text] = (0,len(doc))
        concept2mention[sent] = doc.text
    try:
        assert len(res) > 0
    except Exception:
        logger.info(f"for {sent}, concept not found in hard grounding.")
    return [{
        'mention':concept2mention[concept],
        "start":mention2location[concept2mention[concept]][0],
        "end":mention2location[concept2mention[concept]][1],
        "concepts":[concept],
    } for concept in list(res)]
    # return res


def load_merge_relation():
    relation_mapping = dict()
    for line in RELATION_GROUPS:
        ls = line.strip().split('/')
        rel = ls[0]
        for l in ls:
            if l.startswith("*"):
                relation_mapping[l[1:]] = "*" + rel
            else:
                relation_mapping[l] = rel
    return relation_mapping


def del_pos(s):
    """
    Deletes part-of-speech encoding from an entity string, if present.
    :param s: Entity string.
    :return: Entity string with part-of-speech encoding removed.
    """
    if s.endswith("/n") or s.endswith("/a") or s.endswith("/v") or s.endswith("/r"):
        s = s[:-2]
    return s


def load_cpnet_vocab(cpnet_vocab_path):
    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        cpnet_vocab = [l.strip() for l in fin]
    # cpnet_vocab = [c.replace("_", " ") for c in cpnet_vocab]
    return cpnet_vocab


def create_pattern(nlp, doc, debug=False):
    # Filtering concepts consisting of all stop words and longer than four words.
    if len(doc) >= 5 or doc[0].text in PATTERN_PRONOUN_LIST or doc[-1].text in PATTERN_PRONOUN_LIST or \
            all([(token.text in NLTK_STOPWORDS or token.lemma_ in NLTK_STOPWORDS or token.lemma_ in PATTERN_BACKLIST)
                 for token
                 in doc]):
        if debug:
            return False, doc.text
        return None  # ignore this concept as pattern

    pattern = []
    for token in doc:  # a doc is a concept
        pattern.append({"LEMMA": token.lemma_})
    if debug:
        return True, doc.text
    return pattern


def lemmatize(nlp, concept):
    doc = nlp(concept.replace("_", " "))
    lcs = set()
    lcs.add("_".join([token.lemma_ for token in doc]))  # all lemma
    return lcs


if __name__ == '__main__':
    conceptnet_tagger = ConcetNetTagger(path='/data/hongbang/CogKTR/datapath/knowledge_graph/conceptnet/',)
    # sentence = "When standing miles away from Mount Rushmore the mountains seem very close"
    # answer = "the mountains seem very close"
    sentence = "In following That Bird, Ernie and Bert search for Big Bird by plane."
    # sentence = "Bert likes reading in Sesame street library with elmo."
    tag_result = conceptnet_tagger.tag(sentence)
    # words = ["The","sun","is","responsible","for","puppies","learning","new","tricks","."]
    # words = ['all','of','these']
    # all_concepts = conceptnet_linker.link(sentence)
    # ans_concepts = conceptnet_linker.link(answer)
    # ac = set(all_concepts) - set(ans_concepts)
    # qc = set(ans_concepts)
    # print(qc)
    # print(ac)

