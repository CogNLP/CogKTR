from cogktr.enhancers.tagger import BaseTagger
from allennlp.predictors.predictor import Predictor
import collections
from cogktr.utils.constant.srl_constant.vocab import TAG_VOCAB
from cogktr.utils.vocab_utils import Vocabulary

class SrlTagger(BaseTagger):
    def __init__(self, tool):
        super().__init__()
        if tool not in ["allennlp"]:
            raise ValueError("{} in SrlTagger is not supported!".format(tool))

        self.tool = tool
        self.knowledge_type = "srltagger"
        if self.tool == "allennlp":
            self.srltagger = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")

    def tag(self, sentence):
        tag_dict = {}
        if self.tool == "allennlp":
            tag_dict = self._allennlp_tag(sentence)
        return tag_dict

    def _allennlp_tag(self, sentence):
        tag_dict = {}
        token_list = []
        label_dict = {}
        try:
            tag_result = self.srltagger.predict(sentence=sentence)
        except RuntimeError:
            print("Sentence Too Long! Cut from {} to {}!".format(len(sentence),int(len(sentence)/2)))
            tag_result = self.srltagger.predict(sentence=sentence[0:int(len(sentence)/2)])
        else:
            pass
        token_list = tag_result["words"]
        for item in tag_result["verbs"]:
            label_dict[item["verb"]] = item["tags"]

        tag_dict["words"] = token_list
        tag_dict["labels"] = label_dict

        return tag_dict


class TagTokenizer(object):
    def __init__(self):
        # self.tag_vocab = TAG_VOCAB
        self.tag_vocab = Vocabulary()
        self.tag_vocab.add_sequence(TAG_VOCAB)
        self.tag_vocab.create()

    def convert_tags_to_ids(self, tags):
        """Converts a sequence of tags into ids using the vocab."""
        ids = []
        for tag in tags:
            if tag not in TAG_VOCAB:
                tag = 'O'
            ids.append(self.tag_vocab.label2id(tag))

        return ids

    def convert_ids_to_tags(self, ids):
        """Converts a sequence of ids into tags using the vocab."""
        tags = []
        for i in ids:
            tags.append(self.tag_vocab.id2label(i))
        return tags

#
# class TagTokenizer(object):
#     def __init__(self):
#         self.tag_vocab = TAG_VOCAB
#         self.ids_to_tags = collections.OrderedDict(
#             [(ids, tag) for ids, tag in enumerate(TAG_VOCAB)])
#
#     def convert_tags_to_ids(self, tags):
#         """Converts a sequence of tags into ids using the vocab."""
#         ids = []
#         for tag in tags:
#             if tag not in TAG_VOCAB:
#                 tag = 'O'
#             ids.append(TAG_VOCAB.index(tag))
#
#         return ids
#
#     def convert_ids_to_tags(self, ids):
#         """Converts a sequence of ids into tags using the vocab."""
#         tags = []
#         for i in ids:
#             tags.append(self.ids_to_tags[i])
#         return tags



if __name__ == "__main__":
    tagger = SrlTagger(tool="allennlp")
    sentence = 'Among the best-known are U.S. Presidents William Howard Taft, Gerald Ford, George H. W. Bush, Bill Clinton and George W. Bush; royals Crown Princess Victoria Bernadotte, Prince Rostislav Romanov and Prince Akiiki Hosea Nyabongo; heads of state, including Italian prime minister Mario Monti, Turkish prime minister Tansu Çiller, Mexican president Ernesto Zedillo, German president Karl Carstens, and Philippines president José Paciano Laurel; U.S. Supreme Court Justices Sonia Sotomayor, Samuel Alito and Clarence Thomas; U.S. Secretaries of State John Kerry, Hillary Clinton, Cyrus Vance, and Dean Acheson; authors Sinclair Lewis, Stephen Vincent Benét, and Tom Wolfe; lexicographer Noah Webster; inventors Samuel F. B. Morse and Eli Whitney; patriot and "first spy" Nathan Hale; theologian Jonathan Edwards; actors, directors and producers Paul Newman, Henry Winkler, Vincent Price, Meryl Streep, Sigourney Weaver, Jodie Foster, Angela Bassett, Patricia Clarkson, Courtney Vance, Frances McDormand, Elia Kazan, George Roy Hill, Edward Norton, Lupita Nyong\'o, Allison Williams, Oliver Stone, Sam Waterston, and Michael Cimino; "Father of American football" Walter Camp, James Franco, "The perfect oarsman" Rusty Wailes; baseball players Ron Darling, Bill Hutchinson, and Craig Breslow; basketball player Chris Dudley; football players Gary Fencik, and Calvin Hill; hockey players Chris Higgins and Mike Richter; figure skater Sarah Hughes; swimmer Don Schollander; skier Ryan Max Riley; runner Frank Shorter; composers Charles Ives, Douglas Moore and Cole Porter; Peace Corps founder Sargent Shriver; child psychologist Benjamin Spock; architects Eero Saarinen and Norman Foster; sculptor Richard Serra; film critic Gene Siskel; television commentators Dick Cavett and Anderson Cooper; New York Times journalist David Gonzalez; pundits William F. Buckley, Jr., and Fareed Zakaria; economists Irving Fischer, Mahbub ul Haq, and Paul Krugman; cyclotron inventor and Nobel laureate in Physics, Ernest Lawrence; Human Genome Project director Francis S. Collins; mathematician and chemist Josiah Willard Gibbs; and businesspeople, including Time Magazine co-founder Henry Luce, Morgan Stanley founder Harold Stanley, Boeing CEO James McNerney, FedEx founder Frederick W. Smith, Time Warner president Jeffrey Bewkes, Electronic Arts co-founder Bing Gordon, and investor/philanthropist Sir John Templeton; pioneer in electrical applications Austin Cornelius Dunham.'
    tag_dict = tagger.tag(sentence)
    print("end")
    from tqdm import tqdm
    # pbar = tqdm(range(4096))

    # import nltk
    # tokens = nltk.word_tokenize(sentence)
    # print(len(tokens))
    # for i in tqdm(range(len(tokens))):
    #     sub_sentence = " ".join(tokens[:i+1])
    #     tagger.tag(sub_sentence)



    # for i in pbar:
    #     test_sentence = "hello " * (i+1)
    #     pbar.set_description("num_tokens={}".format(i+1))
    #     tag_dict = tagger.tag(test_sentence)

