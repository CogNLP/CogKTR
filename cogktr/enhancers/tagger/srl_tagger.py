from cogktr.enhancers.tagger import BaseTagger
from allennlp.predictors.predictor import Predictor


class SrlTagger(BaseTagger):
    def __init__(self, tool):
        super().__init__()
        if tool not in ["allennlp"]:
            raise ValueError("{} in SrlTagger is not supported!".format(tool))

        self.tool = tool
        self.knowledge_type = "srltagger"

        print("Loading SrlTagger...")
        if self.tool == "allennlp":
            self.srltagger = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
        print("Finish loading SrlTagger!")

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
            if isinstance(sentence, str):
                tag_result = self.srltagger.predict(sentence=sentence)
            elif isinstance(sentence, list):
                tag_result = self.srltagger.predict_tokenized(tokenized_sentence=sentence)
            else:
                raise ValueError("Sentence type {} is not supported!".format(type(sentence)))
        except RuntimeError:
            print("Sentence Too Long! Cut from {} to {}!".format(len(sentence), int(len(sentence) / 2)))
            if isinstance(sentence, str):
                tag_result = self.srltagger.predict(sentence=sentence[0:int(len(sentence) / 2)])
            elif isinstance(sentence, list):
                tag_result = self.srltagger.predict_tokenized(tokenized_sentence=sentence[0:int(len(sentence) / 2)])
        else:
            pass
        token_list = tag_result["words"]

        frames = []
        for result in tag_result["verbs"]:
            frame_name = result["verb"]
            spans = _bio_tag_to_spans(result["tags"])
            mentions = []
            for span in spans:
                t,(start,end) = span
                mentions.append({
                    "mention":tag_result["words"][start:end],
                    "start":start,
                    "end":end,
                    "type":t,
                })
            frames.append({
                "frame_name":frame_name,
                "mentions":mentions,
                "tags":result["tags"],
            })

        tag_dict["words"] = token_list
        tag_dict["knowledge"] = frames

        return tag_dict

def _bio_tag_to_spans(tags, ignore_labels=None):
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bio_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bio_tag, label = tag[:1], tag[2:]
        if bio_tag == 'b':
            spans.append((label, [idx, idx]))
        elif bio_tag == 'i' and prev_bio_tag in ('b', 'i') and label == spans[-1][0]:
            spans[-1][1][1] = idx
        elif bio_tag == 'o':  # o tag does not count
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bio_tag = bio_tag
    return [(span[0], (span[1][0], span[1][1] + 1)) for span in spans if span[0] not in ignore_labels]

if __name__ == "__main__":
    tagger = SrlTagger(tool="allennlp")
    tagger_dict_1 = tagger.tag(["Bert", "likes", "reading", "in the Sesame", "Street", "Library."])
    tagger_dict_2 = tagger.tag("Bert likes reading in the Sesame Street Library.")
    print(tagger_dict_1)
    print(tagger_dict_2)
    print("end")

    # sentence_3 = 'Among the best-known are U.S. Presidents William Howard Taft, Gerald Ford, George H. W. Bush, Bill Clinton and George W. Bush; royals Crown Princess Victoria Bernadotte, Prince Rostislav Romanov and Prince Akiiki Hosea Nyabongo; heads of state, including Italian prime minister Mario Monti, Turkish prime minister Tansu Çiller, Mexican president Ernesto Zedillo, German president Karl Carstens, and Philippines president José Paciano Laurel; U.S. Supreme Court Justices Sonia Sotomayor, Samuel Alito and Clarence Thomas; U.S. Secretaries of State John Kerry, Hillary Clinton, Cyrus Vance, and Dean Acheson; authors Sinclair Lewis, Stephen Vincent Benét, and Tom Wolfe; lexicographer Noah Webster; inventors Samuel F. B. Morse and Eli Whitney; patriot and "first spy" Nathan Hale; theologian Jonathan Edwards; actors, directors and producers Paul Newman, Henry Winkler, Vincent Price, Meryl Streep, Sigourney Weaver, Jodie Foster, Angela Bassett, Patricia Clarkson, Courtney Vance, Frances McDormand, Elia Kazan, George Roy Hill, Edward Norton, Lupita Nyong\'o, Allison Williams, Oliver Stone, Sam Waterston, and Michael Cimino; "Father of American football" Walter Camp, James Franco, "The perfect oarsman" Rusty Wailes; baseball players Ron Darling, Bill Hutchinson, and Craig Breslow; basketball player Chris Dudley; football players Gary Fencik, and Calvin Hill; hockey players Chris Higgins and Mike Richter; figure skater Sarah Hughes; swimmer Don Schollander; skier Ryan Max Riley; runner Frank Shorter; composers Charles Ives, Douglas Moore and Cole Porter; Peace Corps founder Sargent Shriver; child psychologist Benjamin Spock; architects Eero Saarinen and Norman Foster; sculptor Richard Serra; film critic Gene Siskel; television commentators Dick Cavett and Anderson Cooper; New York Times journalist David Gonzalez; pundits William F. Buckley, Jr., and Fareed Zakaria; economists Irving Fischer, Mahbub ul Haq, and Paul Krugman; cyclotron inventor and Nobel laureate in Physics, Ernest Lawrence; Human Genome Project director Francis S. Collins; mathematician and chemist Josiah Willard Gibbs; and businesspeople, including Time Magazine co-founder Henry Luce, Morgan Stanley founder Harold Stanley, Boeing CEO James McNerney, FedEx founder Frederick W. Smith, Time Warner president Jeffrey Bewkes, Electronic Arts co-founder Bing Gordon, and investor/philanthropist Sir John Templeton; pioneer in electrical applications Austin Cornelius Dunham.'
    # sentence_4 = sentence_1.split()
    # tagger_dict_3 = tagger.tag(sentence_3)
    # tagger_dict_4 = tagger.tag(sentence_4)
