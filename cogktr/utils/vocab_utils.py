class Vocabulary():
    def __init__(self):
        self.label_set = set()
        self.label2id_dict = {}
        self.id2label_dict = {}

    def __len__(self):
        return len(self.label_set)

    def add(self, label):
        self.label_set.add(label)

    def add_sequence(self, labels):
        for label in labels:
            self.add(label)

    def create(self):
        label_list = list(self.label_set)
        label_list.sort()
        for label in label_list:
            if label not in self.label2id_dict.keys():
                current_id = len(self.label2id_dict)
                self.label2id_dict[label] = current_id
                self.id2label_dict[current_id] = label

    def label2id(self, word):
        return self.label2id_dict[word]

    def id2label(self, id):
        return self.id2label_dict[id]

    def get_label2id_dict(self):
        return self.label2id_dict

    def get_id2label_dict(self):
        return self.id2label_dict


if __name__ == "__main__":
    vocab = Vocabulary()
    vocab.add("C")
    vocab.add("A")
    vocab.add("B")
    vocab.add("A")
    vocab.add_sequence(["C", "B", "D"])
    vocab.create()
    print(vocab.label2id("A"))
    print(vocab.id2label(0))
    print(vocab.get_label2id_dict())
    print(vocab.get_id2label_dict())
    print(len(vocab))
    print("end")
