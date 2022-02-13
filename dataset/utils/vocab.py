import nltk
from nltk import WordNetLemmatizer, RegexpTokenizer, pos_tag

nltk.data.path.append('/data/vision/billf/scratch/jerrymei/datasets/nltk_data')
for module in ["punkt", "averaged_perceptron_tagger"]:
    try:
        nltk.data.find(f'tokenizers/{module}')
    except LookupError:
        nltk.download(module)


class WordVocab:
    def __init__(self):
        self.words = {"<start>", "<end>", "<pad>", "<unk>"}
        self._lemmatize = WordNetLemmatizer().lemmatize
        self._tokenize = RegexpTokenizer(r'\w+').tokenize
        self.word2index = {}
        self.index2word = []

    def freeze(self):
        self.words = frozenset(sorted(self.words))
        self.word2index = {w: i for i, w in enumerate(sorted(self.words))}
        self.index2word = list(sorted(self.words))

    def update(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]
        for sentence in sentences:
            self.words.update([self._lemmatize(word.lower()) for word in self._tokenize(sentence)])

    @property
    def unk(self):
        return self.word2index["<unk>"]

    @property
    def start(self):
        return self.word2index["<start>"]

    @property
    def end(self):
        return self.word2index["<end>"]

    @property
    def pad(self):
        return self.word2index["<pad>"]

    @property
    def special_tokens(self):
        return [self.unk, self.start, self.end, self.pad]

    def __getitem__(self, word):
        assert len(self.word2index) > 0, "The vocab should be freezed."
        word = word.lower()
        if word == "unk":
            return self.unk
        if word == "pad":
            return self.pad
        word = self._lemmatize(word)
        assert word in self.word2index, f"Word \'{word}\' not found in vocabulary."
        return self.word2index[word]

    def __call__(self, sentence):
        return [self.start, *(self[word] for word in self._tokenize(sentence)), self.end]

    def is_noun(self, word):
        return pos_tag([word])[0][1] == "NN"

    def __len__(self):
        return len(self.words)


class ProgramVocab:
    def __init__(self):
        self.words = {"<start>", "<end>", "<pad>", "<unk>"}
        self.word2index = {}
        self.index2word = []

    def freeze(self):
        self.words = frozenset(sorted(self.words))
        self.word2index = {w: i for i, w in enumerate(sorted(self.words))}
        self.index2word = list(sorted(self.words))

    def update(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]
        self.words.update(sentences)

    @property
    def unk(self):
        return self.word2index["<unk>"]

    @property
    def start(self):
        return self.word2index["<start>"]

    @property
    def end(self):
        return self.word2index["<end>"]

    @property
    def pad(self):
        return self.word2index["<pad>"]

    @property
    def special_tokens(self):
        return [self.unk, self.start, self.end, self.pad]

    def __getitem__(self, word):
        return self.word2index[word]

    def __call__(self, sentence):
        return [self.start, *(self[word] for word in sentence), self.end]

    def __len__(self):
        return len(self.words)
