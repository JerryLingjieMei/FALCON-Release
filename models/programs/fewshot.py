from models.programs.abstract_program import AbstractProgram


class Fewshot(AbstractProgram):

    def __init__(self, *arg):
        super().__init__(*arg)
        self.gt_embeddings, self.relations, self.train_features, self.parent = arg

    @property
    def is_attached(self):
        return len(self.gt_embeddings) > 0

    @property
    def is_detached(self):
        return len(self.gt_embeddings) == 0

    @property
    def is_fewshot(self):
        return len(self.train_features) > 0

    @property
    def is_zeroshot(self):
        return len(self.train_features) == 0

    @property
    def hypernym_embeddings(self):
        return self.gt_embeddings[self.relations == 0]

    @property
    def samekind_embeddings(self):
        return self.gt_embeddings[self.relations == 2]

    @property
    def support_embeddings(self):
        return self.gt_embeddings

    def __call__(self, executor):
        return executor.network(self)
