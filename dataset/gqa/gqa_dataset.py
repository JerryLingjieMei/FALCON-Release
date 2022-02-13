import os
from collections import defaultdict

import torch
from torchvision.transforms import functional as TF

from dataset.dataset import Dataset, BuilderDataset
from dataset.utils import FixedResizeTransform, WordVocab, ProgramVocab
from utils import load, join, read_image, file_cached, IdentityDict


class GqaDataset(Dataset):
    _scenegraph_file = "sceneGraphs.pkl"
    _pretrain_file = 'pretrain_questions.json'
    _refexp_file = "refexps.json"
    _fewshot_file = "fewshot_questions.json"
    _chunk_size = 16
    _objects_file = "objects/gqa_objects_{}.h5"
    _objects_info_file = "objects/gqa_objects_info.json"
    image_size = (224, 224)
    concept_knowledge = "knowledge/gqa_concept.json"
    synonym_knowledge = "knowledge/gqa_synonym.json"
    isinstanceof_knowledge = "knowledge/gqa_isinstanceof.json"

    @property
    def transform_fn(self):
        return FixedResizeTransform

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.kinds_ = list(sorted(load(self.isinstanceof_knowledge).keys())) + ["other"]
        self.concepts_ = load(self.concept_knowledge)
        self.concepts = list(range(len(self.concepts_)))
        self.entry2idx_ = {c: n for n, c in enumerate(self.concepts_)}
        self.concept2kinds = self._build_concept2kinds()
        self.named_entries_ = self.concepts_
        self.entry2idx_ = {e: i for i, e in enumerate(self.named_entries_)}
        self.entry2kinds = self.concept2kinds

        self.synonym2itself_ = self._build_synonyms()
        self.obj2concepts, self.image_filenames, image_split_specs = self._build_images()
        self.image_split_specs = torch.tensor(image_split_specs)
        if not self.is_builder:
            self.object_chunks, self.objects_info = self._build_object_features()
        else:
            self.object_chunks, self.objects_info = None, None
        self.itself2samekinds = self._build_samekinds()

        self.concept_split_specs = self._build_concept_split_specs()
        self.concept2splits = self.concept_split_specs
        self.names = self.concepts_
        self.word_vocab = self._build_word_vocab()

    def _build_concept2kinds(self):
        concept2kinds = defaultdict(lambda: len(self.kinds_) - 1)
        isinstanceof_knowledge = load(self.isinstanceof_knowledge)
        for group, concepts in isinstanceof_knowledge.items():
            for c in concepts:
                concept2kinds[c] = self.kinds_.index(group)
        return torch.tensor([concept2kinds[c] for c in self.concepts_])

    @file_cached('images')
    def _build_images(self):
        scenegraphs = load(join(self.root, self._scenegraph_file))
        obj2concepts = self._build_obj2concepts(scenegraphs)
        image_filenames = [os.path.split(sg['image_filename'])[-1] for sg in scenegraphs.values()]
        image_split_specs = [self.split2spec[sg['split']] for sg in scenegraphs.values()]
        return obj2concepts, image_filenames, image_split_specs

    def _build_obj2concepts(self, scenegraphs):
        obj2concepts = []
        for sg in scenegraphs.values():
            this_object = []
            for o in sg['objects'].values():
                concepts = set()
                for c in o['concepts_contained']:
                    itself = self.synonym2itself_[c]
                    if itself in self.entry2idx_:
                        concepts.add(self.entry2idx_[itself])
                this_object.append(list(concepts))
            obj2concepts.append(this_object)
        return obj2concepts

    def _build_synonyms(self):
        synonym2itself = IdentityDict()
        for itself, synonyms in load(self.synonym_knowledge).items():
            for s in synonyms:
                synonym2itself[s] = itself
        return synonym2itself

    def _build_object_features(self):
        info = load(join(self.root, self._objects_info_file))
        object_chunks, objects_info = [], []
        for i in range(self._chunk_size):
            object_chunks.append(load(join(self.root, self._objects_file.format(i))))
        for image_filename in self.image_filenames:
            image_id = image_filename.split('.')[0]
            objects_info.append(info[image_id])
        return object_chunks, objects_info

    def _build_samekinds(self):
        isinstanceof_knowledge = load(self.isinstanceof_knowledge)
        itself2samekinds = {}
        for group, concepts in isinstanceof_knowledge.items():
            for c in concepts:
                itself2samekinds[self.entry2idx_[c]] = [self.entry2idx_[_] for _ in concepts if _ != c]
        return itself2samekinds

    def _build_concept_split_specs(self):
        concept_split_specs = [0] * len(self.concepts)
        for c in self.concepts:
            if c % 4 == 1:
                concept_split_specs[c] = 1
            elif c % 4 == 3:
                concept_split_specs[c] = 2
            if self.concept2kinds[c] == len(self.kinds_) - 1:
                concept_split_specs[c] = 0
        return torch.tensor(concept_split_specs)

    def get_image(self, image_index):
        return TF.to_tensor(read_image(join(self.root, "images", self.image_filenames[image_index])))

    def get_mask(self, image_index):
        info = self.objects_info[image_index]
        mask = torch.zeros(info['objectsNum'], info['height'], info['width'], dtype=torch.bool)
        bboxes = self.object_chunks[info['file']]['bboxes'][info['idx'], :info['objectsNum']].astype(int)
        for i in range(info['objectsNum']):
            x, y, xx, yy = bboxes[i]
            mask[i, y:yy, x:xx] = 1
        return mask

    def get_object_features(self, image_index):
        info = self.objects_info[image_index]
        return torch.Tensor(self.object_chunks[info["file"]]["features"][info["idx"], :info["objectsNum"]])

    def get_stacked_scenes(self, image_index):
        assert not torch.is_tensor(image_index)
        if self.has_mask:
            return {"pretrained": self.get_object_features(image_index)}
        else:
            return {'image': self.transform(self.get_image(image_index))}

    def exist_question(self, candidate):
        return f"Is there a {self.names[candidate]} object?"

    def exist_statement(self, candidate):
        return f"There is a {self.names[candidate]} object."

    def filter_question(self, candidate, filters):
        return f"Is the {' '.join(self.names[f] for f in filters)} object a {self.names[candidate]} object? "

    def filter_statement(self, candidate, filters):
        return f"The {' '.join(self.names[f] for f in filters)} object is a {self.names[candidate]} object."

    def metaconcept_text(self, supports, relations, concept_index):
        other_names = list(self.names[s] for e, s in zip(relations, supports) if e != 0)
        return f"{', '.join(other_names + [self.names[concept_index]])} describes the same property of an " \
               f"object.".capitalize()

    @file_cached("word_tokens")
    def _build_word_tokens(self):
        vocabulary = WordVocab()
        vocabulary.update(self.names)
        vocabulary.update(["hypernym", "samekind", "hyponym", "of"])
        vocabulary.update([self.exist_statement(0), self.exist_question(0), self.filter_question(0, [0, 0]),
            self.filter_statement(0, [0, 0]), self.metaconcept_text([0], [0], 0)])
        vocabulary.update(['yes', 'no'])
        return sorted(list(vocabulary.words))


class GqaBuilderDataset(BuilderDataset, GqaDataset):
    num_inputs = {'scene': 0, 'filter': 1, 'exist': 1}
    num_value_inputs = {'scene': 0, 'filter': 1, 'exist': 0}

    assert num_inputs.keys() == num_value_inputs.keys()

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.program_vocab = self._build_program_vocab()

    @file_cached('program_tokens')
    def _build_program_tokens(self):
        vocabulary = ProgramVocab()
        vocabulary.update(self.named_entries_)
        vocabulary.update(self.num_inputs.keys())
        vocabulary.update(['hypernym', 'hyponym', 'samekind'])
        return sorted(list(vocabulary.words))
