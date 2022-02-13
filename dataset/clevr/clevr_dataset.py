import logging

import numpy as np
import torch
from pycocotools import mask as mask_util
from torchvision.transforms import functional as TF
from tqdm import tqdm

from dataset.dataset import Dataset, BuilderDataset
from dataset.utils import FixedTransform, WordVocab, ProgramVocab
from utils import load, join, read_image, file_cached, nonzero, mask2bbox


class ClevrDataset(Dataset):
    _annotation_file = "scenes/CLEVR_train_scenes.json"
    _mask_file = "detectron/CLEVR_train_detect_objs.json"
    _pretrain_file = 'pretrain_questions.json'
    _refexp_file = "refexps.json"
    _fewshot_file = "fewshot_questions.json"
    image_size = (320, 480)
    colors_ = {"gray": [87, 87, 87], "red": [173, 35, 35], "blue": [42, 75, 215], "green": [29, 105, 20],
        "brown": [129, 74, 25], "purple": [129, 38, 192], "cyan": [41, 208, 208], "yellow": [255, 238, 51]}
    val_concepts_ = ["red", "green"]
    test_concepts_ = ["purple", "cyan", "cylinder"]
    shapes_ = ["cube", "sphere", "cylinder"]
    sizes_ = ["large", "small"]
    materials_ = ["rubber", "metal"]
    concept_maps_ = {"color": list(colors_.keys()), "size": sizes_, "shape": shapes_, "material": materials_}
    directions_ = ["behind", "front", "right", "left"]
    attributes_ = list(concept_maps_.keys())
    synonym_knowledge = "knowledge/clevr_synonym.json"

    @property
    def transform_fn(self):
        return FixedTransform

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.kinds_ = list(self.concept_maps_.keys()) + ["direction", "attribute"]
        self.concepts_ = self._build_all_concepts()
        self.concepts = list(range(len(self.concepts_)))
        self.concept2kinds = self._build_concept2kinds()
        self.directions = list(range(len(self.concepts), len(self.concepts) + len(self.directions_)))
        self.direction2kinds = torch.tensor([self.kinds_.index("direction")] * len(self.directions))
        self.attributes_ = list(self.concept_maps_.keys())
        self.attribute2kinds = torch.tensor([self.kinds_.index("attribute")] * len(self.attributes_))
        self.named_entries_ = self.concepts_ + self.directions_ + self.attributes_
        self.entry2idx_ = {e: i for i, e in enumerate(self.named_entries_)}
        self.names = self.named_entries_
        self.entry2kinds = torch.cat([self.concept2kinds, self.direction2kinds, self.attribute2kinds])

        self.images = load(join(self.root, self._annotation_file))["scenes"]
        self.image_filenames = [i["image_filename"] for i in self.images]
        self.objects, self.obj2concepts, self.obj2relations = self._build_obj_concepts()
        self.object_masks = self._load_object_masks()
        self.samekind_pairs, self.itself2samekinds = self._build_samekind_pairs()

        self.image_split_specs = torch.tensor(self._build_image_split_specs())
        self.concept_split_specs = self._build_concept_split_specs()
        self.concept2splits = self.concept_split_specs
        self.word_vocab = self._build_word_vocab()
        self.synonyms = load(self.synonym_knowledge)

    def _build_all_concepts(self):
        all_concepts = []
        for group, concepts in self.concept_maps_.items():
            all_concepts += concepts
        return all_concepts

    def _build_concept2kinds(self):
        concept2kinds = {}
        for group, concepts in self.concept_maps_.items():
            for c in concepts:
                concept2kinds[c] = self.kinds_.index(group)
        return torch.tensor([concept2kinds[c] for c in self.concepts_])

    def _build_obj_concepts(self):
        objects, obj2concepts, obj2relations = [], [], []
        for image_index, image in enumerate(self.images):
            object_in_image, relation_in_image = [], {}
            for object_index, obj in enumerate(image["objects"]):
                object_in_image.append([self.entry2idx_[obj[key]] for key in self.concept_maps_.keys()])
            for relation_name in self.directions_:
                relation_in_image[len(relation_in_image) + len(self.concepts)] = image["relationships"][
                    relation_name]
            obj2concepts.append(object_in_image)
            obj2relations.append(relation_in_image)
            objects.extend([(image_index, _) for _ in range(len(object_in_image))])
        return objects, obj2concepts, obj2relations

    def _load_object_masks(self):
        mask_file = load(join(self.root, self._mask_file))
        object_masks = [[] for _ in range(len(self.images))]
        for image_index, mask in zip(mask_file['image_idxs'], mask_file['object_masks']):
            object_masks[image_index].append(mask)
        return object_masks

    def _build_image_split_specs(self):
        split_specs = []
        for i, r in enumerate(self.split_ratio):
            split_specs.extend([i] * int(r * len(self.images)))
        while len(split_specs) < len(self.images):
            split_specs.append(split_specs[-1])
        return split_specs

    def _build_samekind_pairs(self):
        samekind_pairs, itself2samekinds = [], {}
        for itself in self.concepts:
            itself2samekinds[itself] = []
            for samekind in self.concepts:
                if samekind != itself and self.concept2kinds[itself] == self.concept2kinds[samekind]:
                    samekind_pairs.append((itself, samekind))
                    itself2samekinds[itself].append(samekind)
        return samekind_pairs, itself2samekinds

    def _build_concept_split_specs(self):
        concept_split_specs = [0] * len(self.concepts)
        for test_color in self.val_concepts_:
            concept_split_specs[self.entry2idx_[test_color]] = 1
        for test_color in self.test_concepts_:
            concept_split_specs[self.entry2idx_[test_color]] = 2
        for u in torch.unique(self.concept2kinds):
            indices = nonzero(self.concept2kinds == u)
            start, end = indices[0], indices[-1] + 1
            shift = (5 * self.split_seed) % (end - start) if self.split_seed < 5 else 0
            concept_split_specs[start:end] = concept_split_specs[start + shift:end] + concept_split_specs[
            start:start + shift]
        logger = logging.getLogger("falcon_logger")
        logger.warning(f"val concepts: {[c for c, s in zip(self.concepts_, concept_split_specs) if s == 1]}")
        logger.warning(f"test concepts: {[c for c, s in zip(self.concepts_, concept_split_specs) if s == 2]}")
        return torch.tensor(concept_split_specs)

    def get_stacked_scenes(self, image_index):
        assert not torch.is_tensor(image_index)
        img = self.get_image(image_index)
        if self.has_mask:
            mask = self.get_mask(image_index)
            return {"image": img, "mask": mask}
        else:
            return {"image": img}

    def get_image(self, image_index):
        return TF.to_tensor(read_image(join(self.root, "images", "train", self.image_filenames[image_index])))

    def get_mask(self, image_index):
        return mask2bbox(
            torch.BoolTensor(np.stack([mask_util.decode(m) for m in self.object_masks[image_index]])))

    def metaconcept_text(self, supports, relations, concept_index):
        other_names = list(self.names[s] for e, s in zip(relations, supports) if e != 0)
        return f"{', '.join(other_names + [self.names[concept_index]])} describes the same property of an " \
               f"object.".capitalize()

    @staticmethod
    def answer2target(answer):
        for v in ClevrDataset.concept_maps_.values():
            if answer in v:
                return torch.tensor(v.index(answer))
        return torch.tensor(answer).float()

    @file_cached("word_tokens")
    def _build_word_tokens(self):
        vocabulary = WordVocab()
        for raw_filename in [self._pretrain_file, self._refexp_file, self._fewshot_file]:
            d = load(join(self.augmented_root, raw_filename))
            key = "question" if "question" in raw_filename else "refexp"
            for e in tqdm(d[f"{key}s"]):
                vocabulary.update([e[key]])
        vocabulary.update([str(i) for i in range(11)])
        vocabulary.update(['yes', 'no'])
        vocabulary.update([self.metaconcept_text([0], [0], 0)])
        return sorted(list(vocabulary.words))


class ClevrBuilderDataset(ClevrDataset, BuilderDataset):
    num_inputs = {'scene': 0, 'filter_shape': 1, 'filter_size': 1, 'filter_material': 1, 'filter_color': 1,
        'exist': 1, 'unique': 1, 'relate': 1, 'same_shape': 1, 'same_size': 1, 'same_material': 1,
        'same_color': 1, 'query_shape': 1, 'query_size': 1, 'query_material': 1, 'query_color': 1,
        'equal_shape': 2, 'equal_size': 2, 'equal_material': 2, 'equal_color': 2, 'intersect': 2, 'union': 2,
        'count': 1, 'equal_integer': 2, 'greater_than': 2, 'less_than': 2}
    num_value_inputs = {'scene': 0, 'filter_shape': 1, 'filter_size': 1, 'filter_material': 1,
        'filter_color': 1, 'exist': 0, 'unique': 0, 'relate': 1, 'same_shape': 0, 'same_size': 0,
        'same_material': 0, 'same_color': 0, 'query_shape': 0, 'query_size': 0, 'query_material': 0,
        'query_color': 0, 'equal_shape': 0, 'equal_size': 0, 'equal_material': 0, 'equal_color': 0,
        'intersect': 0, 'union': 0, 'count': 0, 'equal_integer': 0, 'greater_than': 0, 'less_than': 0}

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
