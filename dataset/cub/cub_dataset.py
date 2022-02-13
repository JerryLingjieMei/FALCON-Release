from collections import defaultdict
from copy import copy

import torch
import torchvision.transforms.functional as TF

from dataset.dataset import Dataset, BuilderDataset
from dataset.utils import FixedCropTransform, ProgramVocab
from dataset.utils import sample_with_ratio, WordVocab
from utils import join, load, read_image, file_cached, mask2bbox


class CubDataset(Dataset):

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.kinds_ = self._build_kinds()
        self.classes_ = self._read_concept_names()
        self.classes = list(range(len(self.classes_)))
        self.concepts_, concept2kinds = self._build_concepts()
        self.concept2kinds = torch.tensor(concept2kinds)
        self.concepts = list(range(len(self.concepts_)))
        self.attributes_ = self._read_names(self._attributes_file)
        self.attribute2kinds = torch.tensor(self._build_attribute2kinds())
        self.attributes = list(range(len(self.concepts), len(self.concepts) + len(self.attributes_)))
        self.named_entries_ = self.concepts_ + self.attributes_
        self.entry2idx_ = {e: i for i, e in enumerate(self.named_entries_)}
        self.names = self._build_names()
        self.entry2kinds = torch.cat([self.concept2kinds, self.attribute2kinds])

        self.image2classes = torch.tensor(self._read_image2classes())
        self.class2images = self._build_class2images()
        self.image_filenames = self._read_image_filenames()
        self.bbox = torch.tensor(self._read_bboxes())

        self.hierarchy, self.children = self._load_hierarchy_knowledge()
        self.hypernym_pairs, self.hypo2hyper, self.hyper2hypo = self._load_hypernym_knowledge()
        self.obj2concepts = [[self.hypo2hyper[c] + [c]] for c in self.image2classes.tolist()]
        self.holonym2meronym = self._load_meronym_knowledge()
        self.common_ancesters = self._build_common_ancesters()
        self.samekind_pairs, self.itself2samekinds = self._build_samekind_pairs()

        self.image_split_specs = torch.tensor(self._build_image_split_specs())
        self.concept_split_specs = torch.tensor(self._build_concept_split_specs())
        # noinspection PyUnresolvedReferences
        self.concept2splits = (self.concept_split_specs - 1).float().div(self._hierarchy_n).clamp(
            max=2.).floor() + 1
        self.word_vocab = self._build_word_vocab()

    # Setup utilities

    _image_filename_file = 'images.txt'

    _class_file = 'classes.txt'
    _image2class_file = 'image_class_labels.txt'

    _attributes_file = 'attributes/attributes.txt'
    _image2attribute_file = 'attributes/image_attribute_labels.txt'
    _class2attribute_file = 'attributes/class_attribute_labels_continuous.txt'

    _original_split_file = 'train_test_split.txt'
    _bbox_file = 'bounding_boxes.txt'

    _hierarchy_knowledge = 'knowledge/cub_hierarchy.json'
    _meronym_knowledge = 'knowledge/cub_meronym.json'
    _isinstanceof_knowledge = 'knowledge/cub_isinstanceof.json'
    _hypernym_knowledge = 'knowledge/cub_hypernym.json'

    _hierarchy_n = 5
    _attribute_n = 28
    _confidence = 4
    bird_kinds_ = ["species", "genera", "families", "orders", "classes"]

    def _read_names(self, path):
        with open(join(self.root, path), 'r') as f:
            read_lines = f.readlines()
        return [line.lstrip('0123456789. ').rstrip('\n') for line in read_lines]

    @file_cached(_class_file)
    def _read_concept_names(self):
        return self._read_names(self._class_file)

    @file_cached(_attributes_file)
    def _read_attribute_names(self):
        return self._read_names(self._attributes_file)

    def _read_numbers(self, path, select_columns=None):
        with open(join(self.root, path), 'r') as f:
            lines = f.readlines()
        # Some lines are corrupted, they have extra columns
        return torch.stack(
            [torch.Tensor([float(num) for num in line.rstrip('\n').split(' ') if num != ''])[select_columns] for
                line in lines], dim=0)

    @file_cached(_image2class_file)
    def _read_image2classes(self):
        # class index start with 1
        return (self._read_numbers(self._image2class_file, 1).long() - 1).tolist()

    @file_cached(_bbox_file)
    def _read_bboxes(self):
        return self._read_numbers(self._bbox_file, [1, 2, 3, 4]).long().tolist()

    @file_cached(_image_filename_file)
    def _read_image_filenames(self):
        with open(join(self.root, self._image_filename_file), 'r') as f:
            read_lines = f.readlines()
        return [line.split(' ')[-1].rstrip('\n') for line in read_lines]

    @file_cached("kinds")
    def _build_kinds(self):
        isinstanceof_knowledge = load(self._isinstanceof_knowledge)
        isinstanceof_knowledge.pop("species")
        kinds = copy(self.bird_kinds_)
        for k in isinstanceof_knowledge:
            kinds.append(k)
        return kinds

    @file_cached("concepts")
    def _build_concepts(self):
        hierarchy = load(self._hierarchy_knowledge)

        def _load_concepts_from_dict(x):
            results = []
            for k, n in x.items():
                piece = _load_concepts_from_dict(n)
                if len(piece) >= len(results):
                    for _ in range(len(piece) - len(results) + 1):
                        results.append([])
                for i, concepts in enumerate(piece):
                    results[i].extend(concepts)
                results[len(piece)].append(k)
            return results

        concepts_by_depth = [sorted(_) for _ in _load_concepts_from_dict(hierarchy)]
        species, genera, families, orders, classes = concepts_by_depth
        concept2kinds, concepts = [], []
        for i, cs in enumerate([self.classes_, genera, families, orders, classes]):
            concepts.extend(cs)
            concept2kinds.extend([i] * len(cs))
        return concepts, concept2kinds

    @file_cached("attribute2kinds")
    def _build_attribute2kinds(self):
        attribute2kinds = [0] * len(self.attributes_)
        isinstanceof_knowledge = load(self._isinstanceof_knowledge)
        isinstanceof_knowledge.pop("species")
        for k, vs in isinstanceof_knowledge.items():
            for v in vs:
                attribute2kinds[self.attributes_.index(v)] = self.kinds_.index(k)
        return attribute2kinds

    def _build_class2images(self):
        class2images = defaultdict(list)
        for i, c in enumerate(self.image2classes):
            class2images[c.item()].append(i)
        return class2images

    def _load_hypernym_knowledge(self):
        hypernym_knowledge = load(self._hypernym_knowledge)
        hypernym_pairs = []
        hypo2hyper, hyper2hypo = defaultdict(list), defaultdict(list)
        for hyper, hypos in hypernym_knowledge.items():
            for hypo in hypos:
                hypernym_pairs.append((self.entry2idx_[hyper], self.entry2idx_[hypo]))
                hypo2hyper[self.entry2idx_[hypo]].append(self.entry2idx_[hyper])
                hyper2hypo[self.entry2idx_[hyper]].append(self.entry2idx_[hypo])
        return hypernym_pairs, hypo2hyper, hyper2hypo

    def _load_hierarchy_knowledge(self):
        children = dict()

        def _load_hierarchy_sub(d):
            res = {}
            for k, v in d.items():
                key = self.entry2idx_[k]
                value = _load_hierarchy_sub(v)
                res[key] = value
                children[key] = value
            return res

        hierarchy_knowledge = load(self._hierarchy_knowledge)
        return _load_hierarchy_sub(hierarchy_knowledge), [children[_] for _ in self.concepts]

    @file_cached("meronym")
    def _load_meronym_knowledge(self):
        meronym_knowledge = load(self._meronym_knowledge)
        holonym2meronym = {}
        for cls, spec in meronym_knowledge.items():
            specs = [self.entry2idx_[_] for _ in spec["true"]]
            holonym2meronym[self.entry2idx_[cls]] = specs
        for c in self.concepts[len(self.classes):]:
            classes = [_ for _ in self.hyper2hypo[c]]
            meronyms = set(holonym2meronym[classes[0]])
            for cls in classes:
                meronyms.intersection_update(holonym2meronym[cls])
            holonym2meronym[c] = list(sorted(meronyms))
        holonym2meronym = [holonym2meronym[_] for _ in self.concepts]
        return holonym2meronym

    def _build_common_ancesters(self):
        common_ancesters = [[] for _ in self.concepts]
        for i in self.concepts:
            for j in self.concepts:
                ancesters = [_ for _ in self.hypo2hyper[i] + [i] if _ in self.hypo2hyper[j] + [j]]
                common_ancesters[i].append(max(ancesters, key=lambda x: len(self.hypo2hyper[x])))
        return torch.tensor(common_ancesters)

    def _build_samekind_pairs(self):
        samekind_pairs = []
        itself2samekinds = {}
        for itself, hypers in self.hypo2hyper.items():
            if len(hypers) == 0:
                itself2samekinds[itself] = []
                continue
            hyper = min(hypers)
            for samekind in self.children[hyper].keys():
                if samekind != itself:
                    samekind_pairs.append((samekind, itself))
            itself2samekinds[itself] = list(_ for _ in self.children[hyper].keys() if _ != itself)
        return samekind_pairs, itself2samekinds

    @file_cached("image_split")
    def _build_image_split_specs(self):
        split_specs = []
        for c, image_indices in self.class2images.items():
            split_specs.append(sample_with_ratio(len(image_indices), self.split_ratio, 10))
        return torch.cat(split_specs).tolist()

    split_remainder = {0: [[0, 2], [1], [3]], 1: [[0], [2], [1, 3]]}

    @file_cached("concept_split")
    def _build_concept_split_specs(self):
        train_remainder, val_remainder, test_remainder = self.split_remainder[
            self.split_seed] if self.split_seed in self.split_remainder else self.split_remainder[0]
        train_classes = set(_ for _ in self.classes if _ % 4 in train_remainder)
        val_classes = set(_ for _ in self.classes if _ % 4 in val_remainder)
        test_classes = set(_ for _ in self.classes if _ % 4 in test_remainder)
        train_concepts, val_concepts, test_concepts = set(train_classes), set(val_classes), set(test_classes)
        for train_class in train_classes:
            train_concepts.update(self.hypo2hyper[train_class])
        for val_class in val_classes:
            val_concepts.update(self.hypo2hyper[val_class])
        val_concepts.difference_update(train_concepts)
        for test_class in test_classes:
            test_concepts.update(self.hypo2hyper[test_class])
        test_concepts.difference_update(train_concepts)
        test_concepts.difference_update(val_concepts)
        train_test_concepts = set()
        for c in sorted(test_concepts, reverse=True):
            if min(self.hypo2hyper[c]) in train_test_concepts or min(self.hypo2hyper[c]) in train_concepts:
                train_test_concepts.add(c)
        val_test_concepts = test_concepts.difference(train_test_concepts)
        split_spec = [-100] * len(self.concepts)

        def fill_spec_tree_(split_spec, concepts, base):
            concepts = sorted(concepts, reverse=True)
            for c in concepts:
                split_spec[c] = base
            for c in concepts:
                for d in self.hyper2hypo[c]:
                    split_spec[d] = split_spec[c] + 1

        fill_spec_tree_(split_spec, train_concepts, -4)
        fill_spec_tree_(split_spec, val_concepts, 1)
        fill_spec_tree_(split_spec, train_test_concepts, 6)
        fill_spec_tree_(split_spec, val_test_concepts, 11)
        return split_spec

    def _build_names(self):
        names = [self.named_entries_[_].replace('_', ' ').lower() for _ in self.concepts]
        for a in self.attributes:
            attribute, adj = self.named_entries_[a].split('::')
            part = ' '.join(attribute.split('_')[1:-1])
            if part in ['', 'primary']:
                part = 'body'
            adj_plain = adj.replace('_', ' ')
            if part in adj_plain:
                if '-' in adj_plain:
                    adj_plain = adj_plain.split('-')[0]
                else:
                    adj_plain = adj_plain.split(' ')[0]
            if '(' in adj_plain or 'same' in adj_plain or 'than' in adj_plain:
                attr = f'{part} which are {adj_plain}'
            else:
                attr = f'{adj_plain} {part}'
            if not part.endswith('s'):
                attr = f'a {attr}'
            names.append(attr)
        return names

    @property
    def transform_fn(self):
        return FixedCropTransform

    # Computed
    def get_image(self, image_index):
        return TF.to_tensor(read_image(join(self.root, "images", self.image_filenames[image_index])))

    # Interfaces
    def get_mask(self, image_index):
        return mask2bbox(TF.to_tensor(read_image(
            join(self.root, "segmentations", self.image_filenames[image_index][:-len(".jpg")] + ".png")))[:1])

    def get_stacked_scenes(self, image_index):
        assert not torch.is_tensor(image_index)
        img = self.get_image(image_index)
        if not self.has_mask:
            return {"image": self.transform(img)}
        else:
            mask = self.get_mask(image_index)
            mask, img = self.transform(mask, img)
            return {"image": img, "mask": mask}

    def exist_question(self, candidate):
        return f"Is there a {self.names[candidate]}?"

    def exist_statement(self, candidate):
        return f"There is a {self.names[candidate]}."

    def metaconcept_text(self, supports, relations, concept_index):
        other_names = list(self.names[s] for e, s in zip(relations, supports) if e != 0)
        return f"{', '.join(other_names + [self.names[concept_index]])} are a kind of " \
               f"{self.names[supports[0]] if len(supports) > 0 else 'bird'}.".capitalize()

    @file_cached("word_tokens")
    def _build_word_tokens(self):
        vocabulary = WordVocab()
        vocabulary.update(self.names)
        vocabulary.update(["hypernym", "samekind", "hyponym", "of"])
        vocabulary.update([self.exist_statement(0), self.exist_question(0), self.metaconcept_text([0], [0], 0)])
        vocabulary.update(['yes', 'no'])
        return sorted(list(vocabulary.words))


class CubBuilderDataset(BuilderDataset, CubDataset):
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
