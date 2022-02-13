import os

from ete3 import Tree, TreeStyle, TextFace, add_face_to_node, NodeStyle
from matplotlib import cm
from matplotlib.colors import to_hex

from snippets.snippet_utils import cfg2test_loader
from utils import mkdir, join, ArgumentParser

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def build_tree(dataset):
    spec_dict = {}
    for c, spec in zip(dataset.concepts, dataset.concept_split_specs.tolist()):
        spec_dict[c] = spec
    min_spec = min(spec_dict.values())
    max_spec = max(spec_dict.values())

    def _dict2newick(d):
        if len(d) > 0:
            return "({})".format(
                ','.join("{}{}".format(_dict2newick(value), dataset.named_entries_[key]) for key, value in d.items()))
        else:
            return ""

    tree = Tree(_dict2newick(dataset.hierarchy) + ";", 8)
    for n in tree.traverse():
        ns = NodeStyle()
        ns["size"] = 10
        ns["fgcolor"] = to_hex(cm.spring((spec_dict.get(n.name, 0) - min_spec) / (max_spec - min_spec)))
        n.set_style(ns)
    return tree


def layout(node):
    text_face = TextFace(node.name, tight_text=True)
    add_face_to_node(text_face, node, column=0, position="branch-right")


def plot_tree(t, ratio):
    ts = TreeStyle()
    ts.show_leaf_name = False
    ts.layout_fn = layout

    folder = "output/snippets/concept_hierarchy_plot"
    mkdir(folder)
    t.render(join(folder, "tree_{}.png".format("{:.03f}-{:.03f}".format(*ratio))), tree_style=ts)


def locate(name):
    path = []
    for n in tree.traverse():
        if n.name == name:
            path.append(n.name)
            while n.up:
                path.append(n.up.name)
                n = n.up
            break
    return path[:-1]


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    args = arg_parser.parse_args()
    test_loader = cfg2test_loader(args.config_file, args)
    dataset = test_loader.dataset
    tree = build_tree(dataset)
    plot_tree(tree, dataset.concept_split_ratio)
