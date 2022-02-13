import os
from argparse import ArgumentParser

import yaml

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--prefix", type=str, required=True)
if __name__ == '__main__':
    args = parser.parse_args()
    prefix = args.prefix
    os.makedirs(f"experiments/{prefix}", exist_ok=True)
    if args.dataset == "cub":
        for task in ["fewshot", "detached", "zeroshot"]:
            for method in ["cnn_lstm", "nscl_lstm", "nscl_gnn", "graphical", "recurrent"]:
                for emb in ["box", "cone", "plane"]:
                    if method in ["cnn_lstm", "bottom_up"]:
                        emb = "plane"
                    name = f"{prefix}/{prefix}_{task}_{method.replace('_', '-')}_{emb}"
                    filename = f"experiments/{name}.yaml"
                    config = {'DATASETS': {'TRAIN': f'cub_{task}_train', 'VAL': f'cub_{task}_val',
                        'TEST': f'cub_{task}_test_shallow&cub_{task}_test'}, 'TEMPLATE': ['meta', 'cub', emb],
                        "MODEL": {"NAME": method}, }
                    if method in ["cnn_lstm", "bottom_up"]:
                        config = {**config, "CATALOG": {"USE_TEXT": True}}
                    else:
                        config = {**config,
                            'WEIGHT': {'FILE': f'output/{prefix}_support_{emb}/checkpoints/model_0050000.pth'}}
                    if method in ["cnn_lstm"]:
                        config = {**config, "CATALOG": {"USE_TEXT": True, "HAS_MASK": False}}
                        config['MODEL']['FEATURE_EXTRACTOR'] = {'IN_CHANNELS': 3}
                    with open(filename, "w") as f:
                        yaml.safe_dump(config, f, indent=4)
                    print(f"export NAME={name};tt0")
    elif args.dataset == "clevr":
        for task in ["fewshot", "detached"]:
            for seed in [0, 1, 2, 3]:
                for method in ["cnn_lstm", "nscl_lstm", "nscl_gnn", "graphical", "recurrent"]:
                    for emb in ["box"]:
                        if method in ["cnn_lstm", "bottom_up"]:
                            emb = "plane"
                        name = f"{prefix}/{prefix}_{task}_{method.replace('_', '-')}_{emb}_{seed}"
                        filename = f"experiments/{name}.yaml"
                        config = {'DATASETS': {'TRAIN': f'clevr_{task}_train', 'VAL': f'clevr_{task}_val',
                            'TEST': f'clevr_{task}_debias_test&clevr_{task}_test'},
                            'TEMPLATE': ['meta', 'clevr', emb], "MODEL": {"NAME": method}}
                        if method in ["cnn_lstm", "bottom_up"]:
                            config = {**config, "CATALOG": {"USE_TEXT": True}}
                        else:
                            config = {**config, 'WEIGHT': {
                                'FILE': f'output/{prefix}_support_{seed}/checkpoints/model_0050000.pth'}}
                        if method in ["cnn_lstm"]:
                            config = {**config, "CATALOG": {"USE_TEXT": True, "HAS_MASK": False}}
                            config['MODEL']['FEATURE_EXTRACTOR'] = {'IN_CHANNELS': 3}
                        with open(filename, "w") as f:
                            yaml.safe_dump(config, f, indent=4)
                        print(f"export NAME={name};tt0")
    else:
        for task in ["fewshot", "detached"]:
            for method in ["cnn_lstm", "nscl_lstm", "nscl_gnn", "graphical", "recurrent"]:
                for emb in ["box", "cone", "plane"]:
                    if method in ["cnn_lstm", "bottom_up"]:
                        emb = "plane"
                    name = f"{prefix}/{prefix}_{task}_{method.replace('_', '-')}_{emb}"
                    filename = f"experiments/{name}.yaml"
                    config = {'DATASETS': {'TRAIN': f'gqa_{task}_train', 'VAL': f'gqa_{task}_val',
                        'TEST': f'gqa_{task}_test'}, 'TEMPLATE': ['meta', 'gqa', emb],
                        "MODEL": {"NAME": method}}
                    if method in ["cnn_lstm", "bottom_up"]:
                        config = {**config, "CATALOG": {"USE_TEXT": True}}
                    else:
                        config = {**config,
                            'WEIGHT': {'FILE': f'output/{prefix}_support_{emb}/checkpoints/model_0050000.pth'}}
                    if method in ["cnn_lstm"]:
                        config = {**config, "CATALOG": {"USE_TEXT": True, "HAS_MASK": False}}
                        config['MODEL']['FEATURE_EXTRACTOR'] = {'IN_CHANNELS': 3, 'FROM_FEATURE_DIM': 0}
                    with open(filename, "w") as f:
                        yaml.safe_dump(config, f, indent=4)
                    print(f"export NAME={name};tt0")
