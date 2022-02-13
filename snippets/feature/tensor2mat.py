import os

from scipy.io import savemat

from experiments import cfg_from_args
from snippets.snippet_utils import cfg2test_loader
from utils import load, join, ArgumentParser

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    args = arg_parser.parse_args()
    iteration = args.iteration
    cfg=cfg_from_args(args)

    test_loader = cfg2test_loader(cfg, args)
    test_set = test_loader.dataset

    feature_file = join(cfg.OUTPUT_DIR, "inference", cfg.DATASETS.TEST, f"feature_{iteration:07d}.pth")
    if os.path.exists(feature_file):
        features = load(feature_file)[:, :cfg.MODEL.DIMENSION]

        train_features = features[test_set.image_split_specs <= 0].numpy()
        train_labels = test_set.image2classes[test_set.image_split_specs <= 0].numpy()
        train_attributes = test_set.image_attributes[test_set.image_split_specs <= 0][:,
        test_set.attribute_split_specs <= 0].numpy()

        test_features = features[test_set.image_split_specs == 2].numpy()
        test_labels = test_set.image2classes[test_set.image_split_specs == 2].numpy()
        test_attributes = test_set.image_attributes[test_set.image_split_specs == 2][:,
        test_set.attribute_split_specs <= 0].numpy()

        train_data = {"Trains": train_features.T, "TrainLabel": train_labels, "TrainAttMat": train_attributes}
        savemat(join(cfg.OUTPUT_DIR, "inference", cfg.DATASETS.TEST, f"train_feature_{iteration:07d}.mat"),
            train_data)

        test_data = {"Tests": test_features.T, "TestLabel": test_labels, "TestAttMat": test_attributes}
        savemat(join(cfg.OUTPUT_DIR, "inference", cfg.DATASETS.TEST, f"test_feature_{iteration:07d}.mat"),
            test_data)

    logit_file = join(cfg.OUTPUT_DIR, "inference", cfg.DATASETS.TEST, f"logit_{iteration:07d}.pth")
    if os.path.exists(logit_file):
        logits = load(logit_file).numpy()
        logit_data = {"InputAtt": logits}
        savemat(join(cfg.OUTPUT_DIR, "inference", cfg.DATASETS.TEST, f"logit_{iteration:07d}.mat"), logit_data)
