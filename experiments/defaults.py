from yacs.config import CfgNode as CN

C = CN()

# ---------------------------------------------------------------------------- #
# DataCatalog
# ---------------------------------------------------------------------------- #
C.CATALOG = CN()
C.CATALOG.SHOT_K = 1
C.CATALOG.SPLIT_SEED = 0
C.CATALOG.QUERY_K = 30
C.CATALOG.DROPOUT_RATE = .2
C.CATALOG.HAS_MASK = True
C.CATALOG.TRUNCATED_SIZE = 50
C.CATALOG.USE_TEXT = False
C.CATALOG.SPLIT_RATIO = [.7, .2, .1]
# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
C.DATASETS = CN()
C.DATASETS.TRAIN = ""
C.DATASETS.VAL = ""
C.DATASETS.TEST = ""

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
C.DATALOADER = CN()
# Number of data loading threads
C.DATALOADER.NUM_WORKERS = 10

C.TEMPLATE = []

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
C.MODEL = CN()
C.MODEL.META_ARCHITECTURE = "pretrain"
C.MODEL.REPRESENTATION = "box"
C.MODEL.NAME = "graphical"
C.MODEL.DIMENSION = 100

C.MODEL.BOX_REGISTRY = CN()
C.MODEL.BOX_REGISTRY.PATCH_SIZE = [7, 7]
C.MODEL.BOX_REGISTRY.ENTRIES = 366
C.MODEL.BOX_REGISTRY.INIT = CN()
C.MODEL.BOX_REGISTRY.INIT.METHOD = "uniform"
C.MODEL.BOX_REGISTRY.INIT.CENTER = [-.15, .15]
C.MODEL.BOX_REGISTRY.INIT.OFFSET = [.1, .2]
C.MODEL.BOX_REGISTRY.CLAMP = CN()
C.MODEL.BOX_REGISTRY.CLAMP.CENTER = [-.24, .24]
C.MODEL.BOX_REGISTRY.CLAMP.OFFSET = [.02, .24]

C.MODEL.FEATURE_EXTRACTOR = CN()
C.MODEL.FEATURE_EXTRACTOR.HAS_CACHE = True
C.MODEL.FEATURE_EXTRACTOR.FROM_FEATURE_DIM = 0
C.MODEL.FEATURE_EXTRACTOR.HAS_RELATIONS = True
C.MODEL.FEATURE_EXTRACTOR.IS_PRETRAINED = True
C.MODEL.FEATURE_EXTRACTOR.IN_CHANNELS = 6
C.MODEL.FEATURE_EXTRACTOR.MID_CHANNELS = 64

C.MODEL.BAYES = CN()
C.MODEL.BAYES.MAX_ITER = 1
C.MODEL.BAYES.REDUCTION = "max"
C.MODEL.BAYES.OBSERVATION = "entailment"
C.MODEL.BAYES.N_PARTICLES = 128
C.MODEL.BAYES.PRIOR_LAMBDA = 1.
C.MODEL.BAYES.GAMMA = .99

C.MODEL.BAYES.PRIOR = CN()
C.MODEL.BAYES.PRIOR.BOX_PARAMS = [[21.85, 21.86, 3.96], .5]
C.MODEL.BAYES.PRIOR.PLANE_PARAMS = [0.0044, 0.0844]

C.MODEL.GNN = CN()
C.MODEL.GNN.METHOD = "complete"
C.MODEL.GNN.N_EDGE_TYPES = 3
C.MODEL.GNN.N_LAYERS = 2
C.MODEL.GNN.MID_CHANNELS = 20
C.MODEL.GNN.OUT_CHANNELS = 10
C.MODEL.GNN.MID_CHANNELS_WEIGHT = 1024
C.MODEL.GNN.OUT_CHANNELS_WEIGHT = 512

C.MODEL.LOSS = CN()
C.MODEL.LOSS.NAME = "logit"
C.MODEL.LOSS.TAU = 4.

C.MODEL.FINETUNE = CN()
C.MODEL.FINETUNE.MAX_ITER = 500

C.MODEL.LANGUAGE = CN()
C.MODEL.LANGUAGE.WORD_ENTRIES = 0
C.MODEL.LANGUAGE.PAD_ENTRIES = 4
C.MODEL.LANGUAGE.PROGRAM_ENTRIES = 0
C.MODEL.LANGUAGE.NUM_OBJECTS = 12
C.MODEL.LANGUAGE.HIDDEN_CHANNELS = 256
C.MODEL.LANGUAGE.WORD_CHANNELS = 512
C.MODEL.LANGUAGE.PAD_CHANNELS = 32
C.MODEL.LANGUAGE.N_LAYERS = 2
C.MODEL.LANGUAGE.RNN_CELL = "lstm"
C.MODEL.LANGUAGE.BIDIRECTIONAL = True
C.MODEL.LANGUAGE.USE_ATTENTION = True

C.MODEL.TEMPERATURE = .2

# -----------------------------------------------------------------------------
# Profiler
# -----------------------------------------------------------------------------
C.PROFILE = CN()
C.PROFILE.REGEX = "FewshotModel|BoxRegistry|FeatureExtractor|MessagePassing|Updater|MAP|GNN"

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
C.OUTPUT_DIR = "output"

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
C.SOLVER = CN()
C.SOLVER.MAX_ITER = 50000
C.SOLVER.STEP = []
C.SOLVER.GAMMA = .1

C.SOLVER.BASE_LR = 0.001
C.SOLVER.LR_SPECS = [['prior', 100], ["box_registry", 10]]
C.SOLVER.WEIGHT_DECAY = 0.005
C.SOLVER.CLIP_GRAD_NORM = 10.
C.SOLVER.WARMUP_FACTOR = .1
C.SOLVER.WARMUP_ITER = 2000

C.SOLVER.VALIDATION_PERIOD = 1000
C.SOLVER.CHECKPOINT_PERIOD = 5000
C.SOLVER.VALIDATION_LIMIT = 100
C.SOLVER.PATIENCE = 5

C.SOLVER.BATCH_SIZE = 20

# ---------------------------------------------------------------------------- #
# Visualization
# ---------------------------------------------------------------------------- #
C.VISUALIZATION = []

# ---------------------------------------------------------------------------- #
# Customary weight file
# ---------------------------------------------------------------------------- #
C.WEIGHT = CN()
C.WEIGHT.FILE = ""
C.WEIGHT.REGEX = "box_registry|feature_extractor"
