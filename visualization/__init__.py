from .concepts import *
from .matrix import *
from .polygon import *
from .summary import *
from .visualizer import Visualizer, VisualizerList

__all__ = ["build_visualizer"]


def build_visualizer(visualization_cfg, dataset, summary_writer):
    visualizers = set()
    for part in visualization_cfg:
        visualizers.add(Visualizer.VISUALIZER_REGISTRY[part](dataset, summary_writer))
    visualizers = list(visualizers)
    return VisualizerList(visualizers)
