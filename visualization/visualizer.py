from utils import underscores


class Visualizer:
    VISUALIZER_REGISTRY = {}
    ALPHA = .8

    MAX_WIDTH = 120
    MAX_TEXT_LINES = 8
    MAX_ANSWER_LINES = 2

    per_batch = True

    @property
    def per_result(self):
        return not self.per_batch

    def __init__(self, dataset, summary_writer):
        self.dataset = dataset
        self.summary_writer = summary_writer

    @staticmethod
    def get_name(name):
        return underscores(name[:-len('Visualizer')])

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = Visualizer.get_name(cls.__name__)
        cls.VISUALIZER_REGISTRY[name] = cls
        cls.name = name

    def visualize(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return f"{self.name}-{self.dataset}"

    @staticmethod
    def partition(texts, max_lines):
        if isinstance(texts, list):
            all_parts, currents = [], []
            for text in texts:
                if sum(len(_) for _ in currents) < Visualizer.MAX_WIDTH:
                    currents.append(text)
                else:
                    all_parts.append(' '.join(currents))
                    currents = []
            if len(currents) > 0:
                all_parts.append(' '.join(currents))
            if len(all_parts) > max_lines:
                all_parts[max_lines:] = []
            return all_parts
        else:
            return [texts[Visualizer.MAX_WIDTH * i:Visualizer.MAX_WIDTH * (i + 1)] for i in range(max_lines)]


class VisualizerList:
    def __init__(self, visualizers):
        self.visualizers = visualizers

    def visualize(self, *args, **kwargs):
        for visualizer in self.visualizers:
            if ("mode" in args[0]) != visualizer.per_batch:
                visualizer.visualize(*args, **kwargs)

    def __str__(self):
        return str(self.visualizers)
