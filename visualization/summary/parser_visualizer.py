import torchvision.transforms.functional as TF
from torchvision.utils import make_grid

from utils import text2img, to_cpu_detach
from visualization.visualizer import Visualizer


class ParserVisualizer(Visualizer):
    per_batch = True
    DISPLAY_SIZE = (500, 1500)
    _N_SAMPLES = 4

    def encoded2text(self, encoded):
        return ' '.join([self.dataset.program_vocab.index2word[i] for i in encoded if
            i not in self.dataset.program_vocab.special_tokens])

    def visualize(self, inputs, outputs, model, iteration, **kwargs):
        summaries = []
        inputs = to_cpu_detach(inputs)
        outputs = to_cpu_detach(outputs)
        for i in range(min(self._N_SAMPLES, len(inputs["index"]))):
            lines, colors = [], []
            for t in ['statement', 'metaconcept', 'question']:
                if t not in inputs: continue
                text = inputs[t][i]
                operation = self.encoded2text(inputs[f"{t}_target"][i][..., 0])
                argument = self.encoded2text(inputs[f"{t}_target"][i][..., 1])
                operation_predicted = self.encoded2text(outputs[f"{t}_predicted"][i][..., 0])
                argument_predicted = self.encoded2text(outputs[f"{t}_predicted"][i][..., 1])
                lines.extend([text, operation, argument, operation_predicted, argument_predicted, ""])
                colors.extend(['blue', 'blue', 'blue', 'green' if operation == operation_predicted else 'red',
                    'green' if argument == argument_predicted else 'red', 'blue'])

            lines.extend([f"index:{inputs['index'][i]} question index:{inputs['question_index'][i]}", ""])
            colors.extend(['blue', 'blue'])
            summaries.append(TF.to_tensor(text2img(lines, self.DISPLAY_SIZE, colors)))

        self.summary_writer.add_image(f"parser/{self.dataset.tag}", make_grid(summaries, nrow=1), iteration)
