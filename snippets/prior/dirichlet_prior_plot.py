import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from utils import mkdir

bins = 40
n_samples = 10000
alphas = [43,43,8.2]
# alphas = [5.5657, 5.5696, 2.4127]
# alphas = [64.3572, 64.3615, 5.3968]
gammas = [torch.distributions.gamma.Gamma(alpha, 1) for alpha in alphas]
batch_size = 128

if __name__ == '__main__':
    samples = []
    for i in tqdm(range(n_samples // batch_size)):
        gs = torch.stack([gamma.sample((batch_size,)) for gamma in gammas])
        gs /= gs.sum(0, keepdims=True)
        samples.append(gs)
    samples = torch.cat(samples, -1)

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)

    xs = (-.5 + samples[0]).tolist()
    ys = (.5 - samples[1]).tolist()

    ax.hist2d(xs, ys, bins=bins, range=[[-.5, .5], [-.5, .5]])
    ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel("Box begin")
    ax.set_ylabel("Box end")
    ax.set_title("Box begin vs end.")

    fig.tight_layout()
    mkdir("output/snippets/dirichlet_prior")
    fig.savefig("output/snippets/dirichlet_prior/figure.png")
