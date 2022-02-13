import glob
import os

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw, ImageFont


def get_font():
    font_library = "/usr/share/fonts/truetype"
    if os.path.exists(os.path.join(font_library, "ttf-bitstream-vera", "VeraMono.ttf")):
        return os.path.join(font_library, "ttf-bitstream-vera", "VeraMono.ttf")
    for font_folder in glob.glob(os.path.join(font_library, "*")):
        for font in glob.glob(os.path.join(font_folder, "*Mono.ttf")):
            return font
    for font_folder in glob.glob(os.path.join(font_library, "*")):
        for font in glob.glob(os.path.join(font_folder, "*Regular.ttf")):
            return font


FONT = get_font()
COLORS = {'green': (56, 124, 68), 'red': (178, 34, 34), 'blue': (65, 105, 225)}


def image2figure(image, comments):
    """Add a comment to the image and return a matplotlib figure"""
    plt.clf()
    figure = plt.imshow(image)
    plt.figtext(0.5, 0.01, comments, wrap=True, horizontalalignment='center', fontsize=12)
    return figure


def text2img(lines, display_size, colors=None):
    img = Image.new("RGB", (display_size[1], display_size[0]))
    drawing = ImageDraw.Draw(img)
    plain_text = "\n".join(lines)
    font = None
    for font_size in range(32, 12, -4):
        font = ImageFont.truetype(FONT, font_size)
        total_w, total_h = drawing.textsize(plain_text, font)
        if total_w <= display_size[1] * .9 - 5 and total_h <= display_size[0] * .9 - 5:
            break
    w, h = font.getsize('A')
    if colors is None:
        colors = [COLORS['blue'] for _ in lines]
    else:
        colors = [COLORS[c] for c in colors]
    for i, (l, c) in enumerate(zip(lines, colors)):
        drawing.text((w, display_size[0] * .5 + (h + 4) * (i - len(lines) / 2)), l, fill=c, font=font)
    return img


def to_numbered_list(texts, n_line=1):
    texts = [f"{i}. {t}" for i, t in enumerate(texts)]
    return [' '.join(texts[i] for i in indices.tolist()) for indices in torch.arange(len(texts)).chunk(n_line)]
