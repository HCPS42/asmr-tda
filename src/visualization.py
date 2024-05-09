from IPython.display import Image, display

from config import IMAGE_PATH, INTERACTIVE


def visualize(fig, title):
    if INTERACTIVE:
        fig.show()
    else:
        img_path = f'{IMAGE_PATH}/{title}.png'
        display(Image(img_path))
