# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

Image Players for displaying images in the animation.
"""
from typing import Callable
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio.v2 as imageio
import os

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class ImageStreamPlayer:
    """
    A class for displaying images stream in the animation.
    """

    pass


class ImageSetPlayer:
    """
    A class for displaying images set in the animation.
    """

    IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".bmp", ".gif")

    def __init__(
        self,
        image_dir: str,
        image_filter: Callable = None,
        figure_size: tuple = (10, 5),
        pause: float = 20,
    ):
        self._image_dir = image_dir
        self._filter = image_filter
        self._pause = pause
        self._images = [imageio.imread(f) for f in self.update_images()]

        # initialize the figure and axes
        self._fig, self._ax = plt.subplots(figsize=figure_size)
        self._ax.axis("off")

    def update_images(self) -> list:
        """
        Update the image set.

        Returns:
            Image file paths.
        """
        # retrieve all image files in the directory
        image_files = []
        for f in os.listdir(self._image_dir):
            fpath = os.path.join(self._image_dir, f)
            if not os.path.isfile(fpath):
                continue
            if not fpath.lower().endswith(self.IMAGE_TYPES):
                continue
            image_files.append(fpath)

        # sort the image files by time
        image_files.sort(key=os.path.getmtime)

        # filter the image files
        if self._filter is not None:
            image_files = self._filter(image_files)

        return image_files

    def play(self):
        """
        Play the image set.
        """

        def _init():
            self._ax.clear()
            return (self._ax,)

        def _update(frame):
            self._ax.clear()
            self._ax.imshow(self._images[frame])
            return (self._ax,)

        ani = FuncAnimation(
            self._fig,
            _update,
            init_func=_init,
            frames=len(self._images),
            interval=self._pause,
            blit=True,
            repeat=False,
        )

        plt.show()
        plt.close()


if __name__ == "__main__":
    from core.numerics.fields import NodeField, Scalar
    import random

    # Test ImageStreamPlayer

    # Test ImageSetPlayer
    img_dir = r".\tests\\results"
    player = ImageSetPlayer(img_dir, pause=0.01)
    player.play()
