# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

Image Players for displaying images in the animation.
"""
from core.numerics.fields import Field
from core.numerics.mesh import Mesh

import matplotlib.pyplot as plt


class ImageStreamPlayer:
    """
    A class for displaying images stream in the animation.
    """

    def __init__(
        self, title: str, mesh: Mesh = None, color: str = "g", pause: float = 0.1
    ):
        if mesh is not None and mesh.domain != 1:
            raise ValueError("ImageStreamPlayer only supports 1D meshes yet.")
        self._mesh = mesh

        plt.ion()
        self._title = title
        self._graph = None
        self._pause = pause
        self._color = color

    def __del__(self):
        plt.ioff()
        plt.close()

    def update(self, field: Field):
        """
        Update the image stream.
        """
        if field.dtype != "scalar":
            raise ValueError("ImageStreamPlayer only supports scalar field.")

        values = [e.value for e in field]

        if self._graph is not None:
            self._graph.remove()

        self._graph = plt.plot(values, color=self._color)[0]
        plt.title(self._title)
        plt.pause(self._pause)


class ImageSetPlayer:
    """
    A class for displaying images set in the animation.
    """

    pass


if __name__ == "__main__":
    from core.numerics.fields import NodeField, Scalar
    import random

    player = ImageStreamPlayer("Image Stream")
    field = NodeField(100, Scalar(1))

    num = 0
    while True:
        if num == 20:
            break
        num += 1

        player.update(field)

        for i in random.sample(range(100), 10):
            field[i] = Scalar(random.random())
