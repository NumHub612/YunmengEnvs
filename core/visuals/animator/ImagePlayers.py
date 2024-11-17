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

    def __init__(self, title: str, color: str = "g", pause: float = 0.1):
        plt.ion()
        self._title = title
        self._graph = None
        self._pause = pause
        self._color = color

    def __del__(self):
        plt.ioff()
        plt.close()

    def update(self, field: Field, mesh: Mesh = None):
        """
        Update the image stream.
        """
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False

        if field.dtype == "tensor":
            raise ValueError("ImageStreamPlayer not support tensor fields yet.")

        if self._graph is not None:
            self._graph.remove()

        if field.dtype == "scalar":
            values = [e.value for e in field]
            self._graph = plt.plot(values, color=self._color)[0]
        else:
            elements = None
            if field.etype == "node":
                elements = [node.coord for node in mesh.nodes]
            elif field.etype == "face":
                elements = [face.center for face in mesh.faces]
            elif field.etype == "cell":
                elements = [cell.center for cell in mesh.cells]

            X = [e.x for e in elements]
            Y = [e.y for e in elements]

            values = field.to_np()
            U = [v.x for v in values]
            V = [v.y for v in values]
            W = [v.z for v in values]
            self._graph = plt.quiver(X, Y, U, V, W, color=self._color)

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
