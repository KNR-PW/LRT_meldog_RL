# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Custom terrain generation utilities for Meldog."""

import numpy as np
import trimesh
from scipy.spatial import transform as tf


def make_cylinder_high_poly(
    radius: float, height: float, center: tuple[float, float, float], max_yx_angle: float = 0, degrees: bool = True
) -> trimesh.Trimesh:
    """Generate a high-polygon cylinder mesh with a random orientation.

    This is a custom version of Isaac Lab's make_cylinder with higher polygon count
    for better visual quality (16-24 sections instead of 4-6).

    Args:
        radius: The radius of the cylinder (in m).
        height: The height of the cylinder (in m).
        center: The center of the cylinder (in m).
        max_yx_angle: The maximum angle along the y and x axis. Defaults to 0.
        degrees: Whether the angle is in degrees. Defaults to True.

    Returns:
        A trimesh.Trimesh object for the high-poly cylinder.
    """
    # create a pose for the cylinder
    transform = np.eye(4)
    transform[0:3, -1] = np.asarray(center)
    # -- create a random rotation
    euler_zyx = tf.Rotation.random().as_euler("zyx")  # returns rotation of shape (3,)
    # -- cap the rotation along the y and x axis
    if degrees:
        max_yx_angle = max_yx_angle / 180.0
    euler_zyx[1:] *= max_yx_angle
    # -- apply the rotation
    transform[0:3, 0:3] = tf.Rotation.from_euler("zyx", euler_zyx).as_matrix()
    # create the cylinder with higher polygon count (16-24 sections instead of 4-6)
    return trimesh.creation.cylinder(radius, height, sections=np.random.randint(16, 25), transform=transform)
