import numpy as np
from porteratzo3D.geometry_utils import rotation_matrix_from_vectors
from porteratzo3D.visualization.open3d_pointset_class import O3dPointSetClass
import open3d as o3d
from typing import List, Tuple


def make_plane(
    coefs: List[float] = [0, 0, 1, 0],
    center: List[float] = [0, 0, 0],
    size: List[int] = [-1, 1],
    side_res: int = 50,
) -> np.ndarray:
    """
    Creates a plane.

    Args:
        coefs: Coefficients of the plane equation.
        center: Center of the plane.
        size: Size of the plane.
        side_res: Resolution of the plane sides.

    Returns:
        Numpy array of plane points.
    """
    ll, ul = size
    step = (ul - ll) / side_res
    planepoints = np.meshgrid(np.arange(ll, ul, step), np.arange(ll, ul, step))
    plane = np.array(
        [
            planepoints[0].flatten(),
            planepoints[1].flatten(),
            np.zeros(len(planepoints[0].flatten())),
        ]
    ).T
    R = rotation_matrix_from_vectors([0, 0, 1], coefs[0:3])
    return np.add(np.matmul(R, plane.T).T, center)


def make_pointvector(
    coefs: List[float], centroid: List[float] = [0, 0, 0], length: float = 1, dense: int = 10
) -> np.ndarray:
    """
    Creates a point vector.

    Args:
        coefs: Coefficients of the vector.
        centroid: Centroid of the vector.
        length: Length of the vector.
        dense: Density of the vector points.

    Returns:
        Numpy array of vector points.
    """
    assert len(coefs) == 3, "Need x,y,z normalvector"
    newcoefs = np.array(coefs) / np.linalg.norm(np.array(coefs))
    pointline = np.arange(length / dense, length + length / dense, length / dense)
    pointline = np.vstack([pointline, pointline, pointline])
    out = np.add(np.multiply(pointline.T, newcoefs), centroid)
    return out


def make_sphere(
    centroid: List[float] = [0, 0, 0], radius: float = 1, dense: int = 90
) -> np.ndarray:
    """
    Creates a sphere.

    Args:
        centroid: Centroid of the sphere.
        radius: Radius of the sphere.
        dense: Density of the sphere points.

    Returns:
        Numpy array of sphere points.
    """
    n = np.arange(0, 360, int(360 / dense))
    n = np.deg2rad(n)
    x, y = np.meshgrid(n, n)
    x = x.flatten()
    y = y.flatten()
    sphere = np.vstack(
        [
            centroid[0] + np.sin(x) * np.cos(y) * radius,
            centroid[1] + np.sin(x) * np.sin(y) * radius,
            centroid[2] + np.cos(x) * radius,
        ]
    ).T
    return sphere


def make_cylinder(
    model: List[float] = [0, 0, 0, 1, 0, 0, 1], length: float = 1, dense: int = 50
) -> np.ndarray:
    """
    Creates a cylinder.

    Args:
        model: Model parameters of the cylinder.
        length: Length of the cylinder.
        dense: Density of the cylinder points.

    Returns:
        Numpy array of cylinder points.
    """
    radius = model[6]
    X, Y, Z = model[:3]
    direction = model[3:6] / np.linalg.norm(model[3:6])
    n = np.arange(0, 360, int(360 / dense))
    height = np.arange(0, length, length / dense)
    n = np.deg2rad(n)
    x, z = np.meshgrid(n, height)
    x = x.flatten()
    z = z.flatten()
    cyl = np.vstack([np.cos(x) * radius, np.sin(x) * radius, z]).T
    rotation = rotation_matrix_from_vectors([0, 0, 1], direction)
    rotatedcyl = np.matmul(rotation, cyl.T).T + np.array([X, Y, Z])
    return rotatedcyl


def calculate_zy_rotation_for_arrow(vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the ZY rotation matrices for an arrow.

    Args:
        vec: Vector to calculate the rotation for.

    Returns:
        Tuple of Z and Y rotation matrices.
    """
    gamma = np.arctan2(vec[1], vec[0])
    Rz = np.array(
        [[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]]
    )

    vec = Rz.T @ vec

    beta = np.arctan2(vec[0], vec[2])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    return Rz, Ry


def make_arrow(
    end: np.ndarray, origin: np.ndarray = np.array([0, 0, 0]), scale: float = 1
) -> o3d.geometry.TriangleMesh:
    """
    Creates an arrow.

    Args:
        end: End point of the arrow.
        origin: Origin point of the arrow.
        scale: Scale of the arrow.

    Returns:
        Open3D TriangleMesh of the arrow.
    """
    assert not np.all(end == origin)
    vec = np.array(end)
    size = np.sqrt(np.sum(vec**2))

    Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = o3d.geometry.TriangleMesh.create_arrow(
        cone_radius=size / 17.5 * scale,
        cone_height=size * 0.2 * scale,
        cylinder_radius=size / 30 * scale,
        cylinder_height=size * (1 - 0.2 * scale),
    )
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    return mesh


def make_image_plane(
    im_: np.ndarray,
    coefs: List[float] = [0, 0, 1],
    center: List[float] = [0, 0, 0],
    size: List[int] = [-1, 1],
    side_res: int = 100,
) -> o3d.geometry.PointCloud:
    """
    Creates an image plane.

    Args:
        im_: Image to create the plane from.
        coefs: Coefficients of the plane equation.
        center: Center of the plane.
        size: Size of the plane.
        side_res: Resolution of the plane sides.

    Returns:
        Open3D PointCloud of the image plane.
    """
    #x_, y_ = np.meshgrid(np.arange(0, side_res), np.arange(0, side_res))
    x_, y_ = np.meshgrid(np.linspace(0, 1, side_res, endpoint=False), np.linspace(0, 1, side_res, endpoint=False))
    x_, y_ = x_.flatten(), np.array(list(reversed(y_.flatten())))
    x_, y_ = x_ * im_.shape[1], y_ * im_.shape[0]
    colors = im_[y_.astype(int), x_.astype(int)]
    p = O3dPointSetClass(make_plane(coefs, center, size, side_res), colors)
    return p
