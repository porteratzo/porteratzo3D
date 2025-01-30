import numpy as np
from typing import Optional, Tuple, List


def distance_between_points(a: np.ndarray, b: Optional[np.ndarray] = None) -> float:
    """
    Computes the distance between two points.

    Args:
        a (np.ndarray): First point.
        b (Optional[np.ndarray]): Second point (optional).

    Returns:
        float: Distance between the points.
    """
    if b is not None:
        c = a - b
    else:
        c = a
    anorm = np.sqrt(np.sum(np.multiply(c, c)))
    return anorm


def angle_between_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Computes the angle between two vectors.

    Args:
        a (np.ndarray): First vector.
        b (np.ndarray): Second vector.

    Returns:
        np.ndarray: Angle between the vectors.
    """
    dot_product = np.sum(np.multiply(a, b), axis=-1)
    norms_product = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)

    # Ensure the value is within the valid range for arccos
    value = np.clip(dot_product / norms_product, -1, 1)

    angle = np.arccos(value)
    return angle


def get_principal_vectors(A: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
    """
    Computes the principal vectors and values of a matrix centered around (0,0,0).

    Args:
        A (np.ndarray): Input matrix.

    Returns:
        Tuple[List[np.ndarray], List[float]]: Principal vectors and values.
    """
    centroid = np.mean(A, axis=0)
    centered_points = A - centroid
    VT = np.linalg.eig(np.matmul(centered_points.T, centered_points))
    sort = sorted(zip(VT[0], VT[1].T.tolist()), reverse=True)
    Values, Vectors = zip(*sort)
    return Vectors, Values


def distance_point_plane(point: np.ndarray, plane_coefs: np.ndarray) -> float:
    """
    Computes the minimum distance from a point to a plane.

    Args:
        point (np.ndarray): Point coordinates.
        plane_coefs (np.ndarray): Plane coefficients.

    Returns:
        float: Distance from the point to the plane.
    """
    return (
        plane_coefs[0] * point[0]
        + plane_coefs[1] * point[1]
        + plane_coefs[2] * point[2]
        + plane_coefs[3]
    ) / np.linalg.norm(plane_coefs[0:3])


def dist_point_to_line(
    point: np.ndarray, line_point1: np.ndarray, line_point2: np.ndarray = np.array([0, 0, 0])
) -> float:
    """
    Computes the distance from a point to a line.

    Args:
        point (np.ndarray): Point coordinates.
        line_point1 (np.ndarray): First point on the line.
        line_point2 (np.ndarray): Second point on the line (default is origin).

    Returns:
        float: Distance from the point to the line.
    """
    cross_product = np.cross((point - line_point2), (point - line_point1))
    distance = np.linalg.norm(cross_product, axis=-1) / np.linalg.norm(
        line_point1 - line_point2, axis=-1
    )
    return distance


def rotation_matrix_from_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """
    Computes the rotation matrix that aligns vec1 to vec2.

    Args:
        vec1 (np.ndarray): First vector.
        vec2 (np.ndarray): Second vector.

    Returns:
        np.ndarray: Rotation matrix.
    """
    if all(np.abs(vec1) == np.abs(vec2)):
        return np.eye(3)
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix
