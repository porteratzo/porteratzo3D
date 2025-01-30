import numpy as np
import math
from typing import Tuple


def svd_rigid_body_transform(
    points1: np.ndarray, points2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the rigid body transformation using SVD.

    Args:
        points1 (np.ndarray): Source points.
        points2 (np.ndarray): Destination points.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Translation and rotation matrices.
    """
    centroid1 = np.sum(points1, 0) / len(points1)
    centroid2 = np.sum(points2, 0) / len(points2)
    centered_vector1 = (points1 - centroid1).transpose()
    centered_vector2 = (points2 - centroid2).transpose()

    yi_t = centered_vector2.transpose()
    xi = centered_vector1
    wi = np.eye(xi.shape[1])

    s = np.matmul(xi, np.matmul(wi, yi_t))
    u, sig_n_diag, vt = np.linalg.svd(s, full_matrices=True)
    raro = np.eye(3)
    raro[-1, -1] = np.linalg.det(np.matmul(vt.transpose(), u.transpose()))

    rotacion_final = np.matmul(vt.transpose(), np.matmul(raro, u.transpose()))
    traslation = (centroid2 - np.matmul(rotacion_final, centroid1)).reshape(3, 1)
    return traslation, rotacion_final


def svd_lst_sqr(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Solves the least squares problem using SVD.

    Args:
        X (np.ndarray): Input matrix.
        Y (np.ndarray): Output matrix.

    Returns:
        np.ndarray: Solution matrix.
    """
    U, s, VT = np.linalg.svd(X)
    S = np.diag(s)
    Si = np.zeros([U.shape[1], VT.shape[0]])
    Si[0 : S.shape[0], 0 : S.shape[0]] = np.diag(1 / s)
    return np.matmul(VT.T, np.matmul(Si.T, np.matmul(U.T, Y)))


def euler_angles_to_rotation_matrix(theta: np.ndarray) -> np.ndarray:
    """
    Converts Euler angles to a rotation matrix.

    Args:
        theta (np.ndarray): Euler angles.

    Returns:
        np.ndarray: Rotation matrix.
    """
    r_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta[0]), -np.sin(theta[0])],
            [0, np.sin(theta[0]), np.cos(theta[0])],
        ]
    )
    r_y = np.array(
        [
            [np.cos(theta[1]), 0, np.sin(theta[1])],
            [0, 1, 0],
            [-np.sin(theta[1]), 0, np.cos(theta[1])],
        ]
    )
    r_z = np.array(
        [
            [np.cos(theta[2]), -np.sin(theta[2]), 0],
            [np.sin(theta[2]), np.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )
    r = np.dot(r_z, np.dot(r_y, r_x))
    return r


def euler_angles_to_rotation_matrix_zyx(theta: np.ndarray) -> np.ndarray:
    """
    Converts Euler angles to a rotation matrix using ZYX convention.

    Args:
        theta (np.ndarray): Euler angles.

    Returns:
        np.ndarray: Rotation matrix.
    """
    r_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta[0]), -np.sin(theta[0])],
            [0, np.sin(theta[0]), np.cos(theta[0])],
        ]
    )
    r_y = np.array(
        [
            [np.cos(theta[1]), 0, np.sin(theta[1])],
            [0, 1, 0],
            [-np.sin(theta[1]), 0, np.cos(theta[1])],
        ]
    )
    r_z = np.array(
        [
            [np.cos(theta[2]), -np.sin(theta[2]), 0],
            [np.sin(theta[2]), np.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )
    r = np.dot(r_x, np.dot(r_y, r_z))
    return r


def is_rotation_matrix(R: np.ndarray) -> bool:
    """
    Checks if a matrix is a valid rotation matrix.

    Args:
        R (np.ndarray): Input matrix.

    Returns:
        bool: True if the matrix is a valid rotation matrix, False otherwise.
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    identity = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(identity - shouldBeIdentity)
    return n < 1e-6


def rotation_matrix_to_euler_angles_zyx(r: np.ndarray) -> np.ndarray:
    """
    Converts a rotation matrix to Euler angles using ZYX convention.

    Args:
        r (np.ndarray): Rotation matrix.

    Returns:
        np.ndarray: Euler angles.
    """
    assert is_rotation_matrix(r)

    if r[2, 0] < 1:
        if r[2, 0] > -1:
            y = np.arcsin(-r[2, 0])
            z = np.arctan2(r[1, 0], r[0, 0])
            x = np.arctan2(r[2, 1], r[1, 1])
        else:
            y = np.pi / 2
            z = -np.arctan2(-r[1, 2], r[1, 1])
            x = 0
    else:
        y = -np.pi / 2
        z = np.arctan2(-r[1, 2], r[1, 1])
        x = 0

    return np.array([x, y, z])


def rotation_matrix_to_euler_angles_yzx(r: np.ndarray) -> np.ndarray:
    """
    Converts a rotation matrix to Euler angles using YZX convention.

    Args:
        r (np.ndarray): Rotation matrix.

    Returns:
        np.ndarray: Euler angles.
    """
    assert is_rotation_matrix(r)

    if r[1, 0] < 1:
        if r[1, 0] > -1:
            z = np.arcsin(r[1, 0])
            y = np.arctan2(-r[2, 0], r[0, 0])
            x = np.arctan2(-r[1, 2], r[1, 1])
        else:
            z = -np.pi / 2
            y = -np.arctan2(-r[2, 1], r[2, 2])
            x = 0
    else:
        z = -np.pi / 2
        y = np.arctan2(r[2, 1], r[2, 2])
        x = 0

    return np.array([x, y, z])


def rotation_matrix_to_euler_angles(r: np.ndarray) -> np.ndarray:
    """
    Converts a rotation matrix to Euler angles.

    Args:
        r (np.ndarray): Rotation matrix.

    Returns:
        np.ndarray: Euler angles.
    """
    assert is_rotation_matrix(r)

    sy = math.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(r[2, 1], r[2, 2])
        y = math.atan2(-r[2, 0], sy)
        z = math.atan2(r[1, 0], r[0, 0])
    else:
        x = math.atan2(-r[1, 2], r[1, 1])
        y = math.atan2(-r[2, 0], sy)
        z = 0

    return np.array([x, y, z])
