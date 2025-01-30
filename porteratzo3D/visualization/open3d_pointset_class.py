import numpy as np
from matplotlib import cm
import matplotlib as mpl
from porteratzo3D.visualization.open3d_utils import convertcloud, MyO3dApp
import copy
from typing import Union

try:
    import open3d

    V3V = open3d.utility.Vector3dVector
    use_headless = False
except ImportError:
    use_headless = True


class PointSetClass:
    """
    Point object, should contain points, can contain color and if it should persist between updates
    for non blocking.
    """

    def __init__(
        self, points: np.ndarray = None, color: list = [], persistant: bool = False
    ) -> None:
        """
        Initializes a point set class.

        Args:
            points: Numpy array of points.
            color: List of colors for the points.
            persistant: Whether the points should persist between updates.
        """
        self.plt_colors = None
        self.display_colors = None
        self.points = points
        self.display_points = None
        self.persistant = persistant
        self.updated = False
        if points is not None:
            self.update(points, color)

    def update(self, points: np.ndarray, color: list = []) -> None:
        """
        Updates the points and colors of the point set.

        Args:
            points: Numpy array of points.
            color: List of colors for the points.
        """
        if type(points) is list:
            self.display_points = np.vstack(points)
        else:
            self.display_points = points

        if self.display_points.shape[1] != 3:
            raise ValueError("If points is a list, it needs to be an nx3 array.")

        self.set_colors(points, color)
        self.updated = True

    def set_colors(self, points: np.ndarray, color: list) -> None:
        """
        Sets the colors of the points.

        Args:
            points: Numpy array of points.
            color: List of colors for the points.
        """
        color_array = np.asarray(color)

        color_1d_n_vector = len(color_array.shape) == 1 and color_array.shape[0]
        color_2d_nx1_vector = len(color_array.shape) == 2 and color_array.shape[1] == 1
        color_2d_nx3_matrix = len(color_array.shape) == 2 and color_array.shape[1] == 3
        valid_color_same_length_as_points = (len(color_array) == len(points)) and (
            color_2d_nx1_vector or color_2d_nx3_matrix or color_1d_n_vector
        )

        color_1d_3_vector = len(color_array.shape) == 1 and color_array.shape[0] == 3
        if valid_color_same_length_as_points:
            if color_2d_nx3_matrix:
                self.display_colors = color_array
            else:
                if color_2d_nx1_vector:
                    color_array = color_array[:, 0]
                self.plt_colors = color_array
                normalized_color = (color_array - np.min(color_array)) / (
                    np.max(color_array) - np.min(color_array)
                )
                self.display_colors = cm.jet(normalized_color)[:, :3]
        elif color_1d_3_vector:
            self.display_colors = np.ones_like(points) * color_array
        else:
            self.display_colors = np.vstack(
                np.ones_like(points) * cm.nipy_spectral(np.random.rand(len(points)))[:, :3]
            )

    def draw(self) -> None:
        """
        Draws the point set.
        """
        pass

    def state_dict(self) -> dict:
        """
        Returns the state dictionary of the point set.

        Returns:
            State dictionary of the point set.
        """
        if self.persistant or self.updated:
            self.updated = False
            return {
                "points": (np.asarray(self.points).astype(np.float32)),
                "colors": (np.asarray(self.plt_colors).astype(np.float32)),
            }
        else:
            return {}


class O3dPointSetClass(PointSetClass):
    def __init__(
        self,
        points: np.ndarray = None,
        color: list = [],
        persistant: bool = False,
        name: str = "Pointcloud",
        show_color_bars: bool = False,
    ) -> None:
        """
        Initializes an Open3D point set class.

        Args:
            points: Numpy array of points.
            color: List of colors for the points.
            persistant: Whether the points should persist between updates.
            name: Name of the point cloud.
        """
        self.name = name
        self.cloud = None
        self.is_visible = False
        self.show_color_bars = show_color_bars
        super().__init__(points, color, persistant)

    def update(self, points: np.ndarray = [], color: list = []) -> None:
        """
        Updates the points and colors of the point set.

        Args:
            points: Numpy array of points.
            color: List of colors for the points.
        """
        if len(points) > 0:
            super().update(points, color)
        if self.cloud is None:
            self.cloud = convertcloud(self.display_points)
        else:
            self.cloud.points = V3V(self.display_points)
        if self.display_colors is not None:
            self.cloud.colors = V3V(self.display_colors)
        self.updated = True

    def draw(self, app_object: MyO3dApp) -> None:
        """
        Draws the point set in the visualizer.

        Args:
            app_object: Instance of MyO3dApp.
        """
        if (self.persistant) or (self.updated):
            if not self.is_visible:
                app_object.o3d_visualizer.add_geometry(self.name, self.cloud)
                app_object.global_visible_geometry_list.append(self.name)
                self.is_visible = True
            else:
                app_object.o3d_visualizer.remove_geometry(self.name)
                app_object.o3d_visualizer.add_geometry(self.name, self.cloud)

            if self.show_color_bars:
                if self.plt_colors is not None:
                    self.render_colorbar(app_object)
            self.updated = False
        else:
            if self.is_visible:
                self.remove(app_object)

    def render_colorbar(self, app_object):
        color_d_n_vector = len(self.plt_colors.shape) == 1
        color_2d_nx1_vector = len(self.plt_colors.shape) == 2 and self.plt_colors.shape[1] == 1
        color_2d_nx3_matrix = len(self.plt_colors.shape) == 2 and self.plt_colors.shape[1] == 3

        valid_color_same_length_as_points = (len(self.plt_colors) == len(self.display_points)) and (
            color_2d_nx1_vector or color_2d_nx3_matrix or color_d_n_vector
        )
        if valid_color_same_length_as_points:
            if color_2d_nx1_vector:
                self.plt_colors = self.plt_colors[:, 0]

            color_min, color_max = [np.min(self.plt_colors), np.max(self.plt_colors)]
            norm = mpl.colors.Normalize(vmin=color_min, vmax=color_max)
            colorbar = mpl.cm.ScalarMappable(norm=norm, cmap=cm.jet)
            colorbar.set_array([])

            app_object.ax.set_position([0.85, 0.1, 0.03, 0.7])
            if app_object.first_axis:
                app_object.first_axis = False
            app_object.fig.colorbar(
                colorbar,
                ax=app_object.ax,
                orientation="vertical",
                label=f"{self.name}",
            )

    def remove(self, app_object: MyO3dApp) -> None:
        """
        Removes the point set from the visualizer.

        Args:
            app_object: Instance of MyO3dApp.
        """
        app_object.o3d_visualizer.remove_geometry(self.name)
        app_object.global_visible_geometry_list.remove(self.name)
        self.is_visible = False


class multi_bounding_box:
    def __init__(
        self, workpoints: np.ndarray, color: np.ndarray, name="box_1", persistant=False
    ) -> None:
        """
        Initializes a multi bounding box.

        Args:
            workpoints: Numpy array of points.
            color: Color of the bounding box.
        """

        self.scale_factor = 0.01
        self.number_of_lines = 4
        self.name = name
        self.updated = False
        self.is_visible = False

        try:
            self.main_bb = open3d.geometry.OrientedBoundingBox.create_from_points(
                open3d.utility.Vector3dVector(workpoints)
            )
        except RuntimeError:
            workpoints = np.asanyarray(workpoints)
            workpoints += np.random.normal(scale=1e-6, size=np.asanyarray(workpoints).shape)
            self.main_bb = open3d.geometry.OrientedBoundingBox.create_from_points(
                open3d.utility.Vector3dVector(workpoints)
            )
        self.main_bb.color = color
        self.secondary_bb = []
        for i in range(1, self.number_of_lines + 1):
            self.secondary_bb.append(copy.copy(self.main_bb))
        self.set_points(workpoints)

    def set_points(self, workpoints: np.ndarray) -> None:
        """
        Sets the points of the bounding box.

        Args:
            workpoints: Numpy array of points.
        """
        main_bb = open3d.geometry.OrientedBoundingBox.create_from_points(
            open3d.utility.Vector3dVector(workpoints)
        )
        self.main_bb.center = main_bb.center
        self.main_bb.extent = main_bb.extent
        self.main_bb.R = main_bb.R
        self.updated = True

        for n, i in enumerate(self.secondary_bb):
            sec = copy.copy(main_bb)
            sec.scale(1.0 + (n + 1) * self.scale_factor, sec.get_center())
            i.center = sec.center
            i.extent = sec.extent
            i.R = sec.R

    def draw(self, vis: open3d.visualization.Visualizer) -> None:
        """
        Draws the bounding box in the visualizer.

        Args:
            vis: Instance of Visualizer.
        """
        if self.updated:
            if not self.is_visible:
                vis.add_geometry(self.name, self.main_bb)
                for n, i in enumerate(self.secondary_bb):
                    vis.add_geometry(self.name + f"secondary_{n}", i)
                self.is_visible = True
            else:
                vis.remove_geometry(self.name)
                vis.add_geometry(self.name, self.main_bb)
                for n, i in enumerate(self.secondary_bb):
                    vis.remove_geometry(self.name + f"secondary_{n}")
                    vis.add_geometry(self.name + f"secondary_{n}", i)
            self.updated = False
        else:
            if self.is_visible:
                self.remove(vis)

    def remove(self, vis: open3d.visualization.Visualizer) -> None:
        """
        Removes the bounding box from the visualizer.

        Args:
            vis: Instance of Visualizer.
        """
        vis.remove_geometry(self.name)
        for n, i in enumerate(self.secondary_bb):
            vis.remove_geometry(self.name + f"secondary_{n}")


def check_point_format(nppoints: Union[np.ndarray, list, tuple, O3dPointSetClass]) -> list:
    """
    Ensures points are in a list of np.array or o3d_pointSetClass.

    Args:
        nppoints: Points to check.

    Returns:
        List of points.
    """
    # make sure points are in a list of np.array or o3d_pointSetClass
    assert isinstance(
        nppoints, (np.ndarray, list, tuple, O3dPointSetClass)
    ), "Not valid point_cloud"

    if isinstance(
        nppoints,
        (
            np.ndarray,
            O3dPointSetClass,
            open3d.geometry.TriangleMesh,
            open3d.geometry.PointCloud,
        ),
    ):
        return [nppoints]
    else:
        assert all(
            isinstance(i, (np.ndarray, O3dPointSetClass)) for i in nppoints
        ), f"Not valid list of point_clouds. {nppoints}"
    return nppoints
