from typing import Union

try:
    import open3d as o3d

    use_headless = False
except ImportError:
    use_headless = True
import numpy as np
import matplotlib.pyplot as plt


def convertcloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Converts a numpy array of points to an Open3D PointCloud.

    Args:
        points: Numpy array of points.

    Returns:
        Open3D PointCloud.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def to_tensor_o3d(pcd: o3d.geometry.PointCloud) -> o3d.t.geometry.PointCloud:
    """
    Converts an Open3D PointCloud to a numpy array.

    Args:
        pcd: Open3D PointCloud.

    Returns:
        Numpy array of points.
    """
    device = o3d.core.Device("CPU:0")
    dtype = o3d.core.float32

    # Create an empty point cloud
    # Use pcd.point to access the points' attributes
    t_pcd = o3d.t.geometry.PointCloud(device)
    t_pcd.point["positions"] = o3d.core.Tensor(np.asarray(pcd.points), device=device, dtype=dtype)
    t_pcd.point["colors"] = o3d.core.Tensor(np.asarray(pcd.colors), device=device, dtype=dtype)
    return t_pcd


def cloudsave(points, Path):
    pcd = o3d.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.write_point_cloud(Path + ".ply", pcd)
    pcd_load = o3d.read_point_cloud(Path + ".ply")
    return pcd_load


def downsample(
    point_cloud: np.ndarray, leaf_size: float = 0.005, return_idx: bool = False
) -> Union[np.ndarray, o3d.geometry.PointCloud]:
    """
    Downsamples a point cloud using voxel downsampling.

    Args:
        point_cloud: Numpy array of point cloud.
        leaf_size: Size of the voxels.
        return_idx: Whether to return the indices of the downsampled points.

    Returns:
        Downsampled point cloud or indices of the downsampled points.
    """
    if return_idx:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        _, rest, _ = pcd.voxel_down_sample_and_trace(
            voxel_size=leaf_size,
            min_bound=np.array([-10, -10, -10]),
            max_bound=np.array([10, 10, 10]),
        )
        return rest[rest != -1]
    else:
        return o3d.voxel_down_sample(point_cloud, leaf_size)


class MyO3dApp:

    def __init__(self, pointsize: float, white_background: bool, skybox: bool) -> None:
        """
        Initializes the Open3D application.

        Args:
            pointsize: Size of the points.
            white_background: Whether to use a white background.
            skybox: Whether to show the skybox.
        """
        app = o3d.visualization.gui.Application.instance
        app.initialize()
        vis = o3d.visualization.O3DVisualizer("Open3DVisualizer")
        vis.point_size = pointsize
        self.selected_points = None
        self.global_visible_geometry_list = []
        self.fig, self.ax = plt.subplots()
        self.first_axis = True

        vis.add_action("save selection", self.get_points)
        vis.add_action("hide_all", self.hide_all)
        vis.add_action("show_all", self.show_all)
        vis.add_action("keep_selection", self.keep_selection)

        vis.show_skybox(skybox)
        if white_background:
            vis.set_background(np.asarray([1.0, 1.0, 1.0, 1.0])[:, np.newaxis], None)
        else:
            vis.set_background(np.asarray([0.1, 0.1, 0.1, 1.0])[:, np.newaxis], None)
        self.app = app
        self.o3d_visualizer: o3d.visualization.O3DVisualizer = vis

    def get_points(self, vis: o3d.visualization.O3DVisualizer) -> None:
        """
        Gets the selected points from the visualizer.

        Args:
            vis: Instance of O3DVisualizer.
        """
        self.selected_points = vis.get_selection_sets()
        print([[v for v in i.values()] for i in self.selected_points])

    def hide_all(self, vis: o3d.visualization.O3DVisualizer) -> None:
        """
        Hides all geometries in the visualizer.

        Args:
            vis: Instance of O3DVisualizer.
        """
        for i in self.global_visible_geometry_list:
            vis.show_geometry(i, False)

    def show_all(self, vis: o3d.visualization.O3DVisualizer) -> None:
        """
        Shows all geometries in the visualizer.

        Args:
            vis: Instance of O3DVisualizer.
        """
        for i in self.global_visible_geometry_list:
            vis.show_geometry(i, True)

    def keep_selection(self, vis: o3d.visualization.O3DVisualizer) -> None:
        """
        Keeps the selected geometries in the visualizer.

        Args:
            vis: Instance of O3DVisualizer.
        """
        sel = vis.get_selection_sets()
        leave = [p for i in sel for p in i.keys()]
        for i in self.global_visible_geometry_list:
            if i not in leave:
                vis.show_geometry(i, False)
