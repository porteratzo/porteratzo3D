import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from porteratzo3D.visualization import open3d_utils
from porteratzo3D.visualization.open3d_utils import MyO3dApp
from porteratzo3D.visualization.open3d_pointset_class import (
    check_point_format,
    O3dPointSetClass,
)

try:
    import open3d

    V3V = open3d.utility.Vector3dVector
    use_headless = False
except ImportError:
    use_headless = True
from typing import Union


def sidexsidepaint(
    *point_clusters: np.ndarray,
    color_map: str = "jet",
    pointsize: float = 0.1,
    axis: bool = False,
    for_thesis: bool = False,
) -> None:
    """
    Paints point clusters side by side using Open3D.

    Args:
        point_clusters: Variable length argument list of point clusters.
        color_map: Color map to use for coloring the points.
        pointsize: Size of the points.
        axis: Whether to display the axis.
        for_thesis: Whether to use a white background for thesis.
    """
    new_points = []
    for arg in point_clusters:
        translate = 0
        for n, points in enumerate(arg):
            if points is None or len(points) == 0:
                continue
            _points = points + np.array([[translate, 0, 0]])
            translate = np.max(_points, 0)[0] = translate
            new_points.append(_points)
    open3dpaint(new_points, pointsize=pointsize, axis=axis, white_background=for_thesis)


def open3dpaint(
    geometry_list: Union[
        list,
        tuple,
        np.ndarray,
        O3dPointSetClass,
        open3d.geometry.TriangleMesh,
        open3d.geometry.PointCloud,
    ],
    pointsize: int = 1,
    axis: bool = False,
    white_background: bool = False,
    voxel_size: Union[int, float] = None,
    skybox: bool = False,
    settings: bool = True,
    show_color_bars: bool = False,
) -> Union[None, list]:
    """
    Visualizes geometries using Open3D.

    Args:
        geometry_list: List of geometries to visualize.
        pointsize: Size of the points.
        axis: Whether to display the axis.
        white_background: Whether to use a white background.
        voxel_size: Size of the voxels for downsampling.
        skybox: Whether to show the skybox.
        settings: Whether to show the settings.

    Returns:
        List of selected points if any.
    """
    if not use_headless:
        myapp = MyO3dApp(pointsize, white_background, skybox)

    geometry_list = check_point_format(geometry_list)
    color_list = (
        [cm.tab20c(i)[:3] for i in np.linspace(0, 1, len(geometry_list))]
    )
    color_choice_list = np.random.choice(len(geometry_list), len(geometry_list), replace=False)

    draw_geometries(
        geometry_list,
        voxel_size,
        myapp,
        color_list,
        color_choice_list,
        show_color_bars=show_color_bars,
    )

    if axis:
        add_axis(axis, myapp)

    myapp.o3d_visualizer.reset_camera_to_default()
    myapp.o3d_visualizer.show_settings = settings

    myapp.app.add_window(myapp.o3d_visualizer)
    if show_color_bars:
        set_color_bars(myapp)
        plt.pause(0.5)
    myapp.app.run()
    return myapp.selected_points


def set_color_bars(myapp: MyO3dApp) -> None:
    """
    Sets color bars for the visualizer.

    Args:
        myapp: Instance of MyO3dApp.
    """
    z = 10
    width = z / ((len(myapp.fig.axes) - 1) * (z + 1) + 1)
    spacing = width / z
    pos = spacing
    for n, ax in enumerate(myapp.fig.axes):
        if n == 0:
            ax.set_position([0, 0.0, 0, 1])
        else:
            ax.set_position([pos, 0.1, width, 0.8])
            pos += width + spacing
    plt.show(block=False)


def draw_geometries(
    geometry_list: list,
    voxel_size: Union[int, float],
    myapp: MyO3dApp,
    color_list: list,
    color_choice_list: np.ndarray,
    show_color_bars: bool = False,
) -> None:
    """
    Draws geometries in the visualizer.

    Args:
        geometry_list: List of geometries to draw.
        voxel_size: Size of the voxels for downsampling.
        myapp: Instance of MyO3dApp.
        color_list: List of colors for the geometries.
        color_choice_list: List of color choices for the geometries.
    """
    for n, i in enumerate(geometry_list):
        if isinstance(i, O3dPointSetClass):
            i.is_visible = False
            i.update()
            i.draw(myapp)
        else:
            if isinstance(i, open3d.geometry.TriangleMesh):
                myapp.o3d_visualizer.add_geometry(f"Points {n}", i)
                myapp.global_visible_geometry_list.append(f"Points {n}")
            else:
                if isinstance(i, list) and len(i) == 0:
                    continue
                workpoints = i
                color = color_list[color_choice_list[n]]
                if voxel_size is not None:
                    if isinstance(workpoints, open3d.geometry.PointCloud):
                        _points = np.asarray(workpoints.points)
                    else:
                        _points = workpoints
                    idx = open3d_utils.downsample(_points, leaf_size=voxel_size, return_idx=True)
                    workpoints = _points[idx]
                current_pointset = O3dPointSetClass(
                    name=f"Points {n}", show_color_bars=show_color_bars
                )
                current_pointset.update(workpoints, color)
                current_pointset.draw(myapp)


def add_axis(axis: bool, myapp: MyO3dApp) -> None:
    """
    Adds an axis to the visualizer.

    Args:
        axis: Whether to display the axis.
        myapp: Instance of MyO3dApp.
    """
    myapp.o3d_visualizer.add_geometry(
        "Axis", open3d.geometry.TriangleMesh.create_coordinate_frame(axis)
    )
    myapp.global_visible_geometry_list.append("Axis")
