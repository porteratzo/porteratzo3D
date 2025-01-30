from porteratzo3D.create_geometries import (
    make_sphere,
    make_cylinder,
    make_plane,
    make_image_plane,
)
from porteratzo3D.visualization.open3d_vis import open3dpaint
from porteratzo3D.visualization.open3d_pointset_class import O3dPointSetClass
import open3d as o3d
import numpy as np


def main():

    object_list = []
    shape = make_sphere()
    pointset = O3dPointSetClass(shape, color=[0, 0, 1], name="sphere")
    object_list.append(pointset)
    shape = make_cylinder()
    pointset = O3dPointSetClass(shape, color=shape, name="cylinder")
    object_list.append(pointset)
    shape = make_plane()
    pointset = O3dPointSetClass(shape, color=shape[:, 0], name="plane", show_color_bars=True)
    object_list.append(pointset)
    redwood_rgbd = o3d.data.SampleRedwoodRGBDImages()
    color_raw = o3d.io.read_image(redwood_rgbd.color_paths[0])
    image_plane = make_image_plane(
        np.asarray(color_raw) / 255, center=[0, 0, -2], size=[-3, 3], side_res=300
    )
    object_list.append(image_plane)
    open3dpaint(object_list, pointsize=5, axis=True, white_background=True, show_color_bars=True)


if __name__ == "__main__":
    main()
