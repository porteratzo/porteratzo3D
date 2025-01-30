from porteratzo3D.create_geometries import make_sphere, make_cylinder, make_plane
from porteratzo3D.visualization.open3d_pointset_class import O3dPointSetClass
from porteratzo3D.visualization.open3d_vis_non_blocking import open3dpaint_non_blocking
from typing import List


def main():
    object_list: List[O3dPointSetClass] = []
    shape = make_sphere()
    pointset = O3dPointSetClass(shape, color=[0, 0, 1], name="sphere")
    object_list.append(pointset)
    shape = make_cylinder()
    pointset = O3dPointSetClass(shape, color=shape, name="cylinder")
    object_list.append(pointset)
    shape = make_plane()
    pointset = O3dPointSetClass(shape, color=shape[:, 0], name="plane", show_color_bars=True)
    object_list.append(pointset)
    non_block = open3dpaint_non_blocking(
        pointsize=5, axis=True, white_background=True, show_color_bars=True, file_name="test"
    )

    for pset in object_list[1:]:
        non_block.update_points(
            pset.display_points, pset.display_colors, pset.name, persistant=True
        )

    non_block.update_points(
        object_list[0].display_points,
        object_list[0].display_colors,
        object_list[0].name,
        persistant=False,
    )

    count = 0

    while count < 250:
        non_block.draw(sleep_time=0.01)

        if count < 30:
            # Translate the sphere
            non_block.translate_pointset("sphere", [0, 0, 0.1])
            non_block.translate(0.2, 0, -0.2)
            non_block.rotate(0, -1, 0)
        elif count < 60:
            non_block.translate_pointset("sphere", [0, 0, 0.1])
            non_block.update_points_of_interest_multiline("sphere")
        elif count < 120:
            non_block.translate_pointset("cylinder", [0, 0.1, 0.1])
            non_block.translate_pointset("plane", [0.1, 0, 0.1])
            non_block.rotate_pointset("plane", [1, 5, 5])
            non_block.update_points_of_interest_multiline(['cylinder', 'plane'])
        elif count < 160:
            non_block.translate_pointset("cylinder", [0, 0.1, 0.1])
            non_block.translate_pointset("plane", [0.1, 0, 0.1])
            non_block.rotate_pointset("plane", [1, 5, 5])

        count += 1

    non_block.stop()
    print("Done")


if __name__ == "__main__":
    main()
