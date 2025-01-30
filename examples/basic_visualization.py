from porteratzo3D.create_geometries import make_sphere, make_cylinder, make_plane
from porteratzo3D.visualization.open3d_vis import open3dpaint


def main():
    object_list = []
    object_list.append(make_sphere())
    object_list.append(make_cylinder())
    object_list.append(make_plane())
    open3dpaint(object_list, pointsize=5, axis=True, white_background=True)


if __name__ == "__main__":
    main()
