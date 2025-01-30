import numpy as np
from matplotlib import cm
from porteratzo3D.visualization.open3d_utils import MyO3dApp
from porteratzo3D.visualization.open3d_vis import add_axis
import os
import time
from porteratzo3D.visualization.open3d_pointset_class import (
    O3dPointSetClass,
    PointSetClass,
    multi_bounding_box,
)
from porteratzo3D.transforms import euler_angles_to_rotation_matrix

try:
    import open3d

    V3V = open3d.utility.Vector3dVector
    use_headless = False
except ImportError:
    use_headless = True
import pickle
from typing import Union


class open3dpaint_non_blocking:
    def __init__(
        self,
        pointsize: int = 1,
        axis: bool = False,
        white_background: bool = False,
        voxel_size: Union[int, float] = None,
        skybox: bool = False,
        settings: bool = True,
        show_color_bars: bool = False,
        file_name: str = None,
        disable: bool = False,
        headless: bool = False,
    ) -> None:
        """
        Initializes the non-blocking Open3D painter.

        Args:
            pointsize: Size of the points.
            file_name: Name of the file to save the state.
            axis: Whether to display the axis.
            disable: Whether to disable the painter.
            headless: Whether to run in headless mode.
        """
        self.disable = disable
        self.headless = use_headless or headless

        if not self.disable:
            if not self.headless:
                myapp = MyO3dApp(pointsize, white_background, skybox)
                self.myapp = myapp
                myapp.o3d_visualizer.show_settings = settings
                self.view = self.myapp.o3d_visualizer.setup_camera
                if axis:
                    add_axis(axis, myapp)
                self.myapp.o3d_visualizer.reset_camera_to_default()
                self.myapp.app.add_window(self.myapp.o3d_visualizer)
            self.T = np.eye(4)
            self.all_bounding_box = []
            self.frame = 0
            self.file_name = file_name
            self.pointsets: dict[str, Union[O3dPointSetClass, PointSetClass]] = {}
            self.state_save = []
            self.show_color_bars = show_color_bars
            self.voxel_size = voxel_size
            self.color_list = [cm.tab20c(i)[:3] for i in np.linspace(0, 1, 20)]
            self.first_draw = True
            self.bb_list = None
            self.bb_color_list = None

    def update_points(
        self,
        nppoints: np.ndarray,
        color: np.ndarray = None,
        pointset: str = "0",
        persistant: bool = False,
    ) -> None:
        """
        Updates points in the visualizer.

        Args:
            nppoints: Numpy array of points.
            pointset: Index of the point set.
            color_map: Color map to use for coloring the points.
            persistant: Whether the points should persist between updates.
        """
        if not self.disable:
            if len(nppoints) < 1:
                return

            if color is not None:
                group_colors = color
            else:
                group_colors = self.color_list[np.random.choice(len(self.color_list))]

            if not self.pointsets.get(pointset):
                self.pointsets[pointset] = (
                    O3dPointSetClass(
                        persistant=persistant, name=pointset, show_color_bars=self.show_color_bars
                    )
                    if not self.headless
                    else PointSetClass(
                        persistant=persistant, name=pointset, show_color_bars=self.show_color_bars
                    )
                )
            self.pointsets[pointset].update(nppoints, group_colors)

    def draw(
        self,
        saveim: bool = False,
        save_name: str = None,
        inc: bool = False,
        sleep_time: float = 0.05,
    ) -> bool:
        """
        Draws the visualizer.

        Args:
            saveim: Whether to save the image.
            save: Path to save the image.
            inc: Whether to increment the file name.

        Returns:
            Whether to exit the visualizer.
        """
        if not self.disable:
            for keys in self.pointsets.keys():
                if not self.headless:
                    self.pointsets[keys].draw(self.myapp)
            if self.bb_list is not None:
                for n, (nppoints, color) in enumerate(zip(self.bb_list, self.bb_color_list)):
                    self._update_points_of_interest_multiline(nppoints, color, n)

            if self.all_bounding_box is not None:
                if self.bb_list is None:
                    for _ in range(len(self.all_bounding_box)):
                        self.all_bounding_box[-1].remove(self.myapp.o3d_visualizer)
                        self.all_bounding_box.pop(-1)
                else:
                    while len(self.bb_list) < len(self.all_bounding_box):
                        self.all_bounding_box[-1].remove(self.myapp.o3d_visualizer)
                        self.all_bounding_box.pop(-1)

            self.bb_list = None
            self.bb_color_list = None

            if not self.headless:
                if self.first_draw:
                    self.myapp.o3d_visualizer.reset_camera_to_default()
                    self.first_draw = False
                    self.get_perspective()

                tick_return = self.myapp.app.run_one_tick()
                if tick_return:
                    self.myapp.o3d_visualizer.post_redraw()

                if saveim:
                    if self.file_name is not None:
                        if save_name is None:
                            save_path = self.file_name + str(self.frame).zfill(5) + ".jpg"
                        else:
                            save_path = save_name
                            if inc:
                                n = 0
                                while os.path.isfile(save_path):
                                    save_path = save_name.split(".jpg")[0] + "_" + str(n) + ".jpg"
                                    n += 1
                        self.myapp.o3d_visualizer.export_current_image(save_path)
                        self.frame += 1
                time.sleep(sleep_time)
                return tick_return
            else:
                if saveim:
                    self.state_save.append(
                        {key: self.pointsets[key].state_dict() for key in self.pointsets.keys()}
                    )
                time.sleep(sleep_time)
                return False

    def rotate_pointset(
        self,
        pointset: str = "0",
        rotation: tuple = (0.0, 0.0, 0.0),
        center: Union[str, np.ndarray] = "self",
    ) -> None:
        """
        Rotates a point set.

        Args:
            pointset: Index of the point set.
            rotation: Rotation angles in degrees.
            center: Center of rotation.
        """
        R = self.pointsets[pointset].cloud.get_rotation_matrix_from_xyz(np.deg2rad(rotation))
        if type(center) is str:
            self.pointsets[pointset].cloud.rotate(R)
        else:
            self.pointsets[pointset].cloud.rotate(R, center=center)
        self.pointsets[pointset].updated = True

    def translate_pointset(
        self,
        pointset: str = "0",
        translation: tuple = (0.0, 0.0, 0.0),
    ) -> None:
        """
        Translates a point set.

        Args:
            pointset: Index of the point set.
            translation: Translation vector.
        """
        if not self.disable:
            self.pointsets[pointset].cloud.translate(translation)
            self.pointsets[pointset].updated = True

    def _update_points_of_interest_multiline(
        self, nppoints: np.ndarray, color: np.ndarray = np.array([1.0, 0.0, 0.0]), index=0
    ) -> None:
        """
        Updates points of interest with multiple lines.

        Args:
            nppoints: Numpy array of points.
            color: Color of the points.
        """
        if not self.disable:
            if not self.headless:

                workpoints = nppoints
                if len(workpoints) > 3:
                    if len(self.all_bounding_box) <= index:
                        bb = multi_bounding_box(workpoints, color, f"box_{index}")
                        bb.draw(self.myapp.o3d_visualizer)
                        self.all_bounding_box.append(bb)
                    else:
                        self.all_bounding_box[index].set_points(workpoints)
                        self.all_bounding_box[index].draw(self.myapp.o3d_visualizer)

    def update_points_of_interest_multiline(
        self,
        nppoints_list: Union[list[Union[np.ndarray, str]], Union[np.ndarray, str]],
        color_list: list[np.ndarray] = None,
    ) -> None:
        """
        Updates points of interest with multiple lines for a batch of points.

        Args:
            nppoints_list: List of numpy arrays of points or point set keys.
            color_list: List of colors for the points.
        """
        if not self.disable:
            if not self.headless:
                if not isinstance(nppoints_list, list):
                    nppoints_list = [nppoints_list]

                resolved_points_list = []
                for item in nppoints_list:
                    if isinstance(item, str) and item in self.pointsets:
                        resolved_points_list.append(self.pointsets[item].cloud.points)
                    else:
                        resolved_points_list.append(item)

                if color_list is None:
                    color_list = [np.array([1.0, 0.0, 0.0])] * len(resolved_points_list)

                self.bb_list = resolved_points_list
                self.bb_color_list = color_list

    def rotate(self, x: float, y: float, z: float) -> None:
        """
        Rotates the visualizer.

        Args:
            x: Rotation angle around the x-axis.
            y: Rotation angle around the y-axis.
            z: Rotation angle around the z-axis.
        """
        if not self.disable:
            T = np.eye(4)
            T[:3, :3] = euler_angles_to_rotation_matrix(np.deg2rad([x, y, z]))
            self.T = T @ self.T
            self.set_perspective()

    def translate(self, x: float, y: float, z: float) -> None:
        """
        Translates the visualizer.

        Args:
            x: Translation along the x-axis.
            y: Translation along the y-axis.
            z: Translation along the z-axis.
        """
        if not self.disable:
            T = np.eye(4)
            T[:, 3] = np.array([x, y, z, 1])
            self.T = T @ self.T
            self.set_perspective()

    def set_perspective(self) -> None:
        """
        Sets the perspective of the visualizer.
        """
        if not self.disable:
            if not self.headless:
                camera_extrinsic = self.T  # Your 4x4 transformation matrix
                camera_position = camera_extrinsic[
                    :3, 3
                ]  # The translation vector (camera position)
                camera_orientation = camera_extrinsic[
                    :3, :3
                ]  # The rotation matrix (camera orientation)

                # Define the target point and up vector
                # Typically, the camera looks along the negative Z-axis in its local frame
                camera_target = camera_position + camera_orientation @ np.array(
                    [0, 0, -1]
                )  # Look in -Z direction
                camera_up = camera_orientation @ np.array([0, 1, 0])  # Y-axis as the "up" vector

                # Set the camera in O3DVisualizer
                field_of_view = 60.0  # Adjust as needed
                self.myapp.o3d_visualizer.setup_camera(
                    field_of_view,
                    camera_position.tolist(),
                    camera_target.tolist(),
                    camera_up.tolist(),
                )

                time.sleep(0.02)

    def get_perspective(self) -> None:
        self.T = self.myapp.o3d_visualizer.scene.camera.get_view_matrix()

    def save_state(self) -> None:
        """
        Saves the state of the visualizer.
        """
        with open(self.file_name + ".pkl", "wb") as f:
            pickle.dump(self.state_save, f, pickle.HIGHEST_PROTOCOL)

    def load_state(self, load_path: str) -> None:
        """
        Loads the state of the visualizer.

        Args:
            load_path: Path to the saved state file.
        """
        with open(load_path, "rb") as f:
            self.state_save = pickle.load(f)

    def run_from_state(self, save: bool) -> bool:
        """
        Runs the visualizer from a saved state.

        Args:
            save: Whether to save the state.

        Returns:
            Whether to exit the visualizer.
        """
        state = self.state_save
        exit_vis = False
        for step in state:
            for pointset_key in step.keys():
                self.update_points(
                    step[pointset_key]["points"],
                    pointset=pointset_key,
                    color=step[pointset_key]["colors"],
                )
            self.set_perspective()
            if self.draw(save):
                exit_vis = True
                break
        return exit_vis

    def stop(self) -> None:
        """
        Stops the visualizer.
        """
        if not self.disable:
            if not self.headless:
                self.myapp.app.quit()
