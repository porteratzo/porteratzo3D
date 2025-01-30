# porteratzo3D

Repo to house my utilities for 3D visualization and manipulation using Open3D.

## Overview

Porteratzo3D is a collection of utilities and classes designed to simplify 3D visualization and manipulation with Open3D. It provides functionalities for:  

- Creating and manipulating 3D geometries  
- Visualizing point clouds and meshes  
- Applying transformations to 3D objects  
- Customizing Open3D visualizations for better usability  

## Installation

To install the required dependencies, run:

```bash
git clone https://github.com/porteratzo/porteratzo3D.git
cd porteratzo3D  
pip install .
```  

## Usage

Explore example scripts in the `examples/` directory for practical demonstrations.

### Creating Geometries

Porteratzo3D provides functions to create simple point clouds of shapes with numpy arrays:

```python
from porteratzo3D.visualization.create_geometries import make_sphere, make_cylinder, make_plane

object_list = []
object_list.append(make_sphere())
object_list.append(make_cylinder())
object_list.append(make_plane())
```

### Visualizing Point Clouds

Porteratzo3D has two visualization functions:

1. `open3dpaint` for simple visualization:
```python
from porteratzo3D.visualization.open3d_vis import open3dpaint
from porteratzo3D.visualization.create_geometries import make_sphere
from porteratzo3D.visualization.pointset import O3dPointSetClass

object_list = []
object_list.append(make_sphere())
pointset = O3dPointSetClass(shape, color=[0, 0, 1], name="sphere")
object_list.append(pointset)
open3dpaint(object_list, pointsize=5, axis=True, white_background=True)
```
`open3dpaint` takes a list of numpy arrays representing point clouds or an `O3dPointSetClass` object to display point clouds with specific colors. See `examples/colors.py` for more info.

2. `open3dpaint_non_blocking` for non-blocking visualization:
```python
from porteratzo3D.visualization.open3d_vis_non_blocking import open3dpaint_non_blocking

object_list = []
shape = make_sphere()
pointset = O3dPointSetClass(shape, color=[0, 0, 1], name="sphere")
object_list.append(pointset)
shape = make_cylinder()
pointset = O3dPointSetClass(shape, color=shape, name="cylinder")
object_list.append(pointset)
shape = make_plane()
non_block = open3dpaint_non_blocking(object_list)
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
        non_block.update_points_of_interest_multiline(['cylinder'])
    elif count < 160:
        non_block.translate_pointset("cylinder", [0, 0.1, 0.1])

    count += 1

non_block.stop()
```
`open3dpaint_non_blocking` works in a loop and allows operations such as rotation, translation, and highlighting with bounding boxes. See `examples/non_blocking` for the full example.

## **License**  

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

