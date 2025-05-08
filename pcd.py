import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Sample points3D output
points3D = []
np.random.seed(42)
for i in range(200):
    xyz = np.random.randn(3) * 2   # 3D point with some spread
    rgb = np.random.randint(0, 255, 3)
    point_entry = {
        "id": i,
        "xyz": xyz,
        "rgb": rgb,
        "error": 1.0,
        "track": [(np.random.randint(0, 3), np.random.randint(0, 500))]
    }
    points3D.append(point_entry)

def visualize_point_cloud(points3D, num_points=10000):
    """
    Visualizes the 3D points with their RGB colors.
    Args:
        points3D: List of dicts, each with 'xyz' (3,) and 'rgb' (3,) keys.
        num_points: Max number of points to plot for performance.
    """
    xyz = np.array([pt['xyz'] for pt in points3D])
    rgb = np.array([pt['rgb'] for pt in points3D]) / 255.0  # normalize for matplotlib

    # Sample points for faster plotting if you have a huge point cloud
    if len(xyz) > num_points:
        indices = np.random.choice(len(xyz), num_points, replace=False)
        xyz = xyz[indices]
        rgb = rgb[indices]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=rgb, s=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Point Cloud Visualization')
    plt.show()

# Visualize the example
visualize_point_cloud(points3D)
