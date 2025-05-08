import json
import numpy as np
import matplotlib.pyplot as plt

# --- STEP 1: Load transforms.json ---
with open('transforms.json', 'r') as f:
    data = json.load(f)

# --- STEP 2: Parse Camera Centers and Orientations ---
camera_centers = []
camera_dirs = []  # to show orientation
for frame in data['frames']:
    transform = np.array(frame['transform_matrix'])  # (4,4)
    camera_center = transform[:3, 3]  # (x, y, z)
    # Camera looks along -Z axis in its own frame; in world: -R[:,2]
    R = transform[:3, :3]             # 3x3 rotation
    look_dir = -R[:, 2]               # camera -Z axis in world coords
    camera_centers.append(camera_center)
    camera_dirs.append(look_dir)

camera_centers = np.stack(camera_centers, axis=0)
camera_dirs = np.stack(camera_dirs, axis=0)

# --- STEP 3: Plot Trajectory and Camera Orientations ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Path (in order)
ax.plot(camera_centers[:,0], camera_centers[:,1], camera_centers[:,2],
        c='b', linewidth=2, label='Trajectory')
ax.scatter(camera_centers[:,0], camera_centers[:,1], camera_centers[:,2],
           c='r', s=15, label='Camera Centers')

# Orientation quivers (arrows)
for ct, dir in zip(camera_centers[::max(len(camera_centers)//20,1)], camera_dirs[::max(len(camera_dirs)//20,1)]):
    ax.quiver(ct[0], ct[1], ct[2], dir[0], dir[1], dir[2], length=2, color='k', arrow_length_ratio=0.25)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Camera Trajectory from JSON")
ax.legend()
plt.tight_layout()
plt.show()