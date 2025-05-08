import numpy as np
import matplotlib.pyplot as plt

# ---- Include or import your camera trajectory functions ----
def camera_translation(current, destination, n_poses=360):
    flythrough_cams = []
    R = np.array([[1,0,0],[0,1,0],[0,0,1]])
    current_z, current_x = current
    destination_z, destination_x = destination
    if current_z <  destination_z:
        steps_z = np.linspace(current_z, destination_z, n_poses + 1)[:-1]
    else:
        steps_z = np.linspace(destination_z, current_z, n_poses + 1)[:-1][::-1]
    if current_x <  destination_x:
        steps_x = np.linspace(current_x, destination_x, n_poses + 1)[:-1]
    else:
        steps_x = np.linspace(destination_x, current_x, n_poses + 1)[:-1][::-1]
    for i in range(len(steps_z)):
        new_T = [steps_x[i], 0, -steps_z[i]]
        flythrough_cams += [(new_T, R)]
    return flythrough_cams

def camera_rotation(radius=10, angle=np.pi/9, n_poses=180):
    def circular_pose(theta, phi, radius):
        trans_t = lambda t: np.array([
            [0, 0, -radius],
        ], dtype=float).T
        rotation_phi = lambda phi: np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)],
        ], dtype=float)
        rotation_theta = lambda th: np.array([
            [np.cos(th), 0, -np.sin(th)],
            [0, 1, 0],
            [np.sin(th), 0, np.cos(th)],
        ], dtype=float)
        trans_mat = trans_t(radius)
        rot_mat = rotation_phi(phi / 180. * np.pi) 
        rot_mat = rotation_theta(theta) @ rot_mat
        return (trans_mat, rot_mat)

    circular_cams = []
    if angle > 0:
        thetas = np.linspace(0, angle, n_poses + 1)[:-1]
        for th in thetas:
            circular_cams += [circular_pose(th, 0, radius)]
    elif angle < 0:
        thetas = np.linspace(0, -angle, n_poses + 1)
        for th in thetas:
            circular_cams += [circular_pose(-th, 0, radius)]
    else:
        for idx in range(n_poses):
            circular_cams += [circular_pose(0, 0, radius)]
    
    return circular_cams 

def get_zigzag_trajectory(points, rotations, n_poses):
    currents = points[:-1]
    targets = points[1:]
    trajectory = []
    current_R = np.array([[1,0,0],[0,1,0],[0,0,1]])
    for i in range(len(currents)):
        tran_poses = camera_translation(currents[i], targets[i], n_poses=n_poses[i])
        rot_poses = camera_rotation(radius=10, angle=rotations[i], n_poses=n_poses[i])
        poses = [(tran_poses[i][0], current_R @ rot_poses[i][1]) for i in range(len(tran_poses))]
        trajectory += poses
        current_R = poses[-1][1]
    return trajectory

# ---- Use same parameters as your script ----
points = [[0,0], [-5,0], [-20,-20], [-40,-20], [-50,0],[0,0], [0,0]] # z,x
angles = [0, np.pi/8, -np.pi/4, np.pi/8, 0, np.pi/2]
n_poses = [40,30,30,30,40,30]

zigzag_poses = get_zigzag_trajectory(points, angles, n_poses=n_poses)

# Extract translation as xyz
xyz = np.array([pose[0] for pose in zigzag_poses])
xs, ys, zs = xyz[:,0], xyz[:,1], xyz[:,2]

# ---- Plot ----
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(xs, ys, zs, c='b', linewidth=2, label='Trajectory')
ax.scatter(xs, ys, zs, c='r', marker='o', s=12, label='Camera Position')

# Optionally: Show direction vectors (Z axis of the camera)
for i in range(0, len(zigzag_poses), max(len(zigzag_poses)//20, 1)):
    T, R = zigzag_poses[i]
    # Camera looks along -Z camera axis; let's plot this direction
    direction = -R[:,2] * 5  # scaled for visualization
    ax.quiver(T[0], T[1], T[2], direction[0], direction[1], direction[2], color='k', arrow_length_ratio=0.3)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Camera Zigzag Trajectory (Translation Path and Orientation)')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()