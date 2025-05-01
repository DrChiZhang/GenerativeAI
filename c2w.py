import numpy as np

"""
LookAt function to create a camera-to-world matrix (c2w) for a given camera position, target, and up vector.
The function computes the forward, right, and up vectors to construct the rotation matrix and translation vector.
Parameters:
- eye: Camera position (3D vector)
- center: Target position (3D vector)  
- up: Up vector (3D vector)
Returns:
- c2w: Camera-to-world matrix (4x4 numpy array)
The camera-to-world matrix is constructed such that the camera looks at the target point, with the specified up vector.
"""
def look_at(eye, center, up):
    """
    The normalized direction from the camera position (eye) to the target position (center).
    """
    forward = center - eye
    forward = forward / np.linalg.norm(forward)
    """
    Normalize the up vector to ensure it is a unit vector.
    """
    up = up / np.linalg.norm(up)
    """
    The right vector is perpendicular to the forward vector and the up vector.
    It is computed using the cross product of the forward and up vectors.
    """
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    """
    Re-compute the up vector to ensure it is perpendicular to the forward and right vectors.(for Orthogonality)
    """
    true_up = np.cross(right, forward)
    # Construct rotation: right, true_up, -forward (as columns)
    R = np.stack([right, true_up, -forward], axis=1)
    T = eye
    c2w = np.eye(4)
    c2w[:3,:3] = R
    c2w[:3, 3] = T
    return c2w

# Trajectory parameters
start = np.array([0, 0, 0])       # Camera starts here
end   = np.array([0, 0, 1])       # Camera ends here
target = np.array([0, 0, 2])      # Camera always looks at this point
up = np.array([0, 1, 0])          # Up direction

num_frames = 10

trajectory_c2w = []
for i in range(num_frames):
    t = i / (num_frames - 1)  # Value from 0 to 1
    cam_pos = (1 - t) * start + t * end
    c2w = look_at(cam_pos, target, up)
    trajectory_c2w.append(c2w)

# Print one of the matrices
print("First pose (c2w):\n", trajectory_c2w[0])
print("Last pose (c2w):\n", trajectory_c2w[-1])
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot trajectory points and look-at targets
trajectory = np.array([c2w[:3, 3] for c2w in trajectory_c2w])
ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], 'o-', color='blue', label="Camera Centers")

# Plot camera frustums/orientations
for c2w in trajectory_c2w:
    orig = c2w[:3, 3]
    forward = c2w[:3, 2]
    # Draw forward (view) direction
    ax.quiver(orig[0], orig[1], orig[2], -forward[0], -forward[1], -forward[2], 
              length=0.2, color='red')

ax.scatter([target[0]], [target[1]], [target[2]], c='black', marker='*', s=100, label='Target')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Camera trajectory from (0,0,0) to (0,0,1)')
ax.legend()
plt.show()