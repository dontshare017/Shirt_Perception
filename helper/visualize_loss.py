import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.spatial import distance
from PIL import Image

def inside_polygon(x, y, polygon):
    return Path(polygon).contains_point((x, y))

def compute_distance_to_line(x, y, line):
    # Calculate the minimum distance from (x, y) to any segment of the polyline
    dist = np.inf
    for i in range(len(line) - 1):
        seg_dist = distance_point_to_segment(x, y, line[i], line[i+1])
        if seg_dist < dist:
            dist = seg_dist
    return dist

def distance_point_to_segment(px, py, seg_start, seg_end):
    
    # Vector from start to point
    start_to_point = np.array([px - seg_start[0], py - seg_start[1]])
    # Vector from start to end
    start_to_end = np.array([seg_end[0] - seg_start[0], seg_end[1] - seg_start[1]])
    # Project point onto the line segment
    projection = max(0, min(1, np.dot(start_to_point, start_to_end) / np.dot(start_to_end, start_to_end)))
    # Find the closest point on the segment
    closest = seg_start + projection * start_to_end
    # Return the distance from the point to this closest point
    return np.linalg.norm(closest - np.array([px, py]))

# Assuming knowledge to an optimal distance, d_opt, in pixel space
def compute_loss(image_shape, goal_point, convex_hull, fold_line, d_opt=30):
    height, width = image_shape
    loss_grid = np.zeros((height, width))
    
    for x in range(width):
        for y in range(height):
            if inside_polygon(x, y, convex_hull):
                distance_to_fold_line = compute_distance_to_line(x, y, fold_line)
                # Prioritize selecting a push point on the goal side
                distance_to_goal = np.linalg.norm(np.array([x, y]) - np.array(goal_point))
                loss_fold_line = (distance_to_fold_line - d_opt)**2
                loss_goal = distance_to_goal
                loss_grid[y, x] = loss_fold_line + loss_goal
            else:
                loss_grid[y, x] = 10000
    
    return loss_grid

image_path = '/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/recollection/rgb_1-35/RGB_114_T23.png' 
image = Image.open(image_path)
image_shape = image.size[::-1]  # (height, width)

annotations = [
    {"type": "point", "coordinates": (580, 864)},
    {"type": "polygon", "coordinates": [(2073, 1151), (2369, 1120), (2517, 992), (2598, 786), (2614, 2120), 
                                        (2489, 1949), (2377, 1887), (2209, 1809), (2108, 1735), (1867, 1521),
                                        (1859, 1498)]}, 
    {"type": "polyline", "coordinates": [(2066, 1148), (1855, 1505)]} 
]
goal_point = annotations[0]["coordinates"]
convex_hull = annotations[1]["coordinates"]
fold_line = annotations[2]["coordinates"]

loss_grid = compute_loss(image_shape, goal_point, convex_hull, fold_line)
normalized_loss = (loss_grid - np.min(loss_grid)) / (np.max(loss_grid) - np.min(loss_grid))

fig, ax = plt.subplots()
ax.imshow(image) 
ax.imshow(normalized_loss, cmap='hot', alpha=0.5, interpolation='nearest') 
ax.set_title("Loss Heatmap Overlay")
plt.gca().invert_yaxis()
plt.show()
