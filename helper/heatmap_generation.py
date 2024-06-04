import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter

image = cv2.imread('shirt_annotated/shirt1.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

lower_color = np.array([20, 100, 100])  # Lower bound of yellow
upper_color = np.array([30, 255, 255])  # Upper bound of yellow

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_image, lower_color, upper_color)

moments = cv2.moments(mask)
x = int(moments['m10'] / moments['m00']) if moments['m00'] else 0
y = int(moments['m01'] / moments['m00']) if moments['m00'] else 0

heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
if x != 0 and y != 0:  
    heatmap[y, x] = 255 

sigma = 32  # Increase the spread
smoothed_heatmap = gaussian_filter(heatmap, sigma)
smoothed_heatmap /= smoothed_heatmap.max()

heatmap_colored = cv2.applyColorMap(np.uint8(255 * smoothed_heatmap), cv2.COLORMAP_HOT)

final_image = np.zeros_like(image_rgb)
mask = smoothed_heatmap > 0.01 
final_image[mask] = heatmap_colored[mask]

cv2.imwrite('heatmap_train/heatmap_1.jpg', cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

plt.figure(figsize=(10, 8))
plt.imshow(final_image)
plt.axis('off')
plt.show()
