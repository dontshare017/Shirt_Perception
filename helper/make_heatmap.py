import numpy as np
import cv2 as cv
from scipy.ndimage import gaussian_filter

def generate_heatmap(output_res, center, sigma):
    size = 6 * sigma + 3
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0, y0 = 3*sigma + 1, 3*sigma + 1
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    
    hms = np.zeros(output_res, dtype=np.float32)
    x, y = int(center[0]), int(center[1])
    ul = (int(x - 3*sigma - 1), int(y - 3*sigma - 1))
    br = (int(x + 3*sigma + 2), int(y + 3*sigma + 2))
    
    c, d = max(0, -ul[0]), min(br[0], output_res[1]) - ul[0]
    a, b = max(0, -ul[1]), min(br[1], output_res[0]) - ul[1]
    
    cc, dd = max(0, ul[0]), min(br[0], output_res[1])
    aa, bb = max(0, ul[1]), min(br[1], output_res[0])
    
    hms[aa:bb,cc:dd] = np.maximum(hms[aa:bb,cc:dd], g[a:b,c:d])
    return hms

def process(image_path, output_res=(256, 256), sigma=8):
    image = cv.imread(image_path)
    image = cv.resize(image, output_res)
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    lower_color = np.array([20, 100, 100])
    upper_color = np.array([30, 255, 255])
    mask = cv.inRange(hsv_image, lower_color, upper_color)
    
    moments = cv.moments(mask)
    center = (moments['m10'] / moments['m00'], moments['m01'] / moments['m00']) if moments['m00'] else (0, 0)
    
    heatmap = generate_heatmap(output_res, center, sigma)
    return image, heatmap

def save(image, heatmap, image_file_name, heatmap_file_name):
    cv.imwrite(image_file_name, image)
    heatmap_normalized = cv.normalize(heatmap, None, 0, 255, cv.NORM_MINMAX)
    heatmap_colored = cv.applyColorMap(np.uint8(heatmap_normalized), cv.COLORMAP_JET)
    cv.imwrite(heatmap_file_name, heatmap_colored)

image_path = 'shirt1.jpg'
image, heatmap = process(image_path, sigma=8)
save(image, heatmap, 'processed_image.jpg', 'heatmap.jpg')
