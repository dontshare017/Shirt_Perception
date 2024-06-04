from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def annotate_convex(image_path, depth_path, modified_image_path):
    image = Image.open(image_path).convert('RGB')
    depth_img = Image.open(depth_path)

    annotations = [
        {"type": "point", "coordinates": (580, 864)},  # Goal point
        {"type": "polygon", "coordinates": [(2073, 1151), (2369, 1120), (2517, 992), (2598, 786), (2614, 2120), 
                                            (2489, 1949), (2377, 1887), (2209, 1809), (2108, 1735), (1867, 1521),
                                            (1859, 1498)]},  # Convex Hull
        {"type": "polyline", "coordinates": [(2066, 1148), (1855, 1505)]}  # Fold line 2

        # {"type": "polyline", "coordinates": [(2272, 1085), (2279, 1482)]},  # Fold line 1
        # {"type": "polyline", "coordinates": [(2120, 2011), (2276, 1494), (2283, 1482)]}  # Fold line 2
    ]

    draw = ImageDraw.Draw(image)
    for annotation in annotations:
        if annotation["type"] == "point":
            x, y = annotation["coordinates"]
            draw.ellipse((x-20, y-20, x+20, y+20), fill='green', outline='green', width=5)
        elif annotation["type"] == "polygon":
            draw.polygon(annotation["coordinates"], outline='blue', width=20)
        elif annotation["type"] == "polyline":
            draw.line(annotation["coordinates"], fill='yellow', width=20)

    retain_width = image.width - 1280
    start_width = 0
    image = image.crop((start_width, 0, retain_width, image.height))
    cropped_depth = depth_img.crop((start_width, 0, retain_width, depth_img.height))

    img_arr = np.array(cropped_depth, dtype=np.float32)
    min_val = np.min(img_arr)
    max_val = np.max(img_arr)
    normalized_arr = ((img_arr - min_val) / (max_val - min_val)) * 255
    normalized_depth = Image.fromarray(normalized_arr.astype(np.uint8)).convert('RGB')

    draw = ImageDraw.Draw(normalized_depth)
    for annotation in annotations:
        if annotation["type"] == "point":
            x, y = annotation["coordinates"]
            draw.ellipse((x-20, y-20, x+20, y+20), fill='green', outline='green', width=5)
        elif annotation["type"] == "polygon":
            draw.polygon(annotation["coordinates"], outline='blue', width=20)
        elif annotation["type"] == "polyline":
            draw.line(annotation["coordinates"], fill='yellow', width=20)

    # put RGB and depth images side by side
    combined_width = image.width + normalized_depth.width
    combined_height = max(image.height, normalized_depth.height)
    combined_image = Image.new('RGB', (combined_width, combined_height))
    combined_image.paste(image, (0, 0))
    combined_image.paste(normalized_depth, (image.width, 0))

    legend_items = [
        {"color": 'green', "description": "Goal point"},
        {"color": 'blue', "description": "Convex hull"},
        {"color": 'yellow', "description": "Fold lines"}
    ]
    font_size = 40 
    font_path = "/Users/YifeiHu/Downloads/montserrat/Montserrat-Medium.otf"
    font = ImageFont.truetype(font=font_path, size=font_size)
    legend_height = 500
    rectangle_size = (200, 150)  

    final_image = Image.new('RGB', (combined_image.width, combined_image.height + legend_height), 'white')
    final_image.paste(combined_image, (0, legend_height))
    draw = ImageDraw.Draw(final_image)

    legend_start_y = 10
    for idx, item in enumerate(legend_items):
        color = item["color"]
        description = item["description"]
        draw.rectangle([(combined_image.width - 10 - rectangle_size[0], legend_start_y + idx * (rectangle_size[1] + 10)), 
                        (combined_image.width - 10, legend_start_y + idx * (rectangle_size[1] + 10) + rectangle_size[1])], 
                       fill=color)
        draw.text((combined_image.width - 20 - rectangle_size[0] - 300, legend_start_y + idx * (rectangle_size[1] + 10)), 
                  description, fill='black', font=font)

    final_image.save(modified_image_path)




def annotate_convex_matplotlib(image_path, depth_path, annotations, modified_image_path):
    image = Image.open(image_path).convert('RGB')
    depth_img = Image.open(depth_path)

    retain_width = image.width - 1280
    start_width = 0
    image = image.crop((start_width, 0, retain_width, image.height))
    cropped_depth = depth_img.crop((start_width, 0, retain_width, depth_img.height))

    img_arr = np.array(cropped_depth, dtype=np.float32)
    min_val = np.min(img_arr)
    max_val = np.max(img_arr)
    if min_val != max_val: 
        normalized_arr = ((img_arr - min_val) / (max_val - min_val)) * 255
    else:
        normalized_arr = img_arr 
    normalized_depth = normalized_arr.astype(np.uint8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))

    ax1.imshow(image)
    ax2.imshow(normalized_depth, cmap='gray')

    for annotation in annotations:
        if annotation["type"] == "point":
            x, y = annotation["coordinates"]
            ax1.plot(x, y, 'go', markersize=10)  
        elif annotation["type"] == "polygon":
            poly_coords = np.array(annotation["coordinates"])
            ax1.plot(poly_coords[:, 0], poly_coords[:, 1], 'b-', linewidth=2)  
        elif annotation["type"] == "polyline":
            line_coords = np.array(annotation["coordinates"])
            ax1.plot(line_coords[:, 0], line_coords[:, 1], 'y-', linewidth=2) 

    legend_elements = [
        plt.Line2D([0], [0], color='g', marker='o', markersize=10, label='Goal point'),
        plt.Line2D([0], [0], color='b', linewidth=2, label='Convex hull'),
        plt.Line2D([0], [0], color='y', linewidth=2, label='Fold lines')
    ]

    ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(1.05, 1.2), ncol=1, fontsize=10, frameon=False)
    ax1.axis('off')
    ax2.axis('off')

    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    plt.savefig(modified_image_path, bbox_inches='tight')
    plt.show()




def annotate_points_and_lines(image_path, depth_path, annotations_path, modified_image_path):

    annotations_df = pd.read_csv(annotations_path)
    image_filename = image_path.split('/')[-1]
    image_annotations = annotations_df[annotations_df['filename'] == image_filename]

    if image_annotations.empty:
        print(f"No annotations found for {image_filename}")
        return

    image = Image.open(image_path).convert('RGB')
    depth_img = Image.open(depth_path).convert('L') 

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))

    ax1.imshow(image)
    ax2.imshow(depth_img, cmap='gray')

    for index, row in image_annotations.iterrows():
        x_start, y_start = row['x_start'], row['y_start']
        x_end, y_end = row['x_end'], row['y_end']
        ax1.plot(x_start, y_start, 'go', markersize=10)
        ax1.plot(x_end, y_end, 'bo', markersize=10) 
        ax1.plot([x_start, x_end], [y_start, y_end], 'y-', linewidth=2) 

    legend_elements = [
        plt.Line2D([0], [0], color='g', marker='o', markersize=10, label='Start point'),
        plt.Line2D([0], [0], color='b', marker='o', markersize=10, label='End point'),
    ]

    ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(1.05, 1.15), ncol=1, fontsize=10, frameon=False)
    ax1.axis('off')
    ax2.axis('off')

    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    plt.savefig(modified_image_path, bbox_inches='tight')
    plt.show()


image_path = '/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/downsampled_final/rgb/0-30/downsampled_rgb/T0-T9/RGB_53_T9.png'
depth_path = '/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/downsampled_final/depth/0-30/downsampled_depth/T0-T9/Depth_53_T9.png'
annotations_path = '/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/downsampled_final/depth/0-30/downsampled_depth/T0-T9/updated_annotations_T0-T9.csv'
modified_image_path = '/Users/YifeiHu/Downloads/graphed/annotated_image_with_points_and_lines.png'

annotate_points_and_lines(image_path, depth_path, annotations_path, modified_image_path)


image_path = '/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/recollection/rgb_1-35/RGB_114_T23.png'
depth_path = '/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/recollection/depth_1-35/Depth_114_T23.png'
annotations = [
    {"type": "point", "coordinates": (580, 864)},  # Goal point
    {"type": "polygon", "coordinates": [(2073, 1151), (2369, 1120), (2517, 992), (2598, 786), (2614, 2120), 
                                        (2489, 1949), (2377, 1887), (2209, 1809), (2108, 1735), (1867, 1521),
                                        (1859, 1498)]},  # Convex Hull
    {"type": "polyline", "coordinates": [(2066, 1148), (1855, 1505)]}  # Fold line 2
]
modified_image_path = '/Users/YifeiHu/Downloads/graphed/annotated_image.png'
annotate_convex_matplotlib(image_path, depth_path, annotations, modified_image_path)
