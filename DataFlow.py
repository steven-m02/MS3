import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load YOLOv5 for object detection
yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load MiDaS for depth estimation
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").eval()

# Load image transformation for MiDaS
transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

# Create a folder for processed images
output_dir = "./processed_images"
os.makedirs(output_dir, exist_ok=True)


def detect_pedestrians(img_path):
    """Detects pedestrians and estimates their distance."""

    # Read and convert the image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect objects using YOLO
    results = yolo(img_rgb, size=1024).pandas().xyxy[0]

    # Keep only pedestrians (label: 'person')
    pedestrians = results[results['name'] == 'person']

    # Prepare image for depth estimation
    img_resized = transform(cv2.resize(img_rgb, (256, 256))).to(torch.device('cpu'))

    # Get depth map
    with torch.no_grad():
        depth_map = midas(img_resized).squeeze().cpu().numpy()

    # Resize depth map to match original image
    depth_map = cv2.resize(depth_map, (img.shape[1], img.shape[0]))

    # Extract details for each pedestrian
    return [
        {"bbox": [int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])],
         "confidence": float(row['confidence']),
         "distance": round(np.mean(depth_map[int(row['ymin']):int(row['ymax']), int(row['xmin']):int(row['xmax'])]), 2)}
        for _, row in pedestrians.iterrows()
    ]


def process_images(folder):
    """Processes all images in a folder and saves the results."""

    for file in os.listdir(folder):
        # Process images that start with 'A' or 'C'
        if file.startswith(('A', 'C')) and file.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(folder, file)
            print(f"\nProcessing: {file}")

            # Detect pedestrians
            detections = detect_pedestrians(img_path)
            print(f"Results: {detections}")

            # Read image for drawing
            img = cv2.imread(img_path)
            for d in detections:
                x1, y1, x2, y2 = d['bbox']
                label = f"Person {d['confidence']:.2f}, {d['distance']}m"

                # Draw bounding box and label
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save the processed image
            output_path = os.path.join(output_dir, file)
            cv2.imwrite(output_path, img)
            print(f"Saved: {output_path}")

            # Show the image
            plt.figure(figsize=(10, 6))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()


if __name__ == "__main__":
    # Folder with images to process
    process_images("./Dataset_Occluded_Pedestrian/")
