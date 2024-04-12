import torch
import torchvision
import cv2

# Load pre-trained object detection model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=False)

# Load pre-trained weights for COCO dataset
model.load_state_dict(torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).state_dict())

model.eval()

# Define a function to perform object detection on an image
def detect_people(image):
    with torch.no_grad():
        image_tensor = torch.tensor(image).permute(2, 0, 1).float().unsqueeze(0)
        predictions = model(image_tensor)
    
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    
    # Filter out detections that are people (label 1)
    person_boxes = boxes[labels == 1]
    
    return person_boxes

# Define a function to calculate population density
def calculate_density(boxes, image_width, image_height):
    # Calculate population density based on the number of people detected
    population_density = len(boxes) / (image_width * image_height)
    return population_density

# Main function to process single image
def monitor_street_image(image_path, safe_limit, resize_width=640, resize_height=480):
    # Read the image
    image = cv2.imread(image_path)
    
    if image is None:
        print("Failed to read the image.")
        return
    
    # Resize the image
    image = cv2.resize(image, (resize_width, resize_height))
    
    image_height, image_width, _ = image.shape
    
    # Detect people in the image
    person_boxes = detect_people(image)
    
    # Calculate population density
    density = calculate_density(person_boxes, image_width, image_height)
    
    # Check if population density exceeds safe limit
    if density > safe_limit:
        print("Population density exceeds safe limit!")
    else:
        print("Population density is within safe limit.")
    
    # Display the image with detections
    for box in person_boxes:
        start_point = (int(box[0]), int(box[1]))
        end_point = (int(box[2]), int(box[3]))
        color = (0, 255, 0)  # Green color for bounding box
        thickness = 2
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
    
    cv2.imshow('Street Monitoring', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Set your image file path
    image_path = r"C:\Users\DELL\OneDrive\Desktop\population ndensity\crowd1.jpg"
    
    # Set the safe limit for population density
    safe_limit = 0.0# Example safe limit
    
    # Set the desired resize dimensions
    resize_width = 740
    resize_height = 680
    
    monitor_street_image(image_path, safe_limit, resize_width, resize_height)
