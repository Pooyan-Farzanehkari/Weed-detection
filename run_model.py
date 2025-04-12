from ultralytics import YOLO
import cv2
import numpy as np
import os

def main():
    # Load the model
    model = YOLO('best.pt')
    
    # Use the first image in the directory
    image_path = '33035_jpg.rf.3e4348af760ecc9bbae1e89beff0090c.jpg'
    
    # Run inference
    results = model(image_path)
    
    # Process and save results
    for result in results:
        # Get the image with detections
        img = result.plot()
        
        # Save the image with detections
        output_path = 'results/detection_result.jpg'
        cv2.imwrite(output_path, img)
        print(f"\nDetection results saved to: {os.path.abspath(output_path)}")
        
        # Print detection results
        print("\nDetection Results:")
        print(f"Detected {len(result.boxes)} objects")
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            print(f"Class: {class_name}, Confidence: {confidence:.2f}, Coordinates: {box.xyxy[0].tolist()}")

if __name__ == "__main__":
    main() 
    