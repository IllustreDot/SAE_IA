from Predict import predict_and_save
import os

def process_images_in_folder():
    """
    Process all images in a folder using the predict_and_save function.
    """
    model_path = "model.pth"
    img_folder_path = "img_to_predict"
    label_to_class_path = "label_to_class.json"
    output_folder_path = "predictions"
    # Check if the output folder exists, create it if it doesn't
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # List all image files in the directory
    image_files = [f for f in os.listdir(img_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Iterate over each image and make predictions
    for img_file in image_files:
        img_path = os.path.join(img_folder_path, img_file)
        output_prediction_path = os.path.join(output_folder_path, f"{os.path.splitext(img_file)[0]}_prediction.json")
        
        # Call the prediction function for each image
        predict_and_save(model_path, img_path, label_to_class_path, output_prediction_path)

if __name__ == "__main__":
    process_images_in_folder()
    print("All images processed and predictions saved.")