from PIL import Image
import numpy as np
import os
import glob

def invert_image(input_path, output_path):
    """
    Invert black and white colors in an image using PIL.
    Whites become black and blacks become white.
    """
    try:
        # Open the image
        image = Image.open(input_path)
        
        # Convert to grayscale if it's not already
        if image.mode != 'L':
            image = image.convert('L')
        
        # Convert to numpy array for easier manipulation
        img_array = np.array(image)
        
        # Invert the image (255 - pixel_value)
        inverted_array = 255 - img_array
        
        # Convert back to PIL Image
        inverted_image = Image.fromarray(inverted_array.astype(np.uint8))
        
        # Save the inverted image
        inverted_image.save(output_path)
        print(f"✓ Inverted: {os.path.basename(input_path)}")
        
    except Exception as e:
        print(f"✗ Error processing {input_path}: {e}")

def process_folder(input_folder, output_folder):
    """
    Process all PNG images in a folder and save inverted versions.
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    # Get all PNG files in the input folder
    png_files = glob.glob(os.path.join(input_folder, "*.png"))
    
    if not png_files:
        print(f"No PNG files found in {input_folder}")
        return
    
    print(f"Found {len(png_files)} PNG files to process...")
    print("-" * 50)
    
    # Process each image
    for input_file in png_files:
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_inverted.png")
        invert_image(input_file, output_file)
    
    print("-" * 50)
    print(f"Processing complete! Inverted images saved in: {output_folder}")

if __name__ == "__main__":
    input_folder = "edges"
    output_folder = "edges_inverted"
    
    if os.path.exists(input_folder):
        process_folder(input_folder, output_folder)
    else:
        print(f"Error: Folder {input_folder} not found!") 