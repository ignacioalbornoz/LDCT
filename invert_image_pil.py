from PIL import Image
import numpy as np
import os

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
        print(f"Image inverted successfully!")
        print(f"Original: {input_path}")
        print(f"Inverted: {output_path}")
        
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    input_file = "29030.png"
    output_file = "29030_inverted.png"
    
    if os.path.exists(input_file):
        invert_image(input_file, output_file)
    else:
        print(f"Error: File {input_file} not found!") 