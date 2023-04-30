# Locaal imports
from transformations import *

# External imports
import cv2
import matplotlib.pyplot as plt

def select_transformation(transformation_type=None):
    """
    Define the type of augmentation to apply to the images.
    
    Args:
        transformation_type (str): Type of augmentation to apply to the images.
    
    Returns:
        transformation (list): List of transformations to apply to the images.
    """
    if transformation_type is None:
        transformation = NoOpTransform()
        
    else:
        transformation = eval(transformation_type + "()")
    
    return [transformation]

def test_transformation(img_path="../TCIA_SegPC_dataset/coco/x/106.bmp", transformation_type=None):
    """
    Display the transformation on the image

    Args:
        img_path (str): Path to the image to be transformed.
        transformation_type (str): Type of augmentation to apply to the images.

    Returns:
        None
    """
    
    ## Read the image
    img = cv2.imread(img_path)
    
    ## Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    ## Get the image transformer
    transform_object = select_transformation(transformation_type)[0]
    
    ## APply transformation
    img_transformed = transform_object.apply_image(img)

    if transformation_type == "IlluminationSimulation":
        plt.figure(figsize=(10,10))
        for i in range(12):
            plt.subplot(3, 4, i+1)
            plt.imshow(img_transformed[...,i])
            plt.axis('off')

    else:
        ## Create a figure with two subplots in a single row
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

        ## Display the first image in the first subplot
        axs[0].imshow(img)
        axs[0].set_title('Original Image')

        ## Display the second image in the second subplot
        axs[1].imshow(img_transformed)
        axs[1].set_title('Transformed Image')

    ## Show the figure
    plt.show();