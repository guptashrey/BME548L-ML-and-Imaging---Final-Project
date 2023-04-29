import numpy as np
import cv2
from detectron2.data.transforms import Transform, NoOpTransform

class CorrectColor(Transform):
    """
    Color correct the input image to RGB format
    """
    def __init__(self):
        super().__init__()

    def apply_image(self, img):
        ## Convert image to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def apply_coords(self, coords):
        # This transform does not modify the bounding box coordinates
        return coords

    def get_transform(self, image):
        # This transform does not depend on the input image
        return self

    
class GaussianBlur(Transform):
    """
    Apply a Gaussian blur to the input image.
    """
    def __init__(self, kernel_size=(15, 15), sigma=0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def apply_image(self, img):        
        ## apply Gaussian kernel
        img = cv2.GaussianBlur(img, self.kernel_size, self.sigma)
        return img

    def apply_coords(self, coords):
        # This transform does not modify the bounding box coordinates
        return coords

    def get_transform(self, image):
        # This transform does not depend on the input image
        return self

class ContrastNormalization(Transform):
    """
    Normalize the contrast of the input image.
    """
    def __init__(self, min_value=0, max_value=255):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def apply_image(self, img):
        # Normalize the contrast of the image using linear scaling
        img = (self.max_value - self.min_value) * (img - img.min()) / (img.max() - img.min()) + self.min_value
        return img

    def apply_coords(self, coords):
        # This transform does not modify the bounding box coordinates
        return coords

    def get_transform(self, image):
        # This transform does not depend on the input image
        return self

class Dilation(Transform):
    """
    Perform dilation on the input image.
    """
    def __init__(self, kernel_size=9, iterations=1):
        super().__init__()
        self.radius = 5
        self.kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        #self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*self.radius+1, 2*self.radius+1))
        #self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.iterations = iterations

    def apply_image(self, img):
        # Perform dilation on the input image
        img = cv2.dilate(img, self.kernel, iterations=self.iterations)
        return img

    def apply_coords(self, coords):
        # This transform does not modify the bounding box coordinates
        return coords

    def get_transform(self, image):
        # This transform does not depend on the input image
        return self
    
class Erosion(Transform):
    """
    Perform dilation on the input image.
    """
    def __init__(self, kernel_size=9, iterations=1):
        super().__init__()
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.iterations = iterations

    def apply_image(self, img):
        # Perform dilation on the input image
        img = cv2.erode(img, self.kernel, iterations=1)
        return img

    def apply_coords(self, coords):
        # This transform does not modify the bounding box coordinates
        return coords

    def get_transform(self, image):
        # This transform does not depend on the input image
        return self

class SobelFilterX(Transform):
    """
    Perform dilation on the input image.
    """
    def __init__(self, kernel_size=1, iterations=1):
        super().__init__()
        self.kernel_size = kernel_size

    def apply_image(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Sobel filter
        img = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=self.kernel_size)        
        return img

    def apply_coords(self, coords):
        # This transform does not modify the bounding box coordinates
        return coords

    def get_transform(self, image):
        # This transform does not depend on the input image
        return self
    
class SobelFilterY(Transform):
    """
    Perform dilation on the input image.
    """
    def __init__(self, kernel_size=1, iterations=1):
        super().__init__()
        self.kernel_size = kernel_size

    def apply_image(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Sobel filter
        img = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=self.kernel_size)       
        return img

    def apply_coords(self, coords):
        # This transform does not modify the bounding box coordinates
        return coords

    def get_transform(self, image):
        # This transform does not depend on the input image
        return self
    
class EnhanceRedColor(Transform):
    """
    Perform dilation on the input image.
    """
    def __init__(self):
        super().__init__()

    def apply_image(self, img):
        b, g, r = cv2.split(img)

        # Enhance the red channel
        r = cv2.equalizeHist(r)
        
        mask = np.zeros_like(img)
        mask[:,:,2] = np.ones_like(r)  # Keep the red channel
        img = cv2.merge((img * mask).astype(np.uint8))
        
        return img

    def apply_coords(self, coords):
        # This transform does not modify the bounding box coordinates
        return coords

    def get_transform(self, image):
        # This transform does not depend on the input image
        return self

class EnhanceGreenColor(Transform):
    """
    Enhance the green color channel of the input image.
    """
    def __init__(self):
        super().__init__()

    def apply_image(self, img):
        b, g, r = cv2.split(img)

        # Enhance the green channel
        g = cv2.equalizeHist(g)
        
        mask = np.zeros_like(img)
        mask[:,:,1] = np.ones_like(g)  # Keep the green channel
        img = cv2.merge((img * mask).astype(np.uint8))
        
        return img

    def apply_coords(self, coords):
        # This transform does not modify the bounding box coordinates
        return coords

    def get_transform(self, image):
        # This transform does not depend on the input image
        return self


class EnhanceBlueColor(Transform):
    """
    Enhance the blue color channel of the input image.
    """
    def __init__(self):
        super().__init__()

    def apply_image(self, img):
        b, g, r = cv2.split(img)

        # Enhance the blue channel
        b = cv2.equalizeHist(b)
        
        mask = np.zeros_like(img)
        mask[:,:,0] = np.ones_like(b)  # Keep the blue channel
        img = cv2.merge((img * mask).astype(np.uint8))
        
        return img

    def apply_coords(self, coords):
        # This transform does not modify the bounding box coordinates
        return coords

    def get_transform(self, image):
        # This transform does not depend on the input image
        return self