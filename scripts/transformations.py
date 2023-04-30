# Library imports
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
        """
        Color correct the input image to RGB format

        Args:
            img (np.array): Input image

        Returns:
            np.array: Color corrected image
        """
        ## Convert image to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def apply_coords(self, coords):
        """
        This transform does not modify the bounding box coordinates
        """
        # This transform does not modify the bounding box coordinates
        return coords

    def get_transform(self, image):
        """
        This transform does not depend on the input image
        """
        # This transform does not depend on the input image
        return self

    
class GaussianBlur(Transform):
    """
    Apply a Gaussian blur to the input image.
    """
    def __init__(self, kernel_size=(15, 15), sigma=0.0):
        """
        Args:
            kernel_size (tuple): Kernel size of the Gaussian filter.
            sigma (float): Standard deviation of the Gaussian filter.
        
        Returns:
            None
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def apply_image(self, img): 
        """
        Apply a Gaussian blur to the input image.

        Args:
            img (np.array): Input image

        Returns:
            np.array: Image with Gaussian blur applied
        """       
        ## apply Gaussian kernel
        img = cv2.GaussianBlur(img, self.kernel_size, self.sigma)
        return img

    def apply_coords(self, coords):
        """
        This transform does not modify the bounding box coordinates
        """
        # This transform does not modify the bounding box coordinates
        return coords

    def get_transform(self, image):
        """
        This transform does not depend on the input image
        """
        # This transform does not depend on the input image
        return self

class ContrastNormalization(Transform):
    """
    Normalize the contrast of the input image.
    """
    def __init__(self, min_value=0, max_value=255):
        """
        Args:
            min_value (int): Minimum value of the normalized image.
            max_value (int): Maximum value of the normalized image.
        
        Returns:    
            None
        """
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def apply_image(self, img):
        """
        Normalize the contrast of the input image.

        Args:
            img (np.array): Input image

        Returns:
            np.array: Image with normalized contrast
        """
        # Normalize the contrast of the image using linear scaling
        img = (self.max_value - self.min_value) * (img - img.min()) / (img.max() - img.min()) + self.min_value
        img = (img*255).astype(np.int)
        return img

    def apply_coords(self, coords):
        """
        This transform does not modify the bounding box coordinates
        """
        # This transform does not modify the bounding box coordinates
        return coords

    def get_transform(self, image):
        """
        This transform does not depend on the input image
        """
        # This transform does not depend on the input image
        return self

class Dilation(Transform):
    """
    Perform dilation on the input image.
    """
    def __init__(self, kernel_size=9, iterations=1):
        """
        Args:
            kernel_size (int): Kernel size of the dilation filter.
            iterations (int): Number of times to apply the dilation filter. 

        Returns:    
            None
        """
        super().__init__()
        self.radius = 5
        self.kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        self.iterations = iterations

    def apply_image(self, img):
        """
        Perform dilation on the input image.

        Args:
            img (np.array): Input image

        Returns:    
            np.array: Image with dilation applied
        """
        # Perform dilation on the input image
        img = cv2.dilate(img, self.kernel, iterations=self.iterations)
        return img

    def apply_coords(self, coords):
        """
        This transform does not modify the bounding box coordinates
        """
        # This transform does not modify the bounding box coordinates
        return coords

    def get_transform(self, image):
        """
        This transform does not depend on the input image
        """
        # This transform does not depend on the input image
        return self
    
class Erosion(Transform):
    """
    Perform dilation on the input image.
    """
    def __init__(self, kernel_size=9, iterations=1):
        """
        Args:
            kernel_size (int): Kernel size of the dilation filter.
            iterations (int): Number of times to apply the dilation filter. 

        Returns:
            None
        """
        super().__init__()
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.iterations = iterations

    def apply_image(self, img):
        """
        Perform dilation on the input image.

        Args:
            img (np.array): Input image
        
        Returns:
            np.array: Image with dilation applied
        """
        # Perform dilation on the input image
        img = cv2.erode(img, self.kernel, iterations=1)
        return img

    def apply_coords(self, coords):
        """
        This transform does not modify the bounding box coordinates
        """
        # This transform does not modify the bounding box coordinates
        return coords

    def get_transform(self, image):
        """
        This transform does not depend on the input image
        """
        # This transform does not depend on the input image
        return self
    
class IlluminationSimulation(Transform):
    """
    Simulate the illumination of the input image.
    """
    def __init__(self):
        """
        Args:   
            None

        Returns:
            None
        """
        super().__init__()
        self.wavelength = .5e-3 # units are mm; assuming green light
        delta_x = 0.5*self.wavelength # let's sample at nyquist rate
        num_samples_x = 1440 # number of pixels in x direction
        num_samples_y = 1080 # number of pixels in y direction
        
        # Define the spatial coordinates of the sample
        starting_coordinate_x = (-num_samples_x/2) * delta_x
        ending_coordinate_x = (num_samples_x/2 - 1) * delta_x

        starting_coordinate_y = (-num_samples_y/2) * delta_x
        ending_coordinate_y = (num_samples_y/2 - 1) * delta_x

        # make linspace, meshgrid as needed for sample
        x = np.linspace(starting_coordinate_x, ending_coordinate_x, num=num_samples_x)
        y = np.linspace(starting_coordinate_y, ending_coordinate_y, num=num_samples_y)
        self.xx, self.yy = np.meshgrid(x, y)

        # define total range of spatial frequency axis, 1/mm
        f_range = int(1/delta_x)
        delta_fx_x = f_range/num_samples_x
        delta_fy_y = f_range/num_samples_y

        # make linspace, meshgrid as needed for lens transfer function
        starting_coordinate_fx = (-num_samples_x/2) * delta_fx_x
        ending_coordinate_fx = (num_samples_x/2 - 1) * delta_fx_x
        starting_coordinate_fy = (-num_samples_y/2) * delta_fy_y
        ending_coordinate_fy = (num_samples_y/2 - 1) * delta_fy_y

        # make linspace, meshgrid as needed for lens transfer function
        xf = np.linspace(starting_coordinate_fx, ending_coordinate_fx, num=num_samples_x)
        yf = np.linspace(starting_coordinate_fy, ending_coordinate_fy, num=num_samples_y)
        xxf, yyf = np.meshgrid(xf, yf)

        # Define lens numerical aperture as percentage of total width of spatial freqeuncy domain
        # Let's make the lens transfer function diameter 1/4th the total spatial frequency axis coordinates.
        d =int((ending_coordinate_fx - starting_coordinate_fx+1) / 4)
        r = d/2

        # Define lens transfer function as matrix with 1's within desired radius, 0's outside
        self.trans = np.zeros((num_samples_y, num_samples_x))
        dist = np.sqrt((xxf)**2+(yyf)**2)
        self.trans[np.where(dist<r)]=1

        foo = np.array([[-15, 10], [-5, 10], [5, 10],[15,10], [-15, 0], [-5, 0], [5, 0], [15,0], [-15, -10], [-5, -10], [5, -10], [15,-10]])
        self.plane_wave_angle_xy = ((foo/15) * 12) * np.pi/180        

    def apply_image(self, img):   
        """
        Simulate the illumination of the input image.

        Args:
            img (np.array): Input image

        Returns:
            np.array: Illuminated image
        """     
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        img_convert = self.convert_images(img_gray)
        
        # generate illumination dataset
        illumination_data = np.zeros((1080, 1440, 12))
        for i, plane_wave_angle in enumerate(self.plane_wave_angle_xy):

            # Define plane waves
            illumination_plane_wave = np.exp(1j*2*np.pi/self.wavelength * (np.sin(plane_wave_angle[0]) * self.xx + np.sin(plane_wave_angle[1]) * self.yy))

            # Define field emerging from sample
            emerging_field = np.multiply(illumination_plane_wave, img_convert)
            
            # Take 2D fourier transform of sample
            fourier_field = np.fft.fftshift(np.fft.fft2(emerging_field))
            
            # Create filtered sample spectrum with center crop (64 x 64)
            # trans: only within desired radius is 1
            # so we can crop the outer part
            fourier_field_trans = np.multiply(fourier_field, self.trans)
            
            # 0.5 point
            # Propagate filtered sample spectrum to image plane
            inverse_fourier_field = np.fft.ifft2(np.fft.ifftshift(fourier_field_trans))
            
            # 0.5 point
            # save the intensity of inverse_fourier_field
            illumination_data[:, :,i] = np.square(np.abs(inverse_fourier_field))

            illumination_data = illumination_data.astype(np.uint16)
            foo = cv2.convertScaleAbs(illumination_data, alpha=(255.0/65535.0))

        return foo

    def convert_images(self, sample_amplitude):
        """
        in real world, microscope samples are 3D and have thickness, which introduce a phase shift to the optical field
        For simplicity, let's further assume the sample thickness and amplitude are inversely correlated, which means the thicker the sample is,
        the more light it absorb.
        """
        sample_phase = 1 - sample_amplitude
        optical_thickness = 0.02 * self.wavelength
        return sample_amplitude * np.exp(1j * sample_phase*optical_thickness/self.wavelength)

    def apply_coords(self, coords):
        # This transform does not modify the bounding box coordinates
        return coords

    def get_transform(self, image):
        # This transform does not depend on the input image
        return self
    
class GrayscaleTransform(Transform):
    """
    Convert the input image to grayscale.
    """
    def __init__(self):
        pass
    
    def apply_image(self, img):
        """
        Convert the input image to grayscale.

        Args:
            img (np.array): Input image

        Returns:
            np.array: Grayscale image
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        img = np.expand_dims(img, axis=2)
        return img
    
    def apply_coords(self, coords):
        """
        This transform does not modify the bounding box coordinates
        """
        # This transform does not modify the bounding box coordinates
        return coords

    def get_transform(self, image):
        """
        This transform does not depend on the input image
        """
        # This transform does not depend on the input image
        return self
    
class SobelFilter(Transform):
    """
    Apply Sobel filter on the input image.
    """
    def __init__(self):
        super().__init__()

    def apply_image(self, img):
        """
        Apply Sobel filter on the input image.  

        Args:   
            img (np.array): Input image

        Returns:
            np.array: Image after Sobel filter
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
        sobel = np.clip(sobel, 0, 255).astype(np.uint8)
        sobel = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
        return sobel

    def apply_coords(self, coords):
        """
        This transform does not modify the bounding box coordinates
        """
        # This transform does not modify the bounding box coordinates
        return coords

    def get_transform(self, image):
        """
        This transform does not depend on the input image
        """
        # This transform does not depend on the input image
        return self


class EnhanceGreenColor(Transform):
    """
    Enhance the green channel of an image.
    """
    def __init__(self):
        super().__init__()

    def apply_image(self, img):
        """
        Enhance the green channel of an image.

        Args:   
            img (np.array): Input image

        Returns:
            np.array: Image with enhanced green channel
        """
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Split the image
        b, g, r = cv2.split(img)

        # Enhance the green channel
        g = cv2.equalizeHist(g)

        img = cv2.merge((b, g, r))

        return img

    def apply_coords(self, coords):
        """
        This transform does not modify the bounding box coordinates
        """
        # This transform does not modify the bounding box coordinates
        return coords

    def get_transform(self, image):
        """
        This transform does not depend on the input image
        """
        # This transform does not depend on the input image
        return self

class EnhanceRedColor(Transform):
    """
    Enhance the red channel of an image.
    """
    def __init__(self):
        super().__init__()

    def apply_image(self, img):
        """
        Enhance the red channel of an image. 
    
        Args:
            img (np.array): Input image
        
        Returns:
            np.array: Image with enhanced red channel
        """
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Split the image
        b, g, r = cv2.split(img)

        # Enhance the red channel
        r = cv2.equalizeHist(r)

        img = cv2.merge((b, g, r))

        return img

    def apply_coords(self, coords):
        """
        This transform does not modify the bounding box coordinates
        """
        # This transform does not modify the bounding box coordinates
        return coords

    def get_transform(self, image):
        """
        This transform does not depend on the input image
        """
        # This transform does not depend on the input image
        return self
    
class EnhanceBlueColor(Transform):
    """
    Enhance the blue channel of an image.
    """
    def __init__(self):
        super().__init__()

    def apply_image(self, img):
        """
        Enhance the blue channel of an image.   

        Args:   
            img (np.array): Input image

        Returns:    
            np.array: Image with enhanced blue channel
        """
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Split the image
        b, g, r = cv2.split(img)

        # Enhance the blue channel
        b = cv2.equalizeHist(b)

        # Merge the channels
        img = cv2.merge((b, g, r))

        return img

    def apply_coords(self, coords):
        """
        This transform does not modify the bounding box coordinates
        """
        # This transform does not modify the bounding box coordinates
        return coords

    def get_transform(self, image):
        """
        This transform does not depend on the input image
        """
        # This transform does not depend on the input image
        return self

class MedianFilter(Transform):
    """
    Apply median filtering with a kernel size of 5x5 to an image.
    """
    def __init__(self):
        super().__init__()

    def apply_image(self, img):
        img = cv2.medianBlur(img, 15)
        return img

    def apply_coords(self, coords):
        """
        This transform does not modify the bounding box coordinates
        """
        # This transform does not modify the bounding box coordinates
        return coords

    def get_transform(self, image):
        """
        This transform does not depend on the input image
        """
        # This transform does not depend on the input image
        return self