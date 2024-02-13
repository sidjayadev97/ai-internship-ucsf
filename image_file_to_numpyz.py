import numpy as np

# library to open .png file
from PIL import Image 

import pydicom

from skimage import data, color
from skimage.transform import rescale, resize


# Part 4) function below:
# This function will check the extension of a file (either DICOM or PNG), 
# read the file, and then convert it into a NumpyZ compressed file 

def image_file_to_numpyz(filename, label_category, parent_folder, destination_name, imgsize):
    # Check file extension
    extension = filename.split('.')[-1]
    
    if extension == 'dcm':
        # Read data
        data = pydicom.dcmread(parent_folder+filename)
        dicom_array = data.pixel_array
        
        # Resize the Numpy array
        resized_array = resize(dicom_array, (imgsize[0], imgsize[1]),
                       anti_aliasing=True)
        
        # save NumpZ file
        np.savez_compressed(destination_name, a=resized_array, b=label_category)
        
    elif extension == 'png':
        # Read Data
        data = Image.open(parent_folder+filename)
        png_array = np.asarray(data)
        
        # Resize the Numpy array
        resized_array = resize(png_array, (imgsize[0], imgsize[1]),
                       anti_aliasing=True)
        
        # save NumpZ file
        np.savez_compressed(destination_name, a=resized_array, b=label_category)
        
    elif extension == 'jpg':
        # Read Data
        data = Image.open(parent_folder+filename)
        jpg_array = np.asarray(data)
        
        # Resize the Numpy array
        resized_array = resize(jpg_array, (imgsize[0], imgsize[1]),
                       anti_aliasing=True)
        
        # save NumpZ file
        np.savez_compressed(destination_name, a=resized_array, b=label_category)

