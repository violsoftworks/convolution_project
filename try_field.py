from PIL import Image
import numpy as np
image = Image.open('watermelon.png')
 
numpydata = np.asarray(image)
 
# <class 'numpy.ndarray'>
print(type(numpydata))
 
#  shape
print(numpydata)