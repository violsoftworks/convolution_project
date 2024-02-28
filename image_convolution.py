import numpy as np
from PIL import Image

load_image = Image.open('watermelon.png')
 
image = np.asarray(load_image)

class Convolution1D:
  def __init__(self, kernel):
    self.kernel = kernel

  def apply(self, signal):
    result = np.convolve(signal, self.kernel)
    return result

if __name__ == "__main__":
  signal = np.array([1, 2, 3])
  kernel = np.array([4, 5, 6])

  convolver = Convolution1D(kernel)
  convolved_signal = convolver.apply(signal)
  # print("Convolved Signal:", convolved_signal)


class Convolution2D:
    def __init__(self, kernel):
        self.kernel = kernel

    def convolve(self, image):
        # Get the dimensions of the image and the kernel
        xKernShape = self.kernel.shape[0]
        yKernShape = self.kernel.shape[1]
        xImgShape = image.shape[0]
        yImgShape = image.shape[1]

        # Create an output array with the same size as the image
        output = np.zeros_like(image)

        # Pad the borders of the input image
        pad = xKernShape // 2
        image_padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)))

        # Convolution operation
        for y in range(image.shape[1]):
            for x in range(image.shape[0]):
                for c in range(image.shape[2]):
                    output[x, y, c] = (self.kernel * image_padded[x: x + xKernShape, y: y + yKernShape, c]).sum()

        return output

kernel = np.array([[1, 0, 1],
                   [1, 2, 1],
                   [1, 3, 1]])

convolution = Convolution2D(kernel)

convolved_image = convolution.convolve(image)
# Normalize to 0-255
convolved_image = convolved_image - convolved_image.min()
convolved_image = convolved_image / convolved_image.max() * 255

# Convert to 8-bit unsigned integer format
convolved_image = convolved_image.astype(np.uint8)

# Convert the numpy array to an image
convolved_image_pil = Image.fromarray(convolved_image)

# Save the image
convolved_image_pil.save('watermelon_convolved.png')