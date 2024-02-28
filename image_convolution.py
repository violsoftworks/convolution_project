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
        kernel = np.flipud(np.fliplr(self.kernel))  # Flip the kernel
        kernel_height, kernel_width = kernel.shape
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
        
        image_height, image_width = image.shape
        output = np.zeros_like(image)
        for y in range(image_height):
            for x in range(image_width):
                output[y, x] = np.sum(padded_image[y:y+kernel_height, x:x+kernel_width] * kernel)
        return output
kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])

convolution = Convolution2D(kernel)

convolved_image = convolution.convolve(image)

print("Original Image:")
print(image)
print("\nConvolved Image:")
print(convolved_image)