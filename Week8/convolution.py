import numpy as np


def convolve2d(input_array, kernel):
    n = input_array.shape[0]
    k = kernel.shape[0]
    output_size = n - k + 1
    output_array = np.zeros((output_size, output_size))

    for i in range(output_size):
        for j in range(output_size):
            output_array[i, j] = np.sum(input_array[i:i+k, j:j+k] * kernel)
    
    return output_array

