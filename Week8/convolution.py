import numpy as np


def convolve(input_array, kernel):
    n = input_array.shape[0]
    k = kernel.shape[0]
    output_size = n - k + 1
    output_array = np.zeros((output_size, output_size))

    for i in range(output_size):
        for j in range(output_size):
            sub_array = input_array[i:i+k, j:j+k]
            product = sub_array * kernel
            sum_product = np.sum(product)
            output_array[i, j] = sum_product
    
    return output_array

