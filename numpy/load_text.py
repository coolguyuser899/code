from __future__ import print_function
from numpy import loadtxt

# load 2 x 2 array
array1 = loadtxt('float_data.txt')
print('example 1:')
print(array1)

# skip first row
array2 = loadtxt('float_data_with_header.txt', skiprows=1)
print('example 2:')
print(array2)

# skips comments and columns
array3 = loadtxt("complex_data_file.txt", delimiter=",", comments="%", usecols=(0, 1, 2, 4), dtype=int, skiprows=1)
print('example 3:')
print(array3)



