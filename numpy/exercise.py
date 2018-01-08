# the code of this folder is from - https://github.com/enthought/Numpy-Tutorial-SciPyConf-2017

import matplotlib as plt

test_list = list(range(1000))

#create numpy array
import numpy as np

test_array = np.arange(1000)

import timeit

# start_time = timeit.default_timer()
# sum(test_list)
# print(timeit.default_timer() - start_time)
# #1.4813034795224667e-05
#
# #
# start_time = timeit.default_timer()
# np.sum(test_array)
# print(timeit.default_timer() - start_time)
# #0.00012267997954040766

# normal python array is address pointer to value, one lookup is added when looking for value
# numpy has memory, dense array containing values itself
# number and data structure are different in numpy and python, for performan e
# numpy array has same data type, fixed size (different than python list, resizable)

# a = np.array([-1, 0, 1, 100], dtype='int8')

# a / 0   #get warning, but not error
#RuntimeWarning: divide by zero encountered in true_divide

# a ** 2
# print(a ** 2)   #[ 1  0  1 16]
#
# b = a.astype('float32')
# print(b, b/0)   #[  -1.    0.    1.  100.] [-inf  nan  inf  inf]

# print(np.nan == np.nan)     #False
# print(np.isnan(np.nan))     #True
#
# np.zeros
# np.ones
# np.emyty
# np.emytp((2,2))

# print(a[0], a[-1], a[0:2], a[:2], a[::2])  #slicing, step size
# -1 100 [-1  0] [-1  0] [-1  1]
# a[-1] = 5   #replace value
# print(a)

# b = np.arange(12).reshape(4, 3) #create 1 dem array, change to 2 dem array 4 rows and 3 columns
# print(b)

# v[[ 0  1  2]
# [ 3  4  5]
# [ 6  7  8]
# [ 9 10 11]]

# print(a.shape, b.shape, b[2,2], b[:2, :2], b[1:3, -1]) # check array size

# (4,) (4, 3)
# 8
# [[0 1]
# [3 4]]
# [5 8]

# print(b[:1, :1])    #get first item
# np.loadtxt(skiprows=1)  # read file

# c = np.arange(24).reshape(2, 3,4)
# print(c, c[1, 1, 1], c[0, :, :], c[1, 0, :])        #17
# [[[ 0  1  2  3]
#   [ 4  5  6  7]
#  [ 8  9 10 11]]
#
# [[12 13 14 15]
#  [16 17 18 19]
# [20 21 22 23]]]
# 17
# [[ 0  1  2  3]
# [ 4  5  6  7]
# [ 8  9 10 11]]
# [12 13 14 15]

# change multiple dim array to one dim array
# print(c.flatten())
#[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]

# a = np.arange(25).reshape(5, 5)
# print(a[4, :], a[-1, :], a[:, 1::2], a[1::2, :3:2])    #step size 2
# a[1::2, :3:2] row from 1 to end, step =2; from begin to 3, step 2
# [20 21 22 23 24] [20 21 22 23 24] [[ 1  3]
#                                    [ 6  8]
# [11 13]
# [16 18]
# [21 23]] [[ 5  7]
#           [15 17]]

# numpy has data in one location, lookup instant time
# a = np.arange(4)    #array([0,1,2,3])
# a[0]
# a[[0,1,3]]
# b[[0,2], [2,0]]     # array([2, 6])
# c[[0, 1], [1, 1], [2, 1]]    # array([6, 17])

# print(c > 16)
# [[[False False False False]
#   [False False False False]
#  [False False False False]]
#
# [[False False False False]
#  [False  True  True  True]
# [ True  True  True  True]]]
#

# d = c[:, 1:2]   #this does not create new object, but create pointer to link to c
# print(d.flags)
# C_CONTIGUOUS : False
# F_CONTIGUOUS : False
# OWNDATA : False         # False
# WRITEABLE : True
# ALIGNED : True
# UPDATEIFCOPY : False

# d[0,0,0] = 100  #this changes value in c as well

# e = c[c > 16]
# e.flags # fancy indexing
# print(e.flags)
# [[[False False False False]
#   [False False False False]
#  [False False False False]]
#
# [[False False False False]
#  [False  True  True  True]
# [ True  True  True  True]]]
# C_CONTIGUOUS : True
# F_CONTIGUOUS : True
# OWNDATA : True
# WRITEABLE : True
# ALIGNED : True
# UPDATEIFCOPY : False

# print(c.strides)
# (96, 32, 8)

# print(e.T) # transpose operator T
# [17 18 19 20 21 22 23]

a = np.arange(25).reshape(5,5)
# print(a, a[[0, 1, 2, 3], [1, 2, 3, 4]], a % 3, a % 3 == 0, a[a % 3 == 0])     # fancy indexing, modular operator
# [[ 0  1  2  3  4]
#  [ 5  6  7  8  9]
# [10 11 12 13 14]
# [15 16 17 18 19]
# [20 21 22 23 24]] [ 1  7 13 19] [[0 1 2 0 1]
# [2 0 1 2 0]
# [1 2 0 1 2]
# [0 1 2 0 1]
# [2 0 1 2 0]] [[ True False False  True False]
#               [False  True False False  True]
# [False False  True False False]
# [ True False False  True False]
# [False  True False False  True]] [ 0  3  6  9 12 15 18 21 24]
#

# 1:51
# dow.shape
# dow[dow > 5.5e9]        # volumn 5 billion
# dow[:, 4] > 5.5.e9      #
# mask = dow[:, 4]
# np.sum(mask)
# np.where(mask)

# print(a + 5, a + a, a + np.arange(5), a + np.arange(5).reshape(1, 5))     # shape is important for mapping
# a + np.arange(7)    # this fails

# print(np.arange(5).reshape(5,1) + np.arange(5))     # expand array
# [[0 1 2 3 4]
#  [1 2 3 4 5]
# [2 3 4 5 6]
# [3 4 5 6 7]
# [4 5 6 7 8]]

# a = a.astype('float64')
# a[2,3] = np.nan
# print(a, a + 5, np.sum(a), np.sum(a, axis=1))     # add 5 to everything, but nan is not changed
# [[0 1 2 3 4]
#  [1 2 3 4 5]
# [2 3 4 5 6]
# [3 4 5 6 7]
# [4 5 6 7 8]]
# [[  0.   1.   2.   3.   4.]
#  [  5.   6.   7.   8.   9.]
# [ 10.  11.  12.  nan  14.]
# [ 15.  16.  17.  18.  19.]
# [ 20.  21.  22.  23.  24.]] [[  5.   6.   7.   8.   9.]
#                              [ 10.  11.  12.  13.  14.]
# [ 15.  16.  17.  nan  19.]
# [ 20.  21.  22.  23.  24.]
# [ 25.  26.  27.  28.  29.]] nan [  10.   35.   nan   85.  110.]

# b = np.arange(12).reshape(4,3)
# print(b, b.shape, np.sum(b, axis = 0), np.sum(b, axis = -1))
#
# calc_return # calculate return for all business days in 2008

# prices[1:] - prices[:-1]
# prices.shape
# (prices[1:] - prices[:-1]) / [prices[:-1]]      #better solution than for loop
# np.mean)prices)
# np.std(prices)
# np.var(prices)
# np.max(prices)
# np.min(prices)
# np.argmin(prices)
# np.argmax(prices)   # position
#
# np.argmax(b)    # print position
# np.unravel_index(np.argmax(b), b.shape)     # find position of max value
#
# gind statistics
#
# emptyArray = np.zeros(5)
# emptyArray[1:4] = np.arange(3)
# emptyArray[1::2] = np.random.rand(1)
# print(emptyArray)
# # [ 0.          0.56403691  1.          0.56403691  0.
# #
# emptyArray = np.zeros(15)
# emptyArray[5:10] = np.arange(5)
#
# print(emptyArray)
# # [ 0.  0.  0.  0.  0.  0.  1.  2.  3.  4.  0.  0.  0.  0.  0.]
#
# emptyArray[1::2] = np.random.rand(int(len(emptyArray) / 2))
# print(emptyArray.shape, emptyArray)

# (15,) [ 0.          0.74147071  0.          0.24976153  0.          0.13237944
# 1.          0.95527272  3.          0.26167094  0.          0.57219965
# 0.          0.97995582  0.        ]




# use mask to replace values < mean with mean
a = np.array([(20, 25, 10, 23, 26, 32, 10, 5, 0), (0, 5, 20, 25, 21, 20, 11, 15, 10)])
mean = a.mean()
print(a[a<mean])
# [10 10  5  0  0  5 11 15 10]


a[a<mean] = mean    #replace values < mean with mean
print(a)
# [[20 25 15 23 26 32 15 15 15]
#  [15 15 20 25 21 20 15 15 15]]