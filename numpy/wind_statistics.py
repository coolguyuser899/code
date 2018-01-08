# First of all, from __future__ import print_function needs to be the first line of code in your script
# (aside from some exceptions mentioned below). Second of all, as other answers have said, you have to use print as a function now.
# That's the whole point of from __future__ import print_function; to bring the print function from Python 3 into Python 2.6+.

from __future__ import print_function
from numpy import (loadtxt, arange, searchsorted, add, zeros, unravel_index, where)
import numpy as np

wind_data = loadtxt('wind.data')

data = wind_data[:, 3:]         # : is to select all rows, 3: is to select from 3 column to rest
print(data)

print('2:statistics over all values')
print(' min: ', data.min())
print(' max: ', data.max())
print(' mean: ', data.mean())
print(' standard deviation: ', data.std())

print('3: statistics over all days at each location')   # based on 12 column or locations
print(' min: ', data.min(axis=0))   # axis = 0, vertically downwards across rows (axis 0
print(' max: ', data.max(axis=0))
print(' mean: ', data.mean(axis=0))
print(' standard deviation: ', data.std(axis=0))

print('4: statistics over all locations for each day')  # based on all rows
print(' min: ', data.min(axis=1))
print(' max: ', data.max(axis=1))
print(' mean: ', data.mean(axis=1))
print(' standard deviation: ', data.std(axis=1))

print('5: location of daily maximum')
print(' daily max location: ', data.argmax(axis=1))

#
daily_max = data.max(axis=1)
max_row = daily_max.argmax()
print(max_row)
# 2161

# Or use unravel_index takes a linear index and convert it to a location given the shape of the array
max_row, max_col = unravel_index(data.argmax(), data.shape)
print(max_row, max_col)
# 2161 11

# Or use where to identify all places where max occurs, where returns two array
max_row, max_col = where(data == data.max())
print(max_row, max_col)
# [2161] [11]

# running horizontally across columns (axis 1)

print('6: day of maximum reading')
print(' year:', int(wind_data[max_row, 0]))
print(' month:', int(wind_data[max_row, 1]))
print(' year:', int(wind_data[max_row, 2]))

january_indices = wind_data[0:, 1] == 1
january_data = data[january_indices]

print('7: statistics for january')
print(' mean:', january_data.mean(axis=0))

# compute month number for each day in the dataset
months = (wind_data[:, 0] -61) * 12 + wind_data[:, 1] -1
print((wind_data[:, 0] - 61)*12 +  wind_data[:, 1] - 1)

month_values = set(months)
print(month_values)
monthly_means = zeros(len(month_values))

# for month in month_values:
#     day_indices = (months == month) # find rows correspond to the current month
#     month_data = data[day_indices]  # extract data for current month using fancy indexing
#     monthly_means[month] = month_data.mean()

# above for loop can be rewritten as one line below
monthly_means = np.array([data[months == month].mean() for month in month_values])
print(monthly_means)

# extract first 52 weeks, reshapre to put 7 days for all locations
weekly_data = data[:52 * 7].reshape(-1, 7 *12)
print(weekly_data)

print(' min:', weekly_data.min(axis=1))
print(' max:', weekly_data.max(axis=1))
print(' mean:', weekly_data.mean(axis=1))
print(' standard deviation:', weekly_data.std(axis=1))

# compute the month number of each day
months = (wind_data[:, 0] - 61) * 12 + wind_data[:, 1] -1

# find indices for the start of each month
month_indices = searchsorted(months, arange(months[-1] + 2))
print(month_indices)

# now use add.reduceat to get the sum at each location
monthly_loc_totals = add.reduceat(data, month_indices[:-1])
print(monthly_loc_totals)

monthly_totals = monthly_loc_totals.sum(axis=1)

# find total number of meachsurements for each month
month_days = month_indices[1:] - month_indices[:1]
measurement_count = month_days * 12

# compute mean
monthly_means = monthly_totals / measurement_count

print('mean: ', monthly_means)
