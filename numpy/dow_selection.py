from __future__ import print_function
from numpy import loadtxt, sum, where
import matplotlib.pyplot as plt

OPEN = 0
HIGH = 1
LOW =2
CLOSE =3
VOLUMN = 4
ADJ_CLOSE = 5

dow = loadtxt('dow.csv', delimiter=',')

# volumn > 5.5 billion
high_volumn_mask = dow[:, VOLUMN] > 5.5e9

# how many are there
high_volumn_days = sum(high_volumn_mask)
print("dow volumn above 5.5 billion has {} days this year".format(high_volumn_days))

# find the index of every row (or day) where volumn is > 5.5 billion
high_vol_index = where(high_volumn_mask)[0]

# plot red dot for days with volumn > 5.5 billion
plt.figure()

#plot teh adjusted close for everyday as blue line
plt.plot(dow[:, ADJ_CLOSE], 'p-')       # b- means blue line

# plot days where volumn was high with red dots
plt.plot(high_vol_index, dow[high_vol_index, ADJ_CLOSE], 'ro')

plt.show()
