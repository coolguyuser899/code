from __future__ import print_function
from numpy import arange, loadtxt, zeros
import matplotlib.pyplot as plt

prices = loadtxt("aapl_2008_close_values.csv", usecols=[1], delimiter=",")
print("Prices for AAPL stock in 2008")
# print(prices)

# compute daily return
diff = prices[1:] - prices[:-1]
# [1:] means all but the first, ; [:-1] means all but the last
print(prices[1:], prices[:-1])
diffs = prices[1:] - prices[:-1]
returns = diffs / prices[:-1]

# calculate the line of 0 return, or baseline
days = arange(len(returns))
zero_line = zeros(len(returns))
print(len(returns), zeros(len(returns)))

plt.plot(days, zero_line, 'r-', days, returns * 100, 'b-')
plt.title("Daily return of aapl in 2008 %")

plt.xlim(xmax = len(returns))
plt.show()