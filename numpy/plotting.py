from numpy import linspace, pi, sin, cos
import matplotlib.pyplot as plt

x = linspace(0, 2*pi, 101)  # Return evenly spaced numbers over a specified interval, generate 101
s = sin(x)
c = cos(x)

# print(x, s, c)

img = plt.imread('dc_metro.JPG')

# first plot

plt.close('all')    # closes all the figure windows
plt.subplot(2, 2, 1)    # 2 rows, 2 columns of space, start from top left
plt.plot(x, s, 'b-', x, c, 'r+')    # *args, *kwargs
plt.axis('tight')   # changes *x* and *y* axis limits such that all data is shown.

# second plot
plt.subplot(2, 2, 2)
plt.plot(x, s)
plt.grid()  # show grid line
plt.xlabel('radians')       #  a unit of measurement of angles equal to about 57.3
plt.ylabel('amplitude')     # what is sway or distance from middle between min and max, abs(( max - min) / 2)
plt.title('sin(x)')
plt.axis('tight')

# 3rd plot, image
plt.subplot(2, 2, 3)
# Extent defines the images max and min of the horizontal and vertical values. It takes four values like so:
# extent=[horizontal_min,horizontal_max,vertical_min,vertical_max].
plt.imshow(img, origin = 'upper', extent=[-5, 50, -5, 50], cmap = plt.cm.winter)
plt.tight_layout()

plt.subplot(2, 2, 4)
plt.imshow(img, origin = 'upper', extent=[0, .5, 0, .5], cmap = plt.cm.winter)

plt.show()
plt.savefig('my_plots.png')