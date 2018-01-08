from numpy import dtype, loadtxt, float64, NaN, isfinite, all
import matplotlib.pyplot as plt

log_file = open('short_log.crv')
header = log_file.readline()
print(header)
# create dtype from header field
log_names = header.split()
print(log_names)
# ['DEPTH', 'CALI', 'S-SONIC', 'P-SONIC', 'GR', 'LITH', 'RESISTIVITY', 'NPHI', 'POROS', 'RHOB', 'SWARCH', 'SW_I', 'VP', 'VSH', 'VS']

# construct array dtype describing data, 8 bytes (64 bits) floating point
# f8 - 64-bit floating-point number
fields = list(zip(log_names, ['f8'] * len(log_names)))
print(fields)
# [('DEPTH', 'f8'), ('CALI', 'f8'), ('S-SONIC', 'f8'), ('P-SONIC', 'f8'), ('GR', 'f8'), ('LITH', 'f8'), ('RESISTIVITY', 'f8'), ('NPHI', 'f8'), ('POROS', 'f8'), ('RHOB', 'f8'), ('SWARCH', 'f8'), ('SW_I', 'f8'), ('VP', 'f8'), ('VSH', 'f8'), ('VS', 'f8')]

fields_type = dtype(fields)
print(fields_type)

logs = loadtxt(log_file, dtype=fields_type)
print(logs)
# [ ( 8744.5, -999.25  , -999.25  , -999.25  ,  87.3611, -999.25, -999.25  ,  -9.99250000e+02,  -9.99250000e+02, -999.25  ,  -9.99250000e+02,  -9.99250000e+02,   -999.25  ,  -9.99250000e+02,  -999.25  )
#   ( 8745. , -999.25  , -999.25  , -999.25  ,  86.646 , -999.25, -999.25  ,  -9.99250000e+02,  -9.99250000e+02, -999.25  ,  -9.99250000e+02,  -9.99250000e+02,   -999.25  ,  -9.99250000e+02,  -999.25  )
#
# original data
# 8744.5000   -999.2500   -999.2500   -999.2500     87.3611   -999.2500   -999.2500   -999.2500   -999.2500   -999.2500   -999.2500   -999.2500   -999.2500   -999.2500   -999.2500

# make 2-d, float 64 view of data
values = logs.view(float64)
print("logs", values)
# [  8.74450000e+03  -9.99250000e+02  -9.99250000e+02 ...,   1.34029932e+04
#    3.32900000e-01   7.90716360e+03]

# tuple of array dimensions, -1 in the row shape means that numpy should make this dimension whatever ti needs to be so that rows * cols = size for the array
values.shape = -1, len(fields)
print(values.shape)
# (1060, 15)

# replace -999.25 with NaN, the default value if no actual value is available
values[values == -999.25] = NaN
print(values)
# [[  8.74450000e+03              nan              nan ...,              nan
#     nan              nan]

# make a mask for all rows that don't have any missing values
# `axis` may be negative, in which case it counts from the last to the first axis
data_mask = all(isfinite(values), axis=-1)
print(data_mask)
# [False False False ...,  True  True  True]

good_logs = logs[data_mask]
print(good_logs)
# [ ( 8983. ,  6.0026,  157.1567,  85.317 ,   20.7575,  3.,   19.128 ,  0.2008,  0.1088,  2.4705,  1.    ,  0.7694,  11720.9932,  0.5486,  6363.0757)
#   ( 8983.5,  5.9624,  175.2532,  90.749 ,   17.2627,  3.,   20.1494,  0.2196,  0.0981,  2.4881,  1.    ,  0.7947,  11019.4053,  0.6329,  5706.0303)


plt.plot(good_logs['VS'], good_logs['VP'])
plt.xlabel('VS')
plt.ylabel('VP')
plt.show()

