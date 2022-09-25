from matplotlib import pyplot

# rectified linear function
def rectified(x):
    return max(0, x)

# define a series of inputs
series_in = [x for x in range(-10, 11)]
# output for inputs
series_out = [rectified(x) for x in series_in]

# line plot of raw inputs to rectified outputs
pyplot.plot(series_in, series_out)
pyplot.show()