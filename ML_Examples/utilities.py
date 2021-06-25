import matplotlib.pyplot as plt


def multi_plot_best_results(data, params_values, compare=None, x_zoom=0, y_zoom=0,
                            labels=dict()):
    ''' plot, zoom and show best value for a param in some data
    data: data to plot (matrix)
    params_value: array of param's values tested (ndarray)
    compare: another matrix to plot (matrix with same shape of data)
    x_zoom: bandwith around the best value on x-axis (int|float)
    y_zoom: bandwith around the best value on y-axis (int|float)
    labels: labels to display (dict) {"data": "data", "compare": "comparison",
                                    "x_axis": "x", "y_axis": "y"} 
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    _labels = {"data": "data",
               "compare": "comparison",
               "x_axis": "x",
               "y_axis": "y"}
    _labels.update(labels)
    plt.plot(params_values, data, label=_labels['data'])

    if type(compare) == 'numpy.ndarray':
        print(True)
        plt.plot(params_values, compare, label=_labels['compare'])

    ymax = data.max()
    xpos = data.argmax()
    x = range(len(data))
    xmax = x[xpos]

    ax.annotate(f'Best score:{round(ymax,3)}\nBest value:{xmax}',
                xy=(xmax, ymax), xytext=(xmax, ymax+0.05),
                arrowprops=dict(facecolor='black'),
                )
    ax_x_min = 0
    ax_x_max = len(data)
    ax_y_min = 0
    ax_y_max = ymax

    if(x_zoom > 0):
        ax_x_min = xpos-x_zoom
        ax_x_max = xpos+x_zoom

    ax.set_xlim(ax_x_min, ax_x_max)

    if(y_zoom > 0):
        ax_y_min = ymax-y_zoom
        ax_y_max = ymax+y_zoom

    ax.set_ylim(ax_y_min, ax_y_max)

    plt.ylabel(_labels['y_axis'])
    plt.xlabel(_labels['x_axis'])
    plt.legend()
    plt.show()
