from statistics import mean

def squared_error(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)

def coefficent_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]

    error_regr = squared_error(ys_orig,ys_line)
    error_y_mean = squared_error(ys_orig,y_mean_line)

    return 1 - error_regr/error_y_mean