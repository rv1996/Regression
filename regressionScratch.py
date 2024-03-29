from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from meanError import *
import random

style.use("fivethirtyeight")

# xs = np.array([1,2,3,4,5,6,4,5,6,7,8,4,5,6], dtype=np.float64)
# ys = np.array([5,4,6,5,6,7,6,7,8,9,8,9,8,7], dtype=np.float64) 
# above is the old tested data now we are creating our own data so we must use the newly created data

def create_dataset(hm,variance,step=2,correlation=False):

    val = 1
    ys = []
    for i in range(hm):  # hm-> how many
        y = val + random.randrange(-variance,variance)
        ys.append(y)
        if correlation and correlation=="pos":
            val+=step
        elif correlation and correlation=="neg":
            val-=step
    xs = [i for i in range(len(ys))]        
    return np.array(xs,dtype=np.float64),np.array(ys,dtype=np.float64)

def best_fite_slope_and_intercept(xs,ys):
    m = ( mean(xs)*mean(ys) - mean(xs*ys) ) / ((mean(xs)**2  - mean(xs**2)) )
    b = mean(ys) -m*mean(xs)

    return m,b

xs,ys = create_dataset(40,40,2,correlation="pos")

m,b = best_fite_slope_and_intercept(xs,ys)
print(m,b)

regression_line = [(m*x)+b for x in xs]
predict_x = 11
predict_y = predict_x*m + b

r_squared = coefficent_of_determination(ys,regression_line)
print("R squared mean error ",r_squared)

plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y,color="g")
plt.plot(xs,regression_line)
plt.show()