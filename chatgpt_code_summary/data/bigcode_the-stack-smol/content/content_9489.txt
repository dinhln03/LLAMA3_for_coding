import numpy
import matplotlib.pyplot as plt
import threading
import multiprocessing
from scipy import stats

class TestHist:
    def hist(self, parameter_list):
        x = numpy.random.uniform(0.0, 5.0, 100000)

        plt.hist(x, 100)
        plt.show()

        y = numpy.random.normal(0.0, 5.0, 100000)

        plt.hist(y, 100)
        plt.show()


class TestScatter:
    def scatter(self, parameter_list):
        
        a = numpy.random.normal(5.0, 1.0, 1000)
        b = numpy.random.normal(10.0, 2.0, 1000)

        plt.scatter(a, b)
        plt.show()


class TestLinearRegression:
    def linear(self):
        
        a = numpy.random.uniform(5.0, 1.0, 1000)
        b = numpy.random.uniform(10.0, 2.0, 1000)

        slope, intercept, r, p, std_err = stats.linregress(a, b)

        print(slope, intercept, r, p, std_err )
                
        mymodel = list(map(lambda xa : self.myfunc(xa,slope,intercept),a))
        plt.scatter(a, b )
        plt.plot(a, mymodel)        
        plt.show()

    def myfunc(self,x, slope, intercept):
        """
        input, slope, intercept
        """
        return slope * x + intercept 
    

linear =TestLinearRegression()
linear.linear()        

# numpy.random.seed(12345678)
# x = numpy.random.random(10)
# y = 1.6*x + numpy.random.random(10)


# from scipy import stats

# x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
# y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

# slope, intercept, r, p, std_err = stats.linregress(x, y)

# def myfunc(x):
#   return slope * x + intercept

# speed = myfunc(10)
# print(slope, intercept, r, p, std_err)
# print(speed)