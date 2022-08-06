import math
import numpy as np

###### TOCORRECT
from .utils import AttrDict

all_functions = {
    # https://www.sfu.ca/~ssurjano/stybtang.html
    "Styblinski-Tang" : {"min_x" : -5 , "max_x" : 5, "min_y" : -5, "max_y" : 5, "step_x" : 0.25, "step_y" : 0.25},
    # http://www.sfu.ca/~ssurjano/ackley.html
    "Ackley" : {"min_x" : -5 , "max_x" : 5, "min_y" : -5, "max_y" : 5, "step_x" : 0.25, "step_y" : 0.25},
    # https://www.sfu.ca/~ssurjano/beale.html
    "Beale" : {"min_x" :  -4.5 , "max_x" : 4.5, "min_y" : -4.5, "max_y" : 4.5, "step_x" : 0.25, "step_y" : 0.25},
    # https://www.sfu.ca/~ssurjano/booth.html
    "Booth" : {"min_x" :  -10 , "max_x" : 10, "min_y" : -10, "max_y" : 10, "step_x" : 0.25, "step_y" : 0.25},
    # https://www.sfu.ca/~ssurjano/bukin6.html
    "Bukin" : {"min_x" : -15 , "max_x" : -5, "min_y" : -3, "max_y" : 3, "step_x" : 0.25, "step_y" : 0.25},
    # https://www.sfu.ca/~ssurjano/mccorm.html
    "McCormick" : {"min_x" : -1.5 , "max_x" : 4, "min_y" : -3, "max_y" : 4, "step_x" : 0.25, "step_y" : 0.25},
    # https://www.sfu.ca/~ssurjano/rosen.html
    "Rosenbrock" : {"min_x" : -2 , "max_x" : 2, "min_y" : -2, "max_y" : 2, "step_x" : 0.25, "step_y" : 0.25},
    # Others
    "Sum" : {"min_x" : -10 , "max_x" : 10, "min_y" : -10, "max_y" : 10, "step_x" : 0.25, "step_y" : 0.25},
    "Prod" : {"min_x" : -10 , "max_x" : 10, "min_y" : -10, "max_y" : 10, "step_x" : 0.25, "step_y" : 0.25},

}

# 1) Styblinski-Tang function : https://www.sfu.ca/~ssurjano/stybtang.html 

def STFunction(x, d=2):
    val = 0
    for i in range(d): val += x[i] ** 4 - 16 * x[i] ** 2 + 5 * x[i]
    val *= 0.5
    return val

def STDeriv(index):
    def f(x): return 0.5 * (4 * x[index] ** 3 - 32 * x[index] + 5)
    return f
    
# 2) Ackley function : http://www.sfu.ca/~ssurjano/ackley.html 

def AckleyFunction(x):
    x1, x2 = x[0], x[1]
    part_1 = -0.2*math.sqrt(0.5*(x1*x1 + x2*x2))
    part_2 = 0.5*(math.cos(2*math.pi*x1) + math.cos(2*math.pi*x2))
    return math.exp(1) + 20 -20*math.exp(part_1) - math.exp(part_2)

def AckleyDeriv(index):
    def f(x):
        x1, x2 = x[0], x[1]

        part_1 = -0.2*math.sqrt(0.5*(x1*x1 + x2*x2))
        part_1 = -20*math.exp(part_1)
        coef_deriv_part_1 = -0.2*math.sqrt(0.5)*x[index] / math.sqrt(x1*x1 + x2*x2)

        part_2 = 0.5*(math.cos(2*math.pi*x1) + math.cos(2*math.pi*x2))
        part_2 =  - math.exp(part_2)
        coef_deriv_part_2 = 0.5*(-2*math.pi*math.sin(2*math.pi*x[index]))

        return coef_deriv_part_1*part_1 + coef_deriv_part_2*part_2

    return f
    
# 3) Beale function : https://www.sfu.ca/~ssurjano/beale.html
def BealeFunction(x):
    x1, x2 = x[0], x[1]
    part_1 = (1.5 - x1 + x1*x2)**2
    part_2 = (2.25 - x1 + x1*(x2**2))**2
    part_3 = (2.625 - x1 + x1*(x2**3))**2
    return  part_1 + part_2 + part_3

def BealeDeriv(index):
    def f(x):
        x1, x2 = x[0], x[1]
        part_1 = 1.5 - x1 + x1*x2
        part_2 = 2.25 - x1 + x1*(x2**2)
        part_3 = 2.625 - x1 + x1*(x2**3)
        if index == 0 :
            return 2*(-1 + x2)*part_1 + 2*(-1 + x2**2)*part_2 + 2*(-1 + x2**3)*part_3
        elif index == 1 :
            return 2*x1*part_1 + 2*(2*x1*x2)*part_2 + 2*(3*x1*(x2**2))*part_3 
    return f
    
# 4) Booth function : https://www.sfu.ca/~ssurjano/booth.html
def BoothFunction(x):
    x1, x2 = x[0], x[1]
    part_1 = (x1 + 2*x2 - 7)**2
    part_2 = (2*x1 + x2 - 5)**2
    return  part_1 + part_2
    
def BoothDeriv(index):
    def f(x):
        x1, x2 = x[0], x[1]
        part_1 = (x1 + 2*x2 - 7)
        part_2 = (2*x1 + x2 - 5)
        if index == 0 :
            return 2*part_1 + 4*part_2 
        elif index == 1 :
            return 4*part_1 + 2*part_2 
    return f
    
# 5) Bukin function : https://www.sfu.ca/~ssurjano/bukin6.html

def part_1(x):
    x1, x2 = x[0], x[1] 
    return math.sqrt(abs(x2 - 0.01*(x1**2)))

def part_2(x):
    x1, x2 = x[0], x[1]
    return abs(x1 + 10)

def BukinFunction(x):
    return 100*part_1(x) + 0.01*part_2(x)

def part_1_deriv(x, index):
    x1, x2 = x[0], x[1]
    try : 
        if index == 0 :
            condition = x2 > 0 and -math.sqrt(x2/0.01) < x1 < math.sqrt(x2/0.01)
            return (-1 if condition else 1)*0.01*x1/part_1(x)
        elif index == 1 :
            return (-1 if x2 < 0.01*(x1**2) else 1)/(2*part_1(x)) 
    except ZeroDivisionError :
        assert x2 == 0.01*(x1**2)
        return 0

def part_2_deriv(x, index):
    if index == 0 : return 1 if x[0] > -10 else -1 
    elif index == 1 : return 0

def BukinDeriv(index):
    def f(x): return 100*part_1_deriv(x, index) + 0.01*part_2_deriv(x, index)
    return f
    
# 6) McCormick function : https://www.sfu.ca/~ssurjano/mccorm.html

def McCormickFunction(x):
    x1, x2 = x[0], x[1]
    return math.sin(x1 + x2) + (x1 - x2)**2 - 1.5*x1 + 2.5*x2 + 1
    
def McCormickDeriv(index):
    def f(x):
        x1, x2 = x[0], x[1]
        if index == 0 : return math.cos(x1 + x2) + 2*(x1 - x2) - 1.5
        elif index == 1 : return math.cos(x1 + x2) - 2*(x1 - x2) + 2.5
    return f
    
# 7) Rosenbrock function : https://www.sfu.ca/~ssurjano/mccorm.html

def RosenbrockFunction(x):
    x1, x2 = x[0], x[1]
    return 100*(x2-x1**2)**2 + (x1 - 1)**2 
    
def RosenbrockDeriv(index):
    def f(x):
        x1, x2 = x[0], x[1]
        if index == 0 :
            return -400*x1*(x2 - x1**2) + 2*(x1 - 1)
        elif index == 1 :
            return 200*(x2 - x1**2)
    return f

def get_function(params) :
    f_name = params.f_name
    if f_name == "Styblinski-Tang" :
        callable_function = STFunction
        callable_function_deriv = STDeriv
    elif f_name == "Ackley" :
        callable_function = AckleyFunction
        callable_function_deriv = AckleyDeriv
    elif f_name == "Beale" :
        callable_function = BealeFunction
        callable_function_deriv = BealeDeriv
    elif f_name == "Booth" :
        callable_function = BoothFunction
        callable_function_deriv = BoothDeriv
    elif f_name == "Bukin" :
        callable_function = BukinFunction
        callable_function_deriv = BukinDeriv
    elif f_name == "McCormick" :
        callable_function = McCormickFunction
        callable_function_deriv = McCormickDeriv
    elif f_name == "Rosenbrock" :
        callable_function = RosenbrockFunction
        callable_function_deriv = RosenbrockDeriv
    elif f_name == "Sum" :
        callable_function = lambda x : x.sum()
        def callable_function_deriv(index : int) :  return lambda x : 1
    elif f_name == "Prod" :
        callable_function = lambda x : x.prod()
        def callable_function_deriv(index : int) :  return lambda x : np.delete(x, index).prod() # x[:index-1].prod() * x[index+1:].prod()

    params_ = {
        "callable_function" : callable_function,
        "callable_function_deriv" : callable_function_deriv
    }
    params = AttrDict({**params, **params_})

    params_ = all_functions.get(f_name, {})
    for attr_name in ["min_x", "max_x", "min_y", "max_y", "step_x", "step_y"] :
        setattr(params, attr_name, getattr(params, attr_name, params_.get(attr_name)))

    return AttrDict(params)
