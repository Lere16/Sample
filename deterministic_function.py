from numpy import exp, piecewise

def sigmoid(x, x0, k):
    return 1 / (1 + exp(-k * (x - x0)))

def _function(x, x1, x2, x3, x4, high, base):
    level =[]
    for i in x:
        k1 = (high + base) / (x2 - x1)
        k3 = (high - base) / (x4 - x3)

        term1 = k1 * (i - x1) - high
        term2 = base
        term3 = k3 * (i - x3) + base
    
        n=1e3
        level.append(term1 * sigmoid(i, x1, n) * sigmoid(x2, i, n)
            + term2 * sigmoid(i, x2, n) * sigmoid(x3, i, n)
            + term3 * sigmoid(i, x3, n) * sigmoid(x4, i, n))

    return level

# Example usage:
base = 0
high= 10e3
limit= 5e3
thres = 1e3
x1 = -limit
x2 = -thres
x3 = thres
x4 = limit

x= [1.02e3 , 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 9e3, 10e3]
        
level = _function(x, x1, x2, x3, x4, high, base)
print(level)
