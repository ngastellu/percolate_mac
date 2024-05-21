#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


c1 = np.array([-0.09996723, 0.39951283, 0.70011547])
c2 = np.array([0.15591627, 0.60768372, 0.39195739])

exps = np.array([2.9412494, 0.6834831, 0.2222899])


x = np.linspace(0,15,10000)

exponentials = np.array([((2*a/np.pi)**0.75)*np.exp(-a*(x**2)) for a in exps])
print(exponentials.shape)

f1 = np.dot(c1,exponentials)*x*x
f2 = np.dot(c2,exponentials)*x*x

print(f1.shape)


plt.plot(x,f1,'r-',lw=0.8,label='f1')
plt.plot(x,f2,'b--',lw=0.8,label='f2')

plt.legend()
plt.show()