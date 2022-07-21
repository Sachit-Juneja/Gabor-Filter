import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

f = 0.1
theta = math.radians(0.0) # Converts angle x from degrees to radians.
sigma_x = 7.0
sigma_y = 7.0
radius = 20

M = np.zeros((radius*2,radius*2))

def ChangeBase(x,y,theta):
    x_theta = x * math.cos(theta) + y * math.sin(theta)
    y_theta = y * math.cos(theta) - x * math.sin(theta)
    return x_theta, y_theta

def GaborFunction(x,y,theta,f,sigma_x,sigma_y):
    r1 = ChangeBase(x,y,theta)[0] / sigma_x
    r2 = ChangeBase(x,y,theta)[1] / sigma_y
    arg = - 0.5 * ( r1**2 + r2**2 )
    return math.exp(arg) * math.cos(2*math.pi*f*ChangeBase(x,y,theta)[0])

x = -float(radius)
for i in range(radius*2):
    y = -float(radius)
    for j in range(radius*2):
        M[i,j] = GaborFunction(x,y,theta,f,sigma_x,sigma_y)
        y = y + 1
    x = x + 1

#print M.max(), M.min()
# Normalization from 0 to 255
M[:,:] = ( (M[:,:] - M.min() ) * 255 ) / ( M.max() - M.min() )
#print M.max(), M.min()

plt.imshow(M.T, cmap = cm.Greys_r, origin='lower')

#plt.imshow(M.T, origin='lower')
#plt.colorbar()

plt.savefig('GaborFilter.png')
plt.show()