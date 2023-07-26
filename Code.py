import numpy as np
import matplotlib.pyplot as plt
def expit(x):
    return 1/(1 + np.exp(-x))

# Generating random points
np.random.seed(13)
x_h=np.random.normal(1.1,0.3,5)
x_e=np.random.normal(1.9,0.4,5)

# Assigning 0 and 1
y_h=np.zeros(x_h.shape)
y_h[:]=0.0
y_e=np.zeros(x_e.shape)
y_e[:]=+1.0

plt.plot(x_h,y_h,'co', label="hobbit")
plt.plot(x_e,y_e,'mo', label="elf")
plt.title('Training samples from two different classes c1 ja c2')
plt.legend()
plt.xlabel('height [m]')
plt.ylabel('y [class id]')
plt.show()

#Putting all training data points to same vector
x_tr=np.concatenate((x_h,x_e))
y_tr=np.concatenate((y_h,y_e))
print(f'The size of x is {x_tr.size}')
print(f'The size of y is {y_tr.size}')

# Weight Initialization
w0_t = 0
w1_t = 0

## MSE Calculation with initial weights
y_pred=expit(w1_t*x_tr+w0_t)
MSE=np.sum(((y_tr-y_pred)**2)/(len(y_tr)))
plt.title(f'Epoch=0 w0={w0_t:.2f} w1={w1_t:.2f} MSE={MSE:.2f}')
plt.plot(x_h,y_h,'co', label="hobbit")
plt.plot(x_e,y_e,'mo', label="elf")
x = np.linspace(0.0,+5.0,50)
plt.plot(x,expit(w1_t*x+w0_t),'b-',label='y=sig(w1x+w0)')
plt.xlabel('height [m]')
plt.axis([0.5,3.0,-0.1,+1.1])
plt.legend()
plt.show()

# Gradient Descent Optimization
num_of_epochs = 400
learning_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for mu in learning_rate:
    for e in range(num_of_epochs):
        for x_ind, x in enumerate(x_tr):
            
            y= expit(w1_t*x + w0_t)
            w1_t= w1_t-mu*((-2/len(y_tr))*(y_tr[x_ind]-y)*(y*(1-y)*x))
            w0_t= w0_t-mu*((-2/len(y_tr))*(y_tr[x_ind]-y)*(y*(1-y)*1))
       
        if np.mod(e, 20) == 0 or e == 1: # Plot after every 20th epoch
            y_pred = expit(w1_t*x_tr+w0_t)
            MSE = np.sum((y_tr-y_pred)**2)/(len(y_tr))
            plt.title(f'Epoch={e} w0={w0_t:.2f} w1={w1_t:.2f} MSE={MSE:.2f}')
            plt.plot(x_h,y_h,'co', label="hobbit")
            plt.plot(x_e,y_e,'mo', label="elf")
            x = np.linspace(0.0,+5.0,50)
            plt.plot(x,expit(w1_t*x+w0_t),'b-',label='y=sig(w1x+w0)')
            plt.plot([0.5, 5.0],[0.5,0.5],'k--',label='y=0 (class boundary)')
            plt.xlabel('height [m]')
            plt.legend()
            plt.show()
    print(f'Learning Rate = {mu} ==> MSE = {MSE}')

np.set_printoptions(precision=2)
print(f'True values y={y_tr} and predicted values y_pred={y_pred}')
