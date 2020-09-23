import numpy as np
import matplotlib.pyplot as plt

t_duration = 1.0
t_step = 1.0e-06

data_points= int(t_duration/t_step)

#time array which is integers from 0 to 1million -1
time_array = np.arange(data_points)*t_step

frequency = 60.0            #Hz
omega = 2*np.pi*frequency   #rad/s
mag = np.sqrt(2)*120.0       #Volt = 170V peak

voltage_signal = mag*np.sin(time_array*omega)

t_sample = 200.0e-06
t_skip = int(t_sample/t_step)

#setting up our current input signal
sample_time = time_array[::t_skip]  
voltage_samples = voltage_signal[::t_skip]

#Filter input initialization
u = np.zeros(2)    #present value v(n) and past value v(n-1) respectively
y = np.zeros(2)    #present i(n) and past value i(n-1) respectively

#Initialize the output
current_samples = np.zeros(voltage_samples.size)
L = 1e-03     #1mH inductor
R = 0.005        # 5 mohm loss resistor


#calculating our voltage output as per the delay operator equation D from notes
for voltage_index, voltage_value in np.ndenumerate(voltage_samples):
    
    #Implementation of filter
    u[0] = voltage_value
    y[0] = (t_sample*u[0] + t_sample*u[1] +  2*L*y[1] - t_sample*R*y[1]) / (2*L + t_sample*R)
    u[1] = u[0]     #storing the value calculated for further sample calculations
    y[1] = y[0]
    current_samples[voltage_index] = y[0]
    #End of filter implementation

plt.plot(time_array, voltage_signal, label='full', ds='steps')
plt.plot(sample_time, voltage_samples, label='sample', ds='steps')
plt.legend()
plt.show()

plt.figure()
plt.plot(sample_time, current_samples, label='output I', ds='steps')
plt.legend()
plt.show() 