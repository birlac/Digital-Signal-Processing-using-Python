import numpy as np
import matplotlib.pyplot as plt

#signal which is a sinusoid
# we want a signal of duration of 1 second
# what should be our time step?

# typical power converter switching freq in 5kHz
# sampling time of 200 us

#time step for signal generation should be 1 us
#signal generation is 200 times more accurate than sampled signal

t_duration = 1.0
t_step = 1.0e-06

data_points= int(t_duration/t_step)

#time array which is integers from 0 to 1million -1
time_array = np.arange(data_points)*t_step

frequency = 60.0
omega = 2*np.pi*frequency
mag = 5.0

inp_signal = mag*np.sin(time_array*omega)

t_sample = 200.0e-06
t_skip = int(t_sample/t_step)

sample_time = time_array[::t_skip]  
sample_signal = inp_signal[::t_skip]

plt.plot(time_array, inp_signal, label='full', ds='steps')
plt.plot(sample_time, sample_signal, label='sample', ds='steps')
plt.legend()
plt.show()