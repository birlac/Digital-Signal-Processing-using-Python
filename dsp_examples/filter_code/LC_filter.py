import numpy as np
import matplotlib.pyplot as plt

t_duration = 1.0
t_step = 1.0e-6

no_of_data = int(t_duration/t_step)

# Time array which is integers from 0 to 1 million -1
time_array = np.arange(no_of_data)*t_step

frequency = 50.0        # Hertz
omega = 2*np.pi*frequency       # rad/s
omega_noise = 2*np.pi*650.0        # 650 Hz (13th harmonic) noise signal
mag = np.sqrt(2)*240.0       # Voltage
C = 1000.0e-6    # 100 microFard capacitor
R = 0.05     # 0.05 Ohm resistance as loss
L = 0.001   # 1 milli Henry inductor

# mag*sin(2 pi f t)
ip_voltage_signal = mag*(np.sin(time_array*omega) + 0.1*np.sin(time_array*omega_noise))

t_sample = 200.0e-6     # 5kHz
no_of_skip = int(t_sample/t_step)

tsample_array = time_array[::no_of_skip]
ip_voltage_samples = ip_voltage_signal[::no_of_skip]

# Setup our input

# Initialize our output
op_voltage_samples = np.zeros(ip_voltage_samples.size)


# Filter input
u = np.zeros(3)     # Present value u[0]=vin(n) and past vaue u[1]=vin(n-1), u[2]=vin(n-2)
y = np.zeros(3)     # Present y[0]=vo(n) and past y[1]=vo(n-1), y[2]=vo(n-2)


# Calculate our output
for volt_index, volt_value in np.ndenumerate(ip_voltage_samples):

    # Implementation of the filter
    u[0] = volt_value
    # v(n) = (Ti(n) + Ti(n-1) + 2C v(n-1))/2C
    #y[0] = (t_sample*u[0] + t_sample*u[1] + 2*C*y[1])/(2*C)
    # v(n) = (Ti(n) + Ti(n-1) + 2C v(n-1) - (t_sample/R)*v(n-1))/(2C + T/R)
    #y[0] = (t_sample*u[0] + t_sample*u[1] + 2*C*y[1] - (t_sample/R)*y[1] )/(2*C + (t_sample/R))
    y[0] = ( (1/(L*C))*(u[0] + 2*u[1] + u[2]) - ( -2*(2/t_sample)*(2/t_sample) - 2*(R/L)*(2/t_sample) + (2/(L*C)) )*y[1] - ( (2/t_sample)*(2/t_sample) - (R/L)*(2/t_sample) + (1/(L*C)) )*y[2] ) / ( (2/t_sample)*(2/t_sample) + (R/L)*(2/t_sample) + (1/(L*C)))

    u[2] = u[1]     # u(n-2) = u(n-1)
    y[2] = y[1]     # y(n-2) = y(n-1)
    u[1] = u[0]     # u(n-1) = u(n)
    y[1] = y[0]     # y(n-1) = y(n)

    op_voltage_samples[volt_index] = y[0]

    # End of filter implementation



plt.plot(time_array, ip_voltage_signal, label='full', ds='steps')
plt.plot(tsample_array, ip_voltage_samples, label='sample', ds='steps')
plt.legend()
plt.show()

plt.figure()
plt.plot(tsample_array, op_voltage_samples, label='output', ds='steps')
plt.legend()
plt.show()

plt.figure()
plt.plot(tsample_array, ip_voltage_samples, label='input', ds='steps')
plt.plot(tsample_array, op_voltage_samples, label='output', ds='steps')
plt.legend()
plt.show()
