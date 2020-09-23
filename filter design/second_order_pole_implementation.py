from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


""" R = 0.1
C = 1000.0e-06
L = 0.001 """ 

omega_0 = 5000  #rad/s
zeta = 0.5      #damping ratio

lc_num = [omega_0*omega_0]
lc_den = [1, 2*zeta*omega_0, omega_0*omega_0]      # s^2 + 2*s*zeta*omega_0 + omega_0^2

lc_tf = signal.lti(lc_num, lc_den)

w, mag, phase = signal.bode(lc_tf, np.arange(100000))

plt.figure()
plt.grid()
plt.title('Magnitude in dB')
plt.xlabel('log w')
plt.ylabel('mag in dB')
plt.semilogx(w,mag)


plt.figure()
plt.grid()
plt.title('Phase in degrees')
plt.xlabel('log w')
plt.ylabel('Phase angle')
plt.semilogx(w,phase)



lc_tf_z = lc_tf.to_discrete(dt = 200.0e-06, method ='tustin') # tustin implies bilinear or trapezoidal transformation
#lc_tf_z.num
#lc_tf_z.den
#End of filter design

# start of implementation

t_duration = 1.0
t_step = 1.0e-6

no_of_data = int(t_duration/t_step)

# Time array which is integers from 0 to 1 million -1
time_array = np.arange(no_of_data)*t_step

frequency = 50.0        # Hertz
omega = 2*np.pi*frequency       # rad/s
omega_noise = 2*np.pi*8000.0        #8k Hz noise signal, once done try 11th harmonic freq i.e. 550Hz
inp_mag = np.sqrt(2)*240.0       # input Voltage

# mag*sin(2 pi f t)
ip_voltage_signal = inp_mag*(1.0*np.sin(time_array*omega) + 0.2*np.sin(time_array*omega_noise))

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
    #check jupyter notebook signal_examples file line 34. It should follow the same trend for second order systems
    y[0] = (lc_tf_z.num[0]*u[0] + lc_tf_z.num[1]*u[1] + lc_tf_z.num[2]*u[2] - lc_tf_z.den[1]*y[1] - lc_tf_z.den[2]*y[2])/lc_tf_z.den[0]

    u[2] = u[1]     # u(n-2) = u(n-1)
    y[2] = y[1]     # y(n-2) = y(n-1)
    u[1] = u[0]     # u(n-1) = u(n)
    y[1] = y[0]     # y(n-1) = y(n)

    op_voltage_samples[volt_index] = y[0]

    # End of filter implementation


plt.figure()
plt.plot(time_array, ip_voltage_signal, label='full', ds='steps')
plt.plot(tsample_array, ip_voltage_samples, label='sample', ds='steps')
plt.title('Input voltage signal and samples')
plt.xlabel('time')
plt.ylabel('Volts')
plt.legend()


plt.figure()
plt.plot(tsample_array, op_voltage_samples, label='output', ds='steps')
plt.title('Output voltage signal samples')
plt.xlabel('time')
plt.ylabel('Volts')
plt.legend()


plt.figure()
plt.plot(tsample_array, ip_voltage_samples, label='input', ds='steps')
plt.plot(tsample_array, op_voltage_samples, label='output', ds='steps')
plt.title('Input vs output')
plt.xlabel('time')
plt.ylabel('Volts')
plt.legend()


#plot all figures
plt.show()

