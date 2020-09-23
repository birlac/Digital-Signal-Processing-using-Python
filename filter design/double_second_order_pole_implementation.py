from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

omega_1 = 1700   #rad/s
zeta1 = 0.1      #damping ratio
omega_2 = 1700   #rad/s
zeta2 = 0.1      #damping ratio


lc_num1 = [np.float64(omega_1*omega_1)]
lc_den1 = [1, 2*zeta1*omega_1, omega_1*omega_1]      # s^2 + 2*s*zeta*omega_0 + omega_0^2
lc_tf1 = signal.lti(lc_num1, lc_den1)

lc_num2 = [np.float64(omega_2*omega_2)]
lc_den2 = [1, 2*zeta2*omega_2, omega_2*omega_2]      # s^2 + 2*s*zeta*omega_0 + omega_0^2
lc_tf2 = signal.lti(lc_num2, lc_den2)


lc_tf1_z = lc_tf1.to_discrete(dt = 200.0e-06, method ='tustin') # tustin implies bilinear or trapezoidal transformation
lc_tf2_z = lc_tf2.to_discrete(dt = 200.0e-06, method ='tustin')
#lc_tf1_z.num
#lc_tf1_z.den
#lc_tf2_z.num
#lc_tf2_z.den
#End of filter design

# start of implementation

t_duration = 1.0
t_step = 1.0e-6

no_of_data = int(t_duration/t_step)

# Time array which is integers from 0 to 1 million -1
time_array = np.arange(no_of_data)*t_step

frequency = 50.0        # Hertz
omega = 2*np.pi*frequency       # rad/s
omega_noise = 2*np.pi*550.0        #11th harmonic freq i.e. 550Hz
inp_mag = np.sqrt(2)*240.0       # input Voltage

# mag*sin(2 pi f t)
ip_voltage_signal = (1.0*np.sin(time_array*omega) + 0.2*np.sin(time_array*omega_noise))

t_sample = 200.0e-6     # 5kHz
no_of_skip = int(t_sample/t_step)

tsample_array = time_array[::no_of_skip]
ip_voltage_samples = ip_voltage_signal[::no_of_skip]

# Setup our input

# Initialize our output
op_voltage_samples = np.zeros(ip_voltage_samples.size)


# Filter input, tf1
u1 = np.zeros(3)     # Present value u1[0]=vin(n) and past vaue u1[1]=vin(n-1), u1[2]=vin(n-2)
y1 = np.zeros(3)     # Present y1[0]=vo(n) and past y1[1]=vo(n-1), y1[2]=vo(n-2)

# Filter input, tf2
u2 = np.zeros(3)     # Present value u2[0]=y1[n] and past vaue u2[1]=y1[n-1], u2[2]=y1(n-2), output of stage 1 is input to stage 2(cascading)
y2 = np.zeros(3)     # Present y2[0]=vo(n) and past y2[1]=vo(n-1), y2[2]=vo(n-2)


# Calculate our output
for volt_index, volt_value in np.ndenumerate(ip_voltage_samples):

    # Implementation of the first stage of the filter
    u1[0] = volt_value
    y1[0] = (lc_tf1_z.num[0]*u1[0] + lc_tf1_z.num[1]*u1[1] + lc_tf1_z.num[2]*u1[2] - lc_tf1_z.den[1]*y1[1] - lc_tf1_z.den[2]*y1[2])/lc_tf1_z.den[0]

    u1[2] = u1[1]     # u1(n-2) = u1(n-1)
    y1[2] = y1[1]     # y1(n-2) = y1(n-1)
    u1[1] = u1[0]     # u1(n-1) = u1(n)
    y1[1] = y1[0]     # y1(n-1) = y1(n)

    # Implementation of the first stage of the filter
    u2[0] = y1[0]   #feeding ouput of stage one into the input of stage 2
    y2[0] = (lc_tf2_z.num[0]*u2[0] + lc_tf2_z.num[1]*u2[1] + lc_tf2_z.num[2]*u2[2] - lc_tf2_z.den[1]*y2[1] - lc_tf2_z.den[2]*y2[2])/lc_tf2_z.den[0]

    u2[2] = u2[1]     # u2(n-2) = u2(n-1)
    y2[2] = y2[1]     # y2(n-2) = y2(n-1)
    u2[1] = u2[0]     # u2(n-1) = u2(n)
    y2[1] = y2[0]     # y2(n-1) = y2(n)


    op_voltage_samples[volt_index] = y2[0]

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
plt.grid()
plt.legend()


#plot all figures
plt.show()

