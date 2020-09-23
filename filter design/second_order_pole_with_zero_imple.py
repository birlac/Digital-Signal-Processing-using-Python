from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
#combining second order pole with second order zero 

#Pole
omega_1 = 1700  #rad/s
zeta1 = 0.1      #damping ratio
lc_num1 = [np.float64(omega_1*omega_1)]
lc_den1 = [1, 2*zeta1*omega_1, omega_1*omega_1]      # s^2 + 2*s*zeta*omega_0 + omega_0^2
pole1 = signal.lti(lc_num1,lc_den1)

omega_4 = 5300  #rad/s
zeta4 = 0.5      #damping ratio
lc_num4 = [np.float64(omega_4*omega_4)]
lc_den4 = [1, 2*zeta4*omega_4, omega_4*omega_4]
pole2 = signal.lti(lc_num4,lc_den4)

lc_num_pole = np.polymul(lc_num1,lc_num4)
lc_den_pole = np.polymul(lc_den1,lc_den4)



#zero
omega_2 = 3000  #rad/s
zeta2 = 0.1      #damping ratio
lc_num2 = [1, 2*zeta2*omega_2, omega_2*omega_2]
lc_den2 = [np.float64(omega_2*omega_2)] 
zero1 = signal.lti(lc_num2,lc_den2)

omega_3 = 3000  #rad/s
zeta3 = 0.1      #damping ratio
lc_num3 = [1, 2*zeta3*omega_3, omega_3*omega_3]
lc_den3 = [np.float64(omega_3*omega_3)]
zero2 = signal.lti(lc_num3,lc_den3)

lc_num_zero = np.polymul(lc_num2,lc_num3)
lc_den_zero = np.polymul(lc_den2,lc_den3)

lc_num = np.polymul(lc_num_pole,lc_num_zero)
lc_den = np.polymul(lc_den_pole,lc_den_zero)

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

# End of design


# Conversion to digital

#Cheating python to look at zero as a pole  for the sake
# of implementation because of not being able to discretize
# an improper transfer function
zero1_mod = signal.lti(lc_den2,lc_num2)
zero2_mod = signal.lti(lc_den3,lc_num3)

pole1_z = pole1.to_discrete(dt =200.0e-6, method='tustin')
zero1_z = zero1_mod.to_discrete(dt =200.0e-6, method='tustin') #discretize modified zero1 TF
zero2_z = zero2_mod.to_discrete(dt =200.0e-6, method='tustin')
pole2_z = pole2.to_discrete(dt =200.0e-6, method='tustin')
# start of implementation

t_duration = 1.0
t_step = 1.0e-6

no_of_data = int(t_duration/t_step)

# Time array which is integers from 0 to 1 million -1
time_array = np.arange(no_of_data)*t_step

frequency = 50.0        # Hertz
omega = 2*np.pi*frequency       # rad/s
omega_noise = 2*np.pi*550.0        #11th harmonic freq i.e. 550Hz
omega_pass = 25000              #from freq resp characteristics rad/s
inp_mag = np.sqrt(2)*240.0       # input Voltage

# mag*sin(2 pi f t)
ip_voltage_signal = (1.0*np.sin(time_array*omega) + 0.2*np.sin(time_array*omega_noise) + 0.02*np.sin(time_array*omega_pass))

t_sample = 200.0e-6     # 5kHz
no_of_skip = int(t_sample/t_step)

tsample_array = time_array[::no_of_skip]
ip_voltage_samples = ip_voltage_signal[::no_of_skip]

# Setup our input

# Initialize our output
op_voltage_samples = np.zeros(ip_voltage_samples.size)


# Filter input
u1 = np.zeros(3)     # Present value u[0]=vin(n) and past vaue u[1]=vin(n-1), u[2]=vin(n-2)
y1 = np.zeros(3)     # Present y[0]=vo(n) and past y[1]=vo(n-1), y[2]=vo(n-2)

u2 = np.zeros(3)     # Present value u[0]=vin(n) and past vaue u[1]=vin(n-1), u[2]=vin(n-2)
y2 = np.zeros(3)

u3 = np.zeros(3)     # Present value u[0]=vin(n) and past vaue u[1]=vin(n-1), u[2]=vin(n-2)
y3 = np.zeros(3)

u4 = np.zeros(3)     # Present value u[0]=vin(n) and past vaue u[1]=vin(n-1), u[2]=vin(n-2)
y4 = np.zeros(3)

# Calculate our output
for volt_index, volt_value in np.ndenumerate(ip_voltage_samples):

    # Implementation of the filter
    u1[0] = volt_value

    y1[0] = (pole1_z.num[0]*u1[0] + pole1_z.num[1]*u1[1] + pole1_z.num[2]*u1[2] - pole1_z.den[1]*y1[1] - pole1_z.den[2]*y1[2])/pole1_z.den[0]

    u1[2] = u1[1]     # u(n-2) = u(n-1)
    y1[2] = y1[1]     # y(n-2) = y(n-1)
    u1[1] = u1[0]     # u(n-1) = u(n)
    y1[1] = y1[0]     # y(n-1) = y(n)

    u2[0] = y1[0]
  
    y2[0] = (zero1_z.den[0]*u2[0] + zero1_z.den[1]*u2[1] + zero1_z.den[2]*u2[2] - zero1_z.num[1]*y2[1] - zero1_z.num[2]*y2[2])/zero1_z.num[0]

    u2[2] = u2[1]     # u(n-2) = u(n-1)
    y2[2] = y2[1]     # y(n-2) = y(n-1)
    u2[1] = u2[0]     # u(n-1) = u(n)
    y2[1] = y2[0]     # y(n-1) = y(n)

    u3[0] = y2[0]
  
    y3[0] = (zero2_z.den[0]*u3[0] + zero2_z.den[1]*u3[1] + zero2_z.den[2]*u3[2] - zero2_z.num[1]*y3[1] - zero2_z.num[2]*y3[2])/zero2_z.num[0]

    u3[2] = u3[1]     # u(n-2) = u(n-1)
    y3[2] = y3[1]     # y(n-2) = y(n-1)
    u3[1] = u3[0]     # u(n-1) = u(n)
    y3[1] = y3[0]     # y(n-1) = y(n)

    u4[0] = y3[0]

    y4[0] = (pole2_z.num[0]*u4[0] + pole2_z.num[1]*u4[1] + pole2_z.num[2]*u4[2] - pole2_z.den[1]*y4[1] - pole2_z.den[2]*y4[2])/pole2_z.den[0]

    u4[2] = u4[1]     # u(n-2) = u(n-1)
    y4[2] = y4[1]     # y(n-2) = y(n-1)
    u4[1] = u4[0]     # u(n-1) = u(n)
    y4[1] = y4[0]     # y(n-1) = y(n)



    op_voltage_samples[volt_index] = y4[0]

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