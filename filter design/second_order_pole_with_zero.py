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
signal.lti(lc_num3,lc_den3)

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

#plot all figures
plt.show()