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

#plot all figures
plt.show()