from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
# first and second order pole design


omega_1 = 500  # first order pole H(s) = w0/(s+w0)


omega_2 = 5000  # second order pole 
zeta2 = 0.1      #damping ratio

lc_num1 = [omega_1]
lc_den1 = [1, omega_1]      # s + omega_1

lc_num2 = [omega_2*omega_2]
lc_den2 = [1, 2*zeta2*omega_2, omega_2*omega_2]      # s^2 + 2*s*zeta*omega_0 + omega_0^2

lc_num = np.polymul(lc_num1, lc_num2)
lc_den = np.polymul(lc_den1, lc_den2)

lc_tf = signal.lti(lc_num, lc_den)
print(lc_tf)

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

plt.show()