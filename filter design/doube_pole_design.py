from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

# omega_0 = 1700 rad /s which is half of 3400 rad/s or 550 Hz
#input_voltage_signal is modified to see output against noise input for unity magnitude of signal

omega_0 = 1700  # 0.5* 11th harmonic(550Hz) units-rad/s
zeta = 0.5      #damping ratio

lc_num1 = [np.float64(omega_0*omega_0)]
lc_den1 = [1, 2*zeta*omega_0, omega_0*omega_0]      # s^2 + 2*s*zeta*omega_0 + omega_0^2

lc_num2 = [np.float64(omega_0*omega_0)]
lc_den2 = [1, 2*zeta*omega_0, omega_0*omega_0]      # s^2 + 2*s*zeta*omega_0 + omega_0^2

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