from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


R = 0.1
C = 1000.0e-06
L = 0.001 

lc_num = [1]
lc_den = [L*C, R*C, 1]      # s^2*LC + s*RC + 1

lc_tf = signal.lti(lc_num, lc_den)

w, mag, phase = signal.bode(lc_tf, np.arange(100000))

plt.figure()
plt.grid()
plt.semilogx(w,mag)
plt.show()

plt.figure()
plt.grid()
plt.semilogx(w,phase)
plt.show()