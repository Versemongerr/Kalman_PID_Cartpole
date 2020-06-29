# Kalman_PID_Cartpole
Cartpole with PID and Kalman filter, for DSP Final.
Measured the speed of the Cart and the angle speed of the Pole with some normal random noise, and then estimate the offset of the cart and the angle of the pole.
The model used in the prediction part of Kalman Filter is a linear approximation,where sin(theta)~0,cos(theta)~1.
A mathmatical derivation is provided in Kalman.md in Chineses.
Rough PID parameters were set,because I only need a nice Kalman Filter plot.Adjustment to the parameters can achieve better performance.
