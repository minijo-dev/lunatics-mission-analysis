import numpy as np
 
"""TO BE ITERATED"""
I_tot = 0.06 #kgm^2
w_tot = 0.035 #rad/s slew rate

motor = 4500 # RPM

rho = 2780 # material density
v = 0.33 # poissons ratio
ult_stress = 469e6
"""""" 
# required torque from motor
tau = I_tot*w_tot
print("Required torque",tau)


w_fly =(motor*360/60)*np.pi/180
print(w_fly)

I_fly_ideal = (I_tot*w_tot)/w_fly # ideal flywheel moment of inertia given the motor speed)

""" ITERATED DIMENSIONS"""
h_ring = 0.01
h_disk = 0.002
r_ring = 0.02
r_disk = 0.015
""""""
# mass of flywheel
m_fly = rho*np.pi*(r_disk**2 *h_disk + (r_ring**2 - r_disk**2)*h_ring)

I_fly_actual = (rho*np.pi/2)*(h_ring*(r_ring**4 - r_disk**4) + h_disk* r_disk**4)

if I_fly_actual > I_fly_ideal:
    print("Flywheel dimensions is sufficient")
    print("Motor RPM:",motor, "RPM")
    print("Mass of flywheel", m_fly, "kg")

# checking structural stress of flywheel
sigma_max = ((3+v)/4) * rho*w_fly**2*(r_ring**2 + ((1-v)/(3+v))*r_disk**2)

if sigma_max>ult_stress:
    print("WARNING: Ultimate stress exceeded")