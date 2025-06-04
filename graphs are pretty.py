import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# === MODEL PARAMETERS ===
nu = 0.2         # vaccination rate 
beta_H = 1.0     # host infection rate 
beta_V = 0.2     # vector infection rate 
delta = 0.4      # recovery rate 
epsilon = 0.15    # vaccine efficacy = 1-epsilon

# === INITIAL CONDITIONS ===

sH0 = 1
iH0 = 0
vH0 = 0.0
rH0 = 0.0
sV0 = 0.99
iV0 = 0.01

y0 = [sH0, iH0, vH0, rH0, sV0, iV0]

# === ODES ===
def odes(tau, y):
    sH, iH, vH, rH, sV, iV = y
    dsH_dtau = -iV * sH - (nu / beta_H) * sH
    diH_dtau = iV * sH + epsilon * vH * iH - (delta / beta_H) * iH
    dvH_dtau = (nu / beta_H) * sH - epsilon * vH * iH
    drH_dtau = (delta / beta_H) * iH
    dsV_dtau = - (beta_V / beta_H) * iH * sV
    diV_dtau = (beta_V / beta_H) * iH * sV
    return [dsH_dtau, diH_dtau, dvH_dtau, drH_dtau, dsV_dtau, diV_dtau]

# === TIME GRID ===
t_span = (0, 365)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# === SOLVE ODES ===
sol = solve_ivp(odes, t_span, y0, t_eval=t_eval, method='RK45')

# === PLOT GRAPHS ===
plt.figure(figsize=(10, 7))
plt.plot(sol.t, sol.y[0], label='$s_H$ (Susceptible Host)')
plt.plot(sol.t, sol.y[1], label='$i_H$ (Infected Host)')
plt.plot(sol.t, sol.y[2], label='$v_H$ (Vaccinated Host)')
plt.plot(sol.t, sol.y[3], label='$r_h$ (Recovered Host)')
plt.plot(sol.t, sol.y[4], linestyle = ":", label='$s_V$ (Susceptible Vector)')
plt.plot(sol.t, sol.y[5], linestyle = ":", label='$i_V$ (Infected Vector)')
plt.xlabel('Non-dimensional time $\\tau$')
plt.ylabel('Fraction of population')
plt.title('Dynamics of Vector-Borne Disease Model')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === STABILITY TEST: TRY DIFFERENT NU VALUES ===
nus = np.arange(0,1.25,0.25)
for nu_test in nus:
    def odes_nu(tau, y):
        sH, iH, vH, rH, sV, iV = y
        dsH_dtau = -iV * sH - (nu_test / beta_H) * sH
        diH_dtau = iV * sH + epsilon * vH * iH - (delta / beta_H) * iH
        dvH_dtau = (nu_test / beta_H) * sH - epsilon * vH * iH
        drH_dtau = (delta / beta_H) * iH
        dsV_dtau = - (beta_V / beta_H) * iH * sV
        diV_dtau = (beta_V / beta_H) * iH * sV
        return [dsH_dtau, diH_dtau, dvH_dtau, drH_dtau, dsV_dtau, diV_dtau]
    sol = solve_ivp(odes_nu, t_span, y0, t_eval=t_eval, method='RK45')
    plt.plot(sol.t, sol.y[1], label=f'nu={nu_test}')
    plt.xlabel('Non-dimensional time $\\tau$')
    plt.ylabel('$i_H$ (Infected Host)')
    plt.title('Effect of Vaccination Rate $\\nu$ on Infection')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === STABILITY TEST: TRY DIFFERENT DELTA VALUES ===
nus = np.arange(0,1.25,0.25)
for nu_test in nus:
    def odes_nu(tau, y):
        sH, iH, vH, rH, sV, iV = y
        dsH_dtau = -iV * sH - (nu_test / beta_H) * sH
        diH_dtau = iV * sH + epsilon * vH * iH - (delta / beta_H) * iH
        dvH_dtau = (nu_test / beta_H) * sH - epsilon * vH * iH
        drH_dtau = (delta / beta_H) * iH
        dsV_dtau = - (beta_V / beta_H) * iH * sV
        diV_dtau = (beta_V / beta_H) * iH * sV
        return [dsH_dtau, diH_dtau, dvH_dtau, drH_dtau, dsV_dtau, diV_dtau]
    sol = solve_ivp(odes_nu, t_span, y0, t_eval=t_eval, method='RK45')
    plt.plot(sol.t, sol.y[1], label=f'nu={nu_test}')
    plt.xlabel('Non-dimensional time $\\tau$')
    plt.ylabel('$i_H$ (Infected Host)')
    plt.title('Effect of Vaccination Rate $\\nu$ on Infection')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === STABILITY TEST: TRY DIFFERENT BETA_V VALUES ===
nus = np.arange(0,1.25,0.25)
for nu_test in nus:
    def odes_nu(tau, y):
        sH, iH, vH, rH, sV, iV = y
        dsH_dtau = -iV * sH - (nu_test / beta_H) * sH
        diH_dtau = iV * sH + epsilon * vH * iH - (delta / beta_H) * iH
        dvH_dtau = (nu_test / beta_H) * sH - epsilon * vH * iH
        drH_dtau = (delta / beta_H) * iH
        dsV_dtau = - (beta_V / beta_H) * iH * sV
        diV_dtau = (beta_V / beta_H) * iH * sV
        return [dsH_dtau, diH_dtau, dvH_dtau, drH_dtau, dsV_dtau, diV_dtau]
    sol = solve_ivp(odes_nu, t_span, y0, t_eval=t_eval, method='RK45')
    plt.plot(sol.t, sol.y[1], label=f'nu={nu_test}')
    plt.xlabel('Non-dimensional time $\\tau$')
    plt.ylabel('$i_H$ (Infected Host)')
    plt.title('Effect of Vaccination Rate $\\nu$ on Infection')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# === STABILITY TEST: TRY EPSILON NU VALUES ===
nus = np.arange(0,1.25,0.25)
for nu_test in nus:
    def odes_nu(tau, y):
        sH, iH, vH, rH, sV, iV = y
        dsH_dtau = -iV * sH - (nu_test / beta_H) * sH
        diH_dtau = iV * sH + epsilon * vH * iH - (delta / beta_H) * iH
        dvH_dtau = (nu_test / beta_H) * sH - epsilon * vH * iH
        drH_dtau = (delta / beta_H) * iH
        dsV_dtau = - (beta_V / beta_H) * iH * sV
        diV_dtau = (beta_V / beta_H) * iH * sV
        return [dsH_dtau, diH_dtau, dvH_dtau, drH_dtau, dsV_dtau, diV_dtau]
    sol = solve_ivp(odes_nu, t_span, y0, t_eval=t_eval, method='RK45')
    plt.plot(sol.t, sol.y[1], label=f'nu={nu_test}')
    plt.xlabel('Non-dimensional time $\\tau$')
    plt.ylabel('$i_H$ (Infected Host)')
    plt.title('Effect of Vaccination Rate $\\nu$ on Infection')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

