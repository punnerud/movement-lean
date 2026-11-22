#!/usr/bin/env python3
"""Numbers quoted in section 7, from CODATA 2018 constants."""
import math

c = 299792458.0
h = 6.62607015e-34
hbar = h / (2 * math.pi)
e = 1.602176634e-19
m = 9.1093837015e-31
alpha = 7.2973525693e-3
muB = e * hbar / (2 * m)
mc2 = m * c * c

r = hbar / (2 * m * c)
print(f"1/alpha                         {1/alpha:.3f}")
print(f"spinning ball surface speed     {5/4/alpha:.0f} c")
print(f"circulation radius hbar/2mc     {r:.3e} m")
print(f"reduced Compton wavelength      {hbar/(m*c):.3e} m")
print(f"classical electron radius       {alpha*hbar/(m*c):.3e} m")
print(f"circulation frequency           {c/r/(2*math.pi):.3e} Hz")
print(f"model magnetic moment           {e*c*r/2/muB:.3f} mu_B   (measured 1.00116)")
print(f"circulation speed in hydrogen   {math.sqrt(1-alpha**2):.5f} c")
print(f"model radius / 1e-18 m          {r/1e-18:.0f}")
print(f"fine structure 2P (Dirac)       {alpha**4*mc2/32/h/1e9:.2f} GHz (measured 10.97)")
print(f"Darwin term, hydrogen 1s        {alpha**4*mc2/2/e:.2e} eV")
print(f"alpha / 2 pi                    {alpha/(2*math.pi):.5f}   (measured a_e 0.00116)")
print(f"inner electron speed in gold    {79*alpha:.2f} c")
