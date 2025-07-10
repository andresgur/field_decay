from astropy.constants import G, c, M_sun
import astropy.units as u

Gcgs = G.to(u.cm**3/u.g/u.s**2).value
ccgs = c.to(u.cm/u.s).value
M_suncgs = M_sun.to(u.g).value