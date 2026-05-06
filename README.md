# field_decay
Formulas for magnetic field suppression due to supercritical accretion.

The main script `field_suppression.py` solves self-consistently the torque, magnetic field decay and accretion flow geometry (spherization radius, magnetospheric radius, inner disk radius, etc). For now, advection is ignored and the spin up of the NS due to accretion is treated rather simply. Three prescriptions for the magnetic field decay are possible:

1. [Zhang & Kojima](https://doi.org/10.1111/j.1365-2966.2005.09802.x) 

2. [Payne & Melatos 2004](https://doi.org/10.1111/j.1365-2966.2004.07798.x), [Payne & Melatos 2007](https://doi.org/10.1111/j.1365-2966.2007.11451.x) 

3. [Shibazaki et al. 1989](https://www.nature.com/articles/342656a0)



Run the main script as `python field_suppression.py`. Use `-h` to see the list of input parameters. The output .dat file contains the following columns:
1) Magnetic field in G
2) The spin period in seconds
3) Mdot_0 the mass-transfer rate (usually constant)
4) The instantaneous mass accreted onto the NS in Msun/yr
5) The critical mass-transfer the accretion column can sustain
6) The mass-transfer rate at the magnetosphere
7) Magnetospheric radius (in km)
8) Inner disk radius ($R_{isco}$, $R_{NS} or $R_{mag}$) (in km)
9) Spherization radius (in km)
10) Co-rotation radius (in km)
11) The torque onto the NS in g cm2 / s2
12) Spin-axis angle (in deg)
13) Magnetic angle (in deg)
14) A flag indicating whether the NS is in propeller (1) or not (0)
15) A flag indicating whether the NS is pulsing (1) or not (0)
16) The time in years
