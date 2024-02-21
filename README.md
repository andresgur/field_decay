# field_decay
Formulas for magnetic field suppression due to supercritical accretion.

The main script `field_suppression.py` solves self-consistently the torque, magnetic field decay and accretion flow geometry (spherization radius, magnetospheric radius, inner disk radius, etc). For now, advection is ignored and the spin up of the NS due to accretion is treated rather simply. Several prescriptions for the magnetic field decay are possible:

[Zhang & Kojima](https://doi.org/10.1111/j.1365-2966.2005.09802.x) 

[Payne & Melatos 2004](https://doi.org/10.1111/j.1365-2966.2004.07798.x), [Payne & Melatos 2007](https://doi.org/10.1111/j.1365-2966.2007.11451.x) 

[Shibazaki et al. 1989](https://www.nature.com/articles/342656a0) --> Named Igoshed in the code after the 2021 review



Run the main script as `python field_suppression.py`. Use `-h` to see the list of input parameters.
