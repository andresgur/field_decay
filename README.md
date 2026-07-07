# Field Decay
Formulas for magnetic field suppression due to supercritical accretion.



## Installation

### Git Clone
To clone the repository, run the following command:
```bash
git clone https://github.com/andresgur/field-decay .
```

### Pip Install
Navigate to the cloned directory and install the package using pip:
```bash
cd field_decay
pip install .
```

## Usage

The main script `field_suppression.py` solves self-consistently the torque, magnetic field decay and accretion flow geometry (spherization radius, magnetospheric radius, inner disk radius, etc). Three prescriptions for the magnetic field decay are possible:

1. [Zhang & Kojima](https://doi.org/10.1111/j.1365-2966.2005.09802.x) 

2. [Payne & Melatos 2004](https://doi.org/10.1111/j.1365-2966.2004.07798.x), [Payne & Melatos 2007](https://doi.org/10.1111/j.1365-2966.2007.11451.x) 

3. [Shibazaki et al. 1989](https://www.nature.com/articles/342656a0)

The code also takes into account advection. You can switch off by setting $\epsilon_w = 0$.

Run the main script as `python field_suppression.py`. Use `-h` to see the list of input parameters. Parameters are input via a config file (see [example](https://github.com/andresgur/field_decay/blob/main/scripts/magnetic_decay_test.config)). An example can be found in the [scripts](https://github.com/andresgur/field_decay/tree/main/scripts) folder. The output .dat file contains the following columns:
1) Magnetic field in G
2) The spin period in seconds
3) The instantaneous mass accreted onto the NS in Msun/yr
4) Magnetospheric radius (in km)
5) Spherization radius (in km)
6) Co-rotation radius (in km)
7) The torque onto the NS in g cm2 / s2
8) Spin-axis angle (in deg)
9) Magnetic angle (in deg)
10) A flag indicating whether the NS is in propeller (1) or not (0)
11) A flag indicating whether the NS is pulsing (1) or not (0)
12) The time in years

The script  `python create_data.py` can be used to generate some mock data. Along with the config file, you will also need to pass a file with luminosities and period and the epoch of the observations (see [example](https://github.com/andresgur/field_decay/blob/main/scripts/M82_X2.dat)).
