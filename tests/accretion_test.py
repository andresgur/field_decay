import unittest
from field_decay.constants import M_suncgs, Gcgs, ccgs
from field_decay.accretion import (
    spherization_radius_poutanen,
    mass_transfer_inner_radius,
)
from math import pi


class TestAccretionFlow(unittest.TestCase):
    def test_Poutanen(self):
        mdot = 1000
        ewind = 0.5
        Rin = 10
        Rsph = spherization_radius_poutanen(mdot, Rin, ewind) / Rin / mdot
        # see Poutanen below Eq 23
        self.assertAlmostEqual(
            Rsph, 1.16, msg="Spherization radius is wrong!", places=2
        )
        min = mass_transfer_inner_radius(mdot, ewind)
        self.assertAlmostEqual(
            min, 0.66, msg="Mass transfer inner radius does not work!", places=2
        )

        ewind = 1
        Rsph = spherization_radius_poutanen(mdot, Rin, ewind) / Rin / mdot
        self.assertAlmostEqual(
            Rsph, 1.04, msg="Spherization radius is wrong!", places=2
        )
        min = mass_transfer_inner_radius(mdot, ewind)
        self.assertAlmostEqual(
            min, 0.43, msg="Mass transfer inner radius does not work!", places=2
        )


if __name__ == "__main__":
    unittest.main()
