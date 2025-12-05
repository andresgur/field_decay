import unittest
from field_decay.constants import M_suncgs, Gcgs, ccgs
from field_decay.compact_objects import CO, NS
from math import pi


class TestCompactObject(unittest.TestCase):
    def test_Mass_setter(self):
        M = 10
        co = CO(M=M, spin=0.0)
        self.assertEqual(co.M, M * M_suncgs)

    def test_isco_schwarzchild(self):
        co = CO(M=10, spin=0)
        Risco_expected = 6  # in units of Rg for a=0
        self.assertAlmostEqual(co.Risco / co.Rg, Risco_expected)

    def test_isco_kerr(self):
        co = CO(M=10, spin=1.0)
        Risco_expected = 1  # in units of Rg for a=0
        self.assertAlmostEqual(co.Risco / co.Rg, Risco_expected)

    def test_efficiency(self):
        co = CO(M=10, spin=0)
        eff = co.accretion_efficiency(co.Risco)
        self.assertAlmostEqual(1 / 12, eff, places=0)

    def test_eddington_luminosity(self):
        M = 10
        co = CO(M=M, spin=0)
        L_edd_expected = 1.26 * M  # in 10**38 erg/s (wikipedia)
        self.assertAlmostEqual(co.LEdd / 10**38, L_edd_expected, delta=0.1)


class TestNeutronStar(unittest.TestCase):
    def test_moment_of_intertia(self):
        P = 2
        M = 1.4
        ns = NS(P, M)
        I = 0.4 * M * M_suncgs * ns.R_NS**2.0
        self.assertAlmostEqual(I / ns.I, 1, delta=0.0001)
        for M in [1.8, 2.0]:
            ns.M = M
            I = 0.4 * M * M_suncgs * ns.R_NS**2.0
            self.assertAlmostEqual(
                I / ns.I, 1, delta=0.0001, msg="Mass setter does not work!"
            )

    def test_period(self):
        P = 2
        M = 1.4
        ns = NS(P, M)
        I = 0.4 * M * M_suncgs * ns.R_NS**2.0
        spin = 2 * pi * I * ccgs / P / Gcgs / (M * M_suncgs) ** 2

        self.assertAlmostEqual(
            spin / ns.spin, 1, delta=0.0001, msg="Spin does not work!"
        )

        omega = 2 * pi / P
        self.assertAlmostEqual(
            omega / ns.omega, 1, delta=0.0001, msg="Omega does not work!"
        )
        for P in [10, 20.0]:
            ns.P = P
            spin = 2 * pi * I * ccgs / P / Gcgs / (M * M_suncgs) ** 2

            self.assertAlmostEqual(
                spin / ns.spin, 1, delta=0.0001, msg="Period setter does not work!"
            )

            omega = 2 * pi / P
            self.assertAlmostEqual(
                omega / ns.omega, 1, delta=0.0001, msg="Period setter does not work!"
            )

    def test_corotation_radius(self):
        P = 2
        M = 1.4
        ns = NS(P, M)
        Rco = (Gcgs * M * M_suncgs * P**2 / 4 / pi**2) ** (1 / 3)

        self.assertAlmostEqual(Rco / ns.Rco, 1, msg="Co-rotation radius does not work!")
        for P in [10, 20.0]:
            ns.P = P
            Rco = (Gcgs * M * M_suncgs * P**2 / 4 / pi**2) ** (1 / 3)
            self.assertAlmostEqual(
                Rco / ns.Rco, 1, msg="Period setter does not update Co-rotation radius!"
            )

    def test_light_cylinder(self):
        P = 2
        M = 1.4
        ns = NS(P, M)
        omega = 2 * pi / P
        Rlc = ccgs / omega

        self.assertAlmostEqual(
            Rlc / ns.Rlc, 1, msg="Light cyclinder radius does not work!"
        )
        for P in [10, 20.0]:
            ns.P = P
            omega = 2 * pi / P
            Rlc = ccgs / omega
            self.assertAlmostEqual(
                Rlc / ns.Rlc, 1, msg="Period setter does not update light cylinder!"
            )

    def test_braking_torque(self):
        P = 2
        M = 1.4
        B = 10**12
        ns = NS(P, M, B=B)
        omega = 2 * pi / P
        Rlc = ccgs / omega
        torque = -ns.magnetic_moment() ** 2.0 / Rlc**3.0

        self.assertAlmostEqual(
            torque / ns.braking_torque(), 1, msg="Braking torque does not work!"
        )
        for B in [10**11, 10**13]:
            ns.B = P
            omega = 2 * pi / P
            Rlc = ccgs / omega
            torque = -ns.magnetic_moment() ** 2.0 / Rlc**3.0
            self.assertAlmostEqual(
                torque / ns.braking_torque(),
                1,
                msg="B setter does not update braking torque!",
            )

        # spin = 2 * pi * self.M * self.R_NS**2.0


if __name__ == "__main__":
    unittest.main()
