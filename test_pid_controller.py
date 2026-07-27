"""Unit tests for the reusable timestamp-aware PID controller."""

import unittest

from control.pid import PIDConfig, PIDController


class PIDControllerTests(unittest.TestCase):
    def test_output_is_clamped(self):
        pid = PIDController(
            PIDConfig(kp=10.0, output_min=-1.0, output_max=1.0)
        )
        self.assertEqual(pid.update(2.0, 0.02), 1.0)
        self.assertEqual(pid.update(-2.0, 0.02), -1.0)

    def test_integrator_does_not_wind_up_while_saturated(self):
        pid = PIDController(
            PIDConfig(
                kp=2.0,
                ki=5.0,
                output_min=-1.0,
                output_max=1.0,
                integral_min=-0.5,
                integral_max=0.5,
            )
        )
        for _ in range(100):
            pid.update(1.0, 0.02)
        self.assertEqual(pid.integral, 0.0)
        self.assertLess(pid.update(-0.1, 0.02), 0.0)

    def test_integral_is_clamped_without_output_saturation(self):
        pid = PIDController(
            PIDConfig(
                kp=0.0,
                ki=1.0,
                output_min=-10.0,
                output_max=10.0,
                integral_min=-0.2,
                integral_max=0.2,
            )
        )
        for _ in range(100):
            pid.update(0.1, 0.1)
        self.assertAlmostEqual(pid.integral, 0.2)

    def test_derivative_on_measurement_has_expected_sign(self):
        pid = PIDController(PIDConfig(kp=0.0, kd=0.5))
        self.assertAlmostEqual(
            pid.update(0.0, 0.02, measurement_rate=2.0),
            -1.0,
        )

    def test_invalid_input_resets_to_safe_zero(self):
        pid = PIDController(PIDConfig(kp=1.0, ki=1.0))
        pid.update(0.5, 0.1)
        self.assertNotEqual(pid.integral, 0.0)
        self.assertEqual(pid.update(float("nan"), 0.1), 0.0)
        self.assertEqual(pid.integral, 0.0)


if __name__ == "__main__":
    unittest.main()
