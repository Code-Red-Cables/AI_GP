"""Unit tests for the reusable timestamp-aware PID controller."""

import math
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


class VioControlLoopTests(unittest.TestCase):
    """Closed-loop sanity checks for the VIO yaw-hold and thrust loops.

    These simulate the plant offline with the deployed config.py gains, so a
    gain/sign regression fails here before it reaches the simulator.
    """

    def test_thrust_pi_arrests_descent_and_trims_to_hover(self):
        import config

        pid = PIDController(
            PIDConfig(
                kp=config.KP_THRUST_VEL,
                ki=config.KI_THRUST_VEL,
                output_min=config.VIO_THRUST_MIN - config.MAX_THRUST,
                output_max=config.VIO_THRUST_MAX,
                integral_min=-config.THRUST_INTEGRAL_LIMIT,
                integral_max=config.THRUST_INTEGRAL_LIMIT,
            )
        )
        # Plant fit from the Q2_pnp branch logs: accel_down = -58.2*thrust
        # + 15.57, i.e. true hover thrust ~0.2675 (below HOVER_THRUST=0.27,
        # so the integral must trim the residual).
        vz = 1.0        # NED down: starts descending at 1 m/s
        dt = 0.01
        thrust = config.HOVER_THRUST
        for _ in range(800):
            thrust = config.HOVER_THRUST + pid.update(vz - 0.0, dt)
            thrust = max(
                config.VIO_THRUST_MIN, min(config.VIO_THRUST_MAX, thrust)
            )
            accel_down = -58.2 * thrust + 15.57
            vz += accel_down * dt
        self.assertLess(abs(vz), 0.05)
        self.assertAlmostEqual(thrust, 15.57 / 58.2, delta=0.01)

    def test_thrust_integral_state_is_bounded(self):
        import config

        pid = PIDController(
            PIDConfig(
                kp=config.KP_THRUST_VEL,
                ki=config.KI_THRUST_VEL,
                output_min=config.VIO_THRUST_MIN - config.MAX_THRUST,
                output_max=config.VIO_THRUST_MAX,
                integral_min=-config.THRUST_INTEGRAL_LIMIT,
                integral_max=config.THRUST_INTEGRAL_LIMIT,
            )
        )
        # A blind stretch with a persistent phantom descent must not wind the
        # collective past the clamp.
        for _ in range(5000):
            pid.update(3.0, 0.01)
        self.assertLessEqual(pid.integral, config.THRUST_INTEGRAL_LIMIT)
        self.assertLessEqual(
            config.KI_THRUST_VEL * pid.integral,
            config.KI_THRUST_VEL * config.THRUST_INTEGRAL_LIMIT + 1e-9,
        )


if __name__ == "__main__":
    unittest.main()
