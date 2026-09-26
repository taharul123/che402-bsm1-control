"""
pi_controller.py
A simple, reusable discrete-time PI controller with anti-windup.

u(t) = bias + Kp * e(t) + (Kp/Ti) * integral(e)

Anti-windup: if the computed output would exceed the actuator limits, we do NOT
add to the integral term that step (this is the simplest, "clamping" style of
anti-windup, and is good enough for this assignment).
"""


class PIController:
    def __init__(self, kp, ti, dt, u_min, u_max, bias=0.0):
        """
        kp   : proportional gain
        ti   : integral time [days]
        dt   : controller timestep [days] (should match the simulation timestep)
        u_min, u_max : actuator limits (the controller output will never leave this range)
        bias : nominal/baseline actuator value the controller adjusts around
        """
        self.kp = kp
        self.ti = ti
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max
        self.bias = bias
        self.integral = 0.0

    def update(self, setpoint, measurement):
        error = setpoint - measurement
        integral_candidate = self.integral + error * self.dt

        p_term = self.kp * error
        i_term = (self.kp / self.ti) * integral_candidate
        u = self.bias + p_term + i_term

        if u > self.u_max:
            u = self.u_max
            # don't commit the integral update -> anti-windup
        elif u < self.u_min:
            u = self.u_min
            # don't commit the integral update -> anti-windup
        else:
            self.integral = integral_candidate

        return u, error


def simc_tuning(k, tau, theta, tc=None):
    """
    SIMC (Skogestad) PI tuning rule.
    k, tau, theta : FOPDT model parameters (gain, time constant, dead time)
    tc : desired closed-loop time constant. Defaults to theta (a common,
         reasonably aggressive-but-safe choice).

    Returns (kp, ti).
    """
    if tc is None:
        tc = theta
    kp = (1.0 / k) * (tau / (theta + tc))
    ti = min(tau, 4 * (tc + theta))
    return kp, ti