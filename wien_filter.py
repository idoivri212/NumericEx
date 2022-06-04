from numeric_methods import runge_kutta
import math

# Constants
B = 0.5
R = 3*(10**(-3))
L = 1
q = 1
m = 1
dt = 0.01


def calc_path(E0, Y0):
    A = q / m
    max_t = 2 * math.pi / (A * B)

    t_segments = math.ceil(max_t / dt)

    ay = lambda vz: A * (E0 - vz * B)
    az = lambda vy: A * (vy * B)

    r_kutta, v_kutta = runge_kutta(dt, t_segments, ay, az)
