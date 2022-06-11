import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Constants
E = 5
B = 1
q = 1
m = 1
max_t = 20
analytic_endpoint = analytic = (0, (2 * m * E * math.pi) / (q * B ** 2))
analytic_rz = lambda t: (E / B) * (t - (2 * m / (q * B)) * np.cos(q * B * t / m + np.pi / 2))
analytic_ry = lambda t: (2 * E * m / (q * B * B)) * (np.sin(q * B * t / m + np.pi / 2) - 1)


class Vector:
    def __init__(self, size):
        self.y = np.zeros(size)
        self.z = np.zeros(size)


def compare_euler(dt):
    global max_t
    A = q / m
    max_t = 2 * math.pi / (A * B)

    t_segments = math.ceil(max_t / dt)

    ay = lambda vz: A * (E - vz * B)
    az = lambda vy: A * (vy * B)

    r_euler, v_euler = explicit_euler(dt, t_segments, ay, az)
    fig, plot_ax = plt.subplots()

    # Plot Location
    plot_ax.plot(r_euler.y, r_euler.z, "blue")
    plot_ax.plot(analytic_ry, analytic_rz, "red")
    plot_ax.legend(["Explicit Euler", "Analytic"])
    plot_ax.set_xlabel("y(m)")
    plot_ax.set_ylabel("z(m)")
    plt.title("Euler's method vs. the analytic solution")
    plt.show()


def explicit_euler(dt, t_seg, ay, az):
    r, v = Vector(t_seg), Vector(t_seg)
    # Initial Conditions
    v.y[0] = 0
    v.z[0] = 3 * E / B
    r.y[0] = 0
    r.z[0] = 0
    for i in range(1, t_seg):
        r.y[i] = r.y[i - 1] + dt * v.y[i - 1]
        r.z[i] = r.z[i - 1] + dt * v.z[i - 1]
        v.z[i] = v.z[i - 1] + dt * az(v.y[i - 1])
        v.y[i] = v.y[i - 1] + dt * ay(v.z[i - 1])

    return r, v


def midpoint(dt, t_seg, ay, az):
    r, v = Vector(t_seg), Vector(t_seg)
    # Initial Conditions
    v.y[0] = 0
    v.z[0] = 3 * E / B
    r.y[0] = 0
    r.z[0] = 0
    for i in range(1, t_seg):
        k1vy = ay(v.z[i - 1]) * dt
        k1vz = az(v.y[i - 1]) * dt
        k2vz = az(v.y[i - 1] + 0.5 * k1vy) * dt
        k2vy = ay(v.z[i - 1] + 0.5 * k1vz) * dt

        k1ry = v.y[i - 1] * dt
        k1rz = v.z[i - 1] * dt
        k2rz = (v.z[i - 1] + 0.5 * k1rz) * dt
        k2ry = (v.y[i - 1] + 0.5 * k1ry) * dt

        r.z[i] = r.z[i - 1] + k2rz
        r.y[i] = r.y[i - 1] + k2ry
        v.z[i] = v.z[i - 1] + k2vz
        v.y[i] = v.y[i - 1] + k2vy

    return r, v


def runge_kutta(dt, t_seg, ay, az):
    r, v = Vector(t_seg), Vector(t_seg)
    # Initial Conditions
    v.y[0] = 0
    v.z[0] = 3 * E / B
    r.y[0] = 0
    r.z[0] = 0
    for i in range(1, t_seg):
        k1vz = dt * az(v.y[i - 1])
        k1vy = dt * ay(v.z[i - 1])

        k1rz = dt * (v.z[i - 1])
        k1ry = dt * (v.y[i - 1])

        k2vy = dt * ay(v.z[i - 1] + 0.5 * k1vz)
        k2vz = dt * az(v.y[i - 1] + 0.5 * k1vy)

        k2ry = dt * (v.y[i - 1] + 0.5 * k1vy)
        k2rz = dt * (v.z[i - 1] + 0.5 * k1vz)

        k3vz = dt * az(v.y[i - 1] + 0.5 * k2vy)
        k3vy = dt * ay(v.z[i - 1] + 0.5 * k2vz)

        k3rz = dt * (v.z[i - 1] + 0.5 * k2vz)
        k3ry = dt * (v.y[i - 1] + 0.5 * k2vy)

        k4vz = dt * az(v.y[i - 1] + k3vy)
        k4vy = dt * ay(v.z[i - 1] + k3vz)

        k4rz = dt * (v.z[i - 1] + k3vz)
        k4ry = dt * (v.y[i - 1] + k3vy)

        r.z[i] = r.z[i - 1] + (k1rz + 2 * k2rz + 2 * k3rz + k4rz) / 6
        r.y[i] = r.y[i - 1] + (k1ry + 2 * k2ry + 2 * k3ry + k4ry) / 6
        v.y[i] = v.y[i - 1] + (k1vy + 2 * k2vy + 2 * k3vy + k4vy) / 6
        v.z[i] = v.z[i - 1] + (k1vz + 2 * k2vz + 2 * k3vz + k4vz) / 6

    return r, v


def plot(ax, dicts, colors, labels):
    for i in range(len(dicts)):
        ax.plot(*zip(*sorted(dicts[i].items())), colors[i], label=labels[i])


def advance(dt, show=False):
    global max_t
    A = q / m
    max_t = 2 * math.pi / (A * B)

    t_segments = math.ceil(max_t / dt)

    ay = lambda vz: A * (E - vz * B)
    az = lambda vy: A * (vy * B)

    r_euler, v_euler = explicit_euler(dt, t_segments, ay, az)
    print("done euler")
    r_mid, v_mid = midpoint(dt, t_segments, ay, az)
    print("done mid")
    r_kutta, v_kutta = runge_kutta(dt, t_segments, ay, az)
    print("done kutta")

    times = np.linspace(0, max_t, t_segments)
    ry, rz = [], []
    for t in times:
        ry.append(analytic_ry(t))
        rz.append(analytic_rz(t))

    """print(f"Euler endpoint- ({r_euler.y[len(r_euler.y)-1], r_euler.z[len(r_euler.z)-1]})")
    print(f"Kutta endpoint- ({r_kutta.y[len(r_kutta.y) - 1], r_kutta.z[len(r_kutta.z) - 1]})")
    print(f"Midpoint endpoint- ({r_mid.y[len(r_mid.y) - 1], r_mid.z[len(r_mid.z) - 1]})")"""

    if show:
        fig, plot_ax = plt.subplots(1, 2)

        # Plot Location
        plot_ax[0].plot(r_euler.y, r_euler.z, "blue")
        plot_ax[0].plot(r_mid.y, r_mid.z, "red")

        # plot_ax[0].plot(ry, rz, "yellow")
        plot_ax[0].plot(r_kutta.y, r_kutta.z, "green")

        plot_ax[0].legend(["Explicit Euler", "Midpoint", "Runge-Kutta", "analytic"])
        plot_ax[0].set_xlabel("y(m)")
        plot_ax[0].set_ylabel("z(m)")

        # Plot Velocities
        plot_ax[1].plot(v_euler.y, v_euler.z, "blue")
        plot_ax[1].plot(v_mid.y, v_mid.z, "red")
        plot_ax[1].plot(v_kutta.y, v_kutta.z, "green")
        plot_ax[1].set_xlabel("vy(m/s)")
        plot_ax[1].set_ylabel("vz(m/s)")
        plot_ax[1].legend(["Explicit Euler", "Midpoint", "Runge-Kutta", "analytic"])

        plt.title(f"dt = {dt}")
        plt.show()

    return r_euler, r_mid, r_kutta


def graph_errors(start, stop, n_vals):
    error = lambda r: math.sqrt(
        ((r.y[len(r.y) - 1] - analytic_endpoint[0]) ** 2) + ((r.z[len(r.z) - 1] - analytic_endpoint[1]) ** 2))

    fig, plot_ax = plt.subplots()
    plot_ax.set_yscale('log')
    plt.grid()
    plot_ax.set_xscale('log')

    euler, mid, kutta = {}, {}, {}
    times = np.linspace(start=start, num=n_vals, stop=stop)
    for dt in tqdm(times):
        vectors = advance(dt)
        print("hi")
        euler[dt], mid[dt], kutta[dt] = error(vectors[0]), error(vectors[1]), error(vectors[2])

    plot_ax.plot(euler.keys(), euler.values(), "blue")
    plot_ax.plot(mid.keys(), mid.values(), "red")
    plot_ax.plot(kutta.keys(), kutta.values(), "green")
    plot_ax.legend(["Explicit Euler", "Midpoint", "Runge-Kutta"])
    plt.title("Distance from analytic endpoint for 3 approximation methods, as function of dt")
    plot_ax.set_xlabel("dt(s)")
    plot_ax.set_ylabel("distance(m)")
    plt.show()


if __name__ == '__main__':
    # advance(10**-5, True)
    graph_errors(10**-5, 10**-2, 100)
