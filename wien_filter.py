from numeric_methods import Vector
import math
import numpy as np
import matplotlib.pyplot as plt

# Constants
B = 0.5
R = 0.003
L = 1
q = 1.6 * 10**(-19)
m = 1.672 * 10**(-27)
Mev = 1.6*(10**(-13))
E0 = 5 * Mev
dt = 10**(-8)
dE = 0.25 * Mev
dv = math.sqrt(2*dE/m)
E = math.sqrt((2*E0)/m)*B

A = q / m
ay = lambda vz: A * (E - vz * B)
az = lambda vy: A * (vy * B)


def plot_histog(x, *perc):
    fig, ax = plt.subplots()
    ax.set_xlabel("number of particles")
    ax.set_ylabel("vt(m/s)")
    plt.grid()
    fig.suptitle(f"Velocity distribution for particles with evenly distributed E0, Y0")
    plt.title(f"{perc[0]}% of the particles in the beam passes the filter")
    plt.hist(x)
    plt.show()


def plot_points(v, y):
    fig, ax = plt.subplots()
    ax.set_xlabel("y0/R")
    ax.set_ylabel("dv/v0")
    plt.grid()
    plt.title(f"Initial conditions plane for particles passing the Wien filter")
    plt.scatter(x=y, y=v)
    plt.show()


def plot_paths(good_paths, bad_paths):
    fig, plot_ax = plt.subplots()

    # Plot settings
    plot_ax.set_xlabel("y(m)")
    plot_ax.set_ylabel("z(m)")
    plt.title(f"{len(good_paths) + len(bad_paths)} particles in the Wien filter")

    for gp in good_paths:
        plot_ax.plot(gp.y, gp.z, "green")
    for bp in bad_paths:
        plot_ax.plot(bp.y, bp.z, "red")

    plot_ax.hlines(L, xmin=-R, xmax=0.003, linestyle="-")
    plot_ax.vlines(R, ymin=0, ymax=L)
    plot_ax.vlines(-R, ymin=0, ymax=L)

    plt.xlim(-1.5*R, 1.5*R)
    plt.ylim(0, L*1.5)

    plt.show()


def one_particle(E0, Y0):
    r, v = Vector(1), Vector(1)

    # Initial Conditions
    v.y[0] = 0
    v.z[0] = math.sqrt(2 * E0 / m)
    r.y[0] = Y0
    r.z[0] = 0
    i = 1

    while r.z[i - 1] < 1 and -R <= r.y[i - 1] <= R:

        k1vz = dt * az(v.y[i - 1])
        k1vy = dt * ay(v.z[i - 1])
        k2vz = dt * az(v.y[i - 1] + 0.5 * k1vy)
        k2vy = dt * ay(v.z[i - 1] + 0.5 * k1vz)
        k3vz = dt * az(v.y[i - 1] + 0.5 * k2vy)
        k3vy = dt * ay(v.z[i - 1] + 0.5 * k2vz)
        k4vz = dt * az(v.y[i - 1] + k3vy)
        k4vy = dt * ay(v.z[i - 1] + k3vz)

        k1rz = dt * (v.z[i - 1])
        k1ry = dt * (v.y[i - 1])
        k2rz = dt * (v.z[i - 1] + 0.5 * k1ry)
        k2ry = dt * (v.y[i - 1] + 0.5 * k1rz)
        k3rz = dt * (v.z[i - 1] + 0.5 * k2ry)
        k3ry = dt * (v.y[i - 1] + 0.5 * k2rz)
        k4rz = dt * (v.z[i - 1] + k3ry)
        k4ry = dt * (v.y[i - 1] + k3rz)

        r.z = np.append(r.z, [r.z[i - 1] + 1 / 6 * (k1rz + 2 * k2rz + 2 * k3rz + k4rz)])
        r.y = np.append(r.y, r.y[i - 1] + 1 / 6 * (k1ry + 2 * k2ry + 2 * k3ry + k4ry))
        v.y = np.append(v.y, v.y[i - 1] + 1 / 6 * (k1vy + 2 * k2vy + 2 * k3vy + k4vy))
        v.z = np.append(v.z, v.z[i - 1] + 1 / 6 * (k1vz + 2 * k2vz + 2 * k3vz + k4vz))
        i += 1

    return (r, v), -R < r.y[i - 1] < R


def question_6(n_particles):

    E0_vec = np.linspace(E0-dE, E0+dE, num=n_particles)
    Y0_vec = np.linspace(-R, R, num=n_particles)

    fail_r, fail_v = [], []
    suc_r, suc_v = [], []

    for n in range(n_particles):
        vectors, rez = one_particle(E0_vec[n], Y0_vec[n])

        if rez:
            suc_r += [vectors[0]]
            suc_v += [vectors[1]]
        else:
            fail_r += [vectors[0]]
            fail_v += [vectors[1]]

    plot_paths([suc_r[0]], [fail_r[round(len(fail_r)/2)]])


def question_7(n_particles):
    v_pts, y_pts = [], []
    E0_vec = np.linspace(E0 - dE, E0 + dE, num=n_particles)
    Y0_vec = np.linspace(-R, R, num=n_particles)

    for e_init in E0_vec:
        for y_init in Y0_vec:
            vectors, rez = one_particle(e_init, y_init)
            if rez:
                v0 = vectors[1].z[0]
                y0 = vectors[0].y[0]
                y_pts.append(y0 / R)
                v_pts.append(dv / v0)
    plot_points(v_pts, y_pts)

def question_8(n_particles):
    velocities = []
    for i in range(n_particles):
        e0 = np.random.uniform(E0 - dE, E0 + dE)
        y0 = np.random.uniform(-R, R)
        vectors, rez = one_particle(e0, y0)
        if rez:
            vyt = vectors[1].y[len(vectors[1].y)-1]
            vzt = vectors[1].z[len(vectors[1].y) - 1]
            velocities.append(math.sqrt((vzt**2)+(vyt**2)))

    perc = round((len(velocities)/n_particles), 5) * 100
    plot_histog(velocities, perc)


if __name__ == '__main__':
    # question_6(100)
    # question_7(100)
    question_8(10**5)

