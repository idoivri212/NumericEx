import matplotlib.pyplot as plt

# Constants
dt = 0.01


def explicit_euler_one_dimension(x0, t0, df, dt, max_t):
    values = {}
    t = t0
    x = x0

    while t <= max_t:
        values[t] = x
        x += df(t) * dt
        t += dt

    return values


def heun_one_dimension(x0, t0, df, dt, max_t):
    values = {}
    t = t0
    x = x0

    while t <= max_t:
        values[t] = x
        x += (0.5 * df(t) + 0.5 * df(t + dt)) * dt
        t += dt

    return values


def calculate_orig_f_values(t0, f, dt, max_t):
    values = {}
    t = t0

    while t <= max_t:
        values[t] = f(t)
        t += dt

    return values


def plot_first_part(plot_ax, x0, f, f_dot, max_t):
    euler_results = explicit_euler_one_dimension(x0, 0, f_dot, dt, max_t)
    appx_f_euler = euler_results

    appx_f_heun = heun_one_dimension(x0, 0, f_dot, dt, max_t)

    orig_f_values = calculate_orig_f_values(0, f, dt, max_t)

    plot(plot_ax, [appx_f_euler, appx_f_heun, orig_f_values], ['r', 'g', 'b'], ["Explicit Euler", "Heun", "Ground Truth"])


def plot(ax, dicts, colors, labels):
    for i in range(len(dicts)):
        ax.plot(*zip(*sorted(dicts[i].items())), colors[i], label=labels[i])


fig, plot_ax = plt.subplots()
plot_first_part(plot_ax, 0, lambda x: x**2, lambda x: 2*x, 20)

plot_ax.legend(["Explicit Euler", "Heun", "Ground Truth"])
plt.show()
