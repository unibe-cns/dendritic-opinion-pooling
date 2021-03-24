import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle

import figure_config

mpl.rcParams.update(figure_config.mpl_style)


def gaussian_density(x, mu, sigma):
    return 1. / np.sqrt(2. * np.pi * sigma ** 2) * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))


def plot_rates(ax_early, ax_late):
    ax_early.set_ylabel(r"$r$ (1/s)", fontsize=figure_config.fontsize_tiny)
    ax_early.set_ylim(plot_params["ylim_rate"])
    ax_early.set_xlabel("Time (s)", fontsize=figure_config.fontsize_tiny)
    ax_early.set_xlim(xlim_early)

    ax_late.set_yticks([])
    ax_late.spines["left"].set_visible(False)
    ax_late.set_ylim(plot_params["ylim_rate"])
    ax_late.set_xticks([1897, 1899])

    ax_early.plot(
        times[indices_times_start], res["r_in"][indices_times_start], color="k",
    )
    ax_early.plot(
        times[indices_times_start],
        res["r_in_noisy"][indices_times_start, 1],
        color=figure_config.colors["T"],
    )
    ax_early.plot(
        times[indices_times_start],
        res["r_in_noisy"][indices_times_start, 0],
        color=figure_config.colors["V"],
    )

    ax_late.plot(
        times[indices_times_late], res["r_in"][indices_times_late], color="k",
    )
    ax_late.plot(
        times[indices_times_late],
        res["r_in_noisy"][indices_times_late, 1],
        color=figure_config.colors["T"],
    )
    ax_late.plot(
        times[indices_times_late],
        res["r_in_noisy"][indices_times_late, 0],
        color=figure_config.colors["V"],
    )


def plot_potential(ax_early, ax_late):

    ax_early.set_ylabel("Membrane potential (mV)", fontsize=figure_config.fontsize_tiny)
    ax_early.set_ylim(plot_params["ylim_potential"])
    ax_early.set_xlim(xlim_early)
    ax_early.set_xticklabels([])

    ax_late.set_xticklabels([])
    ax_late.set_yticks([])
    ax_late.spines["left"].set_visible(False)
    ax_late.set_ylim(plot_params["ylim_potential"])

    ax_early.plot(
        times[indices_times_early][1:],
        res["u0_target_sample"][indices_times_early][1:],
        color="k",
    )
    ax_early.plot(
        times[indices_times_before],
        res["u0_sample"][indices_times_before],
        color=figure_config.colors["VT"],
        ls=':',
    )
    ax_early.plot(
        times[indices_times_early],
        res["u0_sample"][indices_times_early],
        color=figure_config.colors["VT"],
        ls='--',
    )

    ax_early.axvline(0.0, color="k", ls="--")
    ax_early.annotate(
        "teacher present",
        xy=(0.3, 1.05),
        xycoords="axes fraction",
        xytext=(0.3, 1.05),
        textcoords="axes fraction",
        fontsize=figure_config.fontsize_tiny,
    )

    ax_late.plot(
        times[indices_times_late],
        res["u0_target_sample"][indices_times_late],
        color="k",
    )
    ax_late.plot(
        times[indices_times_late],
        res["u0_sample"][indices_times_late],
        color=figure_config.colors["VT"],
    )


def plot_dist(ax):
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(plot_params["ylim_potential"])
    ax.set_yticks([])

    x = np.linspace(-85.0, -50.0, 500)

    mu_target = np.mean(res["u0_target"][indices_times_late])
    sigma_target = 1.0 / np.sqrt(np.mean(res["g0_target"][indices_times_late]))
    # print('teacher mu/sigma', mu_target, sigma_target)
    mu_before = np.mean(res["u0"][indices_times_before])
    sigma_before = 1.0 / np.sqrt(np.mean(res["g0"][indices_times_before]))
    mu_initial = np.mean(res["u0"][indices_times_early])
    sigma_initial = 1.0 / np.sqrt(np.mean(res["g0"][indices_times_early]))
    mu_final = np.mean(res["u0"][indices_times_late])
    sigma_final = 1.0 / np.sqrt(np.mean(res["g0"][indices_times_late]))
    # print('final mu/sigma', mu_final, sigma_final)
    ax.plot(
        gaussian_density(x, mu_target, sigma_target), x, color="k", lw=3.0,
    )
    ax.plot(
        gaussian_density(x, mu_before, sigma_before,),
        x,
        color=figure_config.colors["VT"],
        lw=2,
        ls=":",
    )
    ax.plot(
        gaussian_density(x, mu_initial, sigma_initial,),
        x,
        color=figure_config.colors["VT"],
        lw=2,
        ls="--",
    )
    ax.plot(
        gaussian_density(x, mu_final, sigma_final,),
        x,
        color=figure_config.colors["VT"],
        lw=2,
    )


def plot_cond(ax_early, ax_late, ax_rel):
    ax_early.set_ylabel(r"$W_d^\mathsf{E}+W_d^\mathsf{I}$", fontsize=figure_config.fontsize_tiny)
    ax_early.set_xlim(xlim_early)
    ax_early.set_ylim(plot_params["ylim_cond"])
    ax_early.set_xticklabels([])

    ax_late.set_xticklabels([])
    ax_late.set_yticks([])
    ax_late.spines["left"].set_visible(False)
    ax_late.set_ylim(plot_params["ylim_cond"])

    ax_rel.set_xlabel('Time (s)', fontsize=figure_config.fontsize_tiny)
    ax_rel.set_ylabel('Rel. weight.', fontsize=figure_config.fontsize_tiny)
    ax_rel.set_ylim(0., 1.)

    wd = res["wEd"] + res["wId"]
    ax_early.plot(
        times[indices_times_start], wd[:, 0][indices_times_start], color=figure_config.colors["V"],
    )
    ax_early.plot(
        times[indices_times_start], wd[:, 1][indices_times_start], color=figure_config.colors["T"],
    )

    ax_late.plot(
        times[indices_times_late], wd[:, 0][indices_times_late], color=figure_config.colors["V"],
    )
    ax_late.plot(
        times[indices_times_late], wd[:, 1][indices_times_late], color=figure_config.colors["T"],
    )

    rel_sigma = (
        1.0
        / params["sigma_0"] ** 2
        / (1.0 / params["sigma_0"] ** 2 + 1.0 / params["sigma_1"] ** 2)
    )
    wd0_rel = wd[:, 0] / (wd[:, 0] + wd[:, 1])
    wd1_rel = wd[:, 1] / (wd[:, 0] + wd[:, 1])
    print(wd0_rel[0], "->", wd0_rel[-1], "<->", rel_sigma)
    ax_rel.plot(
        times, wd0_rel, ls="-", color=figure_config.colors["V"]
    )
    ax_rel.plot(
        times, wd1_rel, ls="-",  color=figure_config.colors["T"]
    )


if __name__ == "__main__":

    sigma = 0.5

    with open(f"params_{sigma}.pkl", "rb") as f:
        params = pickle.load(f)

    plot_params = {
        "t_before": (-2.0, 0.0),
        "t_early": (0.0, 5.0),
        "t_late": None,
        "ylim_potential": (-85.0, -52.0),
        "ylim_cond": (0.0, 1.5),
        "ylim_rate": (0.0, 3.5),
    }

    with open(f"res_{sigma}.pkl", "rb") as f:
        res = pickle.load(f)

    fig = plt.figure(figsize=(5., 2.5))

    x_early = 0.12
    width_early = 0.38
    x_late = 0.56
    width_late = 0.21

    ax_rates_early = fig.add_axes([x_early, 0.15, width_early, 0.10], zorder=1)
    ax_rates_late = fig.add_axes([x_late, 0.15, width_late, 0.10], zorder=1)
    ax_pot_early = fig.add_axes([x_early, 0.55, width_early, 0.38])
    ax_pot_late = fig.add_axes([x_late, 0.55, width_late, 0.38])
    ax_dist = fig.add_axes([0.81, 0.55, 0.16, 0.4], zorder=-1)
    ax_cond_early = fig.add_axes([x_early, 0.35, width_early, 0.1])
    ax_cond_late = fig.add_axes([x_late, 0.35, width_late, 0.1])
    ax_cond_rel = fig.add_axes([0.87, 0.15, 0.125, 0.25])

    step_size = 1
    window_size = 5
    times = np.arange(
        -params["relative_time_silent_teacher"] * params["trials"],
        (1.0 - params["relative_time_silent_teacher"]) * params["trials"],
        params["recording_interval"],
    )
    alpha = 0.65

    # convert trial indices to time
    times *= 100.0 / params["recording_interval"]  # 1 trial == 100 ms
    times *= 1e-3  # ms to s

    plot_params["t_late"] = (times[-1] - 3.5, times[-1])

    indices_times_before = np.where(
        (plot_params["t_before"][0] <= times) & (times < plot_params["t_before"][1])
    )[0][::step_size]
    indices_times_early = np.where(
        (plot_params["t_early"][0] <= times) & (times < plot_params["t_early"][1])
    )[0][::step_size]
    indices_times_late = np.where(
        (plot_params["t_late"][0] <= times) & (times < plot_params["t_late"][1])
    )[0][::step_size]

    indices_times_start = np.hstack([indices_times_before, indices_times_early])

    xlim_early = (plot_params["t_before"][0], plot_params["t_early"][1])
    xlim_late = plot_params["t_late"]

    plot_rates(ax_rates_early, ax_rates_late)
    plot_potential(ax_pot_early, ax_pot_late)
    plot_dist(ax_dist)
    plot_cond(ax_cond_early, ax_cond_late, ax_cond_rel)

    figname = "learning.{ext}"
    print(f"creating {figname}")
    plt.savefig(figname.format(ext="svg"))
    plt.savefig(figname.format(ext="pdf"), dpi=300)
