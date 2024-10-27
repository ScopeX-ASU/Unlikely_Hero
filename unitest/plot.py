import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pyutils.general import ensure_dir
from pyutils.plot import batch_plot, set_axes_size_ratio, set_ms

color_dict = {
    "black": "#000000",
    "red": "#de425b",  # red
    "blue": "#1F77B4",  # blue
    "orange": "#f58055",  # orange
    "yellow": "#f6df7f",  # yellow
    "green": "#2a9a2a",  # green
    "grey": "#979797",  # grey
    "purple": "#AF69C5",  # purple,
    "mitred": "#A31F34",  # mit red
    "pink": "#CDA2BE",
}


def plot_sparsity(sp_mode, sa_mode):
    set_ms()
    csv_file = f"./Experiment/log/cifar10/SparseBP_MRR_VGG8/sparsity_mode-{sp_mode}_salience_mode-{sa_mode}.csv"
    data = np.loadtxt(csv_file, delimiter=",")
    # [step, cycle, mae, acc]
    steps = data[:, ::4].T
    cycles = data[:, 1::4].T
    maes = data[:, 2::4].T
    accs = data[:, 3::4].T
    sparsitys = np.arange(0.1, 1.01, 0.1)
    print(accs.shape, sparsitys.shape)

    fig, ax = plt.subplots(1, 1)

    name = f"{sp_mode}_{sa_mode}"
    ensure_dir("./figures/sparsity")
    # cmap = mpl.colormaps['viridis'].resampled(8)
    cmap = mpl.colormaps["coolwarm"].resampled(8)
    for i, (cycle, acc, sparsity) in enumerate(zip(cycles, accs, sparsitys)):
        fig, ax, _ = batch_plot(
            "line",
            raw_data={"x": cycle / 1000, "y": acc},
            name=name,
            xlabel="Cycles (K)",
            ylabel="Test Acc (%)",
            fig=fig,
            ax=ax,
            xrange=[0, 250.1, 50],
            yrange=[50, 100.1, 10],
            xformat="%d",
            yformat="%d",
            figscale=[0.65, 0.65 * 9.1 / 8],
            fontsize=9,
            linewidth=1,
            gridwidth=0.5,
            trace_color=cmap(sparsity),
            alpha=1,
            trace_label=f"{sparsity}",
            linestyle="-",
            ieee=True,
        )

        set_axes_size_ratio(0.4, 0.5, fig, ax)

        plt.savefig(f"./figures/sparsity/{name}.png", dpi=300)
        plt.savefig(f"./figures/sparsity/{name}.pdf", dpi=300)


def plot_lowrank_scanning2d_resnet():
    fig, ax = None, None

    Bc = np.arange(1, 9, 1)  ## X-axis Col
    Bi = np.arange(1, 9, 1)  ## Y-axis Row
    acc = np.array(
        [
            [82.62, 87.40, 88.80, 89.97, 90.27, 90.36, 90.73, 91.33],
            [83.75, 88.07, 89.09, 89.58, 91.09, 90.92, 91.00, 91.52],
            [84.18, 87.86, 89.08, 89.91, 90.48, 91.02, 91.47, 91.57],
            [83.95, 88.12, 89.37, 90.78, 91.10, 90.99, 91.10, 91.42],
            [83.93, 87.94, 89.75, 90.01, 90.62, 90.73, 91.53, 91.30],
            [84.07, 88.72, 90.20, 90.49, 91.27, 91.28, 91.27, 91.26],
            [84.69, 88.54, 90.16, 90.32, 91.13, 91.08, 91.52, 91.54],
            [85.99, 88.85, 89.89, 90.90, 91.26, 91.18, 91.03, 91.47],
        ]
    )

    name = "Scan2d"
    fig, ax, _ = batch_plot(
        "mesh2d",
        raw_data={"x": Bc, "y": Bi, "z": acc},
        name=name,
        xlabel="Bc",
        ylabel="Bi",
        fig=fig,
        ax=ax,
        xrange=[1, 8.01, 1],
        yrange=[1, 8.01, 1],
        xformat="%.0f",
        yformat="%.0f",
        figscale=[0.65, 0.65 * 9.1 / 8],
        fontsize=10,
        linewidth=1,
        gridwidth=0.5,
        ieee=True,
    )
    X, Y = np.meshgrid(Bc, Bi)
    ct = ax.contour(X, Y, acc, 5, colors="k", linewidths=0.5)
    ax.clabel(ct, fontsize=10, colors="k")

    # ct = ax.contour(X, Y, acc,5,colors='b', linewidths=0.5)
    # ax.clabel(ct,fontsize=10,colors='b')

    fig.savefig(f"{name}.png")
    fig.savefig(f"{name}.pdf")


for sp_mode in ["uniform", "topk", "IS"]:
    for sa_mode in ["first_grad", "second_grad"]:
        plot_sparsity(sp_mode=sp_mode, sa_mode=sa_mode)

# plot_lowrank_scanning2d_resnet()
