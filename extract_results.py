from tensorflow.python.summary.summary_iterator import summary_iterator
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import os

def print_keys(tfevents_path):
    print("Log keys:\n", "\n".join(sorted(set([
        event.summary.value[0].tag
        for event in summary_iterator(tfevents_path)
        if len(event.summary.value) > 0
    ]))))

def extract_scalar_prop(tfevents_path, tags):
    result = []
    for event in summary_iterator(tfevents_path):
        if (len(event.summary.value) > 0 and (
            event.summary.value[0].tag in tags
        )):
            result.append(event.summary.value[0].simple_value)
    return np.array(result)

def calculate_time_per_point_and_per_100K(tfevents_paths):
    print("Checking timing...")
    all_times = []
    for lod in range(3):
        for tfevents_path in tfevents_paths:
            all_times.extend(extract_scalar_prop(tfevents_path, [
                f"Surface/AverageTime/{lod}", f"Volume/AverageTime/{lod}"
            ]).tolist())
        print(f"LOD {lod+1}")
        print(f"Mean time per 100K points: {(100000 * np.mean(all_times)) * 1000:.2f} ms")
        print(f"Mean time per point:       {np.mean(all_times) * (10**9):.2f} ns")
        print()

def plot_lod_losses(tfevents_paths, exp_name, save_plot=True):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    for (lod, color) in zip(range(3), ['blue', 'red', 'green']):
        all_lod_losses = []
        for tfevents_path in tfevents_paths:
            all_lod_losses.append(np.array(extract_scalar_prop(
                tfevents_path, [f"Loss/l2_loss/{lod}"]
            )))

        max_arr_len = max(arr.shape[0] for arr in all_lod_losses)
        # sum_arr = np.zeros(max_arr_len)
        count_arr = np.zeros(max_arr_len)
        for arr in all_lod_losses:
        #     sum_arr[:arr.shape[0]] += arr
            count_arr[:arr.shape[0]] += 1
        epochs = np.arange(max_arr_len)
        # averages = sum_arr / count_arr
        averages = np.mean([
            np.pad(arr, (0, max_arr_len - len(arr)), 'constant', constant_values=arr[-1]) 
            for arr in all_lod_losses
        ], axis=0)
        axs[0].plot(epochs, averages, color=color, linewidth=2, label=f'avg, lod {lod+1}')
        if lod == 0:
            axs[1].plot(epochs, count_arr, color=color, linewidth=1, label=f'count')

        # minimums = np.min([
        #     np.pad(arr, (0, max_arr_len - len(arr)), 'constant', constant_values=99999) 
        #     for arr in all_lod_losses
        # ], axis=0)
        # axs[0].plot(epochs, minimums, color=color, label=f'min, lod {lod+1}')
        # axs[0].fill_between(epochs, averages, minimums, color=color, alpha=0.1)

        # maximums = np.max([
        #     np.pad(arr, (0, max_arr_len - len(arr)), 'constant', constant_values=0) 
        #     for arr in all_lod_losses
        # ], axis=0)
        # axs[0].plot(epochs, maximums, color=color, label=f'max, lod {lod+1}')
        # axs[0].fill_between(epochs, maximums, averages, color=color, alpha=0.1)
    for ax in axs:
        ax.axvline(x=5-1, color='red', linestyle='-', linewidth=1)
        ax.axvline(x=25-1, color='red', linestyle='-', linewidth=1)
        ax.grid()
        ax.set_xlabel('Epochs')
        ax.legend()
    axs[0].set_ylabel('Loss')
    axs[0].set_yscale('log')
    axs[1].set_ylabel('Count of trained')
    plt.tight_layout()
    if save_plot:
        os.makedirs(f"PLOTS/{exp_name}/", exist_ok=True)
        plt.savefig(f"PLOTS/{exp_name}/lod_losses.png")
    plt.show()

def find_f1_fails(tfevents_paths):
    for tfevents_path in tfevents_paths:
        for (test_name, test_f1_thr) in [
            ("Volume", 0.95), ("Surface", 0.9),
        ]:
            any_lod_works = False
            for lod in range(3):
                all_lod_f1s = extract_scalar_prop(tfevents_path, [f"{test_name}/F1/{lod+1}"])
                argmax_f1 = np.argmax(all_lod_f1s)
                any_lod_works = any_lod_works or (all_lod_f1s[argmax_f1] > test_f1_thr)
                if any_lod_works:
                    break
            if not any_lod_works:
                print(f"TEST: {test_name} OBJECT: {Path(tfevents_path).parent.name} MAX F1: {all_lod_f1s[argmax_f1]} AFTER EPOCH: {argmax_f1+1}")

def plot_loss_and_f1(tfevents_path, lod, exp_name, save_plot=True):
    # plot lod 3 loss on left side
    # plot f1 Volume and Surface F1 on the right side
    volume_f1 = extract_scalar_prop(tfevents_path, [f"Volume/F1/{lod}"])
    surface_f1 = extract_scalar_prop(tfevents_path, [f"Surface/F1/{lod}"])
    l2_loss = extract_scalar_prop(tfevents_path, [f"Loss/l2_loss/{lod-1}"])
    batch_n = np.arange(len(l2_loss))

    _, ax1 = plt.subplots(figsize=(12, 8))
    ax1.set_xlabel('Epoch')
    ax1.axvline(x=25-1, color='green', linestyle='-', linewidth=1, label='First scheduler step')
    ax1.axvline(x=50-1, color='lightgreen', linestyle='-', linewidth=1, label='Second scheduler step')
    ax1.set_ylabel('L2 Loss', color='mediumblue')
    ax1.set_yscale('log')
    ax1.plot(batch_n, l2_loss, color='blue', label="L2 Loss")
    ax1.legend()
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()  
    ax2.set_ylabel('F1 Score (%)', color='red')  
    ax2.plot(batch_n, volume_f1, color='red', label="Volume F1")
    ax2.plot(batch_n, surface_f1, color='lightcoral', label="Surface F1")
    ax2.legend()
    ax2.tick_params(axis='y', labelcolor='red')

    plt.grid()
    plt.tight_layout()
    if save_plot:
        os.makedirs(f"PLOTS/{exp_name}/", exist_ok=True)
        plt.savefig(f"PLOTS/{exp_name}/obj_{Path(tfevents_path).parent.name}_lod{lod}_loss_and_f1.png")
    plt.show()


def create_latex_table(tfevents_paths):

    object_numbers = list(map(lambda p: int(Path(p).parent.name.split("_")[1]), tfevents_paths))
    compression = list(map(
        lambda p: 800517 / os.path.getsize(f"./test_task_meshes/{int(Path(p).parent.name.split('_')[1])}.obj"),
        tfevents_paths
    ))
    batch_criteria_met = list(map(
        lambda p: len(extract_scalar_prop(p, [f"Loss/l2_loss/2"])),
        tfevents_paths
    ))
    volume_f1 = list(map(
        lambda p: max([
            max(extract_scalar_prop(p, [f"Volume/F1/{lod+1}"]))
            for lod in range(3)
        ])*100,
        tfevents_paths
    ))
    surface_f1 = list(map(
        lambda p: max([
            max(extract_scalar_prop(p, [f"Surface/F1/{lod+1}"]))
            for lod in range(3)
        ])*100,
        tfevents_paths
    ))
    min_loss = list(map(
        lambda p: min([
            min(extract_scalar_prop(p, [f"Loss/l2_loss/{lod}"]))
            for lod in range(3)
        ]),
        tfevents_paths
    ))

    print("\\begin{tabular}{|c|c|c|c|c|c|}")
    print("\\hline")
    print("\# Object & Compression & \# Batch, Criteria Met & Volume F1, \% & Surface F1, \% & Min L2 Loss \\\\")
    print("\\hline")

    for i in range(len(object_numbers)):
        print(f"{object_numbers[i]} & {compression[i]:.2f} & {batch_criteria_met[i]} & \\textcolor{{blue}}{{{volume_f1[i]:.2f}}}\% & \\textcolor{{blue}}{{{surface_f1[i]:.2f}}}\% & {min_loss[i]:.2e} \\\\")
        print("\\hline")

    print("\\end{tabular}")

def main(exp_name):
    results_root = Path(f"/Users/michael/uni/UCU 4y-1/cv/module_4_hw/RESULTS/{exp_name}/_results")

    log_runs_root = results_root / 'logs' / 'runs'
    tfevents_paths = []
    for root, _dirs, files in os.walk(log_runs_root):
        for file in files:
            if 'tfevents' in file:
                tfevents_paths.append(str(Path(root) / file))
    tfevents_paths = sorted(tfevents_paths, key=lambda p: int(Path(p).parent.name.split("_")[1]))
    print(f"Found {len(tfevents_paths)} paths")
    # print_keys(tfevents_paths[0])
    # calculate_time_per_point_and_per_100K(tfevents_paths)
    # plot_lod_losses(tfevents_paths, exp_name, save_plot=True)
    # find_f1_fails(tfevents_paths)
    # plot_loss_and_f1(tfevents_paths[0], 3, exp_name, save_plot=True)
    # create_latex_table(tfevents_paths)
    print(np.mean(list(map(
        lambda p: os.path.getsize(f"./test_task_meshes/{int(Path(p).parent.name.split('_')[1])}.obj"),
        tfevents_paths
    ))) / 1024)

if __name__ == "__main__":
    # main("exp_1")
    main("exp_2")
    # main("timing_exp")
    # main("timing_65536")


