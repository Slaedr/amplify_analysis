import json
import matplotlib.pyplot as plt
import numpy as np

data_dir = "bench-amp-gb10-data"

def load_gmres(filename):
    with open(f"{data_dir}/{filename}") as f:
        return json.load(f)["gmres"]


def get_time(entries, format_prefix):
    for e in entries:
        if e["format"].startswith(format_prefix):
            return e["solve_ms"]
    raise ValueError(f"No entry with format prefix '{format_prefix}'")


fp16_csr = load_gmres("with_fp16_gmres_csr_cuda_results.json")
bf16_csr = load_gmres("with_bf16_gmres_csr_cuda_results.json")
fp16_ell = load_gmres("with_fp16_gmres_ell_cuda_results.json")
bf16_ell = load_gmres("with_bf16_gmres_ell_cuda_results.json")

# Extract times: base format (double), AMP FP16, AMP BF16
csr_double = get_time(fp16_csr, "CSR<double>")
csr_amp_fp16 = get_time(fp16_csr, "AMP")
csr_amp_bf16 = get_time(bf16_csr, "AMP")

ell_double = get_time(fp16_ell, "ELL<double>")
ell_amp_fp16 = get_time(fp16_ell, "AMP")
ell_amp_bf16 = get_time(bf16_ell, "AMP")

# Plot
fig, ax = plt.subplots(figsize=(5.5, 4.0))

groups = ["CSR", "ELL"]
x = np.arange(len(groups))
width = 0.22

bars_base = [csr_double, ell_double]
bars_fp16 = [csr_amp_fp16, ell_amp_fp16]
bars_bf16 = [csr_amp_bf16, ell_amp_bf16]

labels = ["FP64", "AMP (FP16)", "AMP (BF16)"]
colors = ["#2c7bb6", "#d7191c", "#fdae61"]

rects0 = ax.bar(x - width, bars_base, width, label=labels[0], color=colors[0],
                edgecolor="black", linewidth=0.6)
rects1 = ax.bar(x, bars_fp16, width, label=labels[1], color=colors[1],
                edgecolor="black", linewidth=0.6)
rects2 = ax.bar(x + width, bars_bf16, width, label=labels[2], color=colors[2],
                edgecolor="black", linewidth=0.6)

ax.set_ylabel("Time (ms)", fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=16)
ax.tick_params(axis="y", labelsize=14)
ax.legend(fontsize=13, loc="upper right")

ax.set_ylim(0, max(bars_base + bars_fp16 + bars_bf16) * 1.25)
ax.yaxis.grid(True, linestyle="--", alpha=0.7, linewidth=0.5)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig("gmres_time.pdf", bbox_inches="tight")
print("Saved gmres_time.pdf")

# Speedup plot: base_time / AMP_time (only AMP bars)
fig2, ax2 = plt.subplots(figsize=(5.5, 4.0))

width2 = 0.25
speedup_fp16 = [csr_double / csr_amp_fp16, ell_double / ell_amp_fp16]
speedup_bf16 = [csr_double / csr_amp_bf16, ell_double / ell_amp_bf16]

ax2.bar(x - width2 / 2, speedup_fp16, width2, label="AMP (FP16)", color=colors[1],
        edgecolor="black", linewidth=0.6)
ax2.bar(x + width2 / 2, speedup_bf16, width2, label="AMP (BF16)", color=colors[2],
        edgecolor="black", linewidth=0.6)

ax2.set_ylabel("Speedup over FP64", fontsize=16)
ax2.set_xticks(x)
ax2.set_xticklabels(groups, fontsize=16)
ax2.tick_params(axis="y", labelsize=14)
ax2.legend(fontsize=13, loc="upper right")
ax2.axhline(y=1.0, color="black", linestyle="-", linewidth=0.8)
all_speedups = [*speedup_fp16, *speedup_bf16]
ax2.set_ylim(min(all_speedups) - 0.1, max(all_speedups) + 0.15)
ax2.yaxis.grid(True, linestyle="--", alpha=0.7, linewidth=0.5)
ax2.set_axisbelow(True)

fig2.tight_layout()
fig2.savefig("gmres_speedup.pdf", bbox_inches="tight")
print("Saved gmres_speedup.pdf")
