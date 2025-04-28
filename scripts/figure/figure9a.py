import re
import os
import numpy as np
import matplotlib.pyplot as plt

def parse_alto_time(log_content, dataset):
    """Parse ALTO creation time from ALTO log."""
    pattern = rf"(?:text_file|quantum_file):.*?{re.escape(dataset)}(?:\.tns)?[\s\S]*?ALTO creation time:\s*([\d.]+)"
    m = re.search(pattern, log_content, re.IGNORECASE)
    return float(m.group(1)) * 1000 if m else None

def parse_blco_time(log_content, dataset):
    """Parse BLCO tensor generation time from BLCO log."""
    pattern = rf"(?:text_file|quantum_file):.*?{re.escape(dataset)}(?:\.tns)?[\s\S]*?BLCO tensor generation time:\s*([\d.]+)"
    m = re.search(pattern, log_content, re.IGNORECASE)
    return float(m.group(1)) * 1000 if m else None

def parse_hash_time(log_content, dataset):
    """Parse hash-table creation time from Sparta log."""
    pattern = rf"Xfname:.*?{re.escape(dataset)}(?:\.tns)?[\s\S]*?\[Create Hash table\]:\s*([\d.]+)\s*s"
    m = re.search(pattern, log_content, re.IGNORECASE)
    return float(m.group(1)) * 1000 if m else None

def parse_slitom_time(log_content, dataset):
    """Parse SLITOM time from GSPARC (SpTC) log and divide by 2."""
    pattern = rf"{re.escape(dataset)}(?:\.tns)?[\s\S]*?SLITOM time:\s*([\d.]+)"
    m = re.search(pattern, log_content, re.IGNORECASE)
    return float(m.group(1)) * 1000 / 2 if m else None

def collect_data():
    datasets = ['nell-2', 'nips', 'patents', 'qt1']  # Qt2 removed
    results = {'Hash': [], 'ALTO': [], 'BLCO': [], 'SLITOM': []}
    base = "./result"

    def _read(fn):
        path = os.path.join(base, fn)
        return open(path, 'r', encoding='utf-8').read() if os.path.exists(path) else ""

    alto_log   = _read('Figure_9a_ALTO.out')
    blco_log   = _read('Figure_9a_BLCO.out')
    sparta_log = _read('Figure_8_Sparta.out')
    gsparc_log = _read('Figure_8_GSPARC.out')

    for d in datasets:
        results['Hash'].append( parse_hash_time(sparta_log, d) )
        results['ALTO'].append( parse_alto_time(alto_log,   d) )
        results['BLCO'].append( parse_blco_time(blco_log,   d) )
        results['SLITOM'].append( parse_slitom_time(gsparc_log, d) )

    return datasets, results

def create_figure9a(save_path, datasets, results):
    labels = ['Nell-2', 'NIPS', 'Patents', 'Qt1']
    impls  = ['Hash', 'ALTO', 'BLCO', 'SLITOM']  # reordered
    colors = ['#d7191c', '#fdae61', '#2ca02c', '#1f77b4']  # match impls
    bar_w  = 0.18
    x = np.arange(len(datasets))

    plt.figure(figsize=(14, 6))
    for i, impl in enumerate(impls):
        times = results[impl]
        vals  = [t if t is not None else 0 for t in times]
        pos = x + (i - (len(impls)-1)/2) * bar_w
        plt.bar(pos, vals, bar_w, color=colors[i], label=impl)
        for p, t in zip(pos, times):
            if t is None:
                plt.text(p, 1, '-', ha='center', va='bottom', fontsize=10)
            else:
                label = f"{t/1000:.1f}K" if t >= 1000 else f"{int(t)}"
                offset = t * 0.05 if t > 0 else 5
                plt.text(p, t + offset, label, ha='center', va='bottom', fontsize=10)

    plt.yscale('log')
    plt.ylim(1)
    plt.xticks(x, labels)
    plt.xlabel('Tensors')
    plt.ylabel('Construction time (ms, log scale)')
    plt.legend(loc='upper right')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path.replace('.png', '.pdf'))
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    save_path = "./Figure/figure9a.png"
    datasets, results = collect_data()
    create_figure9a(save_path, datasets, results)
