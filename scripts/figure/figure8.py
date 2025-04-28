import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# === 1. 로그 읽기 ===
with open("./result/Figure_8_GSPARC.out", encoding="utf-8") as f:
    text_gsparc = f.read().strip()
with open("./result/Figure_8_Sparta.out", encoding="utf-8") as f:
    text_sparta = f.read().strip()
with open("./result/Figure_8_GspTC.out", encoding="utf-8") as f:
    text_gsptc  = f.read().strip()

# === 2. 파싱 함수 정의 ===
def parse_blocks_simple(impl, text):
    recs = []
    for blk in re.split(r"\n{2,}", text):
        low = blk.lower()
        if "out of memory" in low:
            err, tm = "O.O.M", None
        elif "timeout" in low:
            err, tm = "T.O", None
        else:
            err, tm = None, None

        if impl == "GSPARC":
            m_raw   = re.search(r"tensor X: .*/([^\s/]+)(?:\.tns)?", blk)
            m_modes = re.search(r"# contract modes of X:\s*(\d+)", blk)
            m_time  = re.search(r"Total time:\s*([\d.]+)", blk)
        else:  # Sparta
            m_raw   = re.search(r"Xfname: .*/([^\s/]+)(?:\.tns)?", blk)
            m_modes = re.search(r"Number of contraction modes:\s*(\d+)", blk)
            m_time  = re.search(r"\[Total time\]:\s*([\d.]+)\s*s", blk)

        if not (m_raw and m_modes):
            continue

        raw   = m_raw.group(1)
        modes = int(m_modes.group(1))
        if err is None and m_time:
            tm = float(m_time.group(1)) * 1000.0
        recs.append({
            "Impl":    impl,
            "RawName": raw,
            "Modes":   modes,
            "Time_ms": tm,
            "Error":   err
        })
    return recs

def parse_gsptc(text):
    recs = []
    blocks = re.findall(r'num_cmodes:[\s\S]*?(?=num_cmodes:|\Z)', text)
    for blk in blocks:
        low = blk.lower()
        m_modes = re.search(r'num_cmodes:\s*(\d+)', blk)
        m_raw   = re.search(r'Tensor X\s*=\s*(?:.*/)*([^\s/]+)(?:\.tns)?', blk)
        if not (m_modes and m_raw):
            continue

        raw   = m_raw.group(1)
        modes = int(m_modes.group(1))
        if 'out of memory' in low:
            recs.append({"Impl":"GspTC","RawName":raw,"Modes":modes,"Time_ms":None,"Error":"O.O.M"})
            continue
        if 'timeout' in low:
            recs.append({"Impl":"GspTC","RawName":raw,"Modes":modes,"Time_ms":None,"Error":"T.O"})
            continue

        m_time = re.search(r'total time:\s*([\d.]+)', blk, re.IGNORECASE)
        if m_time:
            tm, err = float(m_time.group(1)) * 1000.0, None
        else:
            tm, err = None, "N/A"

        recs.append({
            "Impl":    "GspTC",
            "RawName": raw,
            "Modes":   modes,
            "Time_ms": tm,
            "Error":   err
        })
    return recs

# === 3. 레코드 수집 & DataFrame ===
recs_gsparc = parse_blocks_simple("GSPARC", text_gsparc)
recs_sparta = parse_blocks_simple("Sparta", text_sparta)
recs_gsptc  = parse_gsptc(text_gsptc)

df = pd.DataFrame(recs_gsparc + recs_sparta + recs_gsptc) \
       .drop_duplicates(subset=["Impl","RawName","Modes"])

# === 4. final_seq: GSPARC 순서 → Sparta 순서(남은) → GspTC 순서(남은) ===
seq_gsparc = [(r["RawName"], r["Modes"]) for r in recs_gsparc]

seq_sparta = []
for r in recs_sparta:
    pair = (r["RawName"], r["Modes"])
    if pair not in seq_gsparc and pair not in seq_sparta:
        seq_sparta.append(pair)

seq_gsptc = []
for r in recs_gsptc:
    pair = (r["RawName"], r["Modes"])
    if pair not in seq_gsparc and pair not in seq_sparta and pair not in seq_gsptc:
        seq_gsptc.append(pair)

final_seq = seq_gsparc + seq_sparta + seq_gsptc

# === 5. 레이블 생성 ===
def nice_label(raw, modes):
    rl = raw.lower()
    if "chicago-crime" in rl: base = "Chicago"
    elif "flickr" in rl:       base = "Flickr"
    elif "vast" in rl:         base = "Vast"
    elif "delicious" in rl:    base = "Delicious"
    elif "amazon" in rl:       base = "Amazon"
    elif "reddit" in rl:       base = "Reddit"
    else:
        base = " ".join(w.capitalize() for w in raw.replace(".tns","").split("-"))
    return f"{base} ({modes}-mode)"

tick_labels = [nice_label(r, m) for r, m in final_seq]
N = len(tick_labels)
x = np.arange(N)

# === 6. 차트 그리기 ===
impls  = ["Sparta","GspTC","GSPARC"]
colors = {"Sparta":"#d7191c","GspTC":"#fdae61","GSPARC":"#6da04b"}
w      = 0.25

plt.figure(figsize=(max(14, N*0.6), 6))
for i, impl in enumerate(impls):
    sub = df[df["Impl"]==impl]
    vals = [
        (sub[(sub["RawName"]==raw)&(sub["Modes"]==mode)]["Time_ms"].iloc[0]
         if not sub[(sub["RawName"]==raw)&(sub["Modes"]==mode)].empty else np.nan)
        for raw, mode in final_seq
    ]
    plt.bar(x + i*w, vals, w, color=colors[impl], label=impl)

# 에러 텍스트 표시
for i, impl in enumerate(impls):
    sub = df[df["Impl"]==impl]
    for idx, (raw, mode) in enumerate(final_seq):
        row = sub[(sub["RawName"]==raw)&(sub["Modes"]==mode)]
        xpos = x[idx] + i*w
        if not row.empty:
            tm = row["Time_ms"].iloc[0]
            if pd.notna(tm):
                lbl = f"{tm/1000:.1f}K" if tm>=1000 else f"{tm:.0f}"
                plt.text(xpos, tm, lbl, ha="center", va="bottom", fontsize=8)
            else:
                plt.text(xpos, 1, row["Error"].iloc[0],
                         ha="center", va="bottom", fontsize=7, rotation=90)

plt.yscale("log")
plt.ylim(1)
plt.xticks(x + w, tick_labels, rotation=45, ha="right")
plt.xlabel("Tensor")
plt.ylabel("Execution Time (ms, log scale)")
plt.title("Total Execution Time per Tensor (ALL, ordered by appearance)")
plt.legend()
plt.tight_layout()
plt.savefig("./Figure/figure8.pdf")
plt.savefig("./Figure/figure8.png")
plt.show()
