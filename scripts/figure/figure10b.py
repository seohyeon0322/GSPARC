import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

# === 로그 파일 읽기 ===
log_path = "./result/Figure_10b.out"
with open(log_path, "r", encoding="utf-8") as f:
    log_text = f.read()

# === Execution Breakdown 패턴 ===
# 1) tensor X: .../[NAME](.tns)? 라인에서 NAME만 캡처
# 2) Dynamic Partition, SpTC, Total 차례로 매칭
pattern_parts = (
    r"tensor\s+X:\s+.+/([^/]+?)(?:\.tns)?\s*[\r\n]"  # NAME(.tns)? 캡처, 줄바꿈까지
    r"[\s\S]*?Dynamic\s+Partition\s+time:\s*([\d\.]+)[\s\S]*?"
    r"SpTC\(IdxMatch\s*\+\s*Contraction\s*\+\s*Data\s*Copy\)\s*time:\s*([\d\.]+)[\s\S]*?"
    r"Total\s+time:\s*([\d\.]+)"
)
matches = re.findall(pattern_parts, log_text, re.IGNORECASE)
if not matches:
    raise RuntimeError("로그에서 breakdown 정보를 찾을 수 없습니다.")

# DataFrame 생성 & ms 변환
df = pd.DataFrame(matches, columns=["Tensor","DynamicPartition","SpTC","Total"])
for col in ["DynamicPartition","SpTC","Total"]:
    df[col] = df[col].astype(float) * 1000

# Tensor 이름 정리: 
# 혹시 남아 있는 '.tns' 제거, 'TensorY' 같은 잘못된 텍스트 제거
df["Tensor"] = (
    df["Tensor"]
      .str.replace(r"\.tns$", "", regex=True)
      .str.replace("TensorY", "", regex=False)
      .str.strip()
)

# 중복 이름 고유화 및 레이블 매핑
counts = {}
labels = []
for name in df["Tensor"]:
    counts[name] = counts.get(name, 0) + 1
    # 'delicious-4d' → 'Delicious', 'qt1' → 'Qt1'
    base = {"delicious-4d":"Delicious", "qt1":"Qt1"}.get(name.lower(), name)
    label = base if counts[name] == 1 else f"{base}_{counts[name]}"
    labels.append(label)
df["Label"] = labels

# === 차트 그리기 ===
x = np.arange(len(df))
bar_width = 0.1
fig_width = max(12, len(df) * 0.4)
fig, ax = plt.subplots(figsize=(fig_width, 6))

ax.bar(x - bar_width, df["DynamicPartition"], bar_width, label="Dynamic partition")
ax.bar(x,               df["SpTC"],               bar_width, label="Contraction+Accumulation")
ax.bar(x + bar_width,   df["Total"],               bar_width, label="Total")

ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(df["Label"], rotation=45, ha="right", fontsize=10)
ax.set_ylabel("Execution time (ms)")
ax.set_title("Execution Breakdown: All Tensors")
ax.legend(fontsize=10)

# 값 레이블링
for bar in ax.patches:
    h = bar.get_height()
    text = f"{h/1000:.1f}K" if h >= 1000 else f"{int(h)}"
    ax.text(
        bar.get_x() + bar.get_width()/2,
        h + max(h * 0.03, 5),
        text,
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout(pad=2.0)
os.makedirs("./Figure", exist_ok=True)
fig.savefig("./Figure/figure10b.pdf")
fig.savefig("./Figure/figure10b.png")
plt.show()
