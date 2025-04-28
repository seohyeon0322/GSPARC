import re
import matplotlib.pyplot as plt
import numpy as np

def read_log_file(filename):
    with open(filename, 'r') as f:
        return f.read()

def extract_sptc_times(log_text):
    pattern = (
        r"tensor X: .*/([^/]+)"     
        r"(?:\.tns)?"                
        r"[\s\S]*?num_gpus: (\d+)"   
        r"[\s\S]*?SpTC\(IdxMatch \+ Contraction \+ Data Copy\) time: ([\d\.]+)"
    )
    matches = re.findall(pattern, log_text)
    
    results = {}
    for name, gpus, time in matches:
        nl = name.lower()
        # 2) Qt1 분기 추가
        if 'amazon' in nl:
            simple_name = 'Amazon'
        elif 'patent' in nl:
            simple_name = 'Patents'
        elif 'reddit' in nl:
            simple_name = 'Reddit'
        elif 'qt1' in nl:
            simple_name = 'Qt1'
        elif 'qt2' in nl:
            simple_name = 'Qt2'
        else:
            simple_name = name  # 나머지는 원본 그대로
        
        results.setdefault(simple_name, {})[int(gpus)] = float(time)
    
    return results


# 로그 파일 읽기
log_text = read_log_file("./result/Figure_11a.out")  # 파일명 수정

# SpTC 시간 추출
results = extract_sptc_times(log_text)

# Speedup 계산
speedups = {}
for dataset in results:
    base_time = results[dataset][1]  # 1-GPU time
    speedups[dataset] = [
        base_time / results[dataset].get(1, base_time),  # 1-GPU (항상 1.0)
        base_time / results[dataset].get(2, base_time),  # 2-GPU
        base_time / results[dataset].get(4, base_time),  # 4-GPU
        base_time / results[dataset].get(8, base_time)   # 8-GPU
    ]

# 그래프 설정
fig, ax = plt.subplots(figsize=(10, 6))

# 데이터셋과 GPU 설정
datasets = ['Amazon', 'Patents', 'Reddit', 'Qt2']  # 순서 고정
x = np.arange(len(datasets))
width = 0.15  # 바의 너비

# GPU 설정별 색상
colors = ['#d7191c', '#fdae61', '#6da04b', '#2c7bb6']
gpu_configs = ['1-GPU', '2-GPU', '4-GPU', '8-GPU']

# 각 GPU 설정별로 바 그리기
for i, (gpu, color) in enumerate(zip(gpu_configs, colors)):
    values = [speedups.get(dataset, [1.0, 1.0, 1.0, 1.0])[i] for dataset in datasets]
    bars = ax.bar(x + i*width, values, width, label=gpu, color=color)
    
    # 바 위에 값 표시
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=8)

# 그래프 꾸미기
ax.set_ylabel('Speedup')
ax.set_title('Result of multi-GPU execution')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(datasets)
ax.legend()

# x축 레이블 설정
plt.xlabel('Tensors')

# y축 범위 설정 (0부터 8까지)
plt.ylim(0, 8)

# 레이아웃 조정
plt.tight_layout()

# 그래프 저장
plt.savefig('./Figure/figure11a.pdf')
plt.savefig('./Figure/figure11a.png')
plt.show()
