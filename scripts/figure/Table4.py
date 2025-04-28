import re
import pandas as pd

def read_log_file(filename):
    with open(filename, 'r') as f:
        return f.read()

def extract_times(log_text):
    # 각 텐서의 실행 결과를 찾기 위한 패턴
    pattern = r"tensor X: .*?/([^/]+)\.tns[\s\S]*?First SpTC time: ([\d\.]+)[\s\S]*?Bit reordering time: ([\d\.]+)[\s\S]*?Total time: ([\d\.]+)"
    matches = re.findall(pattern, log_text)
    
    results = []
    for name, sptc_time, bit_time, total_time in matches:
        # 텐서 이름 정리
        if 'nips' in name.lower():
            tensor_name = 'NIPS'
        elif 'uber' in name.lower():
            tensor_name = 'Uber'
        elif 'chicago' in name.lower():
            tensor_name = 'Chicago'
        else:
            continue
            
        # 초 단위를 밀리초 단위로 변환 (x1000)
        bit_time = float(bit_time) * 1000
        sptc_time = float(sptc_time) * 1000
        total_time = float(total_time) * 1000
        
        # bit reordering ratio 계산 (%)
        ratio = (bit_time / total_time) * 100
        
        results.append({
            'Tensors': tensor_name,
            'bit reordering': bit_time,
            'SpTC': sptc_time,
            'multi-SpTCs': total_time,
            'bit reordering ratio': f"{ratio:.2f}%"
        })
    
    return pd.DataFrame(results)

# 로그 파일 읽기
log_text = read_log_file("./result/Table_4.out")  # 파일명 수정

# 데이터 추출 및 데이터프레임 생성
df = extract_times(log_text)

# 표 형식으로 출력
print("\nTable 4: Execution times (in ms) and bit reordering cost in multi-SpTCs.")
print("=" * 80)

# pandas 출력 옵션 설정
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# 데이터프레임 출력
print(df.to_string(index=False))

# CSV 파일로 저장
df.to_csv('./Figure/table4.csv', index=False, float_format='%.3f')
print("\nTable has been saved as 'table4.csv'")