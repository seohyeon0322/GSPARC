# !/bin/bash
mkdir -p result
#Table 5 GSPARC
./scripts/Tab4_SpTC_FROSTT.sh >/dev/null 2> ./result/Tab4_SpTC_FROSTT.out 

#Table 5 synthetic
./scripts/Tab5_SpTC_synthetic.sh >/dev/null 2> ./result/Tab5_SpTC_synthetic.out

#Figure 9a memory
./scripts/Fig9a_memory.sh >/dev/null 2> ./result/Fig9a_memory.out

#Figure 8b multiGPU
./scripts/Fig8b_multiGPU.sh >/dev/null 2> ./result/Fig8b_multiGPU.out

#Table 7 multi sptc
./scripts/Tab7_multi_sptc.sh >/dev/null 2> ./result/Tab7_multi_sptc.out
