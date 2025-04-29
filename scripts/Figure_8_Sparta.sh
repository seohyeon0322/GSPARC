#!/bin/bash
export EXPERIMENT_MODES=3

# 2D tensors
timeout 12000s ./baselines/Sparta/build/benchmark/ttt -X /var/GSPARC/dataset/real-tensor/3d/nell-2.tns -Y /var/GSPARC/dataset/real-tensor/3d/nell-2.tns -m 2 -x 0 2 -y 0 2 -t 32 || echo -e "Timeout\n\n" >&2;
timeout 12000s ./baselines/Sparta/build/benchmark/ttt -X /var/GSPARC/dataset/real-tensor/4d/nips.tns -Y /var/GSPARC/dataset/real-tensor/4d/nips.tns -m 2 -x 1 2 -y 1 2 -t 32 || echo -e "Timeout\n\n" >&2;
timeout 12000s ./baselines/Sparta/build/benchmark/ttt -X /var/GSPARC/dataset/real-tensor/4d/uber.tns -Y /var/GSPARC/dataset/real-tensor/4d/uber.tns -m 2 -x 2 3 -y 2 3 -t 32 || echo -e "Timeout\n\n" >&2;
timeout 12000s ./baselines/Sparta/build/benchmark/ttt -X /var/GSPARC/dataset/real-tensor/4d/chicago-crime-comm.tns -Y /var/GSPARC/dataset/real-tensor/4d/chicago-crime-comm.tns -m 2 -x 0 2 -y 0 2 -t 32 || echo -e "Timeout\n\n" >&2;
timeout 12000s ./baselines/Sparta/build/benchmark/ttt -X /var/GSPARC/dataset/real-tensor/4d/flickr-4d.tns -Y /var/GSPARC/dataset/real-tensor/4d/flickr-4d.tns -m 2 -x 1 2 -y 1 2 -t 32 || echo -e "Timeout\n\n" >&2;
timeout 12000s ./baselines/Sparta/build/benchmark/ttt -X /var/GSPARC/dataset/real-tensor/4d/delicious-4d.tns -Y /var/GSPARC/dataset/real-tensor/4d/delicious-4d.tns -m 2 -x 1 2 -y 1 2 -t 32 || echo -e "Timeout\n\n" >&2;
timeout 12000s ./baselines/Sparta/build/benchmark/ttt -X /var/GSPARC/dataset/real-tensor/5d/vast-2015-mc1-5d.tns -Y /var/GSPARC/dataset/real-tensor/5d/vast-2015-mc1-5d.tns -m 2 -x 0 1 -y 0 1 -t 32 || echo -e "Timeout\n\n" >&2;

# 3D tensors
timeout 12000s ./baselines/Sparta/build/benchmark/ttt -X /var/GSPARC/dataset/real-tensor/4d/nips.tns -Y /var/GSPARC/dataset/real-tensor/4d/nips.tns -m 3 -x 0 1 2 -y 0 1 2 -t 32 || echo -e "Timeout\n\n" >&2;
timeout 12000s ./baselines/Sparta/build/benchmark/ttt -X /var/GSPARC/dataset/real-tensor/4d/uber.tns -Y /var/GSPARC/dataset/real-tensor/4d/uber.tns -m 3 -x 0 2 3 -y 0 2 3 -t 32 || echo -e "Timeout\n\n" >&2;
timeout 12000s ./baselines/Sparta/build/benchmark/ttt -X /var/GSPARC/dataset/real-tensor/4d/chicago-crime-comm.tns -Y /var/GSPARC/dataset/real-tensor/4d/chicago-crime-comm.tns -m 3 -x 0 2 3 -y 0 2 3 -t 32 || echo -e "Timeout\n\n" >&2;
timeout 12000s ./baselines/Sparta/build/benchmark/ttt -X /var/GSPARC/dataset/real-tensor/4d/flickr-4d.tns -Y /var/GSPARC/dataset/real-tensor/4d/flickr-4d.tns -m 3 -x 0 1 2 -y 0 1 2 -t 32 || echo -e "Timeout\n\n" >&2;
timeout 12000s ./baselines/Sparta/build/benchmark/ttt -X /var/GSPARC/dataset/real-tensor/4d/delicious-4d.tns -Y /var/GSPARC/dataset/real-tensor/4d/delicious-4d.tns -m 3 -x 0 1 2 -y 0 1 2 -t 32 || echo -e "Timeout\n\n" >&2;
timeout 12000s ./baselines/Sparta/build/benchmark/ttt -X /var/GSPARC/dataset/real-tensor/5d/vast-2015-mc1-5d.tns -Y /var/GSPARC/dataset/real-tensor/5d/vast-2015-mc1-5d.tns -m 3 -x 0 1 3 -y 0 1 3 -t 32 || echo -e "Timeout\n\n" >&2;


# 2-mode contraction (real-world (large))
timeout 12000s ./baselines/Sparta/build/benchmark/ttt -X /var/GSPARC/dataset/real-tensor/3d/amazon-reviews.tns -Y /var/GSPARC/dataset/real-tensor/3d/amazon-reviews.tns -m 2 -x 0 2 -y 0 2 -t 32|| echo -e "Timeout\n\n" >&2;
timeout 12000s ./baselines/Sparta/build/benchmark/ttt -X /var/GSPARC/dataset/real-tensor/3d/patents.tns -Y /var/GSPARC/dataset/real-tensor/3d/patents.tns -m 2 -x 1 2 -y 1 2 -t 32 || echo -e "Timeout\n\n" >&2;
timeout 12000s ./baselines/Sparta/build/benchmark/ttt -X /var/GSPARC/dataset/real-tensor/3d/reddit-2015.tns -Y /var/GSPARC/dataset/real-tensor/3d/reddit-2015.tns -m 2 -x 0 2 -y 0 2 -t 32 || echo -e "Timeout\n\n" >&2;

# 4D tensors
timeout 12000s ./baselines/Sparta/build/benchmark/ttt -X /var/GSPARC/dataset/quantum-tensor/Qt1 -Y /var/GSPARC/dataset/quantum-tensor/Qt1 -m 4 -x 0 1 2 3 -y 0 1 2 3 -t 32 -q 1 || echo -e "Timeout\n\n" >&2;
timeout 12000s ./baselines/Sparta/build/benchmark/ttt -X /var/GSPARC/dataset/quantum-tensor/Qt2 -Y /var/GSPARC/dataset/quantum-tensor/Qt2 -m 4 -x 0 1 2 3 -y 0 1 2 3 -t 32 -q 1 || echo -e "Timeout\n\n" >&2;



