# !/bin/bash

# 2-mode contraction (real-world (small))
./baselines/GspTC/build/GspTC -X /var/GSPARC/dataset/real-tensor/3d/nell-2.tns -Y /var/GSPARC/dataset/real-tensor/3d/nell-2.tns -c 2 -x 0 2 -y 0 2 -g 1 -p 2;
./baselines/GspTC/build/GspTC -X /var/GSPARC/dataset/real-tensor/4d/nips.tns -Y /var/GSPARC/dataset/real-tensor/4d/nips.tns -c 2 -x 1 2 -y 1 2 -g 1 -p 2;
./baselines/GspTC/build/GspTC -X /var/GSPARC/dataset/real-tensor/4d/uber.tns -Y /var/GSPARC/dataset/real-tensor/4d/uber.tns -c 2 -x 2 3 -y 2 3 -g 1 -p 2;
./baselines/GspTC/build/GspTC -X /var/GSPARC/dataset/real-tensor/4d/chicago-crime-comm.tns -Y /var/GSPARC/dataset/real-tensor/4d/chicago-crime-comm.tns -c 2 -x 0 2 -y 0 2 -p 2;
./baselines/GspTC/build/GspTC -X /var/GSPARC/dataset/real-tensor/4d/flickr-4d.tns -Y /var/GSPARC/dataset/real-tensor/4d/flickr-4d.tns -c 2 -x 1 2 -y 1 2  -p 2;
./baselines/GspTC/build/GspTC -X /var/GSPARC/dataset/real-tensor/4d/delicious-4d.tns -Y /var/GSPARC/dataset/real-tensor/4d/delicious-4d.tns -c 2 -x 1 2 -y 1 2 -g 1 -p 2;
./baselines/GspTC/build/GspTC -X /var/GSPARC/dataset/real-tensor/5d/vast-2015-mc1-5d.tns -Y /var/GSPARC/dataset/real-tensor/5d/vast-2015-mc1-5d.tns -c 2 -x 0 1 -y 0 1 -g 1 -p 2;

# 3-mode contraction (real-world (small))
./baselines/GspTC/build/GspTC -X /var/GSPARC/dataset/real-tensor/4d/nips.tns -Y /var/GSPARC/dataset/real-tensor/4d/nips.tns -c 3 -x 0 1 2 -y 0 1 2  -g 1 -p 2;
./baselines/GspTC/build/GspTC -X /var/GSPARC/dataset/real-tensor/4d/uber.tns -Y /var/GSPARC/dataset/real-tensor/4d/uber.tns -c 3 -x 0 2 3 -y 0 2 3 -g 1 -p 2;
./baselines/GspTC/build/GspTC -X /var/GSPARC/dataset/real-tensor/4d/chicago-crime-comm.tns -Y /var/GSPARC/dataset/real-tensor/4d/chicago-crime-comm.tns -c 3 -x 0 2 3 -y 0 2 3 -p 2;
./baselines/GspTC/build/GspTC -X /var/GSPARC/dataset/real-tensor/4d/flickr-4d.tns -Y /var/GSPARC/dataset/real-tensor/4d/flickr-4d.tns -c 3 -x 0 1 2 -y 0 1 2 -p 2;
./baselines/GspTC/build/GspTC -X /var/GSPARC/dataset/real-tensor/4d/delicious-4d.tns -Y /var/GSPARC/dataset/real-tensor/4d/delicious-4d.tns -c 3 -x 0 1 2 -y 0 1 2 -g 1 -p 2;
./baselines/GspTC/build/GspTC -X /var/GSPARC/dataset/real-tensor/5d/vast-2015-mc1-5d.tns -Y /var/GSPARC/dataset/real-tensor/5d/vast-2015-mc1-5d.tns -c 3 -x 0 1 3 -y 0 1 3-g 1 -p 2;

# 2-mode contraction (real-world (large))
./baselines/GspTC/build/GspTC -X /var/GSPARC/dataset/real-tensor/3d/amazon-reviews.tns -Y /var/GSPARC/dataset/real-tensor/3d/amazon-reviews.tns -c 2 -x 0 2 -y 0 2 -g 1 -p 2;
./baselines/GspTC/build/GspTC -X /var/GSPARC/dataset/real-tensor/3d/patents.tns -Y /var/GSPARC/dataset/real-tensor/3d/patents.tns -c 2 -x 1 2 -y 1 2 -g 1 -p 2;
./baselines/GspTC/build/GspTC -X /var/GSPARC/dataset/real-tensor/3d/reddit-2015.tns -Y /var/GSPARC/dataset/real-tensor/3d/reddit-2015.tns -c 2 -x 0 2 -y 0 2 -g 1 -p 2;



# 4-mode contraction(quantum dataset)
./baselines/GspTC/build/GspTC -X /var/GSPARC/dataset/quantum-tensor/Qt1 -Y /var/GSPARC/dataset/quantum-tensor/Qt1 -c 4 -x 0 1 2 3 -y 0 1 2 3 -g 1 -q 1 -p 3;
./baselines/GspTC/build/GspTC -X /var/GSPARC/dataset/quantum-tensor/Qt2  -Y /var/GSPARC/dataset/quantum-tensor/Qt2  -c 4 -x 0 1 2 3 -y 0 1 2 3 -q 1 -g 1 -p 3;
