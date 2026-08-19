# !/bin/bash


./build/gsparc -X /var/GSPARC/dataset/synthetic-tensor/base-6d.tns -Y /var/GSPARC/dataset/synthetic-tensor/base-6d.tns -c 4 -x 0 1 2 3 -y 0 1 2 3 -g 1 -n 10;
./build/gsparc -X /var/GSPARC/dataset/synthetic-tensor/skew-6d.tns -Y /var/GSPARC/dataset/synthetic-tensor/skew-6d.tns -c 4 -x 0 1 2 3 -y 0 1 2 3 -g 1 -n 10;
./build/gsparc -X /var/GSPARC/dataset/synthetic-tensor/skew-nips.tns -Y /var/GSPARC/dataset/synthetic-tensor/skew-nips.tns -c 4 -x 0 1 2 3 -y 0 1 2 3 -g 1 -n 10;
./build/gsparc -X /var/GSPARC/dataset/synthetic-tensor/largeout.tns -Y /var/GSPARC/dataset/synthetic-tensor/largeout.tns -c 4 -x 0 1 2 3 -y 0 1 2 3 -g 1 -n 10;
./build/gsparc -X /var/GSPARC/dataset/synthetic-tensor/unbalX.tns -Y /var/GSPARC/dataset/synthetic-tensor/unbalY.tns -c 2 -x 1 2 -y 2 3 -g 1 -n 10;
