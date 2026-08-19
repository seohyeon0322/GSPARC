# !/bin/bash

./build/gsparc -X /var/GSPARC/dataset/real-tensor/3d/amazon-reviews.tns -Y /var/GSPARC/dataset/real-tensor/3d/amazon-reviews.tns -c 2 -x 0 2 -y 0 2 -g 1 -n 1 -m 12;
./build/gsparc -X /var/GSPARC/dataset/real-tensor/3d/amazon-reviews.tns -Y /var/GSPARC/dataset/real-tensor/3d/amazon-reviews.tns -c 2 -x 0 2 -y 0 2 -g 1 -n 1 -m 24;
./build/gsparc -X /var/GSPARC/dataset/real-tensor/3d/amazon-reviews.tns -Y /var/GSPARC/dataset/real-tensor/3d/amazon-reviews.tns -c 2 -x 0 2 -y 0 2 -g 1 -n 1 -m 48;
./build/gsparc -X /var/GSPARC/dataset/real-tensor/3d/amazon-reviews.tns -Y /var/GSPARC/dataset/real-tensor/3d/amazon-reviews.tns -c 2 -x 0 2 -y 0 2 -g 1 -n 1 -m 96;