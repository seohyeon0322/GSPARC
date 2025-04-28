#!/bin/bash


./build/gsparc -X /var/GSPARC/dataset/real-tensor/4d/nips.tns -Y /var/GSPARC/dataset/real-tensor/4d/nips.tns -c 2 -a 2 -x 1 2 -y 1 2 -z 0 3 -g 1 -m 1 -n 1;
./build/gsparc -X /var/GSPARC/dataset/real-tensor/4d/uber.tns -Y /var/GSPARC/dataset/real-tensor/4d/uber.tns -c 2 -a 2 -x 2 3 -y 2 3 -z 0 1 -m 1 -g 1 -n 1;
./build/gsparc -X /var/GSPARC/dataset/real-tensor/4d/chicago-crime-comm.tns -Y /var/GSPARC/dataset/real-tensor/4d/chicago-crime-comm.tns -c 2 -a 2 -x 0 2 -y 0 2 -z 1 3 -m 1 -g 1 -n 1;

