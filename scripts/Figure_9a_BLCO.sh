#!/bin/bash

./baselines/BLCO/build/cpd64 -i /var/GSPARC/dataset/real-tensor/3d/nell-2.tns -k 1
./baselines/BLCO/build/cpd64 -i /var/GSPARC/dataset/real-tensor/4d/nips.tns -k 1
./baselines/BLCO/build/cpd64 -i /var/GSPARC/dataset/real-tensor/3d/patents.tns -k 1
./baselines/BLCO/build/cpd64 -q /var/GSPARC/dataset/quantum-tensor/Qt1 -k 1
