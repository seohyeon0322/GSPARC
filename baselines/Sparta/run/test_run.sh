#!/bin/bash

export EXPERIMENT_MODES=3
# 
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/quantum-tensor/F2_0.15 -Y /home/asdf0322/data/jihye/quantum-tensor/F2_0.15 -m 3 -x 0 1 2 -y 0 1 2 -t 32 ;
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/quantum-tensor/F2_0.15 -Y /home/asdf0322/data/jihye/quantum-tensor/F2_0.15 -m 3 -x 0 1 3 -y 0 1 3 -t 32 ;

# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/quantum-tensor/F2_0.15 -Y /home/asdf0322/data/jihye/quantum-tensor/F2_0.15 -m 4 -x 0 1 2 3 -y 0 1 2 3 -t 32 ;

# numactl --cpunodebind=0 --membind=0  $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/quantum-tensor/F2_0.09 -Y /home/asdf0322/data/jihye/quantum-tensor/F2_0.09 -m 4 -x 0 1 2 3 -y 0 1 2 3 -t 64; 
# numactl --cpunodebind=0 --membind=0  $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/quantum-tensor/Si_0.19 -Y /home/asdf0322/data/jihye/quantum-tensor/Si_0.19 -m 4 -x 0 1 2 3 -y 0 1 2 3 -t 32; 



# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/vast.bin -Y $TENSOR_DIR/vast.bin -m 2 -x 2 3 -y 2 3 -t 12 ;
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X  /home/asdf0322/data/jihye/real-tensor/3d/amazon-reviews.tns -Y  /home/asdf0322/data/jihye/real-tensor/3d/amazon-reviews.tns -m 2 -x 0 2 -y 0 2 -t 64 ;
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X  /home/asdf0322/data/jihye/real-tensor/4d/nips.tns -Y  /home/asdf0322/data/jihye/real-tensor/4d/nips.tns -Z ./output3.tns -m 3 -x 0 1 2 -y 0 1 2 -t 32 ;
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X  /home/asdf0322/data/jihye/real-tensor/4d/nips.tns -Y  /home/asdf0322/data/jihye/real-tensor/4d/nips.tns -Z ./output3.tns -m 3 -x 0 2 3 -y 0 2 3 -t 32 ;
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X  /home/asdf0322/data/jihye/real-tensor/4d/nips.tns -Y  /home/asdf0322/data/jihye/real-tensor/4d/nips.tns -Z ./output3.tns -m 3 -x 1 2 3 -y 1 2 3 -t 32 ;

# # #mode 3
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X  /home/asdf0322/data/jihye/real-tensor/4d/delicious-4d.tns -Y  /home/asdf0322/data/jihye/real-tensor/4d/delicious-4d.tns -m 3 -x 0 1 2 -y 0 1 2 -t 12 ;

# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/real-tensor/4d/chicago-crime-comm.tns -Y /home/asdf0322/data/jihye/real-tensor/4d/chicago-crime-comm.tns -m 3 -x 0 2 3 -y 0 2 3 -t 12;
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/real-tensor/4d/chicago-crime-comm.tns -Y /home/asdf0322/data/jihye/real-tensor/5d/chicago-crime-geo.tns -m 3 -x 0 1 3 -y 0 1 4 -t 12;
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/real-tensor/5d/chicago-crime-geo.tns -Y /home/asdf0322/data/jihye/real-tensor/5d/chicago-crime-geo.tns -m 3 -x 0 2 3 -y 0 2 3 -t 12;

# #NIPS
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nips.bin -Y $TENSOR_DIR/nips.bin -m 3 -x 0 1 2 -y 0 1 2 -t 40 ;
# $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/real-tensor/4d/nips.tns -Y /home/asdf0322/data/jihye/real-tensor/4d/nips.tns -m 3 -x 0 1 3 -y 0 1 3 -t 12 ;

# #Uber
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/uber.bin -Y $TENSOR_DIR/uber.bin -m 3 -x 0 2 3 -y 0 2 3 -t 32 ;
# $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/uber.bin -Y $TENSOR_DIR/uber.bin -m 3 -x 0 1 2 -y 0 1 2 -t 12 ;

# #Chicago
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/chicago.bin -Y $TENSOR_DIR/chicago.bin -m 3 -x 0 2 3 -y 0 2 3 - t 32 ;
# $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/real-tensor/4d/chicago-crime-comm.tns -Y /home/asdf0322/data/jihye/real-tensor/4d/chicago-crime-comm.tns -m 3 -x 1 2 3 -y 1 2 3 -t 12 ;

# #Uracil
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/uracil_trimer.sto-3g_T2.bin -Y $TENSOR_DIR/uracil_trimer.sto-3g_T2.bin -m 3 -x 1 2 3 -y 1 2 3 -t 12 ;
# $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/uracil_trimer.sto-3g_T2.bin -Y $TENSOR_DIR/uracil_trimer.sto-3g_T2.bin -m 3 -x 0 1 2 -y 0 1 2 -t 12 ;

# #Flickr
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/flickr-4d.bin -Y $TENSOR_DIR/flickr-4d.bin  -m 3 -x 0 1 2 -y 0 1 2 -t 24 ;
# $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/flickr-4d.bin -Y $TENSOR_DIR/flickr-4d.bin -m 3 -x 0 2 3 -y 0 2 3 -t 12 ;

# #Delicious
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/delicious.bin -Y $TENSOR_DIR/delicious.bin -m 3 -x 0 1 2 -y 0 1 2 -t 24 ;
# $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/delicious.bin -Y $TENSOR_DIR/delicious.bin -m 3 -x 0 2 3 -y 0 2 3 -t 12 ;

# #Vast
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/vast.bin -Y $TENSOR_DIR/vast.bin -m 3 -x 0 1 3 -y 0 1 3 -t 12 ;
# $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/vast.bin -Y $TENSOR_DIR/vast.bin -m 3 -x 2 3 4 -y 2 3 4 -t 12 ;

# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/shkim/tensors/enron.tns -Y /home/asdf0322/data/shkim/tensors/enron.tns -m 3 -x 0 1 2 -y 0 1 2 -t 12;

# #mode 2
# #NELL-2
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nell-2.bin -Y $TENSOR_DIR/nell-2.bin -m 2 -x 0 2 -y 0 2 -t 32 ;
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nell-2.bin -Y $TENSOR_DIR/nell-2.bin -m 2 -x 0 1 -y 0 1 -t 32 ;
# $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nell-2.bin -Y $TENSOR_DIR/nell-2.bin -m 2 -x 0 1 -y 0 1 -t 32 ;

# #NIPS
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nips.bin -Y $TENSOR_DIR/nips.bin -m 2 -x 1 2 -y 1 2 -t 12 ;
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nips.bin -Y $TENSOR_DIR/nips.bin -m 2 -x 0 3 -y 0 3 -t 32 ;
# $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nips.bin -Y $TENSOR_DIR/nips.bin -m 2 -x 0 3 -y 0 3 -t 32 ;

# #Uber
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/uber.bin -Y $TENSOR_DIR/uber.bin -m 2 -x 2 3 -y 2 3 -t 32 ;
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/uber.bin -Y $TENSOR_DIR/uber.bin -m 2 -x 0 1 -y 0 1 -t 32 ;
# $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/uber.bin -Y $TENSOR_DIR/uber.bin -m 2 -x 0 1 -y 0 1 -t 32 ;

# #Chicago
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/chicago.bin -Y $TENSOR_DIR/chicago.bin -m 2 -x 0 2 -y 0 2 -t 32 ;
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/chicago.bin -Y $TENSOR_DIR/chicago.bin -m 2 -x 1 3 -y 1 3 -t 32 ;
# $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/chicago.bin -Y $TENSOR_DIR/chicago.bin -m 2 -x 1 3 -y 1 3 -t 32 ;


# #Uracil
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/uracil_trimer.sto-3g_T2.bin -Y $TENSOR_DIR/uracil_trimer.sto-3g_T2.bin -m 2 -x 2 3 -y 2 3 -t 12 ;
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/uracil_trimer.sto-3g_T2.bin -Y $TENSOR_DIR/uracil_trimer.sto-3g_T2.bin -m 2 -x 0 1 -y 0 1 -t 12 ;

# #Flickr
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/flickr-4d.bin -Y $TENSOR_DIR/flickr-4d.bin -m 2 -x 1 2 -y 1 2 -t 12 ;
# $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/flickr-4d.bin -Y $TENSOR_DIR/flickr-4d.bin -m 2 -x 0 3 -y 0 3 -t 32 ;

# #Delicious
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/delicious.bin -Y $TENSOR_DIR/delicious.bin -m 2 -x 1 2 -y 1 2 -t 3 ;
# $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/delicious.bin -Y $TENSOR_DIR/delicious.bin -m 2 -x 0 3 -y 0 3 -t 32 ;

# #Vast
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/vast.bin -Y $TENSOR_DIR/vast.bin -m 2 -x 0 1 -y 0 1 -t 24 ;
# $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/vast.bin -Y $TENSOR_DIR/vast.bin -m 2 -x 2 4 -y 2 4 -t 12 ;

# # Amazon Reviews
$SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/real-tensor/3d/amazon-reviews.tns -Y /home/asdf0322/data/jihye/real-tensor/3d/amazon-reviews.tns -m 2 -x 0 2 -y 0 2 -t 32 ;
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/real-tensor/3d/amazon-reviews.tns -Y /home/asdf0322/data/jihye/real-tensor/3d/amazon-reviews.tns -m 2 -x 1 2 -y 1 2 -t 12 ;

# # #Patents
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/real-tensor/3d/patents.tns -Y /home/asdf0322/data/jihye/real-tensor/3d/patents.tns -m 2 -x 0 2 -y 0 2 -t 12 ;
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/real-tensor/3d/patents.tns -Y /home/asdf0322/data/jihye/real-tensor/3d/patents.tns -m 2 -x 1 2 -y 1 2 -t 12 ;

# #Reddit-2015
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/real-tensor/3d/reddit-2015.tns -Y /home/asdf0322/data/jihye/real-tensor/3d/reddit-2015.tns -m 2 -x 0 2 -y 0 2 -t 12 ;
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/real-tensor/3d/reddit-2015.tns -Y /home/asdf0322/data/jihye/real-tensor/3d/reddit-2015.tns -m 2 -x 1 2 -y 1 2 -t 12 ;

