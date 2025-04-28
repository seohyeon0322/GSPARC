#!/bin/bash

export EXPERIMENT_MODES=3
# $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nell-2.bin -Y $TENSOR_DIR/nell-2.bin -m 2 -x 0 2 -y 0 2 -t 32 ;
# ./build/benchmark/ttt -X $TENSOR_DIR/nips.bin -Y $TENSOR_DIR/nips.bin -m 3 -x 0 1 2 -y 0 1 2 -t 32 ;
./build/benchmark/ttt -X $TENSOR_DIR/nips.bin -Y $TENSOR_DIR/nips.bin -m 2 -x 1 2 -y 1 2 -t 32 ;

# $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/uber.bin -Y $TENSOR_DIR/uber.bin -m 3 -x 0 2 3 -y 0 2 3 -t 32 ;
# $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/chicago.bin -Y $TENSOR_DIR/chicago.bin -m 3 -x 0 2 3 -y 0 2 3 - t 32 ;
# $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/shkim/tensors/enron.tns -Y /home/asdf0322/data/shkim/tensors/enron.tns -m 3 -x 0 1 2 -y 0 1 2 -t 32;
# $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/real-tensor/5d/chicago-crime-geo.tns -Y /home/asdf0322/data/jihye/real-tensor/5d/chicago-crime-geo.tns -m 4 -x 0 2 3 4 -y 0 2 3 4 -t 32;


# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nell-2.bin -Y $TENSOR_DIR/nell-2.bin -m 2 -x 0 2 -y 0 2 -t 32 ;
# timeout 10s numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nips.bin -Y $TENSOR_DIR/nips.bin -m 2 -x 1 2 -y 1 2 -t 32 || echo "Timeout occurred";
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/uber.bin -Y $TENSOR_DIR/uber.bin -m 2 -x 2 3 -y 2 3 -t 32 ;
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/chicago.bin -Y $TENSOR_DIR/chicago.bin -m 2 -x 0 2 -y 0 2 -t 32 ;
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/flickr-4d.bin -Y $TENSOR_DIR/flickr-4d.bin -m 2 -x 1 2 -y 1 2 -t 32 ;
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/delicious.bin -Y $TENSOR_DIR/delicious.bin -m 2 -x 1 2 -y 1 2 -t 32 ;
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/vast.bin -Y $TENSOR_DIR/vast.bin -m 2 -x 0 1 -y 0 1 -t 32 ;
# $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/delicious.bin -Y $TENSOR_DIR/delicious.bin -m 2 -x 1 2 -y 1 2 -t 32 ;


# timeout 12000s $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/real-tensor/3d/patents.tns -Y /home/asdf0322/data/jihye/real-tensor/3d/patents.tns -m 2 -x 1 2 -y 1 2 -t 32 || echo "Timeout occurred";
# timeout 12000s $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/real-tensor/3d/amazon-reviews.tns -Y /home/asdf0322/data/jihye/real-tensor/3d/amazon-reviews.tns -m 2 -x 0 2 -y 0 2 -t 32 || echo "Timeout occurred";
# timeout 12000s $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/real-tensor/3d/reddit-2015.tns -Y /home/asdf0322/data/jihye/real-tensor/3d/reddit-2015.tns -m 2 -x 0 2 -y 0 2 -t 32 || echo "Timeout occurred";
# $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/real-tensor/3d/reddit-2015.tns -Y /home/asdf0322/data/jihye/real-tensor/3d/reddit-2015.tns -m 2 -x 0 2 -y 0 2 -t 12;

#  $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/real-tensor/3d/patents.tns -Y /home/asdf0322/data/jihye/real-tensor/3d/patents.tns -m 2 -x 1 2 -y 1 2 -t 32 
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/shkim/tensors/enron.tns -Y /home/asdf0322/data/shkim/tensors/enron.tns -m 3 -x 0 1 2 -y 0 1 2 -t 32;
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/real-tensor/5d/chicago-crime-geo.tns -Y /home/asdf0322/data/jihye/real-tensor/5d/chicago-crime-geo.tns -m 4 -x 0 2 3 4 -y 0 2 3 4 -t 32;



# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/quantum-tensor/F2_0.15 -Y /home/asdf0322/data/jihye/quantum-tensor/F2_0.15 -m 4 -x 0 1 2 3 -y 0 1 2 3 -t 32 ;
# numactl --cpunodebind=0 --membind=0  $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/quantum-tensor/F2_0.09 -Y /home/asdf0322/data/jihye/quantum-tensor/F2_0.09 -m 4 -x 0 1 2 3 -y 0 1 2 3 -t 32; 
# numactl --cpunodebind=0 --membind=0  $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/quantum-tensor/Si_0.19 -Y /home/asdf0322/data/jihye/quantum-tensor/Si_0.19 -m 4 -x 0 1 2 3 -y 0 1 2 3 -t 32; 


# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nips.bin -Y $TENSOR_DIR/nips.bin -m 3 -x 0 1 2 -y 0 1 2 -t 32 ;
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/uber.bin -Y $TENSOR_DIR/uber.bin -m 3 -x 0 2 3 -y 0 2 3 -t 32 ;
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/chicago.bin -Y $TENSOR_DIR/chicago.bin -m 3 -x 0 2 3 -y 0 2 3 - t 32 ;
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/flickr-4d.bin -Y $TENSOR_DIR/flickr-4d.bin  -m 3 -x 0 1 2 -y 0 1 2 -t 32 ;
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/delicious.bin -Y $TENSOR_DIR/delicious.bin -m 3 -x 0 1 2 -y 0 1 2 -t 32 ;
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/vast.bin -Y $TENSOR_DIR/vast.bin -m 3 -x 0 1 3 -y 0 1 3 -t 32 ;

# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/shkim/tensors/enron.tns -Y /home/asdf0322/data/shkim/tensors/enron.tns -m 3 -x 0 1 2 -y 0 1 2 -t 32;
# numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/real-tensor/5d/chicago-crime-geo.tns -Y /home/asdf0322/data/jihye/real-tensor/5d/chicago-crime-geo.tns -m 4 -x 0 2 3 4 -y 0 2 3 4 -t 32;

# $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/quantum-tensor/F2_0.15 -Y /home/asdf0322/data/jihye/quantum-tensor/F2_0.15 -m 4 -x 0 1 2 3 -y 0 1 2 3 -t 32 ;
# $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/quantum-tensor/Si_0.19 -Y /home/asdf0322/data/jihye/quantum-tensor/Si_0.19 -m 4 -x 0 1 2 3 -y 0 1 2 3 -t 32; 
# $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/quantum-tensor/F2_0.09 -Y /home/asdf0322/data/jihye/quantum-tensor/F2_0.09 -m 4 -x 0 1 2 3 -y 0 1 2 3 -t 32; 

# $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/quantum-tensor/F2_0.15 -Y /home/asdf0322/data/jihye/quantum-tensor/F2_0.15 -m 3 -x 0 1 2 -y 0 1 2 -t 32 ;
# $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/quantum-tensor/Si_0.19 -Y /home/asdf0322/data/jihye/quantum-tensor/Si_0.19 -m 3 -x 0 1 2 -y 0 1 2 -t 32; 
# $SPARTA_DIR/build/benchmark/ttt -X /home/asdf0322/data/jihye/quantum-tensor/F2_0.09 -Y /home/asdf0322/data/jihye/quantum-tensor/F2_0.09 -m 3 -x 0 1 2 -y 0 1 2 -t 32; 
