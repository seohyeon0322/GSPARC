cd dataset

# Synthetic dataset
wget -O synthetic-tensor.tar.gz "https://zenodo.org/records/22005817/files/synthetic-tensor.tar.gz"
tar xzf synthetic-tensor.tar.gz
rm synthetic-tensor.tar.gz


## 3D tensors
mkdir -p real-tensor/3d
cd real-tensor/3d
wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/amazon/amazon-reviews.tns.gz
wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/delicious/delicious-3d.tns.gz
wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/flickr/flickr-3d.tns.gz
wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/nell/nell-2.tns.gz
wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/patents/patents.tns.gz
wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/reddit-2015/reddit-2015.tns.gz
wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/uber-pickups/uber.tns.gz
gunzip *.gz

# 4D tensors
mkdir ../4d
cd ../4d
wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/chicago-crime/comm/chicago-crime-comm.tns.gz
wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/delicious/delicious-4d.tns.gz
wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/flickr/flickr-4d.tns.gz
wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/nips/nips.tns.gz
gunzip *.gz

# 5D tensors
mkdir -p ../5d
cd ../5d
wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/vast-2015-mc1/vast-2015-mc1-5d.tns.gz
gunzip *.gz
cd ../..


### Reference
# frostt.io/tensors