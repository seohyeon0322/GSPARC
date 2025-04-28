# GSPARC Installation and Deployment

## 1. Clone the Source Code
```bash
git clone https://github.com/seohyeon0322/GSPARC.git
cd GSPARC
```

## 2. Download Real-World Datasets
```bash
# (Optional) Edit the dataset path inside the script if needed.
./scripts/download_data.sh
```
> By default, datasets are downloaded into the `./dataset` directory.


## 3. Change `CMakeLists.txt` file

```bash
# If the installed GPU has a Compute Capability of 7.5,


set(CMAKE_CUDA_FLAGS  "-O3 -arch=sm_75 -Xcompiler=-fopenmp -mbmi2 -rdc=true")
```


## 4. Build the Docker Image
```bash
docker build -t gsparc .
```

## 5. Run the Docker Container
```bash
# Replace /dataset/path with your actual dataset storage path
docker run -it --name gsparc \
  --runtime=nvidia \
  --gpus all \
  -v /dataset/path:/var/GSPARC/dataset \
  gsparc
```

## 6. Run GSPARC on Sample Data
```bash
./scripts/sample_GSPARC.sh
```

---

# Running GSPARC on All Datasets
```bash
./scripts/run_GSPARC.sh
```

# Running Experiments with Comparison Methods
```bash
./scripts/run_all_exp.sh
./scripts/create_figure.sh
```
