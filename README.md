# Llama-4-Scout-Inference

### 1. Prerequisites : 
- Install Docker & Nvidia Docker. Follow [Link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) </br>
- Make sure all the 8 GPUs are visible. </br>
- Check GPUs status with Command : `nvidia-smi`


### 2. Setup TRT-LLM Docker Container : 

###### Replace this with your Work Space Path. Minimum Disk Space Required : 400GB

```
export HOSTSPACE="/mnt/Scratch_space"  
```
```
sudo docker run --runtime=nvidia --name=TensorRT_LLM_8xGPU_TRT_LLM_0.20_CUDA_12.8.1 --gpus=all --entrypoint /bin/bash \
                --net=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --cap-add=SYS_PTRACE \
                --cap-add=SYS_ADMIN --cap-add=DAC_READ_SEARCH --security-opt seccomp=unconfined -it \
                -v ${HOSTSPACE}:/home/user -w /home/user nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
```

### 3. Install Dependencies ( Inside Docker ) : 

```
apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git git-lfs vim
apt-get -y install libopenmpi-dev && pip3 install --upgrade pip setuptools
apt-get update && apt-get -y install git git-lfs
git lfs install
```

### 4. Clone the Repository : 
```
mkdir /home/user/TRT_LLM_0.20 && cd /home/user/TRT_LLM_0.20 
git clone https://github.com/NVIDIA/TensorRT-LLM.git && cd TensorRT-LLM
git submodule update --init --recursive
git lfs pull
```

### 4. Install TRT-LLM : 

```
git clone https://github.com/ajithAI/Llama-4-Scout-Inference.git
cp Llama-4-Scout-Inference/Makefile /home/user/TRT_LLM_0.20/TensorRT-LLM/docker/Makefile
cd /home/user/TRT_LLM_0.20/TensorRT-LLM
make -C docker release_build
pip3 install numpy==1.26.4
```

### 5. Check Installation : 

```
python3 -c "import tensorrt_llm"  # < Prints TRT Version >
```
```
[TensorRT-LLM] TensorRT-LLM version: 0.20.0
```

### 6. Download Mixtral 8x7B Model from HuggingFace : 
###### Use Huggingface Login Credentials to download the Mixtral Model.
```
huggingface-cli download meta-llama/Llama-4-Scout-17B-16E-Instruct --local-dir Llama-4-Scout-17B-16E-Instruct
```

### 7. Run Quantization & Create Checkpoint :
```
export DIR=/home/user/TRT_LLM_0.20/Llama4_Scout_Inference && mkdir $DIR
```
```
cd /home/user/TRT_LLM_0.20/TensorRT-LLM/examples
python3 quantization/quantize.py --model_dir /home/user/Llama-4-Scout-17B-16E-Instruct --dtype float16 \
      --qformat fp8 --kv_cache_dtype fp8 --calib_size 512 --tp_size 8 \
      --output_dir /home/user/TRT_LLM_0.20/Llama4_Scout_Inference/CHECKPOINT
```

### 8. Create TRT Engine :

```
export DIR=/home/user/TRT_LLM_0.20/Llama4_Scout_Inference && cd $DIR && mkdir $DIR/TRT_Engines
```
```
trtllm-build --checkpoint_dir $DIR/CHECKPOINT --output_dir $DIR/TRT_Engines/LLlama4_Scout_TRT_Engine_FP8_TP8_MaxBatch_8192_MaxSeqLen_4096 \
             --gemm_plugin auto --use_fp8_context_fmha enable --use_fused_mlp enable --use_paged_context_fmha enable \
             --workers 8 --max_batch_size 8192 --max_seq_len 4096
```

### 9. To Run Benchmark inside Docker : 
```
mkdir LLaMA3.1_BM_LOGS
```

```
USAGE : <RUN_SCRIPT> <BATCH_SIZE> <INPUT_LENGTH> <OUTPUT_LENGTH> <WARMUP_ITER> <BENCHMARK_ITER> <OUTPUT_FILENAME>

./run_llama_inference.sh 64 2048 2048 25 75 LLaMA3.1_BM_LOGS/LLaMA3.1_70B_TRT_Batch_64_Input_2048_Output_2048
./run_llama_inference.sh 96 2048 128 50 200 LLaMA3.1_BM_LOGS/LLaMA3.1_70B_TRT_Batch_96_Input_2048_Output_128
./run_llama_inference.sh 1024 128 128 50 150 LLaMA3.1_BM_LOGS/LLaMA3.1_70B_TRT_Batch_1024_Input_128_Output_128
./run_llama_inference.sh 1024 128 2048 5 25 LLaMA3.1_BM_LOGS/LLaMA3.1_70B_TRT_Batch_1024_Input_128_Output_2048

./run_llama_inference.sh 64 2048 1 50 250 LLaMA3.1_BM_LOGS/LLaMA3.1_70B_TRT_Batch_64_Input_2048_Output_1
./run_llama_inference.sh 96 2048 1 50 250 LLaMA3.1_BM_LOGS/LLaMA3.1_70Bl_TRT_Batch_96_Input_2048_Output_1
./run_llama_inference.sh 1024 128 1 50 250 LLaMA3.1_BM_LOGS/LLaMA3.1_70B_TRT_Batch_1024_Input_128_Output_1
```
###### Note : To save CPU Turbostat Logs & NVIDIA-SMI Logs, run from Docker Outside. 

### 10. To Run Benchmark from Docker Outside : 
```
cd ${HOSTSPACE}/LLaMA3.1_Inference
```

```
./docker_run_llama_inference.sh 64 2048 2048 10 50 MaxFreq
./docker_run_llama_inference.sh 96 2048 128 50 200 MaxFreq
./docker_run_llama_inference.sh 1024 128 128 25 100 MaxFreq
./docker_run_llama_inference.sh 1024 128 2048 5 25 MaxFreq

./docker_run_llama_inference.sh 64 2048 1 50 200 MaxFreq
./docker_run_llama_inference.sh 96 2048 1 50 200 MaxFreq
./docker_run_llama_inference.sh 1024 128 1 50 200 MaxFreq
```
##### (OR) Simply : 
```
cd ${HOSTSPACE}/LLaMA3.1_Inference && bash ./docker_run_benchmark.sh 
```

### 11. To Re-Enter TRT-LLM Docker Environment :
```
sudo docker restart TensorRT_LLM_8xGPU_CUDA_12.6.0
sudo docker exec -it TensorRT_LLM_8xGPU_CUDA_12.6.0 bash
``` 

### 12. Error Handlings : 

###### For Error : CUDA initialization: Unexpected error from cudaGetDeviceCount()

```
sudo systemctl stop nvidia-fabricmanager
sudo systemctl restart nvidia-fabricmanager
sudo systemctl status nvidia-fabricmanager
```
