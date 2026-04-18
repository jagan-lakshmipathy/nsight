## Some Commands in Docker and within Docker 
 
 1. docker build -f Dockerfile -t nsight:1.0 . 
 2. docker run --gpus all --cap-add=SYS_ADMIN --security-opt seccomp=unconfined --rm -it -p 8888:8888 eccfcb1b4fac /bin/bash
 3. jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
 4. ncu --set full \
    --target-processes all \
    --export /workspace/my_profile \
    python train_cnn.py
5. different parameter sweeps for regression_normalform.py
   a. ncu --kernel-name "geqr2|ampere_sgemm" --set full --target-processes all --export /workspace/my_profile python regression_normalform.py --method lstsq --n 100000 --d 128 --iters 10
   b. ncu --kernel-name "geqr2|ampere_sgemm" --set full --target-processes all --export /workspace/my_profile python regression_normalform.py --method lstsq --n 100000 --d 128 --iters 10
