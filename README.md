# Predicting Time Series with LSTM and Reformer

lorem ipsum

# Setup

To build the docker image with all the dependencies run:
```bash
docker build --rm -t fiskio/ml-base .
```
This is an optional step as the public image `fiskio/ml-base` will be pulled on first run. 

# Run

To run the tests:
```bash
./run_docker.sh python -m pytest -s xtx/tests/tests.py
```

To run a jupyter server within the container run:
```bash
./run_docker.sh jupyter notebook --no-browser --allow-root --ip=0.0.0.0
```
and click on the link generated (e.g. `http://127.0.0.1:8888/?token=a676cbe6c4f4422f92bcbd514a51fa9356cfe65bd2e7fd15`)

To train an LSTM model run:
```bash
./run_docker.sh python xtx/apps/train_lstm.py --mat_file data/data.mat
```
or on multiple GPUs:
```bash
python -m torch.distributed.launch --nproc_per_node=2  xtx/apps/train_lstm.py --mat_file data/data.mat --fp16_opt_level O2
```
