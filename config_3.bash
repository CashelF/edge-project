#!/bin/bash

# This script is used to generate commands to run cloud.py and device.py on edge devices
# Change the parameters below and run "bash run.bash" on terminal
# It will also run "generate_configs.py" based on the given parameters

############### TODO CHANGE
cloud_ip="172.29.197.214" #Change every time when new VPN is connected

model_name="conv3smallBN"
loss_func_name="cross_entropy"
loss_type="fedavg" # change to fedmax or fedprox
mu=0.2 # For FedProx
beta=100.0 # For FedMAX
learning_rate=0.015

declare -a experiment_configs=(
# experiment | run | data_iid
  # "1 1 true" # Experiment 1, Run 1, IID
  # "2 1 false" # Experiment 2, Run 1, Non-IID
  # "3 1 false" # Experiment 3, Run 1, Non-IID  (LEARNING RATE FROM 0.01 TO 0.1)
  # "4 1 false" # Experiment 4, Run 1, Non-IID  (LOCAL EPOCHS FROM 1 TO 2)
  # "5 1 false" # Experiment 5, Run 1, Non-IID  (LEARNING RATE FROM 0.01 TO 0.02)
  # "6 1 false" # Experiment 6, Run 1, Non-IID  (ADD BATCH NORMALIZATION TO CONV5SMALL)
  # "7 1 true" # Experiment 7, Run 1, IID  (Try FedMax)
  # "8 1 false" # Experiment 8, Run 1, Non-IID  (Try FedMax)
  # "9 1 true" # Experiment 9, Run 1, IID  (Try FedProx)
  # "10 1 false" # Experiment 10, Run 1, Non-IID  (Try FedProx)
  # "11 1 false" # Experiment 11, Run 1, Non-IID  (Try FedProx, mu=0.5)
  # "12 1 false" # Experiment 12, Run 1, Non-IID  (FedAvg, BN, lr=0.015)
  # "13 1 false" # Experiment 13, Run 1, Non-IID  (FedAvg, BN, lr=0.025)
  # "14 1 false" # Experiment 14, Run 1, Non-IID  (FedAvg, BN, lr=0.02)
  # "15 1 false" # Experiment 15, Run 1, Non-IID  (FedMax, BN, lr=0.015, Beta=100)
  # "16 1 false" # Experiment 16, Run 1, Non-IID  (FedMax, BN, lr=0.015, Beta=10)
  # "17 1 false" # Experiment 17, Run 1, Non-IID  (FedProx, BN, lr=0.015, mu=.2)
  # "18 1 false" # Experiment 18, Run 1, Non-IID  (FedAvg, BN, lr=0.015, mu=.5, Conv3smallBN)
  # "19 1 false" # Experiment 19, Run 1, Non-IID  (FedAvg, BN, lr=0.015, mu=.5, Conv1smallBN)
  # "20 1 false" # Experiment 20, Run 1, Non-IID  (FedAvg, BN, lr=0.015, mu=.5, Conv1smallBN, mc1-2local rpi-1local)
  # "21 1 false" # Experiment 21, Run 1, Non-IID  (FedAvg, BN, lr=0.015, Conv3smallBN, mc1-2local rpi-1local)
  # "21 2 false" # Experiment 21, Run 2, Non-IID  (FedAvg, BN, lr=0.015, Conv3smallBN, mc1-2local rpi-1local)
  "21 3 false" # Experiment 21, Run 3, Non-IID  (FedAvg, BN, lr=0.015, Conv3smallBN, mc1-2local rpi-1local)
  # "22 1 false" # Experiment 22, Run 1, Non-IID  (FedAvg, BN, lr=0.015, Conv3smallBN, mc1-2local rpi-1local, not divide by 255)
  # "23 1 false" # Experiment 23, Run 1, Non-IID  (FedAvg, BN, lr=0.015, Conv2smallBN, mc1-2local rpi-1local)
  # "24 1 false" # Experiment 24, Run 1, Non-IID  (FedAvg, BN, lr=0.015, Conv3smallBN, depth-wise, mc1-2local rpi-1local)
  # "25 1 false" # Experiment 24, Run 1, Non-IID  (FedAvg, BN, lr=0.015, Conv3smallBN, depth-wise take 2, mc1-2local rpi-1local)
)

declare -a devices_configs=(
# hw_type | host | port | cuda_name | local_epochs
  "rpi sld-rpi-15.ece.utexas.edu 9090 cpu 1" #Change number of device
  "mc1 sld-mc1-15.ece.utexas.edu 9090 cpu 2" #Change number of device
)
############### TODO END CHANGE

verbose='false'
laptop_number='laptop_1'
cloud_port="9090"
cloud_cuda="cpu"
comm_rounds=30
num_devices=2
for experiment_config in "${experiment_configs[@]}"
do
  read -a exp_config <<< "$experiment_config"
  experiment="${exp_config[0]}"
  run="${exp_config[1]}"
  # Seeds for all runs are predetermined.
  if [ "$run" -eq 1 ]; then
    seed=2
  elif [ "$run" -eq 2 ]; then
    seed=14
  elif [ "$run" -eq 3 ]; then
    seed=26
  else
    echo "Invalid run value. Please specify 1, 2, or 3."
    exit 1
  fi
  data_iid="${exp_config[2]}"

  declare -a dev_hw_types
  declare -a ips
  declare -a ports
  declare -a model_names
  declare -a dev_local_epochs
  for devs_configs in "${devices_configs[@]}"
  do
    read -a dev_config <<< "$devs_configs"
    hw_type="${dev_config[0]}"
    dev_hw_types+=("$hw_type")

    host="${dev_config[1]}"
    hosts+=("$host")

    port="${dev_config[2]}"
    ports+=("$port")

    cuda_name="${dev_config[3]}"
    cuda_names+=("$cuda_name")

    model_names+=("$model_name")
    local_epochs="${dev_config[4]}"
    dev_local_epochs+=("$local_epochs")
  done

  cloud_config_filename="cloud_cfg_exp${experiment}_run${run}.json"
  dev_config_filename="dev_cfg_exp${experiment}_run${run}.json"

  python generate_configs.py \
  --cloud_config_filename "${cloud_config_filename}" \
  --dev_config_filename "${dev_config_filename}" \
  --cloud_ip "${cloud_ip}" \
  --cloud_port "${cloud_port}" \
  --cloud_cuda "${cloud_cuda}" \
  --model_name "${model_name}" \
  --loss_func_name "${loss_func_name}" \
  --loss_type "${loss_type}" \
  --mu "${mu}" \
  --beta "${beta}" \
  --comm_rounds "${comm_rounds}" \
  --learning_rate "${learning_rate}" \
  --verbose "${verbose}" \
  --experiment "${experiment}" \
  --run "${run}" \
  --seed "${seed}" \
  --laptop_number "${laptop_number}" \
  --data_iid "${data_iid}" \
  --num_devices "${num_devices}" \
  --dev_hw_types "${dev_hw_types[*]}" \
  --hosts "${hosts[*]}" \
  --ports "${ports[*]}" \
  --cuda_names "${cuda_names[*]}" \
  --model_names "${model_names[*]}" \
  --dev_local_epochs "${dev_local_epochs[*]}"
done
