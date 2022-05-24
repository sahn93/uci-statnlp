#!/bin/bash

# exit when any command fails
set -e

for task in ner pos; do
  BASE_PATH=./model/neural_crf_${task}
  # default
  echo ${BASE_PATH}
  python train.py ./config/neural_crf_$task.json \
    -s ${BASE_PATH}

  # w/ glove
  echo ${BASE_PATH}_GloVe
  python train.py ./config/neural_crf_$task.json \
    --glove \
    -s ${BASE_PATH}_GloVe


  # w/ encoder
  for hidden_size in $(seq 20 5 30); do
    for num_layer in $(seq 1 3); do
      # w/o glove
      echo ${BASE_PATH}_h${hidden_size}_l${num_layer}
      python train.py ./config/neural_crf_$task.json \
        --encoder --hidden_size $hidden_size -l $num_layer \
        -s ${BASE_PATH}_h${hidden_size}_l${num_layer}

      # w/ glove
      echo ${BASE_PATH}_h${hidden_size}_l${num_layer}_GloVe
      python train.py ./config/neural_crf_$task.json \
        --glove \
        --encoder --hidden_size $hidden_size -l $num_layer \
        -s ${BASE_PATH}_h${hidden_size}_l${num_layer}_GloVe
    done
  done
done
