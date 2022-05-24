#!/bin/bash

# exit when any command fails
set -e

for task in ner pos; do
  BASE_PATH=./model/neural_crf_${task}
  # w/ encoder
  for hidden_size in $(seq 20 5 30); do
    for num_layer in $(seq 2 2); do
      # w/o glove
      echo ${BASE_PATH}_h${hidden_size}_l${num_layer}
      python train.py ./config/neural_crf_$task.json \
        --encoder --hidden_size $hidden_size -l $num_layer \
        -s ${BASE_PATH}_h${hidden_size}_l${num_layer}

      MODEL_PATH=${BASE_PATH}_h${hidden_size}_l${num_layer}
      python evaluate.py "$MODEL_PATH" \
        ./data/twitter_test.${task} > "${MODEL_PATH}"/test_scores.json

      # w/ glove
      echo ${BASE_PATH}_h${hidden_size}_l${num_layer}_GloVe
      python train.py ./config/neural_crf_$task.json \
        --glove \
        --encoder --hidden_size $hidden_size -l $num_layer \
        -s ${BASE_PATH}_h${hidden_size}_l${num_layer}_GloVe

      MODEL_PATH=${BASE_PATH}_h${hidden_size}_l${num_layer}_GloVe
      python evaluate.py "$MODEL_PATH" \
        ./data/twitter_test.${task} > "${MODEL_PATH}"/test_scores.json
    done
  done
done
