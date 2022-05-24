#!/bin/bash

# exit when any command fails
set -e

for task in ner pos; do
  BASE_PATH=./model/neural_crf_${task}
  # default
  MODEL_PATH=$BASE_PATH
  echo $MODEL_PATH
  python evaluate.py $MODEL_PATH \
    ./data/twitter_test.${task} > ${MODEL_PATH}/test_scores.json

  # w/ glove
  MODEL_PATH=${BASE_PATH}_GloVe
  echo $MODEL_PATH
  python evaluate.py $MODEL_PATH \
    ./data/twitter_test.${task} > ${MODEL_PATH}/test_scores.json

  # w/ encoder
  for hidden_size in $(seq 20 5 30); do
    for num_layer in $(seq 1 3); do
      # w/o glove
      MODEL_PATH=${BASE_PATH}_h${hidden_size}_l${num_layer}
      echo "$MODEL_PATH"
      python evaluate.py "$MODEL_PATH" \
        ./data/twitter_test.${task} > "${MODEL_PATH}"/test_scores.json

      # w/ glove
      MODEL_PATH=${BASE_PATH}_h${hidden_size}_l${num_layer}_GloVe
      echo "$MODEL_PATH"
      python evaluate.py "$MODEL_PATH" \
        ./data/twitter_test.${task} > "${MODEL_PATH}"/test_scores.json
    done
  done
done
