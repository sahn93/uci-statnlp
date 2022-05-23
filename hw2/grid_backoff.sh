#!/bin/bash
DIR=backoff
mkdir -p $DIR
for n in $(seq 1 9); do
    dir=$DIR/$n
    mkdir -p $dir
    python data.py --output_dir $dir --n $n --backoff 2> $dir/avg_ppl
done

GRID_RESULTS=$DIR/avg_dev_ppl

> $GRID_RESULTS
for n in $(seq 1 9); do
    dir=$DIR/$n
    echo $n $k  $(cat $dir/avg_ppl | head -n2 | tail -n1) >> $GRID_RESULTS
done
