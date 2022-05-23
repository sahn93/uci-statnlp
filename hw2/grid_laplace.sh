#!/bin/bash
DIR=laplace
mkdir -p $DIR
for n in $(seq 1 4); do
    for k in $(seq 0.0 0.2 1.0); do
        dir=$DIR/${n}_${k}
        mkdir -p $dir
        python data.py --output_dir $dir --n $n --laplace $k 2> $dir/avg_ppl
    done
done

GRID_RESULTS=$DIR/avg_dev_ppl
> $GRID_RESULTS
for n in $(seq 1 4); do
    for k in $(seq 0.0 0.2 1.0); do
        dir=$DIR/${n}_${k}
        echo $n $k  $(cat $dir/avg_ppl | head -n2 | tail -n1) >> $GRID_RESULTS
    done
done
