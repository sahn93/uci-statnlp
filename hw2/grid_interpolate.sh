#!/bin/bash
DIR=interpolate

mkdir -p $DIR
for l1 in $(seq 0.1 0.2 0.9); do
    l2=$(echo "print(round(1-$l1, 2))" | python)
    dir=$DIR/${l1}_${l2}
    mkdir -p $dir
    python data.py --output_dir $dir --n 2 --interpolat $l1 $l2 2> $dir/avg_ppl
done

GRID_RESULTS=$DIR/avg_dev_ppl

> $GRID_RESULTS
for l1 in $(seq 0.1 0.2 0.9); do
    l2=$(echo "print(round(1-$l1, 2))" | python)
    dir=$DIR/${l1}_${l2}
    echo $l1 $l2 $(cat $dir/avg_ppl | head -n2 | tail -n1) >> $GRID_RESULTS
done
