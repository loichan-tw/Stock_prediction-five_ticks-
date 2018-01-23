#!/bin/bash

for c in "CRNN.py" "lstm.py" "logistic_regression.py"
do
    for s in "2330" "3008"
    do 
        for t in "_3_average" "_5_average" "_3_one" "_5_one"
        do 
            sudo CUDA_VISIBLE_DEVICES=1 python3 ${c} ${s} ${t}
        done
    done
done

 
