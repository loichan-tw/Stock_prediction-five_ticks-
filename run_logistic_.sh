#!/bin/bash

for s in "2330" "3008"
do 
    for t in "_3_average" "_5_average" "_3_one" "_5_one"
    do 
#        sudo CUDA_VISIBLE_DEVICES=1 python3 logistic_regression.py ${s} ${t}
	sudo CUDA_VISIBLE_DEVICES=1 python3 lstm_test.py ${s} ${t}
#	sudo CUDA_VISIBLE_DEVICES=1 python3 CRNN_test.py ${s} ${t}
    done
done

 
