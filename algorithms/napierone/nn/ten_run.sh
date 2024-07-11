#!/bin/bash
for i in {1..3}
do
    python neural_network.py
    echo "Executed:" $i >> runs.txt
done


