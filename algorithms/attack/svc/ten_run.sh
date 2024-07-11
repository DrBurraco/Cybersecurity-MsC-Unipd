#!/bin/bash
for i in {1..3}
do
    python svc.py
    echo "Executed:" $i >> runs.txt
done


