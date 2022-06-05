#!/bin/bash
for i in {1..188}
do
   echo "====Run $i===="
   python main.py random_test --load_checkpoint "CTGR-$i"
done