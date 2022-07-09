#!/bin/bash

## Run the param search serially
#lrs=(0.01 0.001 0.0001 0.0001 0.00001)
#devices=(3 3 3 3 3)
#num=${#lrs[*]}
#for i in $(seq 1 "$num"); do
#  lr=${lrs[i-1]}
#  device=${devices[i-1]}
#  echo "Running lr=${lr} on device ${device}"
#  PYTHONPATH=~/CogKTR python test_param_search.py --device ${device} --batch_size 64 --lr ${lr}
#done

# Run the param search parallely
pids=()
function trap_ctrlc ()
{
    echo "Ctrl-C caught...performing clean up"
    num=${#pids[*]}
    for i in $(seq 1 "$num"); do
      kill ${pids[${i-1}]}
      lr=${lrs[i-1]}
      device=${devices[i-1]}
      echo "Kill process ${pids[${i-1}]} lr=${lr} on device ${device}!"
    done
    exit
}
trap "trap_ctrlc" 2

lrs=(0.000001 0.000003 0.000005 0.00001 0.00003 0.00005 0.0001)
devices=(2 3 4 5 8 9 1)
num=${#lrs[*]}
export PYTHONPATH=~/CogKTR
for i in $(seq 1 "$num"); do
  lr=${lrs[i-1]}
  device=${devices[i-1]}
  nohup python test_param_search.py --device ${device} --batch_size 16 --lr ${lr} > /dev/null 2>&1 &
  pids[${i-1}]=$!
  echo "Running lr=${lr} on device ${device} in process ${pids[${i-1}]}.."
done

for i in $(seq 1 "$num"); do
  wait ${pids[${i-1}]}
  lr=${lrs[i-1]}
  device=${devices[i-1]}
  echo "Finished process ${pids[${i-1}]} lr=${lr} on device ${device}!"
done

echo "All Process Done"