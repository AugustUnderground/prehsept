#!/bin/sh

#PDK="xh018 xt018 gpdk180 gpdk090 gpdk045"
PDK="xt018"
DEV="nmos pmos"

for pdk in $(echo "$PDK"); do
    for dev in $(echo "$DEV"); do
        stack exec -- prehsept-exe --pdk $pdk --dev $dev --dir "./data/$pdk-$dev.pt" --size 32 --num 64 --exp
    done
done
