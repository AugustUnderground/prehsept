#!/bin/sh

PDK="xh035 xh018 xt18 gpdk180 gkd090 gpdk045"
DEV="nmos pmos"

for pdk in $(echo "$PDK"); do
    for dev in $(echo "$DEV"); do
        stack exec -- prehsept-exe --pdk $pdk --dev $dev --dir "./data/$pdk-$dev.pt" --size 512 --num 24
    done
done
