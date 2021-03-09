#!/usr/bin/env bash

# Train Motifnet using different orderings
export PYTHONPATH=$HOME/workspaces/unbiased-sg
export PYTHONIOENCODING=utf-8


for d in ./checkpoints/*; do

    echo $d
    for f in $d/*.pth; do
        #echo $f
        if echo $f | grep '0002000'; then
            rm $f

        elif echo $f | grep '0004000'; then
            rm $f

        elif echo $f | grep '0006000'; then
            rm $f

        elif echo $f | grep '0008000'; then
            rm $f

        elif echo $f | grep '0010000'; then
            rm $f

        elif echo $f | grep '0012000'; then
            rm $f

        elif echo $f | grep '0014000'; then
            rm $f

        elif echo $f | grep '0016000'; then
            rm $f
        fi
    done

done

