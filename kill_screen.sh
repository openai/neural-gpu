#!/bin/bash
#count=$(screen -ls | awk '/Sock/{print $1}')
#for sess in `seq 1 $count`; do
for i in `seq 0 8`; do
    #-S $sess 
    screen -X -p $i stuff $"^C\n"
done
for i in `seq 0 8`; do
    screen -X -p $i stuff $"echo exit\n"
done
#done
if [ "x$TERM" != "x" ]; then
    screen -r
fi
