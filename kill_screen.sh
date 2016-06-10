#!/bin/bash
#count=$(screen -ls | awk '/Sock/{print $1}')
#for sess in `seq 1 $count`; do
if [ "x$1" != "x" ]; then
    sess="-S $1"
else
    sess=""
fi
echo $sess
for i in `seq 0 8`; do
    #-S $sess 
    screen $sess -X -p $i stuff $"^C\n"
done
for i in `seq 0 8`; do
    screen $sess -X -p $i stuff $"echo exit\n"
done
#done
if [ "x$TERM" != "x" ]; then
    screen -r
fi
