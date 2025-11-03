#!/bin/bash

numfogs=${1}
shift

for (( i=0; i<$numfogs; i++ ))
do
   ./soswirlyfog fog$i.json > out$i.txt &
   echo $!
   sleep 1
done