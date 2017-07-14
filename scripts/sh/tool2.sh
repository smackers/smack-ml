#!/bin/sh

file=$1 ; ../../../sloopy/llvm-build/bin/sloopy -ml-format $file -- && \
  echo -n $file >> $2 && \
  ../../../sloopy/llvm-build/bin/sloopy -ml $file -- >> $2
  printf "\n" >> $2
