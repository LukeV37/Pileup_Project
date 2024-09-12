#!/bin/bash
dirlist=(/tmp/*.eps)
for ((i=0; i < ${#dirlist[@]}; i++)); do
  input=${dirlist[$i]}
  output=${input:5}
  output=${output/eps/pdf}
  echo "Processing $input"
  echo "Writing to diagrams/$output"
  epspdf $input ./diagrams/$output
  echo ""
done
