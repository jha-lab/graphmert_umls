#!/bin/bash

# 1st parameter - folder with ftp list;
file="/scratch/gpfs/JHA/mb5157/large_data/diabetes_2025_6years/non-eng_and_bad_pmcids.txt"

while IFS= read -r pmcid; do
    filename="/scratch/gpfs/JHA/mb5157/large_data/diabetes_2025_6years/papers/$pmcid"
    rm -r $filename || echo "couldnt remove $pmcid"
done <"$file"
