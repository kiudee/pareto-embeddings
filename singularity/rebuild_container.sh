#!/bin/bash

rm -rf sandbox
rm -rf pareto.sif
singularity build -s ./sandbox ./pareto.def
singularity build ./pareto.sif ./sandbox/
