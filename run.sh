#!/usr/bin/bash

module load compiler vtune

cd ver7
make
vtune -collect uarch-exploration -r ../vtune_lab5_ver7 ./nbody.x 5000 2000

cd ver8
make
vtune -collect uarch-exploration -r ../vtune_lab5_ver8 ./nbody.x 5000 2000
