#!/usr/bin/env bash

source ./model_files

bmk_dir=`git rev-parse --show-toplevel`/../build/bin

${bmk_dir}/smaug-instrumented \
  ${topo_file} ${params_file} --sample-level=high --debug-level=0 --num-accels=1
