#!/usr/bin/env bash

. ./model_files

cfg_home=`pwd`
gem5_dir=${ALADDIN_HOME}/../..
bmk_dir=`git rev-parse --show-toplevel`/../build/bin

${gem5_dir}/build/X86/gem5.opt \
  --debug-flags=Aladdin,HybridDatapath \
  --outdir=${cfg_home}/outputs_cpu \
  --stats-db-file=stats.db \
  ${gem5_dir}/configs/example/se.py \
  --num-cpus=1 \
  --mem-size=4GB \
  --mem-type=LPDDR4_3200_2x16  \
  --sys-clock=1.25GHz \
  --cpu-clock=2.5GHz \
  --cpu-type=DerivO3CPU \
  --ruby \
  --access-backing-store \
  --l2_size=2097152 \
  --l2_assoc=16 \
  --cacheline_size=32 \
  --fast-forward=10000000000 \
  -c ${bmk_dir}/smaug \
  -o "${topo_file} ${params_file} --sample-level=high --gem5 --debug-level=0" \
  > stdout_cpu 2> stderr_cpu
