#!/usr/bin/env bash

. ./model_files

cfg_home=`pwd`
gem5_dir=${ALADDIN_HOME}/../..
bmk_dir=`git rev-parse --show-toplevel`/../build/bin

${gem5_dir}/build/X86/gem5.opt \
  --debug-flags=Aladdin,HybridDatapath \
  --outdir=${cfg_home}/outputs \
  --stats-db-file=stats.db \
  ${gem5_dir}/configs/aladdin/aladdin_se.py \
  --num-cpus=2 \
  --mem-size=128MB \
  --mem-type=LPDDR3_1600_1x32  \
  --sys-clock=3GHz \
  --cpu-clock=3GHz \
  --cpu-type=DerivO3CPU \
  --ruby \
  --access-backing-store \
  --l2_size=64000 \
  --l2_assoc=16 \
  --cacheline_size=32 \
  --accel_cfg_file=gem5.cfg \
  --fast-forward=10000000000 \
  -c ${bmk_dir}/smaug \
  -o "${topo_file} ${params_file} --sample-level=high --gem5 --debug-level=0 --num-accels=1" \
  > stdout 2> stderr
