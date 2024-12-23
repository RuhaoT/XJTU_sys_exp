# compile dependencies for workloads
# m5ops, etc.

# path to gem5
gem5_path=/home/ruhaotian/XJTU_sys_exp/gem5

# current path
current_path=$(pwd)

# parameters
TARGET_ISA=riscv
TARGET_ISA_CROSS_COMPILER=riscv64-unknown-linux-gnu-g++

cd $gem5_path/util/m5

# compile m5ops
scons [{$TARGET_ISA}.CROSS_COMPILE={$TARGET_ISA_CROSS_COMPILER}] build/$TARGET_ISA/out/m5 -j8

# copy all dependencies to the workload library
cp build/$TARGET_ISA/out/* $current_path/lib/$TARGET_ISA

# copy the m5ops header file to the workload library
cd $gem5_path

cp -r ./include $current_path