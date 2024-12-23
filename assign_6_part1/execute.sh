# execute a python configuration file with gem5 library
gem5_path=/home/ruhaotian/XJTU_sys_exp/gem5/build/RISCV/gem5.opt

# get all arguments
args=$@

# execute the python configuration file
$gem5_path $args