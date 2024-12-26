"""This module is the step 1 of Lab 6. 

It creates a hybrid CPU with 3-level coherent cache hierarchy.
This lab intends to explore how cache configurations affect the performance of a hybrid CPU.
"""

import dataclasses
import pandas as pd
import subprocess
import os
import tqdm

from utils import parameterization, step1_dataclass


def generate_metaparam_combinations(
    meta_params: step1_dataclass.ExperimentMetaParameter, save_as: str
) -> int:
    """Generate all possible cache configurations based on the meta parameters."""

    # generate experiment cache configuration based on the meta parameter
    meta_params_list = parameterization.recursive_iterate_dataclass(meta_params)
    meta_params_dicts = []
    for index, meta_params in enumerate(meta_params_list):
        meta_params.experiment_index = index
        meta_params_dicts.append(dataclasses.asdict(meta_params))

    # convert the list of dataclasses to a pandas dataframe
    meta_params_df = pd.DataFrame(meta_params_dicts)
    # save the dataframe to a csv file
    meta_params_df.to_csv(save_as, index=False)

    return len(meta_params_list)


if __name__ == "__main__":
    # step 1: experiment preparation
    EXP_NAME = "step2_cpu_only"
    META_PARAM_FILE_NAME = "step1_experiment_metaparams.csv"
    GEM5_RAW_FOLDER_NAME = "gem5_raw_output"
    DATA_FILE_NAME = "step1_experiment_data.csv"
    DATA_FILE_COLUMNS = [
        "experiment_index",
        "simSeconds",
        "simInsts",
        "avg_big_l1i_missrate",
        "avg_big_l1d_missrate",
        "avg_big_l2_missrate",
        "avg_little_l1i_missrate",
        "avg_little_l1d_missrate",
        "avg_little_l2_missrate",
        "avg_l3_missrate",
    ]

    CURR_DIR_ABS_PATH = os.path.dirname(os.path.abspath(__file__))
    EXECUTOR_REL_PATH = "step2_experiment_executor.py"
    WORKLOAD_REL_PATH = "workload/matmul/mm-ijk-gem5"
    RESULT_FOLDER_REL_PATH = "results"
    GEM5_ABS_PATH = "/home/ruhaotian/XJTU_sys_exp/gem5/build/RISCV/gem5.opt"

    # create the result directory
    result_dir = os.path.join(CURR_DIR_ABS_PATH, RESULT_FOLDER_REL_PATH, EXP_NAME)
    # remove the result directory if it already exists
    if os.path.exists(result_dir):
        subprocess.run(["rm", "-rf", result_dir])
    os.makedirs(result_dir)
    os.makedirs(os.path.join(result_dir, GEM5_RAW_FOLDER_NAME))
    # create the data file and header
    data_file = os.path.join(result_dir, DATA_FILE_NAME)
    with open(data_file, "w") as f:
        f.write(",".join(DATA_FILE_COLUMNS) + "\n")
    print("Step 1: experiment preparation done.")

    # step 2: parameterization
    meta_params = step1_dataclass.ExperimentMetaParameter(
        replacement_policy=["LRURP"],
        prefetcher_type=["Stride"],
        l1_cache_sample_seed=[1],
        l2_cache_sample_seed=[4],
        l3_cache_sample_seed=[4],
        big_core_width=10,
        big_core_rob_size=40,
        big_core_num_int_regs=50,
        big_core_num_fp_regs=50,
        small_core_width=2,
        small_core_rob_size=30,
        small_core_num_int_regs=40,
        small_core_num_fp_regs=40,
        big_core_num=1,
        small_core_num=1,
        matsize=[256],
    )
    # save the metaparam combinations to a csv file
    meta_file = os.path.join(result_dir, META_PARAM_FILE_NAME)
    experiment_num = generate_metaparam_combinations(meta_params, meta_file)
    print("Step 2: metaparam combinations generated and saved.")

    # step 3: experiment execution
    # create a progress bar
    progress_bar = tqdm.tqdm(total=experiment_num)
    for index in range(experiment_num):
        # make index dir
        index_dir = os.path.join(result_dir, GEM5_RAW_FOLDER_NAME, str(index))
        os.makedirs(index_dir)
        redirect_command = "--outdir=" + index_dir
        subprocess.run(
            [
                GEM5_ABS_PATH,
                redirect_command,
                os.path.join(CURR_DIR_ABS_PATH, EXECUTOR_REL_PATH),
                f"--param_file={meta_file}",
                f"--target_index={index}",
                f"--workload={os.path.join(CURR_DIR_ABS_PATH, WORKLOAD_REL_PATH)}",
            ]
        )

        # summarize results
        # load stats file
        stats_file = os.path.join(index_dir, "stats.txt")
        # only keep the stats after "End Simulation Statistics"
        with open(stats_file, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "End Simulation Statistics" in line:
                    break
            lines = lines[i:]

        # read and categorize results
        simSeconds = 0.0
        simInsts = 0
        big_l1i_missrate = []
        big_l1d_missrate = []
        big_l2_missrate = []
        little_l1i_missrate = []
        little_l1d_missrate = []
        little_l2_missrate = []
        l3_missrate = []
        for line in lines:
            curr_cluster = None
            # find simSeconds
            # example line: simSeconds                                   0.002261
            if "simSeconds" in line:
                simSeconds = float(line.split()[1])
            # find simInsts
            # example line: simInsts                                      8018609
            if "simInsts" in line:
                simInsts = int(line.split()[1])
            # find cluster line
            # example line: board.cache_hierarchy.clusters0.dptw_cache.tags.sampledRefs
            if "clusters" in line:
                # big core cluster num: 0-metaparams.big_core_num-1
                # little core cluster num: metaparams.big_core_num-metaparams.big_core_num+metaparams.small_core_num-1
                line_split = line.split(".")
                # find the chunk that contains the 'clusters' keyword
                for i, chunk in enumerate(line_split):
                    if "clusters" in chunk:
                        break
                    
                try:
                    cluster_num = int(line_split[i][8])
                except ValueError:
                    raise ValueError("Cluster number is not an integer. Full line: " + line)
                    exit(1)
                
                if cluster_num < meta_params.big_core_num:
                    curr_cluster = "big"
                else:
                    curr_cluster = "little"

            # find missrate line
            # find line with 'overallMissRate::total'
            # example line: board.cache_hierarchy.l3_cache.overallMissRate::total     0.096511
            if "overallMissRate::total" in line:
                # extract cache type: l1icache, l1dcache, l2cache, l3_cache
                # first split by ' ', get the first part, then split by '.', get the second last part
                cache_type = line.split()[0].split(".")[-2]
                if cache_type == "l1icache":
                    if curr_cluster == "big":
                        big_l1i_missrate.append(float(line.split()[1]))
                    else:
                        little_l1i_missrate.append(float(line.split()[1]))
                elif cache_type == "l1dcache":
                    if curr_cluster == "big":
                        big_l1d_missrate.append(float(line.split()[1]))
                    else:
                        little_l1d_missrate.append(float(line.split()[1]))
                elif cache_type == "l2cache":
                    if curr_cluster == "big":
                        big_l2_missrate.append(float(line.split()[1]))
                    else:
                        little_l2_missrate.append(float(line.split()[1]))
                elif cache_type == "l3_cache":
                    l3_missrate.append(float(line.split()[1]))
                else:
                    raise ValueError("Unknown cache type. Full line: " + line)

        # calculate average missrate
        if len(big_l1i_missrate) == 0:
            avg_big_l1i_missrate = 0.0
        else: 
            avg_big_l1i_missrate = sum(big_l1i_missrate) / len(big_l1i_missrate)
        if len(big_l1d_missrate) == 0:
            avg_big_l1d_missrate = 0.0
        else:
            avg_big_l1d_missrate = sum(big_l1d_missrate) / len(big_l1d_missrate)
        if len(big_l2_missrate) == 0:
            avg_big_l2_missrate = 0.0
        else:
            avg_big_l2_missrate = sum(big_l2_missrate) / len(big_l2_missrate)
        if len(little_l1i_missrate) == 0:
            avg_little_l1i_missrate = 0.0
        else:
            avg_little_l1i_missrate = sum(little_l1i_missrate) / len(little_l1i_missrate)
        if len(little_l1d_missrate) == 0:
            avg_little_l1d_missrate = 0.0
        else:
            avg_little_l1d_missrate = sum(little_l1d_missrate) / len(little_l1d_missrate)
        if len(little_l2_missrate) == 0:
            avg_little_l2_missrate = 0.0
        else:
            avg_little_l2_missrate = sum(little_l2_missrate) / len(little_l2_missrate)
        if len(l3_missrate) == 0:
            avg_l3_missrate = 0.0
        else:
            avg_l3_missrate = sum(l3_missrate) / len(l3_missrate)

        # write results to the data file
        with open(data_file, "a") as f:
            f.write(
                f"{index},{simSeconds},{simInsts},{avg_big_l1i_missrate},{avg_big_l1d_missrate},{avg_big_l2_missrate},{avg_little_l1i_missrate},{avg_little_l1d_missrate},{avg_little_l2_missrate},{avg_l3_missrate}\n"
            )

        # update progress bar
        progress_bar.update(1)
