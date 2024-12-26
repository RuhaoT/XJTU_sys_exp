#!/usr/bin/env python3

import re
import csv
import sys

HEADER_LINES = [
    "===============================\n",
    "        Aladdin Results        \n",
    "===============================\n"
]

def is_aladdin_header(lines, idx):
    """
    Checks if lines[idx], lines[idx+1], lines[idx+2] match the known 3-line header block.
    Returns True if yes, otherwise False.
    Be mindful of indexing, so call only if idx+2 < len(lines).
    """
    return (
        lines[idx]   == HEADER_LINES[0] and
        lines[idx+1] == HEADER_LINES[1] and
        lines[idx+2] == HEADER_LINES[2]
    )

def parse_result_block(block_lines):
    """
    Given all lines from a single Aladdin Results block (between a valid open/close pair),
    parse out the information of interest. Return a dictionary of the extracted fields.
    Adjust the regex or parsing logic as needed.
    """

    result = {
        "Running"                  : None,
        "Top level function"       : None,
        "Cycle"                    : None,
        "Upsampled Cycle"          : None,
        "Avg Power (mW)"           : None,
        "Idle FU Cycles"           : None,
        "Avg FU Power (mW)"        : None,
        "Avg FU Dynamic Power (mW)": None,
        "Avg FU Leakage Power (mW)": None,
        "Avg MEM Power (mW)"       : None,
        "Avg MEM Dynamic Power (mW)": None,
        "Avg MEM Leakage Power (mW)": None,
        "Total Area (uM^2)"        : None,
        "FU Area (uM^2)"           : None,
        "MEM Area (uM^2)"          : None,
        "Num Multipliers"          : None,
        "Num Adders"               : None,
        "Num Bit-wise Operators"   : None,
        "Num Shifters"             : None,
        "Num Registers"            : None
    }

    # Some example regex patterns to capture fields
    # Adjust them as needed for your output format
    regex_patterns = {
        "Running"                   : r"^Running\s*:\s*(.*)$",
        "Top level function"        : r"^Top level function:\s*(.*)$",
        "Cycle"                     : r"^Cycle\s*:\s*([\d.]+)\s+cycles",
        "Upsampled Cycle"           : r"^Upsampled Cycle\s*:\s*([\d.]+)\s+cycles",
        "Avg Power (mW)"            : r"^Avg Power:\s*([\d.]+)\s+mW",
        "Idle FU Cycles"            : r"^Idle FU Cycles:\s*([\d.]+)\s+cycles",
        "Avg FU Power (mW)"         : r"^Avg FU Power:\s*([\d.]+)\s+mW",
        "Avg FU Dynamic Power (mW)" : r"^Avg FU Dynamic Power:\s*([\d.]+)\s+mW",
        "Avg FU Leakage Power (mW)" : r"^Avg FU leakage Power:\s*([\d.]+)\s+mW",
        "Avg MEM Power (mW)"        : r"^Avg MEM Power:\s*([\d.]+)\s+mW",
        "Avg MEM Dynamic Power (mW)": r"^Avg MEM Dynamic Power:\s*([\d.]+)\s+mW",
        "Avg MEM Leakage Power (mW)": r"^Avg MEM Leakage Power:\s*([\d.]+)\s+mW",
        "Total Area (uM^2)"         : r"^Total Area:\s*([\d.eE\+]+)\s+uM\^2",
        "FU Area (uM^2)"            : r"^FU Area:\s*([\d.eE\+]+)\s+uM\^2",
        "MEM Area (uM^2)"           : r"^MEM Area:\s*([\d.eE\+]+)\s+uM\^2",
        "Num Multipliers"           : r"^Num of Multipliers \(32-bit\):\s*(\d+)",
        "Num Adders"                : r"^Num of Adders \(32-bit\):\s*(\d+)",
        "Num Bit-wise Operators"    : r"^Num of Bit-wise Operators \(32-bit\):\s*(\d+)",
        "Num Shifters"              : r"^Num of Shifters \(32-bit\):\s*(\d+)",
        "Num Registers"             : r"^Num of Registers \(32-bit\):\s*(\d+)",
    }

    for line in block_lines:
        for key, pattern in regex_patterns.items():
            match = re.match(pattern, line.strip())
            if match:
                result[key] = match.group(1)
                break
    
    return result

def main(stdout_filename, csv_filename="aladdin_results.csv"):
    with open(stdout_filename, "r") as f:
        lines = f.readlines()
    
    # We'll collect dictionaries of results here
    results = []

    # State machine: we look for pairs of the 3-line header blocks.
    i = 0
    n = len(lines)
    block_count = 0

    # We toggle "collecting" on/off every time we detect a header.
    # If we see the 1st header, we start collecting. Next time we see
    # the header, we stop collecting (that's one block). Next time, we start again.
    collecting = False
    current_block_lines = []

    while i + 2 < n:
        if is_aladdin_header(lines, i):
            # We found a header
            if not collecting:
                # If we were NOT collecting, now we start
                collecting = True
                current_block_lines = []
            else:
                # If we WERE collecting, then the block ends here
                # We'll parse the block, store results, and turn collecting off
                block_data = parse_result_block(current_block_lines)
                results.append(block_data)
                collecting = False
            
            # Skip past the 3-line header
            i += 3
        else:
            # If we are collecting, add line to current block
            if collecting:
                current_block_lines.append(lines[i])
            i += 1

    # Edge case: if the file ends while still collecting, we ignore that incomplete block
    # But if your log always closes with a valid header, you can handle differently.

    # Write out CSV
    # Gather all possible keys from the first block (or use a known stable list)
    if not results:
        print("No Aladdin results sections found.")
        return

    # Derive fieldnames from the dictionary keys (or define them manually)
    fieldnames = list(results[0].keys())

    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Extracted {len(results)} Aladdin result sections into '{csv_filename}'.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <stdout_file> [output_csv_file]")
        sys.exit(1)
    
    stdout_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_csv = sys.argv[2]
    else:
        output_csv = "aladdin_results.csv"

    main(stdout_file, output_csv)
