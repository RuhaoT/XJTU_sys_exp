#!/usr/bin/env python
#
# Run an nnet_fwd simulation on Condor.
#
# This uses the runscripts that the user prepares, and all simulation output is
# placed into those directories.
#
# This script will also search for condor_req files that the user places within
# directories. condor_req files contain specific condor requirements for all
# jobs in that directory and all its subdirectories, unless there is another
# condor_req file lower in the directory tree that matches to a particular
# runscript better. This script will add those requirements to the header of the
# condor job file.

import argparse
import ConfigParser
import fnmatch
import getpass
import re
import os
import sys
import socket

ACC_MACHINES = ["acc41", "acc29"]
RB_MACHINES = ["%s.int.seas.harvard.edu" % host for host in
               ["rb22", "rb23", "rb24", "rb25", "rb26", "rb27"]]
ACC_HEADER = "# ACC CONDOR HEADER"
RB_HEADER = "# RB CONDOR HEADER"

def check_path_exists(path):
  if not os.path.exists(path):
    print "%s does not exist!" % path
    sys.exit(1)

def read_condor_req(req_file):
  if not req_file:
    # Be super conservative and request the largest amount of memory we've
    # observed our jobs to take up.
    return "request_memory = 48g\n"

  contents = []
  with open(req_file, "r") as f:
    for line in f:
      contents.append(line.strip())

  return "\n\n".join(contents) + "\n"

def get_rb_script_header():
  header = (
      "%s\n"
      "Universe = vanilla\n"
      "GetEnv = True\n"
      "Requirements = (OpSys == \"LINUX\") && (Arch == \"X86_64\") && ("
      "TotalMemory > 128000)\n"
      "Notification = Complete\n"
      "notify_user = %s@seas.harvard.edu\n"
      "Executable = /bin/bash\n" % (RB_HEADER, getpass.getuser()))
  return header

def get_acc_script_header():
  header = (
      "%s\n"
      "Universe = vanilla\n"
      "GetEnv = True\n"
      "Requirements = (OpSys == \"LINUX\") && (Arch == \"X86_64\") && ("
      "TotalMemory > 128000) && (!HasGpu)\n"
      "Notification = Complete\n"
      "notify_user = %s@seas.harvard.edu\n"
      "Executable = /bin/bash\n" % (ACC_HEADER, getpass.getuser()))
  return header

def detect_condor_script_host_type(condor_script_fname):
  with open(condor_script_fname, "r") as f:
    for line in f:
      line = line.strip()
      if line == ACC_HEADER:
        return "acc"
      elif line == RB_HEADER:
        return "rb"
  return "Unknown"

def condor_script_matches_target(condor_script_fname, curr_hostname):
  host_type = detect_condor_script_host_type(condor_script_fname)
  if host_type == "acc" and curr_hostname in ACC_MACHINES:
    return True
  if host_type == "rb" and curr_hostname in RB_MACHINES:
    return True
  return False

def write_condor_script(sim_dir, condor_script_fname, condor_req):
  hostname = socket.gethostname()
  if hostname in ACC_MACHINES:
    header_template = get_acc_script_header()
  else:
    header_template = get_rb_script_header()

  # Attach a basic description to the arguments. This description does nothing,
  # but it helps us know what is still running when we run condor_q.
  split = sim_dir.split(os.path.sep)
  desc = "%s-%s-%s-%s" % (split[-4], split[-3], split[-2], split[-1])

  # The initial directory will be the sim_dir, and all subsequent paths will be
  # relative to that.
  job_template = ("InitialDir = %s\n"
                  "Arguments = run.sh %s\n"
                  "Log = log\n"
                  "Queue\n") % (os.path.abspath(sim_dir), desc)

  file_exists = os.path.exists(condor_script_fname)
  if file_exists and not condor_script_matches_target(
      condor_script_fname, hostname):
    print ("Current condor script was generated on a different host type "
           "than the current one! It will be overwritten.")
    file_exists = False
    open_mode = "w"
  else:
    open_mode = "a"

  with open(condor_script_fname, open_mode) as f:
    if not file_exists:
      f.write(header_template)

    reqs = read_condor_req(condor_req)
    f.write(reqs)
    f.write(job_template)

def parse_run_config(config, sim_dir):
  """ Return a dict of all files that need symlinking or copying. """
  orig = {"gem5": "", "outdir": "", "binary-args": "", "gem5-cfg-file": ""}
  symlinks = {}
  copies = {"binary-build": ""}
  environment_values = {"SIMDIR": sim_dir, "TOT": top_of_tree}

  with open(config, "r") as f:
    for line in f:
      split = line.split(":")
      assert(len(split) == 2 and
             "Invalid syntax: Cannot have more than one colon, in \"%s\"" %
             line)
      param_name = split[0].strip()
      param_value = split[1].strip()
      for envkey, envval in environment_values.iteritems():
        param_value = param_value.replace("$%s" % envkey, envval)

      if param_name in orig:
        orig[param_name] = param_value
      if param_name in symlinks:
        symlinks[param_name] = param_value
      if param_name in copies:
        copies[param_name] = param_value

  orig["gem5"] = os.path.join(os.environ["ALADDIN_HOME"], orig["gem5"])

  # Check for existence of all the paths.
  check_path_exists(orig["outdir"])
  check_path_exists(orig["gem5-cfg-file"])
  check_path_exists(copies["binary-build"])

  return orig, symlinks, copies

def find_condor_reqs_for_sim_dir(sim_dir, condor_reqs):
  """ Find the condor_req file that best matches this sim_dir. """
  best_match = ""
  best_dirmatches = 0

  sim_dirpath = os.path.normpath(os.path.dirname(sim_dir)).split(os.sep)
  for req in condor_reqs:
    curr_match = 0
    req_dirpath = os.path.normpath(os.path.dirname(req)).split(os.sep)
    for i, d in enumerate(req_dirpath):
      if i < len(sim_dirpath) and d == sim_dirpath[i]:
        curr_match += 1
        if curr_match > best_dirmatches:
          best_dirmatches = curr_match
          best_match = req
      else:
        break

  return best_match

def find_sim_dirs(base_dir, filt=None):
  sim_dirs = []
  condor_reqs = []
  for root, dirs, files in os.walk(base_dir):
    for item in fnmatch.filter(files, "run.sh"):
      if filt == None or re.search(filt, root):
        sim_dirs.append(root)

  # Find all condor_req files directly in the path of this directory.
  curr_dir = base_dir
  while curr_dir:
    if "condor_req" in os.listdir(curr_dir):
      condor_reqs.append(os.path.join(curr_dir, "condor_req"))
    curr_dir = os.path.dirname(curr_dir)

  return sim_dirs, condor_reqs

def check_sim_dir(sim_dir):
  files = os.listdir(sim_dir)
  has_gem5_cfg = "gem5.cfg" in files
  if not has_gem5_cfg:
    print "Could not find gem5.cfg in this simulation directory!"
    return False

  parser = ConfigParser.SafeConfigParser()
  parser.read(os.path.join(sim_dir, "gem5.cfg"))
  if not parser.sections():
    print "gem5.cfg contained no accelerator configuration!"
    return False
  sections = parser.sections()
  if "DEFAULT" in sections:
    sections.remove("DEFAULT")
  assert(len(sections) >= 1)

  for i in range(len(sections)):
    accel = sections[i]
    try:
      dynamic_trace_fname = parser.get(accel, "trace_file_name")
      if not os.path.exists(os.path.join(sim_dir, dynamic_trace_fname)):
        print sim_dir
        print ("Dynamic trace %s (specified in gem5.cfg), for accelerator %s, "
               "does not exist!" % (dynamic_trace_fname, accel))
        return False
    except ConfigParser.NoOptionError:
     pass

    try:
      aladdin_cfg_fname = parser.get(accel, "config_file_name")
      if not os.path.exists(os.path.join(sim_dir, aladdin_cfg_fname)):
        print sim_dir
        print ("Aladdin config file %s (specified in gem5.cfg), for accelerator "
               "%s, does not exist!" % (aladdin_cfg_fname, accel))
        return False
    except ConfigParser.NoOptionError:
      pass

  if not os.path.exists(os.path.join(sim_dir, "run.sh")):
    print "Runscript does not exist!"
    return False

  return True

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("sim_dir")
  parser.add_argument("--filter",
      help="Only parse directories with this string in its path")
  args = parser.parse_args()

  sim_dirs, condor_reqs = find_sim_dirs(args.sim_dir, args.filter)
  for sim_dir in sorted(sim_dirs):
    is_valid_dir = check_sim_dir(sim_dir)
    if not is_valid_dir:
      continue
    condor_req = find_condor_reqs_for_sim_dir(sim_dir, condor_reqs)
    write_condor_script(sim_dir, "condor.con", condor_req)

if __name__ == "__main__":
  main()
