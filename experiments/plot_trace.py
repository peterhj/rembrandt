#!/usr/bin/env python3

from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from ast import literal_eval
import math
import sys

def main():
  cfg_path = sys.argv[1]
  with open(cfg_path, "r") as f:
    cfg = literal_eval(f.read())
  print(cfg)

  out_prefix = cfg["out_prefix"]
  num_procs = cfg["num_procs"]
  num_traces = len(cfg["traces"])

  x_upper = cfg["x_upper"]
  show_legend = cfg["show_legend"]

  #rc("text", usetex=True)

  trace_colors = []
  trace_labels = []
  trace_dfs = []
  step_dfs = []
  valid_dfs = []

  for trace_idx in range(num_traces):
    trace = cfg["traces"][trace_idx]
    trace_color = trace[0]
    trace_label = trace[1]
    trace_prefix = trace[2]
    trace_path = "{}/trace_sgd.0.log".format(trace_prefix)

    trace_df = pd.read_csv(trace_path, index_col="t")
    #trace_df = pd.read_csv(trace_path)
    #print(trace_df)

    step_df = trace_df[trace_df["event"] == "step"]
    #print(step_df)

    step_df["tstamps"] = np.cumsum(step_df["elapsed"])
    #print(step_df)

    valid_df = trace_df[trace_df["event"] == "valid"]
    #print(valid_df.index)

    valid_df["tstamps"] = step_df["tstamps"].loc[valid_df.index] / 3600.0
    #print(valid_df)

    trace_colors.append(trace_color)
    trace_labels.append(trace_label)
    trace_dfs.append(trace_df)
    step_dfs.append(step_df)
    valid_dfs.append(valid_df)

  plots = []

  plt.figure()
  ax = plt.gca()

  for trace_color, trace_label, trace_df, step_df, valid_df in zip(trace_colors, trace_labels, trace_dfs, step_dfs, valid_dfs):
    if "acc" in valid_df.columns:
      valid_df["error"] = 1.0 - valid_df["acc"]
    plot_h, = ax.plot(valid_df["tstamps"], valid_df["error"], color=trace_color, label=trace_label)
    plots.append(plot_h)

  plt.title("p = {}".format(num_procs))
  plt.xlabel("training time (hours)")
  plt.xlim([0.0, x_upper])
  plt.ylabel("validation error")
  plt.ylim([0.0, 1.0])
  ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0%}".format(y)))
  if show_legend:
    plt.legend(handles=plots)
  plt.show()
  plt.savefig("{}_error.pdf".format(out_prefix))

  plots = []

  plt.figure()
  ax = plt.gca()

  for trace_color, trace_label, trace_df, step_df, valid_df in zip(trace_colors, trace_labels, trace_dfs, step_dfs, valid_dfs):
    plot_h, = ax.plot(valid_df["tstamps"], valid_df["loss"], color=trace_color, label=trace_label)
    plots.append(plot_h)

  plt.title("p = {}".format(num_procs))
  plt.xlabel("training time (hours)")
  plt.xlim([0.0, x_upper])
  plt.ylabel("validation loss")
  plt.ylim([0.0, math.log(1000.0)])
  if show_legend:
    plt.legend(handles=plots)
  plt.show()
  plt.savefig("{}_loss.pdf".format(out_prefix))

if __name__ == "__main__":
  main()
