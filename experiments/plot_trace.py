#!/usr/bin/env python3

from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from ast import literal_eval
import math
import sys

FONT_SIZE = 28
TICK_FONT_SIZE = 28
LEGEND_FONT_SIZE = 22

def _make_error_plot(fig, ax, plots, num_procs, x_label, x_range, y_range, show_legend, out_prefix, out_suffix):
  fig.subplots_adjust(bottom=0.15, left=0.20)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  plt.title("p = {}".format(num_procs), fontsize=FONT_SIZE)
  plt.xlabel(x_label, fontsize=FONT_SIZE)
  plt.xlim(x_range[:2])
  if len(x_range) == 2:
    pass
  elif len(x_range) == 3:
    #ax.yaxis.set_ticks(np.arange(x_range[0], x_range[1], x_range[2]))
    ax.xaxis.set_ticks(np.linspace(x_range[0], x_range[1], num=int(round((x_range[1] - x_range[0]) / x_range[2]))+1, endpoint=True))
  else:
    raise NotImplementedError
  plt.setp(ax.get_xticklabels(), fontsize=TICK_FONT_SIZE)
  ax.xaxis.set_ticks_position("bottom")
  ax.axhline(linewidth=2.0, color="k")
  plt.ylabel("validation error", fontsize=FONT_SIZE)
  plt.ylim(y_range[:2])
  if len(y_range) == 2:
    pass
  elif len(y_range) == 3:
    #ax.yaxis.set_ticks(np.arange(y_range[0], y_range[1], y_range[2]))
    ax.yaxis.set_ticks(np.linspace(y_range[0], y_range[1], num=int(round((y_range[1] - y_range[0]) / y_range[2]))+1, endpoint=True))
  else:
    raise NotImplementedError
  plt.setp(ax.get_yticklabels(), fontsize=TICK_FONT_SIZE)
  ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0%}".format(y)))
  ax.yaxis.set_ticks_position("left")
  ax.axvline(linewidth=2.0, color="k")
  if show_legend:
    plt.legend(handles=plots, fontsize=LEGEND_FONT_SIZE)
  plt.show()
  plt.savefig("{}_{}.pdf".format(out_prefix, out_suffix))

def main():
  cfg_path = sys.argv[1]
  with open(cfg_path, "r") as f:
    cfg = literal_eval(f.read())
  print(cfg)

  out_prefix = cfg["out_prefix"]
  num_procs = cfg["num_procs"]
  minibatch_sz = cfg["minibatch_sz"]
  num_traces = len(cfg["traces"])

  show_legend = cfg["show_legend"]
  x_upper = cfg["x_upper"]
  hours_range = cfg["hours_range"]
  epochs_range = cfg["epochs_range"]
  error_range = cfg["error_range"]

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

    valid_df["epochs"] = valid_df.index.map(lambda i: float(i) / (1281167.0 / (float(num_procs) * float(minibatch_sz))))

    trace_colors.append(trace_color)
    trace_labels.append(trace_label)
    trace_dfs.append(trace_df)
    step_dfs.append(step_df)
    valid_dfs.append(valid_df)

  plots = []

  fig = plt.figure()
  ax = plt.gca()

  for trace_color, trace_label, trace_df, step_df, valid_df in zip(trace_colors, trace_labels, trace_dfs, step_dfs, valid_dfs):
    if "acc" in valid_df.columns:
      valid_df["error"] = 1.0 - valid_df["acc"]
    plot_h, = ax.plot(valid_df["tstamps"], valid_df["error"], color=trace_color, label=trace_label, linewidth=2.0)
    plots.append(plot_h)

  if False:
    fig.subplots_adjust(bottom=0.15, left=0.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.title("p = {}".format(num_procs), fontsize=FONT_SIZE)
    plt.xlabel("training time (hours)", fontsize=FONT_SIZE)
    plt.xlim(hours_range)
    plt.setp(ax.get_xticklabels(), fontsize=TICK_FONT_SIZE)
    ax.xaxis.set_ticks_position("bottom")
    ax.axhline(linewidth=2.0, color="k")
    plt.ylabel("validation error", fontsize=FONT_SIZE)
    plt.ylim(error_range)
    plt.setp(ax.get_yticklabels(), fontsize=TICK_FONT_SIZE)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0%}".format(y)))
    ax.yaxis.set_ticks_position("left")
    ax.axvline(linewidth=2.0, color="k")
    if show_legend:
      plt.legend(handles=plots, fontsize=LEGEND_FONT_SIZE)
    plt.show()
    plt.savefig("{}_error.pdf".format(out_prefix))
  _make_error_plot(fig, ax, plots, num_procs, "training time (hours)", hours_range, error_range, show_legend, out_prefix, "error")

  plots = []

  fig = plt.figure()
  ax = plt.gca()

  for trace_color, trace_label, trace_df, step_df, valid_df in zip(trace_colors, trace_labels, trace_dfs, step_dfs, valid_dfs):
    plot_h, = ax.plot(valid_df["tstamps"], valid_df["loss"], color=trace_color, label=trace_label, linewidth=2.0)
    plots.append(plot_h)

  fig.subplots_adjust(bottom=0.15, left=0.20)
  plt.title("p = {}".format(num_procs), fontsize=FONT_SIZE)
  plt.xlabel("training time (hours)", fontsize=FONT_SIZE)
  plt.xlim([0.0, x_upper])
  plt.setp(ax.get_xticklabels(), fontsize=TICK_FONT_SIZE)
  plt.ylabel("validation loss", fontsize=FONT_SIZE)
  plt.ylim([0.0, math.log(1000.0)])
  plt.setp(ax.get_yticklabels(), fontsize=TICK_FONT_SIZE)
  if show_legend:
    plt.legend(handles=plots, fontsize=LEGEND_FONT_SIZE)
  plt.show()
  plt.savefig("{}_loss.pdf".format(out_prefix))

  plots = []

  fig = plt.figure()
  ax = plt.gca()

  for trace_color, trace_label, trace_df, step_df, valid_df in zip(trace_colors, trace_labels, trace_dfs, step_dfs, valid_dfs):
    plot_h, = ax.plot(valid_df.index, valid_df["error"], color=trace_color, label=trace_label, linewidth=2.0)
    plots.append(plot_h)

  #fig.tight_layout()
  fig.subplots_adjust(bottom=0.15, left=0.20)
  plt.title("p = {}".format(num_procs), fontsize=FONT_SIZE)
  plt.xlabel("iterations", fontsize=FONT_SIZE)
  #plt.xlim()
  plt.setp(ax.get_xticklabels(), fontsize=TICK_FONT_SIZE)
  plt.ylabel("validation error", fontsize=FONT_SIZE)
  plt.ylim([0.0, 1.0])
  plt.setp(ax.get_yticklabels(), fontsize=TICK_FONT_SIZE)
  ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0%}".format(y)))
  if show_legend:
    plt.legend(handles=plots, fontsize=LEGEND_FONT_SIZE)
  plt.show()
  plt.savefig("{}_iter.pdf".format(out_prefix))

  plots = []

  fig = plt.figure()
  ax = plt.gca()

  for trace_color, trace_label, trace_df, step_df, valid_df in zip(trace_colors, trace_labels, trace_dfs, step_dfs, valid_dfs):
    plot_h, = ax.plot(valid_df["epochs"], valid_df["error"], color=trace_color, label=trace_label, linewidth=2.0)
    plots.append(plot_h)

  if False:
    #fig.tight_layout()
    fig.subplots_adjust(bottom=0.15, left=0.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.title("p = {}".format(num_procs), fontsize=FONT_SIZE)
    plt.xlabel("training epochs", fontsize=FONT_SIZE)
    plt.xlim(epochs_range)
    plt.setp(ax.get_xticklabels(), fontsize=TICK_FONT_SIZE)
    ax.xaxis.set_ticks_position("bottom")
    ax.axhline(linewidth=2.0, color="k")
    plt.ylabel("validation error", fontsize=FONT_SIZE)
    plt.ylim(error_range)
    plt.setp(ax.get_yticklabels(), fontsize=TICK_FONT_SIZE)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0%}".format(y)))
    ax.yaxis.set_ticks_position("left")
    ax.axvline(linewidth=2.0, color="k")
    if show_legend:
      plt.legend(handles=plots, fontsize=LEGEND_FONT_SIZE)
    plt.show()
    #plt.savefig("{}_epoch.pdf".format(out_prefix))
  _make_error_plot(fig, ax, plots, num_procs, "training epochs", epochs_range, error_range, show_legend, out_prefix, "epoch")

if __name__ == "__main__":
  main()
