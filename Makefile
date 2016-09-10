LIBRARY_PATH := /usr/local/cudnn_v5/lib64:/usr/local/cuda-7.5/lib64:/opt/openmpi/lib

.PHONY: all debug clean

all:
	#CARGOFLAGS=-j1 cargo build --release --features openmpi
	LIBRARY_PATH=$(LIBRARY_PATH) cargo build --release

debug:
	#CARGOFLAGS=-j1 cargo build --features openmpi
	LIBRARY_PATH=$(LIBRARY_PATH) cargo build

clean:
	cargo clean
