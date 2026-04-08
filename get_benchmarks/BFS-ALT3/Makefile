# Makefile for Local Traversal BFS on UPMEM DPU

# Configuration
NR_TASKLETS ?= 16
NR_DPUS ?= 64

# Compiler settings
DPU_CC = dpu-upmem-dpurte-clang
HOST_CXX = g++

# Flags
DPU_CFLAGS = -DNR_TASKLETS=$(NR_TASKLETS) -O3 -Wall
HOST_CXXFLAGS = -O3 -Wall -std=c++11
HOST_CXXFLAGS = -O3 -Wall -std=c++11 $(shell pkg-config --cflags dpu)
HOST_LDFLAGS = $(shell pkg-config --libs dpu)
# Directories
BIN_DIR = bin
SRC_DIR = .
SUPPORT_DIR = ../support

# Targets
DPU_TARGET = $(BIN_DIR)/dpu_code
HOST_TARGET = $(BIN_DIR)/app_local

# Source files
DPU_SRC = $(SRC_DIR)/dpu/task.c
HOST_SRC = $(SRC_DIR)/host/app.cpp

# Headers
DPU_HEADERS = $(SRC_DIR)/support/nodeObject.h \
              $(SRC_DIR)/support/dpu_params_local.h \
              $(SRC_DIR)/dpu/dpu-utils.h
              
HOST_HEADERS = $(SRC_DIR)/support/nodeObject.h \
               $(SRC_DIR)/support/dpu_params_local.h \
               $(SRC_DIR)/host/mram-management.h

# Default target
all: $(DPU_TARGET) $(HOST_TARGET)

# Create bin directory
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Build DPU code
$(DPU_TARGET): $(DPU_SRC) $(DPU_HEADERS) | $(BIN_DIR)
	$(DPU_CC) $(DPU_CFLAGS) -o $@ $(DPU_SRC)

# Build host code
$(HOST_TARGET): $(HOST_SRC) $(HOST_HEADERS) | $(BIN_DIR)
	$(HOST_CXX) $(HOST_CXXFLAGS) -o $@ $(HOST_SRC) $(HOST_LDFLAGS)

# Clean build artifacts
clean:
	rm -f $(DPU_TARGET) $(HOST_TARGET)
	rm -rf $(BIN_DIR)

# Run with default parameters
run: all
	$(HOST_TARGET) -f graph.txt -v 1

# Run with detailed logging
debug: all
	$(HOST_TARGET) -f graph.txt -v 2

# Display configuration
info:
	@echo "Configuration:"
	@echo "  NR_TASKLETS: $(NR_TASKLETS)"
	@echo "  NR_DPUS: $(NR_DPUS)"
	@echo "  DPU_CC: $(DPU_CC)"
	@echo "  HOST_CXX: $(HOST_CXX)"

# Help target
help:
	@echo "Available targets:"
	@echo "  all      - Build both DPU and host code (default)"
	@echo "  clean    - Remove build artifacts"
	@echo "  run      - Build and run with default parameters"
	@echo "  debug    - Build and run with detailed logging"
	@echo "  info     - Display build configuration"
	@echo "  help     - Display this help message"
	@echo ""
	@echo "Configuration variables:"
	@echo "  NR_TASKLETS - Number of tasklets per DPU (default: 16)"
	@echo "  NR_DPUS     - Number of DPUs to allocate (default: 64)"
	@echo ""
	@echo "Example usage:"
	@echo "  make NR_TASKLETS=24 NR_DPUS=128"
	@echo "  make run"

.PHONY: all clean run debug info help