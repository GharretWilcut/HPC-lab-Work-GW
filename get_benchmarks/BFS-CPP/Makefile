DPU_DIR := dpu
HOST_DIR := host
BUILDDIR ?= bin

NR_TASKLETS ?= 16
BL ?= 10
NR_DPUS ?= 64
TYPE ?= INT32
ENERGY ?= 0

define conf_filename
	${BUILDDIR}/.NR_DPUS_$(1)_NR_TASKLETS_$(2)_BL_$(3)_TYPE_$(4).conf
endef

CONF := $(call conf_filename,${NR_DPUS},${NR_TASKLETS},${BL},${TYPE})

HOST_TARGET := ${BUILDDIR}/host_code
DPU_TARGET := ${BUILDDIR}/dpu_code

COMMON_INCLUDES := support

# 🔁 Host is now C++
HOST_SOURCES := $(wildcard ${HOST_DIR}/*.cpp)
DPU_SOURCES := $(wildcard ${DPU_DIR}/*.c)

.PHONY: all clean test

__dirs := $(shell mkdir -p ${BUILDDIR})

COMMON_FLAGS := -Wall -Wextra -g -I${COMMON_INCLUDES}

# 🔁 C++ flags
HOST_FLAGS := ${COMMON_FLAGS} \
	-std=c++17 -O3 \
	`dpu-pkg-config --cflags dpu` \
	`dpu-pkg-config --libs dpu` \
	-lstdc++ \
	-DNR_TASKLETS=${NR_TASKLETS} \
	-DNR_DPUS=${NR_DPUS} \
	-DBL=${BL} \
	-D${TYPE} \
	-DENERGY=${ENERGY}

DPU_FLAGS := ${COMMON_FLAGS} \
	-O2 \
	-DNR_TASKLETS=${NR_TASKLETS} \
	-DBL=${BL} \
	-D${TYPE}

all: ${HOST_TARGET} ${DPU_TARGET}

${CONF}:
	$(RM) $(call conf_filename,*,*,*,*)
	touch ${CONF}

# 🔁 Use C++ compiler for host
${HOST_TARGET}: ${HOST_SOURCES} ${COMMON_INCLUDES} ${CONF}
	$(CXX) -o $@ ${HOST_SOURCES} ${HOST_FLAGS}

${DPU_TARGET}: ${DPU_SOURCES} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -o $@ ${DPU_SOURCES}

clean:
	$(RM) -r $(BUILDDIR)

test: all
	./${HOST_TARGET}
