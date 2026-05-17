# PRIM-BFS Fragmented Graph Project

> This project is aimed at building a new, performant version of BFS leveraging PRIM

---

## Overview

The basic overview is that we are designing and testing a new way to do BFS on PRIM DPUs. We are leveraging the memory storage of DPUs and breaking up graphs into fragments unique to each dpu, they then conduct BFS on the graph fragments and sync back and forth with the CPU global truth nodelists. 

The fastest verison is in the get_benchmarks folder which is basically BFS_ALT3 with CSR implemented within it which reduces the data size being moved. Though it is still slower than the benchmark which does level/frontier based iterations.

---

## Getting Started

### Prerequisites

<!-- List any dependencies, software versions, or environment requirements needed before setup. -->

- A server with UPMEM DRAM
- use this for reference with UPMEM https://sdk.upmem.com/2025.1.0/index.html
- some python files will require certain libraries most should be built in

### Installation

```bash
# Clone the repository
git clone https://github.com/GharretWilcut/HPC-lab-Work-GW.git
cd HPC-lab-Work-GW

```

---

## Usage

### Basic Usage
non benchmark usage within the folder
```bash
./bin/app_local -v 3 ./data/Graph_1.txt 
```
-v == verbosity changes detail from reports

comparison of different bfs versions within get_benchmarks
```bash
chmod +x ./test_run.sh 
./test_run.sh ./data/Graph_1.txt 
```
### 


---


---

## Notes & Known Issues

<!-- Any caveats, limitations, or things teammates should be aware of. -->

- having the project completely to spec has been an issue the part of the spec with the global truth being sent and not just changes has been an issue that has not been resolved yet right now it just sends any change made 

---

## Contact

| Name | Role | Contact |
|---|---|---|
| Gharret Wilcut| repo maker| gharretwilcut@gmail.com|
if you have any questions reach out