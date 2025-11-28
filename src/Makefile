# ==========================================
# 整合版 Makefile: Pthread, OpenMP, MPI, CUDA
# ==========================================

# --- 編譯器設定 ---
CXX      = g++
MPICXX   = mpicxx
NVCC     = nvcc

# --- 編譯旗標 ---
# 一般 C++ (C++17, O3 優化)
CXXFLAGS = -std=c++17 -O3 -Wall -g -fPIC -I./common -Iobjs/ -lm
# OpenMP
OMPFLAGS = -fopenmp
# Pthread
PTHREADFLAGS = -pthread

# CUDA (依你的 GPU 調整 compute_xx / sm_xx)
# 使用 sm_50 以相容更多 GPU，並加入 --extended-lambda 支援 thrust lambda
CUDA_FLAGS = --extended-lambda -arch=sm_50 -O3

# --- 路徑與資料夾 ---
OBJDIR    = objs
COMMONDIR = ./common

# ==========================================
# WorldCount / MapReduce / MPI 區
# ==========================================

SEQ_SRC     = WorldCount_sequential.cpp
MR_SRC      = WorldCount_map_reduce.cpp
PTH_SRC     = WorldCount_map_reduce_pthread.cpp
OMP_SRC     = WorldCount_map_reduce_omp.cpp
MPI_SRC     = WorldCountMPI.cpp
MPI_MR_SRC  = WorldCount_map_reduce_mpi.cpp

SEQ_BIN     = WorldCount_sequential
MR_BIN      = WorldCount_map_reduce
PTH_BIN     = WorldCount_map_reduce_pthread
OMP_BIN     = WorldCount_map_reduce_omp
MPI_BIN     = WorldCountMPI
MPI_MR_BIN  = WorldCount_map_reduce_mpi

# --- CUDA 版本 ---
CUDA_SRC    = WorldCount_cuda.cu
CUDA_BIN    = WorldCount_cuda

# MPI 預設執行緒數量
NP ?= 4

# ==========================================
# 主要目標 / phony targets
# ==========================================

.PHONY: all clean dirs \
	run_all run_seq run_mr run_pth run_omp run_mpi run_cuda

# 一次編所有東西
all: dirs \
	$(SEQ_BIN) $(MR_BIN) $(PTH_BIN) $(OMP_BIN) \
	$(MPI_BIN) $(MPI_MR_BIN) \
	$(CUDA_BIN)

# 建立物件檔資料夾
dirs:
	/bin/mkdir -p $(OBJDIR)/

# ===========================
# 編譯規則: CUDA WordCount
# ===========================

$(CUDA_BIN): $(CUDA_SRC)
	$(NVCC) $(CUDA_FLAGS) -o $@ $<

# ===========================
# 編譯規則: World Count (CPU)
# ===========================

$(SEQ_BIN): $(SEQ_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(MR_BIN): $(MR_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(PTH_BIN): $(PTH_SRC)
	$(CXX) $(CXXFLAGS) $(PTHREADFLAGS) -o $@ $<

$(OMP_BIN): $(OMP_SRC)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<

$(MPI_BIN): $(MPI_SRC)
	$(MPICXX) $(CXXFLAGS) -o $@ $<

$(MPI_MR_BIN): $(MPI_MR_SRC)
	$(MPICXX) $(CXXFLAGS) -o $@ $<

# ===========================
# 執行規則 (Run Targets)
# ===========================

run_all: run_seq run_mr run_pth run_omp run_mpi run_cuda

run_seq: $(SEQ_BIN)
	@echo ">>> Running Sequential"
	./$(SEQ_BIN)

run_mr: $(MR_BIN)
	@echo ">>> Running MapReduce (sequential)"
	./$(MR_BIN)

run_pth: $(PTH_BIN)
	@echo ">>> Running Pthread"
	./$(PTH_BIN)

run_omp: $(OMP_BIN)
	@echo ">>> Running OpenMP"
	./$(OMP_BIN)

run_mpi: $(MPI_BIN) $(MPI_MR_BIN)
	@echo ">>> Running MPI (basic) with $(NP) processes"
	mpirun -np $(NP) ./$(MPI_BIN)
	@echo ">>> Running MPI (MapReduce) with $(NP) processes"
	mpirun -np $(NP) ./$(MPI_MR_BIN)

run_cuda: $(CUDA_BIN)
	@echo ">>> Running CUDA WordCount"
	./$(CUDA_BIN)

# ===========================
# 清除
# ===========================

clean:
	rm -f $(SEQ_BIN) $(MR_BIN) $(PTH_BIN) $(OMP_BIN) $(MPI_BIN) $(MPI_MR_BIN) $(CUDA_BIN)
	rm -rf $(OBJDIR) *.ppm *.out
