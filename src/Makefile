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
CUDA_COMPILE_FLAGS = --device-c -gencode=arch=compute_61,code=sm_61 -Xcompiler "-fPIC" -g -O3
CUDA_LINK_FLAGS    = -rdc=true -gencode=arch=compute_61,code=sm_61 -Xcompiler "-fPIC"

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

# ==========================================
# CUDA 區（從你的 Makefilecuda 合併而來）
# ==========================================

# CUDA 執行檔名字：**這裡改成你想要的名字**
# 例如：CUDA_APP = hw5_cuda  或  CUDA_APP = mandelbrot
CUDA_APP = cuda_app

# 通常不用動：common/ppm.cpp
PPM_SRC = $(COMMONDIR)/ppm.cpp
PPM_OBJ = $(OBJDIR)/ppm.o

# 這邊照你原本 Makefilecuda 寫的物件檔。
# 如果你的 .cpp 檔名跟這些不一樣，就改這裡。
CUDA_OBJS = \
	$(OBJDIR)/main.o \
	$(OBJDIR)/kernel.o \
	$(OBJDIR)/mandelbrotSerial.o \
	$(OBJDIR)/mandelbrotThread.o \
	$(PPM_OBJ)

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
	$(CUDA_APP)

# 建立物件檔資料夾
dirs:
	/bin/mkdir -p $(OBJDIR)/

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
# 編譯規則: CUDA
# ===========================

# 最終 CUDA 執行檔
$(CUDA_APP): dirs $(CUDA_OBJS)
	$(NVCC) $(CUDA_LINK_FLAGS) -o $@ $(CUDA_OBJS) mandelbrotThreadRef.a

# 一般 C++ -> .o（含 CUDA 專案用的 .cpp）
$(OBJDIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# common/*.cpp -> .o
$(OBJDIR)/%.o: $(COMMONDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# main.cpp 依賴 common/CycleTimer.h
$(OBJDIR)/main.o: main.cpp $(COMMONDIR)/CycleTimer.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# kernel.cu -> kernel.o
$(OBJDIR)/kernel.o: kernel.cu kernel.h
	$(NVCC) $(CUDA_COMPILE_FLAGS) -c kernel.cu -o $@

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

# run_cuda: $(CUDA_APP)
# 	@echo ">>> Running CUDA app: $(CUDA_APP)"
# 	./$(CUDA_APP)

# ===========================
# 清除
# ===========================

clean:
	rm -f $(SEQ_BIN) $(MR_BIN) $(PTH_BIN) $(OMP_BIN) $(MPI_BIN) $(MPI_MR_BIN) $(CUDA_APP)
	rm -rf $(OBJDIR) *.ppm *.out
