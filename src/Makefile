# Compiler settings
CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++11 -O3
# Note: Adjust -arch based on your GPU (sm_50 for GTX 950M, sm_60 for GTX 1060, etc.)
NVCCFLAGS = -std=c++11 -O3 -arch=sm_50

# Targets
all: sequential mapreduce cuda

# Sequential version
sequential:
	@echo "Building sequential version..."
	@$(CXX) $(CXXFLAGS) WorldCount.cpp -o WorldCount.out
	@echo "Running sequential version..."
	@./WorldCount.out

# MapReduce version
mapreduce:
	@echo "Building MapReduce version..."
	@$(CXX) $(CXXFLAGS) WorldCount_map_reduce.cpp -o WorldCount_map_reduce.out
	@echo "Running MapReduce version..."
	@./WorldCount_map_reduce.out

# CUDA version
cuda:
	@echo "Building CUDA version..."
	@$(NVCC) $(NVCCFLAGS) WorldCount_cuda.cu -o WorldCount_cuda.out
	@echo "Running CUDA version..."
	@./WorldCount_cuda.out

# Run only CUDA version
run-cuda:
	@./WorldCount_cuda.out

# Build only CUDA version
build-cuda:
	@echo "Building CUDA version..."
	@$(NVCC) $(NVCCFLAGS) WorldCount_cuda.cu -o WorldCount_cuda.out

# Clean all executables
clean:
	@rm -f *.out

# Help
help:
	@echo "Available targets:"
	@echo "  all        - Build and run all versions"
	@echo "  sequential - Build and run sequential version"
	@echo "  mapreduce  - Build and run MapReduce version"
	@echo "  cuda       - Build and run CUDA version"
	@echo "  build-cuda - Build CUDA version only"
	@echo "  run-cuda   - Run CUDA version only"
	@echo "  clean      - Remove all executables"

.PHONY: all sequential mapreduce cuda run-cuda build-cuda clean help
