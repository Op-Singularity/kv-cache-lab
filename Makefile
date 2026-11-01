CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic
INCLUDES := -Isrc

BIN  := bin/kvcalc
TEST := bin/test_kvcalc
SRC  := src/kvcalc.cpp src/main.cpp

.PHONY: all test clean run

all: $(BIN)

$(BIN): $(SRC) src/kvcalc.hpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) -o $@

$(TEST): src/kvcalc.cpp tests/test_kvcalc.cpp src/kvcalc.hpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $(INCLUDES) src/kvcalc.cpp tests/test_kvcalc.cpp -o $@

test: $(TEST)
	./$(TEST)

run: $(BIN)
	./$(BIN) --layers 32 --full-attn 8 --kv-heads 4 --head-dim 256 \
	         --vram-mib 8151 --util 0.85 --weights-mib 3891 --check 76933

clean:
	rm -rf bin
