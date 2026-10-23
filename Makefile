# flint-metal 构建
GGML      := third_party/ggml
BUILD     := build
CXX       := clang++
CXXFLAGS  := -std=c++20 -O2 -Wall -fobjc-arc
METALFLAGS:= -std=metal4.0 -O3 -Wall -I$(GGML) -Isrc/metal
FRAMEWORKS:= -framework Metal -framework Foundation -framework QuartzCore

SRCS := src/main.mm src/engine.mm src/gguf.cpp src/ref/ref_model.cpp src/ref/dequant_row.cpp
HDRS := $(wildcard src/*.h src/ref/*.h src/metal/*.h)

all: $(BUILD)/flint $(BUILD)/flint.metallib $(BUILD)/fused.metallib $(BUILD)/prefill.metallib $(BUILD)/ref_check $(BUILD)/libflint.a

$(BUILD)/flint.metallib: src/metal/flint.metal src/metal/args.h
	@mkdir -p $(BUILD)
	xcrun metal $(METALFLAGS) -c $< -o $(BUILD)/flint.air
	xcrun metallib $(BUILD)/flint.air -o $@

$(BUILD)/fused.metallib: src/metal/fused.metal src/metal/fused_args.h src/metal/flint_config.h
	@mkdir -p $(BUILD)
	xcrun metal $(METALFLAGS) -c $< -o $(BUILD)/fused.air
	xcrun metallib $(BUILD)/fused.air -o $@

$(BUILD)/prefill.metallib: src/metal/prefill.metal src/metal/prefill_args.h src/metal/flint_config.h
	@mkdir -p $(BUILD)
	xcrun metal $(METALFLAGS) -c $< -o $(BUILD)/prefill.air
	xcrun metallib $(BUILD)/prefill.air -o $@

$(BUILD)/dequant.o: src/ref/dequant.c src/ref/dequant_gen.c
	@mkdir -p $(BUILD)
	clang -std=c11 -O2 -I$(GGML) -c $< -o $@

$(BUILD)/flint: $(SRCS) $(HDRS) $(BUILD)/dequant.o
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -o $@ $(SRCS) $(BUILD)/dequant.o

$(BUILD)/ref_check: src/ref/ref_check.cpp src/ref/ref_model.cpp src/ref/dequant_row.cpp src/gguf.cpp $(HDRS) $(BUILD)/dequant.o
	$(CXX) -std=c++20 -O2 -Wall -o $@ src/ref/ref_check.cpp src/ref/ref_model.cpp src/ref/dequant_row.cpp src/gguf.cpp $(BUILD)/dequant.o

# 供 Rust 服务链接的静态库（引擎 + C 接口）
LIB_SRCS := src/capi.mm src/engine.mm src/gguf.cpp src/ref/ref_model.cpp src/ref/dequant_row.cpp
$(BUILD)/libflint.a: $(LIB_SRCS) $(HDRS) src/capi.h $(BUILD)/dequant.o
	@mkdir -p $(BUILD)/lib
	cd $(BUILD)/lib && for f in $(LIB_SRCS); do $(CXX) $(CXXFLAGS) -c ../../$$f -o $$(basename $$f).o || exit 1; done
	rm -f $@ && ar rcs $@ $(BUILD)/lib/*.o $(BUILD)/dequant.o

clean:
	rm -rf $(BUILD)/flint $(BUILD)/*.metallib $(BUILD)/*.air $(BUILD)/*.o

.PHONY: all clean
