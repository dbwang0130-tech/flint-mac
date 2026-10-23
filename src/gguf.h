// GGUF 文件只读解析：mmap 整个文件，暴露 metadata 与张量表。
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace flint {

// 与 ggml_type 编号一致，只列本引擎关心的类型。
enum class GgmlType : uint32_t {
    F32 = 0, F16 = 1, Q4_0 = 2, Q8_0 = 8,
    Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14,
    IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ1_S = 19, IQ4_NL = 20,
    IQ3_S = 21, IQ2_S = 22, IQ4_XS = 23, IQ1_M = 29, BF16 = 30,
};

struct TypeTraits {
    const char *name;
    int block_elems;  // 每块元素数
    int block_bytes;  // 每块字节数
};

const TypeTraits &type_traits(GgmlType t);

struct Tensor {
    std::string name;
    GgmlType type;
    std::vector<uint64_t> dims;  // ggml 顺序：dims[0] 是最内层（行长）
    uint64_t offset;             // 相对文件起始的绝对偏移
    uint64_t nbytes;
    const uint8_t *data;         // 指向 mmap 区域

    uint64_t nelements() const;
    uint64_t row_bytes() const;  // dims[0] 个元素的字节数
};

using MetaValue = std::variant<int64_t, double, bool, std::string,
                               std::vector<int64_t>, std::vector<double>,
                               std::vector<bool>, std::vector<std::string>>;

class GgufFile {
public:
    explicit GgufFile(const std::string &path);
    ~GgufFile();
    GgufFile(const GgufFile &) = delete;
    GgufFile &operator=(const GgufFile &) = delete;

    const uint8_t *base() const { return base_; }
    size_t size() const { return size_; }
    uint64_t data_offset() const { return data_offset_; }

    bool has(const std::string &key) const { return meta_.count(key) != 0; }
    int64_t get_int(const std::string &key) const;
    double get_float(const std::string &key) const;
    bool get_bool(const std::string &key) const;
    const std::string &get_string(const std::string &key) const;
    std::vector<int64_t> get_int_array(const std::string &key) const;
    std::vector<bool> get_bool_array(const std::string &key) const;
    const std::vector<std::string> &get_string_array(const std::string &key) const;

    const Tensor &tensor(const std::string &name) const;
    const Tensor *find_tensor(const std::string &name) const;
    const std::vector<Tensor> &tensors() const { return tensors_; }

private:
    const MetaValue &meta(const std::string &key) const;

    int fd_ = -1;
    const uint8_t *base_ = nullptr;
    size_t size_ = 0;
    uint64_t data_offset_ = 0;
    std::map<std::string, MetaValue> meta_;
    std::vector<Tensor> tensors_;
    std::map<std::string, size_t> tensor_index_;
};

}  // namespace flint
