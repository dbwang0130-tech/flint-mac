#include "gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>

namespace flint {

namespace {

enum MetaType : uint32_t {
    U8 = 0, I8 = 1, U16 = 2, I16 = 3, U32 = 4, I32 = 5, F32 = 6, BOOL = 7,
    STRING = 8, ARRAY = 9, U64 = 10, I64 = 11, F64 = 12,
};

class Cursor {
public:
    Cursor(const uint8_t *p, size_t size) : p_(p), end_(p + size) {}

    template <typename T>
    T read() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, p_, sizeof(T));
        p_ += sizeof(T);
        return v;
    }

    std::string read_string() {
        uint64_t n = read<uint64_t>();
        need(n);
        std::string s(reinterpret_cast<const char *>(p_), n);
        p_ += n;
        return s;
    }

    size_t pos(const uint8_t *base) const { return p_ - base; }

private:
    void need(size_t n) const {
        if (static_cast<size_t>(end_ - p_) < n) throw std::runtime_error("gguf: unexpected end of file");
    }

    const uint8_t *p_;
    const uint8_t *end_;
};

bool is_integer(uint32_t t) {
    return t == U8 || t == I8 || t == U16 || t == I16 || t == U32 || t == I32 || t == U64 || t == I64;
}

int64_t read_integer(Cursor &c, uint32_t t) {
    switch (t) {
        case U8: return c.read<uint8_t>();
        case I8: return c.read<int8_t>();
        case U16: return c.read<uint16_t>();
        case I16: return c.read<int16_t>();
        case U32: return c.read<uint32_t>();
        case I32: return c.read<int32_t>();
        case U64: return static_cast<int64_t>(c.read<uint64_t>());
        case I64: return c.read<int64_t>();
    }
    throw std::runtime_error("gguf: not an integer type");
}

MetaValue read_value(Cursor &c, uint32_t t) {
    if (is_integer(t)) return read_integer(c, t);
    switch (t) {
        case F32: return static_cast<double>(c.read<float>());
        case F64: return c.read<double>();
        case BOOL: return c.read<uint8_t>() != 0;
        case STRING: return c.read_string();
        case ARRAY: {
            uint32_t et = c.read<uint32_t>();
            uint64_t n = c.read<uint64_t>();
            if (is_integer(et)) {
                std::vector<int64_t> v(n);
                for (auto &x : v) x = read_integer(c, et);
                return v;
            }
            if (et == F32 || et == F64) {
                std::vector<double> v(n);
                for (auto &x : v) x = et == F32 ? c.read<float>() : c.read<double>();
                return v;
            }
            if (et == BOOL) {
                std::vector<bool> v(n);
                for (size_t i = 0; i < n; i++) v[i] = c.read<uint8_t>() != 0;
                return v;
            }
            if (et == STRING) {
                std::vector<std::string> v(n);
                for (auto &x : v) x = c.read_string();
                return v;
            }
            throw std::runtime_error("gguf: unsupported array element type");
        }
    }
    throw std::runtime_error("gguf: unknown metadata type " + std::to_string(t));
}

}  // namespace

const TypeTraits &type_traits(GgmlType t) {
    static const std::map<GgmlType, TypeTraits> traits = {
        {GgmlType::F32, {"F32", 1, 4}},         {GgmlType::F16, {"F16", 1, 2}},
        {GgmlType::BF16, {"BF16", 1, 2}},       {GgmlType::Q4_0, {"Q4_0", 32, 18}},
        {GgmlType::Q8_0, {"Q8_0", 32, 34}},     {GgmlType::Q2_K, {"Q2_K", 256, 84}},
        {GgmlType::Q3_K, {"Q3_K", 256, 110}},   {GgmlType::Q4_K, {"Q4_K", 256, 144}},
        {GgmlType::Q5_K, {"Q5_K", 256, 176}},   {GgmlType::Q6_K, {"Q6_K", 256, 210}},
        {GgmlType::IQ2_XXS, {"IQ2_XXS", 256, 66}}, {GgmlType::IQ2_XS, {"IQ2_XS", 256, 74}},
        {GgmlType::IQ3_XXS, {"IQ3_XXS", 256, 98}}, {GgmlType::IQ1_S, {"IQ1_S", 256, 50}},
        {GgmlType::IQ4_NL, {"IQ4_NL", 32, 18}}, {GgmlType::IQ3_S, {"IQ3_S", 256, 110}},
        {GgmlType::IQ2_S, {"IQ2_S", 256, 82}},  {GgmlType::IQ4_XS, {"IQ4_XS", 256, 136}},
        {GgmlType::IQ1_M, {"IQ1_M", 256, 56}},
    };
    auto it = traits.find(t);
    if (it == traits.end()) throw std::runtime_error("unsupported ggml type " + std::to_string(uint32_t(t)));
    return it->second;
}

uint64_t Tensor::nelements() const {
    uint64_t n = 1;
    for (auto d : dims) n *= d;
    return n;
}

uint64_t Tensor::row_bytes() const {
    const auto &tt = type_traits(type);
    return dims[0] / tt.block_elems * tt.block_bytes;
}

GgufFile::GgufFile(const std::string &path) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("cannot open " + path);
    struct stat st {};
    fstat(fd_, &st);
    size_ = st.st_size;
    void *p = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (p == MAP_FAILED) throw std::runtime_error("mmap failed: " + path);
    base_ = static_cast<const uint8_t *>(p);

    Cursor c(base_, size_);
    if (c.read<uint32_t>() != 0x46554747) throw std::runtime_error("not a GGUF file: " + path);  // "GGUF"
    uint32_t version = c.read<uint32_t>();
    if (version < 3) throw std::runtime_error("unsupported GGUF version");
    uint64_t n_tensors = c.read<uint64_t>();
    uint64_t n_kv = c.read<uint64_t>();

    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = c.read_string();
        uint32_t t = c.read<uint32_t>();
        meta_[key] = read_value(c, t);
    }

    struct RawTensor {
        std::string name;
        std::vector<uint64_t> dims;
        uint32_t type;
        uint64_t rel_offset;
    };
    std::vector<RawTensor> raw(n_tensors);
    for (auto &rt : raw) {
        rt.name = c.read_string();
        uint32_t nd = c.read<uint32_t>();
        rt.dims.resize(nd);
        for (auto &d : rt.dims) d = c.read<uint64_t>();
        rt.type = c.read<uint32_t>();
        rt.rel_offset = c.read<uint64_t>();
    }

    uint64_t alignment = has("general.alignment") ? get_int("general.alignment") : 32;
    uint64_t pos = c.pos(base_);
    data_offset_ = (pos + alignment - 1) / alignment * alignment;

    tensors_.reserve(n_tensors);
    for (auto &rt : raw) {
        Tensor t;
        t.name = rt.name;
        t.type = static_cast<GgmlType>(rt.type);
        t.dims = rt.dims;
        t.offset = data_offset_ + rt.rel_offset;
        const auto &tt = type_traits(t.type);
        t.nbytes = t.nelements() / tt.block_elems * tt.block_bytes;
        if (t.offset + t.nbytes > size_) throw std::runtime_error("tensor out of bounds: " + t.name);
        t.data = base_ + t.offset;
        tensor_index_[t.name] = tensors_.size();
        tensors_.push_back(std::move(t));
    }
}

GgufFile::~GgufFile() {
    if (base_) munmap(const_cast<uint8_t *>(base_), size_);
    if (fd_ >= 0) close(fd_);
}

const MetaValue &GgufFile::meta(const std::string &key) const {
    auto it = meta_.find(key);
    if (it == meta_.end()) throw std::runtime_error("gguf: missing key " + key);
    return it->second;
}

int64_t GgufFile::get_int(const std::string &key) const { return std::get<int64_t>(meta(key)); }

double GgufFile::get_float(const std::string &key) const {
    const auto &v = meta(key);
    if (auto *i = std::get_if<int64_t>(&v)) return static_cast<double>(*i);
    return std::get<double>(v);
}

bool GgufFile::get_bool(const std::string &key) const { return std::get<bool>(meta(key)); }

const std::string &GgufFile::get_string(const std::string &key) const {
    return std::get<std::string>(meta(key));
}

std::vector<int64_t> GgufFile::get_int_array(const std::string &key) const {
    return std::get<std::vector<int64_t>>(meta(key));
}

std::vector<bool> GgufFile::get_bool_array(const std::string &key) const {
    return std::get<std::vector<bool>>(meta(key));
}

const std::vector<std::string> &GgufFile::get_string_array(const std::string &key) const {
    return std::get<std::vector<std::string>>(meta(key));
}

const Tensor *GgufFile::find_tensor(const std::string &name) const {
    auto it = tensor_index_.find(name);
    return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
}

const Tensor &GgufFile::tensor(const std::string &name) const {
    const Tensor *t = find_tensor(name);
    if (!t) throw std::runtime_error("missing tensor " + name);
    return *t;
}

}  // namespace flint
