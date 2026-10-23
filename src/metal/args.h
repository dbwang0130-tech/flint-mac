// v1.1 的参数结构（重建，仅用于 A/B 对比）
#pragma once
#ifdef __METAL_VERSION__
#define FL_U64 ulong
#else
#include <cstdint>
#define FL_U64 uint64_t
typedef uint32_t uint;
#endif
struct MvArgs { FL_U64 mat_stride; uint K; uint N; uint row_bytes; uint x_stride; uint y_stride; uint use_ids; uint add; float scale; };
struct MvDownArgs { FL_U64 mat_stride; uint K; uint N; uint row_bytes; uint n_mat; uint use_ids; uint add; };
struct NormArgs { uint n; float eps; };
struct QkArgs { uint n_head; uint n_head_kv; uint head_dim; uint pos; uint slot; uint has_rope; float eps; float rope_theta; };
struct AttnArgs { uint n_head; uint n_head_kv; uint head_dim; uint n_keys; uint n_slots; uint first_slot; float scale; };
struct TopkArgs { uint n_expert; uint k; };
struct RowArgs { uint K; uint row_bytes; uint row; };
