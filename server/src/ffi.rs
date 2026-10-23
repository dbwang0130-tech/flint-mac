//! flint-metal 引擎的 FFI 绑定（对应 ../src/capi.h）。

use std::ffi::{c_char, CStr, CString};

#[repr(C)]
struct FlintConfig {
    model_path: *const c_char,
    metallib_dir: *const c_char,
    pool_gb: f64,
    n_ctx: i32,
}

#[repr(C)]
struct FlintEngineRaw {
    _private: [u8; 0],
}

extern "C" {
    fn flint_open(cfg: *const FlintConfig, err: *mut c_char, err_len: i32) -> *mut FlintEngineRaw;
    fn flint_close(e: *mut FlintEngineRaw);
    fn flint_n_ctx(e: *const FlintEngineRaw) -> i32;
    fn flint_n_vocab() -> i32;
    fn flint_pos(e: *const FlintEngineRaw) -> i32;
    fn flint_truncate(e: *mut FlintEngineRaw, pos: i32) -> i32;
    fn flint_prefill(e: *mut FlintEngineRaw, tokens: *const i32, n: i32, err: *mut c_char, err_len: i32) -> i32;
    fn flint_decode(e: *mut FlintEngineRaw, token: i32, err: *mut c_char, err_len: i32) -> i32;
    fn flint_logits(e: *const FlintEngineRaw) -> *const f32;
}

/// 引擎句柄。单会话、非线程安全：由调用方加锁串行使用。
pub struct Engine {
    raw: *mut FlintEngineRaw,
}

// 引擎内部的 Metal 对象可在线程间移动使用（同一时刻只有一个线程调用，由外层 Mutex 保证）
unsafe impl Send for Engine {}

fn err_buf() -> Vec<c_char> {
    vec![0; 1024]
}

fn err_string(buf: &[c_char]) -> String {
    unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned()
}

impl Engine {
    pub fn open(model: &str, metallib_dir: &str, pool_gb: f64, n_ctx: i32) -> Result<Engine, String> {
        let model_c = CString::new(model).map_err(|e| e.to_string())?;
        let dir_c = CString::new(metallib_dir).map_err(|e| e.to_string())?;
        let cfg = FlintConfig { model_path: model_c.as_ptr(), metallib_dir: dir_c.as_ptr(), pool_gb, n_ctx };
        let mut err = err_buf();
        let raw = unsafe { flint_open(&cfg, err.as_mut_ptr(), err.len() as i32) };
        if raw.is_null() {
            return Err(err_string(&err));
        }
        Ok(Engine { raw })
    }

    pub fn n_ctx(&self) -> usize {
        unsafe { flint_n_ctx(self.raw) as usize }
    }

    pub fn n_vocab() -> usize {
        unsafe { flint_n_vocab() as usize }
    }

    pub fn pos(&self) -> usize {
        unsafe { flint_pos(self.raw) as usize }
    }

    pub fn truncate(&mut self, pos: usize) {
        unsafe { flint_truncate(self.raw, pos as i32) };
    }

    /// 从当前位置起处理一批 token，之后 `logits()` 为最后一个位置的分布
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<(), String> {
        let t: Vec<i32> = tokens.iter().map(|&x| x as i32).collect();
        let mut err = err_buf();
        let r = unsafe { flint_prefill(self.raw, t.as_ptr(), t.len() as i32, err.as_mut_ptr(), err.len() as i32) };
        if r < 0 { Err(err_string(&err)) } else { Ok(()) }
    }

    pub fn decode(&mut self, token: u32) -> Result<(), String> {
        let mut err = err_buf();
        let r = unsafe { flint_decode(self.raw, token as i32, err.as_mut_ptr(), err.len() as i32) };
        if r < 0 { Err(err_string(&err)) } else { Ok(()) }
    }

    pub fn logits(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(flint_logits(self.raw), Self::n_vocab()) }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { flint_close(self.raw) };
    }
}
