//! 采样：temperature → top-k → top-p（官方推荐 1.0 / 20 / 0.95）。temperature ≤ 0 时贪心。

pub struct Sampler {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    state: u64,
}

impl Sampler {
    pub fn new(temperature: f32, top_k: usize, top_p: f32, seed: u64) -> Sampler {
        Sampler { temperature, top_k: top_k.max(1), top_p, state: seed | 1 }
    }

    fn next_f32(&mut self) -> f32 {
        // xorshift64*
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        let v = self.state.wrapping_mul(0x2545F4914F6CDD1D);
        (v >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        // 取 top-k（k 很小，单遍维护有序小数组）
        let k = self.top_k.min(logits.len());
        let mut top: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
        for (i, &v) in logits.iter().enumerate() {
            if top.len() < k || v > top[top.len() - 1].0 {
                let pos = top.partition_point(|&(x, _)| x >= v);
                top.insert(pos, (v, i as u32));
                if top.len() > k {
                    top.pop();
                }
            }
        }
        if self.temperature <= 0.0 {
            return top[0].1;
        }
        let m = top[0].0;
        let mut probs: Vec<f32> = top.iter().map(|&(v, _)| ((v - m) / self.temperature).exp()).collect();
        let sum: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= sum);
        // top-p：按概率降序累计到 top_p 为止
        let mut cut = probs.len();
        let mut acc = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            acc += p;
            if acc >= self.top_p {
                cut = i + 1;
                break;
            }
        }
        let total: f32 = probs[..cut].iter().sum();
        let mut r = self.next_f32() * total;
        for i in 0..cut {
            r -= probs[i];
            if r <= 0.0 {
                return top[i].1;
            }
        }
        top[cut - 1].1
    }
}
