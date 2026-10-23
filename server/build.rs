// 链接 flint-metal 引擎静态库（由项目根目录 make 生成）及其依赖的系统框架
fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("build");
    println!("cargo:rustc-link-search=native={}", root.display());
    println!("cargo:rustc-link-lib=static=flint");
    for fw in ["Metal", "Foundation", "QuartzCore"] {
        println!("cargo:rustc-link-lib=framework={fw}");
    }
    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=objc");
    println!("cargo:rerun-if-changed={}", root.join("libflint.a").display());
}
