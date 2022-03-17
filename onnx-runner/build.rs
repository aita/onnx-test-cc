use cmake;
use std::env;
use std::path::Path;

fn main() {
    let dst = cmake::build("cpp");
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=onnx-runner");

    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=onnxruntime");

    let out_dir = env::var("OUT_DIR").unwrap();
    std::fs::copy(
        Path::new(format!("{}/lib/libonnxruntime.so", dst.display()).as_str()),
        Path::new(&out_dir).join("libonnxruntime.so"),
    )
    .unwrap();
    std::fs::copy(
        Path::new(format!("{}/lib/libonnxruntime.so.1.10.0", dst.display()).as_str()),
        Path::new(&out_dir).join("libonnxruntime.so.1.10.0"),
    )
    .unwrap();
}
