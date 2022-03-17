use cmake;
use std::env;
use std::path::Path;
use std::path::PathBuf;

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

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I/{}/include", dst.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
