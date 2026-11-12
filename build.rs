// Compiles the Pulse beat tracker (C++17, no libraries) into the app. Same code as the DECK001 firmware.
fn main() {
    tauri_build::build();

    let files = ["native/pulse/pulse_beat.cpp", "native/pulse/pulse_ffi.cpp"];
    let mut b = cc::Build::new();
    b.cpp(true).std("c++17").include("native/pulse");
    for f in files {
        b.file(f);
        println!("cargo:rerun-if-changed={f}");
    }
    println!("cargo:rerun-if-changed=native/pulse/pulse_beat.h");
    println!("cargo:rerun-if-changed=native/pulse/pulse_weights.h");
    if b.get_compiler().is_like_msvc() {
        b.flag("/O2").flag("/fp:fast");
    } else {
        // GCC and Clang (macOS and Linux): -O3 -ffast-math, as in the firmware build.
        b.opt_level(3).flag("-ffast-math");
    }
    b.compile("pulse");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-lib=pthread");   // std::thread on older glibc
    }
}