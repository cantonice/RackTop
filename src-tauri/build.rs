fn main() {
    // tauri-build compiles this into the Windows PE resource but does not watch it.
    println!("cargo:rerun-if-changed=icons/icon.ico");
    tauri_build::build()
}
