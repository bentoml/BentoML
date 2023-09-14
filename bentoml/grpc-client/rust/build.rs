use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=../bentoml/grpc/v1/service.proto");
    fs::create_dir("src/v1/pb").unwrap_or(());
    let mut config = prost_build::Config::new();
    config.protoc_arg("--experimental_allow_proto3_optional");
    tonic_build::configure()
        .out_dir("./src/v1/pb")
        .build_client(true)
        .build_server(false)
        .include_file("mod.rs")
        .compile_with_config(
            config,
            &["../bentoml/grpc/v1/service.proto"],
            &[".."],
        ).unwrap_or_else(|e| panic!("protobuf compilation failed: {e}"));

    fs::create_dir("src/v1alpha1/pb").unwrap_or(());
    let mut config = prost_build::Config::new();
    config.protoc_arg("--experimental_allow_proto3_optional");
    tonic_build::configure()
        .out_dir("./src/v1alpha1/pb")
        .build_client(true)
        .build_server(false)
        .include_file("mod.rs")
        .compile_with_config(
            config,
            &["../bentoml/grpc/v1alpha1/service.proto"],
            &[".."],
        ).unwrap_or_else(|e| panic!("protobuf compilation failed: {e}"));
    Ok(())
}
