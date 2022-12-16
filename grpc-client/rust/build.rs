fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .out_dir("./src/v1/")
        .compile(&["../bentoml/grpc/v1/service.proto"], &[".."])
        .unwrap();
    tonic_build::configure()
        .out_dir("./src/v1alpha1/")
        .compile(&["../bentoml/grpc/v1alpha1/service.proto"], &[".."])
        .unwrap();
    Ok(())
}
