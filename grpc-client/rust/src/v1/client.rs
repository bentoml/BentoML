#[path = "bentoml.grpc.v1.rs"]
pub mod api;

use api::bento_service_client::BentoServiceClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let channel = tonic::transport::Channel::from_static("http://[::1]:3000")
        .connect()
        .await?;
    let mut client = BentoServiceClient::new(channel);

    let request = tonic::Request::new(api::Request {
        api_name: String::from("classify"),
        content: Some(api::request::Content::Ndarray(api::NdArray {
            float_values: vec![5.9, 3.0, 5.1, 1.8],
            string_values: vec![],
            double_values: vec![],
            bool_values: vec![],
            int32_values: vec![],
            int64_values: vec![],
            uint32_values: vec![],
            uint64_values: vec![],
            shape: vec![1, 4],
            dtype: 1,
        })),
    });
    let response = client.call(request).await?.into_inner();
    println!("{:?}", response);
    Ok(())
}
