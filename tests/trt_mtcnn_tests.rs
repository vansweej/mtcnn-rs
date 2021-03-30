use mtcnn_rs::trt_mtcnn::*;

#[test]
fn test_mtcnn_new() {
    let mt = Mtcnn::new("./test_resources");
}

#[test]
fn test_detect() {
    let mt = Mtcnn::new("./test_resources").unwrap();

    let img = image::open("test_resources/DSC_0003.JPG").unwrap();

    let face = mt.detect(&img, 40);

    println!("{:?}", face);
}
