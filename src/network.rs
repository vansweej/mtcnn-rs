// extern crate image;
// extern crate tensorrt_rs;

// // use std::format;

// // use tensorrt::builder::Builder;
// // use tensorrt::dims::Dims3;
// use tensorrt_rs::engine::Engine;
// use tensorrt_rs::runtime::Logger;
// use tensorrt_rs::runtime::Runtime;

// use std::fs::File;
// use std::io::prelude::*;
// use std::path::Path;

// pub struct TensorRTNN {
//     pnet_engine: Engine,
//     rnet_engine: Engine,
//     onet_engine: Engine,
// }

// impl TensorRTNN {
//     // pub fn new(engine_path: &str) -> Result<TensorRTNN, String> {
//     //     let logger = Logger::new();

//     //     if Path::new(engine_path).exists() {
//     //         let pnet_runtime = Runtime::new(&logger);
//     //         let mut f = File::open(engine_path.to_string() + "/det1.engine").unwrap();
//     //         let mut pnet_buffer = Vec::new();
//     //         f.read_to_end(&mut pnet_buffer).unwrap();
//     //         drop(f);

//     //         let rnet_runtime = Runtime::new(&logger);
//     //         f = File::open(engine_path.to_string() + "/det2.engine").unwrap();
//     //         let mut rnet_buffer = Vec::new();
//     //         f.read_to_end(&mut rnet_buffer).unwrap();
//     //         drop(f);

//     //         let onet_runtime = Runtime::new(&logger);
//     //         f = File::open(engine_path.to_string() + "/det3.engine").unwrap();
//     //         let mut onet_buffer = Vec::new();
//     //         f.read_to_end(&mut onet_buffer).unwrap();
//     //         drop(f);

//     //         Ok(TensorRTNN {
//     //             pnet_engine: Engine::new(pnet_runtime, pnet_buffer),
//     //             rnet_engine: Engine::new(rnet_runtime, rnet_buffer),
//     //             onet_engine: Engine::new(onet_runtime, onet_buffer),
//     //         })
//     //     } else {
//     //         Err("Path does not exists".to_string())
//     //     }
//     // }

//     pub fn infer(&self, input_tensor: &[u8]) -> Result<Vec<f32>, String> {
//         let mut vec_in = Vec::<f32>::with_capacity(784);

//         for i in 0..784 {
//             let norm_pixel = 1.0 as f32 - (input_tensor[i] as f32 / 255.0);
//             vec_in.push(norm_pixel);
//         }

//         let mut vec_out = vec![0.0 as f32; 10];

//         let context = self.pnet_engine.create_execution_context();

//         let size_in = vec_in.len() * 4;
//         let size_out = vec_out.len() * 4;

//         //        context.execute(vec_in, size_in, 0, &mut vec_out, size_out, 1);

//         Ok(vec_out)
//     }
// }

// unsafe impl Send for TensorRTNN {}
// unsafe impl Sync for TensorRTNN {}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use std::path::Path;
//     // use image::GenericImageView;

//     #[test]
//     fn test_create_tensorrt_nn() {
//         let engine_path = "test_resources";
//         assert_eq!(Path::new(engine_path).exists(), true);

//         let tensorrt_nn = TensorRTNN::new(engine_path);
//         assert_eq!(tensorrt_nn.is_ok(), true);
//     }

//     // #[test]
//     // fn test_infer_tensorrt_nn() {
//     //     let tensorrt_nn = TensorRTNN::new("test_resources/lenet5.uff");
//     //     assert_eq!(tensorrt_nn.is_ok(), true);
//     //     let tensorrt = tensorrt_nn.unwrap();

//     //     {
//     //         let img = image::open("test_resources/0.pgm").unwrap();
//     //         let mut vec_in = Vec::<u8>::with_capacity(784);
//     //         for y in 0..28 {
//     //             for x in 0..28 {
//     //                 let pixel = img.get_pixel(x, y);
//     //                 vec_in.push(pixel[0]);
//     //             }
//     //         }

//     //         let res = tensorrt.infer(&vec_in);
//     //         assert_eq!(res.is_ok(), true);
//     //         let resvec = res.unwrap();
//     //         let resvec0 = [
//     //             19.033106,
//     //             -5.3259997,
//     //             1.1770546,
//     //             -6.69033,
//     //             -0.51755106,
//     //             -9.703967,
//     //             2.6883106,
//     //             -6.8328795,
//     //             -0.82511306,
//     //             0.29254186,
//     //         ];
//     //         for i in 0..10 {
//     //             relative_eq!(resvec[i], resvec0[i]);
//     //         }
//     //     }
//     // }
// }
