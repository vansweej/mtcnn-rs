use crate::helper::*;
extern crate image;

use crate::trt_pnet::*;
use crate::trt_rnet::*;
use image::*;
use ndarray::prelude::*;
use tensorrt_rs::runtime::*;

pub struct Mtcnn {
    pnet: TrtPnet,
    rnet: TrtRnet,
    mlogger: Logger,
}

impl Mtcnn {
    pub fn new(engine_path: &str, logger: &Logger) -> Result<Mtcnn, String> {
        let log = Logger::new();
        
        let pnet_t = TrtPnet::new(&std::format!("{}/det1.engine", engine_path)[..], &log)?;
        let rnet_t = TrtRnet::new(&std::format!("{}/det2.engine", engine_path)[..], &log)?;

        Ok(Mtcnn {
            pnet: pnet_t,
            rnet: rnet_t,
            mlogger: log,
        })
    }

    pub fn detect(&self, image: &DynamicImage, minsize: u32) -> Vec<[f32; 5]> {
        let (rescaled_img, min_size) = rescale(image, minsize);
        let img = DynamicImage::ImageRgb8(rescaled_img);
        let pnet_dets = self.pnet.detect(&img, min_size, 0.709, 0.7);
        let rnet_dets = self.rnet.detect(&img, &pnet_dets, 256, 0.7);

        let scale = (720. / image.height() as f32).min(1280. / image.width() as f32);
        let result = rnet_dets
            .axis_iter(Axis(0))
            .map(|v| {
                if scale < 1.0 {
                    [v[0] / scale, v[1] / scale, v[2] / scale, v[3] / scale, v[4]]
                } else {
                    [v[0], v[1], v[2], v[3], v[4]]
                }
            })
            .collect::<Vec<_>>();

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helper::*;

    #[test]
    fn test_mtcnn_new() {
        let logger = Logger::new();

        let mt = Mtcnn::new("./test_resources", &logger);
    }

    #[test]
    fn test_detect() {
        let logger = Logger::new();
        let mt = Mtcnn::new("./test_resources", &logger).unwrap();

        let img = image::open("test_resources/DSC_0003.JPG").unwrap();

        let face = mt.detect(&img, 40);

        println!("{:?}", face);
    }
}
