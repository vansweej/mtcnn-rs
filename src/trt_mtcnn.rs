use crate::helper::*;
extern crate image;

use crate::trt_pnet::*;
use crate::trt_rnet::*;
use image::*;
use ndarray::prelude::*;
use tensorrt_rs::runtime::*;

pub struct mtcnn {
    pnet: TrtPnet,
    rnet: TrtRnet,
}

impl mtcnn {
    pub fn new(logger: &Logger) -> Result<mtcnn, String> {
        let pnet_t = TrtPnet::new("./test_resources/det1.engine", &logger)?;
        let rnet_t = TrtRnet::new("./test_resources/det2.engine", &logger)?;

        Ok(mtcnn {
            pnet: pnet_t,
            rnet: rnet_t,
        })
    }

    pub fn detect(&self, image: &DynamicImage, minsize: u32) -> Array2<f32> {
        let (rescaled_img, min_size) = rescale(image, minsize);
        let img = DynamicImage::ImageRgb8(rescaled_img);
        let pnet_dets = self.pnet.detect(&img, min_size, 0.709, 0.7);
        let rnet_dets = self.rnet.detect(&img, &pnet_dets, 256, 0.7);

        rnet_dets
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helper::*;

    #[test]
    fn test_mtcnn_new() {
        let logger = Logger::new();

        let mt = mtcnn::new(&logger);
    }

    #[test]
    fn test_detect() {
        let logger = Logger::new();
        let mt = mtcnn::new(&logger).unwrap();

        let img = image::open("test_resources/DSC_0003.JPG").unwrap();

        let face = mt.detect(&img, 40);

        println!("{:?}", face);
    }
}
