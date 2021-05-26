use crate::helper::*;
extern crate image;
use crate::trt_pnet::*;
use crate::trt_rnet::*;
use image::*;
use ndarray::prelude::Axis;
use rustacuda::prelude::*;
use tensorrt_rs::runtime::*;
use npp_rs::image::CudaImage;
use npp_rs::imageops::resize;
use rustacuda::error::CudaError;
use std::convert::TryFrom;
use std::cmp;

pub struct Mtcnn {
    pnet: TrtPnet,
    rnet: TrtRnet,
    mlogger: Logger,
    scaled_img: CudaImage<u8>,
    cuda_ctx: Context,
}

impl Mtcnn {
    pub fn new(engine_path: &str) -> Result<Mtcnn, String> {
        rustacuda::init(rustacuda::CudaFlags::empty()).unwrap();
        let device = Device::get_device(0).unwrap();
        let ctx =
            Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
                .unwrap();

        let log = Logger::new();
        let pnet_t = TrtPnet::new(&std::format!("{}/det1.engine", engine_path)[..], &log)?;
        let rnet_t = TrtRnet::new(&std::format!("{}/det2.engine", engine_path)[..], &log)?;
        let img = CudaImage::<u8>::new(1280, 720, ColorType::Rgb8).unwrap();

        Ok(Mtcnn {
            pnet: pnet_t,
            rnet: rnet_t,
            mlogger: log,
            scaled_img: img,
            cuda_ctx: ctx,
        })
    }

    pub fn detect(&self, image: &DynamicImage, minsize: u32) -> Vec<[f32; 5]> {
        let (rescaled_img, min_size) = self.rescale(image, minsize);
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

    pub fn rescale(&self, image: &DynamicImage, min_size: u32) -> (RgbImage, u32) {
        let scale = f32::min(720.0 / image.height() as f32, 1280.0 / image.width() as f32);
        let (width, height) = if scale < 1.0 {
            (
                (image.width() as f32 * scale).ceil() as u32,
                (image.height() as f32 * scale).ceil() as u32,
            )
        } else {
            (image.width(), image.height())
        };
        let ms = || {
            if scale < 1.0 {
                return cmp::max((min_size as f32 * scale).ceil() as u32, 40);
            } else {
                return min_size;
            }
        };
        let img_layout_src = image.as_rgb8().unwrap().sample_layout();

        let cuda_src = CudaImage::try_from(image.as_rgb8().unwrap()).unwrap();

        let mut cuda_dst = self.scaled_img.sub_image(0, 0, width.clamp(1, 1280), height.clamp(1, 720)).unwrap();
        let _res = resize(&cuda_src, &mut cuda_dst).unwrap();
        (RgbImage::try_from(&cuda_dst).unwrap(), ms())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_npy::read_npy;
    use rustacuda::prelude::*;

    #[test]
    fn test_rescale() {
        rustacuda::init(rustacuda::CudaFlags::empty()).unwrap();
        let device = Device::get_device(0).unwrap();
        let _ctx =
            Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
                .unwrap();

        let mt = Mtcnn::new("./test_resources").unwrap();

        let img1 = image::open("test_resources/2020-11-21-144033.jpg").unwrap();

        let (scaled_image1, min_size) = mt.rescale(&img1, 40);

        assert_eq!(min_size, 40);
        assert_eq!(scaled_image1.width(), 640);
        assert_eq!(scaled_image1.height(), 360);

        let img2 = image::open("test_resources/DSC_0003.JPG").unwrap();

        let (scaled_image2, min_size) = mt.rescale(&img2, 40);

        assert_eq!(min_size, 40);
        assert_eq!(scaled_image2.width(), 1076);
        assert_eq!(scaled_image2.height(), 720);
    }
}
