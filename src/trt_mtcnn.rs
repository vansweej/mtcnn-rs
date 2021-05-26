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

        Ok(Mtcnn {
            pnet: pnet_t,
            rnet: rnet_t,
            mlogger: log,
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

        let mut cuda_dst = match img_layout_src.channels {
            3 => CudaImage::<u8>::new(width, height, ColorType::Rgb8),
            _ => Err(CudaError::UnknownError),
        }
            .unwrap();
        let _res = resize(&cuda_src, &mut cuda_dst).unwrap();
        (RgbImage::try_from(&cuda_dst).unwrap(), ms())
    }
}
