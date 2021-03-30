use image::DynamicImage;

pub trait ImageOps {
    fn resize(&self, image: &DynamicImage, width: usize, height: usize) -> DynamicImage;
}

pub struct CudaOps {}

impl ImageOps for CudaOps {
    fn resize(&self, image: &DynamicImage, width: usize, height: usize) -> DynamicImage {
        unimplemented!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustacuda::prelude::*;

    #[test]
    fn test_cuda_resize() {
        let image_ops = CudaOps {};
        rustacuda::init(rustacuda::CudaFlags::empty()).unwrap();
        let device = Device::get_device(0).unwrap();
        let _ctx =
            Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
                .unwrap();

        let img_src = image::open("test_resources/DSC_0003.JPG").unwrap();

        let img_dst = image_ops.resize(&img_src, 640, 480);
    }
}
