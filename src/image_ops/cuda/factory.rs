use crate::image_ops::cuda::resize_ops::*;
use crate::image_ops::image_ops::*;
use npp_rs::image::CudaImage;
use rustacuda::error::CudaError;

pub type CudaFactory = Box<dyn ImageOpFactory<CudaImage<u8>, CudaError>>;

pub struct CudaImageOpFactory {}

impl ImageResizeOpFactory<CudaImage<u8>, CudaError> for CudaImageOpFactory {
    fn create_resize_imageop(&self, w: u32, h: u32) -> Box<dyn ImageOp<CudaImage<u8>, CudaError>> {
        Box::new(CudaResizeOp::new(w, h))
    }
}

impl ImageOpFactory<CudaImage<u8>, CudaError> for CudaImageOpFactory {}

#[cfg(test)]
mod tests {
    use super::*;
    use rustacuda::prelude::*;

    #[test]
    fn test_cuda_factory() {
        rustacuda::init(rustacuda::CudaFlags::empty()).unwrap();
        let device = Device::get_device(0).unwrap();
        let _ctx =
            Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
                .unwrap();

        let test = CudaImageOpFactory {};

        let op1 = test.create_resize_imageop(1024, 768);
    }
}
