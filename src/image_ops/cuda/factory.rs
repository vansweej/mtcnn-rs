use crate::image_ops::cuda::resize_ops::*;
use crate::image_ops::image_ops::*;
use npp_rs::image::CudaImage;
use rustacuda::error::CudaError;

struct CudaImageOpFactory {}

impl ImageResizeOpFactory<CudaImage<u8>, CudaError> for CudaImageOpFactory {
    type Output = CudaResizeOp;

    fn create_resize_imageop(w: u32, h: u32) -> Self::Output {
        CudaResizeOp::new(w, h)
    }
}

impl ImageOpFactory for CudaImageOpFactory where
    CudaImageOpFactory: ImageResizeOpFactory<CudaImage<u8>, CudaError>
{
}
