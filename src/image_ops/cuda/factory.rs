use crate::image_ops::cuda::resize_ops::*;
use crate::image_ops::image_ops::*;
use npp_rs::image::CudaImage;
use rustacuda::error::CudaError;

pub type CudaFactory = Box<dyn ImageOpFactory<CudaImage<u8>, CudaError>>;
pub type CudaImageOp = Box<dyn ImageOp<CudaImage<u8>, CudaError>>;

pub struct CudaImageOpFactory {}

impl ImageResizeOpFactory<CudaImage<u8>, CudaError> for CudaImageOpFactory {
    fn create_resize_imageop(&self, w: u32, h: u32) -> CudaImageOp {
        Box::new(CudaResizeOp::new(w, h))
    }
}

impl ImageOpFactory<CudaImage<u8>, CudaError> for CudaImageOpFactory {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_ops::*;
    use crate::img_op;
    use image::RgbImage;
    use npp_rs::cuda::initialize_cuda_device;
    use pretty_assertions::assert_eq;
    use std::cell::RefCell;
    use std::convert::TryFrom;
    use std::rc::Rc;

    #[test]
    fn test_cuda_factory() {
        let _ctx = initialize_cuda_device().unwrap();

        let test = CudaImageOpFactory {};

        let op1 = img_op!(test.create_resize_imageop(1024, 768));

        let img1 = image::open("test_resources/2020-11-21-144033.jpg").unwrap();

        let cuda_src1 = Rc::new(RefCell::new(
            CudaImage::try_from(img1.as_rgb8().unwrap()).unwrap(),
        ));

        let res1 = op1(Rc::clone(&cuda_src1)).and_then(cuda::into_inner);

        let result_img1 = res1.map(|c| RgbImage::try_from(&c)).unwrap().unwrap();

        let dim1 = result_img1.dimensions();
        assert_eq!(dim1.0, 1024);
        assert_eq!(dim1.1, 768);
    }
}
