use crate::image_ops::image_ops::*;
use image::ColorType;
use npp_rs::image::CudaImage;
use npp_rs::imageops::resize;
use rustacuda::error::CudaError;
use std::cell::RefCell;
use std::rc::Rc;
use std::result::Result;

pub struct CudaResizeOp<T> {
    image: Rc<RefCell<T>>,
}

impl CudaResizeOp<CudaImage<u8>> {
    pub fn new(w: u32, h: u32) -> CudaResizeOp<CudaImage<u8>> {
        CudaResizeOp {
            image: Rc::new(RefCell::new(
                CudaImage::<u8>::new(w, h, ColorType::Rgb8).unwrap(),
            )),
        }
    }
}

impl ImageOp<CudaImage<u8>, CudaError> for CudaResizeOp<CudaImage<u8>> {
    fn execute(
        &self,
        img: Rc<RefCell<CudaImage<u8>>>,
    ) -> Result<Rc<RefCell<CudaImage<u8>>>, CudaError> {
        let res = resize(&img.borrow(), &mut self.image.borrow_mut());
        match res {
            Ok(_) => Ok(Rc::clone(&self.image)),
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_ops::*;
    use crate::img_op;
    use image::RgbImage;
    use pretty_assertions::{assert_eq, assert_ne};
    use rustacuda::prelude::*;
    use std::convert::TryFrom;

    #[test]
    fn test_cuda_resize_op() {
        rustacuda::init(rustacuda::CudaFlags::empty()).unwrap();
        let device = Device::get_device(0).unwrap();
        let _ctx =
            Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
                .unwrap();

        let resize1 = img_op!(CudaResizeOp::new(1024, 768));

        let resize2 = img_op!(CudaResizeOp::new(800, 600));

        let resize3 = img_op!(CudaResizeOp::new(640, 480));

        let img1 = image::open("test_resources/2020-11-21-144033.jpg").unwrap();

        let cuda_src1 = Rc::new(RefCell::new(
            CudaImage::try_from(img1.as_rgb8().unwrap()).unwrap(),
        ));

        let res1 = resize1(Rc::clone(&cuda_src1)).and_then(cuda::into_inner);

        let result_img1 = res1.map(|c| RgbImage::try_from(&c)).unwrap().unwrap();

        let dim1 = result_img1.dimensions();
        assert_eq!(dim1.0, 1024);
        assert_eq!(dim1.1, 768);

        let res2 = resize1(Rc::clone(&cuda_src1))
            .and_then(resize2)
            .and_then(resize3)
            .and_then(cuda::into_inner);

        let result_img2 = res2.map(|c| RgbImage::try_from(&c)).unwrap().unwrap();

        let dim2 = result_img2.dimensions();
        assert_eq!(dim2.0, 640);
        assert_eq!(dim2.1, 480);
    }
}
