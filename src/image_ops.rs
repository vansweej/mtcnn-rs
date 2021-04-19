use image::imageops::FilterType;
use image::{ColorType, DynamicImage};
use npp_rs::image::CudaImage;
use npp_rs::imageops::resize;
use rustacuda::error::CudaError;
use std::cell::RefCell;
use std::rc::Rc;
use std::result::Result;

#[macro_use]
macro_rules! img_op {
    ($I:expr) => {
        |img| $I.execute(img);
    };
}

trait ImageOp<T, E> {
    fn execute(&self, img: Rc<T>) -> Result<Rc<T>, E>;
}

struct RustResizeOp {
    width: u32,
    height: u32,
}

impl RustResizeOp {
    fn new(w: u32, h: u32) -> RustResizeOp {
        RustResizeOp {
            width: w,
            height: h,
        }
    }
}

impl ImageOp<DynamicImage, String> for RustResizeOp {
    fn execute(&self, img: Rc<DynamicImage>) -> Result<Rc<DynamicImage>, String> {
        Ok(Rc::new(img.resize(
            self.width,
            self.height,
            FilterType::Nearest,
        )))
    }
}

struct CudaResizeOp {
    width: u32,
    height: u32,
    image: Option<Rc<RefCell<CudaImage<u8>>>>,
}

impl CudaResizeOp {
    fn new(w: u32, h: u32) -> CudaResizeOp {
        CudaResizeOp {
            width: w,
            height: h,
            image: Some(Rc::new(RefCell::new(
                CudaImage::<u8>::new(w, h, ColorType::Rgb8).unwrap(),
            ))),
        }
    }
}

impl ImageOp<RefCell<CudaImage<u8>>, CudaError> for CudaResizeOp {
    fn execute(
        &self,
        img: Rc<RefCell<CudaImage<u8>>>,
    ) -> Result<Rc<RefCell<CudaImage<u8>>>, CudaError> {
        let im = self.image.as_ref().unwrap();
        let res = resize(&img.borrow(), &mut im.borrow_mut());
        match res {
            Ok(_) => Ok(Rc::clone(&im)),
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;
    use pretty_assertions::{assert_eq, assert_ne};
    use rustacuda::prelude::*;
    use std::convert::TryFrom;

    #[test]
    fn test_rust_resize_op() {
        let resize1 = img_op!(RustResizeOp::new(1024, 768));

        let resize2 = img_op!(RustResizeOp::new(800, 600));

        let resize3 = img_op!(RustResizeOp::new(640, 480));

        let img1 = Rc::new(image::open("test_resources/2020-11-21-144033.jpg").unwrap());

        let res = resize1(Rc::clone(&img1));
        let res = resize1(Rc::clone(&img1));

        assert_eq!(res.is_ok(), true);

        let img2 = Rc::new(image::open("test_resources/DSC_0003.JPG").unwrap());
        let res2 = resize1(Rc::clone(&img2))
            .and_then(resize2)
            .and_then(resize3);
        assert_eq!(res2.is_ok(), true);
    }

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

        let res1 = resize1(Rc::clone(&cuda_src1))
            .and_then(resize2)
            .and_then(resize3);

        assert_eq!(res1.is_ok(), true);

        let o = res1.unwrap();
        let r = Rc::try_unwrap(o);

        assert_eq!(r.is_ok(), true);

        let s = r.unwrap();
        let result_img = RgbImage::try_from(&s.into_inner()).unwrap();

        result_img.save("/tmp/test1.png");

        let img2 = image::open("test_resources/DSC_0003.JPG").unwrap();

        let cuda_src2 = Rc::new(RefCell::new(
            CudaImage::try_from(img2.as_rgb8().unwrap()).unwrap(),
        ));

        let res2 = resize1(Rc::clone(&cuda_src2))
            .and_then(resize2)
            .and_then(resize3);

        assert_eq!(res2.is_ok(), true);

        let o2 = res2.unwrap();
        let r2 = Rc::try_unwrap(o2);

        assert_eq!(r2.is_ok(), true);

        let s2 = r2.unwrap();
        let result_img2 = RgbImage::try_from(&s2.into_inner()).unwrap();

        result_img2.save("/tmp/test2.png");
    }
}
