use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView};
use npp_rs::image::CudaImage;
use std::path::Path;
use std::rc::Rc;
use std::result::Result;

macro_rules! img_op {
    ($I:expr) => {
        |img| $I.execute(img);
    };
}

trait image_op<T, E> {
    fn execute(&self, img: Rc<T>) -> Result<Rc<T>, E>;
}

struct rust_resize_op {
    width: u32,
    height: u32,
}

impl rust_resize_op {
    fn new(w: u32, h: u32) -> rust_resize_op {
        rust_resize_op {
            width: w,
            height: h,
        }
    }
}

impl image_op<DynamicImage, String> for rust_resize_op {
    fn execute(&self, img: Rc<DynamicImage>) -> Result<Rc<DynamicImage>, String> {
        Ok(Rc::new(img.resize(
            self.width,
            self.height,
            FilterType::Nearest,
        )))
    }
}

struct cuda_resize_op {
    width: u32,
    height: u32,
    img: CudaImage<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::{assert_eq, assert_ne};

    #[test]
    fn test_rustops() {
        let resize1 = img_op!(rust_resize_op::new(1024, 768));

        let resize2 = img_op!(rust_resize_op::new(800, 600));

        let resize3 = img_op!(rust_resize_op::new(640, 480));

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
}
