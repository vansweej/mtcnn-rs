use crate::image_ops::image_ops::*;
use image::imageops::FilterType;
use image::DynamicImage;
use std::cell::RefCell;
use std::rc::Rc;
use std::result::Result;

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

// impl ImageOp<DynamicImage, String> for RustResizeOp {
//     fn execute(&self, img: Rc<RefCell<DynamicImage>>) -> Result<Rc<RefCell<DynamicImage>>, String> {
//         Ok(Rc::new(RefCell::new(img.borrow().resize(
//             self.width,
//             self.height,
//             FilterType::Nearest,
//         ))))
//     }
// }

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::img_op;
//     use pretty_assertions::{assert_eq, assert_ne};

//     #[test]
//     fn test_rust_resize_op() {
//         let resize1 = img_op!(RustResizeOp::new(1024, 768));

//         let resize2 = img_op!(RustResizeOp::new(800, 600));

//         let resize3 = img_op!(RustResizeOp::new(640, 480));

//         let img1 = Rc::new(RefCell::new(
//             image::open("test_resources/2020-11-21-144033.jpg").unwrap(),
//         ));

//         let res = resize1(Rc::clone(&img1));
//         let res = resize1(Rc::clone(&img1));

//         assert_eq!(res.is_ok(), true);

//         let img2 = Rc::new(RefCell::new(
//             image::open("test_resources/DSC_0003.JPG").unwrap(),
//         ));
//         let res2 = resize1(Rc::clone(&img2))
//             .and_then(resize2)
//             .and_then(resize3);
//         assert_eq!(res2.is_ok(), true);
//     }
// }
