use image::imageops::FilterType;
use image::{ColorType, DynamicImage};
use npp_rs::image::CudaImage;
use npp_rs::imageops::resize;
use rustacuda::error::CudaError;
use std::cell::RefCell;
use std::rc::Rc;
use std::result::Result;

#[macro_export]
macro_rules! img_op {
    ($I:expr) => {
        |img| $I.execute(img);
    };
}

pub trait ImageOp<T, E> {
    fn execute(&self, img: Rc<RefCell<T>>) -> Result<Rc<RefCell<T>>, E>;
}
