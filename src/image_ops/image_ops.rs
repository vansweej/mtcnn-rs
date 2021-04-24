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

pub enum ImageOpType {
    Rust,
    Cuda,
}

pub trait AbstractImageOpFactory<T> {
    fn create_imageop_factory(imageop_type: ImageOpType) -> Result<T, String>;
}

struct ConcreteAbstractImageOpFactory {}

impl<T> AbstractImageOpFactory<T> for ConcreteAbstractImageOpFactory
where
    T: ImageOpFactory,
{
    fn create_imageop_factory(imageop_type: ImageOpType) -> Result<T, String> {
        Err("nothing else matters".to_string())
    }
}

pub trait ImageOpFactory {}

pub trait ImageResizeOpFactory<T, E> {
    type Output: ImageOp<T, E>;

    fn create_resize_imageop(w: u32, h: u32) -> Self::Output;
}
