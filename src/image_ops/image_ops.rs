use crate::image_ops::cuda::factory::*;
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

pub enum ImageOpsType {
    Cuda,
}

pub enum FactoryType {
    CudaType(CudaFactory),
}

trait CudaFactoryType {
    fn create_resize_imageop(&self, w: u32, h: u32) -> CudaImageOp;
}

impl CudaFactoryType for FactoryType {
    fn create_resize_imageop(&self, w: u32, h: u32) -> CudaImageOp {
        let out = if let FactoryType::CudaType(cuda_factory) = self {
            Some(cuda_factory.create_resize_imageop(w, h))
        } else {
            None
        };
        out.unwrap()
    }
}

struct AbstractImageOpFactory {}

impl AbstractImageOpFactory {
    fn create_imageop_factory(image_op_type: ImageOpsType) -> FactoryType {
        match image_op_type {
            ImageOpsType::Cuda => FactoryType::CudaType(Box::new(CudaImageOpFactory {})),
        }
    }
}

pub trait ImageOpFactory<T, E>: ImageResizeOpFactory<T, E> {}

pub trait ImageResizeOpFactory<T, E> {
    fn create_resize_imageop(&self, w: u32, h: u32) -> Box<dyn ImageOp<T, E>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustacuda::prelude::*;

    #[test]
    fn test_abstract_factory() {
        rustacuda::init(rustacuda::CudaFlags::empty()).unwrap();
        let device = Device::get_device(0).unwrap();
        let _ctx =
            Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
                .unwrap();

        let factory = AbstractImageOpFactory::create_imageop_factory(ImageOpsType::Cuda);

        let resize_op = factory.create_resize_imageop(1024, 768);
    }
}
