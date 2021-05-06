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
    use crate::image_ops::*;
    use crate::img_op;
    use image::RgbImage;
    use npp_rs::cuda::initialize_cuda_device;
    use npp_rs::image::CudaImage;
    use pretty_assertions::assert_eq;
    use std::cell::RefCell;
    use std::convert::TryFrom;
    use std::rc::Rc;

    #[test]
    fn test_abstract_factory() {
        let _ctx = initialize_cuda_device().unwrap();

        let factory = AbstractImageOpFactory::create_imageop_factory(ImageOpsType::Cuda);

        let op1 = img_op!(factory.create_resize_imageop(1024, 768));

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
