use rustacuda::error::CudaError;

fn into_inner(
    wrapped_img: std::rc::Rc<std::cell::RefCell<npp_rs::image::CudaImage<u8>>>,
) -> Result<npp_rs::image::CudaImage<u8>, CudaError> {
    match std::rc::Rc::try_unwrap(wrapped_img) {
        Ok(r) => Ok(r.into_inner()),
        Err(_) => Err(CudaError::UnknownError),
    }
}

pub mod resize_ops;

pub mod factory;
