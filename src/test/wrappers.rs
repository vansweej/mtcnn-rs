#[cfg(test)]
pub mod tests {
    use std::panic;
    use npp_rs::cuda::initialize_cuda_device;

    pub fn run_cuda_test<T>(test_f: T) -> ()
        where
            T: FnOnce() -> () + panic::UnwindSafe,
    {
        let result = panic::catch_unwind(|| {
            let _ctx = initialize_cuda_device();
            test_f()
        });

        assert!(result.is_ok())
    }
}