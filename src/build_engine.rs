use std::fs::File;
use std::io::Read;
use tensorrt_rs::engine::Engine;
use tensorrt_rs::runtime::{Logger, Runtime};

pub(crate) fn build_engine(engine_file: &str, logger: &Logger) -> Engine {
    let runtime = Runtime::new(&logger);
    let mut f = File::open(engine_file).unwrap();
    let mut buffer = Vec::new();
    f.read_to_end(&mut buffer).unwrap();
    drop(f);
    runtime.deserialize_cuda_engine(buffer)
}