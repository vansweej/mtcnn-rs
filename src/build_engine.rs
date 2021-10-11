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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tensorrt_rs::builder::{Builder, NetworkBuildFlags};
    use tensorrt_rs::data_size::GB;
    use tensorrt_rs::dims::Dims4;
    use tensorrt_rs::onnx::{OnnxFile, OnnxParser};

    #[test]
    fn test_onnx_parser() {
        let batch_size = 1;
        let workspace_size = 1 * GB;
        let file = OnnxFile::new(&PathBuf::from("./test_resources/pnet.onnx")).unwrap();

        let logger = Logger::new();

        let builder = Builder::new(&logger);

        let network = builder.create_network_v2(NetworkBuildFlags::EXPLICIT_BATCH);
        let verbosity = 7;

        builder.set_max_batch_size(batch_size);
        builder.set_max_workspace_size(workspace_size);

        let parser = OnnxParser::new(&network, &logger);
        parser.parse_from_file(&file, verbosity).unwrap();

        println!("{:?}",network.get_nb_inputs());
        println!("{:?}",network.get_nb_outputs());

        let dim = Dims4::new(batch_size, 710, 384, 3);
        network.get_input(0).set_dimensions(dim);

        let _engine = builder.build_cuda_engine(&network);
    }
}