# MTCNN face detection in rust
This crate implements MTCNN facial and facial feature detection in the rust programming language.
It is based on the [python demo](https://github.com/jkjung-avt/tensorrt_demos) of this neural network.
Nvidia's Tensorrt rust wrappings are provided by the [repo of Mason Stallmo](https://github.com/mstallmo/tensorrt-rs). Currently the Cargo.toml file links to [this local version](https://github.com/vansweej/tensorrt-rs), since this library is extended to fit the needs of this crate.
Neural network files are taken from the [python demo](https://github.com/jkjung-avt/tensorrt_demos) and converted to an optimized tensorrt dump (which matters for the hardware it was ran on). 
Current development state:
* Pnet functionality is written, needs to be integrated
* Rnet not implemented yet
* Onet not implemented yet