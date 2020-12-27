extern crate tensorrt_rs;
use crate::helper::*;

use image::imageops::*;
use image::*;
use ndarray::prelude::*;
use ndarray::{s, stack};
use ndarray_image;
use tensorrt_rs::context::ExecuteInput;
use tensorrt_rs::engine::Engine;
use tensorrt_rs::runtime::*;
use std::fs::File;
use std::io::Read;

pub struct TrtRnet {
    rnet_engine: Engine,
}

impl TrtRnet {
    pub fn new(engine_file: &str, logger: &Logger) -> Result<TrtRnet, String> {
        let runtime = Runtime::new(&logger);
        let mut f = File::open(engine_file).unwrap();
        let mut pnet_buffer = Vec::new();
        f.read_to_end(&mut pnet_buffer).unwrap();
        drop(f);
        let engine = runtime.deserialize_cuda_engine(pnet_buffer);

        Ok(TrtRnet {
            rnet_engine: engine,
        })
    }

    fn convert_to_1x1(boxes: &Array2<f32>) -> Array2<f32> {
        let hh = boxes.axis_iter(Axis(0)).map(|v| v[3] - v[1] + 1.0).collect::<Vec<_>>();
        let ww = boxes.axis_iter(Axis(0)).map(|v| v[2] - v[0] + 1.0).collect::<Vec<_>>();
        let mm = hh.iter().zip(&ww).map(|v| v.0.max(*v.1)).collect::<Vec<_>>();
        
        let boxes_1x1_t = boxes.axis_iter(Axis(0)).enumerate()
        .map(|(i, v)| [ 
            f32::trunc(v[0] + ww[i] * 0.5 - mm[i] * 0.5), 
            f32::trunc(v[1] + hh[i] * 0.5 - mm[i] * 0.5), 
            f32::trunc((v[0] + ww[i] * 0.5 - mm[i] * 0.5) + mm[i] - 1.0), 
            f32::trunc((v[1] + hh[i] * 0.5 - mm[i] * 0.5) + mm[i] - 1.0), 
            v[4] ])
        .collect::<Vec<_>>();
        let boxes_1x1 = Array::from_shape_vec(
            (boxes.dim().0, boxes.dim().1),
            boxes_1x1_t.iter().flatten().map(|v| *v).collect::<Vec<_>>(),
        )
        .unwrap();
        boxes_1x1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_npy::read_npy;

    #[test]
    fn test_convert_to_1x1() {
        let boxes: Array2<f32> = read_npy("test_resources/dets.npy").unwrap();
        TrtRnet::convert_to_1x1(&boxes);
    }
}