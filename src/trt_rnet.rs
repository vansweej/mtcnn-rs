extern crate tensorrt_rs;
use crate::helper::*;

use image::imageops::*;
use image::*;
use ndarray::prelude::*;
use ndarray::{s, stack};
use ndarray_image;
use std::fs::File;
use std::io::Read;
use tensorrt_rs::context::ExecuteInput;
use tensorrt_rs::engine::Engine;
use tensorrt_rs::runtime::*;

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
        let hh = boxes
            .axis_iter(Axis(0))
            .map(|v| v[3] - v[1] + 1.0)
            .collect::<Vec<_>>();
        let ww = boxes
            .axis_iter(Axis(0))
            .map(|v| v[2] - v[0] + 1.0)
            .collect::<Vec<_>>();
        let mm = hh
            .iter()
            .zip(&ww)
            .map(|v| v.0.max(*v.1))
            .collect::<Vec<_>>();

        let boxes_1x1_t = boxes
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(i, v)| {
                [
                    f32::trunc(v[0] + ww[i] * 0.5 - mm[i] * 0.5),
                    f32::trunc(v[1] + hh[i] * 0.5 - mm[i] * 0.5),
                    f32::trunc((v[0] + ww[i] * 0.5 - mm[i] * 0.5) + mm[i] - 1.0),
                    f32::trunc((v[1] + hh[i] * 0.5 - mm[i] * 0.5) + mm[i] - 1.0),
                    v[4],
                ]
            })
            .collect::<Vec<_>>();
        let boxes_1x1 = Array::from_shape_vec(
            (boxes.dim().0, boxes.dim().1),
            boxes_1x1_t.iter().flatten().map(|v| *v).collect::<Vec<_>>(),
        )
        .unwrap();
        boxes_1x1
    }

    pub fn detect(
        &self,
        image: &DynamicImage,
        boxes: &Array2<f32>,
        max_batch: u32,
        threshold: f32,
    ) {
        const PIXEL_MEAN: f32 = 127.5;
        const PIXEL_SCALE: f32 = 0.0078125;
        // if max_batch > 256:
        //     raise ValueError('Bad max_batch: %d' % max_batch)
        // boxes = boxes[:max_batch]  # assuming boxes are sorted by score
        // if boxes.shape[0] == 0:
        //     return boxes

        //display_image(image);
        let mut crops: Vec<_> = vec![];
        for (i, det) in boxes.axis_iter(Axis(0)).enumerate() {
            let w = det[2] - det[0];
            let h = det[3] - det[1];

            let cropped_img = crop_imm(image, det[0] as u32, det[1] as u32, w as u32, h as u32);
            let resized_img = resize(&cropped_img, 24, 24, FilterType::Nearest);
            let rotated_img = rotate270(&resized_img);
            let img = DynamicImage::ImageRgba8(rotated_img).to_rgb8();
            crops.push(img);
        }

        let mut conv_crops: Vec<_> = vec![];
        for crop in crops.iter() {
            let mut im_array: ndarray_image::NdColor = ndarray_image::NdImage(crop).into();
            im_array.swap_axes(0, 2);
            im_array.swap_axes(1, 2);
            let pre_processed = im_array.map(|&x| {
                if x == 0 {
                    0.0
                } else {
                    ((x as f32) - PIXEL_MEAN) * PIXEL_SCALE
                }
            });
            conv_crops.push(pre_processed);
        }
        let x = Array::from_shape_vec(
            (conv_crops.len(), 3, 24, 24),
            conv_crops.iter().flatten().collect::<Vec<_>>(),
        )
        .unwrap();
        println!("{:?}", x.dim());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_npy::read_npy;

    #[test]
    fn test_convert_to_1x1() {
        let boxes: Array2<f32> = read_npy("test_resources/dets.npy").unwrap();
        let np_boxes_1x1: Array2<f32> = read_npy("test_resources/boxes_1x1.npy").unwrap();
        let boxes_1x1 = TrtRnet::convert_to_1x1(&boxes);

        assert_eq!(np_boxes_1x1, boxes_1x1);
    }

    #[test]
    fn test_detect() {
        let logger = Logger::new();
        let rnet = TrtRnet::new("./test_resources/det2.engine", &logger).unwrap();

        let rnet_img = image::open("test_resources/rnet_img.jpg").unwrap();
        let np_boxes_1x1: Array2<f32> = read_npy("test_resources/boxes_1x1.npy").unwrap();

        rnet.detect(&rnet_img, &np_boxes_1x1, 256, 0.7);
    }
}
