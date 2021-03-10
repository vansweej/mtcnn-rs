extern crate tensorrt_rs;
use crate::helper::*;

use image::imageops::*;
use image::*;
use itertools::*;
use ndarray::prelude::*;
use ndarray::s;
use ndarray_image;
use std::fs::File;
use std::io::Read;
use tensorrt_rs::context::ExecuteInput;
use tensorrt_rs::engine::Engine;
use tensorrt_rs::runtime::*;

pub struct TrtRnet {
    data_dims: (u32, u32, u32),
    prob1_dims: (u32, u32, u32),
    boxes_dims: (u32, u32, u32),
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
            data_dims: (3, 24, 24),
            prob1_dims: (2, 1, 1),
            boxes_dims: (4, 1, 1),
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

        let boxes_1x1 = boxes
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(i, v)| {
                vec![
                    f32::trunc(v[0] + ww[i] * 0.5 - mm[i] * 0.5),
                    f32::trunc(v[1] + hh[i] * 0.5 - mm[i] * 0.5),
                    f32::trunc((v[0] + ww[i] * 0.5 - mm[i] * 0.5) + mm[i] - 1.0),
                    f32::trunc((v[1] + hh[i] * 0.5 - mm[i] * 0.5) + mm[i] - 1.0),
                    v[4],
                ]
            })
            .concat()
            .into_iter()
            .collect::<Array<_, _>>()
            .into_shape((boxes.dim().0, boxes.dim().1))
            .unwrap();

        println!("{:?}", boxes_1x1);

        boxes_1x1
    }

    fn generate_rnet_bboxes(
        conf: &Array1<f32>,
        reg: &Array2<f32>,
        pboxes: &Array2<f32>,
        t: f32,
    ) -> Array2<f32> {
        let boxes = pboxes
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(i, v)| [v[0], v[1], v[2], v[3], conf[i]])
            .filter(|v| v[4] > t)
            .collect::<Vec<_>>();

        let reg_f = reg
            .axis_iter(Axis(0))
            .enumerate()
            .filter(|(i, v)| conf[*i] > t)
            .map(|(i, v)| v)
            .collect::<Vec<_>>();

        let mut s_boxes_t: Vec<_> = vec![];
        for (i, ibox) in boxes.iter().enumerate() {
            let reg = reg_f[i];
            let s_box = [
                ibox[0] + ((ibox[2] - ibox[0] + 1.0) * reg[[0]]),
                ibox[1] + ((ibox[3] - ibox[1] + 1.0) * reg[[1]]),
                ibox[2] + ((ibox[2] - ibox[0] + 1.0) * reg[[2]]),
                ibox[3] + ((ibox[3] - ibox[1] + 1.0) * reg[[3]]),
                ibox[4],
            ];
            s_boxes_t.push(s_box);
        }

        let s_boxes = Array::from_shape_vec(
            (boxes.len(), 5),
            s_boxes_t.iter().flatten().map(|v| *v).collect(),
        )
        .unwrap();

        s_boxes
    }

    fn execute(&self, mut rnet_input: &mut Array4<f32>) -> (Array4<f32>, Array4<f32>) {
        let batch_size = rnet_input.dim().0;
        let im_input = ExecuteInput::Float(&mut rnet_input);

        let mut prob1 = ndarray::Array4::<f32>::zeros((batch_size, 2, 1, 1));
        let mut boxes = ndarray::Array4::<f32>::zeros((batch_size, 4, 1, 1));
        let pnet_output = vec![
            ExecuteInput::Float(&mut boxes),
            ExecuteInput::Float(&mut prob1),
        ];

        let context = self.rnet_engine.create_execution_context();
        context
            .execute(im_input, pnet_output, Some(batch_size as i32))
            .unwrap();

        (prob1, boxes)
    }

    pub fn detect(
        &self,
        image: &DynamicImage,
        boxes: &Array2<f32>,
        max_batch: u32,
        threshold: f32,
    ) -> Array2<f32> {
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
            let flipped_img = flip_horizontal(&resized_img);
            let rotated_img = rotate270(&flipped_img);
            let img = DynamicImage::ImageRgba8(rotated_img).to_rgb8();
            crops.push(img);
        }

        let mut conv_crops: Vec<_> = vec![];
        for crop in crops.iter() {
            let im_array: ndarray_image::NdColor = ndarray_image::NdImage(crop).into();
            let pre_processed = im_array
                .permuted_axes([2, 0, 1])
                .map(|&x| ((x as f32) - PIXEL_MEAN) * PIXEL_SCALE);
            conv_crops.push(pre_processed);
        }
        let mut pre_processed = Array::from_shape_vec(
            (conv_crops.len(), 3, 24, 24),
            conv_crops.iter().flatten().map(|v| *v).collect::<Vec<_>>(),
        )
        .unwrap();

        let (prob1, boxs) = self.execute(&mut pre_processed);
        let pp = prob1.slice(s![.., 1, 0, 0]);
        let cc = boxs.slice(s![.., .., 0, 0]);

        let rnet_boxes =
            TrtRnet::generate_rnet_bboxes(&pp.to_owned(), &cc.to_owned(), &boxes, threshold);
        // if boxes.shape[0] == 0:
        //     return boxes

        let total_pick = nms(&rnet_boxes, 0.7, SuppressionType::Union);
        let indexed_rnet_boxes_t = total_pick
            .iter()
            .map(|v| rnet_boxes.index_axis(Axis(0), *v))
            .collect::<Vec<_>>();
        let indexed_rnet_boxes = Array::from_shape_vec(
            (total_pick.len(), 5),
            indexed_rnet_boxes_t
                .iter()
                .flatten()
                .map(|v| *v)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let dets = clip_dets(&indexed_rnet_boxes, image.width(), image.height());
        dets
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

    #[test]
    fn test_generate_rnet_bboxes() {
        let np_pp: Array1<f32> = read_npy("test_resources/pp_rnet.npy").unwrap();
        let np_cc: Array2<f32> = read_npy("test_resources/cc_rnet.npy").unwrap();
        let np_boxes_1x1: Array2<f32> = read_npy("test_resources/boxes_1x1.npy").unwrap();
        let np_bboxes: Array2<f32> = read_npy("test_resources/boxes_rnet.npy").unwrap();

        let bboxes = TrtRnet::generate_rnet_bboxes(&np_pp, &np_cc, &np_boxes_1x1, 0.7);

        assert_eq!(np_bboxes, bboxes);
    }
}
