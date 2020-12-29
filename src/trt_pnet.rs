extern crate tensorrt_rs;
use crate::helper::*;

use image::imageops::*;
use image::*;
use ndarray::prelude::*;
use ndarray::{s, stack};
use ndarray_image;
use std::cmp;
use std::fs::File;
use std::io::Read;
use tensorrt_rs::context::ExecuteInput;
use tensorrt_rs::engine::Engine;
use tensorrt_rs::runtime::*;

pub struct TrtPnet {
    data_dims: (u32, u32, u32),
    prob1_dims: (u32, u32, u32),
    boxes_dims: (u32, u32, u32),
    pnet_engine: Engine,
}

enum SuppressionType {
    Union,
    Min,
}

impl TrtPnet {
    pub fn new(engine_file: &str, logger: &Logger) -> Result<TrtPnet, String> {
        let runtime = Runtime::new(&logger);
        let mut f = File::open(engine_file).unwrap();
        let mut pnet_buffer = Vec::new();
        f.read_to_end(&mut pnet_buffer).unwrap();
        drop(f);
        let engine = runtime.deserialize_cuda_engine(pnet_buffer);

        Ok(TrtPnet {
            data_dims: (3, 710, 384),
            prob1_dims: (2, 350, 187),
            boxes_dims: (4, 350, 187),
            pnet_engine: engine,
        })
    }

    fn get_scales(width: u32, height: u32, minsize: u32, factor: f64) -> Vec<f64> {
        let mut m = 12.0 / minsize as f64;
        let mut minl = cmp::min(width, height) as f64 * m;

        // create scale pyramid
        let mut scales = Vec::new();
        while minl >= 12.0 {
            scales.push(m);
            m *= factor;
            minl *= factor;
        }

        // if len(scales) > self.max_n_scales:  # probably won't happen...
        //     raise ValueError('Too many scales, try increasing minsize '
        //                      'or decreasing factor.')

        scales
    }

    fn execute(&self, mut pnet_input: &mut Array3<f32>) -> (Array3<f32>, Array3<f32>) {
        let im_input = ExecuteInput::Float(&mut pnet_input);

        let mut prob1 = ndarray::Array3::<f32>::zeros((2, 350, 187));
        let mut boxes = ndarray::Array3::<f32>::zeros((4, 350, 187));
        let pnet_output = vec![
            ExecuteInput::Float(&mut boxes),
            ExecuteInput::Float(&mut prob1),
        ];

        let context = self.pnet_engine.create_execution_context();
        context.execute(im_input, pnet_output, None).unwrap();

        (prob1, boxes)
    }

    fn generate_pnet_bboxes(
        conf: &Array2<f32>,
        reg: &Array3<f32>,
        scale: &f64,
        t: f32,
    ) -> Array2<f32> {
        let conf_t = conf.t();

        let dx1 = reg.slice(s![0, .., ..]).reversed_axes();
        let dy1 = reg.slice(s![1, .., ..]).reversed_axes();
        let dx2 = reg.slice(s![2, .., ..]).reversed_axes();
        let dy2 = reg.slice(s![3, .., ..]).reversed_axes();

        // as shown in:
        // https://github.com/rust-ndarray/ndarray/issues/466
        let indexes = conf_t
            .indexed_iter()
            .filter_map(|(index, &item)| if item >= t { Some(index) } else { None })
            .collect::<Vec<_>>();

        let score = Array::from_shape_vec(
            (indexes.len(), 1),
            indexes.iter().map(|(x, y)| conf_t[[*x, *y]]).collect(),
        )
        .unwrap();

        let reg = indexes
            .iter()
            .map(|(x, y)| [dx1[[*x, *y]], dy1[[*x, *y]], dx2[[*x, *y]], dy2[[*x, *y]]])
            .collect::<Vec<_>>()
            .iter()
            .flatten()
            .map(|v| v * 12.0)
            .collect::<Array<_, _>>()
            .into_shape((indexes.len(), 4))
            .unwrap();

        let topleft_t: Vec<_> = indexes
            .iter()
            .map(|(x, y)| [*x as f32, *y as f32])
            .collect();
        let topleft = Array::from_shape_vec(
            (indexes.len(), 2),
            topleft_t.iter().flatten().map(|v| v * 2.0).collect(),
        )
        .unwrap();
        let bottomright = topleft.clone() + array![11.0, 11.0];
        let boxes = (stack![Axis(1), topleft, bottomright] + reg) / *scale as f32;
        stack!(Axis(1), boxes, score)
    }

    fn nms(boxes: &Array2<f32>, threshold: f32, s_type: SuppressionType) -> Vec<usize> {
        let areas = boxes
            .axis_iter(Axis(0))
            .map(|x| (x[2] - x[0] + 1.0) * (x[3] - x[1] + 1.0))
            .collect::<Array<_, _>>();

        let mut sorted_idx = boxes
            .slice(s![.., 4])
            .indexed_iter()
            .map(|(index, &item)| index)
            .collect::<Vec<usize>>();
        sorted_idx.sort_unstable_by(|a, b| boxes[[*a, 4]].partial_cmp(&boxes[[*b, 4]]).unwrap());

        let xx1 = boxes.slice(s![.., 0]);
        let yy1 = boxes.slice(s![.., 1]);
        let xx2 = boxes.slice(s![.., 2]);
        let yy2 = boxes.slice(s![.., 3]);

        let mut pick: Vec<usize> = vec![];
        loop {
            if sorted_idx.len() <= 0 {
                break;
            }

            let (begin, last) = sorted_idx.split_at(sorted_idx.len() - 1);

            let tx1 = maximum(&xx1[last[0]], map_index(begin, &xx1));
            let ty1 = maximum(&yy1[last[0]], map_index(begin, &yy1));
            let tx2 = minimum(&xx2[last[0]], map_index(begin, &xx2));
            let ty2 = minimum(&yy2[last[0]], map_index(begin, &yy2));

            let tw = maximum(&0.0, tx2 - tx1 + 1.0);
            let th = maximum(&0.0, ty2 - ty1 + 1.0);
            let inter = tw * th;

            let iou = match s_type {
                SuppressionType::Min => {
                    inter / minimum(&areas[last[0]], map_index(begin, &areas.view()))
                }
                SuppressionType::Union => {
                    inter.clone() / (areas[last[0]] + map_index(begin, &areas.view()) - inter)
                }
            };
            pick.push(last[0]);
            sorted_idx = sorted_idx
                .iter()
                .enumerate()
                .filter_map(|(index, &item)| {
                    if (index < iou.len()) && (iou[index] <= threshold) {
                        Some(item)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
        }

        pick
    }

    fn clip_dets(in_dets: &Array2<f32>, img_w: u32, img_h: u32) -> Array2<f32> {
        let out_dets_t = in_dets
            .axis_iter(Axis(0))
            .map(|v| {
                [
                    clamp(f32::trunc(v[0]), 0.0, (img_w - 1) as f32),
                    clamp(f32::trunc(v[1]), 0.0, (img_h - 1) as f32),
                    clamp(f32::trunc(v[2]), 0.0, (img_w - 1) as f32),
                    clamp(f32::trunc(v[3]), 0.0, (img_h - 1) as f32),
                    v[4],
                ]
            })
            .collect::<Vec<_>>();

        let out_dets = Array::from_shape_vec(
            (in_dets.dim().0, in_dets.dim().1),
            out_dets_t.iter().flatten().map(|v| *v).collect::<Vec<_>>(),
        )
        .unwrap();

        out_dets
    }

    fn extract_outputs(
        width: u32,
        height: u32,
        scales: &Vec<f64>,
        prob1: &Array3<f32>,
        boxes: &Array3<f32>,
        threshold: f32,
    ) -> Array2<f32> {
        const OUTPUT_H_OFFSETS: [i32; 9] = [0, 108, 185, 239, 278, 305, 324, 338, 348];

        let mut total_boxes = Array2::<f32>::zeros((0, 5));

        for (i, scale) in scales.iter().enumerate() {
            let h_offset = OUTPUT_H_OFFSETS[i];
            let h = ((height as f64 * scale) as i32 - 12) / 2 + 1;
            let w = ((width as f64 * scale) as i32 - 12) / 2 + 1;
            let pp = prob1.slice(s![1, h_offset..(h_offset + h), ..w]);
            let cc = boxes.slice(s![.., h_offset..(h_offset + h), ..w]);
            let boxes =
                TrtPnet::generate_pnet_bboxes(&pp.to_owned(), &cc.to_owned(), scale, threshold);

            if boxes.shape()[0] > 0 {
                let pick = TrtPnet::nms(&boxes, 0.5, SuppressionType::Union);
                //println!("{:?}", boxes);
                if pick.len() > 0 {
                    let boxes_slice_t = pick
                        .iter()
                        .map(|v| {
                            [
                                boxes[[*v, 0]],
                                boxes[[*v, 1]],
                                boxes[[*v, 2]],
                                boxes[[*v, 3]],
                                boxes[[*v, 4]],
                            ]
                        })
                        .collect::<Vec<_>>();
                    let boxes_slice = Array::from_shape_vec(
                        (pick.len(), 5),
                        boxes_slice_t.iter().flatten().map(|v| *v).collect(),
                    )
                    .unwrap();
                    if boxes_slice.len() > 0 {
                        total_boxes = stack![Axis(0), total_boxes, boxes_slice];
                    }
                }
            }
        }
        if total_boxes.shape()[0] == 0 {
            total_boxes
        } else {
            let total_pick = TrtPnet::nms(&total_boxes, 0.7, SuppressionType::Union);

            //dets = clip_dets(total_boxes[pick, :], img_w, img_h)
            let indexed_boxes_t = total_pick
                .iter()
                .map(|v| total_boxes.index_axis(Axis(0), *v))
                .collect::<Vec<_>>();
            let indexed_boxes = Array::from_shape_vec(
                (total_pick.len(), 5),
                indexed_boxes_t
                    .iter()
                    .flatten()
                    .map(|v| *v)
                    .collect::<Vec<_>>(),
            )
            .unwrap();

            let dets = TrtPnet::clip_dets(&indexed_boxes, width, height);
            dets
        }
    }

    pub fn detect(&self, image: &DynamicImage, minsize: u32, factor: f64, threshold: f64) {
        const INPUT_H_OFFSETS: [u32; 9] = [0, 216, 370, 478, 556, 610, 648, 676, 696];
        const OUTPUT_H_OFFSETS: [i32; 9] = [0, 108, 185, 239, 278, 305, 324, 338, 348];
        const MAX_N_SCALES: u8 = 9;
        const PIXEL_MEAN: f32 = 127.5;
        const PIXEL_SCALE: f32 = 0.0078125;

        // if minsize < 40:
        // raise ValueError("TrtPNet is currently designed with "
        //                  "'minsize' >= 40")
        // if factor > 0.709:
        // raise ValueError("TrtPNet is currently designed with "
        //                  "'factor' <= 0.709")

        let scales = TrtPnet::get_scales(image.width(), image.height(), minsize, factor);

        let mut im_data = DynamicImage::new_rgb8(384, 710);

        for (i, scale) in scales.iter().enumerate() {
            let h_offset = INPUT_H_OFFSETS[i];
            let h = (image.height() as f64 * scale) as u32;
            let w = (image.width() as f64 * scale) as u32;
            {
                let dst = image.resize(w, h, FilterType::Nearest);

                im_data.copy_from(&dst, 0, h_offset).unwrap();
                //display_image(&im_data);
            }
        }

        let im_data_rgb = im_data.to_rgb8();
        let mut im_array: ndarray_image::NdColor = ndarray_image::NdImage(&im_data_rgb).into();

        im_array.swap_axes(0, 2);
        im_array.swap_axes(1, 2);

        let mut pre_processed = im_array.map(|&x| {
            if x == 0 {
                0.0
            } else {
                ((x as f32) - PIXEL_MEAN) * PIXEL_SCALE
            }
        });

        let (prob1, boxes) = self.execute(&mut pre_processed);
    }
}

impl Drop for TrtPnet {
    fn drop(&mut self) {
        drop(&self.pnet_engine);
    }
}

fn rescale(image: &DynamicImage, min_size: u32) -> (RgbImage, u32) {
    let scale = f32::min(720.0 / image.height() as f32, 1280.0 / image.width() as f32);
    let (width, height) = if scale < 1.0 {
        (
            (image.width() as f32 * scale).ceil() as u32,
            (image.height() as f32 * scale).ceil() as u32,
        )
    } else {
        (image.width(), image.height())
    };
    let ms = || {
        if scale < 1.0 {
            return cmp::max((min_size as f32 * scale).ceil() as u32, 40);
        } else {
            return min_size;
        }
    };
    (
        image.resize(width, height, FilterType::Nearest).to_rgb8(),
        ms(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helper::*;
    use ndarray::{Array, Array3};
    use ndarray_npy::read_npy;

    #[test]
    fn create_pnet() {
        let logger = Logger::new();
        let pnet = TrtPnet::new("./test_resources/det1.engine", &logger).unwrap();

        assert_eq!(pnet.data_dims, (3, 710, 384));
        assert_eq!(pnet.prob1_dims, (2, 350, 187));
        assert_eq!(pnet.boxes_dims, (4, 350, 187));
    }

    #[test]
    fn test_rescale() {
        let img1 = image::open("test_resources/2020-11-21-144033.jpg").unwrap();

        let (scaled_image1, min_size) = rescale(&img1, 40);

        assert_eq!(min_size, 40);
        assert_eq!(scaled_image1.width(), 640);
        assert_eq!(scaled_image1.height(), 360);

        let img2 = image::open("test_resources/DSC_0003.JPG").unwrap();

        let (scaled_image2, min_size) = rescale(&img2, 40);

        assert_eq!(min_size, 40);
        assert_eq!(scaled_image2.width(), 1076);
        assert_eq!(scaled_image2.height(), 720);

        // display_image(&scaled_image1);

        // display_image(&img2);

        // display_image(&scaled_image2);
    }

    #[test]
    fn test_pnet_detect() {
        let logger = Logger::new();
        let pnet = TrtPnet::new("./test_resources/det1.engine", &logger).unwrap();

        let img2 = image::open("test_resources/DSC_0003.JPG").unwrap();

        let (scaled_image2, min_size) = rescale(&img2, 40);

        assert_eq!(min_size, 40);
        assert_eq!(scaled_image2.width(), 1076);
        assert_eq!(scaled_image2.height(), 720);

        pnet.detect(
            &DynamicImage::ImageRgb8(scaled_image2),
            min_size,
            0.709,
            0.7,
        );
    }

    #[test]
    fn test_pnet_execute() {
        let logger = Logger::new();
        let pnet = TrtPnet::new("./test_resources/det1.engine", &logger).unwrap();

        let mut np_im_data: Array3<f32> = read_npy("test_resources/im_data.npy").unwrap();

        let np_boxes: Array3<f32> = read_npy("test_resources/boxes.npy").unwrap();
        let np_prob1: Array3<f32> = read_npy("test_resources/prob1.npy").unwrap();

        let (prob1, boxes) = pnet.execute(&mut np_im_data);

        assert_eq!(prob1.dim(), np_prob1.dim());
        assert_eq!(boxes.dim(), np_boxes.dim());

        assert_eq!(boxes, np_boxes);
        //     assert_eq!(prob1, np_prob1);
    }

    #[test]
    fn test_pnet_extract_outputs() {
        let scales = TrtPnet::get_scales(1076, 720, 40, 0.709);
        let np_boxes: Array3<f32> = read_npy("test_resources/boxes.npy").unwrap();
        let np_prob1: Array3<f32> = read_npy("test_resources/prob1.npy").unwrap();

        let outputs = TrtPnet::extract_outputs(1076, 720, &scales, &np_prob1, &np_boxes, 0.7);
    }

    #[test]
    fn test_generate_pnet_bboxes() {
        let scales = TrtPnet::get_scales(1076, 720, 40, 0.709);
        for (i, scale) in scales.iter().enumerate() {
            let cc: Array3<f32> = read_npy(format!("test_resources/cc{}.npy", i)).unwrap();
            let pp: Array2<f32> = read_npy(format!("test_resources/pp{}.npy", i)).unwrap();
            let pnetboxes: Array2<f32> =
                read_npy(format!("test_resources/pnetboxes{}.npy", i)).unwrap();
            let boxes = TrtPnet::generate_pnet_bboxes(&pp, &cc, scale, 0.7);

            assert_eq!(boxes, pnetboxes);
        }
    }

    #[test]
    fn test_nms() {
        let pick_res: Vec<Vec<usize>> = vec![
            vec![36, 12, 43, 31, 22, 6, 44, 3, 19, 47, 33, 2, 0, 1, 8],
            vec![20, 25, 5, 1, 13, 6, 7],
            vec![10, 2, 15, 0, 17, 18, 3],
            vec![3, 9, 0, 10, 1, 2],
        ];

        let scales = TrtPnet::get_scales(1076, 720, 40, 0.709);
        for (i, scale) in scales.iter().enumerate() {
            let pnetboxes: Array2<f32> =
                read_npy(format!("test_resources/pnetboxes{}.npy", i)).unwrap();
            let pick = TrtPnet::nms(&pnetboxes, 0.5, SuppressionType::Union);
            if pick.len() > 0 {
                assert_eq!(pick, pick_res[i]);
            }
        }
    }

    #[test]
    fn test_clip_dets() {
        let indexedboxes: Array2<f32> = read_npy("test_resources/indexedboxes.npy").unwrap();
        let dts: Array2<f32> = read_npy("test_resources/dets.npy").unwrap();

        let dets = TrtPnet::clip_dets(&indexedboxes, 1076, 720);

        assert_eq!(dts, dets);
    }
}
