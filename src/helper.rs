use ndarray::prelude::*;
use std::cmp;

// Some numpy functions
pub fn maximum<A, D>(num: &A, num_array: Array<A, D>) -> Array1<A>
where
    A: std::cmp::PartialOrd + std::marker::Copy,
    D: Dimension,
{
    num_array
        .iter()
        .map(|v| if *v > *num { *v } else { *num })
        .collect::<Array<_, _>>()
}

pub fn minimum<A, D>(num: &A, num_array: Array<A, D>) -> Array1<A>
where
    A: std::cmp::PartialOrd + std::marker::Copy,
    D: Dimension,
{
    num_array
        .iter()
        .map(|v| if *v > *num { *num } else { *v })
        .collect::<Array<_, _>>()
}

pub enum SuppressionType {
    Union,
    #[allow(dead_code)]
    Min,
}

pub fn get_scales(width: u32, height: u32, minsize: u32, factor: f64) -> Vec<f64> {
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

pub fn nms(boxes: &Array2<f32>, threshold: f32, s_type: SuppressionType) -> Vec<usize> {
    let areas = boxes
        .axis_iter(Axis(0))
        .map(|x| (x[2] - x[0] + 1.0) * (x[3] - x[1] + 1.0))
        .collect::<Array<_, _>>();

    let mut sorted_idx = boxes
        .slice(s![.., 4])
        .indexed_iter()
        .map(|(index, &_item)| index)
        .collect::<Vec<usize>>();
    sorted_idx.sort_unstable_by(|a, b| boxes[[*a, 4]].partial_cmp(&boxes[[*b, 4]]).unwrap());

    let xx1 = boxes.slice(s![.., 0]);
    let yy1 = boxes.slice(s![.., 1]);
    let xx2 = boxes.slice(s![.., 2]);
    let yy2 = boxes.slice(s![.., 3]);

    let mut pick: Vec<usize> = vec![];
    loop {
        if sorted_idx.is_empty() {
            break;
        }

        let (begin, last) = sorted_idx.split_at(sorted_idx.len() - 1);
        let tx1 = maximum(&xx1[last[0]], xx1.select(Axis(0), begin));
        let ty1 = maximum(&yy1[last[0]], yy1.select(Axis(0), begin));
        let tx2 = minimum(&xx2[last[0]], xx2.select(Axis(0), begin));
        let ty2 = minimum(&yy2[last[0]], yy2.select(Axis(0), begin));

        let tw = maximum(&0.0, tx2 - tx1 + 1.0);
        let th = maximum(&0.0, ty2 - ty1 + 1.0);
        let inter = tw * th;

        let iou = match s_type {
            SuppressionType::Min => inter / minimum(&areas[last[0]], areas.select(Axis(0), begin)),
            SuppressionType::Union => {
                inter.clone() / (areas[last[0]] + areas.select(Axis(0), begin) - inter)
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

pub fn clip_dets(in_dets: &Array2<f32>, img_w: u32, img_h: u32) -> Array2<f32> {
    let out_dets_t = in_dets
        .axis_iter(Axis(0))
        .map(|v| {
            [
                (f32::trunc(v[0])).clamp(0.0, (img_w - 1) as f32),
                (f32::trunc(v[1])).clamp(0.0, (img_h - 1) as f32),
                (f32::trunc(v[2])).clamp(0.0, (img_w - 1) as f32),
                (f32::trunc(v[3])).clamp(0.0, (img_h - 1) as f32),
                v[4],
            ]
        })
        .collect::<Vec<_>>();

    let out_dets = Array::from_shape_vec(
        (in_dets.dim().0, in_dets.dim().1),
        out_dets_t.iter().flatten().copied().collect::<Vec<_>>(),
    )
    .unwrap();

    out_dets
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_npy::read_npy;

    #[test]
    fn test_nms() {
        let pick_res: Vec<Vec<usize>> = vec![
            vec![36, 12, 43, 31, 22, 6, 44, 3, 19, 47, 33, 2, 0, 1, 8],
            vec![20, 25, 5, 1, 13, 6, 7],
            vec![10, 2, 15, 0, 17, 18, 3],
            vec![3, 9, 0, 10, 1, 2],
        ];

        let scales = get_scales(1076, 720, 40, 0.709);
        for (i, _scale) in scales.iter().enumerate() {
            let pnetboxes: Array2<f32> =
                read_npy(format!("test_resources/pnetboxes{}.npy", i)).unwrap();
            let pick = nms(&pnetboxes, 0.5, SuppressionType::Union);
            if pick.len() > 0 {
                assert_eq!(pick, pick_res[i]);
            }
        }
    }

    #[test]
    fn test_clip_dets() {
        let indexedboxes: Array2<f32> = read_npy("test_resources/indexedboxes.npy").unwrap();
        let dts: Array2<f32> = read_npy("test_resources/dets.npy").unwrap();

        let dets = clip_dets(&indexedboxes, 1076, 720);

        assert_eq!(dts, dets);
    }
}
