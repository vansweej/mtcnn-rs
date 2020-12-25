use ndarray::prelude::*;
use show_image::{make_window, ImageData, KeyCode};
use std::time::Duration;

pub fn display_image(image: impl ImageData) {
    // Create a window and display the image.
    let window = make_window("image").unwrap();
    window.set_image(image, "image-001").unwrap();

    // Print keyboard events until Escape is pressed, then exit.
    // If the user closes the window, wait_key() will return an error and the loop also exits.
    while let Ok(event) = window.wait_key(Duration::from_millis(100)) {
        if let Some(event) = event {
            if event.key == KeyCode::Escape {
                break;
            }
        }
    }

    // Make sure all background tasks are stopped cleanly.
    //show_image::stop().unwrap();
}

// Some numpy functions
pub fn maximum<A, D>(
    num: &A,
    num_array: Array<A, D>,
) -> ArrayBase<ndarray::OwnedRepr<A>, Dim<[usize; 1]>>
where
    A: std::cmp::PartialOrd + std::marker::Copy,
    D: Dimension,
{
    num_array
        .iter()
        .map(|v| if *v > *num { *v } else { *num })
        .collect::<Array<_, _>>()
}

pub fn minimum<A, D>(
    num: &A,
    num_array: Array<A, D>,
) -> ArrayBase<ndarray::OwnedRepr<A>, Dim<[usize; 1]>>
where
    A: std::cmp::PartialOrd + std::marker::Copy,
    D: Dimension,
{
    num_array
        .iter()
        .map(|v| if *v > *num { *num } else { *v })
        .collect::<Array<_, _>>()
}

pub fn map_index(
    index: &[usize],
    arr: &ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<[usize; 1]>>,
) -> ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 1]>> {
    index.iter().map(|v| arr[*v]).collect::<Array<_, _>>()
}

pub fn clamp<T: PartialOrd>(input: T, min: T, max: T) -> T {
    debug_assert!(min <= max, "min must be less than or equal to max");
    if input < min {
        min
    } else if input > max {
        max
    } else {
        input
    }
}
