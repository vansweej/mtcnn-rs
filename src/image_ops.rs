extern crate modifier;

use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView};
use modifier::*;
use std::path::Path;
use std::rc::Rc;

pub struct Image {
    value: Rc<DynamicImage>,
}

impl Set for Image {}

impl Image {
    pub fn new(image: Rc<DynamicImage>) -> Image {
        Image { value: image }
    }

    pub fn from_path<P>(path: P) -> Image
    where
        P: AsRef<Path>,
    {
        Image {
            value: Rc::new(image::open(path).unwrap()),
        }
    }
}

#[derive(Debug, Clone, Copy)]
/// Resize Modifier for `Image`
pub struct Resize {
    /// The resized width of the new Image
    pub width: u32,
    /// The resized heigt of the new Image
    pub height: u32,
}

impl Modifier<Image> for Resize {
    fn modify(self, image: &mut Image) {
        image.value = Rc::new(
            image
                .value
                .resize(self.width, self.height, FilterType::Nearest),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_from_dynamic_image() {
        let dimg = Rc::new(image::open("test_resources/DSC_0003.JPG").unwrap());
        let img = Image::new(Rc::clone(&dimg));
        assert_eq!(img.value.width(), 3872);
        assert_eq!(img.value.height(), 2592);

        let op = Resize {
            width: 640,
            height: 480,
        };
        let img1 = img.set(op);
        assert_eq!(img1.value.width(), 640);
        assert_eq!(img1.value.height(), 428);

        let op2 = Resize {
            width: 1024,
            height: 768,
        };
        let img2 = Image::new(Rc::clone(&dimg));
        let img3 = img2.set(op2);
        assert_eq!(img3.value.width(), 1024);
        assert_eq!(img3.value.height(), 685);

        let img4 = Image::new(Rc::clone(&dimg));
        assert_eq!(img4.value.width(), 3872);
        assert_eq!(img4.value.height(), 2592);

        let img5 = img4.set((op, op2));
        assert_eq!(img5.value.width(), 1024);
        assert_eq!(img5.value.height(), 684);

        let op_composed = (op, op2);
    }

    #[test]
    fn test_image() {
        let img = Image::from_path("test_resources/DSC_0003.JPG");
        assert_eq!(img.value.width(), 3872);
        assert_eq!(img.value.height(), 2592);

        let op = Resize {
            width: 640,
            height: 480,
        };
        let img = img.set(op);
        assert_eq!(img.value.width(), 640);
        assert_eq!(img.value.height(), 428);

        let op2 = Resize {
            width: 1024,
            height: 768,
        };
        let img = img.set(op2);
        assert_eq!(img.value.width(), 1024);
        assert_eq!(img.value.height(), 684);
    }
}
