// extern crate ole32;

// use std::ffi::CStr;
// use std::mem;
// use std::mem::{drop, transmute};
// use std::os::raw::c_char;
// use std::path::Path;
// use std::ptr;
// use std::slice;

// //use ole32::CoTaskMemAlloc;

// use crate::network::TensorRTNN;

// pub struct Baton {
//     rtnn: TensorRTNN,
// }

// impl<'a> Baton {
//     fn to_ptr(self) -> *mut Baton {
//         unsafe { transmute(Box::new(self)) }
//     }

//     fn from_ptr(ptr: *mut Baton) -> &'a mut Baton {
//         unsafe { &mut *ptr }
//     }

//     // fn init(engine_path_str: &str) -> Result<Baton, String> {
//     //     if Path::new(&engine_path_str).exists() {
//     //         let tensorrt_nn = TensorRTNN::new(&engine_path_str);
//     //         if tensorrt_nn.is_ok() {
//     //             Ok(Baton {
//     //                 rtnn: tensorrt_nn.unwrap(),
//     //             })
//     //         } else {
//     //             Err("Error instantiating mtcnn network".to_string())
//     //         }
//     //     } else {
//     //         Err("engine path does not exists".to_string())
//     //     }
//     // }

//     // fn infer(&mut self, len: &mut u32) -> Result<*mut face_box, String> {
//     //     let amount = 2 as u32;

//     //     *len = amount;
//     //     let bytes_to_alloc = amount * mem::size_of::<face_box>() as u32;
//     //     let face_box_array_ptr = unsafe { CoTaskMemAlloc(bytes_to_alloc.into()) as *mut face_box };
//     //     let array = unsafe { slice::from_raw_parts_mut(face_box_array_ptr, amount as usize) };
//     //     for (index, fbox) in array.iter_mut().enumerate() {
//     //         fbox.x0 = index as f32;
//     //         fbox.y0 = index as f32;
//     //         fbox.x1 = index as f32;
//     //         fbox.y1 = index as f32;
//     //     }
//     //     Ok(face_box_array_ptr)
//     // }

//     fn dispose(ptr: *mut *mut Baton) {
//         let baton: Box<Baton> = unsafe { transmute(*ptr) };

//         drop(baton);
//     }
// }

// // #[no_mangle]
// // pub extern "C" fn init(ptr: *mut *const Baton, engine_path_c_char: *const c_char) -> bool {
// //     let engine_path = unsafe { CStr::from_ptr(engine_path_c_char) };
// //     match Baton::init(engine_path.to_str().unwrap()) {
// //         Ok(baton) => {
// //             unsafe {
// //                 *ptr = baton.to_ptr();
// //             }

// //             true
// //         }
// //         Err(message) => {
// //             unsafe {
// //                 *ptr = ptr::null();
// //             }
// //             false
// //         }
// //     }
// // }

// #[repr(C)]
// pub struct face_box {
//     pub x0: f32,
//     pub y0: f32,
//     pub x1: f32,
//     pub y1: f32,
// }

// // #[no_mangle]
// // pub extern "C" fn infer(ptr: *mut Baton, face_box_list: *mut *mut face_box, len: &mut u32) -> bool {
// //     if !ptr.is_null() {
// //         let res = Baton::from_ptr(ptr).infer(len);
// //         if res.is_ok() {
// //             unsafe { *face_box_list = res.unwrap() };
// //             true
// //         } else {
// //             false
// //         }
// //     } else {
// //         false
// //     }
// // }

// #[no_mangle]
// pub extern "C" fn dispose(ptr: *mut *mut Baton) {
//     if !ptr.is_null() && unsafe { !(*ptr).is_null() } {
//         Baton::dispose(ptr);

//         unsafe {
//             *ptr = ptr::null_mut();
//         }
//     }
// }
