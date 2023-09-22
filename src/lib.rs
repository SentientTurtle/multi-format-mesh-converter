#![feature(vec_into_raw_parts)]
#![feature(slice_take)]
#![feature(const_trait_impl)]
#![feature(array_chunks)]
#![feature(const_mut_refs)]

pub mod util;
pub mod geometry;
#[cfg(feature = "wavefront-obj")]
pub mod wavefront_obj;
#[cfg(feature = "studiomdl-data")]
pub mod studiomdl_data;