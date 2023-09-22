#![feature(vec_into_raw_parts)]
#![feature(slice_take)]
#![feature(const_trait_impl)]
#![feature(array_chunks)]
#![feature(const_mut_refs)]

use std::fs::File;
use std::io::Read;
use crate::geometry::ModelWriter;
use crate::geometry::ModelReader;
use crate::studiomdl_data::StudioMDLData;
use crate::wavefront_obj::WavefrontObj;

pub mod util;
pub mod geometry;
#[cfg(feature = "wavefront-obj")]
pub mod wavefront_obj;
#[cfg(feature = "studiomdl-data")]
pub mod studiomdl_data;

fn main() {
    let model_obj = WavefrontObj::read_model("./rsc/shapes.obj", None).unwrap();

    WavefrontObj::write_model(File::create("./test_out/obj-to-obj.zip").unwrap(), &model_obj).expect("");
    StudioMDLData::write_model(File::create("./test_out/obj-to-smd.smd").unwrap(), &model_obj).expect("");

    let mut smd_buffer = String::new();
    File::open("./rsc/shapes.smd").expect("smd test file").read_to_string(&mut smd_buffer).expect("smd test file");
    let model_smd = StudioMDLData::read_model(&*smd_buffer, Some("shapes SMD".to_string())).expect("");

    WavefrontObj::write_model(File::create("./test_out/smd-to-obj.zip").unwrap(), &model_smd).expect("");
    StudioMDLData::write_model(File::create("./test_out/smd-to-smd.smd").unwrap(), &model_smd).expect("");
}