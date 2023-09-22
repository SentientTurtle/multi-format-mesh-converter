use std::collections::HashMap;
use std::{fmt, io};
use std::fmt::Display;
use std::io::{Error, Seek, Write};
use std::path::Path;
use indexmap::{IndexMap, IndexSet};
use tobj::{LoadOptions};
use zip::result::ZipError;
use zip::write::FileOptions;
use crate::geometry;
use crate::geometry::{CoordinateDirections3D, Direction, FaceIndex, GeometryNumber, Material, Mesh, ModelError, Faces, Vector2D, Vector3D, VertexPropertyIndices};
use crate::util::{FakeHash, SliceExtension};

// tobj doesn't export it's float type, so we re-determine it. This is somewhat fragile, but the compiler will catch and error if this breaks
#[cfg(not(feature = "tobj::use_f64"))]
pub type ObjFloat = f32;
#[cfg(feature = "tobj::use_f64")]
pub type ObjFloat = f64;


/// Wavefront .obj parser & generator
pub struct WavefrontObj;

/// Coordinate system axes setup for .obj files; +X Right, +Y Up, -Z Forward
pub const OBJ_AXES: CoordinateDirections3D = CoordinateDirections3D::new_const([Direction::Forward { axis_is_negative: false }, Direction::Up { axis_is_negative: false }, Direction::Right { axis_is_negative: false }]);

impl<P: AsRef<Path> + fmt::Debug> geometry::ModelReader<P, tobj::LoadError> for WavefrontObj {
    type Float = ObjFloat;

    fn read_model(input_path: P, name: Option<String>) -> Result<geometry::Model<Self::Float>, tobj::LoadError> {
        let load_opts = LoadOptions::default();
        let (models, materials_result) = tobj::load_obj(input_path, &load_opts)?;
        let materials = materials_result?;

        let mut add_none_material = false;

        // tobj splits models with multiple materials, ::geometry::model uses per-face materials, so we merge the models with identical name
        let mut model_map = HashMap::new();
        for model in models {
            model_map.entry(model.name.clone())
                .or_insert_with(Vec::new)
                .push(model)
        }

        let single_model_name = if model_map.len() == 1 {
            Some(model_map.keys().next().unwrap().clone())
        } else {
            None
        };

        let mut meshes = Vec::new();
        for models in model_map.into_values() {
            for model in models.iter() {
                if model.mesh.positions.len() % 3 != 0 {
                    return Err(tobj::LoadError::PositionParseError)
                }
            }

            let mut vertex_positions = IndexSet::<[FakeHash<ObjFloat>; 3]>::new();

            let mut normal_vectors = IndexSet::<[FakeHash<ObjFloat>; 3]>::new();
            let mut uv_vectors = IndexSet::<[FakeHash<ObjFloat>; 2]>::new();

            let mut face_indices = Faces::new();
            let mut face_materials = Vec::new();
            let mut normal_indices = Vec::new();
            let mut uv_indices = Vec::new();

            // Buffer for building slices of face vertex-indices
            let mut face_index_buffer = Vec::new();

            for model in &models {
                if model.mesh.face_arities.len() == 0 {
                    for face in model.mesh.indices.chunks_exact(3) {
                        face_index_buffer.clear();
                        for vertex_index in face {
                            if let Some(vertex) = model.mesh.positions.select_array::<3>(*vertex_index as usize * 3) {
                                let (set_index, _) = vertex_positions.insert_full(vertex.map( FakeHash::new));
                                face_index_buffer.push(set_index);
                            } else {
                                return Err(tobj::LoadError::FaceVertexOutOfBounds);
                            }
                        }
                        face_indices.push_face(&*face_index_buffer)
                            .map_err(|_| tobj::LoadError::FaceParseError)?;
                        face_materials.push(model.mesh.material_id.unwrap_or(materials.len()));
                        add_none_material |= model.mesh.material_id.is_none();  // If no material, set add_none_material to true
                    }
                } else {
                    let mut indices = &model.mesh.indices[..];
                    for arity in &model.mesh.face_arities {
                        if let Some(face) = indices.take(..*arity as usize) {
                            face_index_buffer.clear();
                            for vertex_index in face {
                                if let Some(vertex) = model.mesh.positions.select_array::<3>(*vertex_index as usize * 3) {
                                    let (set_index, _) = vertex_positions.insert_full(vertex.map(FakeHash::new));
                                    face_index_buffer.push(set_index);
                                } else {
                                    return Err(tobj::LoadError::FaceVertexOutOfBounds);
                                }
                            }
                            face_indices.push_face(&*face_index_buffer)
                                .map_err(|_| tobj::LoadError::FaceParseError)?;
                            face_materials.push(model.mesh.material_id.unwrap_or(materials.len()));
                            add_none_material |= model.mesh.material_id.is_none();  // If no material, set add_none_material to true
                        } else {
                            return Err(tobj::LoadError::FaceParseError);
                        }
                    }
                }

                if model.mesh.normal_indices.len() == 0 {
                    for normal_index in &model.mesh.indices {
                        if let Some(normal) = model.mesh.normals.select_array::<3>(*normal_index as usize * 3) {
                            let (set_index, _) = normal_vectors.insert_full(normal.map(FakeHash::new));
                            normal_indices.push(set_index)
                        } else {
                            return Err(tobj::LoadError::FaceNormalOutOfBounds);
                        }
                    }
                } else if model.mesh.normal_indices.len() == model.mesh.indices.len() {
                    for normal_index in &model.mesh.normal_indices {
                        if let Some(normal) = model.mesh.normals.select_array::<3>(*normal_index as usize * 3) {
                            let (set_index, _) = normal_vectors.insert_full(normal.map(FakeHash::new));
                            normal_indices.push(set_index)
                        } else {
                            return Err(tobj::LoadError::FaceNormalOutOfBounds);
                        }
                    }
                } else {
                    return Err(tobj::LoadError::NormalParseError);
                }

                if model.mesh.texcoord_indices.len() == 0 {
                    for uv_index in &model.mesh.indices {
                        if let Some(uv) = model.mesh.texcoords.select_array::<2>(*uv_index as usize * 2) {
                            let (set_index, _) = uv_vectors.insert_full(uv.map(FakeHash::new));
                            uv_indices.push(set_index)
                        } else {
                            return Err(tobj::LoadError::FaceTexCoordOutOfBounds);
                        }
                    }
                } else if model.mesh.texcoord_indices.len() == model.mesh.indices.len() {
                    for uv_index in &model.mesh.texcoord_indices {
                        if let Some(uv) = model.mesh.texcoords.select_array::<2>(*uv_index as usize * 2) {
                            let (set_index, _) = uv_vectors.insert_full(uv.map(FakeHash::new));
                            uv_indices.push(set_index)
                        } else {
                            return Err(tobj::LoadError::FaceTexCoordOutOfBounds);
                        }
                    }
                } else {
                    return Err(tobj::LoadError::TexcoordParseError);
                }
            }

            meshes.push(
                Mesh::new(
                    Some(models[0].name.clone()),
                    OBJ_AXES,
                    vertex_positions.into_iter().map(|vertex| Vector3D::new(vertex.map(FakeHash::unwrap))).collect(),
                    face_indices,
                    face_materials,
                    normal_vectors.into_iter().map(|vertex| Vector3D::new(vertex.map(FakeHash::unwrap))).collect(),
                    VertexPropertyIndices::PerVertexPerFace(normal_indices),
                    uv_vectors.into_iter().map(|vertex| Vector2D::new(vertex.map(FakeHash::unwrap))).collect(),
                    VertexPropertyIndices::PerVertexPerFace(uv_indices),
                    Vec::new(),
                    Vec::new(),
                ).expect("OBJ read should produce valid mesh")
            );
        }


        let mut materials: Vec<Material<ObjFloat>> = materials.into_iter().map(|obj_material| {
            Material::Phong {
                name: obj_material.name,
                ambient: obj_material.ambient,
                diffuse: obj_material.diffuse,
                specular: obj_material.specular,
                specular_exponent: obj_material.shininess,
                ambient_texture: obj_material.ambient_texture,
                diffuse_texture: obj_material.diffuse_texture,
                specular_texture: obj_material.specular_texture,
            }
        }).collect();


        if add_none_material {
            materials.push(Material::None)
        }

        Ok(
            geometry::Model::new(
                name.or(single_model_name).unwrap_or("UNNAMED_MODEL".to_string()),
                meshes,
                materials,
            )
                .expect("OBJ read should produce valid model")
        )
    }
}

#[derive(Debug)]
pub enum ObjWriteError {
    IO(io::Error),
    ZIP(zip::result::ZipError),
    ModelError(geometry::ModelError)
}

impl From<io::Error> for ObjWriteError {
    fn from(value: Error) -> Self {
        ObjWriteError::IO(value)
    }
}

impl From<ZipError> for ObjWriteError {
    fn from(value: ZipError) -> Self {
        ObjWriteError::ZIP(value)
    }
}

impl From<ModelError> for ObjWriteError {
    fn from(value: ModelError) -> Self {
        ObjWriteError::ModelError(value)
    }
}

impl<W: Write + Seek, Float: GeometryNumber + Copy + Display> geometry::ModelWriter<W, ObjWriteError, Float> for WavefrontObj {
    fn write_model(output: W, model: &geometry::Model<Float>) -> Result<(), ObjWriteError> {
        let mut zip_writer = zip::ZipWriter::new(output);

        zip_writer.start_file(model.name().to_string() + ".obj", FileOptions::default())?;
        writeln!(zip_writer, "# Multimesh OBJ output")?;
        if model.materials().len() > 0 {
            writeln!(zip_writer, "mtllib {}.mtl", model.name())?;
        }
        for (i, mesh) in model.meshes().iter().enumerate() {
            let mesh = mesh.view(OBJ_AXES);

            writeln!(zip_writer, "o {}", mesh.name().as_ref().unwrap_or(&format!("mesh_{}", i)))?;
            for vertex in mesh.vertices() {
                writeln!(zip_writer, "v {} {} {}", vertex[0], vertex[1], vertex[2])?;
            }
            for uv in mesh.uv_vectors() {
                writeln!(zip_writer, "vt {} {}", uv[0], uv[1])?;
            }
            for normal in mesh.normal_vectors() {
                writeln!(zip_writer, "vn {} {} {}", normal[0], normal[1], normal[2])?;
            }

            let mut face_material_map = IndexMap::new();

            for idx in mesh.faces() {
                face_material_map.entry(idx.material_id())
                    .or_insert_with(Vec::new)
                    .push(idx);
            }

            for (material_id, faces) in face_material_map {
                let material_name = match &model.materials()[material_id] {
                    Material::None => "(null)",
                    Material::External(name) | Material::Phong { name, .. } => name.as_str()
                };

                writeln!(zip_writer, "usemtl {}", material_name)?;
                for face in faces {
                    write!(zip_writer, "f")?;

                    for FaceIndex{ vertex, uv, normal } in face.indices() {
                        write!(zip_writer, " {}/{}/{}", vertex + 1, uv + 1, normal + 1)?;
                    }
                    writeln!(zip_writer)?;
                }
            }
        }

        zip_writer.flush()?;

        if model.materials().len() > 0 {
            zip_writer.start_file(format!("{}.mtl", model.name()), FileOptions::default())?;
            writeln!(zip_writer, "# Multimesh MTL output")?;
            for material in model.materials() {
                match material {
                    Material::None => {}
                    Material::External(name) => {
                        writeln!(zip_writer, "newmtl {}", name)?;
                    }
                    Material::Phong {
                        name,
                        ambient,
                        diffuse,
                        specular,
                        specular_exponent,
                        ambient_texture,
                        diffuse_texture,
                        specular_texture
                    } => {
                        writeln!(zip_writer, "newmtl {}", name)?;
                        if let Some([r, g,b]) = ambient {
                            writeln!(zip_writer, "Ka {} {} {}", r, g, b)?;
                        }
                        if let Some([r, g,b]) = diffuse {
                            writeln!(zip_writer, "Kd {} {} {}", r, g, b)?;
                        }
                        if let Some([r, g,b]) = specular {
                            writeln!(zip_writer, "Ks {} {} {}", r, g, b)?;
                        }
                        if let Some(specular_exponent) = specular_exponent {
                            writeln!(zip_writer, "Ns {}", specular_exponent)?;
                        }
                        if let Some(ambient_texture) = ambient_texture {
                            writeln!(zip_writer, "map_Ka {}", ambient_texture)?;
                        }
                        if let Some(diffuse_texture) = diffuse_texture {
                            writeln!(zip_writer, "map_Kd {}", diffuse_texture)?;
                        }
                        if let Some(specular_texture) = specular_texture {
                            writeln!(zip_writer, "map_Ks {}", specular_texture)?;
                        }
                    }
                }
                writeln!(zip_writer)?;
            }
            zip_writer.flush()?;
        }

        zip_writer.finish()?;

        Ok(())
    }
}