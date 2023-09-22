use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash};
use std::io;
use std::io::{Seek, Write};
use std::str::FromStr;
use indexmap::{IndexMap, IndexSet};
use nom::bytes::complete::{tag, take_till};
use nom::character::complete::{digit1};
use nom::combinator::{map, opt, recognize};
use nom::{IResult};
use nom::error::{ErrorKind, ParseError};
use nom::multi::{count, many0, many1};
use nom::sequence::tuple;
use crate::geometry::{Bone, GeometryNumber, Material, Mesh, Model, ModelReader, ModelWriter, RotMatrix, ModelError, Faces, VertexPropertyIndices};
use crate::geometry::{CoordinateDirections3D, Direction, Vector2D, Vector3D};
use crate::util::FakeHash;

/// MDL-Parsing error
#[derive(Debug)]
pub enum MDLError {
    IO(io::Error),
    IncompleteDocument,
    DataAfterDocumentEnd { line: usize, character: usize },
    ParseError { line: usize, character: usize, message: String },
    DuplicateBoneID(i32),
    /// Bone whose (grand)parents form a cycle.
    CyclicBone(i32),
    UnknownParentBone(i32),
    PositionForUnknownBone(i32),
    NoPositionForBone(i32),
    /// Vertex specifies an unknown bone
    UnknownVertexBone(i32),
}

impl Display for MDLError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            MDLError::IO(io_error) => write!(f, "IO error while parsing MDL document: {}", io_error),
            MDLError::IncompleteDocument => write!(f, "Incomplete MDL document"),
            MDLError::DataAfterDocumentEnd { line, character } => write!(f, "unexpected input after document end (line {}:{})", line, character),
            MDLError::ParseError { line, character, message } => write!(f, "syntax error: {} (line {}:{})", message, line, character),
            MDLError::DuplicateBoneID(id) => write!(f, "duplicate bone id {}", id),
            MDLError::CyclicBone(id) => write!(f, "bone attachment cycle for bone id {}", id),
            MDLError::UnknownParentBone(id) => write!(f, "unknown parent bone id {}", id),
            MDLError::PositionForUnknownBone(id) => write!(f, "found position for unknown bone id {}", id),
            MDLError::NoPositionForBone(id) => write!(f, "missing position for bone id {}", id),
            MDLError::UnknownVertexBone(id) => write!(f, "vertex with invalid bone id {}", id)
        }
    }
}

impl Error for MDLError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        if let MDLError::IO(err) = self {
            Some(err)
        } else {
            None
        }
    }
}

impl From<io::Error> for MDLError {
    fn from(error: io::Error) -> Self {
        MDLError::IO(error)
    }
}

#[derive(Debug, PartialEq)]
enum MDLSyntaxError {
    // Generic nom parsing error
    Nom(usize),
    ExpectedInteger(usize),
    ExpectedFloat(usize),
    ExpectedWhitespace(usize),
    ExpectedNewline(usize),
    ExpectedBlockHeader(usize, &'static str),
    ExpectedBlockEnd(usize),
    WeightsDoNotMatchWeightCount { remaining_input_length: usize, specified_len: i32, found_len: usize },
}

impl ParseError<&str> for MDLSyntaxError {
    fn from_error_kind(input: &str, _kind: ErrorKind) -> Self {
        MDLSyntaxError::Nom(input.len())
    }

    fn append(_input: &str, _kind: ErrorKind, other: Self) -> Self {
        other
    }
}

/// Datatype representing a single animation bone
#[derive(Debug)]
pub struct MDLBone {
    bone_id: i32,
    bone_name: String,
    parent_bone_id: i32,
}

/// Datatype representing the position of a single bone
#[derive(Debug)]
pub struct MDLAnimationPosition {
    /// Bone for which this data specified position/rotation
    bone_id: i32,
    /// Position of the specified bone. Relative to parent bone, or absolute for root bone
    position: Vector3D<f32>,
    /// Rotation of the specified bone. Relative to parent bone, or absolute for root bone
    rotation: Vector3D<f32>,
}

/// Datatype representing a single keyframe of animation bone positions
#[derive(Debug)]
pub struct MDLAnimationFrame {
    _time: i32,
    positions: Vec<MDLAnimationPosition>,
}

/// Datatype representing the whole skeleton, and it's initial position
#[derive(Debug)]
pub struct MDLSkeleton {
    bones: Vec<MDLBone>,
    initial_frame: MDLAnimationFrame,
}

impl MDLSkeleton {
    pub fn validate(&self) -> Result<HashSet<i32>, MDLError> {
        let mut bone_parent_map = HashMap::with_capacity(self.bones.len());
        let mut valid_bones = HashSet::with_capacity(self.bones.len() + 1);
        valid_bones.insert(-1); // Bone parent-id -1 is the global root

        for bone in &self.bones {
            if bone_parent_map.insert(bone.bone_id, bone.parent_bone_id).is_some() {
                return Err(MDLError::DuplicateBoneID(bone.bone_id));
            }
        }
        for bone in &self.bones {
            let mut current_bone_id = bone.bone_id;
            let mut encountered_ids = HashSet::new();
            loop {
                if !encountered_ids.insert(current_bone_id) {
                    return Err(MDLError::CyclicBone(current_bone_id));   // If already encountered this ID, return an error to avoid looping forever
                } else {
                    valid_bones.insert(current_bone_id);    // Mark bone as valid before checking it's parents; Cycle detection prevents this value from being used before all parents are marked valid, and if parent is not valid, we return out of this function
                    let parent_id = bone_parent_map.get(&current_bone_id).expect("bone parent map has been populated with all bones");
                    if valid_bones.contains(parent_id) {
                        break;
                    } else if bone_parent_map.contains_key(parent_id) {
                        current_bone_id = *parent_id;
                        continue;
                    } else {
                        return Err(MDLError::UnknownParentBone(*parent_id));
                    }
                }
            }
        }

        for pos in &self.initial_frame.positions {
            if bone_parent_map.remove(&pos.bone_id).is_none() {
                return Err(MDLError::PositionForUnknownBone(pos.bone_id));
            }
        }

        if let Some(bone_id) = bone_parent_map.into_keys().next() {
            return Err(MDLError::NoPositionForBone(bone_id));
        }

        valid_bones.remove(&-1);

        Ok(valid_bones)
    }
}

/// Datatype representing a single vertex
///
/// Position, Normal vector, UV coordinates, and animation bone weights are stored as "fakehash" permitting this type to be used in hashed maps, for de-duplication
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct MDLVertex {
    parent_bone_id: i32,
    position: [FakeHash<f32>; 3],
    normal: [FakeHash<f32>; 3],
    uv: [FakeHash<f32>; 2],
    weights: Vec<(i32, FakeHash<f32>)>,
}

/// Datatype representing single MDL Face, which always has exactly 3 vertices
#[derive(Debug)]
pub struct MDLTriangle {
    material_index: usize,
    vertex_1: MDLVertex,
    vertex_2: MDLVertex,
    vertex_3: MDLVertex,
}

impl MDLTriangle {
    pub fn validate(&self, bones: &HashSet<i32>) -> Result<(), MDLError> {
        let first_invalid_bone = self.vertex_1.weights.iter()
            .chain(&self.vertex_2.weights)
            .chain(&self.vertex_3.weights)
            .map(|(bone_id, _weight)| *bone_id)
            .chain([self.vertex_1.parent_bone_id])
            .chain([self.vertex_2.parent_bone_id])
            .chain([self.vertex_3.parent_bone_id])
            .find(|id| !bones.contains(&id));

        if let Some(bone_id) = first_invalid_bone {
            Err(MDLError::UnknownVertexBone(bone_id))
        } else {
            Ok(())
        }
    }

    pub fn iter_vertices(&self) -> impl Iterator<Item=&MDLVertex> {
        [&self.vertex_1, &self.vertex_2, &self.vertex_3].into_iter()
    }
}

/// Coordinate system axes setup for MDL files; +X North, +Y East and +Z up
const MDL_AXES: CoordinateDirections3D = CoordinateDirections3D::new_const([Direction::Forward { axis_is_negative: false }, Direction::Right { axis_is_negative: true }, Direction::Up { axis_is_negative: false }]);

/// Data struct representing an MDL mesh. Each MDL file has exactly one mesh so there is no MDL Model datatype
#[derive(Debug)]
pub struct MDLMesh {
    skeleton: MDLSkeleton,
    triangles: Vec<MDLTriangle>,
    materials: Vec<String>,
}

impl MDLMesh {
    pub fn validate(self) -> Result<Self, MDLError> {
        let bone_ids = self.skeleton.validate()?;
        for triangle in &self.triangles {
            triangle.validate(&bone_ids)?;
        }
        Ok(self)
    }


    /// Converts this Mesh into a [`geometry::Model`]
    ///
    /// MDL format specifies floating point precision to be up to 6 decimal digits, so f32 is used as numeric type
    ///
    /// MDL file contents do not specify a name, so this must be specified
    ///
    /// # Arguments
    ///
    /// * `name`: Name of this model
    ///
    /// returns: Result<Model<f32>, MDLError>
    pub fn into_geometry_model(mut self, name: String) -> Result<Model<f32>, MDLError> {
        self = self.validate()?;

        let mut bone_position_map = HashMap::with_capacity(self.skeleton.initial_frame.positions.len());
        for position in self.skeleton.initial_frame.positions {
            bone_position_map.insert(position.bone_id, position);
        }

        // Insert bones into an indexmap to establish an ordering that can be queried by bone_id
        let mut bone_map = IndexMap::with_capacity(self.skeleton.bones.len());
        for bone in self.skeleton.bones {
            bone_map.insert(bone.bone_id, bone);
        }

        let mut bones = Vec::with_capacity(bone_map.len());
        for bone in bone_map.values() {
            let anim_position = bone_position_map.get(&bone.bone_id)
                .ok_or(MDLError::NoPositionForBone(bone.bone_id))?; // We validate bone_position_map

            // TODO: This function performs the incorrect rotation conversion; Stub implementation needs to be tested against Blender
            // SMD format doesn't describe it's rotation format in detail, this function currently performs a generic XYZ euler angle conversion.
            fn euler_to_rot_matrix(angles: Vector3D<f32>) -> RotMatrix<f32> {
                let [x, y, z] = angles.to_array();

                // "Textbook" implementation; Rust-c appears to optimize out the 1 and 0 multiplications.
                RotMatrix::from_row_major([
                    [1.0, 0.0, 0.0],
                    [0.0, x.cos(), -x.sin()],
                    [0.0, x.sin(), x.cos()]
                ])
                    *
                    RotMatrix::from_row_major([
                        [y.cos(), 0.0, y.sin()],
                        [0.0, 1.0, 0.0],
                        [-y.sin(), 0.0, y.cos()]
                    ])
                    *
                    RotMatrix::from_row_major([
                        [z.cos(), -z.sin(), 0.0],
                        [z.sin(), z.cos(), 0.0],
                        [0.0, 0.0, 1.0]
                    ])
            }

            bones.push(
                Bone::new(
                    bone.bone_name.clone(),
                    anim_position.position,
                    euler_to_rot_matrix(anim_position.rotation),
                    if bone.parent_bone_id == -1 {
                        None
                    } else {
                        // If parent_bone_id is not -1 and unknown, raise an error
                        // This is an index into the Vec we are building, which has identical order as bone_map
                        Some(bone_map.get_index_of(&bone.parent_bone_id).ok_or(MDLError::UnknownParentBone(bone.parent_bone_id))?)
                    }
                )
            )
        }

        let mut face_indices = Faces::new();
        let mut face_materials = Vec::with_capacity(self.triangles.len());

        let mut position_set = IndexSet::new();
        let mut normal_set = IndexSet::new();
        let mut uv_set = IndexSet::new();

        let mut normal_indices = Vec::new();
        let mut uv_indices = Vec::new();

        let mut bone_weights = {
            if bone_map.len() == 0 {
                Vec::new()  // If we have no bones, don't allocate
            } else {
                Vec::with_capacity(position_set.len())
            }
        };

        for tri in self.triangles {
            // Hash both position and bone weights; This duplicates vertices on the same position but with different weights
            // Bone weights are dynamically-sized, so must be cloned.
            let (index_1, _) = position_set.insert_full((tri.vertex_1.position, tri.vertex_1.weights.clone(), tri.vertex_1.parent_bone_id));
            let (index_2, _) = position_set.insert_full((tri.vertex_2.position, tri.vertex_2.weights.clone(), tri.vertex_2.parent_bone_id));
            let (index_3, _) = position_set.insert_full((tri.vertex_3.position, tri.vertex_3.weights.clone(), tri.vertex_3.parent_bone_id));
            face_indices.push_face(&[index_1, index_2, index_3]).unwrap();  // We can unwrap; 3 values are always passed

            for vertex in tri.iter_vertices() {
                let (normal_index, _) = normal_set.insert_full(vertex.normal);
                let (uv_index, _) = uv_set.insert_full(vertex.uv);
                normal_indices.push(normal_index);
                uv_indices.push(uv_index);
            }

            face_materials.push(tri.material_index);  // Reuse material indices, we convert the materials list in identical order
        }

        // Build bone_weights in same order as position set/vertex_positions
        for (_, vertex_weights, parent_bone) in &position_set {
            let mut weight_total = 1.0;
            let mut weights = Vec::with_capacity(vertex_weights.len() + 1);
            for (bone_id, weight) in vertex_weights {
                weights.push((bone_map.get_index_of(bone_id).ok_or(MDLError::UnknownVertexBone(*bone_id))?, weight.unwrap()));
                weight_total -= weight.unwrap();
            }
            // Assign whatever weight remains to the parent bone
            weights.push((bone_map.get_index_of(parent_bone).ok_or(MDLError::UnknownVertexBone(*parent_bone))?, weight_total.max(0.0)));

            bone_weights.push(weights);
        }

        let materials = self.materials.into_iter().map(Material::External).collect();
        Ok(
            Model::new(
                name.clone(),
                vec![Mesh::new(
                    Some(name),
                    MDL_AXES,
                    position_set.into_iter().map(|(position, _, _)| Vector3D::new(position.map(FakeHash::unwrap))).collect(),
                    face_indices,
                    face_materials,
                    normal_set.into_iter().map(|normal| Vector3D::new(normal.map(FakeHash::unwrap))).collect(),
                    VertexPropertyIndices::PerVertexPerFace(normal_indices),
                    uv_set.into_iter().map(|uv| Vector2D::new(uv.map(FakeHash::unwrap))).collect(),
                    VertexPropertyIndices::PerVertexPerFace(uv_indices),
                    bones,
                    bone_weights,
                ).expect("MDL read should provide valid mesh")],
                materials,
            )
                .expect("MDL read should provide valid model")
        )
    }
}

// Using parser combinators here is somewhat inefficient, and involves a lot of heap allocation, but it works for now

fn parse_i32(input: &str) -> IResult<&str, i32, MDLSyntaxError> {
    map(
        recognize(tuple((opt(tag("-")), digit1))),
        |digits| i32::from_str(digits).unwrap(),
    )(input)
        .map_err(|err| err.map(|err: (&str, ErrorKind)| MDLSyntaxError::ExpectedInteger(err.0.len())))
}

fn parse_f32(input: &str) -> IResult<&str, f32, MDLSyntaxError> {
    map(
        recognize(tuple((opt(tag("-")), digit1, opt(tuple((tag("."), digit1)))))),
        |digits| f32::from_str(digits).unwrap(),
    )(input)
        .map_err(|err| err.map(|err: (&str, ErrorKind)| MDLSyntaxError::ExpectedFloat(err.0.len())))
}

fn whitespace_not_newline(c: char) -> bool {
    c != '\n' && c != '\r' && c.is_whitespace()
}

fn whitespace(input: &str) -> IResult<&str, (), MDLSyntaxError> {
    if input.starts_with(whitespace_not_newline) {
        Ok((input.trim_start_matches(whitespace_not_newline), ()))
    } else {
        Err(nom::Err::Error(MDLSyntaxError::ExpectedWhitespace(input.len())))
    }
}

fn newline_and_whitespace(mut input: &str) -> IResult<&str, (), MDLSyntaxError> {
    input = input.trim_start_matches(whitespace_not_newline);

    if input.starts_with("#") || input.starts_with(";") {
        input = input.trim_start_matches(|c| c != '\n' && c != '\r');
    }
    if input.starts_with("\r\n") {
        input = &input[2..];
    } else if input.starts_with('\n') || input.starts_with('\r') {
        input = &input[1..];
    } else {
        return Err(nom::Err::Error(MDLSyntaxError::ExpectedNewline(input.len())));
    }

    loop {
        let trimmed_input = input.trim_start_matches(whitespace_not_newline);
        if trimmed_input.starts_with("#") || trimmed_input.starts_with(";") || trimmed_input.starts_with("//") {
            input = trimmed_input.trim_start_matches(|c| c != '\n' && c != '\r');

            if input.starts_with("\r\n") {
                input = &input[2..];
            } else if input.starts_with('\n') || input.starts_with('\r') {
                input = &input[1..];
            }
            continue;
        } else {
            break;  // Break, discarding trimming of whitespace
        }
    }

    Ok((input, ()))
}

fn parse_block_header(header: &'static str) -> impl Fn(&str) -> IResult<&str, (), MDLSyntaxError> {
    move |input| tag(header)(input)
        .map(|(input, _tag)| (input, ()))
        .map_err(|err| err.map(|err: (&str, ErrorKind)| MDLSyntaxError::ExpectedBlockHeader(err.0.len(), header)))
}

fn parse_block_end(input: &str) -> IResult<&str, (), MDLSyntaxError> {
    tag("end")(input)
        .map(|(input, _tag)| (input, ()))
        .map_err(|err| err.map(|err: (&str, ErrorKind)| MDLSyntaxError::ExpectedBlockEnd(err.0.len())))
}

/// This function parses the whole skeletal animation data, then discards all but the first frame; First frame specifies initial position of all bones
fn parse_skeleton(input: &str) -> IResult<&str, MDLSkeleton, MDLSyntaxError> {
    map(
        tuple((
            parse_block_header("nodes"), newline_and_whitespace,
            many1(
                map(
                    tuple((parse_i32, whitespace, tag("\""), take_till(|c| c == '"'), tag("\""), whitespace, parse_i32, newline_and_whitespace)),
                    |(bone_id, _, _, bone_name, _, _, parent_bone_id, _)| {
                        MDLBone {
                            bone_id,
                            bone_name: bone_name.to_string(),
                            parent_bone_id,
                        }
                    },
                )
            ),
            parse_block_end, newline_and_whitespace,
            parse_block_header("skeleton"), newline_and_whitespace,
            many1(map(
                tuple((
                    parse_block_header("time"), whitespace, parse_i32, newline_and_whitespace,
                    many1(map(
                        tuple((
                            parse_i32, whitespace,
                            map(tuple((parse_f32, whitespace, parse_f32, whitespace, parse_f32)), |(x, _, y, _, z)| Vector3D::new([x, y, z])), whitespace,
                            map(tuple((parse_f32, whitespace, parse_f32, whitespace, parse_f32)), |(x, _, y, _, z)| Vector3D::new([x, y, z])), newline_and_whitespace
                        ))
                        , |(time, _, position, _, rotation, _)| (time, position, rotation),
                    ))
                )),
                |(_, _, time, _, animation)| (time, animation),
            )),
            parse_block_end, newline_and_whitespace,
        )),
        |(_, _, bones, _, _, _, _, animation, _, _)| {
            MDLSkeleton {
                bones,
                initial_frame: {
                    // Take first frame, which specifies bone positions, and discard all following animation frames as we currently do not support animation data
                    let (time, positions) = animation.into_iter().next().unwrap();
                    MDLAnimationFrame {
                        _time: time,
                        positions: positions.into_iter().map(
                            |(bone_id, position, rotation)| MDLAnimationPosition {
                                bone_id,
                                position,
                                rotation,
                            }
                        ).collect(),
                    }
                },
            }
        },
    )(input)
}

fn parse_bone_weights(input: &str) -> IResult<&str, Vec<(i32, FakeHash<f32>)>, MDLSyntaxError> {
    let (new_input, (weights_length, weights)) = tuple((parse_i32, many0(map(tuple((whitespace, parse_i32, whitespace, parse_f32)), |(_, bone_id, _, weight)| (bone_id, FakeHash::new(weight))))))(input)?;
    if weights_length < 0 || weights_length as usize != weights.len() {
        return Err(nom::Err::Failure(MDLSyntaxError::WeightsDoNotMatchWeightCount { remaining_input_length: input.len(), specified_len: weights_length, found_len: weights.len() }));
    } else {
        Ok((new_input, weights))
    }
}

fn parse_vertices(input: &str) -> IResult<&str, (Vec<MDLTriangle>, Vec<String>), MDLSyntaxError> {
    let mut material_name_cache = Vec::new();

    // Store parser in a variable so we can explicitly drop it before building the final Result
    let mut parser = map(
        tuple((
            parse_block_header("triangles"), newline_and_whitespace,
            many1(map(
                tuple((
                    take_till(char::is_whitespace), newline_and_whitespace,   // todo: more specific rules on material name
                    count(map(
                        tuple((
                            parse_i32, whitespace,
                            map(tuple((parse_f32, whitespace, parse_f32, whitespace, parse_f32, whitespace)), |(x, _, y, _, z, _)| [x, y, z]),
                            map(tuple((parse_f32, whitespace, parse_f32, whitespace, parse_f32, whitespace)), |(x, _, y, _, z, _)| [x, y, z]),
                            map(tuple((parse_f32, whitespace, parse_f32, whitespace)), |(u, _, v, _)| [u, v]),
                            opt(parse_bone_weights),
                            newline_and_whitespace
                        )),
                        |(| parent_bone_id, _, pos, normal, uv, weights, _)| {
                            MDLVertex {
                                parent_bone_id,
                                position: pos.map(FakeHash::new),
                                normal: normal.map(FakeHash::new),
                                uv: uv.map(FakeHash::new),
                                weights: weights.unwrap_or(Vec::new()),
                            }
                        },
                    ), 3)
                )),
                |(material_name, _, vertices)| {
                    let [vertex_1, vertex_2, vertex_3] = <[MDLVertex; 3]>::try_from(vertices).expect("MDL triangle must have 3 vertices or count() parser will fail");
                    let name_index = if let Some(name_index) = material_name_cache.iter().position(|name| &*name == material_name) {
                        name_index
                    } else {
                        material_name_cache.push(material_name.to_string());
                        material_name_cache.len() - 1
                    };
                    MDLTriangle {
                        material_index: name_index,
                        vertex_1,
                        vertex_2,
                        vertex_3,
                    }
                },
            )),
            parse_block_end, newline_and_whitespace
        )),
        |(_, _, triangles, _, _)| triangles,
    );

    let parse_result = (parser)(input);
    drop(parser);   // Parser captures material_name_cache and must be dropped before we can move material_name_cache into the result
    parse_result.map(|(input, triangles)| (input, (triangles, material_name_cache)))
}

/// This crate currently does not support vertex animation, so this information is parsed to verify SMD document validity, then discarded
fn discard_vertex_animation(input: &str) -> IResult<&str, (), MDLSyntaxError> {
    map(
        tuple((
            parse_block_header("vertexanimation"), newline_and_whitespace,
            many1(
                map(tuple((parse_i32, whitespace, tag("\""), take_till(|c| c == '"'), whitespace, parse_i32, newline_and_whitespace)), |(bone_id, _, _, bone_name, _, parent_bone_id, _)| {
                    MDLBone {
                        bone_id,
                        bone_name: bone_name.to_string(),
                        parent_bone_id,
                    }
                })
            ),
            parse_block_end, newline_and_whitespace,
            parse_block_header("skeleton"), newline_and_whitespace,
            many1(map(
                tuple((
                    parse_block_header("time"), whitespace, parse_i32, newline_and_whitespace,
                    many1(map(
                        tuple((
                            parse_i32, whitespace,
                            map(tuple((parse_f32, whitespace, parse_f32, whitespace, parse_f32)), |(x, _, y, _, z)| Vector3D::new([x, y, z])), whitespace,
                            map(tuple((parse_f32, whitespace, parse_f32, whitespace, parse_f32)), |(x, _, y, _, z)| Vector3D::new([x, y, z])), newline_and_whitespace
                        ))
                        , |(time, _, position, _, rotation, _)| (time, position, rotation),
                    )),
                    parse_block_end, newline_and_whitespace,
                )),
                |(_, _, time, _, animation, _, _)| (time, animation),
            )),
            parse_block_end, newline_and_whitespace,
        )),
        |_| (),
    )(input)
}

fn parse_mdl(input: &str) -> IResult<&str, MDLMesh, MDLSyntaxError> {
    map(
        tuple((
            tuple((parse_block_header("version 1"), newline_and_whitespace)),
            parse_skeleton,
            parse_vertices,
            opt(discard_vertex_animation)
        )),
        |(_, skeleton, (triangles, materials), _)| MDLMesh {
            skeleton,
            triangles,
            materials,
        },
    )(input)
}

/// Retrieves a line number and character number from a given byte index into the specified string (Document contents)
///
/// Debug-panics if the byte index is out of range. Outside debug returns an out of bounds character index for the last line in the document
///
/// # Arguments
///
/// * `string`: string to look into
/// * `byte_index`: index of the byte to find
///
/// returns: (usize, usize) (line number, character number within that line)
fn line_number_for_character(string: &str, byte_index: usize) -> (usize, usize) {
    debug_assert!(string.len() >= byte_index, "byte index must be within string");
    let mut line = 1;
    let mut character = byte_index;

    let mut i = 0;
    let bytes = string.as_bytes();
    while i < bytes.len() {
        if i > byte_index {
            break;
        } else if bytes[i] == b'\r' {
            line += 1;
            if bytes.len() > i + 1 && bytes[i + 1] == b'\n' {
                i += 2;
            } else {
                i += 1;
            }
            character = byte_index - i;
        } else if bytes[i] == b'\n' {
            line += 1;
            i += 1;
            character = byte_index - i;
        } else {
            i += 1;
        }
    }
    (line, character + 1)   // Column/character indices start at 1
}

/// StudioMDLData parser & generator
pub struct StudioMDLData;

impl ModelReader<&str, MDLError> for StudioMDLData {
    type Float = f32;

    fn read_model(input_document: &str, name: Option<String>) -> Result<Model<Self::Float>, MDLError> {
        match parse_mdl(input_document) {
            Ok((remaining_input, mesh)) => {
                if remaining_input.len() == 0 {
                    mesh.into_geometry_model(name.unwrap_or("UNNAMED MODEL".to_string()))
                } else {
                    let (line, character) = line_number_for_character(input_document, input_document.len().saturating_sub(remaining_input.len()));
                    Err(MDLError::DataAfterDocumentEnd { line, character })
                }
            }
            Err(err) => {
                match err {
                    nom::Err::Incomplete(_) => Err(MDLError::IncompleteDocument),
                    nom::Err::Error(e) | nom::Err::Failure(e) => {
                        let (index, message) = match e {
                            MDLSyntaxError::Nom(index) => (index, "".to_string()),
                            MDLSyntaxError::ExpectedInteger(index) => (index, "expected integer".to_string()),
                            MDLSyntaxError::ExpectedFloat(index) => (index, "expected number".to_string()),
                            MDLSyntaxError::ExpectedWhitespace(index) => (index, "expected space".to_string()),
                            MDLSyntaxError::ExpectedNewline(index) => (index, "expected newline".to_string()),
                            MDLSyntaxError::ExpectedBlockHeader(index, header) => (index, format!("expected {} block", header)),
                            MDLSyntaxError::ExpectedBlockEnd(index) => (index, "expected block end".to_string()),
                            MDLSyntaxError::WeightsDoNotMatchWeightCount { remaining_input_length, specified_len, found_len } => {
                                (remaining_input_length, format!("{} bone weights were expected but {} were found (each weight has 2 values)", specified_len, found_len))
                            }
                        };
                        let (line, character) = line_number_for_character(input_document, input_document.len().saturating_sub(index));
                        Err(MDLError::ParseError { line, character, message })
                    }
                }
            }
        }
    }
}

#[derive(Debug)]
pub enum SMDWriteError {
    IO(io::Error),
    ModelError(ModelError),
    NoMeshInModel,
    MoreThanOneMeshInModel,
    ModelNotTriangulated

}

impl From<io::Error> for SMDWriteError {
    fn from(value: io::Error) -> Self {
        SMDWriteError::IO(value)
    }
}

impl From<ModelError> for SMDWriteError {
    fn from(value: ModelError) -> Self {
        SMDWriteError::ModelError(value)
    }
}

impl<W: Write + Seek, Float: Debug + Display + Copy + GeometryNumber> ModelWriter<W, SMDWriteError, Float> for StudioMDLData {
    fn write_model(mut output: W, model: &Model<Float>) -> Result<(), SMDWriteError> {
        let mesh = if model.meshes().len() > 1 {
            return Err(SMDWriteError::MoreThanOneMeshInModel);
        } else {
            if let Some(mesh) = model.meshes().get(0) {
                mesh.view(MDL_AXES)
            } else {
                return Err(SMDWriteError::NoMeshInModel);
            }
        };

        if !mesh.is_triangulated() {
            return Err(SMDWriteError::ModelNotTriangulated);
        }

        // SMD files use either CR or CRLF but not LF newlines, and writeln! macro uses LF, so we use explicit CRLF newlines and the write! macro
        // We use CRLF for maximum compatibility with normal text editors, line endings may become configurable later if needed for compatibility with 3D graphics/modelling software.

        let write_additional_root_bone = mesh.bones().len() != 0 && mesh.bone_weights().iter().any(Vec::is_empty);

        write!(output, "version 1\r\n")?;

        write!(output, "nodes\r\n")?;
        if mesh.bones().len() == 0 {  // If no bones are specified, create a new root bone to parent vertices to
            write!(output, "0 \"root\" -1\r\n")?;
        } else {
            for (index, bone) in mesh.bones().enumerate() {
                // SMD only reads names until the first double-quote mark; We can't escape these, so instead double-quotes are replaced with single-quotes
                write!(output, "{} \"{}\" {}\r\n", index, bone.name().replace('\"', "'"), bone.parent().map(|id| id as isize).unwrap_or(-1))?;
            }
            if write_additional_root_bone {
                write!(output, "{} \"multimesh_unparented_vertices\" -1\r\n", mesh.bones().len())?;
            }
        }
        write!(output, "end\r\n")?;

        write!(output, "skeleton\r\n")?;
        write!(output, "time 0\r\n")?;
        if mesh.bones().len() == 0 {  // If no bones are specified, create a new bone to parent vertices to
            write!(output, "0 \t 0 0 0 \t 0 0 0\r\n")?;
        } else {
            for (index, bone) in mesh.bones().enumerate() {
                // TODO: SMD rotations are poorly documented, implementation below only provides appropriate types. Verify which rotations SMD expects and convert from euler angles
                let [euler_x, euler_y, euler_z] = bone.rotation().euler_factors();
                let [pos_x, pos_y, pos_z] = bone.position().to_array();
                write!(output, "{} \t {} {} {} \t {} {} {}\r\n", index, pos_x, pos_y, pos_z, euler_x, euler_y, euler_z)?;
            }
            // If we have any vertices with no bone weights, create an additional bone to parent those to
            if write_additional_root_bone {
                write!(output, "{} \t 0 0 0 \t 0 0 0\r\n", mesh.bones().len())?;
            }
        }
        write!(output, "end\r\n")?;

        write!(output, "triangles\r\n")?;

            for (idx, face) in mesh.triangles().unwrap().enumerate() {
                // SMD files must have face vertices listed in clockwise order
                // geometry::Mesh has no requirements on the order of face indices, so instead we check whether the existing order matches the normal vector of the vertices
                // If vertex normals point in the opposite direction, face is flipped by swapping the order of the first and third vertex
                let face_values = {
                    let [norm_1, norm_2, norm_3] = face.normal_vectors();
                    let vertex_normal_sum: Vector3D<Float> = norm_1 + norm_2 + norm_3;


                    let [vertex_1, vertex_2, vertex_3] = face.vertices();

                    let face_normal = (vertex_1 - vertex_2).cross_product(vertex_3 - vertex_2);
                    if vertex_normal_sum.dot(face_normal) < Float::from_int(0) {
                        let [one, two, three] = face.values();
                        [three, two, one]
                    } else {
                        face.values()
                    }
                };

                write!(output, "{}\r\n", mesh.face_material(idx)
                    .and_then(|material_id| model.materials().get(material_id))
                    .and_then(Material::name)
                    .unwrap_or("no_material"))?;

                for value in face_values {
                    let [pos_x, pos_y, pos_z] = value.vertex.to_array();
                    let [norm_x, norm_y, norm_z] = value.normal.to_array();
                    let [u, v] = value.uv.to_array();

                    let mut weights = mesh.bone_weights().get(value.vertex_index).map(Vec::clone).unwrap_or_else(Vec::new);
                    weights.sort_unstable_by(|(_, weight_1), (_, weight_2)| {
                        weight_1.partial_cmp(weight_2)
                            .map(Ordering::reverse) // Sort greatest to smallest weight
                            .unwrap_or_else(|| {
                                match (weight_1 != weight_1, weight_2 != weight_2) {   // Override NaN sorting to provide consistent behaviour even if NaN is present
                                    (true, true) => Ordering::Equal,
                                    (false, true) => Ordering::Less,
                                    (true, false) => Ordering::Greater,
                                    (false, false) => Ordering::Equal
                                }
                            })
                    });
                    let parent_bone_id = weights.get(0)
                        .map(|(bone_id, _weight)| *bone_id)
                        // If vertex has no weights assigned, assign it to
                        .unwrap_or(if mesh.bones().len() == 0 {
                            0
                        } else {
                            mesh.bones().len()
                        });

                    write!(output, "{}\t{} {} {}\t{} {} {}\t{} {}", parent_bone_id, pos_x, pos_y, pos_z, norm_x, norm_y, norm_z, u, v)?;
                    if weights.len() > 1 {
                        write!(output, "\t{}", weights.len() - 1)?;
                        for (bone_id, weight) in &weights[2..] {
                            write!(output, " {} {}", bone_id, weight)?;
                        }
                    }
                    write!(output, "\r\n")?;
                }
            }

        write!(output, "end\r\n")?;
        Ok(())
    }
}