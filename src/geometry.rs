//! General geometry types for 3D graphics and 2D textures

use std::iter::{Sum};
use std::mem;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

/// Trait for supported number types
/// Implementation provided for `f32` and `f64`
pub trait GeometryNumber:
    Sized
    + Neg<Output=Self>
    + Add<Self, Output=Self>
    + AddAssign<Self>
    + Sub<Self, Output=Self>
    + SubAssign<Self>
    + Mul<Self, Output=Self>
    + MulAssign<Self>
    + Div<Self, Output=Self>
    + DivAssign<Self>
    + PartialOrd
    + PartialEq
    + Sum
{
    /// Pi constant
    ///
    /// Ratio constants can be obtained through [`GeometryNumber::from_int`] and division
    const PI: Self;

    /// Convert an integer value to this Number type
    /// Primarily used for constructing constant values
    fn from_int(int: i32) -> Self;
    /// Square root, equivalent to [`f64::sqrt`]
    fn sqrt(self) -> Self;
    /// Arc-cosine, equivalent to [`f64::acos`]
    fn acos(self) -> Self;
    /// Arc-sine, equivalent to [`f64::asin`]
    fn asin(self) -> Self;
    /// Arc-tangent, equivalent to [`f64::atan2`]
    fn atan2(lhs: Self, rhs: Self) -> Self;
}

impl GeometryNumber for f32 {
    const PI: Self = std::f32::consts::PI;

    fn from_int(int: i32) -> Self {
        int as f32
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn acos(self) -> Self {
        self.acos()
    }

    fn asin(self) -> Self {
        self.asin()
    }

    fn atan2(lhs: Self, rhs: Self) -> Self {
        f32::atan2(lhs, rhs)
    }
}

impl GeometryNumber for f64 {
    const PI: Self = std::f64::consts::PI;

    fn from_int(int: i32) -> Self {
        int as f64
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn acos(self) -> Self {
        self.acos()
    }

    fn asin(self) -> Self {
        self.asin()
    }

    fn atan2(lhs: Self, rhs: Self) -> Self {
        f64::atan2(lhs, rhs)
    }
}

/// N-dimensional vector
///
/// Caution: Vectors are context-sensitive; Vectors from one context may not share the same coordinate system as vectors from another.
/// E.g. Vectors from a Y-up coordinate system, and vectors from a Z-up coordinate system may both be stored as [X, Y, Z]
///
/// [`Model`] and [`Mesh`] provide a view which translates into the desired context.
///
/// Type aliases for 2D and 3D ([`Vector3D`] , [`Vector2D`]) are provided by this module
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct VectorN<T, const N: usize> {
    array: [T; N],
}

impl<T, const N: usize> VectorN<T, N> {
    /// Create a new vector from an array of components
    ///
    /// Vectors retain the order in which components were passed in.
    ///
    /// # Arguments
    ///
    /// * `array`: Array of components
    ///
    /// returns: VectorN<T, { N }>
    pub fn new(array: [T; N]) -> Self { VectorN { array } }

    /// Convert to a borrowed array
    pub fn as_array(&self) -> &[T; N] { &self.array }

    /// Unwrap to array
    pub fn to_array(self) -> [T; N] { self.array }

    /// Performs an element-wise operation on each component of this vector
    ///
    /// # Arguments
    ///
    /// * `f`: Operation to apply
    ///
    /// returns: VectorN<U, { N }>
    ///
    /// # Examples
    ///
    /// ```
    /// let vector = Vector3D::new([1, 2, 3]);
    /// let scaled = x.map(|v| v * 2);
    /// assert_eq!(scaled, [2, 4, 6]);
    /// ```
    #[inline]
    pub fn map<U, F: FnMut(T) -> U>(self, f: F) -> VectorN<U, N> {
        VectorN { array: self.array.map(f) }
    }

    /// Performs a pair-wise operation on each component of this, and another, vector
    ///
    /// # Arguments
    ///
    /// * `rhs`: Right-hand side vector
    /// * `f`: Operation to apply
    ///
    /// returns: VectorN<V, { N }>
    ///
    /// # Examples
    ///
    /// ```
    /// let left_vector = Vector3D::new([1, 2, 3]);
    /// let right_vector = Vector3D::new([4, 5, 6]);
    /// let sum_vector = left_vector.map_pairwise(right_vector, |lhs, rhs| lhs + rhs);
    /// assert_eq!(sum_vector, [5, 7, 9]);
    /// ```
    pub fn map_pairwise<U, V>(self, rhs: VectorN<U, N>, f: fn(T, U) -> V) -> VectorN<V, N> {
        let mut iter = self.array.into_iter()
            .zip(rhs.array.into_iter())
            .map(|(lhs, rhs)| (f)(lhs, rhs));
        VectorN::new(std::array::from_fn(|_index| iter.next().unwrap()))    // This is a stupid hack, but for small arrays this optimizes well; Using MaybeUninit to consume the arrays is unstable and clunky
    }
}

impl<T: Neg, const N: usize> Neg for VectorN<T, N> {
    type Output = VectorN<<T as Neg>::Output, N>;

    fn neg(self) -> Self::Output {
        self.map(T::neg)
    }
}

impl<T: Add, const N: usize> Add for VectorN<T, N> {
    type Output = VectorN<<T as Add>::Output, N>;

    fn add(self, rhs: Self) -> Self::Output {
        self.map_pairwise(rhs, |l, r| l + r)
    }
}

impl<T: AddAssign, const N: usize> AddAssign for VectorN<T, N> {
    fn add_assign(&mut self, rhs: Self) {
        self.array.iter_mut()
            .zip(rhs.array.into_iter())
            .for_each(|(lhs, rhs)| lhs.add_assign(rhs));
    }
}

impl<T: Sub, const N: usize> Sub for VectorN<T, N> {
    type Output = VectorN<<T as Sub>::Output, N>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.map_pairwise(rhs, |l, r| l - r)
    }
}

impl<T: SubAssign + Copy, const N: usize> SubAssign for VectorN<T, N> {
    fn sub_assign(&mut self, rhs: Self) {
        self.array.iter_mut()
            .zip(rhs.array.into_iter())
            .for_each(|(lhs, rhs)| lhs.sub_assign(rhs));
    }
}

// T has to be copy here as multiplication between T and Vector<T> is element-wise; We need N copies of T
// This could be reduced to clone if there's need for a non-copy numerical type
impl<T: Mul<Output=T> + Copy, const N: usize> Mul<T> for VectorN<T, N> {
    type Output = VectorN<T, N>;

    fn mul(self, rhs: T) -> Self::Output {
        VectorN::new(self.array.map(|n| n * rhs))
    }
}

impl<T: MulAssign<T> + Copy, const N: usize> MulAssign<T> for VectorN<T, N> {
    fn mul_assign(&mut self, rhs: T) {
        self.array.iter_mut().for_each(|n| T::mul_assign(n, rhs))
    }
}


// T has to be copy here as division between T and Vector<T> is element-wise; We need N copies of T
// This could be reduced to clone if there's need for a non-copy numerical type
impl<T: Div<Output=T> + Copy, const N: usize> Div<T> for VectorN<T, N> {
    type Output = VectorN<T, N>;

    fn div(self, rhs: T) -> Self::Output {
        VectorN::new(self.array.map(|n| n / rhs))
    }
}

impl<T: DivAssign<T> + Copy, const N: usize> DivAssign<T> for VectorN<T, N> {
    fn div_assign(&mut self, rhs: T) {
        self.array.iter_mut().for_each(|n| T::div_assign(n, rhs))
    }
}

impl<T, const N: usize, Idx> Index<Idx> for VectorN<T, N> where [T; N]: Index<Idx> {
    type Output = <[T; N] as Index<Idx>>::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.array[index]
    }
}

impl<T, const N: usize, Idx> IndexMut<Idx> for VectorN<T, N> where [T; N]: IndexMut<Idx> {
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
        &mut self.array[index]
    }
}

impl<T, const N: usize> IntoIterator for VectorN<T, N> {
    type Item = <[T; N] as IntoIterator>::Item;
    type IntoIter = <[T; N] as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.array.into_iter()
    }
}

// T has to be copy; Calculating T^2 requires two copies of T.
impl<T: Copy + GeometryNumber, const N: usize> VectorN<T, N> {
    /// Calculates the magnitude of this vector
    pub fn magnitude(self) -> T {
        self.into_iter()
            .map(|value| value * value)
            .sum::<T>()
            .sqrt()
    }
}

impl<T: GeometryNumber, const N: usize> VectorN<T, N> {
    /// Calculates the "scalar" dot product between this and another equally-sized vector
    pub fn dot(self, rhs: Self) -> T {
        self.into_iter()
            .zip(rhs.into_iter())
            .map(|(lhs, rhs)| lhs * rhs)
            .sum::<T>()
    }
}

/// Type alias for 3 dimensional [`VectorN`]
pub type Vector3D<T> = VectorN<T, 3>;

// T has to be copy; Cross product needs 2 of each T
impl<T: Copy + Mul<Output=T> + Sub<Output=T>> Vector3D<T> {
    /// Calculates vector cross product `self × rhs`
    ///
    /// # Arguments
    ///
    /// * `rhs`: Right hand side
    ///
    /// returns: VectorN<T, 3>
    pub fn cross_product(self, rhs: Self) -> Self {
        let [r_1, r_2, r_3] = self.array;
        let [l_1, l_2, l_3] = rhs.array;
        Vector3D::new([
            (r_2 * l_3) - (r_3 * l_2),
            (r_3 * l_1) - (r_1 * l_3),
            (r_1 * l_2) - (r_2 * l_1)
        ])
    }
}

/// Type alias for 2 dimensional [`VectorN`]
pub type Vector2D<T> = VectorN<T, 2>;

/// 3D rotation matrix
///
/// Rotations are performed "pre-multiplied" with column vectors when using row-major matrices ([`RotMatrix::from_row_major`])
/// ```text
/// [[r11, r12, r13],   ⎡x⎤   ⎡(r11 * x) + (r12 * y) + (r13 * z)⎤
///  [r21, r22, r23], . ⎢y⎥ = ⎢(r21 * x) + (r22 * y) + (r23 * z)⎥
///  [r31, r32, r33]]   ⎣z⎦   ⎣(r31 * x) + (r32 * y) + (r33 * z)⎦
/// ```
///
/// Matrix multiplication is performed through the [`Mul`] trait.
///
/// Matrices are represented internally as column-major arrays of Vector3D, such that:
///
/// ```text
/// ⎡⎡r11⎤ ⎡r12⎤ ⎡r13⎤⎤   ⎡x⎤
/// ⎢⎢r21⎥ ⎢r22⎥ ⎢r23⎥⎥ . ⎢y⎥
/// ⎣⎣r31⎦ ⎣r32⎦ ⎣r33⎦⎦   ⎣z⎦
///
///   ⎛    ⎡r11⎤⎞   ⎛    ⎡r12⎤⎞   ⎛    ⎡r13⎤⎞
/// = ⎜x * ⎢r21⎥⎟ + ⎜y * ⎢r22⎥⎟ + ⎜z * ⎢r23⎥⎟
///   ⎝    ⎣r31⎦⎠   ⎝    ⎣r32⎦⎠   ⎝    ⎣r33⎦⎠
///
///   ⎡r11 * x⎤   ⎡r12 * y⎤   ⎡r13 * z⎤
/// = ⎢r21 * x⎥ + ⎢r22 * y⎥ + ⎢r23 * z⎥
///   ⎣r31 * x⎦   ⎣r32 * y⎦   ⎣r33 * z⎦
///
///   ⎡(r11 * x) + (r12 * y) + (r13 * z)⎤
/// = ⎢(r21 * x) + (r22 * y) + (r23 * z)⎥
///   ⎣(r31 * x) + (r32 * y) + (r33 * z)⎦
/// ```
///
/// Caution: Because of this difference in representation, this type should never be destructured directly.
/// Instead, use [`RotMatrix::to_row_major`] to ensure layout matches source code.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RotMatrix<T>([Vector3D<T>; 3]);

impl<T: GeometryNumber + Copy> RotMatrix<T> {
    /// Apply this rotation to the specified vector
    ///
    /// Rotations are performed "pre-multiplied" with column vectors, when using row-major matrices ([`RotMatrix::from_row_major`])
    /// ```text
    /// [[r11, r12, r13],   ⎡x⎤   ⎡(r11 * x) + (r12 * y) + (r13 * z)⎤
    ///  [r21, r22, r23], . ⎢y⎥ = ⎢(r21 * x) + (r22 * y) + (r23 * z)⎥
    ///  [r31, r32, r33]]   ⎣z⎦   ⎣(r31 * x) + (r32 * y) + (r33 * z)⎦
    /// ```
    /// # Arguments
    ///
    /// * `vector`: Vector to rotate
    ///
    /// returns: VectorN<T, 3>
    pub fn apply(self, vector: Vector3D<T>) -> Vector3D<T> {
        let [matrix_x, matrix_y, matrix_z] = self.0;
        let [x, y, z] = vector.array;
        (matrix_x * x) + (matrix_y * y) + (matrix_z * z)
    }

    /// Construct a new rotation matrix from a row-major set of 3x3 arrays
    ///
    /// # Arguments
    ///
    /// * `matrix`: Matrix data
    ///
    /// returns: RotMatrix<T>
    ///
    /// # Examples
    /// ```
    /// RotMatrix::from_row_major([
    ///     [1.0, 0.0, 0.0],
    ///     [0.0, x.cos(), -x.sin()],
    ///     [0.0, x.sin(), x.cos()]
    /// ])
    /// ```
    /// Results in
    /// ```text
    /// ⎡1.0  0.0    0.0   ⎤
    /// ⎢0.0 cos(x) -sin(x)⎥
    /// ⎣0.0 sin(x) cos(x) ⎦
    /// ```
    #[inline]   // Inlining is likely to optimize the transposition away
    pub fn from_row_major(matrix: [[T; 3]; 3]) -> RotMatrix<T> {
        let [
            [r11, r12, r13],
            [r21, r22, r23],
            [r31, r32, r33]
        ] = matrix;

        RotMatrix([   // Caution: This is intentionally transposed as the vectors are in column-major order
            Vector3D::new([r11, r21, r31]),
            Vector3D::new([r12, r22, r32]),
            Vector3D::new([r13, r23, r33])
        ])
    }

    /// Construct a new rotation matrix from a row-major set of 3x3 arrays
    ///
    /// # Arguments
    ///
    /// * `matrix`: Matrix data
    ///
    /// returns: RotMatrix<T>
    ///
    /// # Examples
    /// Given `matrix`
    /// ```text
    /// ⎡1.0  0.0    0.0   ⎤
    /// ⎢0.0 cos(x) -sin(x)⎥
    /// ⎣0.0 sin(x) cos(x) ⎦
    /// ```
    ///
    /// ```
    /// let [
    ///     [r11, r12, r13],
    ///     [r21, r22, r23],
    ///     [r31, r32, r33]
    /// ] = matrix.to_row_major();
    ///
    /// assert_eq!(r11, 1.0);
    /// assert_eq!(r12, 0.0);
    /// assert_eq!(r13, 0.0);
    ///
    /// assert_eq!(r21, 0.0);
    /// assert_eq!(r22, x.cos());
    /// assert_eq!(r23, -x.sin());
    ///
    /// assert_eq!(r31, 0.0);
    /// assert_eq!(r32, x.sin());
    /// assert_eq!(r33, x.cos());
    /// ```
    #[inline]   // Inlining is likely to optimize the transposition away
    pub fn to_row_major(self) -> [[T; 3]; 3] {
        let [   // Caution: This destructuring is intentionally transposed as the vectors are in column-major order
            Vector3D { array: [r11, r21, r31] },
            Vector3D { array: [r12, r22, r32] },
            Vector3D { array: [r13, r23, r33] }
        ] = self.0;

        [
            [r11, r12, r13],
            [r21, r22, r23],
            [r31, r32, r33]
        ]
    }

    /**
     * Factorization into extrinsic euler angles, in order X\*Y\*Z
     * TODO: Currently a stub implementation to provide values of appropriate type
     */
    // 'pub(crate)' until rotations are confirmed to be correct.
    // Algorithm from https://www.geometrictools.com/Documentation/EulerAngles.pdf
    pub(crate) fn euler_factors(self) -> [T; 3] {
        let [
            [r11, r12, r13],
            [r21, r22, r23],
            [_r31, _r32, r33]
        ] = self.to_row_major();

        if r13 < T::from_int(1) {
            if r13 > T::from_int(-1) {
                [
                    r13.asin(),
                    T::atan2(-r23, r33),
                    T::atan2(-r12, r11)
                ]
            } else {
                // Gimbal lock
                [
                    -T::PI/T::from_int(2),
                    -T::atan2(r21, r22),
                    T::from_int(0)
                ]
            }
        } else {
            // Gimbal lock
            [
                T::PI/T::from_int(2),
                T::atan2(r21, r22),
                T::from_int(0)
            ]
        }
    }
}

impl<T: GeometryNumber + Copy> Mul for RotMatrix<T> {
    type Output = RotMatrix<T>;

    #[inline]   // Multiplication with literals can often be (partially) optimized
    fn mul(self, rhs: Self) -> Self::Output {
        let [
            [a11, a12, a13],
            [a21, a22, a23],
            [a31, a32, a33]
        ] = self.to_row_major();
        let [
            [b11, b12, b13],
            [b21, b22, b23],
            [b31, b32, b33]
        ] = rhs.to_row_major();

        RotMatrix([
            Vector3D::new([a11 * b11 + a12 * b21 + a13 * b31, a11 * b12 + a12 * b22 + a13 * b32, a11 * b13 + a12 * b23 + a13 * b33]),
            Vector3D::new([a21 * b11 + a22 * b21 + a23 * b31, a21 * b12 + a22 * b22 + a23 * b32, a21 * b13 + a22 * b23 + a23 * b33]),
            Vector3D::new([a31 * b11 + a32 * b21 + a23 * b31, a31 * b12 + a32 * b22 + a33 * b32, a31 * b13 + a32 * b23 + a33 * b33])
        ])
    }
}

/// Cardinal direction enum
///
/// Used in conjunction with [`CoordinateDirections3D`] to encode coordinate system directions.
///
/// bool indicates whether a direction is in the negative of an axis. (E.g. True for -X, -Y, or -Z. False for +X, +Y or +Z)
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Eq, Debug)]   // Partialeq is provided as a const impl until const-derive is a thing
pub enum Direction {
    Up { axis_is_negative: bool },
    Right { axis_is_negative: bool },
    Forward { axis_is_negative: bool },
}

impl Direction {
    pub const fn opposite(self) -> Self {
        match self {
            Direction::Up { axis_is_negative: is_negative } => Direction::Up { axis_is_negative: !is_negative },
            Direction::Right { axis_is_negative: is_negative } => Direction::Right { axis_is_negative: !is_negative },
            Direction::Forward { axis_is_negative: is_negative } => Direction::Forward { axis_is_negative: !is_negative }
        }
    }
}

impl const PartialEq for Direction {
    fn eq(&self, other: &Self) -> bool {
        match self {
            Direction::Up { axis_is_negative: is_negative } => if let Direction::Up { axis_is_negative: is_other_negative } = other { *is_negative == *is_other_negative } else { false }
            Direction::Right { axis_is_negative: is_negative } => if let Direction::Right { axis_is_negative: is_other_negative } = other { *is_negative == *is_other_negative } else { false }
            Direction::Forward { axis_is_negative: is_negative } => if let Direction::Forward { axis_is_negative: is_other_negative } = other { *is_negative == *is_other_negative } else { false }
        }
    }
}

/// Struct to encode the directions of the axes in a 3D coordinate system
///
/// # Example
///
/// For axes [X, Y, Z]
///
/// Y up -Z forward:
///
/// [Direction::Right_East { axis_is_negative: false }, Direction::Up { axis_is_negative: false }, Direction::Forward_North { axis_is_negative: true }]
///
/// Y forward, Z up:
///
/// [Direction::Right_East { axis_is_negative: false }, Direction::Forward_North { axis_is_negative: false }, Direction::Up { axis_is_negative: false }]
///
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct CoordinateDirections3D {
    one: Direction,
    two: Direction,
    three: Direction
}

impl CoordinateDirections3D {
    /// Create a new coordinate-directions value at const compile time
    ///
    /// Panics if an invalid set of directions is passed
    ///
    /// # Arguments
    ///
    /// * `directions`: Directions in order of their axes'
    ///
    /// returns: CoordinateDirections3D
    pub const fn new_const(directions: [Direction; 3]) -> Self {
        if directions[0] == directions[1] || directions[0] == directions[1].opposite() {
            panic!("overlapping directions 0 and 1")
        } else if directions[1] == directions[2] || directions[1] == directions[2].opposite() {
            panic!("overlapping directions 1 and 2")
        } else if directions[0] == directions[2] || directions[0] == directions[2].opposite() {
            panic!("overlapping directions 0 and 2")
        } else {
            CoordinateDirections3D {
                one: directions[0],
                two: directions[1],
                three: directions[2],
            }
        }
    }

    /// Create a new coordinate-directions value at runtime
    ///
    /// Yields and error if an invalid set of directions is passed
    ///
    /// # Arguments
    ///
    /// * `directions`: Directions in order of their axes'
    ///
    /// returns: CoordinateDirections3D
    pub fn new(directions: [Direction; 3]) -> Result<Self, &'static str> {
        if mem::discriminant(&directions[0]) == mem::discriminant(&directions[1]) {
            Err("overlapping directions 0 and 1")
        } else if mem::discriminant(&directions[1]) == mem::discriminant(&directions[2]) {
            Err("overlapping directions 1 and 2")
        } else if mem::discriminant(&directions[0]) == mem::discriminant(&directions[2]) {
            Err("overlapping directions 0 and 2")
        } else {
            Ok(CoordinateDirections3D {
                one: directions[0],
                two: directions[1],
                three: directions[2],
            })
        }
    }


    /// Returns a mapper function to convert vectors from one set of directions to another
    ///
    /// 'const' mapper functions may be inlined better
    ///
    /// # Arguments
    ///
    /// * `from`: Direction context for the vector input
    /// * `to`: Direction context for the vector output
    ///
    /// returns: fn(&mut [T; 3])
    pub const fn mapper_fn<T: Copy + Neg<Output=T>>(from: Self, to: Self) -> fn(&mut [T; 3]) {
        const fn index_of(direction: Direction, coordinates: CoordinateDirections3D) -> (u8, bool) {
            match direction {
                Direction::Up { axis_is_negative: is_dir_negative } => {
                    if let Direction::Up { axis_is_negative: is_coord_negative } = coordinates.one {
                        (1, is_dir_negative != is_coord_negative)
                    } else if let Direction::Up { axis_is_negative: is_coord_negative } = coordinates.two {
                        (2, is_dir_negative != is_coord_negative)
                    } else if let Direction::Up { axis_is_negative: is_coord_negative } = coordinates.three {
                        (3, is_dir_negative != is_coord_negative)
                    } else {
                        unreachable!()
                    }
                }
                Direction::Right { axis_is_negative: is_dir_negative } => {
                    if let Direction::Right { axis_is_negative: is_coord_negative } = coordinates.one {
                        (1, is_dir_negative != is_coord_negative)
                    } else if let Direction::Right { axis_is_negative: is_coord_negative } = coordinates.two {
                        (2, is_dir_negative != is_coord_negative)
                    } else if let Direction::Right { axis_is_negative: is_coord_negative } = coordinates.three {
                        (3, is_dir_negative != is_coord_negative)
                    } else {
                        unreachable!()
                    }
                }
                Direction::Forward { axis_is_negative: is_dir_negative } => {
                    if let Direction::Forward { axis_is_negative: is_coord_negative } = coordinates.one {
                        (1, is_dir_negative != is_coord_negative)
                    } else if let Direction::Forward { axis_is_negative: is_coord_negative } = coordinates.two {
                        (2, is_dir_negative != is_coord_negative)
                    } else if let Direction::Forward { axis_is_negative: is_coord_negative } = coordinates.three {
                        (3, is_dir_negative != is_coord_negative)
                    } else {
                        unreachable!()
                    }
                }
            }
        }


        let (index_1, invert_one) = index_of(to.one, from);
        let (index_2, invert_two) = index_of(to.two, from);
        let (index_3, invert_three) = index_of(to.three, from);

        // The following macro generates a "cross-table" match, combining the order change and negation functions into a single function pointer
        // Commented out because of impact on compile times; Commented out block to be removed when included in version control and output to be confirmed correct

        // macro_rules! cross_match {
        //     (
        //         match $scrutinee_one:expr => {
        //             $($pattern_one:pat => $func_one:expr,)+
        //         }
        //         match $scrutinee_two:expr => {
        //             $($pattern_two:pat => $func_two:expr,)+
        //         }
        //     ) => {
        //         cross_match!(@generate: ($scrutinee_one, $scrutinee_two); $($pattern_one => $func_one,)+; $($pattern_two => $func_two,)+;)
        //     };
        //
        //     (@generate: $scrutinee:expr; $head:pat => $head_func:expr, $($tail:pat => $tail_func:expr,)*; $($two:pat => $func_two:expr,)*; $($arms:tt)*) => {
        //         cross_match!(@generate: $scrutinee; $($tail => $tail_func,)*; $($two => $func_two,)*; $(($head, $two) => |array| {$head_func(array);$func_two(array);},)* $($arms)*)
        //     };
        //
        //     (@generate: $scrutinee:expr;; $($two:pat => $func_two:expr,)*; $($arms:tt)*) => {
        //         match $scrutinee {
        //             $($arms)*
        //             _ => unreachable!()
        //         }
        //     }
        // }
        //
        // let func_ptr: fn(&mut [T; 3]) = cross_match!(
        //     match (index_1, index_2, index_3) => {
        //         (1, 2, 3) => |_array: &mut [T; 3]| {},
        //         (1, 3, 2) => |[_one, two, three]: &mut [T; 3]| mem::swap(two, three),
        //         (2, 1, 3) => |[one, two, _three]: &mut [T; 3]| mem::swap(one, two),
        //         (3, 2, 1) => |[one, _two, three]: &mut [T; 3]| mem::swap(one, three),
        //         (2, 3, 1) => |[one, two, three]: &mut [T; 3]| {mem::swap(one, three); mem::swap(two, three)},
        //         (3, 1, 2) => |[one, two, three]: &mut [T; 3]| {mem::swap(one, two); mem::swap(two, three)},
        //     }
        //
        //     match (invert_one, invert_two, invert_three) => {
        //         (false, false, false) => |_array: &mut [T; 3]| {},
        //         (false, false, true) => |[_one, _two, three]: &mut [T; 3]| *three = -*three,
        //         (false, true, false) => |[_one, two, _three]: &mut [T; 3]| *two = -*two,
        //         (false, true, true) => |[_one, two, three]: &mut [T; 3]| { *two = -*two; *three = -*three},
        //         (true, false, false) => |[one, _two, _three]: &mut [T; 3]| *one = -*one,
        //         (true, false, true) => |[one, _two, three]: &mut [T; 3]| { *one = -*one; *three = -*three },
        //         (true, true, false) => |[one, two, _three]: &mut [T; 3]| { *one = -*one; *two = -*two },
        //         (true, true, true) => |[one, two, three]: &mut [T; 3]| { *one = -*one; *two = -*two; *three = -*three },
        //     }
        // );

        // Expanded (and cleaned) version of commented-out macro above. Included for compile-performance
        #[allow(unused)]
        let func_ptr: fn(&mut [T; 3]) = match ((index_1, index_2, index_3), (invert_one, invert_two, invert_three)) {
            ((3, 1, 2), (false, false, false)) => |[one, two, three]| { mem::swap(one, two); mem::swap(two, three); },
            ((3, 1, 2), (false, false, true)) => |[one, two, three]| { mem::swap(one, two); mem::swap(two, three); *three = -*three; },
            ((3, 1, 2), (false, true, false)) => |[one, two, three]| { mem::swap(one, two); mem::swap(two, three); *two = -*two; },
            ((3, 1, 2), (false, true, true)) => |[one, two, three]| { mem::swap(one, two); mem::swap(two, three); *two = -*two; *three = -*three; },
            ((3, 1, 2), (true, false, false)) => |[one, two, three]| { mem::swap(one, two); mem::swap(two, three); *one = -*one; },
            ((3, 1, 2), (true, false, true)) => |[one, two, three]| { mem::swap(one, two); mem::swap(two, three); *one = -*one; *three = -*three; },
            ((3, 1, 2), (true, true, false)) => |[one, two, three]| { mem::swap(one, two); mem::swap(two, three); *one = -*one; *two = -*two; },
            ((3, 1, 2), (true, true, true)) => |[one, two, three]| { mem::swap(one, two); mem::swap(two, three); *one = -*one; *two = -*two; *three = -*three; },
            ((2, 3, 1), (false, false, false)) => |[one, two, three]| { mem::swap(one, three); mem::swap(two, three); },
            ((2, 3, 1), (false, false, true)) => |[one, two, three]| { mem::swap(one, three); mem::swap(two, three); *three = -*three; },
            ((2, 3, 1), (false, true, false)) => |[one, two, three]| { mem::swap(one, three); mem::swap(two, three); *two = -*two; },
            ((2, 3, 1), (false, true, true)) => |[one, two, three]| { mem::swap(one, three); mem::swap(two, three); *two = -*two; *three = -*three; },
            ((2, 3, 1), (true, false, false)) => |[one, two, three]| { mem::swap(one, three); mem::swap(two, three); *one = -*one; },
            ((2, 3, 1), (true, false, true)) => |[one, two, three]| { mem::swap(one, three); mem::swap(two, three); *one = -*one; *three = -*three; },
            ((2, 3, 1), (true, true, false)) => |[one, two, three]| { mem::swap(one, three); mem::swap(two, three); *one = -*one; *two = -*two; },
            ((2, 3, 1), (true, true, true)) => |[one, two, three]| { mem::swap(one, three); mem::swap(two, three); *one = -*one; *two = -*two; *three = -*three; },
            ((3, 2, 1), (false, false, false)) => |[one, two, three]| { mem::swap(one, three); },
            ((3, 2, 1), (false, false, true)) => |[one, two, three]| { mem::swap(one, three); *three = -*three; },
            ((3, 2, 1), (false, true, false)) => |[one, two, three]| { mem::swap(one, three); *two = -*two; },
            ((3, 2, 1), (false, true, true)) => |[one, two, three]| { mem::swap(one, three); *two = -*two; *three = -*three; },
            ((3, 2, 1), (true, false, false)) => |[one, two, three]| { mem::swap(one, three); *one = -*one; },
            ((3, 2, 1), (true, false, true)) => |[one, two, three]| { mem::swap(one, three); *one = -*one; *three = -*three; },
            ((3, 2, 1), (true, true, false)) => |[one, two, three]| { mem::swap(one, three); *one = -*one; *two = -*two; },
            ((3, 2, 1), (true, true, true)) => |[one, two, three]| { mem::swap(one, three); *one = -*one; *two = -*two; *three = -*three; },
            ((2, 1, 3), (false, false, false)) => |[one, two, three]| { mem::swap(one, two); },
            ((2, 1, 3), (false, false, true)) => |[one, two, three]| { mem::swap(one, two); *three = -*three; },
            ((2, 1, 3), (false, true, false)) => |[one, two, three]| { mem::swap(one, two); *two = -*two; },
            ((2, 1, 3), (false, true, true)) => |[one, two, three]| { mem::swap(one, two); *two = -*two; *three = -*three; },
            ((2, 1, 3), (true, false, false)) => |[one, two, three]| { mem::swap(one, two); *one = -*one; },
            ((2, 1, 3), (true, false, true)) => |[one, two, three]| { mem::swap(one, two); *one = -*one; *three = -*three; },
            ((2, 1, 3), (true, true, false)) => |[one, two, three]| { mem::swap(one, two); *one = -*one; *two = -*two; },
            ((2, 1, 3), (true, true, true)) => |[one, two, three]| { mem::swap(one, two); *one = -*one; *two = -*two; *three = -*three; },
            ((1, 3, 2), (false, false, false)) => |[one, two, three]| { mem::swap(two, three); },
            ((1, 3, 2), (false, false, true)) => |[one, two, three]| { mem::swap(two, three); *three = -*three; },
            ((1, 3, 2), (false, true, false)) => |[one, two, three]| { mem::swap(two, three); *two = -*two; },
            ((1, 3, 2), (false, true, true)) => |[one, two, three]| { mem::swap(two, three); *two = -*two; *three = -*three; },
            ((1, 3, 2), (true, false, false)) => |[one, two, three]| { mem::swap(two, three); *one = -*one; },
            ((1, 3, 2), (true, false, true)) => |[one, two, three]| { mem::swap(two, three); *one = -*one; *three = -*three; },
            ((1, 3, 2), (true, true, false)) => |[one, two, three]| { mem::swap(two, three); *one = -*one; *two = -*two; },
            ((1, 3, 2), (true, true, true)) => |[one, two, three]| { mem::swap(two, three); *one = -*one; *two = -*two; *three = -*three; },
            ((1, 2, 3), (false, false, false)) => |[one, two, three]| { },
            ((1, 2, 3), (false, false, true)) => |[one, two, three]| { *three = -*three; },
            ((1, 2, 3), (false, true, false)) => |[one, two, three]| { *two = -*two; },
            ((1, 2, 3), (false, true, true)) => |[one, two, three]| { *two = -*two; *three = -*three; },
            ((1, 2, 3), (true, false, false)) => |[one, two, three]| { *one = -*one; },
            ((1, 2, 3), (true, false, true)) => |[one, two, three]| { *one = -*one; *three = -*three; },
            ((1, 2, 3), (true, true, false)) => |[one, two, three]| { *one = -*one; *two = -*two; },
            ((1, 2, 3), (true, true, true)) => |[one, two, three]| { *one = -*one; *two = -*two; *three = -*three; },
            _ => unreachable!(),
        };

        func_ptr
    }

    /// Builds an [`AxesTransformer`] to convert vectors from one set of directions to another
    ///
    /// # Arguments
    ///
    /// * `from`: Direction context for the vector input
    /// * `to`: Direction context for the vector output
    ///
    /// returns: AxesTransformer<T>
    pub const fn transformer<T: GeometryNumber + Copy>(from: Self, to: Self) -> AxesTransformer<T> {
        AxesTransformer {
            vector_transformer: Self::mapper_fn(from, to),
            matrix_transformer: Self::mapper_fn(from, to),
        }
    }
}

/// Struct containing the faces of a [`Mesh`]
///
/// This struct retains the order in which faces were added
///
/// Meshes contain their vertices as a de-duplicated list, this struct contains the indices into that list for each face.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Faces {
    /// Vertex indices for all faces, concatenated
    /// Length is equal to the sum of all values in vertex_counts
    vertex_indices: Vec<usize>,
    /// The amount of vertices in each face.
    /// This vec's length is equal to the number of faces; Face N's entry is at index N in this array (Starting at 0)
    vertex_counts: Vec<usize>,
    /// The starting offset of the face's first vertex; Such that for any face N, the offset equals the sum of vertex_counts[0..N]
    /// This vec's length is equal to the number of faces; Face N's entry is at index N in this array
    vertex_offsets: Vec<usize>,
    /// True if all faces have exactly three vertices
    /// If true, the length of vertex_indices must be a multiple of three.
    /// If true, all values of vertex_counts must equal 3
    /// If true, all values of vertex_offsets must be N*3 (0, 3, 6, .. , (len-1)*3)
    is_triangulated: bool
}

impl Faces {
    /// Construct a new Faces instance
    ///
    /// A [`Faces`] with 0 faces is triangulated
    pub fn new() -> Faces {
        Faces {
            vertex_indices: Vec::new(),
            vertex_counts: Vec::new(),
            vertex_offsets: Vec::new(),
            is_triangulated: true   // Initialise to true, set to false if a non-triangulated face is appended
        }
    }

    /// Appends a face onto the end of this Faces
    ///
    /// If the added face consists of more than 3 vertices, this faces object will be non-triangulated
    ///
    /// # Arguments
    ///
    /// * `vertex_indices`: Vertex indices of the face to be added
    ///
    /// returns: Result<(), ValidationError> Error if vertices.len() < 3
    pub fn push_face(&mut self, vertex_indices: &[usize]) -> Result<(), ModelError>{
        if vertex_indices.len() < 3 {
            return Err(ModelError::from(format!("Added face must have at least 3 vertices, had {}", vertex_indices.len())))
        } else {
            self.vertex_offsets.push(self.vertex_indices.len());
            self.vertex_indices.extend_from_slice(vertex_indices);
            self.vertex_counts.push(vertex_indices.len());
            self.is_triangulated |= vertex_indices.len() == 3;
            Ok(())
        }
    }

    /// Returns true if all faces are triangles
    ///
    /// This is true if no faces were pushed
    pub fn is_triangulated(&self) -> bool {
        self.is_triangulated
    }
}

/// View object for an individual face
///
/// View objects contain a [`GeometryTransformer`] that translates values into the target coordinate system
///
/// Obtained from [`MeshView::faces`]
///
/// See [`TriangleView`] for a special-case alternative for 3-vertex faces
#[derive(Copy, Clone, Debug)]
pub struct FaceView<'a, T: Copy, C: GeometryTransformer<T>> {
    face_index: usize,
    face_vertex_number: usize,
    vertex_indices: &'a [usize],
    mesh_view: MeshView<'a, T, C>
}

/// Data type for a Face's values, each instance corresponds to one vertex and it's properties
///
/// Obtained from [`FaceView::values`], translated to View's target coordinate directions
pub struct FaceValue<T> {
    /// Index of this vertex in the mesh's lists of vertices
    // TODO: This is currently only used to retrieve bone weights, maybe provide those directly in this type?
    pub vertex_index: usize,
    /// Vertex position
    pub vertex: Vector3D<T>,
    /// Vertex UV coordinates
    pub uv: Vector2D<T>,
    /// Vertex normal vector
    pub normal: Vector3D<T>
}

/// Data type for a face's indices, each instance corresponds to one vertex and it's properties.
///
/// Obtained from [`FaceView::indices`]
///
/// This type is intended for exporting data in formats that use indices into a list
/// [`FaceValue`] / [`FaceView::values`] should be used to retrieve values
pub struct FaceIndex {
    /// Index into mesh's vertex position list
    pub vertex: usize,
    /// Index into mesh's UV coordinate list
    pub uv: usize,
    /// Index into mesh's normal vector list
    pub normal: usize
}

impl<'a, T: Copy, C: GeometryTransformer<T>> FaceView<'a, T, C> {
    /// Returns the material ID for this face; An index into the model's material list
    pub fn material_id(&self) -> usize {
        self.mesh_view.mesh.face_materials[self.face_index]
    }

    /// Returns an iterator over this face's vertex positions in this face, translated to target coordinate directions
    ///
    /// This iterator has equal length and encounter order as [`FaceView::normal_vectors`] and [`FaceView::uv_vectors`]
    ///
    /// If all three values are desired, [`FaceView::values`] may be used instead
    ///
    /// This is an [`ExactSizeIterator`], the length may be retrieved by calling [`ExactSizeIterator::len`]
    #[inline]
    pub fn vertices(&'a self) -> impl Iterator<Item=Vector3D<T>> + ExactSizeIterator + 'a {
        self.vertex_indices.iter()
            .map(|vertex_index| self.mesh_view.resolve_vertex(*vertex_index).unwrap())
    }

    /// Returns an iterator over this face's normal vectors in this face, translated to target coordinate directions
    ///
    /// This iterator has equal length and encounter order as [`FaceView::vertices`] and [`FaceView::uv_vectors`]
    ///
    /// If all three values are desired, [`FaceView::values`] may be used instead
    ///
    /// This is an [`ExactSizeIterator`], the length may be retrieved by calling [`ExactSizeIterator::len`]
    #[inline]
    pub fn normal_vectors(&'a self) -> impl Iterator<Item=Vector3D<T>> + ExactSizeIterator + 'a {
        self.vertex_indices.iter().enumerate()
            .map(|(face_vertex_count, vertex_index)| self.mesh_view.resolve_normal(*vertex_index, self.face_vertex_number + face_vertex_count).unwrap())
    }

    /// Returns an iterator over this face's UV coordinates/vectors in this face, translated to target coordinate directions
    ///
    /// This iterator has equal length and encounter order as [`FaceView::vertices`] and [`FaceView::normal_vectors`]
    ///
    /// If all three values are desired, [`FaceView::values`] may be used instead
    ///
    /// This is an [`ExactSizeIterator`], the length may be retrieved by calling [`ExactSizeIterator::len`]
    #[inline]
    pub fn uv_vectors(&'a self) -> impl Iterator<Item=Vector2D<T>> + ExactSizeIterator + 'a {
        self.vertex_indices.iter().enumerate()
            .map(|(face_vertex_count, vertex_index)| self.mesh_view.resolve_uv(*vertex_index, self.face_vertex_number + face_vertex_count).unwrap())
    }

    /// Returns an iterator over this face's vertices and their normal + uv vectors, translated to target coordinate directions
    ///
    /// This iterator has equal length and encounter order as [`FaceView::vertices`], [`FaceView::uv_vectors`], and [`FaceView::normal_vectors`]
    ///
    /// If only one of the values is needed [`FaceView::vertices`], [`FaceView::uv_vectors`], or [`FaceView::normal_vectors`] may be used instead.
    ///
    /// This is an [`ExactSizeIterator`], the length may be retrieved by calling [`ExactSizeIterator::len`]
    #[inline]
    pub fn values(&'a self) -> impl Iterator<Item=FaceValue<T>> + ExactSizeIterator + 'a {
        self.vertex_indices.iter().enumerate()
            .map(|(face_vertex_count, vertex_index)| {
                FaceValue {
                    vertex_index: *vertex_index,
                    vertex: self.mesh_view.resolve_vertex(*vertex_index).unwrap(),
                    uv: self.mesh_view.resolve_uv(*vertex_index, self.face_vertex_number + face_vertex_count).unwrap(),
                    normal: self.mesh_view.resolve_normal(*vertex_index, self.face_vertex_number + face_vertex_count).unwrap(),
                }
            })
    }

    /// Returns an iterator over the indices of this face's vertices and their normal + uv vectors
    ///
    /// These indices are an index into the vertex, normal, and uv lists of the face's mesh
    ///
    /// This is an [`ExactSizeIterator`], the length may be retrieved by calling [`ExactSizeIterator::len`]
    #[inline]
    pub fn indices(&'a self) -> impl Iterator<Item=FaceIndex> + ExactSizeIterator + 'a {
        self.vertex_indices.iter().enumerate()
            .map(|(face_vertex_count, vertex_index)| {
                FaceIndex {
                    vertex: *vertex_index,
                    uv: match &self.mesh_view.mesh.uv_indices {
                        VertexPropertyIndices::PerVertex(indices) => indices[*vertex_index],
                        VertexPropertyIndices::PerVertexPerFace(indices) => indices[self.face_vertex_number + face_vertex_count]
                    },
                    normal: match &self.mesh_view.mesh.normal_indices {
                        VertexPropertyIndices::PerVertex(indices) => indices[*vertex_index],
                        VertexPropertyIndices::PerVertexPerFace(indices) => indices[self.face_vertex_number + face_vertex_count]
                    }
                }
            })
    }
}

/// View object for an individual face with exactly 3 vertices
///
/// View objects contain a [`GeometryTransformer`] that translates values into the target coordinate system
///
/// Obtained from [`MeshView::triangles`]
///
/// See [`FaceView`] for a general-case alternative for N-vertex faces
#[derive(Copy, Clone, Debug)]
pub struct TriangleView<'a, T: Copy, C: GeometryTransformer<T>> {
    face_index: usize,
    vertex_indices: [usize; 3],
    mesh_view: MeshView<'a, T, C>
}

impl<'a, T: Copy, C: GeometryTransformer<T>> TriangleView<'a, T, C> {
    /// Returns the material ID for this face; An index into the model's material list
    pub fn material_id(&self) -> usize {
        self.mesh_view.mesh.face_materials[self.face_index]
    }

    /// Returns vertex positions in this face, translated to target coordinate directions
    ///
    /// If all three values are desired, [`TriangleView::values`] may be used instead
    pub fn vertices(&'a self) -> [Vector3D<T>; 3] {
        [
            self.mesh_view.resolve_vertex(self.vertex_indices[0]).unwrap(),
            self.mesh_view.resolve_vertex(self.vertex_indices[1]).unwrap(),
            self.mesh_view.resolve_vertex(self.vertex_indices[2]).unwrap(),
        ]
    }

    /// Returns normal vectors in this face, translated to target coordinate directions
    ///
    /// If all three values are desired, [`TriangleView::values`] may be used instead
    pub fn normal_vectors(&'a self) -> [Vector3D<T>; 3] {
        [
            self.mesh_view.resolve_normal(self.vertex_indices[0], self.face_index * 3 + 0).unwrap(),
            self.mesh_view.resolve_normal(self.vertex_indices[1], self.face_index * 3 + 1).unwrap(),
            self.mesh_view.resolve_normal(self.vertex_indices[2], self.face_index * 3 + 2).unwrap(),
        ]
    }

    /// Returns UV coordinates/vectors in this face, translated to target coordinate directions
    ///
    /// If all three values are desired, [`TriangleView::values`] may be used instead
    pub fn uv_vectors(&'a self) -> [Vector2D<T>; 3] {
        [
            self.mesh_view.resolve_uv(self.vertex_indices[0], self.face_index * 3 + 0).unwrap(),
            self.mesh_view.resolve_uv(self.vertex_indices[1], self.face_index * 3 + 1).unwrap(),
            self.mesh_view.resolve_uv(self.vertex_indices[2], self.face_index * 3 + 2).unwrap(),
        ]
    }

    /// Returns vertices and their normal + uv vectors, translated to target coordinate directions
    ///
    /// If only one of the values is needed [`TriangleView::vertices`], [`TriangleView::uv_vectors`], or [`TriangleView::normal_vectors`] may be used instead.
    pub fn values(&'a self) -> [FaceValue<T>; 3] {
        [
            FaceValue {
                vertex_index: self.vertex_indices[0],
                vertex: self.mesh_view.resolve_vertex(self.vertex_indices[0]).unwrap(),
                uv: self.mesh_view.resolve_uv(self.vertex_indices[0], self.face_index * 3 + 0).unwrap(),
                normal: self.mesh_view.resolve_normal(self.vertex_indices[0], self.face_index * 3 + 0).unwrap()
            },
            FaceValue {
                vertex_index: self.vertex_indices[1],
                vertex: self.mesh_view.resolve_vertex(self.vertex_indices[1]).unwrap(),
                uv: self.mesh_view.resolve_uv(self.vertex_indices[1], self.face_index * 3 + 1).unwrap(),
                normal: self.mesh_view.resolve_normal(self.vertex_indices[1], self.face_index * 3 + 1).unwrap()
            },
            FaceValue {
                vertex_index: self.vertex_indices[2],
                vertex: self.mesh_view.resolve_vertex(self.vertex_indices[2]).unwrap(),
                uv: self.mesh_view.resolve_uv(self.vertex_indices[2], self.face_index * 3 + 2).unwrap(),
                normal: self.mesh_view.resolve_normal(self.vertex_indices[2], self.face_index * 3 + 2).unwrap()
            }
        ]
    }


    /// Returns the indices of this face's vertices and their normal + uv vectors
    ///
    /// These indices are an index into the vertex, normal, and uv lists of the face's mesh
    pub fn indices(&'a self) -> [FaceIndex; 3] {
        [
            FaceIndex {
                vertex: self.vertex_indices[0],
                uv: match &self.mesh_view.mesh.uv_indices {
                    VertexPropertyIndices::PerVertex(indices) => indices[self.vertex_indices[0]],
                    VertexPropertyIndices::PerVertexPerFace(indices) => indices[self.face_index * 3 + 0]
                },
                normal: match &self.mesh_view.mesh.normal_indices {
                    VertexPropertyIndices::PerVertex(indices) => indices[self.vertex_indices[0]],
                    VertexPropertyIndices::PerVertexPerFace(indices) => indices[self.face_index * 3 + 0]
                }
            },
            FaceIndex {
                vertex: self.vertex_indices[1],
                uv: match &self.mesh_view.mesh.uv_indices {
                    VertexPropertyIndices::PerVertex(indices) => indices[self.vertex_indices[1]],
                    VertexPropertyIndices::PerVertexPerFace(indices) => indices[self.face_index * 3 + 1]
                },
                normal: match &self.mesh_view.mesh.normal_indices {
                    VertexPropertyIndices::PerVertex(indices) => indices[self.vertex_indices[1]],
                    VertexPropertyIndices::PerVertexPerFace(indices) => indices[self.face_index * 3 + 1]
                }
            },
            FaceIndex {
                vertex: self.vertex_indices[2],
                uv: match &self.mesh_view.mesh.uv_indices {
                    VertexPropertyIndices::PerVertex(indices) => indices[self.vertex_indices[2]],
                    VertexPropertyIndices::PerVertexPerFace(indices) => indices[self.face_index * 3 + 2]
                },
                normal: match &self.mesh_view.mesh.normal_indices {
                    VertexPropertyIndices::PerVertex(indices) => indices[self.vertex_indices[2]],
                    VertexPropertyIndices::PerVertexPerFace(indices) => indices[self.face_index * 3 + 2]
                }
            },
        ]
    }
}

/// Enum for vertex properties
///
/// Vertex properties may be defined either 'per-vertex'; One per vertex, or 'per-vertex-per-face'; Each vertex has separate properties for each face that it is part of
///
/// With 'per-vertex-per-face' normal vectors, a vertex that is part of two (adjacent) faces has two normal vectors.
///
/// Note: Normal vectors may be specified per-vertex-per-face whilst uv coordinates are specified per-vertex, or vice versa.
///
/// These indices index into the mesh's list of normal or uv vectors
#[derive(Clone, PartialEq, Debug)]
pub enum VertexPropertyIndices {
    /// One index into Mesh::normal_vectors / Mesh::uv_vectors for each vertex in Mesh::vertex_positions
    PerVertex(Vec<usize>),
    /// One index into Mesh::normal_vectors / Mesh::uv_vectors for each vertex index in Mesh::faces
    PerVertexPerFace(Vec<usize>)
}

impl VertexPropertyIndices {
    /// Resolve this property to an index into it's list.
    ///
    /// # Arguments
    ///
    /// * `vertex_index`: The vertex index; The vertex's position into the mesh's list of vertices
    /// * `face_vertex_index`: The "face_vertex_index"; The vertex's position in the Mesh's faces data. ([`Faces::vertex_indices`])
    ///
    /// returns: Option<usize>
    pub fn resolve(&self, vertex_index: usize, face_vertex_index: usize) -> Option<usize> {
        match self {
            VertexPropertyIndices::PerVertex(indices) => indices.get(vertex_index).copied(),
            VertexPropertyIndices::PerVertexPerFace(indices) => indices.get(face_vertex_index).copied()
        }
    }
}

/// Animation bone
///
/// Values retrieved through [`BoneView`]
#[derive(Clone, PartialEq, Debug)]
pub struct Bone<T> {
    name: String,
    position: Vector3D<T>,
    rotation: RotMatrix<T>,
    /// Bone parent, index into Mesh's list of bones
    parent: Option<usize>,
}

impl<T> Bone<T> {
    /// Construct a new animation bone
    ///
    /// # Arguments
    ///
    /// * `name`: Bone name
    /// * `position`: Bone position
    /// * `rotation`: Bone rotation
    /// * `parent`: Bone parent, index into Mesh's list of bones
    ///
    /// returns: Bone<T>
    pub fn new(name: String, position: Vector3D<T>, rotation: RotMatrix<T>, parent: Option<usize>) -> Self {
        Self { name, position, rotation, parent }
    }
}


/// View object for an animation bone
///
/// View objects contain a [`GeometryTransformer`] that translates values into the target coordinate system
///
/// Obtained from [`MeshView::bones`]
pub struct BoneView<'a, T: Copy, C: GeometryTransformer<T>> {
    bone: &'a Bone<T>,
    mesh_view: MeshView<'a, T, C>
}

impl<'a, T: Copy, C: GeometryTransformer<T>> BoneView<'a, T, C> {
    /// Name of this animation bone
    pub fn name(&self) -> &String {
        &self.bone.name
    }

    /// Bone parent, index into Mesh's list of bones
    pub fn parent(&self) -> Option<usize> {
        self.bone.parent
    }

    /// Bone position, translated to target coordinate directions
    pub fn position(&self) -> Vector3D<T> {
        self.mesh_view.transformer.transform_vector(self.bone.position)
    }

    /// Bone rotation, translated to target coordinate directions
    pub fn rotation(&self) -> RotMatrix<T> {
        self.mesh_view.transformer.transform_rotmatrix(self.bone.rotation)
    }


}

/// Mesh type, representing 3D model meshes
#[allow(non_snake_case)]
#[derive(Clone, PartialEq, Debug)]
pub struct Mesh<T> {
    /// Name of this mesh, if supported/provided by reader's format
    name: Option<String>,

    /// Coordinate system directions of the data in this mesh object
    axes: CoordinateDirections3D,

    /// List of all vertex positions, in no guaranteed order.
    /// This list represents all the vertices of this mesh, with one entry per vertex in the model.
    ///
    /// Positions may have duplicate entries for vertices that have the same position but are part of different faces that are not connected to each-other.
    ///
    /// It is permitted for this list to contain vertices that are not referenced by any face, but no guarantees about their inclusion in exported formats is given.
    /// Implementers of [`ModelReader`] and [`ModelWriter`] are free to discard such vertices.
    /// If provided, such 'orphan' vertices must still have normal vectors and UVs provided in `normal_indices` and `uv_indices`.
    vertex_positions: Vec<Vector3D<T>>,

    /// Faces of this mesh, consisting of 3 or more indices into vertex_positions per face.
    /// Two variants are provided: One optimized for triangulated meshes where each face has exactly 3 vertices, and a general version for variable amounts of vertices per face.
    /// Each index must be valid for `vertex_positions` i.e. less than [`vertex_positions:len()`].
    face_indices: Faces,

    /// Materials for each face, index into [`Model`]::materials.
    /// One index must be provided for each face in Faces. i.e. Length of this vec has to equal [`Faces::face_count()`]
    face_materials: Vec<usize>,

    /// De-duplicated list of normal vectors.
    normal_vectors: Vec<Vector3D<T>>,

    /// Bindings between vertices and `normal_vectors` list.
    /// Normal vectors may be defined per-vertex; One normal vector per entry in `vertex_positions`, or per-vertex-per-face; One normal vector per vertex index in `face_indices`
    normal_indices: VertexPropertyIndices,

    /// De-duplicated list of UV vectors/coordinates
    /// Currently only supports 2D UV-mapped textures.
    uv_vectors: Vec<Vector2D<T>>,

    /// Bindings between vertices and `uv_vectors` list.
    /// UV vectors may be defined per-vertex; One uv vector per entry in `vertex_positions`, or per-vertex-per-face; One uv vector per vertex index in `face_indices`
    uv_indices: VertexPropertyIndices,

    /// Animation bones for this mesh
    bones: Vec<Bone<T>>,

    /// Bindings between vertices and animation bones
    bone_weights: Vec<Vec<(usize, T)>>
}

impl<T: GeometryNumber + Copy> Mesh<T> {
    /// Constructs a new Mesh object
    ///
    /// All parameters' invariants must be valid at this point, see [`Mesh`] documentation or function source for exact details
    ///
    /// # Arguments
    ///
    /// * `name`: Name of this Mesh, or None if no name is set or available
    /// * `axes`: Coordinate system directions of the data in this mesh object
    /// * `vertex_positions`: List of all vertex positions, in no guaranteed order.
    /// * `face_indices`: Faces of this mesh, consisting of 3 or more indices into vertex_positions per face.
    /// * `face_materials`: Materials for each face, index into [`Model`]::materials. One index must be provided for each face in Faces.
    /// * `normal_vectors`: De-duplicated list of normal vectors.
    /// * `normal_indices`: Bindings between vertices and `normal_vectors` list.
    /// * `uv_vectors`: De-duplicated list of UV vectors/coordinates
    /// * `uv_indices`: Bindings between vertices and `uv_vectors` list.
    /// * `bones`: Animation bones for this mesh
    /// * `bone_weights`: Bindings between vertices and animation bones
    ///
    /// returns: Result<Mesh<T>, ValidationError>
    pub fn new(
        name: Option<String>,
        axes: CoordinateDirections3D,
        vertex_positions: Vec<Vector3D<T>>,
        face_indices: Faces,
        face_materials: Vec<usize>,
        normal_vectors: Vec<Vector3D<T>>,
        normal_indices: VertexPropertyIndices,
        uv_vectors: Vec<Vector2D<T>>,
        uv_indices: VertexPropertyIndices,
        bones: Vec<Bone<T>>,
        bone_weights: Vec<Vec<(usize, T)>>
    ) -> Result<Mesh<T>, ModelError> {
        // No need to validate name
        // No need to validate Axes; CoordinateDirections3D struct maintains it's own invariants
        let vertex_count = vertex_positions.len();
        let face_count = face_indices.vertex_counts.len();
        let face_vertex_count = face_indices.vertex_indices.len();

        if face_materials.len() != face_count {
            return Err(ModelError::from(format!("Invalid amount of face materials, expected 1 per face for {} faces, got {}", face_count, face_materials.len())))
        }

        let normal_index_vec = match &normal_indices {
            VertexPropertyIndices::PerVertex(normal_indices) => {
                if normal_indices.len() != vertex_count {
                    return Err(ModelError::from(format!("Invalid normal index count, expected {} per-vertex normal indices, found {}", vertex_count, normal_indices.len())))
                }
                normal_indices
            }
            VertexPropertyIndices::PerVertexPerFace(normal_indices) => {
                if normal_indices.len() != face_vertex_count {
                    return Err(ModelError::from(format!("Invalid normal index count, expected {} per-vertex-per-face normal indices, found {}", face_vertex_count, normal_indices.len())))
                }
                normal_indices
            }
        };
        for normal_index in normal_index_vec {
            if *normal_index >= normal_vectors.len() {
                return Err(ModelError::from(format!("Normal vector index out of bounds, was {} must be less than {}", normal_index, normal_vectors.len())))
            }
        }

        let uv_index_vec = match &uv_indices {
            VertexPropertyIndices::PerVertex(uv_indices) => {
                if uv_indices.len() != vertex_count {
                    return Err(ModelError::from(format!("Invalid uv index count, expected {} per-vertex uv indices, found {}", vertex_count, uv_indices.len())))
                }
                uv_indices
            }
            VertexPropertyIndices::PerVertexPerFace(uv_indices) => {
                if uv_indices.len() != face_vertex_count {
                    return Err(ModelError::from(format!("Invalid uv index count, expected {} per-vertex-per-face uv indices, found {}", face_vertex_count, uv_indices.len())))
                }
                uv_indices
            }
        };
        for uv_index in uv_index_vec {
            if *uv_index >= uv_vectors.len() {
                return Err(ModelError::from(format!("UV index out of bounds, was {} must be less than {}", uv_index, uv_vectors.len())))
            }
        }

        for (face_index, (vertex_offset, face_vertex_count))  in face_indices.vertex_offsets.iter().zip(face_indices.vertex_counts.iter()).enumerate() {
            for (face_vertex_index, vertex_index) in face_indices.vertex_indices[*vertex_offset..(*vertex_offset + *face_vertex_count)].iter().enumerate() {
                if *vertex_index >= vertex_count {
                    return Err(ModelError::from(format!("Invalid face vertex index, face {} vertex {}, must be less than {}", face_index, face_vertex_index, vertex_count)))
                }
            }
        }

        let bone_count = bones.len();
        for bone in &bones {
            if bone.parent.is_some_and(|id| id >= bone_count) {
                return Err(ModelError::from(format!("Invalid bone parent ID {}, must be < {}", bone.parent.unwrap(), bone_count)));
            }
        }

        if bone_weights.len() != 0 && bone_weights.len() != vertex_count {
            return Err(ModelError::from(format!("Missing bone weights for vertices, {} weights {} vertices", bone_weights.len(), vertex_count)));

        }
        for (vertex_id, weights) in bone_weights.iter().enumerate() {
            for (bone_id, _weight) in weights {
                if *bone_id >= bone_count {
                    return Err(ModelError::from(format!("Vertex {} has invalid bone ID {}, mesh has {} bones", vertex_id, bone_id, bone_count)));
                }
            }
        }


        Ok(Mesh { name, axes, vertex_positions, face_indices, face_materials, normal_vectors, normal_indices, uv_vectors, uv_indices, bones, bone_weights })
    }


    /// Create a view object for this mesh, applying the specified transformation
    ///
    /// Generally you'll want to use [`Mesh::view`], which builds a transformer from this mesh's directions to the specified directions
    ///
    /// # Arguments
    ///
    /// * `transformer`: Transformer to apply to vectors and rotations. `()` provides a no-op translation
    ///
    /// returns: MeshView<T, C>
    pub fn transformer_view<C: GeometryTransformer<T>>(&self, transformer: C) -> MeshView<T, C> {
        MeshView {
            mesh: self,
            transformer,
        }
    }

    /// Returns the coordinate directions this mesh's vectors and rotations use
    pub const fn coordinate_directions(&self) -> CoordinateDirections3D {
        self.axes
    }


    /// Create a view object for this mesh, applying translation into the specified coordinate directions
    ///
    /// If a different (or no) translation is required, [`Mesh::transformer_view`] allows the selection of a specific transformer
    ///
    /// # Arguments
    ///
    /// * `to_directions`:
    ///
    /// returns: MeshView<T, AxesTransformer<T>>
    pub fn view(&self, to_directions: CoordinateDirections3D) -> MeshView<T, AxesTransformer<T>> {
        MeshView {
            mesh: self,
            transformer: CoordinateDirections3D::transformer(self.axes, to_directions),
        }
    }
}


/// View object for a mesh
///
/// View objects contain a [`GeometryTransformer`] that translates values into the target coordinate system
///
/// Obtained from [`Mesh::view`] or [`Mesh::transformer_view`]
#[derive(Copy, Clone, Debug)]
pub struct MeshView<'a, T: Copy, C: GeometryTransformer<T>> {
    mesh: &'a Mesh<T>,
    transformer: C
}

impl<'a, T: Copy, C: GeometryTransformer<T>> MeshView<'a, T, C> {
    /// Name of this mesh, if supported/provided by reader's format
    pub fn name(&self) -> &Option<String> {
        &self.mesh.name
    }

    /// Resolves a vertex from it's index in the vertex list, translated to target coordinate directions
    ///
    /// # Arguments
    ///
    /// * `vertex_index`: Index of vertex to retrieve
    ///
    /// returns: Option<Vector3D<T>>
    pub fn resolve_vertex(&self, vertex_index: usize) -> Option<Vector3D<T>> {
        self.mesh.vertex_positions.get(vertex_index)
            .map(|vertex| self.transformer.transform_vector(*vertex))
    }

    /// Resolves a vertex' normal vector, translated to target coordinate directions
    ///
    /// Normal vectors may be either per-vertex or per-face-vertex (See [`VertexPropertyIndices`])
    ///
    /// # Arguments
    ///
    /// * `vertex_index`: Index of vertex in the Mesh's list of vertices
    /// * `face_vertex_index`: Index of the vertex in list of face-vertex indices
    ///
    /// returns: Option<Vector3D<T>>
    pub fn resolve_normal(&self, vertex_index: usize, face_vertex_index: usize) -> Option<Vector3D<T>> {
        self.mesh.normal_indices.resolve(vertex_index, face_vertex_index)
            .and_then(|normal_index| self.mesh.normal_vectors.get(normal_index).copied())
            .map(|normal_vector| self.transformer.transform_vector(normal_vector))
    }

    /// Resolves a vertex' UV vector/coordinates, translated to target coordinate directions
    ///
    /// UV vectors may be either per-vertex or per-face-vertex (See [`VertexPropertyIndices`])
    ///
    /// # Arguments
    ///
    /// * `vertex_index`: Index of vertex in the Mesh's list of vertices
    /// * `face_vertex_index`: Index of the vertex in list of face-vertex indices
    ///
    /// returns: Option<Vector3D<T>>
    pub fn resolve_uv(&self, vertex_index: usize, face_vertex_index: usize) -> Option<Vector2D<T>> {
        self.mesh.uv_indices.resolve(vertex_index, face_vertex_index)
            .and_then(|uv_index| self.mesh.uv_vectors.get(uv_index).copied())
    }

    /// Retrieves the material index for a given face, or None if face has no material
    ///
    /// This is an index into the list of materials of the Mesh's parent Model
    ///
    /// NOTE: Even when returning Some, the pointed-at material may be of type [`Material::None`]
    ///
    /// # Arguments
    ///
    /// * `face_index`:
    ///
    /// returns: Option<usize>
    pub fn face_material(&self, face_index: usize) -> Option<usize> {
        self.mesh.face_materials.get(face_index).copied()
    }

    /// Returns an iterator over the vertices in this mesh, translated to target coordinate directions
    ///
    /// This is an [`ExactSizeIterator`], the length may be retrieved by calling [`ExactSizeIterator::len`]
    pub fn vertices(&'a self) -> impl Iterator<Item=Vector3D<T>> + ExactSizeIterator + 'a {
        self.mesh.vertex_positions.iter()
            .map(|vertex| self.transformer.transform_vector(*vertex))
    }


    /// Returns an iterator over the normal vectors in this mesh, translated to target coordinate directions
    ///
    /// This is an [`ExactSizeIterator`], the length may be retrieved by calling [`ExactSizeIterator::len`]
    pub fn normal_vectors(&'a self) -> impl Iterator<Item=Vector3D<T>> + ExactSizeIterator + 'a {
        self.mesh.normal_vectors.iter()
            .map(|vector| self.transformer.transform_vector(*vector))
    }


    /// Returns an iterator over the UV vectors/coordinates in this mesh, translated to target coordinate directions
    ///
    /// This is an [`ExactSizeIterator`], the length may be retrieved by calling [`ExactSizeIterator::len`]
    pub fn uv_vectors(&'a self) -> impl Iterator<Item=Vector2D<T>> + ExactSizeIterator + 'a {
        self.mesh.uv_vectors.iter()
            .copied()
    }

    /// Returns true if this mesh is triangulated; If all it's faces have exactly 3 vertices
    pub fn is_triangulated(&self) -> bool {
        self.mesh.face_indices.is_triangulated
    }

    /// Returns an iterator over the faces in this mesh, providing a view object to each face
    ///
    /// This is an [`ExactSizeIterator`], the length may be retrieved by calling [`ExactSizeIterator::len`]
    pub fn faces(&self) -> impl Iterator<Item=FaceView<T, C>> + ExactSizeIterator {
        (0..self.mesh.face_indices.vertex_offsets.len())
            .into_iter()
            .map(|face_index| {
                FaceView {
                    face_index,
                    face_vertex_number: self.mesh.face_indices.vertex_offsets[face_index],
                    vertex_indices: &self.mesh.face_indices.vertex_indices[
                        self.mesh.face_indices.vertex_offsets[face_index]..(self.mesh.face_indices.vertex_offsets[face_index] + self.mesh.face_indices.vertex_counts[face_index])
                    ],
                    mesh_view: *self
                }
            })
    }


    /// If this mesh is triangulated, returns an iterator over the faces in this mesh, providing a view object to each face
    ///
    /// This is an [`ExactSizeIterator`], the length may be retrieved by calling [`ExactSizeIterator::len`]
    pub fn triangles(&'a self) -> Option<impl Iterator<Item=TriangleView<T, C>> + ExactSizeIterator + 'a> {
        if self.mesh.face_indices.is_triangulated {
            debug_assert!(self.mesh.face_indices.vertex_counts.len() % 3 == 0);
            Some((0..self.mesh.face_indices.vertex_offsets.len())
                .into_iter()
                .map(|face_index| {
                    TriangleView {
                        face_index,
                        vertex_indices: [
                            self.mesh.face_indices.vertex_indices[face_index * 3 + 0],
                            self.mesh.face_indices.vertex_indices[face_index * 3 + 1],
                            self.mesh.face_indices.vertex_indices[face_index * 3 + 2]
                        ],
                        mesh_view: *self
                    }
                }))
        } else {
            None
        }
    }


    /// Returns an iterator over the bones in this mesh, providing a view object to each face
    ///
    /// This is an [`ExactSizeIterator`], the length may be retrieved by calling [`ExactSizeIterator::len`]
    pub fn bones(&'a self) -> impl Iterator<Item=BoneView<T, C>> + ExactSizeIterator + 'a {
        self.mesh.bones.iter().map(|bone| {
            BoneView {
                bone,
                mesh_view: *self,
            }
        })
    }

    /// Returns a reference to this mesh's bone weight assignment
    ///
    /// This is a vector with length equal to the list of vertices ([`MeshView::vertices`]),
    /// Storing for each vertices a Vec of bone IDs and the amount of weight that bone has for the vertex
    // TODO: This is a somewhat clunky way of accessing bone weights, accessing through FaceView may be more ergonomic
    pub fn bone_weights(&'a self) -> &Vec<Vec<(usize, T)>> {
        &self.mesh.bone_weights
    }
}

/// Enum of material types
#[derive(Clone, PartialEq, Debug)]
pub enum Material<T> {
    /// No material
    None,
    /// Reference to an external material we do not have access to
    External(String),
    /// Phong Shader material
    Phong {
        name: String,
        /// Ambient reflectivity colour/amount in [R, G, B] ranges from 0 to 1.0
        ambient: Option<[T; 3]>,
        /// Diffuse reflectivity colour/amount in [R, G, B] ranges from 0 to 1.0
        diffuse: Option<[T; 3]>,
        /// Specular reflectivity colour/amount in [R, G, B] ranges from 0 to 1.0
        specular: Option<[T; 3]>,
        /// Specular exponent scalar value in range [0..]
        specular_exponent: Option<T>,
        ambient_texture: Option<String>,
        diffuse_texture: Option<String>,
        specular_texture: Option<String>
    }
}

impl<T> Material<T> {
    /// Retrieves the name of this material, if present
    ///
    /// None-type materials have no name, all other materials must have a name set
    pub fn name(&self) -> Option<&str> {
        match self {
            Material::None => None,
            Material::External(name) => Some(name.as_str()),
            Material::Phong { name, .. } => Some(name.as_str())
        }
    }
}

/// Model type, representing 3D models, consisting of one or more meshes
#[derive(Clone, PartialEq, Debug)]
pub struct Model<T> {
    name: String,
    meshes: Vec<Mesh<T>>,
    materials: Vec<Material<T>>
}

impl<T> Model<T> {

    /// Constructs a new Model object
    ///
    /// All meshes' material indices must be valid for the passed list of materials, if any mesh has an out-of-bounds index, an error is returned
    ///
    /// Models must have at least one mesh
    ///
    /// # Arguments
    ///
    /// * `name`: Name of this model
    /// * `meshes`: Meshes of this model, if length 0, an error is returned
    /// * `materials`: Materials for this model
    ///
    /// returns: Result<Model<T>, ValidationError>
    pub fn new(name: String, meshes: Vec<Mesh<T>>, materials: Vec<Material<T>>) -> Result<Self, ModelError> {
        if meshes.len() == 0 {
            return Err(ModelError::from("models must have at least one mesh"))
        } else {
            for (mesh_index, mesh) in meshes.iter().enumerate() {
                for material_id in &mesh.face_materials {
                    if *material_id >= materials.len() {
                        return Err(ModelError::from(format!("Mesh {} expected material id ({}), only had up to < {}", mesh_index, material_id, materials.len())))
                    }
                }
            }
            Ok(Model { name, meshes, materials })
        }
    }


    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn meshes(&self) -> &Vec<Mesh<T>> { // TODO: Maybe make return iterator?
        &self.meshes
    }
    pub fn materials(&self) -> &Vec<Material<T>> {
        &self.materials
    }
}

/// Error type for attempting to construct invalid [`Model`]s, [`Mesh`]es, or [`Faces`]
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ModelError {
    pub message: String
}

impl From<&str> for ModelError {
    fn from(value: &str) -> Self {
        ModelError {
            message: value.to_string()
        }
    }
}

impl From<String> for ModelError {
    fn from(value: String) -> Self {
        ModelError {
            message: value
        }
    }
}

/// Trait for geometric "transformers"; Objects describing a transformation of vectors and rotation matrices
///
/// Various 3D graphics data formats use different directions for the axes in the carthesian coordinate system (e.g. Y-up vs Z-up)
/// To convert between these formats, transformers are used.
///
/// Implementations of this trait should be linear transformations for best results
pub trait GeometryTransformer<T>: Copy {
    /// Apply transformation to a vector
    fn transform_vector(&self, vector: Vector3D<T>) -> Vector3D<T>;
    /// Apply transformation to a rotation matrix
    fn transform_rotmatrix(&self, rotation: RotMatrix<T>) -> RotMatrix<T>;
}

/// No-op transformer, performs no transformation
#[derive(Copy, Clone, Debug)]
pub struct NoOpTransformer;

impl<T> GeometryTransformer<T> for NoOpTransformer {
    fn transform_vector(&self, vector: Vector3D<T>) -> Vector3D<T> {
        vector
    }

    fn transform_rotmatrix(&self, rotation: RotMatrix<T>) -> RotMatrix<T> {
        rotation
    }
}

/// Transformer that translates between coordinate systems with different directions for their cardinal axes.
///
/// Performs 90-degree rotations, reflections and transpositions of the cardinal axes
///
/// Delegates to [`CoordinateDirections3D::mapper_fn`]
#[derive(Debug)]
pub struct AxesTransformer<T: GeometryNumber> {
    vector_transformer: fn(&mut [T; 3]),
    matrix_transformer: fn(&mut [Vector3D<T>; 3]),
}

// Derive only works if T is also clone/copy, but as we only store function pointers we are copy even if T is not
impl<T: GeometryNumber> Clone for AxesTransformer<T> {
    fn clone(&self) -> Self { *self }
}
impl<T: GeometryNumber> Copy for AxesTransformer<T> {}

impl<T: GeometryNumber> GeometryTransformer<T> for AxesTransformer<T> {
    fn transform_vector(&self, mut vector: Vector3D<T>) -> Vector3D<T> {
        (self.vector_transformer)(&mut vector.array);
        vector
    }

    fn transform_rotmatrix(&self, mut rotation: RotMatrix<T>) -> RotMatrix<T> {
        (self.matrix_transformer)(&mut rotation.0);
        rotation
    }
}

pub trait ModelReader<Input, Error> {
    type Float;

    fn read_model(input: Input, name: Option<String>) -> Result<Model<Self::Float>, Error>;
}
pub trait ModelWriter<Output, Error, Float> {
    fn write_model(output: Output, model: &Model<Float>) -> Result<(), Error>;
}