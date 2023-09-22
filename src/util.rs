use std::hash::{Hash, Hasher};

/// Extension trait for slices
pub(crate) trait SliceExtension<T> {
    /// "Selects" an array out of a slice at a given index
    ///
    /// Effectively:
    /// ```
    /// let array: [T; N] = &slice[index..index+N]
    /// ```
    ///
    /// # Arguments
    ///
    /// * `index`: Start position of array, inclusive
    ///
    /// returns: Option<&[T; N]>
    fn select_array<const N: usize>(&self, index: usize) -> Option<&[T; N]>;
}

impl<T> SliceExtension<T> for [T] {
    fn select_array<const N: usize>(&self, index: usize) -> Option<&[T; N]> {
        if let Some(slice) = self.get(index..(index+N)) {
            <&[T; N]>::try_from(slice).ok()
        } else {
            None
        }
    }
}

/// Hash "faking" type; Enables non-[`Hash`] types to pretend they are [`Hash`] and [`Eq`]
///
/// This crate requires de-duplication of floating point values, parsed from text. In lieu of (complex) equality comparisons, we use conventional hashed map types to perform this
///
/// We make the assumption that for a given text-string representation of a number will always be parsed to the same bit-identical floating point value.
/// Values are de-duplicated before any calculations are performed, so this should be a relatively safe assumption.
///
/// If this assumptions fails, de-duplication will be performed incorrectly, but no unsafe or other failure will occur.
///
/// This breaks normal floating point equality rules, particularly those of NaN values.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub(crate) struct FakeHash<T: PretendsToBeHashable>(T::Target); // This is a fairly bad hack, but sets with approximate equality are rather hard to get right and using exact comparisons is acceptable here

impl<T: PretendsToBeHashable> FakeHash<T> {
    /// Wraps the not-[`Hash`] value into a FakeHash that does implement Hash
    pub fn new(not_hash: T) -> Self {
        FakeHash(not_hash.into_hashable())
    }

    /// Unwraps the FakeHash value back to it's original value
    pub fn unwrap(self) -> T {
        T::from_hashable(self.0)
    }
}

// Derive doesn't see that we only hold T::Target (which is Hash and Eq), and not T. So we manually implement and delegate those traits
impl<T: PretendsToBeHashable> PartialEq for FakeHash<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}
impl<T: PretendsToBeHashable> Eq for FakeHash<T> {}

impl<T: PretendsToBeHashable> Hash for FakeHash<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

/// Trait for values that can be "faked" to be [`Hash`]
///
/// Conversion must be round-trip identical
///
/// See [`FakeHash`]
pub(crate) trait PretendsToBeHashable {
    type Target: Hash + Eq;

    /// Convert this value into it's hashable target type
    fn into_hashable(self) -> Self::Target;

    /// Convert the hashable target value back to the main type
    fn from_hashable(fake_hash: Self::Target) -> Self;
}

impl PretendsToBeHashable for f32 {
    type Target = u32;

    fn into_hashable(self) -> Self::Target {
        self.to_bits()
    }

    fn from_hashable(fake_hash: Self::Target) -> Self {
        f32::from_bits(fake_hash)
    }
}

impl PretendsToBeHashable for f64 {
    type Target = u64;

    fn into_hashable(self) -> Self::Target {
        self.to_bits()
    }

    fn from_hashable(fake_hash: Self::Target) -> Self {
        f64::from_bits(fake_hash)
    }
}