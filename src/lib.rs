// Copyright 2017-2018 Niklas Fiekas <niklas.fiekas@backscattering.de>
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A thread-safe lazily initialized cell using double-checked locking.
//!
//! Provides a memory location that can be safely shared between threads and
//! initialized at most once. Once the cell is initialized it becomes
//! immutable.
//!
//! You can only initialize a `DoubleCheckedCell<T>` once, but then it is
//! more efficient than a `Mutex<Option<T>>`.
//!
//! # Examples
//!
//! ```
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use double_checked_cell::DoubleCheckedCell;
//! use futures::future::ready;
//!
//! let cell = DoubleCheckedCell::new();
//!
//! // The cell starts uninitialized.
//! assert_eq!(cell.get().await, None);
//!
//! // Perform potentially expensive initialization.
//! let value = cell.get_or_init(|| ready(21 + 21)).await;
//! assert_eq!(*value, 42);
//! assert_eq!(cell.get().await, Some(&42));
//!
//! // The cell is already initialized.
//! let value = cell.get_or_init(|| ready(unreachable!())).await;
//! assert_eq!(*value, 42);
//! assert_eq!(cell.get().await, Some(&42));
//! # Ok(())
//! # }
//! ```
//!
//! # Errors
//!
//! `DoubleCheckedCell` supports fallible initialization.
//!
//! ```
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use tokio::fs::File;
//! use tokio::prelude::*;
//! use double_checked_cell::DoubleCheckedCell;
//!
//! let cell = DoubleCheckedCell::new();
//!
//! let contents: Result<_, tokio::io::Error> = cell.get_or_try_init(|| async {
//!     let mut file = File::open("not-found.txt").await?;
//!     let mut contents = String::new();
//!     file.read_to_string(&mut contents).await?;
//!     Ok(contents)
//! }).await;
//!
//! // File not found.
//! assert!(contents.is_err());
//!
//! // Cell remains uninitialized for now.
//! assert_eq!(cell.get().await, None);
//! # Ok(())
//! # }
//! ```
//!
//! # Unwind safety
//!
//! If an initialization closure panics, the `DoubleCheckedCell` remains
//! uninitialized, however the `catch_unwind` future combinator currently can't be
//! applied to the futures returned from `get_or_init` and `get_or_try_init`.

#![doc(html_root_url = "https://docs.rs/async-double-checked-cell/0.1.0")]
#![warn(missing_debug_implementations)]

use std::cell::UnsafeCell;
use std::future::Future;
use std::panic::RefUnwindSafe;
use std::sync::atomic::{AtomicBool, Ordering};

use futures_util::future::ready;
use futures_util::FutureExt;
use futures_util::lock::Mutex;
use unreachable::UncheckedOptionExt;
use void::ResultVoidExt;

/// A thread-safe lazily initialized cell.
///
/// The cell is immutable once it is initialized.
/// See the [module-level documentation](index.html) for more.
#[derive(Debug)]
pub struct DoubleCheckedCell<T> {
    value: UnsafeCell<Option<T>>,
    initialized: AtomicBool,
    lock: Mutex<()>,
}

impl<T> Default for DoubleCheckedCell<T> {
    fn default() -> DoubleCheckedCell<T> {
        DoubleCheckedCell::new()
    }
}

impl<T> DoubleCheckedCell<T> {
    /// Creates a new uninitialized `DoubleCheckedCell`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use double_checked_cell::DoubleCheckedCell;
    ///
    /// let cell = DoubleCheckedCell::<u32>::new();
    /// assert_eq!(cell.get().await, None);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new() -> DoubleCheckedCell<T> {
        DoubleCheckedCell {
            value: UnsafeCell::new(None),
            initialized: AtomicBool::new(false),
            lock: Mutex::new(()),
        }
    }

    /// Borrows the value if the cell is initialized.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use double_checked_cell::DoubleCheckedCell;
    ///
    /// let cell = DoubleCheckedCell::from("hello");
    /// assert_eq!(cell.get().await, Some(&"hello"));
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get(&self) -> Option<&T> {
        self.get_or_try_init(|| ready(Err(()))).await.ok()
    }

    /// Borrows the value if the cell is initialized or initializes it from
    /// a closure.
    ///
    /// # Panics
    ///
    /// Panics or deadlocks when trying to access the cell from the
    /// initilization closure.
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use double_checked_cell::DoubleCheckedCell;
    /// use futures::future::ready;
    ///
    /// let cell = DoubleCheckedCell::new();
    ///
    /// // Initialize the cell.
    /// let value = cell.get_or_init(|| ready(1 + 2)).await;
    /// assert_eq!(*value, 3);
    ///
    /// // The cell is now immutable.
    /// let value = cell.get_or_init(|| ready(42)).await;
    /// assert_eq!(*value, 3);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_or_init<F, Fut>(&self, init: F) -> &T
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = T>
    {
        self.get_or_try_init(|| init().map(Ok)).await.void_unwrap()
    }

    /// Borrows the value if the cell is initialized or attempts to initialize
    /// it from a closure.
    ///
    /// # Errors
    ///
    /// Forwards any error from the closure if the cell is not yet initialized.
    /// The cell then remains uninitialized.
    ///
    /// # Panics
    ///
    /// Panics or deadlocks when trying to access the cell from the
    /// initilization closure.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use double_checked_cell::DoubleCheckedCell;
    /// use futures::future::ready;
    ///
    /// let cell = DoubleCheckedCell::new();
    ///
    /// let result = cell.get_or_try_init(|| ready("not an integer".parse())).await;
    /// assert!(result.is_err());
    ///
    /// let result = cell.get_or_try_init(|| ready("42".parse())).await;
    /// assert_eq!(result, Ok(&42));
    ///
    /// let result = cell.get_or_try_init(|| ready("irrelevant".parse())).await;
    /// assert_eq!(result, Ok(&42));
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_or_try_init<F, E, Fut>(&self, init: F) -> Result<&T, E>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T, E>>
    {
        // Safety comes down to the double checked locking here. All other
        // borrowing methods are implemented by calling this.

        if !self.initialized.load(Ordering::Acquire) {
            // Lock the internal mutex.
            let _lock = self.lock.lock().await;

            if !self.initialized.load(Ordering::Relaxed) {
                // We claim that it is safe to make a mutable reference to
                // `self.value` because no other references exist. The only
                // places that could have taken another reference are
                // (A) and (B).
                //
                // We will be the only one holding a mutable reference, because
                // we are holding a mutex. The mutex guard lives longer
                // than the reference taken at (A).
                //
                // No thread could have reached (B) yet, because that implies
                // the cell is already initialized. When we last checked the
                // cell was not yet initialized, and no one else could have
                // initialized it, because that requires holding the mutex.
                {
                    let result = init().await?;

                    // Consider all possible control flows:
                    // - init returns Ok(T)
                    // - init returns Err(E)
                    // - init recursively tries to initialize the cell
                    // - init panics
                    let value = unsafe { &mut *self.value.get() }; // (A)
                    value.replace(result);
                }

                self.initialized.store(true, Ordering::Release);
            }
        }

        // The cell is now guaranteed to be initialized.

        // We claim that it is safe to take a shared reference of `self.value`.
        // The only place that could have created a conflicting mutable
        // reference is (A). But no one can be in that scope while the cell
        // is already initialized.
        let value = unsafe { &*self.value.get() }; // (B)

        // Value is guaranteed to be initialized.
        Ok(unsafe { value.as_ref().unchecked_unwrap() })
    }

    /// Unwraps the value.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use double_checked_cell::DoubleCheckedCell;
    ///
    /// let cell = DoubleCheckedCell::from(42);
    /// let contents = cell.into_inner();
    /// assert_eq!(contents, Some(42));
    /// # Ok(())
    /// # }
    /// ```
    pub fn into_inner(self) -> Option<T> {
        // into_inner() is actually unconditionally safe:
        // https://github.com/rust-lang/rust/issues/35067
        #[allow(unused_unsafe)]
        unsafe { self.value.into_inner() }
    }
}

impl<T> From<T> for DoubleCheckedCell<T> {
    fn from(t: T) -> DoubleCheckedCell<T> {
        DoubleCheckedCell {
            value: UnsafeCell::new(Some(t)),
            initialized: AtomicBool::new(true),
            lock: Mutex::new(()),
        }
    }
}

// Can DoubleCheckedCell<T> be Sync?
//
// The internal state of the DoubleCheckedCell is only mutated while holding
// a mutex, so we only need to consider T.
//
// We need T: Send, because we can share a DoubleCheckedCell with another
// thread, initialize it there and unpack it on the original thread.
// We trivially need T: Sync, because a reference to the contents can be
// retrieved on multiple threads.
unsafe impl<T: Send + Sync> Sync for DoubleCheckedCell<T> {}

// A panic during initialization will leave the cell in a valid, uninitialized
// state.
impl<T> RefUnwindSafe for DoubleCheckedCell<T> {}

#[cfg(test)]
mod tests {
    use std::rc::Rc;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    use futures_util::future::join_all;

    use super::*;

    #[tokio::test]
    async fn test_drop() {
        let rc = Rc::new(true);
        assert_eq!(Rc::strong_count(&rc), 1);

        {
            let cell = DoubleCheckedCell::new();
            cell.get_or_init(|| ready(rc.clone())).await;

            assert_eq!(Rc::strong_count(&rc), 2);
        }

        assert_eq!(Rc::strong_count(&rc), 1);
    }

    #[tokio::test(threaded_scheduler)]
    async fn test_threading() {
        let n = Arc::new(AtomicUsize::new(0));
        let cell = Arc::new(DoubleCheckedCell::new());

        let join_handles = (0..1000).map(|_| {
            let n = n.clone();
            let cell = cell.clone();
            tokio::task::spawn(async move {
                let value = cell.get_or_init(|| {
                    n.fetch_add(1, Ordering::Relaxed);
                    ready(true)
                }).await;

                assert!(*value);
            })
        }).collect::<Vec<_>>();
        join_all(join_handles).await;

        assert_eq!(n.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_sync_send() {
        fn assert_sync<T: Sync>(_: T) {}
        fn assert_send<T: Send>(_: T) {}

        assert_sync(DoubleCheckedCell::<usize>::new());
        assert_send(DoubleCheckedCell::<usize>::new());
        let cell = DoubleCheckedCell::<usize>::new();
        assert_send(cell.get_or_init(|| ready(1)));
    }

    struct _AssertObjectSafe(Box<DoubleCheckedCell<usize>>);
}
