//! Module for System V ABI unwind registry.

use crate::prelude::*;
use crate::runtime::vm::SendSyncPtr;
use core::ptr::NonNull;

/// Represents a registration of function unwind information for System V ABI.
pub struct UnwindRegistration {
    registrations: TryVec<SendSyncPtr<u8>>,
}

cfg_select! {
    // FIXME: at least on the `gcc-arm-linux-gnueabihf` toolchain on Ubuntu
    // these symbols are not provided by default like they are on other targets.
    // I'm not ARM expert so I don't know why. For now though consider this an
    // optional integration feature with the platform and stub out the functions
    // to do nothing which won't break any tests it just means that
    // runtime-generated backtraces won't have the same level of fidelity they
    // do on other targets.
    target_arch = "arm" => {
        unsafe extern "C" fn __register_frame(_: *const u8) {}
        unsafe extern "C" fn __deregister_frame(_: *const u8) {}
        unsafe extern "C" fn wasmtime_using_libunwind() -> bool {
            false
        }
    }
    _ => {
        unsafe extern "C" {
            // libunwind import
            fn __register_frame(fde: *const u8);
            fn __deregister_frame(fde: *const u8);
            #[wasmtime_versioned_export_macros::versioned_link]
            fn wasmtime_using_libunwind() -> bool;
        }
    }
}

/// There are two primary unwinders on Unix platforms: libunwind and libgcc.
///
/// Unfortunately their interface to `__register_frame` is different. The
/// libunwind library takes a pointer to an individual FDE while libgcc takes a
/// null-terminated list of FDEs. This means we need to know what unwinder
/// is being used at runtime.
///
/// This detection is done currently by looking for a libunwind-specific symbol.
/// This specific symbol was somewhat recommended by LLVM's
/// "RTDyldMemoryManager.cpp" file which says:
///
/// > We use the presence of __unw_add_dynamic_fde to detect libunwind.
///
/// I'll note that there's also a different libunwind project at
/// https://www.nongnu.org/libunwind/ but that doesn't appear to have
/// `__register_frame` so I don't think that interacts with this.
fn using_libunwind() -> bool {
    // On macOS the libgcc interface is never used so libunwind is always used.
    // Otherwise delegate to `helpers.c` since weak symbols can't be used from
    // Rust at this time.
    cfg!(target_os = "macos") || unsafe { wasmtime_using_libunwind() }
}

/// Returns the offset of every FDE in a Wasmtime-encoded `.eh_frame` section.
///
/// The section holds one CIE per batch of FDEs and ends with a 32-bit zero
/// terminator; CIEs, whose CIE id is zero, are skipped.
fn fde_offsets(eh_frame: &[u8]) -> impl Iterator<Item = usize> + '_ {
    let read_u32 = |offset: usize| {
        u32::from_ne_bytes(eh_frame[offset..offset + 4].try_into().unwrap())
    };
    let end = eh_frame.len() - 4;
    let mut offset = 0;
    core::iter::from_fn(move || {
        while offset < end {
            let entry = offset;
            // +4 because the length itself is not inclusive.
            offset = entry + read_u32(entry) as usize + 4;
            if read_u32(entry + 4) != 0 {
                return Some(entry);
            }
        }
        None
    })
}

impl UnwindRegistration {
    pub const SECTION_NAME: &'static str = ".eh_frame";

    /// Registers precompiled unwinding information with the system.
    ///
    /// The `_base_address` field is ignored here (only used on other
    /// platforms), but the `unwind_info` and `unwind_len` parameters should
    /// describe an in-memory representation of a `.eh_frame` section. This is
    /// typically arranged for by the `wasmtime-obj` crate.
    pub unsafe fn new(
        _base_address: *const u8,
        unwind_info: *const u8,
        unwind_len: usize,
    ) -> Result<UnwindRegistration> {
        #[cfg(has_virtual_memory)]
        debug_assert_eq!(
            unwind_info as usize % crate::runtime::vm::host_page_size(),
            0,
            "The unwind info must always be aligned to a page"
        );

        let mut registrations = TryVec::new();
        unsafe {
            if using_libunwind() {
                // For libunwind, `__register_frame` takes a pointer to a single
                // FDE.
                let eh_frame = core::slice::from_raw_parts(unwind_info, unwind_len);
                for offset in fde_offsets(eh_frame) {
                    let fde = unwind_info.add(offset);
                    __register_frame(fde);
                    let fde = NonNull::new(fde.cast_mut()).unwrap();
                    registrations.push(SendSyncPtr::new(fde))?;
                }
            } else {
                // On gnu (libgcc), `__register_frame` will walk the FDEs until an
                // entry of length 0
                __register_frame(unwind_info);
                let info = NonNull::new(unwind_info.cast_mut()).unwrap();
                registrations.push(SendSyncPtr::new(info))?;
            }
        }

        Ok(UnwindRegistration { registrations })
    }
}

impl Drop for UnwindRegistration {
    fn drop(&mut self) {
        unsafe {
            // libgcc stores the frame entries as a linked list in decreasing
            // sort order based on the PC value of the registered entry.
            //
            // As we store the registrations in increasing order, it would be
            // O(N^2) to deregister in that order.
            //
            // To ensure that we just pop off the first element in the list upon
            // every deregistration, walk our list of registrations backwards.
            for fde in self.registrations.iter().rev() {
                __deregister_frame(fde.as_ptr());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fde_offsets;
    use crate::prelude::*;

    fn entry(section: &mut Vec<u8>, id: u32, body: usize) -> usize {
        let offset = section.len();
        section.extend_from_slice(&u32::try_from(4 + body).unwrap().to_ne_bytes());
        section.extend_from_slice(&id.to_ne_bytes());
        section.resize(section.len() + body, 0xcc);
        offset
    }

    #[test]
    fn libunwind_registration_skips_every_repeated_cie() {
        let mut section = Vec::new();
        let mut fdes = Vec::new();
        for _batch in 0..3 {
            let cie = entry(&mut section, 0, 12);
            for body in [8, 16] {
                let pointer = u32::try_from(section.len() + 4 - cie).unwrap();
                fdes.push(entry(&mut section, pointer, body));
            }
        }
        section.extend_from_slice(&0u32.to_ne_bytes());

        assert_eq!(fde_offsets(&section).collect::<Vec<_>>(), fdes);
    }
}
