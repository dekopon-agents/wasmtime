//! Native unwinding through Wasm code whose `.eh_frame` is written as several
//! CIE-headed batches of FDEs.

#![cfg(all(
    not(miri),
    any(target_os = "linux", target_os = "macos"),
    any(target_arch = "x86_64", target_arch = "aarch64")
))]

use std::ffi::{c_int, c_void};
use std::fmt::Write as _;
use std::sync::{Arc, Mutex};
use wasmtime::*;

const FILLERS: usize = 2050;

fn chain_module(fillers: usize, leaf: &str) -> String {
    let mut wat = String::from("(module (import \"\" \"capture\" (func $capture))\n");
    for _ in 0..fillers {
        wat.push_str("(func)\n");
    }
    write!(
        wat,
        "(func $outer (export \"outer\") call $middle)
         (func $middle call $inner)
         (func $inner {leaf}))"
    )
    .unwrap();
    wat
}

fn engine(padding: usize, force_veneers: bool) -> Result<Engine> {
    let mut config = Config::new();
    config.native_unwind_info(true);
    unsafe {
        config.cranelift_flag_set(
            "wasmtime_linkopt_padding_between_functions",
            &padding.to_string(),
        );
        config.cranelift_flag_set(
            "wasmtime_linkopt_force_jump_veneer",
            if force_veneers { "true" } else { "false" },
        );
    }
    Engine::new(&config)
}

#[repr(C)]
struct UnwindContext {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn _Unwind_Backtrace(
        trace: extern "C" fn(*mut UnwindContext, *mut c_void) -> c_int,
        arg: *mut c_void,
    ) -> c_int;
    fn _Unwind_GetIP(context: *mut UnwindContext) -> usize;
}

extern "C" fn record_ip(context: *mut UnwindContext, ips: *mut c_void) -> c_int {
    let ips = unsafe { &mut *ips.cast::<Vec<usize>>() };
    ips.push(unsafe { _Unwind_GetIP(context) });
    0
}

fn native_ips() -> Vec<usize> {
    let mut ips = Vec::new();
    unsafe {
        _Unwind_Backtrace(record_ip, (&raw mut ips).cast());
    }
    ips
}

#[test]
fn native_unwinder_walks_wasm_frames_across_eh_frame_batches() -> Result<()> {
    for (padding, force_veneers, fillers) in
        [(0, false, FILLERS), (0, true, FILLERS), (1 << 20, false, 0)]
    {
        let engine = engine(padding, force_veneers)?;
        let module = Module::new(&engine, chain_module(fillers, "call $capture"))?;
        let image = module.image_range();
        let image = image.start as usize..image.end as usize;

        let walked = Arc::new(Mutex::new(Vec::new()));
        let mut store = Store::new(&engine, ());
        let capture = {
            let walked = walked.clone();
            Func::wrap(&mut store, move || *walked.lock().unwrap() = native_ips())
        };
        let instance = Instance::new(&mut store, &module, &[capture.into()])?;
        instance
            .get_typed_func::<(), ()>(&mut store, "outer")?
            .call(&mut store, ())?;

        let ips = walked.lock().unwrap().clone();
        let in_wasm = |ip: &usize| image.contains(ip);
        let first = ips.iter().position(in_wasm).expect("no Wasm frame unwound");
        let last = ips.iter().rposition(in_wasm).unwrap();
        assert!(
            ips[first..=last].iter().filter(|ip| in_wasm(ip)).count() >= 3,
            "padding {padding}, veneers {force_veneers}: {ips:x?} in {image:x?}"
        );
        assert!(
            last + 1 < ips.len(),
            "unwinding stopped inside Wasm: padding {padding}, veneers {force_veneers}"
        );
    }
    Ok(())
}

#[test]
fn traps_after_eh_frame_batches_keep_their_code_and_frames() -> Result<()> {
    let engine = engine(0, false)?;
    let module = Module::new(&engine, chain_module(FILLERS, "unreachable"))?;
    let mut store = Store::new(&engine, ());
    let capture = Func::wrap(&mut store, || {});
    let instance = Instance::new(&mut store, &module, &[capture.into()])?;
    let err = instance
        .get_typed_func::<(), ()>(&mut store, "outer")?
        .call(&mut store, ())
        .unwrap_err();

    assert_eq!(
        err.downcast_ref::<Trap>(),
        Some(&Trap::UnreachableCodeReached)
    );
    let frames = err.downcast_ref::<WasmBacktrace>().unwrap().frames();
    let first = u32::try_from(FILLERS + 1).unwrap();
    assert_eq!(
        frames.iter().map(|f| f.func_index()).collect::<Vec<_>>(),
        [first + 2, first + 1, first]
    );
    assert!(frames.iter().all(|f| f.func_offset().is_some()));
    Ok(())
}

#[test]
fn eh_frame_repeats_one_equivalent_cie_per_batch_of_fdes() -> Result<()> {
    let engine = engine(0, false)?;
    let cwasm = engine.precompile_module(chain_module(FILLERS, "call $capture").as_bytes())?;
    let elf = object::File::parse(&cwasm[..]).unwrap();
    let eh_frame =
        object::Object::section_by_name(&elf, ".eh_frame").expect("module has native unwind info");
    let eh_frame = object::ObjectSection::data(&eh_frame).unwrap();

    let read_u32 = |at: usize| u32::from_ne_bytes(eh_frame[at..at + 4].try_into().unwrap());
    let mut cies: Vec<&[u8]> = Vec::new();
    let mut batches: Vec<usize> = Vec::new();
    let mut cie_at = None;
    let mut at = 0;
    while read_u32(at) != 0 {
        let len = usize::try_from(read_u32(at)).unwrap();
        let id = usize::try_from(read_u32(at + 4)).unwrap();
        if id == 0 {
            cies.push(&eh_frame[at + 8..at + 4 + len]);
            batches.push(0);
            cie_at = Some(at);
        } else {
            assert_eq!(
                Some(at + 4 - id),
                cie_at,
                "FDE at {at:#x} names a stale CIE"
            );
            *batches.last_mut().unwrap() += 1;
        }
        at += 4 + len;
    }
    assert_eq!(at + 4, eh_frame.len(), "one terminator ends the section");

    let fdes = batches.iter().sum::<usize>();
    assert!(fdes > FILLERS, "{fdes} FDEs");
    assert_eq!(cies.len(), fdes.div_ceil(1024));
    assert!(cies.iter().all(|cie| *cie == cies[0]));
    assert!(batches[..batches.len() - 1].iter().all(|&n| n == 1024));
    Ok(())
}
