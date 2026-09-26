//! Compute device selection (port of `device.py`, macOS only: Metal or CPU).

use candle_core::Device;

use crate::error::{Result, VonError};

/// Resolves `requested` (or `VON_DEVICE`, default `auto`) to a device.
///
/// `auto` means Metal when a GPU is visible to this process, otherwise CPU.
/// `mps` is accepted as an alias for `metal` so Python-era settings keep working.
pub fn resolve_device(requested: Option<&str>) -> Result<Device> {
    let spec = requested
        .map(str::to_string)
        .or_else(|| std::env::var("VON_DEVICE").ok())
        .unwrap_or_else(|| "auto".into());
    match spec.trim().to_lowercase().as_str() {
        "" | "auto" => Ok(if metal_available() {
            new_metal(&spec)?
        } else {
            Device::Cpu
        }),
        "metal" | "mps" => new_metal(&spec),
        "cpu" => Ok(Device::Cpu),
        _ => Err(VonError::UnsupportedDevice(spec)),
    }
}

pub fn describe(device: &Device) -> &'static str {
    match device {
        Device::Metal(_) => "Apple Silicon [Metal]",
        _ => "CPU",
    }
}

/// candle's `Device::new_metal` panics when no Metal device exists (for example,
/// in a process without GPU access), so probe first.
#[cfg(feature = "metal")]
fn metal_available() -> bool {
    !candle_metal_kernels::metal::Device::all().is_empty()
}

#[cfg(not(feature = "metal"))]
fn metal_available() -> bool {
    false
}

fn new_metal(requested: &str) -> Result<Device> {
    if !cfg!(feature = "metal") {
        return Err(VonError::DeviceUnavailable {
            requested: requested.into(),
            reason: "von-rs was built without the `metal` feature".into(),
        });
    }
    if !metal_available() {
        return Err(VonError::DeviceUnavailable {
            requested: requested.into(),
            reason: "no Metal device is visible to this process".into(),
        });
    }
    Ok(Device::new_metal(0)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_macos_accelerators() {
        let openvino = [
            "openvino",
            "ov",
            "intel",
            "intel_gpu",
            "openvino:gpu",
            "ov:cpu",
        ];
        let others = ["cuda", "cuda:0", "rocm", "hip", "dml", "directml", "tpu"];
        for d in others.into_iter().chain(openvino) {
            assert!(
                matches!(resolve_device(Some(d)), Err(VonError::UnsupportedDevice(_))),
                "{d}"
            );
        }
    }

    #[test]
    fn cpu_is_always_available() {
        assert!(resolve_device(Some(" CPU ")).unwrap().is_cpu());
    }
}
