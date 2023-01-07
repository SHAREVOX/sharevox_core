use super::*;
use libc::c_int;

pub use voicevox_core::result_code::VoicevoxResultCode;

static ERROR_MESSAGE: Lazy<Mutex<String>> = Lazy::new(|| Mutex::new(String::new()));

fn set_message(message: &str) {
    ERROR_MESSAGE
        .lock()
        .unwrap()
        .replace_range(.., &format!("{}\0", message));
}

#[no_mangle]
pub extern "C" fn initialize(
    root_dir_path: *const c_char,
    use_gpu: bool,
    cpu_num_threads: c_int,
    load_all_models: bool,
) -> bool {
    let Ok(root_dir_path) = unsafe { CStr::from_ptr(root_dir_path) }.to_str() else {
        return false;
    };
    let result = lock_internal().initialize(
        root_dir_path.as_ref(),
        voicevox_core::InitializeOptions {
            acceleration_mode: if use_gpu {
                voicevox_core::AccelerationMode::Gpu
            } else {
                voicevox_core::AccelerationMode::Cpu
            },
            cpu_num_threads: cpu_num_threads as u16,
            load_all_models,
            ..Default::default()
        },
    );
    if let Some(err) = result.err() {
        set_message(&format!("{}", err));
        false
    } else {
        true
    }
}

#[no_mangle]
pub extern "C" fn load_model(speaker_id: i64) -> bool {
    let result = lock_internal().load_model(speaker_id as u32);
    if let Some(err) = result.err() {
        set_message(&format!("{}", err));
        false
    } else {
        true
    }
}

#[no_mangle]
pub extern "C" fn is_model_loaded(speaker_id: i64) -> bool {
    lock_internal().is_model_loaded(speaker_id as u32)
}

#[no_mangle]
pub extern "C" fn finalize() {
    lock_internal().finalize()
}

#[no_mangle]
pub extern "C" fn metas() -> *const c_char {
    voicevox_get_metas_json()
}

#[no_mangle]
pub extern "C" fn last_error_message() -> *const c_char {
    ERROR_MESSAGE.lock().unwrap().as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn supported_devices() -> *const c_char {
    voicevox_get_supported_devices_json()
}

#[no_mangle]
pub extern "C" fn variance_forward(
    length: i64,
    phoneme_list: *mut i64,
    accent_list: *mut i64,
    speaker_id: *mut i64,
    pitch_output: *mut f32,
    duration_output: *mut f32,
) -> bool {
    let result = lock_internal().variance_forward(
        unsafe { std::slice::from_raw_parts_mut(phoneme_list, length as usize) },
        unsafe { std::slice::from_raw_parts_mut(accent_list, length as usize) },
        unsafe { *speaker_id as u32 },
    );
    match result {
        Ok(output_vec_pair) => {
            let pitch_output_slice =
                unsafe { std::slice::from_raw_parts_mut(pitch_output, length as usize) };
            pitch_output_slice.clone_from_slice(&output_vec_pair.0);
            let duration_output_slice =
                unsafe { std::slice::from_raw_parts_mut(duration_output, length as usize) };
            duration_output_slice.clone_from_slice(&output_vec_pair.1);
            true
        }
        Err(err) => {
            set_message(&format!("{}", err));
            false
        }
    }
}

#[no_mangle]
pub extern "C" fn decode_forward(
    length: i64,
    phonemes: *mut i64,
    pitches: *mut f32,
    durations: *mut f32,
    speaker_id: *mut i64,
    output: *mut f32,
) -> bool {
    let length = length as usize;
    let result = lock_internal().decode_forward(
        unsafe { std::slice::from_raw_parts_mut(phonemes, length) },
        unsafe { std::slice::from_raw_parts_mut(pitches, length) },
        unsafe { std::slice::from_raw_parts_mut(durations, length) },
        unsafe { *speaker_id as u32 },
    );
    match result {
        Ok(output_vec) => {
            let output_slice = unsafe { std::slice::from_raw_parts_mut(output, output_vec.len()) };
            output_slice.clone_from_slice(&output_vec);
            true
        }
        Err(err) => {
            set_message(&format!("{}", err));
            false
        }
    }
}
