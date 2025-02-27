use std::fmt::Debug;

use thiserror::Error;

use super::*;

pub(crate) fn into_result_code_with_error(result: CApiResult<()>) -> SharevoxResultCode {
    if let Err(err) = &result {
        display_error(err);
    }
    return into_result_code(result);

    fn display_error(err: &CApiError) {
        eprintln!("Error(Display): {err}");
        eprintln!("Error(Debug): {err:#?}");
    }

    fn into_result_code(result: CApiResult<()>) -> SharevoxResultCode {
        use voicevox_core::{result_code::SharevoxResultCode::*, Error::*};
        use CApiError::*;

        match result {
            Ok(()) => SHAREVOX_RESULT_OK,
            Err(RustApi(NotLoadedOpenjtalkDict)) => SHAREVOX_RESULT_NOT_LOADED_OPENJTALK_DICT_ERROR,
            Err(RustApi(GpuSupport)) => SHAREVOX_RESULT_GPU_SUPPORT_ERROR,
            Err(RustApi(LoadModel { .. })) => SHAREVOX_RESULT_LOAD_MODEL_ERROR,
            Err(RustApi(LoadMetas(_))) => SHAREVOX_RESULT_LOAD_METAS_ERROR,
            Err(RustApi(GetSupportedDevices(_))) => SHAREVOX_RESULT_GET_SUPPORTED_DEVICES_ERROR,
            Err(RustApi(UninitializedStatus)) => SHAREVOX_RESULT_UNINITIALIZED_STATUS_ERROR,
            Err(RustApi(InvalidSpeakerId { .. })) => SHAREVOX_RESULT_INVALID_SPEAKER_ID_ERROR,
            Err(RustApi(InvalidModelIndex { .. })) => SHAREVOX_RESULT_INVALID_MODEL_INDEX_ERROR,
            Err(RustApi(InferenceFailed)) => SHAREVOX_RESULT_INFERENCE_ERROR,
            Err(RustApi(ExtractFullContextLabel(_))) => {
                SHAREVOX_RESULT_EXTRACT_FULL_CONTEXT_LABEL_ERROR
            }
            Err(RustApi(ParseKana(_))) => SHAREVOX_RESULT_PARSE_KANA_ERROR,
            Err(RustApi(LoadLibraries(_))) => SHAREVOX_RESULT_LOAD_LIBRARIES_ERROR,
            Err(RustApi(LoadModelConfig { .. })) => SHAREVOX_RESULT_LOAD_MODEL_CONFIG_ERROR,
            Err(RustApi(InvalidLibraryUuid { .. })) => SHAREVOX_RESULT_INVALID_LIBRARY_UUID_ERROR,
            Err(InvalidUtf8Input) => SHAREVOX_RESULT_INVALID_UTF8_INPUT_ERROR,
            Err(InvalidAudioQuery(_)) => SHAREVOX_RESULT_INVALID_AUDIO_QUERY_ERROR,
        }
    }
}

type CApiResult<T> = std::result::Result<T, CApiError>;

#[derive(Error, Debug)]
pub(crate) enum CApiError {
    #[error("{0}")]
    RustApi(#[from] voicevox_core::Error),
    #[error("UTF-8として不正な入力です")]
    InvalidUtf8Input,
    #[allow(dead_code)]
    #[error("無効なAudioQueryです: {0}")]
    InvalidAudioQuery(serde_json::Error),
}

#[allow(dead_code)]
pub(crate) fn create_audio_query(
    japanese_or_kana: &CStr,
    speaker_id: u32,
    method: fn(
        &mut Internal,
        &str,
        u32,
        voicevox_core::AudioQueryOptions,
    ) -> Result<AudioQueryModel>,
    options: SharevoxAudioQueryOptions,
) -> CApiResult<CString> {
    let japanese_or_kana = ensure_utf8(japanese_or_kana)?;

    let audio_query = method(
        &mut lock_internal(),
        japanese_or_kana,
        speaker_id,
        options.into(),
    )?;
    Ok(CString::new(audio_query_model_to_json(&audio_query)).expect("should not contain '\\0'"))
}

#[allow(dead_code)]
fn audio_query_model_to_json(audio_query_model: &AudioQueryModel) -> String {
    serde_json::to_string(audio_query_model).expect("should be always valid")
}

#[allow(dead_code)]
pub(crate) unsafe fn write_json_to_ptr(output_ptr: *mut *mut c_char, json: &CStr) {
    let n = json.to_bytes_with_nul().len();
    let json_heap = libc::malloc(n);
    libc::memcpy(json_heap, json.as_ptr() as *const c_void, n);
    output_ptr.write(json_heap as *mut c_char);
}

#[allow(dead_code)]
pub(crate) unsafe fn write_wav_to_ptr(
    output_wav_ptr: *mut *mut u8,
    output_length_ptr: *mut usize,
    data: &[u8],
) {
    write_data_to_ptr(output_wav_ptr, output_length_ptr, data);
}

pub(crate) unsafe fn write_predict_pitch_and_duration_to_ptr(
    output_predict_pitch_ptr: *mut *mut f32,
    output_predict_duration_ptr: *mut *mut f32,
    output_predict_length_ptr: *mut usize,
    pitch_data: &[f32],
    duration_data: &[f32],
) {
    write_data_to_ptr(
        output_predict_pitch_ptr,
        output_predict_length_ptr,
        pitch_data,
    );
    write_data_to_ptr(
        output_predict_duration_ptr,
        output_predict_length_ptr,
        duration_data,
    );
}

#[allow(dead_code)]
pub(crate) unsafe fn write_predict_intonation_to_ptr(
    output_predict_intonation_ptr: *mut *mut f32,
    output_predict_intonation_length_ptr: *mut usize,
    data: &[f32],
) {
    write_data_to_ptr(
        output_predict_intonation_ptr,
        output_predict_intonation_length_ptr,
        data,
    );
}

#[allow(dead_code)]
pub(crate) unsafe fn write_decode_to_ptr(
    output_decode_ptr: *mut *mut f32,
    output_decode_length_ptr: *mut usize,
    data: &[f32],
) {
    write_data_to_ptr(output_decode_ptr, output_decode_length_ptr, data);
}

unsafe fn write_data_to_ptr<T>(
    output_data_ptr: *mut *mut T,
    output_length_ptr: *mut usize,
    data: &[T],
) {
    output_length_ptr.write(data.len());
    use std::mem;
    let num_bytes = mem::size_of_val(data);
    let data_heap = libc::malloc(num_bytes);
    libc::memcpy(data_heap, data.as_ptr() as *const c_void, num_bytes);
    output_data_ptr.write(data_heap as *mut T);
}

pub(crate) fn ensure_utf8(s: &CStr) -> CApiResult<&str> {
    s.to_str().map_err(|_| CApiError::InvalidUtf8Input)
}

impl From<voicevox_core::AudioQueryOptions> for SharevoxAudioQueryOptions {
    fn from(options: voicevox_core::AudioQueryOptions) -> Self {
        Self { kana: options.kana }
    }
}
impl From<SharevoxAudioQueryOptions> for voicevox_core::AudioQueryOptions {
    fn from(options: SharevoxAudioQueryOptions) -> Self {
        Self { kana: options.kana }
    }
}

impl From<SharevoxSynthesisOptions> for voicevox_core::SynthesisOptions {
    fn from(options: SharevoxSynthesisOptions) -> Self {
        Self {
            enable_interrogative_upspeak: options.enable_interrogative_upspeak,
        }
    }
}

impl From<voicevox_core::AccelerationMode> for SharevoxAccelerationMode {
    fn from(mode: voicevox_core::AccelerationMode) -> Self {
        use voicevox_core::AccelerationMode::*;
        match mode {
            Auto => Self::SHAREVOX_ACCELERATION_MODE_AUTO,
            Cpu => Self::SHAREVOX_ACCELERATION_MODE_CPU,
            Gpu => Self::SHAREVOX_ACCELERATION_MODE_GPU,
        }
    }
}

impl From<SharevoxAccelerationMode> for voicevox_core::AccelerationMode {
    fn from(mode: SharevoxAccelerationMode) -> Self {
        use SharevoxAccelerationMode::*;
        match mode {
            SHAREVOX_ACCELERATION_MODE_AUTO => Self::Auto,
            SHAREVOX_ACCELERATION_MODE_CPU => Self::Cpu,
            SHAREVOX_ACCELERATION_MODE_GPU => Self::Gpu,
        }
    }
}

impl Default for SharevoxInitializeOptions {
    fn default() -> Self {
        let options = voicevox_core::InitializeOptions::default();
        Self {
            acceleration_mode: options.acceleration_mode.into(),
            cpu_num_threads: options.cpu_num_threads,
            load_all_models: options.load_all_models,
            open_jtalk_dict_dir: null(),
        }
    }
}

impl SharevoxInitializeOptions {
    pub(crate) unsafe fn try_into_options(self) -> CApiResult<voicevox_core::InitializeOptions> {
        let open_jtalk_dict_dir = (!self.open_jtalk_dict_dir.is_null())
            .then(|| ensure_utf8(CStr::from_ptr(self.open_jtalk_dict_dir)).map(Into::into))
            .transpose()?;
        Ok(voicevox_core::InitializeOptions {
            acceleration_mode: self.acceleration_mode.into(),
            cpu_num_threads: self.cpu_num_threads,
            load_all_models: self.load_all_models,
            open_jtalk_dict_dir,
        })
    }
}

impl From<voicevox_core::TtsOptions> for SharevoxTtsOptions {
    fn from(options: voicevox_core::TtsOptions) -> Self {
        Self {
            kana: options.kana,
            enable_interrogative_upspeak: options.enable_interrogative_upspeak,
        }
    }
}

impl From<SharevoxTtsOptions> for voicevox_core::TtsOptions {
    fn from(options: SharevoxTtsOptions) -> Self {
        Self {
            kana: options.kana,
            enable_interrogative_upspeak: options.enable_interrogative_upspeak,
        }
    }
}

impl Default for SharevoxSynthesisOptions {
    fn default() -> Self {
        let options = voicevox_core::TtsOptions::default();
        Self {
            enable_interrogative_upspeak: options.enable_interrogative_upspeak,
        }
    }
}
