use self::engine::*;
use self::result_code::VoicevoxResultCode;
use self::status::*;
use super::*;
use once_cell::sync::Lazy;
use onnxruntime::{
    ndarray,
    session::{AnyArray, NdArray},
};
use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

// const PHONEME_LENGTH_MINIMAL: f32 = 0.01;

// static SPEAKER_ID_MAP: Lazy<BTreeMap<u32, (usize, u32)>> =
//     Lazy::new(|| include!("include_speaker_id_map.rs").into_iter().collect());

pub struct VoicevoxCore {
    synthesis_engine: SynthesisEngine,
    use_gpu: bool,
}

impl VoicevoxCore {
    pub fn new_with_initialize(root_dir_path: &Path, options: InitializeOptions) -> Result<Self> {
        let mut this = Self::new();
        this.initialize(root_dir_path, options)?;
        Ok(this)
    }

    pub fn new_with_mutex() -> Mutex<VoicevoxCore> {
        Mutex::new(Self::new())
    }

    fn new() -> Self {
        #[cfg(windows)]
        list_windows_video_cards();

        Self {
            synthesis_engine: SynthesisEngine::new(
                InferenceCore::new(false, None),
                OpenJtalk::initialize(),
            ),
            use_gpu: false,
        }
    }

    pub fn initialize(&mut self, root_dir_path: &Path, options: InitializeOptions) -> Result<()> {
        let use_gpu = match options.acceleration_mode {
            AccelerationMode::Auto => {
                let supported_devices = SupportedDevices::get_supported_devices()?;

                cfg_if! {
                    if #[cfg(feature="directml")]{
                        *supported_devices.dml()

                    } else {
                        *supported_devices.cuda()
                    }
                }
            }
            AccelerationMode::Cpu => false,
            AccelerationMode::Gpu => true,
        };
        self.use_gpu = use_gpu;
        self.synthesis_engine.inference_core_mut().initialize(
            root_dir_path,
            use_gpu,
            options.cpu_num_threads,
            options.load_all_models,
        )?;
        if let Some(open_jtalk_dict_dir) = options.open_jtalk_dict_dir {
            self.synthesis_engine
                .load_openjtalk_dict(open_jtalk_dict_dir)?;
        }
        Ok(())
    }

    pub fn is_gpu_mode(&self) -> bool {
        self.use_gpu
    }

    pub fn load_model(&mut self, speaker_id: u32) -> Result<()> {
        self.synthesis_engine
            .inference_core_mut()
            .load_model(speaker_id)
    }

    pub fn is_model_loaded(&self, speaker_id: u32) -> bool {
        self.synthesis_engine
            .inference_core()
            .is_model_loaded(speaker_id)
    }

    pub fn finalize(&mut self) {
        self.synthesis_engine.inference_core_mut().finalize()
    }

    pub const fn get_version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    pub fn get_metas_json(&mut self) -> &CStr {
        self.synthesis_engine.inference_core_mut().metas()
    }

    pub fn get_supported_devices_json(&self) -> &'static CStr {
        &SUPPORTED_DEVICES_CSTRING
    }

    pub fn variance_forward(
        &mut self,
        phoneme_vector: &[i64],
        accent_vector: &[i64],
        speaker_id: u32,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.synthesis_engine.inference_core_mut().variance_forward(
            phoneme_vector,
            accent_vector,
            speaker_id,
        )
    }

    pub fn decode_forward(
        &mut self,
        phoneme_vector: &[i64],
        pitch_vector: &[f32],
        duration_vector: &[f32],
        speaker_id: u32,
    ) -> Result<Vec<f32>> {
        self.synthesis_engine.inference_core_mut().decode_forward(
            phoneme_vector,
            pitch_vector,
            duration_vector,
            speaker_id,
        )
    }

    pub fn audio_query(
        &mut self,
        text: &str,
        speaker_id: u32,
        options: AudioQueryOptions,
    ) -> Result<AudioQueryModel> {
        if !self.synthesis_engine.is_openjtalk_dict_loaded() {
            return Err(Error::NotLoadedOpenjtalkDict);
        }
        let accent_phrases = if options.kana {
            parse_kana(text)?
        } else {
            self.synthesis_engine
                .create_accent_phrases(text, speaker_id)?
        };

        let kana = create_kana(&accent_phrases);

        Ok(AudioQueryModel::new(
            accent_phrases,
            1.,
            0.,
            1.,
            1.,
            0.1,
            0.1,
            SynthesisEngine::DEFAULT_SAMPLING_RATE,
            false,
            kana,
        ))
    }

    pub fn synthesis(
        &mut self,
        audio_query: &AudioQueryModel,
        speaker_id: u32,
        options: SynthesisOptions,
    ) -> Result<Vec<u8>> {
        self.synthesis_engine.synthesis_wave_format(
            audio_query,
            speaker_id,
            options.enable_interrogative_upspeak,
        )
    }

    pub fn tts(&mut self, text: &str, speaker_id: u32, options: TtsOptions) -> Result<Vec<u8>> {
        let audio_query = &self.audio_query(text, speaker_id, AudioQueryOptions::from(&options))?;
        self.synthesis(audio_query, speaker_id, SynthesisOptions::from(&options))
    }
}

#[derive(Default)]
pub struct AudioQueryOptions {
    pub kana: bool,
}

impl From<&TtsOptions> for AudioQueryOptions {
    fn from(options: &TtsOptions) -> Self {
        Self { kana: options.kana }
    }
}

#[derive(Default, Debug, PartialEq, Eq)]
pub enum AccelerationMode {
    #[default]
    Auto,
    Cpu,
    Gpu,
}

#[derive(Default)]
pub struct InitializeOptions {
    pub acceleration_mode: AccelerationMode,
    pub cpu_num_threads: u16,
    pub load_all_models: bool,
    pub open_jtalk_dict_dir: Option<PathBuf>,
}

pub struct SynthesisOptions {
    pub enable_interrogative_upspeak: bool,
}

impl From<&TtsOptions> for SynthesisOptions {
    fn from(options: &TtsOptions) -> Self {
        Self {
            enable_interrogative_upspeak: options.enable_interrogative_upspeak,
        }
    }
}

pub struct TtsOptions {
    pub kana: bool,
    pub enable_interrogative_upspeak: bool,
}

impl Default for TtsOptions {
    fn default() -> Self {
        Self {
            enable_interrogative_upspeak: true,
            kana: Default::default(),
        }
    }
}

#[derive(new)]
pub struct InferenceCore {
    initialized: bool,
    status_option: Option<Status>,
}

impl InferenceCore {
    pub fn initialize(
        &mut self,
        root_dir_path: &Path,
        use_gpu: bool,
        cpu_num_threads: u16,
        load_all_models: bool,
    ) -> Result<()> {
        self.initialized = false;
        if !use_gpu || self.can_support_gpu_feature()? {
            let mut status = Status::new(root_dir_path, use_gpu, cpu_num_threads);

            status.load()?;

            if load_all_models {
                for library_uuid in status.usable_libraries.clone() {
                    status.load_model(&library_uuid)?;
                }
            }

            self.status_option = Some(status);
            self.initialized = true;
            Ok(())
        } else {
            Err(Error::GpuSupport)
        }
    }
    fn can_support_gpu_feature(&self) -> Result<bool> {
        let supported_devices = SupportedDevices::get_supported_devices()?;

        cfg_if! {
            if #[cfg(feature = "directml")]{
                Ok(*supported_devices.dml())
            } else{
                Ok(*supported_devices.cuda())
            }
        }
    }
    pub fn load_model(&mut self, speaker_id: u32) -> Result<()> {
        if self.initialized {
            let status = self
                .status_option
                .as_mut()
                .ok_or(Error::UninitializedStatus)?;
            if let Some(library_uuid) = status.get_library_uuid_from_speaker_id(speaker_id) {
                status.load_model(&library_uuid)
            } else {
                Err(Error::InvalidSpeakerId { speaker_id })
            }
        } else {
            Err(Error::UninitializedStatus)
        }
    }
    pub fn is_model_loaded(&self, speaker_id: u32) -> bool {
        if let Some(status) = self.status_option.as_ref() {
            if let Some(library_uuid) = status.get_library_uuid_from_speaker_id(speaker_id) {
                status.is_model_loaded(&library_uuid)
            } else {
                false
            }
        } else {
            false
        }
    }
    pub fn finalize(&mut self) {
        self.initialized = false;
        self.status_option = None;
    }

    pub fn metas(&mut self) -> &CStr {
        if let Some(status) = self.status_option.as_mut() {
            &status.metas_str
        } else {
            <&CStr>::default()
        }
    }

    pub fn variance_forward(
        &mut self,
        phoneme_vector: &[i64],
        accent_vector: &[i64],
        speaker_id: u32,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        if !self.initialized {
            return Err(Error::UninitializedStatus);
        }

        let status = self
            .status_option
            .as_mut()
            .ok_or(Error::UninitializedStatus)?;

        let library_uuid =
            if let Some(library_uuid) = status.get_library_uuid_from_speaker_id(speaker_id) {
                library_uuid
            } else {
                return Err(Error::InvalidSpeakerId { speaker_id });
            };

        // NOTE: statusのusable_model_mapが不正でありうる場合、エラーを報告すべき？
        let start_speaker_id = status
            .usable_model_map
            .get(&library_uuid)
            .unwrap()
            .model_config
            .start_id as i64;
        let model_speaker_id = speaker_id as i64 - start_speaker_id;

        let mut phoneme_vector_array = NdArray::new(
            ndarray::arr1(phoneme_vector)
                .into_shape([1, phoneme_vector.len()])
                .unwrap(),
        );
        let mut accent_vector_array = NdArray::new(
            ndarray::arr1(accent_vector)
                .into_shape([1, accent_vector.len()])
                .unwrap(),
        );
        let mut speaker_id_array = NdArray::new(ndarray::arr1(&[model_speaker_id]));

        let input_tensors: Vec<&mut dyn AnyArray> = vec![
            &mut phoneme_vector_array,
            &mut accent_vector_array,
            &mut speaker_id_array,
        ];

        status.variance_session_run(&library_uuid, input_tensors)
    }

    pub fn decode_forward(
        &mut self,
        phoneme_vector: &[i64],
        pitch_vector: &[f32],
        duration_vector: &[f32],
        speaker_id: u32,
    ) -> Result<Vec<f32>> {
        if !self.initialized {
            return Err(Error::UninitializedStatus);
        }

        let status = self
            .status_option
            .as_mut()
            .ok_or(Error::UninitializedStatus)?;

        let library_uuid =
            if let Some(library_uuid) = status.get_library_uuid_from_speaker_id(speaker_id) {
                library_uuid
            } else {
                return Err(Error::InvalidSpeakerId { speaker_id });
            };

        // NOTE: statusのusable_model_mapが不正でありうる場合、エラーを報告すべき？
        let model_config = status
            .usable_model_map
            .get(&library_uuid)
            .unwrap()
            .model_config
            .clone();
        let start_speaker_id = model_config.start_id as i64;
        let model_speaker_id = speaker_id as i64 - start_speaker_id;

        let mut phoneme_vector_array = NdArray::new(
            ndarray::arr1(phoneme_vector)
                .into_shape([1, phoneme_vector.len()])
                .unwrap(),
        );
        let mut pitch_vector_array = NdArray::new(
            ndarray::arr1(pitch_vector)
                .into_shape([1, pitch_vector.len()])
                .unwrap(),
        );
        let mut speaker_id_array = NdArray::new(ndarray::arr1(&[model_speaker_id]));

        let embedder_input_tensors: Vec<&mut dyn AnyArray> = vec![
            &mut phoneme_vector_array,
            &mut pitch_vector_array,
            &mut speaker_id_array,
        ];

        let embedded_vector =
            &status.embedder_session_run(&library_uuid, embedder_input_tensors)?;

        let length_regulator_type = &model_config.length_regulator;

        let length_regulated_vector: Vec<f32>;
        if length_regulator_type == "normal" {
            length_regulated_vector = status.length_regulator(
                phoneme_vector.len(),
                embedded_vector,
                duration_vector,
                None,
                None,
                None,
            );
        } else if length_regulator_type == "gaussian" {
            length_regulated_vector = status.gaussian_upsampling(
                phoneme_vector.len(),
                embedded_vector,
                duration_vector,
                None,
                None,
            );
        } else {
            return Err(Error::InvalidLengthRegulator {
                length_regulator_type: length_regulator_type.to_owned(),
            });
        }
        let new_length = length_regulated_vector.len() / Status::HIDDEN_SIZE;

        let mut length_regulated_vector_array = NdArray::new(
            ndarray::arr1(length_regulated_vector.as_slice())
                .into_shape([1, new_length, Status::HIDDEN_SIZE])
                .unwrap(),
        );

        let decoder_input_tensors: Vec<&mut dyn AnyArray> =
            vec![&mut length_regulated_vector_array];

        status.decoder_session_run(&library_uuid, decoder_input_tensors)
    }
}

pub static SUPPORTED_DEVICES: Lazy<SupportedDevices> =
    Lazy::new(|| SupportedDevices::get_supported_devices().unwrap());

pub static SUPPORTED_DEVICES_CSTRING: Lazy<CString> =
    Lazy::new(|| CString::new(SUPPORTED_DEVICES.to_json().to_string()).unwrap());

// fn get_model_index_and_speaker_id(speaker_id: u32) -> Option<(usize, u32)> {
//     SPEAKER_ID_MAP.get(&speaker_id).copied()
// }

pub const fn error_result_to_message(result_code: VoicevoxResultCode) -> &'static str {
    // C APIのため、messageには必ず末尾にNULL文字を追加する
    use VoicevoxResultCode::*;
    match result_code {
        VOICEVOX_RESULT_NOT_LOADED_OPENJTALK_DICT_ERROR => {
            "OpenJTalkの辞書が読み込まれていません\0"
        }
        VOICEVOX_RESULT_LOAD_MODEL_ERROR => {
            "modelデータ読み込み中にOnnxruntimeエラーが発生しました\0"
        }
        VOICEVOX_RESULT_LOAD_METAS_ERROR => "メタデータ読み込みに失敗しました\0",

        VOICEVOX_RESULT_GPU_SUPPORT_ERROR => "GPU機能をサポートすることができません\0",
        VOICEVOX_RESULT_GET_SUPPORTED_DEVICES_ERROR => {
            "サポートされているデバイス情報取得中にエラーが発生しました\0"
        }

        VOICEVOX_RESULT_OK => "エラーが発生しませんでした\0",
        VOICEVOX_RESULT_UNINITIALIZED_STATUS_ERROR => "Statusが初期化されていません\0",
        VOICEVOX_RESULT_INVALID_SPEAKER_ID_ERROR => "無効なspeaker_idです\0",
        VOICEVOX_RESULT_INVALID_MODEL_INDEX_ERROR => "無効なmodel_indexです\0",
        VOICEVOX_RESULT_INFERENCE_ERROR => "推論に失敗しました\0",
        VOICEVOX_RESULT_EXTRACT_FULL_CONTEXT_LABEL_ERROR => {
            "入力テキストからのフルコンテキストラベル抽出に失敗しました\0"
        }
        VOICEVOX_RESULT_INVALID_UTF8_INPUT_ERROR => "入力テキストが無効なUTF-8データでした\0",
        VOICEVOX_RESULT_PARSE_KANA_ERROR => {
            "入力テキストをAquesTalkライクな読み仮名としてパースすることに失敗しました\0"
        }
        VOICEVOX_RESULT_INVALID_AUDIO_QUERY_ERROR => "無効なaudio_queryです\0",
        SHAREVOX_RESULT_LOAD_LIBRARIES_ERROR => "libraries.jsonの読み込みに失敗しました\0",
        SHAREVOX_RESULT_LOAD_MODEL_CONFIG_ERROR => "model_config.jsonの読み込みに失敗しました\0",
        SHAREVOX_RESULT_INVALID_LIBRARY_UUID_ERROR => "無効なlibrary_uuidです\0",
        SHAREVOX_RESULT_INVALID_LENGTH_REGULATOR_ERROR => {
            "model_config.jsonのlength_regulatorが無効です\0"
        }
    }
}

#[cfg(windows)]
fn list_windows_video_cards() {
    use std::{ffi::OsString, os::windows::ffi::OsStringExt as _};

    use humansize::BINARY;
    use tracing::{error, info};
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory, IDXGIFactory, DXGI_ADAPTER_DESC, DXGI_ERROR_NOT_FOUND,
    };

    info!("検出されたGPU (DirectMLには1番目のGPUが使われます):");
    match list_windows_video_cards() {
        Ok(descs) => {
            for desc in descs {
                let description = OsString::from_wide(trim_nul(&desc.Description));
                let vram = humansize::format_size(desc.DedicatedVideoMemory, BINARY);
                info!("  - {description:?} ({vram})");
            }
        }
        Err(err) => error!("{err}"),
    }

    fn list_windows_video_cards() -> windows::core::Result<Vec<DXGI_ADAPTER_DESC>> {
        #[allow(unsafe_code)]
        unsafe {
            let factory = CreateDXGIFactory::<IDXGIFactory>()?;
            (0..)
                .map(|i| factory.EnumAdapters(i)?.GetDesc())
                .take_while(|r| !matches!(r, Err(e) if e.code() == DXGI_ERROR_NOT_FOUND))
                .collect()
        }
    }

    fn trim_nul(s: &[u16]) -> &[u16] {
        &s[..s.iter().position(|&c| c == 0x0000).unwrap_or(s.len())]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numerics::F32Ext as _;
    use pretty_assertions::assert_eq;

    #[rstest]
    fn finalize_works() {
        let internal = VoicevoxCore::new_with_mutex();
        let result = internal.lock().unwrap().initialize(
            Path::new(concat!(env!("CARGO_WORKSPACE_DIR"), "/model/")),
            InitializeOptions::default(),
        );
        assert_eq!(Ok(()), result);
        internal.lock().unwrap().finalize();
        assert_eq!(
            false,
            internal
                .lock()
                .unwrap()
                .synthesis_engine
                .inference_core()
                .initialized
        );
        assert_eq!(
            true,
            internal
                .lock()
                .unwrap()
                .synthesis_engine
                .inference_core()
                .status_option
                .is_none()
        );
    }

    #[rstest]
    #[case(0, Err(Error::UninitializedStatus), Ok(()))]
    #[case(1, Err(Error::UninitializedStatus), Ok(()))]
    #[case(999, Err(Error::UninitializedStatus), Err(Error::InvalidSpeakerId{speaker_id:999}))]
    fn load_model_works(
        #[case] speaker_id: u32,
        #[case] expected_result_at_uninitialized: Result<()>,
        #[case] expected_result_at_initialized: Result<()>,
    ) {
        let internal = VoicevoxCore::new_with_mutex();
        let result = internal.lock().unwrap().load_model(speaker_id);
        assert_eq!(expected_result_at_uninitialized, result);

        internal
            .lock()
            .unwrap()
            .initialize(
                Path::new(concat!(env!("CARGO_WORKSPACE_DIR"), "/model/")),
                InitializeOptions {
                    acceleration_mode: AccelerationMode::Cpu,
                    ..Default::default()
                },
            )
            .unwrap();
        let result = internal.lock().unwrap().load_model(speaker_id);
        assert_eq!(
            expected_result_at_initialized, result,
            "got load_model result"
        );
    }

    #[rstest]
    fn is_use_gpu_works() {
        let internal = VoicevoxCore::new_with_mutex();
        assert_eq!(false, internal.lock().unwrap().is_gpu_mode());
        internal
            .lock()
            .unwrap()
            .initialize(
                Path::new(concat!(env!("CARGO_WORKSPACE_DIR"), "/model/")),
                InitializeOptions {
                    acceleration_mode: AccelerationMode::Cpu,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(false, internal.lock().unwrap().is_gpu_mode());
    }

    #[rstest]
    #[case(0, true)]
    #[case(1, true)]
    #[case(999, false)]
    fn is_model_loaded_works(#[case] speaker_id: u32, #[case] expected: bool) {
        let internal = VoicevoxCore::new_with_mutex();
        assert!(
            !internal.lock().unwrap().is_model_loaded(speaker_id),
            "expected is_model_loaded to return false, but got true",
        );

        internal
            .lock()
            .unwrap()
            .initialize(
                Path::new(concat!(env!("CARGO_WORKSPACE_DIR"), "/model/")),
                InitializeOptions {
                    acceleration_mode: AccelerationMode::Cpu,
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(
            !internal.lock().unwrap().is_model_loaded(speaker_id),
            "expected is_model_loaded to return false, but got true",
        );

        internal
            .lock()
            .unwrap()
            .load_model(speaker_id)
            .unwrap_or(());
        assert_eq!(
            internal.lock().unwrap().is_model_loaded(speaker_id),
            expected,
            "expected is_model_loaded return value against speaker_id `{}` is `{}`, but got `{}`",
            speaker_id,
            expected,
            !expected
        );
    }

    #[rstest]
    fn supported_devices_works() {
        let internal = VoicevoxCore::new_with_mutex();
        let cstr_result = internal.lock().unwrap().get_supported_devices_json();
        assert!(cstr_result.to_str().is_ok(), "{:?}", cstr_result);

        let json_result: std::result::Result<SupportedDevices, _> =
            serde_json::from_str(cstr_result.to_str().unwrap());
        assert!(json_result.is_ok(), "{:?}", json_result);
    }

    // #[rstest]
    // #[case(0, Some((0,0)))]
    // #[case(1, Some((0,1)))]
    // #[case(999, None)]
    // fn get_model_index_and_speaker_id_works(
    //     #[case] speaker_id: u32,
    //     #[case] expected: Option<(usize, u32)>,
    // ) {
    //     let actual = get_model_index_and_speaker_id(speaker_id);
    //     assert_eq!(expected, actual);
    // }

    #[rstest]
    fn variance_forward_works() {
        let internal = VoicevoxCore::new_with_mutex();
        internal
            .lock()
            .unwrap()
            .initialize(
                Path::new(concat!(env!("CARGO_WORKSPACE_DIR"), "/model/")),
                InitializeOptions {
                    load_all_models: true,
                    acceleration_mode: AccelerationMode::Cpu,
                    ..Default::default()
                },
            )
            .unwrap();

        // 「こんにちは、音声合成の世界へようこそ」という文章を変換して得た phoneme_vector
        let phoneme_vector = [
            0, 23, 30, 4, 28, 21, 10, 21, 42, 7, 0, 30, 4, 35, 14, 14, 16, 30, 30, 35, 14, 14, 28,
            30, 35, 14, 23, 7, 21, 14, 43, 30, 30, 23, 30, 35, 30, 0,
        ];
        let accent_vector = vec![0; phoneme_vector.len()];

        let result = internal
            .lock()
            .unwrap()
            .variance_forward(&phoneme_vector, &accent_vector, 0);

        assert!(result.is_ok(), "{:?}", result);

        let (pitch, duration) = result.unwrap();
        assert_eq!(pitch.len(), phoneme_vector.len());
        assert_eq!(duration.len(), phoneme_vector.len());
    }

    // #[rstest]
    // fn predict_intonation_works() {
    //     let internal = VoicevoxCore::new_with_mutex();
    //     internal
    //         .lock()
    //         .unwrap()
    //         .initialize(InitializeOptions {
    //             load_all_models: true,
    //             acceleration_mode: AccelerationMode::Cpu,
    //             ..Default::default()
    //         })
    //         .unwrap();

    //     // 「テスト」という文章に対応する入力
    //     let vowel_phoneme_vector = [0, 14, 6, 30, 0];
    //     let consonant_phoneme_vector = [-1, 37, 35, 37, -1];
    //     let start_accent_vector = [0, 1, 0, 0, 0];
    //     let end_accent_vector = [0, 1, 0, 0, 0];
    //     let start_accent_phrase_vector = [0, 1, 0, 0, 0];
    //     let end_accent_phrase_vector = [0, 0, 0, 1, 0];

    //     let result = internal.lock().unwrap().predict_intonation(
    //         vowel_phoneme_vector.len(),
    //         &vowel_phoneme_vector,
    //         &consonant_phoneme_vector,
    //         &start_accent_vector,
    //         &end_accent_vector,
    //         &start_accent_phrase_vector,
    //         &end_accent_phrase_vector,
    //         0,
    //     );

    //     assert!(result.is_ok(), "{:?}", result);
    //     assert_eq!(result.unwrap().len(), vowel_phoneme_vector.len());
    // }

    #[rstest]
    fn decode_forward_works() {
        let internal = VoicevoxCore::new_with_mutex();
        internal
            .lock()
            .unwrap()
            .initialize(
                Path::new(concat!(env!("CARGO_WORKSPACE_DIR"), "/model/")),
                InitializeOptions {
                    acceleration_mode: AccelerationMode::Cpu,
                    load_all_models: true,
                    ..Default::default()
                },
            )
            .unwrap();

        // 「こんにちは、音声合成の世界へようこそ」という文章を変換して得た phoneme_vector
        let phoneme_vector = [
            0, 23, 30, 4, 28, 21, 10, 21, 42, 7, 0, 30, 4, 35, 14, 14, 16, 30, 30, 35, 14, 14, 28,
            30, 35, 14, 23, 7, 21, 14, 43, 30, 30, 23, 30, 35, 30, 0,
        ];
        let pitch_vector = vec![5.5; phoneme_vector.len()];
        let duration_vector = vec![0.1; phoneme_vector.len()];

        let result = internal.lock().unwrap().decode_forward(
            &phoneme_vector,
            &pitch_vector,
            &duration_vector,
            0,
        );

        assert!(result.is_ok(), "{:?}", result);
        assert_eq!(
            result.unwrap().len(),
            (0.1 * 93.75).round_ties_even_() as usize * 512 * phoneme_vector.len()
        );
    }

    #[rstest]
    #[async_std::test]
    async fn audio_query_works() {
        let open_jtalk_dic_dir = download_open_jtalk_dict_if_no_exists().await;

        let core = VoicevoxCore::new_with_mutex();
        core.lock()
            .unwrap()
            .initialize(
                Path::new(concat!(env!("CARGO_WORKSPACE_DIR"), "/model/")),
                InitializeOptions {
                    acceleration_mode: AccelerationMode::Cpu,
                    load_all_models: true,
                    open_jtalk_dict_dir: Some(open_jtalk_dic_dir),
                    ..Default::default()
                },
            )
            .unwrap();

        let query = core
            .lock()
            .unwrap()
            .audio_query("これはテストです", 0, Default::default())
            .unwrap();

        assert_eq!(query.accent_phrases().len(), 2);

        assert_eq!(query.accent_phrases()[0].moras().len(), 3);
        for (i, (text, consonant, vowel)) in [("コ", "k", "o"), ("レ", "r", "e"), ("ワ", "w", "a")]
            .iter()
            .enumerate()
        {
            let mora = query.accent_phrases()[0].moras().get(i).unwrap();
            assert_eq!(mora.text(), text);
            assert_eq!(mora.consonant(), &Some(consonant.to_string()));
            assert_eq!(mora.vowel(), vowel);
        }
        assert_eq!(query.accent_phrases()[0].accent(), &3);

        assert_eq!(query.accent_phrases()[1].moras().len(), 5);
        for (i, (text, consonant, vowel)) in [
            ("テ", "t", "e"),
            ("ス", "s", "U"),
            ("ト", "t", "o"),
            ("デ", "d", "e"),
            ("ス", "s", "U"),
        ]
        .iter()
        .enumerate()
        {
            let mora = query.accent_phrases()[1].moras().get(i).unwrap();
            assert_eq!(mora.text(), text);
            assert_eq!(mora.consonant(), &Some(consonant.to_string()));
            assert_eq!(mora.vowel(), vowel);
        }
        assert_eq!(query.accent_phrases()[1].accent(), &1);
        assert_eq!(query.kana(), "コレワ'/テ'_ストデ_ス");
    }

    #[rstest]
    fn get_version_works() {
        assert_eq!("0.0.0", VoicevoxCore::get_version());
    }
}
