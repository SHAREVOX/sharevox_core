/// cbindgen:ignore
mod compatible_engine;
mod helpers;
use self::helpers::*;
use libc::c_void;
use once_cell::sync::Lazy;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr::null;
use std::sync::{Mutex, MutexGuard};
use std::{env, io};
use tracing_subscriber::EnvFilter;
use voicevox_core::AudioQueryModel;
use voicevox_core::Result;
use voicevox_core::VoicevoxCore;

#[cfg(test)]
use rstest::*;

type Internal = VoicevoxCore;

static INTERNAL: Lazy<Mutex<Internal>> = Lazy::new(|| {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(if env::var_os(EnvFilter::DEFAULT_ENV).is_some() {
            EnvFilter::from_default_env()
        } else {
            "error,sharevox_core=info,sharevox_core_c_api=info,onnxruntime=info".into()
        })
        .with_writer(io::stderr)
        .try_init();

    Internal::new_with_mutex()
});

pub(crate) fn lock_internal() -> MutexGuard<'static, Internal> {
    INTERNAL.lock().unwrap()
}

/*
 * Cの関数として公開するための型や関数を定義するこれらの実装はvoicevox_core/publish.rsに定義してある対応する関数にある
 * この関数ではvoicevox_core/publish.rsにある対応する関数の呼び出しと、その戻り値をCの形式に変換する処理のみとする
 * これはC文脈の処理と実装をわけるためと、内部実装の変更がAPIに影響を与えにくくするためである
 * voicevox_core/publish.rsにある対応する関数とはこのファイルに定義してある公開関数からsharevoxプレフィックスを取り除いた名前の関数である
 */

pub use voicevox_core::result_code::SharevoxResultCode;

/// ハードウェアアクセラレーションモードを設定する設定値
#[repr(i32)]
#[derive(Debug, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum SharevoxAccelerationMode {
    /// 実行環境に合った適切なハードウェアアクセラレーションモードを選択する
    SHAREVOX_ACCELERATION_MODE_AUTO = 0,
    /// ハードウェアアクセラレーションモードを"CPU"に設定する
    SHAREVOX_ACCELERATION_MODE_CPU = 1,
    /// ハードウェアアクセラレーションモードを"GPU"に設定する
    SHAREVOX_ACCELERATION_MODE_GPU = 2,
}

/// 初期化オプション
#[repr(C)]
pub struct SharevoxInitializeOptions {
    /// ハードウェアアクセラレーションモード
    acceleration_mode: SharevoxAccelerationMode,
    /// CPU利用数を指定
    /// 0を指定すると環境に合わせたCPUが利用される
    cpu_num_threads: u16,
    /// 全てのモデルを読み込む
    load_all_models: bool,
    /// open_jtalkの辞書ディレクトリ
    open_jtalk_dict_dir: *const c_char,
}

/// デフォルトの初期化オプションを生成する
/// @return デフォルト値が設定された初期化オプション
#[no_mangle]
pub extern "C" fn sharevox_make_default_initialize_options() -> SharevoxInitializeOptions {
    SharevoxInitializeOptions::default()
}

/// 初期化する
/// @param [in] options 初期化オプション
/// @return 結果コード #SharevoxResultCode
///
/// # Safety
/// @param root_dir_path NUL-terminatedな文字列を指す、有効なポインタであること
/// @param options open_jtalk_dict_dirがNUL-terminatedな文字列を指す、有効なポインタであること
#[no_mangle]
pub unsafe extern "C" fn sharevox_initialize(
    root_dir_path: *const c_char,
    options: SharevoxInitializeOptions,
) -> SharevoxResultCode {
    into_result_code_with_error((|| {
        let root_dir_path = ensure_utf8(CStr::from_ptr(root_dir_path))?.as_ref();
        let options = options.try_into_options()?;
        lock_internal().initialize(root_dir_path, options)?;
        Ok(())
    })())
}

static SHAREVOX_VERSION: once_cell::sync::Lazy<CString> =
    once_cell::sync::Lazy::new(|| CString::new(Internal::get_version()).unwrap());

/// sharevoxのバージョンを取得する
/// @return SemVerでフォーマットされたバージョン
#[no_mangle]
pub extern "C" fn sharevox_get_version() -> *const c_char {
    SHAREVOX_VERSION.as_ptr()
}

/// モデルを読み込む
/// @param [in] speaker_id 読み込むモデルの話者ID
/// @return 結果コード #SharevoxResultCode
#[no_mangle]
pub extern "C" fn sharevox_load_model(speaker_id: u32) -> SharevoxResultCode {
    into_result_code_with_error(lock_internal().load_model(speaker_id).map_err(Into::into))
}

/// ハードウェアアクセラレーションがGPUモードか判定する
/// @return GPUモードならtrue、そうでないならfalse
#[no_mangle]
pub extern "C" fn sharevox_is_gpu_mode() -> bool {
    lock_internal().is_gpu_mode()
}

/// 指定したspeaker_idのモデルが読み込まれているか判定する
/// @return モデルが読み込まれているのであればtrue、そうでないならfalse
#[no_mangle]
pub extern "C" fn sharevox_is_model_loaded(speaker_id: u32) -> bool {
    lock_internal().is_model_loaded(speaker_id)
}

/// このライブラリの利用を終了し、確保しているリソースを解放する
#[no_mangle]
pub extern "C" fn sharevox_finalize() {
    lock_internal().finalize()
}

/// メタ情報をjsonで取得する
/// @return メタ情報のjson文字列
#[no_mangle]
pub extern "C" fn sharevox_get_metas_json() -> *const c_char {
    lock_internal().get_metas_json().as_ptr()
}

/// サポートデバイス情報をjsonで取得する
/// @return サポートデバイス情報のjson文字列
#[no_mangle]
pub extern "C" fn sharevox_get_supported_devices_json() -> *const c_char {
    lock_internal().get_supported_devices_json().as_ptr()
}

/// 音素ごとのピッチと長さを推論する
/// @param [in] length phoneme_vector, accent_vector, output のデータ長
/// @param [in] phoneme_vector  音素データ
/// @param [in] accent_vector  アクセントデータ
/// @param [in] speaker_id 話者ID
/// @param [out] output_predict_length 出力データのサイズ
/// @param [out] output_predict_pitch_data ピッチデータの出力先
/// @param [out] output_predict_duration_data 音素長データの出力先
/// @return 結果コード #SharevoxResultCode
///
/// # Safety
/// @param phoneme_vector 必ずlengthの長さだけデータがある状態で渡すこと
/// @param accent_vector 必ずlengthの長さだけデータがある状態で渡すこと
/// @param output_predict_data_length uintptr_t 分のメモリ領域が割り当てられていること
/// @param output_predict_pitch_data 成功後にメモリ領域が割り当てられるので ::sharevox_predict_pitch_and_duration_data_free で解放する必要がある
/// @param output_predict_duration_data 成功後にメモリ領域が割り当てられるので ::sharevox_predict_pitch_and_duration_data_free で解放する必要がある
#[no_mangle]
pub unsafe extern "C" fn sharevox_predict_pitch_and_duration(
    length: usize,
    phoneme_vector: *mut i64,
    accent_vector: *mut i64,
    speaker_id: u32,
    output_predict_data_length: *mut usize,
    output_predict_pitch_data: *mut *mut f32,
    output_predict_duration_data: *mut *mut f32,
) -> SharevoxResultCode {
    into_result_code_with_error((|| {
        let output_vec_pair = lock_internal().predict_pitch_and_duration(
            std::slice::from_raw_parts_mut(phoneme_vector, length),
            std::slice::from_raw_parts_mut(accent_vector, length),
            speaker_id,
        )?;
        write_predict_pitch_and_duration_to_ptr(
            output_predict_pitch_data,
            output_predict_duration_data,
            output_predict_data_length,
            &output_vec_pair.0,
            &output_vec_pair.1,
        );
        Ok(())
    })())
}

/// ::sharevox_predict_pitch_and_durationで出力されたデータを解放する
/// @param[in] predict_pitch_data 確保されたメモリ領域
/// @param[in] predict_duration_data 確保されたメモリ領域
///
/// # Safety
/// @param predict_pitch_data 実行後に割り当てられたメモリ領域が解放される
/// @param predict_duration_data 実行後に割り当てられたメモリ領域が解放される
#[no_mangle]
pub unsafe extern "C" fn sharevox_predict_pitch_and_duration_data_free(
    predict_pitch_data: *mut f32,
    predict_duration_data: *mut f32,
) {
    libc::free(predict_pitch_data as *mut c_void);
    libc::free(predict_duration_data as *mut c_void);
}

/// decodeを実行する
/// @param [in] length  phoneme_vector, pitch_vector, duration_vector のデータ長
/// @param [in] phoneme_vector 音素データ
/// @param [in] pitch_vector ピッチデータ
/// @param [in] duration_vector 音素長データ
/// @param [in] speaker_id 話者ID
/// @param [out] output_decode_data_length 出力先データのサイズ
/// @param [out] output_decode_data データ出力先
/// @return 結果コード #SharevoxResultCode
///
/// # Safety
/// @param phoneme_vector 必ず length の長さだけデータがある状態で渡すこと
/// @param pitch_vector 必ず length の長さだけデータがある状態で渡すこと
/// @param duration_vector 必ず length の長さだけデータがある状態で渡すこと
/// @param output_decode_data_length uintptr_t 分のメモリ領域が割り当てられていること
/// @param output_decode_data 成功後にメモリ領域が割り当てられるので ::sharevox_decode_data_free で解放する必要がある
#[no_mangle]
pub unsafe extern "C" fn sharevox_decode(
    length: usize,
    phoneme_vector: *mut i64,
    pitch_vector: *mut f32,
    duration_vector: *mut f32,
    speaker_id: u32,
    output_decode_data_length: *mut usize,
    output_decode_data: *mut *mut f32,
) -> SharevoxResultCode {
    into_result_code_with_error((|| {
        let output_vec = lock_internal().decode(
            std::slice::from_raw_parts_mut(phoneme_vector, length),
            std::slice::from_raw_parts_mut(pitch_vector, length),
            std::slice::from_raw_parts_mut(duration_vector, length),
            speaker_id,
        )?;
        write_decode_to_ptr(output_decode_data, output_decode_data_length, &output_vec);
        Ok(())
    })())
}

/// ::sharevox_decodeで出力されたデータを解放する
/// @param[in] decode_data 確保されたメモリ領域
///
/// # Safety
/// @param decode_data 実行後に割り当てられたメモリ領域が解放される
#[no_mangle]
pub unsafe extern "C" fn sharevox_decode_data_free(decode_data: *mut f32) {
    libc::free(decode_data as *mut c_void);
}

/// Audio query のオプション
#[repr(C)]
pub struct SharevoxAudioQueryOptions {
    /// aquestalk形式のkanaとしてテキストを解釈する
    kana: bool,
}

/// デフォルトの AudioQuery のオプションを生成する
/// @return デフォルト値が設定された AudioQuery オプション
#[no_mangle]
pub extern "C" fn sharevox_make_default_audio_query_options() -> SharevoxAudioQueryOptions {
    voicevox_core::AudioQueryOptions::default().into()
}

/// AudioQuery を実行する
/// @param [in] text テキスト
/// @param [in] speaker_id 話者ID
/// @param [in] options AudioQueryのオプション
/// @param [out] output_audio_query_json AudioQuery を json でフォーマットしたもの
/// @return 結果コード #SharevoxResultCode
///
/// # Safety
/// @param text null終端文字列であること
/// @param output_audio_query_json 自動でheapメモリが割り当てられるので ::sharevox_audio_query_json_free で解放する必要がある
#[no_mangle]
pub unsafe extern "C" fn sharevox_audio_query(
    text: *const c_char,
    speaker_id: u32,
    options: SharevoxAudioQueryOptions,
    output_audio_query_json: *mut *mut c_char,
) -> SharevoxResultCode {
    into_result_code_with_error((|| {
        let text = CStr::from_ptr(text);
        let audio_query = &create_audio_query(text, speaker_id, Internal::audio_query, options)?;
        write_json_to_ptr(output_audio_query_json, audio_query);
        Ok(())
    })())
}

/// `sharevox_synthesis` のオプション
#[repr(C)]
pub struct SharevoxSynthesisOptions {
    /// 疑問文の調整を有効にする
    enable_interrogative_upspeak: bool,
}

/// デフォルトの `sharevox_synthesis` のオプションを生成する
/// @return デフォルト値が設定された `sharevox_synthesis` のオプション
#[no_mangle]
pub extern "C" fn sharevox_make_default_synthesis_options() -> SharevoxSynthesisOptions {
    SharevoxSynthesisOptions::default()
}

/// AudioQuery から音声合成する
/// @param [in] audio_query_json jsonフォーマットされた AudioQuery
/// @param [in] speaker_id  話者ID
/// @param [in] options AudioQueryから音声合成オプション
/// @param [out] output_wav_length 出力する wav データのサイズ
/// @param [out] output_wav wav データの出力先
/// @return 結果コード #SharevoxResultCode
///
/// # Safety
/// @param output_wav_length 出力先の領域が確保された状態でpointerに渡されていること
/// @param output_wav 自動で output_wav_length 分のデータが割り当てられるので ::sharevox_wav_free で解放する必要がある
#[no_mangle]
pub unsafe extern "C" fn sharevox_synthesis(
    audio_query_json: *const c_char,
    speaker_id: u32,
    options: SharevoxSynthesisOptions,
    output_wav_length: *mut usize,
    output_wav: *mut *mut u8,
) -> SharevoxResultCode {
    into_result_code_with_error((|| {
        let audio_query_json = CStr::from_ptr(audio_query_json)
            .to_str()
            .map_err(|_| CApiError::InvalidUtf8Input)?;
        let audio_query =
            &serde_json::from_str(audio_query_json).map_err(CApiError::InvalidAudioQuery)?;
        let wav = &lock_internal().synthesis(audio_query, speaker_id, options.into())?;
        write_wav_to_ptr(output_wav, output_wav_length, wav);
        Ok(())
    })())
}

/// テキスト音声合成オプション
#[repr(C)]
pub struct SharevoxTtsOptions {
    /// aquestalk形式のkanaとしてテキストを解釈する
    kana: bool,
    /// 疑問文の調整を有効にする
    enable_interrogative_upspeak: bool,
}

/// デフォルトのテキスト音声合成オプションを生成する
/// @return テキスト音声合成オプション
#[no_mangle]
pub extern "C" fn sharevox_make_default_tts_options() -> SharevoxTtsOptions {
    voicevox_core::TtsOptions::default().into()
}

/// テキスト音声合成を実行する
/// @param [in] text テキスト
/// @param [in] speaker_id 話者ID
/// @param [in] options テキスト音声合成オプション
/// @param [out] output_wav_length 出力する wav データのサイズ
/// @param [out] output_wav wav データの出力先
/// @return 結果コード #SharevoxResultCode
///
/// # Safety
/// @param output_wav_length 出力先の領域が確保された状態でpointerに渡されていること
/// @param output_wav は自動で output_wav_length 分のデータが割り当てられるので ::sharevox_wav_free で解放する必要がある
#[no_mangle]
pub unsafe extern "C" fn sharevox_tts(
    text: *const c_char,
    speaker_id: u32,
    options: SharevoxTtsOptions,
    output_wav_length: *mut usize,
    output_wav: *mut *mut u8,
) -> SharevoxResultCode {
    into_result_code_with_error((|| {
        let text = ensure_utf8(CStr::from_ptr(text))?;
        let output = lock_internal().tts(text, speaker_id, options.into())?;
        write_wav_to_ptr(output_wav, output_wav_length, output.as_slice());
        Ok(())
    })())
}

/// jsonフォーマットされた AudioQuery データのメモリを解放する
/// @param [in] audio_query_json 解放する json フォーマットされた AudioQuery データ
///
/// # Safety
/// @param wav 確保したメモリ領域が破棄される
#[no_mangle]
pub unsafe extern "C" fn sharevox_audio_query_json_free(audio_query_json: *mut c_char) {
    libc::free(audio_query_json as *mut c_void);
}

/// wav データのメモリを解放する
/// @param [in] wav 解放する wav データ
///
/// # Safety
/// @param wav 確保したメモリ領域が破棄される
#[no_mangle]
pub unsafe extern "C" fn sharevox_wav_free(wav: *mut u8) {
    libc::free(wav as *mut c_void);
}

/// エラー結果をメッセージに変換する
/// @param [in] result_code メッセージに変換する result_code
/// @return 結果コードを元に変換されたメッセージ文字列
#[no_mangle]
pub extern "C" fn sharevox_error_result_to_message(
    result_code: SharevoxResultCode,
) -> *const c_char {
    voicevox_core::error_result_to_message(result_code).as_ptr() as *const c_char
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;
    use pretty_assertions::assert_eq;
    use voicevox_core::Error;

    #[rstest]
    #[case(Ok(()), SharevoxResultCode::SHAREVOX_RESULT_OK)]
    #[case(
        Err(Error::NotLoadedOpenjtalkDict),
        SharevoxResultCode::SHAREVOX_RESULT_NOT_LOADED_OPENJTALK_DICT_ERROR
    )]
    #[case(
        Err(Error::LoadModel(anyhow!("some load model error"))),
        SharevoxResultCode::SHAREVOX_RESULT_LOAD_MODEL_ERROR
    )]
    #[case(
        Err(Error::GetSupportedDevices(anyhow!("some get supported devices error"))),
        SharevoxResultCode::SHAREVOX_RESULT_GET_SUPPORTED_DEVICES_ERROR
    )]
    fn into_result_code_with_error_works(
        #[case] result: Result<()>,
        #[case] expected: SharevoxResultCode,
    ) {
        let actual = into_result_code_with_error(result.map_err(Into::into));
        assert_eq!(expected, actual);
    }
}
