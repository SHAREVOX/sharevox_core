/// 処理結果を示す結果コード
#[repr(i32)]
#[derive(Debug, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum SharevoxResultCode {
    // C でのenum定義に合わせて大文字で定義している
    // 出力フォーマットを変更すればRustでよく使われているUpperCamelにできるが、実際に出力されるコードとの差異をできるだけ少なくするため
    /// 成功
    SHAREVOX_RESULT_OK = 0,
    /// open_jtalk辞書ファイルが読み込まれていない
    SHAREVOX_RESULT_NOT_LOADED_OPENJTALK_DICT_ERROR = 1,
    /// modelの読み込みに失敗した
    SHAREVOX_RESULT_LOAD_MODEL_ERROR = 2,
    /// サポートされているデバイス情報取得に失敗した
    SHAREVOX_RESULT_GET_SUPPORTED_DEVICES_ERROR = 3,
    /// GPUモードがサポートされていない
    SHAREVOX_RESULT_GPU_SUPPORT_ERROR = 4,
    /// メタ情報読み込みに失敗した
    SHAREVOX_RESULT_LOAD_METAS_ERROR = 5,
    /// ステータスが初期化されていない
    SHAREVOX_RESULT_UNINITIALIZED_STATUS_ERROR = 6,
    /// 無効なspeaker_idが指定された
    SHAREVOX_RESULT_INVALID_SPEAKER_ID_ERROR = 7,
    /// 無効なmodel_indexが指定された
    SHAREVOX_RESULT_INVALID_MODEL_INDEX_ERROR = 8,
    /// 推論に失敗した
    SHAREVOX_RESULT_INFERENCE_ERROR = 9,
    /// コンテキストラベル出力に失敗した
    SHAREVOX_RESULT_EXTRACT_FULL_CONTEXT_LABEL_ERROR = 10,
    /// 無効なutf8文字列が入力された
    SHAREVOX_RESULT_INVALID_UTF8_INPUT_ERROR = 11,
    /// aquestalk形式のテキストの解析に失敗した
    SHAREVOX_RESULT_PARSE_KANA_ERROR = 12,
    /// 無効なAudioQuery
    SHAREVOX_RESULT_INVALID_AUDIO_QUERY_ERROR = 13,
    /// libraries.jsonの読み込みに失敗した
    SHAREVOX_RESULT_LOAD_LIBRARIES_ERROR = 100,
    /// model_config.jsonの読み込みに失敗した
    SHAREVOX_RESULT_LOAD_MODEL_CONFIG_ERROR = 101,
    /// 無効なlibrary_uuidが指定された
    SHAREVOX_RESULT_INVALID_LIBRARY_UUID_ERROR = 102,
}
