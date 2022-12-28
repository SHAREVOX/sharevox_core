use super::*;
use once_cell::sync::Lazy;
use onnxruntime::{
    environment::Environment,
    session::{AnyArray, Session},
    GraphOptimizationLevel, LoggingLevel,
};
use serde::{Deserialize, Serialize};

cfg_if! {
    if #[cfg(not(feature="directml"))]{
        use onnxruntime::CudaProviderOptions;
    }
}
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

pub struct Status {
    root_dir_path: PathBuf,
    light_session_options: SessionOptions, // 軽いモデルはこちらを使う
    heavy_session_options: SessionOptions, // 重いモデルはこちらを使う
    libraries: Option<BTreeMap<String, bool>>,
    pub usable_libraries: BTreeSet<String>,
    usable_model_data_map: BTreeMap<String, ModelData>,
    pub usable_model_map: BTreeMap<String, Models>,
    pub speaker_id_map: BTreeMap<u64, String>,
    metas_str: String,
}

pub struct Models {
    variance_session: Session<'static>,
    embedder_session: Session<'static>,
    decoder_session: Session<'static>,
    pub model_config: ModelConfig,
}

#[derive(new, Getters)]
struct SessionOptions {
    cpu_num_threads: u16,
    use_gpu: bool,
}

struct ModelData {
    variance_model: Vec<u8>,
    embedder_model: Vec<u8>,
    decoder_model: Vec<u8>,
    model_config: ModelConfig,
}

#[derive(Clone, Serialize, Deserialize, Getters)]
struct Meta {
    name: String,
    speaker_uuid: String,
    styles: Vec<Style>,
    version: String,
}

#[derive(Clone, Serialize, Deserialize, Getters)]
struct Style {
    name: String,
    id: u64,
}

static ENVIRONMENT: Lazy<Environment> = Lazy::new(|| {
    cfg_if! {
        if #[cfg(debug_assertions)]{
            const LOGGING_LEVEL: LoggingLevel = LoggingLevel::Verbose;
        } else{
            const LOGGING_LEVEL: LoggingLevel = LoggingLevel::Warning;
        }
    }
    Environment::builder()
        .with_name(env!("CARGO_PKG_NAME"))
        .with_log_level(LOGGING_LEVEL)
        .build()
        .unwrap()
});

#[derive(Getters, Debug, Serialize, Deserialize)]
pub struct SupportedDevices {
    cpu: bool,
    cuda: bool,
    dml: bool,
}

impl SupportedDevices {
    pub fn get_supported_devices() -> Result<Self> {
        let mut cuda_support = false;
        let mut dml_support = false;
        for provider in onnxruntime::session::get_available_providers()
            .map_err(|e| Error::GetSupportedDevices(e.into()))?
            .iter()
        {
            match provider.as_str() {
                "CUDAExecutionProvider" => cuda_support = true,
                "DmlExecutionProvider" => dml_support = true,
                _ => {}
            }
        }

        Ok(SupportedDevices {
            cpu: true,
            cuda: cuda_support,
            dml: dml_support,
        })
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("should not fail")
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub length_regulator: String,
    pub start_id: usize,
}

fn open_metas(root_dir_path: &Path, library_uuid: &str) -> Result<Vec<Meta>> {
    let metas_path = root_dir_path.join(library_uuid).join("metas.json");
    (|| {
        let metas = serde_json::from_str(&fs_err::read_to_string(metas_path)?)?;
        Ok(metas)
    })()
    .map_err(Error::LoadMetas)
}

fn open_model_files(root_dir_path: &Path, library_uuid: &str) -> Result<ModelData> {
    let path = |file_name| root_dir_path.join(library_uuid).join(file_name);

    let variance_model = open_model_file(&path("variance_model.onnx"))?;
    let embedder_model = open_model_file(&path("embedder_model.onnx"))?;
    let decoder_model = open_model_file(&path("decoder_model.onnx"))?;
    let model_config = {
        let path = path("model_config.json");
        (|| {
            let model_config = serde_json::from_str(&fs_err::read_to_string(&path)?)?;
            Ok(model_config)
        })()
        .map_err(move |cause| Error::LoadModelConfig { path, cause })?
    };

    return Ok(ModelData {
        variance_model,
        embedder_model,
        decoder_model,
        model_config,
    });

    fn open_model_file(path: &Path) -> Result<Vec<u8>> {
        fs_err::read(path)
            .map_err(Into::into)
            .map_err(Error::LoadModel)
    }
}

fn open_libraries(root_dir_path: &Path) -> Result<BTreeMap<String, bool>> {
    (|| {
        let path = root_dir_path.join("libraries.json");
        let libraries = serde_json::from_str(&fs_err::read_to_string(path)?)?;
        Ok(libraries)
    })()
    .map_err(Error::LoadLibraries)
}

#[allow(unsafe_code)]
unsafe impl Send for Status {}

impl Status {
    pub const METAS_STR: &'static str =
        include_str!(concat!(env!("CARGO_WORKSPACE_DIR"), "/model/metas.json"));

    pub fn new(root_dir_path: &Path, use_gpu: bool, cpu_num_threads: u16) -> Self {
        Self {
            root_dir_path: root_dir_path.to_path_buf(),
            light_session_options: SessionOptions::new(cpu_num_threads, false),
            heavy_session_options: SessionOptions::new(cpu_num_threads, use_gpu),
            libraries: None,
            usable_libraries: BTreeSet::new(),
            usable_model_data_map: BTreeMap::new(),
            usable_model_map: BTreeMap::new(),
            speaker_id_map: BTreeMap::new(),
            metas_str: String::new(),
        }
    }

    pub fn load(&mut self) -> Result<()> {
        self.libraries = Some(open_libraries(&self.root_dir_path)?);
        self.usable_libraries = self
            .libraries
            .iter()
            .flatten()
            .filter(|(_, &v)| v)
            .map(|(k, _)| k.to_owned())
            .collect();

        let mut all_metas: Vec<Meta> = Vec::new();
        for library_uuid in self.usable_libraries.iter() {
            let mode_data = open_model_files(&self.root_dir_path, library_uuid)?;
            let start_speaker_id = mode_data.model_config.start_id;

            let mut metas = open_metas(&self.root_dir_path, library_uuid)?;

            self.usable_model_data_map
                .insert(library_uuid.clone(), mode_data);

            for meta in metas.as_mut_slice() {
                let mut speaker_index: Option<usize> = None;
                for (count, all_meta) in all_metas.iter().enumerate() {
                    if meta.speaker_uuid == all_meta.speaker_uuid {
                        speaker_index = Some(count);
                    }
                }
                for style in meta.styles.as_mut_slice() {
                    let metas_style_id = start_speaker_id as u64 + style.id;
                    style.id = metas_style_id;
                    self.speaker_id_map
                        .insert(metas_style_id, library_uuid.clone());
                    if let Some(speaker_index) = speaker_index {
                        all_metas[speaker_index].styles.push(style.clone());
                    }
                }

                if speaker_index.is_none() {
                    all_metas.push(meta.clone());
                }
            }
        }
        self.metas_str = serde_json::to_string(&all_metas).unwrap();
        Ok(())
    }

    pub fn load_model(&mut self, library_uuid: &str) -> Result<()> {
        let model_data = self
            .usable_model_data_map
            .remove(library_uuid)
            .ok_or_else(|| Error::InvalidLibraryUuid {
                library_uuid: library_uuid.to_owned(),
            })?;
        let variance_session = self
            .new_session(&model_data.variance_model, &self.light_session_options)
            .map_err(Error::LoadModel)?;
        let embedder_session = self
            .new_session(&model_data.embedder_model, &self.light_session_options)
            .map_err(Error::LoadModel)?;
        let decoder_session = self
            .new_session(&model_data.decoder_model, &self.heavy_session_options)
            .map_err(Error::LoadModel)?;

        self.usable_model_map.insert(
            library_uuid.to_string(),
            Models {
                variance_session,
                embedder_session,
                decoder_session,
                model_config: model_data.model_config,
            },
        );
        Ok(())
    }

    // pub fn is_model_loaded(&self, model_index: usize) -> bool {
    //     self.models.predict_intonation.contains_key(&model_index)
    //         && self.models.predict_duration.contains_key(&model_index)
    //         && self.models.decode.contains_key(&model_index)
    // }

    fn new_session<B: AsRef<[u8]>>(
        &self,
        model_bytes: B,
        session_options: &SessionOptions,
    ) -> anyhow::Result<Session<'static>> {
        let session_builder = ENVIRONMENT
            .new_session_builder()?
            .with_optimization_level(GraphOptimizationLevel::Basic)?
            .with_intra_op_num_threads(*session_options.cpu_num_threads() as i32)?
            .with_inter_op_num_threads(*session_options.cpu_num_threads() as i32)?;

        let session_builder = if *session_options.use_gpu() {
            cfg_if! {
                if #[cfg(feature = "directml")]{
                    session_builder
                        .with_disable_mem_pattern()?
                        .with_execution_mode(onnxruntime::ExecutionMode::ORT_SEQUENTIAL)?
                        .with_append_execution_provider_directml(0)?
                } else {
                    let options = CudaProviderOptions::default();
                    session_builder.with_append_execution_provider_cuda(options)?
                }
            }
        } else {
            session_builder
        };

        Ok(session_builder.with_model_from_memory(model_bytes)?)
    }

    pub fn variance_session_run(
        &mut self,
        library_uuid: &str,
        inputs: Vec<&mut dyn AnyArray>,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        if let Some(models) = self.usable_model_map.get_mut(library_uuid) {
            let model = &mut models.variance_session;
            if let Ok(output_tensors) = model.run(inputs) {
                // NOTE: 暗黙的に２つのTensorが返ることを想定している
                //       返ってくるTensorの数が不正である時にエラーを報告することを検討しても良さそう
                Ok((
                    output_tensors[0].as_slice().unwrap().to_owned(),
                    output_tensors[1].as_slice().unwrap().to_owned(),
                ))
            } else {
                Err(Error::InferenceFailed)
            }
        } else {
            Err(Error::InvalidLibraryUuid {
                library_uuid: library_uuid.to_owned(),
            })
        }
    }

    pub fn embedder_session_run(
        &mut self,
        library_uuid: &str,
        inputs: Vec<&mut dyn AnyArray>,
    ) -> Result<Vec<f32>> {
        if let Some(models) = self.usable_model_map.get_mut(library_uuid) {
            let model = &mut models.embedder_session;
            if let Ok(output_tensors) = model.run(inputs) {
                Ok(output_tensors[0].as_slice().unwrap().to_owned())
            } else {
                Err(Error::InferenceFailed)
            }
        } else {
            Err(Error::InvalidLibraryUuid {
                library_uuid: library_uuid.to_owned(),
            })
        }
    }

    pub fn decoder_session_run(
        &mut self,
        library_uuid: &str,
        inputs: Vec<&mut dyn AnyArray>,
    ) -> Result<Vec<f32>> {
        if let Some(models) = self.usable_model_map.get_mut(library_uuid) {
            let model = &mut models.decoder_session;
            if let Ok(output_tensors) = model.run(inputs) {
                Ok(output_tensors[0].as_slice().unwrap().to_owned())
            } else {
                Err(Error::InferenceFailed)
            }
        } else {
            Err(Error::InvalidLibraryUuid {
                library_uuid: library_uuid.to_owned(),
            })
        }
    }

    pub fn get_library_uuid_from_speaker_id(&self, speaker_id: u32) -> Option<String> {
        self.speaker_id_map.get(&(speaker_id as u64)).cloned()
    }
}

// #[cfg(test)]
// mod tests {
//
//     use super::*;
//     use pretty_assertions::assert_eq;
//
//     #[rstest]
//     #[case(true, 0)]
//     #[case(true, 1)]
//     #[case(true, 8)]
//     #[case(false, 2)]
//     #[case(false, 4)]
//     #[case(false, 8)]
//     #[case(false, 0)]
//     fn status_new_works(#[case] use_gpu: bool, #[case] cpu_num_threads: u16) {
//         let status = Status::new(use_gpu, cpu_num_threads);
//         assert_eq!(false, status.light_session_options.use_gpu);
//         assert_eq!(use_gpu, status.heavy_session_options.use_gpu);
//         assert_eq!(
//             cpu_num_threads,
//             status.light_session_options.cpu_num_threads
//         );
//         assert_eq!(
//             cpu_num_threads,
//             status.heavy_session_options.cpu_num_threads
//         );
//         assert!(status.models.predict_duration.is_empty());
//         assert!(status.models.predict_intonation.is_empty());
//         assert!(status.models.decode.is_empty());
//         assert!(status.supported_styles.is_empty());
//     }
//
//     #[rstest]
//     fn supported_devices_get_supported_devices_works() {
//         let result = SupportedDevices::get_supported_devices();
//         // 環境によって結果が変わるので、関数呼び出しが成功するかどうかの確認のみ行う
//         assert!(result.is_ok(), "{:?}", result);
//     }
//
//     #[rstest]
//     fn status_load_model_works() {
//         let mut status = Status::new(false, 0);
//         let result = status.load_model(0);
//         assert_eq!(Ok(()), result);
//         assert_eq!(1, status.models.predict_duration.len());
//         assert_eq!(1, status.models.predict_intonation.len());
//         assert_eq!(1, status.models.decode.len());
//     }
//
//     #[rstest]
//     fn status_is_model_loaded_works() {
//         let mut status = Status::new(false, 0);
//         let model_index = 0;
//         assert!(
//             !status.is_model_loaded(model_index),
//             "model should  not be loaded"
//         );
//         let result = status.load_model(model_index);
//         assert_eq!(Ok(()), result);
//         assert!(
//             status.is_model_loaded(model_index),
//             "model should be loaded"
//         );
//     }
// }
