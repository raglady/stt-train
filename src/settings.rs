use serde::{Deserialize, Serialize};

use crate::types::Float;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Settings {
    pub monophone_training: MonophoneTrainSettings,
    pub predict: PredictSettings,
    pub storage: StorageSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonophoneTrainSettings {
    pub train_dir: String,
    pub iteration: usize,
    pub tolerance: Float,
    pub convergence: Float,
    pub component_per_state: usize,
    pub enable: bool,
}

impl Default for MonophoneTrainSettings {
    fn default() -> Self {
        Self {
            train_dir: "train".to_string(),
            iteration: 15,
            tolerance: 1e-4,
            convergence: 1e-4,
            component_per_state: 1,
            enable: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictSettings {
    pub predict_dir: String,
    pub beam_size: Float,
    pub real_time: bool,
    pub enable: bool,
}

impl Default for PredictSettings {
    fn default() -> Self {
        Self {
            predict_dir: "predict".to_string(),
            beam_size: 20.0,
            real_time: false,
            enable: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageSettings {
    pub monophone_modele_file: String,
    pub phonemes_file: String,
    pub log_prob_bigram_file: String,
}

impl Default for StorageSettings {
    fn default() -> Self {
        Self {
            monophone_modele_file: "monophone.json".to_string(),
            phonemes_file: "phonemes.json".to_string(),
            log_prob_bigram_file: "log-prob-bigram.json".to_string(),
        }
    }
}
