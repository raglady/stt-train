use std::{
    collections::{BTreeMap, HashMap},
    fmt::Debug,
    hash::Hash,
    sync::Arc,
};

use indexmap::IndexMap;
use ndarray::{ArcArray2, Array1, Array2};
use tokio::sync::Mutex;

use crate::types::Float;

pub trait BaumWelchTrait: Send + Sync {
    type Entry: Debug + Clone + Eq + Hash + Debug + Sync + Send + 'static;
    type Key: Debug + Clone + Eq + Hash + Debug + Sync + Send + 'static;
    type Data: Debug + Clone + Debug + Sync + Send + 'static;
    /// Return (log forward, log likelihood)
    /// The return values are
    fn log_forward(
        self: Arc<Self>,
        phones: &[Self::Key],
        mfcc: ArcArray2<Float>,
    ) -> impl std::future::Future<Output = (IndexMap<Self::Key, Array2<Float>>, Float)> + Send;

    /// Return Beta and log likelihood of all the hmm
    /// The return value are log
    fn log_backward(
        self: Arc<Self>,
        phones: &[Self::Key],
        mfcc: ArcArray2<Float>,
    ) -> impl std::future::Future<Output = (IndexMap<Self::Key, Array2<Float>>, Float)> + Send;

    // Acc = accumulé
    // Occupation = poids (axe 0 = state, axe 1 = mixture)
    // pounded_sum (axe 0 = state, axe 1 = mixture)
    // pounded_sum_square (axe 0 = state, axe 1 = mixture)
    fn e_step(
        self: Arc<Self>,
        phones: Arc<Vec<Self::Key>>,
        observation: ArcArray2<Float>,
        acc_log_occupation: Arc<Mutex<HashMap<Self::Key, Array2<Float>>>>,
        acc_pounded_sum: Arc<Mutex<HashMap<Self::Key, Array2<Array1<Float>>>>>,
        acc_pounded_sum_square: Arc<Mutex<HashMap<Self::Key, Array2<Array1<Float>>>>>,
        acc_log_epsilon_i_j: Arc<Mutex<HashMap<Self::Key, Array2<Float>>>>,
    ) -> impl std::future::Future<Output = Float> + Send;

    fn m_step(
        data: &mut HashMap<Self::Key, Self::Data>,
        phones: Arc<Vec<Self::Key>>,
        phone_acc_log_occupation: &HashMap<Self::Key, Array2<Float>>,
        phone_acc_pounded_sum: &HashMap<Self::Key, Array2<Array1<Float>>>,
        phone_acc_pounded_sum_square: &HashMap<Self::Key, Array2<Array1<Float>>>,
        phone_acc_log_epsilon_i_j: &HashMap<Self::Key, Array2<Float>>,
    ) -> impl std::future::Future<Output = ()> + Send;

    fn baum_welch(
        &mut self,
        phone_mfccs: Arc<BTreeMap<Self::Entry, Vec<ArcArray2<Float>>>>,
        n_iter: usize,
    ) -> impl std::future::Future<Output = BTreeMap<Self::Key, Float>> + Send;
}
