use std::{
    collections::{BTreeMap, HashMap},
    pin::Pin,
    sync::Arc,
};

use indexmap::IndexMap;
use log::info;
use ndarray::{ArcArray2, Array1, Array2, Array3, Axis, Zip, par_azip, s};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, ParallelBridge, ParallelIterator,
};
use tokio::{
    sync::{
        Mutex, RwLock,
        mpsc::{self, UnboundedSender},
        oneshot,
    },
    task::JoinSet,
};

use crate::{hmm_gmm::HMMGMM, log_sum_exp, traits::baum_welch_trait::BaumWelchTrait, types::Float};

use super::MonoPhone;

impl BaumWelchTrait for MonoPhone {
    type Key = String;
    type Entry = String;
    type Data = HMMGMM;

    /// Return (log forward, log likelihood)
    /// The return values are
    async fn log_forward(
        self: Arc<Self>,
        phones: &[String],
        mfcc: ArcArray2<Float>,
    ) -> (IndexMap<Self::Key, Array2<Float>>, Float) {
        let n_obs = mfcc.nrows();
        let mut phone_alpha = phones
            .iter()
            .map(|phone| {
                let hmm = self.map_phone_hmm.get(phone).unwrap();
                (
                    phone.to_string(),
                    Array2::<Float>::from_elem([hmm.get_n_states(), n_obs], Float::NEG_INFINITY),
                )
            })
            .collect::<IndexMap<String, Array2<Float>>>();

        let (map_index_phone, vec_state_logprob) =
            self.compute_log_prob_for_state_for_every_phone(mfcc.clone());

        if let Some(phone) = phones.first() {
            let hmm = self.map_phone_hmm.get(phone).unwrap();
            // Initialize
            for j in 0..hmm.get_n_states() {
                let val = hmm.get_log_initial()[j]
                    + hmm.get_states()[j].log_probability_density(mfcc.row(0));
                phone_alpha.get_mut(phone).unwrap()[[j, 0]] = val;
            }
        }

        mfcc.axis_iter(Axis(0)).enumerate().for_each(|(t, _)| {
            if t > 0 {
                phones.iter().enumerate().for_each(|(phone_index, phone)| {
                    let log_prob_index = map_index_phone.get(phone).cloned().unwrap();
                    let hmm = self.map_phone_hmm.get(phone).unwrap();
                    for j in 0..hmm.get_n_states() {
                        let vals: Vec<Float> = (0..hmm.get_n_states())
                            .map(|i| {
                                phone_alpha.get(phone).unwrap()[[i, t - 1]]
                                    + hmm.get_log_transition()[[i, j]]
                                    + vec_state_logprob[log_prob_index][[j, t]]
                            })
                            .collect();

                        phone_alpha.get_mut(phone).unwrap()[[j, t]] = log_sum_exp(
                            &([vals, [phone_alpha.get(phone).unwrap()[[j, t]]].to_vec()]
                                .iter()
                                .flatten()
                                .cloned()
                                .collect::<Vec<_>>()),
                        );

                        let next_phone_index = phone_index + 1;
                        if next_phone_index < phones.len() {
                            let next_phone = phones[next_phone_index].clone();
                            let log_prob_index = map_index_phone.get(&next_phone).cloned().unwrap();
                            let next_hmm = self.map_phone_hmm.get(&next_phone).unwrap();
                            for j in 0..next_hmm.get_n_states() {
                                let vals: Vec<Float> = (0..hmm.get_n_states())
                                    .map(|i| {
                                        phone_alpha.get(phone).unwrap()[[i, t - 1]]
                                            + hmm.get_log_final()[i]
                                            + next_hmm.get_log_initial()[j]
                                            + vec_state_logprob[log_prob_index][[j, t]]
                                    })
                                    .collect();
                                phone_alpha.get_mut(&next_phone).unwrap()[[j, t]] = log_sum_exp(
                                    &([
                                        vals,
                                        [phone_alpha.get(&next_phone).unwrap()[[j, t]]].to_vec(),
                                    ]
                                    .iter()
                                    .flatten()
                                    .cloned()
                                    .collect::<Vec<_>>()),
                                );
                            }
                        }
                        if phone == "foana" {
                            let next_phone = phones[phone_index].clone();
                            let log_prob_index = map_index_phone.get(&next_phone).cloned().unwrap();
                            let next_hmm = self.map_phone_hmm.get(&next_phone).unwrap();
                            for j in 0..next_hmm.get_n_states() {
                                let vals: Vec<Float> = (0..hmm.get_n_states())
                                    .map(|i| {
                                        phone_alpha.get(phone).unwrap()[[i, t - 1]]
                                            + hmm.get_log_final()[i]
                                            + next_hmm.get_log_initial()[j]
                                            + vec_state_logprob[log_prob_index][[j, t]]
                                    })
                                    .collect();
                                phone_alpha.get_mut(&next_phone).unwrap()[[j, t]] = log_sum_exp(
                                    &([
                                        vals,
                                        [phone_alpha.get(&next_phone).unwrap()[[j, t]]].to_vec(),
                                    ]
                                    .iter()
                                    .flatten()
                                    .cloned()
                                    .collect::<Vec<_>>()),
                                );
                            }
                        }
                    }
                });
            }
        });
        let last_phone = phone_alpha.last().unwrap().0.clone();

        let ll = phone_alpha
            .iter()
            .filter(|&(phone, _)| phone == &last_phone)
            .map(|(phone, alpha)| {
                alpha
                    .axis_iter(Axis(0))
                    .enumerate()
                    .map(|(state, row)| {
                        let hmm = self.map_phone_hmm.get(phone).unwrap();
                        row.last().unwrap() + hmm.get_log_final()[state]
                    })
                    .max_by(|a, b| a.total_cmp(b))
                    .unwrap()
            })
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();
        (phone_alpha.clone(), ll)
    }

    /// Return Beta and log likelihood of all the hmm
    /// The return value are log
    async fn log_backward(
        self: Arc<Self>,
        phones: &[String],
        mfcc: ArcArray2<Float>,
    ) -> (IndexMap<Self::Key, Array2<Float>>, Float) {
        let n_obs = mfcc.nrows();
        let mut phone_beta = phones
            .iter()
            .map(|phone| {
                let hmm = self.map_phone_hmm.get(phone).unwrap();
                (
                    phone.to_string(),
                    Array2::<Float>::from_elem([hmm.get_n_states(), n_obs], Float::NEG_INFINITY),
                )
            })
            .collect::<IndexMap<String, Array2<Float>>>();

        let (map_index_phone, vec_state_logprob) =
            self.compute_log_prob_for_state_for_every_phone(mfcc.clone());

        // Initialize
        if let Some(phone) = phones.last() {
            let hmm = self.map_phone_hmm.get(phone).unwrap();
            for j in 0..hmm.get_n_states() {
                phone_beta.get_mut(phone).unwrap()[[j, n_obs - 1]] = hmm.get_log_final()[j];
            }
        }

        mfcc.axis_iter(Axis(0))
            .enumerate()
            .rev()
            .for_each(|(t, _)| {
                if t < n_obs - 1 {
                    phones
                        .iter()
                        .enumerate()
                        .rev()
                        .for_each(|(phone_index, phone)| {
                            let log_prob_index = map_index_phone.get(phone).cloned().unwrap();
                            let hmm = self.map_phone_hmm.get(phone).unwrap();
                            for j in (0..hmm.get_n_states()).rev() {
                                let vals: Vec<Float> = (0..hmm.get_n_states())
                                    .rev()
                                    .map(|i| {
                                        hmm.get_log_transition()[[j, i]]
                                            + vec_state_logprob[log_prob_index][[i, t + 1]]
                                            + phone_beta.get(phone).unwrap()[[i, t + 1]]
                                    })
                                    .collect();

                                phone_beta.get_mut(phone).unwrap()[[j, t]] = log_sum_exp(
                                    &([vals, [phone_beta.get(phone).unwrap()[[j, t]]].to_vec()]
                                        .iter()
                                        .flatten()
                                        .cloned()
                                        .collect::<Vec<_>>()),
                                );

                                if phone_index > 1 {
                                    let next_phone_index = phone_index - 1;
                                    let next_phone = phones[next_phone_index].clone();
                                    let log_prob_index =
                                        map_index_phone.get(phone).cloned().unwrap();
                                    let next_hmm = self.map_phone_hmm.get(&next_phone).unwrap();
                                    for k in (0..next_hmm.get_n_states()).rev() {
                                        let vals: Vec<Float> = (0..hmm.get_n_states())
                                            .rev()
                                            .map(|i| {
                                                phone_beta.get(phone).unwrap()[[i, t + 1]]
                                                    + hmm.get_log_initial()[i]
                                                    + next_hmm.get_log_final()[k]
                                                    + vec_state_logprob[log_prob_index][[i, t + 1]]
                                            })
                                            .collect();
                                        phone_beta.get_mut(&next_phone).unwrap()[[k, t]] =
                                            log_sum_exp(
                                                &([
                                                    vals,
                                                    [phone_beta.get(&next_phone).unwrap()[[k, t]]]
                                                        .to_vec(),
                                                ]
                                                .iter()
                                                .flatten()
                                                .cloned()
                                                .collect::<Vec<_>>()),
                                            );
                                    }
                                }
                                if phone == "foana" {
                                    let next_phone_index = phone_index;
                                    let next_phone = phones[next_phone_index].clone();
                                    let log_prob_index =
                                        map_index_phone.get(phone).cloned().unwrap();
                                    let next_hmm = self.map_phone_hmm.get(&next_phone).unwrap();
                                    for k in (0..next_hmm.get_n_states()).rev() {
                                        let vals: Vec<Float> = (0..hmm.get_n_states())
                                            .rev()
                                            .map(|i| {
                                                phone_beta.get(phone).unwrap()[[i, t + 1]]
                                                    + hmm.get_log_initial()[i]
                                                    + next_hmm.get_log_final()[k]
                                                    + vec_state_logprob[log_prob_index][[i, t + 1]]
                                            })
                                            .collect();
                                        phone_beta.get_mut(&next_phone).unwrap()[[k, t]] =
                                            log_sum_exp(
                                                &([
                                                    vals,
                                                    [phone_beta.get(&next_phone).unwrap()[[k, t]]]
                                                        .to_vec(),
                                                ]
                                                .iter()
                                                .flatten()
                                                .cloned()
                                                .collect::<Vec<_>>()),
                                            );
                                    }
                                }
                            }
                        });
                }
            });

        let first_phone = phone_beta.first().unwrap().0.clone();

        let beta_0 = phone_beta
            .iter()
            .filter(|&(phone, _)| phone == &first_phone)
            .map(|(phone, beta)| {
                beta.axis_iter(Axis(0))
                    .enumerate()
                    .map(|(state, row)| {
                        let hmm = self.map_phone_hmm.get(phone).unwrap();
                        let log_prob_index = map_index_phone.get(phone).cloned().unwrap();
                        row.first().unwrap()
                            + hmm.get_log_initial()[state]
                            + vec_state_logprob[log_prob_index][[state, 0]]
                    })
                    .max_by(|a, b| a.total_cmp(b))
                    .unwrap()
            })
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();
        (phone_beta.clone(), beta_0)
    }

    async fn e_step(
        self: Arc<Self>,
        phones: Arc<Vec<String>>,
        observation: ArcArray2<Float>,
        acc_log_occupation: Arc<Mutex<HashMap<Self::Key, Array2<Float>>>>,
        acc_pounded_sum: Arc<Mutex<HashMap<Self::Key, Array2<Array1<Float>>>>>,
        acc_pounded_sum_square: Arc<Mutex<HashMap<Self::Key, Array2<Array1<Float>>>>>,
        acc_log_epsilon_i_j: Arc<Mutex<HashMap<Self::Key, Array2<Float>>>>,
    ) -> Float {
        let (forward_tx, forward_rx) = oneshot::channel();
        let observation_clone = observation.clone();
        let phones_clone = phones.to_vec();
        let this = self.clone();
        tokio::spawn(async move {
            let forward: (IndexMap<Self::Key, Array2<Float>>, Float) =
                this.log_forward(&phones_clone, observation_clone).await;
            forward_tx.send(forward).unwrap();
        });

        let (backward_tx, backward_rx) = oneshot::channel();

        let observation_clone = observation.clone();

        let phones_clone = phones.to_vec();

        let this = self.clone();
        tokio::spawn(async move {
            let backward: (IndexMap<Self::Key, Array2<Float>>, Float) =
                this.log_backward(&phones_clone, observation_clone).await;
            backward_tx.send(backward).unwrap();
        });

        let phone_log_epsilon_j_m_t = Arc::new(RwLock::new(
            phones
                .iter()
                .map(|phone| {
                    let hmm = self.map_phone_hmm.get(phone).unwrap();
                    (
                        phone.to_string(),
                        Array3::<Float>::from_elem(
                            [
                                hmm.get_n_states(),
                                hmm.get_states()[0].num_component(),
                                observation.clone().nrows(),
                            ],
                            Float::NEG_INFINITY,
                        ),
                    )
                })
                .collect::<HashMap<String, Array3<Float>>>(),
        ));

        let phone_log_epsilon_j_m_t_clone = phone_log_epsilon_j_m_t.clone();

        let (phone_log_alpha, log_likelihood) = forward_rx.await.unwrap();

        let (phone_log_beta, _log_beta_1_0) = backward_rx.await.unwrap();

        for phone in phones.iter() {
            let hmm = self.map_phone_hmm.get(phone).unwrap();
            // compute log_epsilon_j_m_t
            for (state, gmm) in hmm.get_states().iter().enumerate() {
                for component in 0..gmm.num_component() {
                    for (t, obs) in observation.axis_iter(Axis(0)).enumerate() {
                        if t > 1 {
                            let value = log_sum_exp(
                                &(0..hmm.get_n_states())
                                    .map(|prev_state| {
                                        phone_log_alpha.get(phone).unwrap()[[state, t - 1]]
                                            + hmm.get_log_transition()[[prev_state, state]]
                                            + gmm.get_component(component).1.ln()
                                            + gmm
                                                .get_component(component)
                                                .0
                                                .log_multivar_gauss_dist(obs)
                                            + phone_log_beta.get(phone).unwrap()[[state, t]]
                                    })
                                    .collect::<Vec<Float>>(),
                            );
                            let phone_log_epsilon_j_m_t_clone =
                                phone_log_epsilon_j_m_t_clone.clone();
                            let mut phone_log_epsilon_j_m_t_guard =
                                phone_log_epsilon_j_m_t_clone.write().await;
                            let log_epsilon_j_m_t =
                                phone_log_epsilon_j_m_t_guard.get_mut(phone).unwrap();
                            (*log_epsilon_j_m_t)[[state, component, t]] = (value
                                - self
                                    .log_alpha_t_n(
                                        &phones,
                                        &phone_log_alpha,
                                        &phone_log_beta,
                                        t - 1,
                                    )
                                    .get(phone)
                                    .unwrap())
                            .max(Float::NEG_INFINITY);
                        }
                    }
                }
            }
        }

        let mut phone_log_occupation = phones
            .iter()
            .map(|phone| {
                let hmm = self.map_phone_hmm.get(phone).unwrap();
                (
                    phone.to_string(),
                    Array2::<Float>::from_elem(
                        [hmm.get_n_states(), hmm.get_states()[0].num_component()],
                        Float::NEG_INFINITY,
                    ),
                )
            })
            .collect::<HashMap<String, Array2<Float>>>();

        let mut join_set = JoinSet::new();

        let acc_log_occupation_clone = acc_log_occupation.clone();
        let phone_log_epsilon_j_m_t_clone = phone_log_epsilon_j_m_t.clone();

        join_set.spawn(async move {
            let phone_log_epsilon_j_m_t_clone = phone_log_epsilon_j_m_t_clone.clone();
            let mut acc_log_occupation_guard = acc_log_occupation_clone.lock().await;
            for (phone, log_occupation) in phone_log_occupation.iter_mut() {
                for ((row, col), value) in log_occupation.indexed_iter_mut() {
                    let phone_log_epsilon_j_m_t_clone = phone_log_epsilon_j_m_t_clone.clone();
                    let phone_log_epsilon_j_m_t_guard = phone_log_epsilon_j_m_t_clone.read().await;
                    *value = log_sum_exp(
                        &phone_log_epsilon_j_m_t_guard
                            .get(phone)
                            .unwrap()
                            .slice(s![row, col, ..])
                            .to_vec(),
                    );
                    drop(phone_log_epsilon_j_m_t_guard);
                }
                let acc = acc_log_occupation_guard.get_mut(phone).unwrap();
                // let mut acc_occupation = acc_occupation_guard.clone();
                par_azip!((a in acc, b in log_occupation) *a = log_sum_exp(&[*a,*b]));
            }
            drop(acc_log_occupation_guard);
        });

        let mut phone_pounded_sum = phones
            .iter()
            .map(|phone| {
                let hmm = self.map_phone_hmm.get(phone).unwrap();
                (
                    phone.to_string(),
                    Array2::<Array1<Float>>::from_elem(
                        [hmm.get_n_states(), hmm.get_states()[0].num_component()],
                        Array1::<Float>::zeros(observation.clone().ncols()),
                    ),
                )
            })
            .collect::<HashMap<String, Array2<Array1<Float>>>>();

        let acc_pounded_sum_clone = acc_pounded_sum.clone();

        let phone_log_epsilon_j_m_t_clone = phone_log_epsilon_j_m_t.clone();
        let observation_clone = observation.clone();

        join_set.spawn(async move {
            let mut acc_pounded_sum_guard = acc_pounded_sum_clone.lock().await;

            // Probleme, obs(t) contient neg number, quit log domaine
            for (phone, pounded_sum) in phone_pounded_sum.iter_mut() {
                for ((row, col), value) in pounded_sum.indexed_iter_mut() {
                    let phone_log_epsilon_j_m_t_clone = phone_log_epsilon_j_m_t_clone.clone();
                    let phone_log_epsilon_j_m_t_guard = phone_log_epsilon_j_m_t_clone.read().await;

                    // On somme toutes les lignes pondérées pour obtenir un seul Array1
                    let log_epsilon_j_m_t = phone_log_epsilon_j_m_t_guard.get(phone).unwrap();

                    let sum_vector = log_epsilon_j_m_t
                        .slice(s![row, col, ..])
                        .iter()
                        .enumerate()
                        .map(|(t, &log_epsilon_j_m_t)| {
                            observation_clone.row(t).map(|v| {
                                let val = v * log_epsilon_j_m_t.exp();
                                if val.is_infinite() || val.is_nan() {
                                    println!("error: {:#?}", (v, log_epsilon_j_m_t.exp()));
                                    panic!();
                                }
                                val
                            })
                        })
                        .fold(Array1::zeros(observation_clone.ncols()), |acc, x| {
                            Zip::from(&acc).and(&x).map_collect(|&a, &k| a + k)
                        });

                    drop(phone_log_epsilon_j_m_t_guard);
                    *value = sum_vector; // 'value' doit être de type Array1<Float>
                }
                let acc = acc_pounded_sum_guard.get_mut(phone).unwrap();

                par_azip!((a in acc, b in pounded_sum) Zip::from(a).and(b).for_each(|x,y|{
                    *x += *y
                }));
            }

            drop(acc_pounded_sum_guard);
        });

        let mut phone_pounded_sum_square = phones
            .iter()
            .map(|phone| {
                let hmm = self.map_phone_hmm.get(phone).unwrap();
                (
                    phone.to_string(),
                    Array2::<Array1<Float>>::from_elem(
                        [hmm.get_n_states(), hmm.get_states()[0].num_component()],
                        Array1::<Float>::zeros(observation.clone().ncols()),
                    ),
                )
            })
            .collect::<HashMap<String, Array2<Array1<Float>>>>();

        let acc_pounded_sum_square_clone = acc_pounded_sum_square.clone();

        let phone_log_epsilon_j_m_t_clone = phone_log_epsilon_j_m_t.clone();

        let observation_clone = observation.clone();
        join_set.spawn(async move {
            let mut acc_pounded_sum_square_guard = acc_pounded_sum_square_clone.lock().await;

            for (phone, pounded_sum_square) in phone_pounded_sum_square.iter_mut() {
                for ((row, col), value) in pounded_sum_square.indexed_iter_mut() {
                    let phone_log_epsilon_j_m_t_clone = phone_log_epsilon_j_m_t_clone.clone();
                    let phone_log_epsilon_j_m_t_guard = phone_log_epsilon_j_m_t_clone.read().await;
                    // On somme toutes les lignes pondérées pour obtenir un seul Array1
                    let sum_vector = phone_log_epsilon_j_m_t_guard
                        .get(phone)
                        .unwrap()
                        .slice(s![row, col, ..])
                        .iter()
                        .enumerate()
                        .map(|(t, &log_epsilon_j_m_t)| {
                            observation_clone
                                .row(t)
                                .map(|v| v * v * log_epsilon_j_m_t.exp())
                        })
                        .fold(Array1::zeros(observation_clone.ncols()), |acc, x| {
                            Zip::from(&acc).and(&x).map_collect(|&a, &k| a + k)
                        });

                    drop(phone_log_epsilon_j_m_t_guard);
                    *value = sum_vector; // 'value' doit être de type Array1<f32>
                }
                let acc = acc_pounded_sum_square_guard.get_mut(phone).unwrap();
                par_azip!((a in acc, b in pounded_sum_square) Zip::from(a).and(b).for_each(|x,y|{
                    *x +=  *y
                }));
            }

            drop(acc_pounded_sum_square_guard);
        });

        // Join all tasks as they finish
        while let Some(res) = join_set.join_next().await {
            res.unwrap();
            //println!("Task finished: {:?}", res.unwrap());
        }

        let mut phone_log_epsilon_i_j_t = phones
            .iter()
            .map(|phone| {
                let hmm = self.map_phone_hmm.get(phone).unwrap();
                (
                    phone.to_string(),
                    Array3::<Float>::from_elem(
                        [hmm.get_n_states(), hmm.get_n_states(), observation.nrows()],
                        Float::NEG_INFINITY,
                    ),
                )
            })
            .collect::<HashMap<String, Array3<Float>>>();

        let phone_log_emission = phones
            .iter()
            .map(|phone| {
                let hmm = self.map_phone_hmm.get(phone).unwrap();
                (
                    phone.to_string(),
                    (0..hmm.get_n_states())
                        .map(|state| hmm.compute_log_emissions(state, observation.clone()))
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<HashMap<String, Vec<_>>>();

        phone_log_epsilon_i_j_t
            .iter_mut()
            .for_each(|(phone, log_epsilon_i_j_t)| {
                let hmm = self.map_phone_hmm.get(phone).unwrap();
                log_epsilon_i_j_t
                    .axis_iter_mut(Axis(0))
                    .enumerate()
                    .for_each(|(i, mut axis_i)| {
                        axis_i
                            .axis_iter_mut(Axis(0))
                            .enumerate()
                            .for_each(|(j, mut axis_j)| {
                                axis_j.iter_mut().enumerate().for_each(|(t, val)| {
                                    if t < observation.nrows() - 1 {
                                        *val = (phone_log_alpha.get(phone).unwrap()[[i, t]]
                                            + hmm.get_log_transition()[[i, j]]
                                            + phone_log_emission.get(phone).unwrap()[j][[t + 1]]
                                            + phone_log_beta.get(phone).unwrap()[[j, t + 1]]
                                            - self
                                                .log_alpha_t_n(
                                                    &phones,
                                                    &phone_log_alpha,
                                                    &phone_log_beta,
                                                    t,
                                                )
                                                .get(phone)
                                                .unwrap())
                                        .max(Float::NEG_INFINITY)
                                    }
                                });
                            });
                    });
            });

        let mut phone_log_epsilon_i_j = phones
            .iter()
            .map(|phone| {
                let hmm = self.map_phone_hmm.get(phone).unwrap();
                (
                    phone.to_string(),
                    Array2::<Float>::from_elem(
                        [hmm.get_n_states(), hmm.get_n_states()],
                        Float::NEG_INFINITY,
                    ),
                )
            })
            .collect::<HashMap<String, Array2<Float>>>();

        let mut acc_log_epsilon_i_j_guard = acc_log_epsilon_i_j.lock().await;
        phone_log_epsilon_i_j
            .iter_mut()
            .for_each(|(phone, log_epsilon_i_j)| {
                log_epsilon_i_j
                    .indexed_iter_mut()
                    .par_bridge()
                    .for_each(|((i, j), val)| {
                        *val = log_sum_exp(
                            &phone_log_epsilon_i_j_t
                                .get(phone)
                                .unwrap()
                                .slice(s![i, j, ..])
                                .to_vec(),
                        );
                    });
                let acc = acc_log_epsilon_i_j_guard.get_mut(phone).unwrap();
                par_azip!((a in acc, b in log_epsilon_i_j) *a = log_sum_exp(&[*a,*b]));
            });

        drop(acc_log_epsilon_i_j_guard);

        log_likelihood
    }

    async fn m_step(
        data: &mut HashMap<Self::Key, Self::Data>,
        phones: Arc<Vec<String>>,
        phone_acc_log_occupation: &HashMap<Self::Key, Array2<Float>>,
        phone_acc_pounded_sum: &HashMap<Self::Key, Array2<Array1<Float>>>,
        phone_acc_pounded_sum_square: &HashMap<Self::Key, Array2<Array1<Float>>>,
        phone_acc_log_epsilon_i_j: &HashMap<Self::Key, Array2<Float>>,
    ) {
        for phone in phones.iter() {
            let mut join_set = JoinSet::new();
            let (mean_tx, mean_rx) = oneshot::channel();

            let arc_phone_acc_log_occupation = Arc::new(phone_acc_log_occupation.clone());
            let arc_phone_acc_log_occupation_clone = arc_phone_acc_log_occupation.clone();

            let arc_phone_acc_pounded_sum = Arc::new(phone_acc_pounded_sum.clone());
            let arc_phone_acc_pounded_sum_clone = arc_phone_acc_pounded_sum.clone();

            let arc_phone = Arc::new(phone.clone());
            let arc_phone_clone = arc_phone.clone();

            join_set.spawn_blocking(move || {
                mean_tx
                    .send(
                        Zip::from(
                            arc_phone_acc_pounded_sum_clone
                                .get(arc_phone_clone.as_str())
                                .unwrap(),
                        )
                        .and(
                            arc_phone_acc_log_occupation_clone
                                .get(arc_phone_clone.as_str())
                                .unwrap(),
                        )
                        .par_map_collect(|pounded_sum, &log_occupation| {
                            pounded_sum.map(|v| {
                                let m = v / log_occupation.exp();
                                if m.is_nan() || m.is_infinite() {
                                    // println!(
                                    //     "n {:?} d {:?}",
                                    //     pounded_sum,
                                    //     (log_occupation, log_occupation.exp())
                                    // );
                                    panic!("mean is nan");
                                }
                                m
                            })
                        }),
                    )
                    .unwrap();
            });

            let arc_phone_acc_log_occupation_clone = arc_phone_acc_log_occupation.clone();

            let arc_phone_acc_pounded_sum_square = Arc::new(phone_acc_pounded_sum_square.clone());
            let arc_phone_acc_pounded_sum_square_clone = arc_phone_acc_pounded_sum_square.clone();

            let arc_phone_clone = arc_phone.clone();

            let (covar_tx, covar_rx) = oneshot::channel();

            join_set.spawn_blocking(move || {
                covar_tx
                    .send(
                        Zip::from(
                            arc_phone_acc_pounded_sum_square_clone
                                .get(arc_phone_clone.as_str())
                                .unwrap(),
                        )
                        .and(
                            arc_phone_acc_log_occupation_clone
                                .get(arc_phone_clone.as_str())
                                .unwrap(),
                        )
                        .par_map_collect(
                            |pounded_sum_square, &log_occupation| {
                                pounded_sum_square.map(|v| v - log_occupation.exp())
                            },
                        ),
                    )
                    .unwrap();
            });

            let arc_phone_acc_log_occupation_clone = arc_phone_acc_log_occupation.clone();

            let arc_phone_clone = arc_phone.clone();

            let (weight_tx, weight_rx) = oneshot::channel();

            join_set.spawn_blocking(move || {
                let log_sum_occupation: Array1<Float> = arc_phone_acc_log_occupation_clone
                    .get(arc_phone_clone.as_str())
                    .unwrap()
                    .axis_iter(Axis(0))
                    .map(|state_occ| log_sum_exp(&state_occ.to_vec()))
                    .collect();
                weight_tx
                    .send(
                        Array2::from_shape_vec(
                            arc_phone_acc_log_occupation_clone
                                .get(arc_phone_clone.as_str())
                                .unwrap()
                                .raw_dim(),
                            arc_phone_acc_log_occupation_clone
                                .get(arc_phone_clone.as_str())
                                .unwrap()
                                .indexed_iter()
                                .map(|((state, _mixture), val)| {
                                    (val - log_sum_occupation[[state]]).exp()
                                })
                                .collect::<Vec<_>>(),
                        )
                        .unwrap(),
                    )
                    .unwrap();
            });

            let mean = mean_rx.await.unwrap();
            let covar = covar_rx.await.unwrap();
            let weight = weight_rx.await.unwrap();

            while let Some(ret) = join_set.join_next().await {
                ret.unwrap();
            }

            let hmm = data.get_mut(phone).unwrap();
            // let mut hmm_clone = hmm.clone();
            let variance_floor = hmm.get_variance_floor().to_owned();
            hmm.get_states_mut()
                .iter_mut()
                .enumerate()
                .for_each(|(state, gmm)| {
                    for component in 0..gmm.num_component() {
                        let (mut gaussian, _w) = gmm.get_component(component).clone();

                        gaussian.set_mean(mean[[state, component]].view());

                        gaussian.set_covar(
                            Zip::from(&covar[[state, component]])
                                .and(&variance_floor)
                                .par_map_collect(|v, v_f| v.max(*v_f))
                                .view(),
                        );

                        let w = weight[[state, component]];

                        gmm.set_component(component, &gaussian, w);
                    }
                });
            // Update transition matrix
            let vec_denominator: Vec<_> = phone_acc_log_epsilon_i_j
                .get(phone)
                .unwrap()
                .axis_iter(Axis(0))
                .map(|row| log_sum_exp(&row.to_vec()))
                .collect();

            Zip::indexed(hmm.get_log_transition_mut()).par_for_each(|(i, j), val| {
                *val = phone_acc_log_epsilon_i_j.get(phone).unwrap()[[i, j]] - vec_denominator[i]
            });
        }
    }

    /// observations is mfcc_features
    /// return true if converged
    fn baum_welch(
        &mut self,
        phone_mfccs: Arc<BTreeMap<String, Vec<ArcArray2<Float>>>>,
        n_iter: usize,
    ) -> impl Future<Output = BTreeMap<String, Float>> + Send {
        async move {
            let last_ll = Arc::new(RwLock::new(
                phone_mfccs
                    .keys()
                    .map(|phone| (phone.to_string(), Float::NEG_INFINITY))
                    .collect::<HashMap<String, Float>>(),
            ));

            let phone_converged = Arc::new(RwLock::new(
                phone_mfccs
                    .keys()
                    .map(|phone| (phone.to_string(), false))
                    .collect::<HashMap<String, bool>>(),
            ));

            let ret = Arc::new(RwLock::new(BTreeMap::new()));

            let train = |this: Self,
                         phone_mfccs: BTreeMap<String, Vec<ArcArray2<Float>>>,
                         tx: UnboundedSender<
                Pin<Box<dyn Future<Output = Option<(String, HMMGMM)>> + Send>>,
            >,
                         iter: usize| {
                this.get_phone_hmm_gmm()
                    .par_iter()
                    .for_each(|(phoneme, _hmm)| {
                        if let Some(observations) = phone_mfccs.get(phoneme) {
                            let last_ll = last_ll.clone();
                            let phone_converged = phone_converged.clone();
                            let ret = ret.clone();
                            let this_clone = this.clone();
                            let phoneme = phoneme.clone();
                            let observations = observations.clone();
                            tx.send(Box::pin(async move {
                                let mut msg = None;
                                info!("Training {} ...", phoneme);
                                if observations.is_empty() {
                                    panic!("baum welch: observations vide !");
                                }

                                let phones = if phoneme != "foana" {
                                    vec![
                                        "foana".to_string(),
                                        phoneme.to_string(),
                                        "foana".to_string(),
                                    ]
                                } else {
                                    vec![phoneme.to_string()]
                                };

                                let mut current_ll = 0.0;

                                let acc_log_occupation = Arc::new(Mutex::new(
                                    phones
                                        .iter()
                                        .map(|phone| {
                                            let hmm = this_clone.map_phone_hmm.get(phone).unwrap();
                                            (
                                                phone.to_string(),
                                                Array2::<Float>::zeros([
                                                    hmm.get_n_states(),
                                                    hmm.get_states()[0].num_component(),
                                                ]),
                                            )
                                        })
                                        .collect::<HashMap<String, Array2<Float>>>(),
                                ));

                                let acc_pounded_sum = Arc::new(Mutex::new(
                                    phones
                                        .iter()
                                        .map(|phone| {
                                            let hmm = this_clone.map_phone_hmm.get(phone).unwrap();
                                            (
                                                phone.to_string(),
                                                Array2::<Array1<Float>>::from_elem(
                                                    [
                                                        hmm.get_n_states(),
                                                        hmm.get_states()[0].num_component(),
                                                    ],
                                                    Array1::zeros(observations[0].ncols()),
                                                ),
                                            )
                                        })
                                        .collect::<HashMap<String, Array2<Array1<Float>>>>(),
                                ));

                                let acc_pounded_sum_square = Arc::new(Mutex::new(
                                    phones
                                        .iter()
                                        .map(|phone| {
                                            let hmm = this_clone.map_phone_hmm.get(phone).unwrap();
                                            (
                                                phone.to_string(),
                                                Array2::<Array1<Float>>::from_elem(
                                                    [
                                                        hmm.get_n_states(),
                                                        hmm.get_states()[0].num_component(),
                                                    ],
                                                    Array1::zeros(observations[0].ncols()),
                                                ),
                                            )
                                        })
                                        .collect::<HashMap<String, Array2<Array1<Float>>>>(),
                                ));

                                let acc_log_epsilon_i_j = Arc::new(Mutex::new(
                                    phones
                                        .iter()
                                        .map(|phone| {
                                            let hmm = this_clone.map_phone_hmm.get(phone).unwrap();
                                            (
                                                phone.to_string(),
                                                Array2::<Float>::from_elem(
                                                    [hmm.get_n_states(), hmm.get_n_states()],
                                                    Float::NEG_INFINITY,
                                                ),
                                            )
                                        })
                                        .collect::<HashMap<String, Array2<Float>>>(),
                                ));

                                let acc_log_occupation_clone = acc_log_occupation.clone();
                                let acc_pounded_sum_clone = acc_pounded_sum.clone();
                                let acc_pounded_sum_square_clone = acc_pounded_sum_square.clone();
                                let acc_log_epsilon_i_j_clone = acc_log_epsilon_i_j.clone();
                                let arc_self = Arc::new(this_clone.clone());

                                let phone_converged_guard = phone_converged.read().await;
                                let converged = *phone_converged_guard.get(&phoneme).unwrap();
                                drop(phone_converged_guard);

                                if !converged {
                                    let mut join_set = JoinSet::new();

                                    let phones_clone = Arc::new(phones.clone());

                                    let mut vec_task = observations
                                        .par_iter()
                                        .with_min_len(rayon::current_num_threads())
                                        .map(|obs| {
                                            let arc_self = arc_self.clone();
                                            arc_self.e_step(
                                                phones_clone.clone(),
                                                obs.to_shared(),
                                                acc_log_occupation_clone.clone(),
                                                acc_pounded_sum_clone.clone(),
                                                acc_pounded_sum_square_clone.clone(),
                                                acc_log_epsilon_i_j_clone.clone(),
                                            )
                                        })
                                        .fold(Vec::new, |mut vec, task| {
                                            vec.push(Some(task));
                                            vec
                                        })
                                        .reduce(Vec::new, |mut acc, js| {
                                            acc.extend(js);
                                            acc
                                        });

                                    for task in vec_task.iter_mut() {
                                        join_set.spawn(task.take().unwrap());
                                    }

                                    while let Some(ll) = join_set.join_next().await {
                                        current_ll += ll.unwrap();
                                        //println!("Task finished: {:?}", res.unwrap());
                                    }

                                    let mut last_ll_guard = last_ll.write().await;
                                    let mut ret_guard = ret.write().await;

                                    current_ll /= observations.len() as Float;
                                    let diff_iter = current_ll.abs()
                                        - last_ll_guard.get(&phoneme).unwrap().abs();

                                    if diff_iter > 0.0 && diff_iter < this.convergence {
                                        info!("phoneme {} converged at iter {}", phoneme, iter);

                                        let mut phone_converged_guard =
                                            phone_converged.write().await;

                                        let converged =
                                            phone_converged_guard.get_mut(&phoneme).unwrap();
                                        *converged = true;
                                        drop(phone_converged_guard);
                                    }

                                    last_ll_guard.insert(phoneme.to_string(), current_ll);
                                    ret_guard.insert(phoneme.to_string(), current_ll);
                                    drop(last_ll_guard);
                                    drop(ret_guard);
                                }

                                let phone_converged_guard = phone_converged.read().await;
                                let converged = *phone_converged_guard.get(&phoneme).unwrap();
                                drop(phone_converged_guard);

                                if !converged {
                                    let acc_log_occupation_clone = acc_log_occupation.clone();
                                    let acc_pounded_sum_clone = acc_pounded_sum.clone();
                                    let acc_pounded_sum_square_clone =
                                        acc_pounded_sum_square.clone();
                                    let acc_log_epsilon_i_j_clone = acc_log_epsilon_i_j.clone();

                                    let acc_log_occupation_guard =
                                        acc_log_occupation_clone.lock().await;
                                    let acc_pounded_sum_guard = acc_pounded_sum_clone.lock().await;
                                    let acc_pounded_sum_square_guard =
                                        acc_pounded_sum_square_clone.lock().await;
                                    let acc_log_epsilon_i_j_guard =
                                        acc_log_epsilon_i_j_clone.lock().await;

                                    let mut hash_map = this_clone
                                        .map_phone_hmm
                                        .iter()
                                        .filter(|(k, _v)| k == &&phoneme)
                                        .map(|(k, v)| (k.clone(), v.clone()))
                                        .collect::<HashMap<String, HMMGMM>>();

                                    Self::m_step(
                                        &mut hash_map,
                                        Arc::new(vec![phoneme.clone()]),
                                        &acc_log_occupation_guard,
                                        &acc_pounded_sum_guard,
                                        &acc_pounded_sum_square_guard,
                                        &acc_log_epsilon_i_j_guard,
                                    )
                                    .await;

                                    drop(acc_log_occupation_guard);
                                    drop(acc_pounded_sum_guard);
                                    drop(acc_pounded_sum_square_guard);
                                    drop(acc_log_epsilon_i_j_guard);
                                    msg =
                                        Some((phoneme.clone(), hash_map.remove(&phoneme).unwrap()));
                                }
                                msg
                            }))
                            .unwrap();
                        }
                    });
            };

            for iter in 0..n_iter {
                info!("Monophone Baum Welch iteration {}", iter);
                let (tx, mut rx) = mpsc::unbounded_channel();
                train(
                    self.clone(),
                    phone_mfccs
                        .par_iter()
                        .filter_map(|(phone, obs)| {
                            if phone.eq(&"foana") {
                                Some((phone.clone(), obs.clone()))
                            } else {
                                None
                            }
                        })
                        .collect::<BTreeMap<_, _>>(),
                    tx.clone(),
                    iter,
                );
                drop(tx);
                let mut join_set = JoinSet::new();
                while let Some(task) = rx.recv().await {
                    join_set.spawn(task);
                }
                while let Some(res) = join_set.join_next().await {
                    if let Some((phoneme, hmm)) = res.unwrap() {
                        self.map_phone_hmm.insert(phoneme, hmm);
                    }
                }
                let (tx, mut rx) = mpsc::unbounded_channel();
                train(
                    self.clone(),
                    phone_mfccs
                        .par_iter()
                        .filter_map(|(phone, obs)| {
                            if phone.ne(&"foana") {
                                Some((phone.clone(), obs.clone()))
                            } else {
                                None
                            }
                        })
                        .collect::<BTreeMap<_, _>>(),
                    tx.clone(),
                    iter,
                );
                drop(tx);
                let mut join_set = JoinSet::new();
                while let Some(task) = rx.recv().await {
                    join_set.spawn(task);
                }
                while let Some(res) = join_set.join_next().await {
                    if let Some((phoneme, hmm)) = res.unwrap() {
                        self.map_phone_hmm.insert(phoneme.clone(), hmm);
                        info!("Phone {} updated!", phoneme);
                    }
                }
            }

            let guard = ret.read().await;
            let r = guard.clone();
            drop(guard);
            r
        }
    }
}
