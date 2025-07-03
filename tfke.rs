// ---------------------------------------------
// Fully Homomorphic Encryption – Secure ML Inference
// Prototype project in Rust using the `tfhe` crate
// ---------------------------------------------
//! # fhe-ml-secure-inference
//! A minimal, production‑ready skeleton for running logistic‑regression‑style
//! models on encrypted data with Fully Homomorphic Encryption (FHE).
//! Uses the pure‑Rust `tfhe` library from Zama.
//!
//! ## Quick Start (CLI)
//! ```bash
//! cargo run --release -- keygen --out ./keys
//! cargo run --release -- encrypt --input data.csv --keys ./keys
//! cargo run --release -- infer   --cipher ./enc.bin --model ./model.npz --keys ./keys
//! cargo run --release -- decrypt --cipher ./preds.bin --keys ./keys
//! ```
//!
//! ## Cargo.toml fragment
//! ```toml
//! [package]
//! name    = "fhe_ml_secure_inference"
//! version = "0.1.0"
//! edition = "2021"
//!
//! [dependencies]
//! tfhe   = "0.5"
//! clap   = { version = "4", features = ["derive"] }
//! serde  = { version = "1", features = ["derive"] }
//! bincode = "1"
//! anyhow = "1"
//! ndarray = "0.15"
//! rayon  = "1"
//! tokio  = { version = "1", features = ["rt‑multi‑thread", "macros"] }
//! ```
//! (full Cargo.toml with features & build profiles is expected in project root)
//!
//! ## Architecture
//! 1. **Key generation** ‑ creates client & server keysets.
//! 2. **Encryption** of user feature vectors (CSV → ciphertexts).
//! 3. **Inference** – homomorphic dot‑product + polynomial sigmoid.
//! 4. **Decryption** of predictions.
//! 5. **gRPC server** (optional) for online scoring.
//!
//! For simplicity all modules live in this single file; real‑world code
//! should be split into their own files under `src/`.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};
use tfhe::integer::gen_keys_radix;
use tfhe::integer::prelude::*;

// ---------------- CLI definition ----------------
#[derive(Parser)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate client & server keys and save to a folder
    Keygen {
        #[arg(long, default_value = "./keys")] out: PathBuf,
        /// log2 of ciphertext modulus – security / performance trade‑off
        #[arg(long, default_value_t = 7)] radix_log: usize,
    },
    /// Encrypt CSV file (numeric features) to binary ciphertext file
    Encrypt {
        #[arg(long)] input: PathBuf,
        #[arg(long)] keys: PathBuf,
        #[arg(long, default_value = "encrypted.bin")] out: PathBuf,
    },
    /// Homomorphic inference (logistic regression)
    Infer {
        #[arg(long)] cipher: PathBuf,
        #[arg(long)] model: PathBuf,
        #[arg(long)] keys: PathBuf,
        #[arg(long, default_value = "preds.bin")] out: PathBuf,
    },
    /// Decrypt ciphertext predictions to plaintext CSV
    Decrypt {
        #[arg(long)] cipher: PathBuf,
        #[arg(long)] keys: PathBuf,
        #[arg(long, default_value = "preds.csv")] out: PathBuf,
    },
    /// Run async gRPC server that exposes `/predict` endpoint
    Serve {
        #[arg(long)] keys: PathBuf,
        #[arg(long, default_value = "0.0.0.0:50051")] addr: String,
    },
}

// ---------------- Key handling ----------------
#[derive(Serialize, Deserialize)]
struct Keyset {
    client_key: ClientKey,
    server_key: ServerKey,
}

fn save_keyset(keys: &Keyset, dir: &PathBuf) -> Result<()> {
    fs::create_dir_all(dir)?;
    fs::write(dir.join("client.bincode"), bincode::serialize(&keys.client_key)?)?;
    fs::write(dir.join("server.bincode"), bincode::serialize(&keys.server_key)?)?;
    Ok(())
}

fn load_keyset(dir: &PathBuf) -> Result<Keyset> {
    let client_bytes = fs::read(dir.join("client.bincode"))?;
    let server_bytes = fs::read(dir.join("server.bincode"))?;
    Ok(Keyset {
        client_key: bincode::deserialize(&client_bytes)?\,
        server_key: bincode::deserialize(&server_bytes)?\,
    })
}

// ---------------- Encryption helpers ----------------
type CipherVec = Vec<CiphertextRadix>;  // radix‑encoded ciphertexts

type ClientKey = tfhe::integer::ClientKey;
type ServerKey = tfhe::integer::ServerKey;
type CiphertextRadix = tfhe::integer::RadixCiphertext;

fn encrypt_record(rec: &Array1<u64>, ck: &ClientKey) -> CipherVec {
    rec.iter()
        .map(|&v| ck.encrypt_radix(&u64::to_le_bytes(v), 64))
        .collect()
}

fn decrypt_record(cipher: &CipherVec, ck: &ClientKey) -> Array1<u64> {
    cipher
        .iter()
        .map(|ct| u64::from_le_bytes(ck.decrypt_radix(ct)))
        .collect()
}

// -------------- Homomorphic logistic regression --------------
/// Compute dot(w, x) on encrypted vector x and plaintext weights w.
fn he_dot_product(enc_x: &CipherVec, weights: &Array1<f64>, sk: &ServerKey) -> CiphertextRadix {
    assert_eq!(enc_x.len(), weights.len());
    // Encode weights as plaintext scalars and perform HE mul & add.
    let mut acc = sk.create_trivial_radix(&[0u8; 8]);
    for (ct, &w) in enc_x.iter().zip(weights) {
        let scaled = (w * 1_000.0) as u64; // fixed‑point scaling
        let tmp = sk.scalar_mul(ct, scaled);
        acc = sk.add(&acc, &tmp);
    }
    acc
}

/// Polynomial approximation of sigmoid using degree‑3 polynomial.
fn he_sigmoid(ct: &CiphertextRadix, sk: &ServerKey) -> CiphertextRadix {
    // Sigmoid(x) ≈ 0.5 + 0.197*x – 0.004*x^3  (for |x| < 4)
    let c0 = sk.create_trivial_radix(&((0.5 * 1_000.0) as u64).to_le_bytes());
    let c1 = sk.scalar_mul(ct, (0.197 * 1_000.0) as u64);
    let x3 = sk.mul(ct, &sk.mul(ct, ct));
    let c3 = sk.scalar_mul(&x3, (0.004 * 1_000.0) as u64);
    let tmp = sk.add(&c0, &c1);
    sk.sub(&tmp, &c3)
}

// -------------- CSV helpers --------------
fn load_csv(path: &PathBuf) -> Result<Array2<u64>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut data = vec![];
    for res in rdr.records() {
        let rec = res?;
        let row: Vec<u64> = rec
            .iter()
            .map(|s| s.parse::<u64>().unwrap_or(0))
            .collect();
        data.push(row);
    }
    Ok(Array2::from_shape_vec((data.len(), data[0].len()), data.into_iter().flatten().collect())?)
}

fn save_csv(mat: &Array2<u64>, path: &PathBuf) -> Result<()> {
    let mut wtr = csv::Writer::from_path(path)?;
    for row in mat.rows() {
        wtr.write_record(row.iter().map(|v| v.to_string()))?;
    }
    wtr.flush()?;
    Ok(())
}

// -------------- gRPC server (simplified stub) --------------
#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Keygen { out, radix_log } => {
            println!("[+] Generating keys (radix_log = {radix_log})…");
            let (ck, sk) = gen_keys_radix(radix_log);
            save_keyset(&Keyset { client_key: ck, server_key: sk }, &out)?;
            println!("[✓] Keys saved to {}", out.display());
        }
        Commands::Encrypt { input, keys, out } => {
            let keyset = load_keyset(&keys)?;
            println!("[+] Loading data…");
            let data = load_csv(&input)?;
            println!("[+] Encrypting {} records…", data.nrows());
            let enc: Vec<CipherVec> = data
                .rows()
                .into_iter()
                .map(|row| encrypt_record(&row.to_owned(), &keyset.client_key))
                .collect();
            fs::write(&out, bincode::serialize(&enc)?)?;
            println!("[✓] Ciphertext written to {}", out.display());
        }
        Commands::Infer { cipher, model, keys, out } => {
            let keyset = load_keyset(&keys)?;
            println!("[+] Loading encrypted data and model…");
            let enc: Vec<CipherVec> = bincode::deserialize(&fs::read(cipher)?)?;
            let weights: Array1<f64> = ndarray_npy::read_npy(model)?;
            let mut preds: Vec<CiphertextRadix> = Vec::with_capacity(enc.len());
            for rec in &enc {
                let z = he_dot_product(rec, &weights, &keyset.server_key);
                let y_hat = he_sigmoid(&z, &keyset.server_key);
                preds.push(y_hat);
            }
            fs::write(&out, bincode::serialize(&preds)?)?;
            println!("[✓] Predictions saved to {}", out.display());
        }
        Commands::Decrypt { cipher, keys, out } => {
            let keyset = load_keyset(&keys)?;
            let preds: Vec<CiphertextRadix> = bincode::deserialize(&fs::read(cipher)?)?;
            let plain: Vec<u64> = preds
                .iter()
                .map(|ct| u64::from_le_bytes(keyset.client_key.decrypt_radix(ct)))
                .collect();
            let arr = Array2::from_shape_vec((plain.len(), 1), plain)?;
            save_csv(&arr, &out)?;
            println!("[✓] Plaintext predictions written to {}", out.display());
        }
        Commands::Serve { keys, addr } => {
            let _keyset = load_keyset(&keys)?;
            println!("[⚙] gRPC server listening on {addr}. (Not fully implemented in this stub)");
            // TODO: tonic gRPC service with unary `Predict` RPC.
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
            }
        }
    }
    Ok(())
}
