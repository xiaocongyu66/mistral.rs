use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use mistralrs::{BlockingModel, GgufModelBuilder, RequestBuilder, TextMessageRole};
use mistralrs_core::SamplingParams;
use serde_json::json;

uniffi::setup_scaffolding!();

#[derive(Debug, thiserror::Error, uniffi::Error)]
#[uniffi(flat_error)]
pub enum FfiError {
    #[error("{0}")]
    NotLoaded(String),
    #[error("{0}")]
    Load(String),
    #[error("{0}")]
    Inference(String),
    #[error("{0}")]
    Json(String),
}

static MODEL: OnceLock<Mutex<Option<Arc<BlockingModel>>>> = OnceLock::new();
static LOADING: AtomicBool = AtomicBool::new(false);

fn model_slot() -> &'static Mutex<Option<Arc<BlockingModel>>> {
    MODEL.get_or_init(|| Mutex::new(None))
}

#[uniffi::export]
fn load_model(
    model_dir: String,
    gguf_file: String,
    tokenizer_json: Option<String>,
) -> Result<String, FfiError> {
    if LOADING.swap(true, Ordering::SeqCst) {
        return Err(FfiError::Load("load_model already in progress".to_string()));
    }
    let t0 = Instant::now();
    let result = (|| {
        let mut builder = GgufModelBuilder::new(model_dir, vec![gguf_file]).with_force_cpu();
        if let Some(tok) = tokenizer_json {
            builder = builder.with_tokenizer_json(tok);
        }
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| FfiError::Load(e.to_string()))?;
        let model = rt
            .block_on(builder.build())
            .map_err(|e| FfiError::Load(e.to_string()))?;
        *model_slot()
            .lock()
            .map_err(|e| FfiError::Load(e.to_string()))? =
            Some(Arc::new(BlockingModel::new(model, Arc::new(rt))));
        Ok(json!({ "load_ms": t0.elapsed().as_millis() as u64 }).to_string())
    })();
    LOADING.store(false, Ordering::SeqCst);
    result
}

#[uniffi::export]
fn complete(prompt: String, max_tokens: u32, temperature: f32) -> Result<String, FfiError> {
    let model = model_slot()
        .lock()
        .map_err(|e| FfiError::NotLoaded(e.to_string()))?
        .clone()
        .ok_or_else(|| FfiError::NotLoaded("call load_model first".to_string()))?;
    let t0 = Instant::now();
    let request = RequestBuilder::new()
        .add_message(TextMessageRole::User, prompt)
        .set_sampling(SamplingParams {
            max_len: Some(max_tokens as usize),
            temperature: (temperature > 0.0).then(|| temperature as f64),
            top_k: (temperature <= 0.0).then_some(1),
            ..SamplingParams::deterministic()
        });
    let response = model
        .send_chat_request(request)
        .map_err(|e| FfiError::Inference(e.to_string()))?;
    let text = response
        .choices
        .first()
        .and_then(|c| c.message.content.clone())
        .unwrap_or_default();
    Ok(json!({
        "text": text,
        "prompt_toks": response.usage.prompt_tokens,
        "out_toks": response.usage.completion_tokens,
        "ms": t0.elapsed().as_millis() as u64,
    })
    .to_string())
}

#[uniffi::export]
fn unload_model() {
    if let Ok(mut slot) = model_slot().lock() {
        *slot = None;
    }
}
