use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use jni::objects::{JClass, JString};
use jni::sys::jstring;
use jni::JNIEnv;
use mistralrs::blocking::BlockingModel;
use mistralrs::{GgufModelBuilder, RequestBuilder, TextMessageRole};
use mistralrs_core::SamplingParams;
use serde_json::json;

#[derive(Debug, thiserror::Error)]
pub enum FfiError {
    #[error("model not loaded: {0}")]
    NotLoaded(String),
    #[error("load failed: {0}")]
    Load(String),
    #[error("inference failed: {0}")]
    Inference(String),
    #[error("jni error: {0}")]
    Jni(String),
}

static MODEL: OnceLock<Mutex<Option<Arc<BlockingModel>>>> = OnceLock::new();
static LOADING: AtomicBool = AtomicBool::new(false);

fn model_slot() -> &'static Mutex<Option<Arc<BlockingModel>>> {
    MODEL.get_or_init(|| Mutex::new(None))
}

fn load_model_inner(
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

fn complete_inner(prompt: String, max_tokens: u32, temperature: f32) -> Result<String, FfiError> {
    let model = model_slot()
        .lock()
        .map_err(|e| FfiError::NotLoaded(e.to_string()))?
        .clone()
        .ok_or_else(|| FfiError::NotLoaded("call loadModel first".to_string()))?;
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

fn jstr_to_rust(env: &mut JNIEnv, s: &JString) -> Result<String, FfiError> {
    env.get_string(s)
        .map(|v| v.to_string_lossy().into_owned())
        .map_err(|e| FfiError::Jni(e.to_string()))
}

fn throw(env: &mut JNIEnv, err: &FfiError) {
    let _ = env.throw_new("java/lang/RuntimeException", &err.to_string());
}

#[no_mangle]
pub extern "system" fn Java_ai_duan_demo_NativeBridge_loadModel<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    j_model_dir: JString<'local>,
    j_gguf_file: JString<'local>,
    j_tokenizer: JString<'local>,
) -> jstring {
    let result = (|| {
        let model_dir = jstr_to_rust(&mut env, &j_model_dir)?;
        let gguf_file = jstr_to_rust(&mut env, &j_gguf_file)?;
        let tokenizer = if j_tokenizer.is_null() {
            None
        } else {
            Some(jstr_to_rust(&mut env, &j_tokenizer)?)
        };
        load_model_inner(model_dir, gguf_file, tokenizer)
    })();
    match result {
        Ok(json) => env
            .new_string(&json)
            .map(|v| v.into_raw())
            .unwrap_or(std::ptr::null_mut()),
        Err(e) => {
            throw(&mut env, &e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_ai_duan_demo_NativeBridge_complete<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    j_prompt: JString<'local>,
    j_max_tokens: i32,
    j_temperature: f32,
) -> jstring {
    let result = (|| {
        let prompt = jstr_to_rust(&mut env, &j_prompt)?;
        complete_inner(prompt, j_max_tokens.max(1) as u32, j_temperature)
    })();
    match result {
        Ok(json) => env
            .new_string(&json)
            .map(|v| v.into_raw())
            .unwrap_or(std::ptr::null_mut()),
        Err(e) => {
            throw(&mut env, &e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_ai_duan_demo_NativeBridge_unloadModel<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
) {
    if let Ok(mut slot) = model_slot().lock() {
        *slot = None;
    }
}
