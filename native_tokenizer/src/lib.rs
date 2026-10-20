//! Small owned-string C ABI. No Python is involved in runtime tokenization.
use regex::Regex;
use serde_json::{json, Value};
use std::ffi::{c_char, CStr, CString};
use tokenizers::Tokenizer;

unsafe fn string<'a>(p: *const c_char) -> &'a str {
    CStr::from_ptr(p).to_str().expect("invalid UTF-8")
}
fn output(f: impl FnOnce() -> Result<Value, String> + std::panic::UnwindSafe) -> *mut c_char {
    let result = std::panic::catch_unwind(f);
    let value = match result {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => json!({"error":e}),
        Err(_) => json!({"error":"text processing failed"}),
    };
    CString::new(value.to_string()).unwrap().into_raw()
}
#[no_mangle]
pub unsafe extern "C" fn pocket_tokenizer_new(path: *const c_char) -> *mut Tokenizer {
    match Tokenizer::from_file(string(path)) {
        Ok(t) => Box::into_raw(Box::new(t)),
        Err(_) => std::ptr::null_mut(),
    }
}
#[no_mangle]
pub unsafe extern "C" fn pocket_tokenizer_free(t: *mut Tokenizer) {
    if !t.is_null() {
        drop(Box::from_raw(t));
    }
}
#[no_mangle]
pub unsafe extern "C" fn pocket_text_free(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}
fn encode(t: &Tokenizer, s: &str) -> Result<Vec<u32>, String> {
    t.encode(s, true)
        .map(|e| e.get_ids().to_vec())
        .map_err(|e| e.to_string())
}
fn flag(c: &Value, k: &str, default: bool) -> bool {
    c[k].as_bool().unwrap_or(default)
}
fn prepare(raw: &str, c: &Value) -> Result<(String, usize), String> {
    let mut text = raw.trim().to_owned();
    if let Some(replacements) = c["replace_characters"]
        .as_object()
        .filter(|m| !m.is_empty())
    {
        text = text
            .chars()
            .map(|ch| {
                replacements
                    .get(&ch.to_string())
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .unwrap_or_else(|| ch.to_string())
            })
            .collect();
        text = text.split_whitespace().collect::<Vec<_>>().join(" ");
        text = Regex::new(r"([.!?…])\s*[,;:]")
            .unwrap()
            .replace_all(&text, "$1")
            .into_owned();
    }
    if text.is_empty() {
        return Err("Text prompt cannot be empty".into());
    }
    text = text.replace(['\n', '\r'], " ").replace("  ", " ");
    if flag(c, "remove_semicolons", false) {
        text = text.replace(';', ",");
    }
    let words = text.split_whitespace().count();
    if flag(c, "capitalize_first_letter", true) {
        let first = text.chars().next().unwrap();
        if !first.is_uppercase() {
            text = first.to_uppercase().to_string() + &text[first.len_utf8()..];
        }
    }
    if flag(c, "append_terminal_punctuation", true) {
        let closers = "\"'”’)]» ";
        let core = text.trim_end_matches(|ch| closers.contains(ch));
        if !core.is_empty() && !".!?…".contains(core.chars().last().unwrap()) {
            let weak = ",;:-–—";
            if weak.contains(core.chars().last().unwrap()) {
                let tail = text[core.len()..].trim();
                text = core.trim_end_matches(|ch| weak.contains(ch)).to_owned() + "." + tail;
            } else {
                text.push('.');
            }
        }
    }
    if flag(c, "pad_with_spaces_for_short_inputs", false) && words < 5 {
        text = "        ".to_owned() + &text;
    }
    Ok((text, if words <= 4 { 3 } else { 1 }))
}
fn segments(
    t: &Tokenizer,
    ids: &[u32],
    marks: &[u32],
    decimal: bool,
) -> Result<Vec<(usize, String)>, String> {
    let mut boundaries = vec![0];
    let mut previous = false;
    for (i, id) in ids.iter().enumerate() {
        if marks.contains(id) {
            previous = true;
        } else {
            if previous {
                let is_decimal = if decimal {
                    let left = t.decode(&ids[..i], true).map_err(|e| e.to_string())?;
                    let right = t.decode(&ids[i..], true).map_err(|e| e.to_string())?;
                    let mut chars = left.chars().rev();
                    chars.next() == Some('.')
                        && chars.next().is_some_and(|c| c.is_numeric())
                        && right.chars().next().is_some_and(|c| c.is_numeric())
                } else {
                    false
                };
                if !is_decimal {
                    boundaries.push(i);
                }
            }
            previous = false;
        }
    }
    boundaries.push(ids.len());
    boundaries
        .windows(2)
        .map(|b| {
            Ok((
                b[1] - b[0],
                t.decode(&ids[b[0]..b[1]], true)
                    .map_err(|e| e.to_string())?,
            ))
        })
        .collect()
}
#[no_mangle]
pub unsafe extern "C" fn pocket_text_process(
    t: *const Tokenizer,
    text: *const c_char,
    config: *const c_char,
) -> *mut c_char {
    output(|| {
        let t = &*t;
        let c: Value = serde_json::from_str(string(config)).map_err(|e| e.to_string())?;
        let raw = string(text);
        if c["encode_only"] == true {
            return Ok(json!({"ids":encode(t,raw)?}));
        }
        let (prepared, _) = prepare(raw, &c)?;
        let ids = encode(t, prepared.trim())?;
        let end = encode(t, ".!...?")?;
        let fallback = encode(t, ",;:")?;
        let max = c["max_tokens"].as_u64().unwrap_or(50) as usize;
        let mut refined = Vec::new();
        for (n, s) in segments(t, &ids, &end[1..], true)? {
            if n > max {
                let sub = segments(t, &encode(t, s.trim())?, &fallback[1..], false)?;
                if sub.len() > 1 {
                    refined.extend(sub);
                    continue;
                }
            }
            refined.push((n, s));
        }
        let mut chunks = Vec::new();
        let mut current = String::new();
        let mut count = 0;
        for (n, s) in refined {
            if !current.is_empty() && count + n > max {
                chunks.push(current.trim().to_owned());
                current.clear();
                count = 0;
            }
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(&s);
            count += n;
        }
        if !current.is_empty() {
            chunks.push(current.trim().to_owned());
        }
        let chunks:Result<Vec<_>,String>=chunks.iter().map(|s| {
            let (p,guess)=prepare(s,&c)?;
            Ok(json!({"text":p,"ids":encode(t,&p)?,"eos_extra":c["model_recommended_frames_after_eos"].as_u64().unwrap_or((guess+2) as u64)}))
        }).collect();
        Ok(json!({"chunks":chunks?}))
    })
}
