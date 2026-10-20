#pragma once
extern "C" {
void* pocket_tokenizer_new(const char* path);
void pocket_tokenizer_free(void* tokenizer);
char* pocket_text_process(const void* tokenizer, const char* text, const char* config_json);
void pocket_text_free(char* text);
}
