// JNI bridge between WhisperEngine.kt and whisper.cpp.

#include <android/log.h>
#include <jni.h>

#include <string>

#include "whisper.h"

#define TAG "WhisperJni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)

extern "C" {

JNIEXPORT jlong JNICALL
Java_uz_univoice_offline_stt_WhisperEngine_nativeInit(JNIEnv *env, jclass, jstring model_path) {
    const char *path = env->GetStringUTFChars(model_path, nullptr);
    whisper_context_params params = whisper_context_default_params();
    params.use_gpu = false;
    whisper_context *ctx = whisper_init_from_file_with_params(path, params);
    LOGI("loaded %s: %s", path, ctx ? "ok" : "failed");
    env->ReleaseStringUTFChars(model_path, path);
    return reinterpret_cast<jlong>(ctx);
}

JNIEXPORT void JNICALL
Java_uz_univoice_offline_stt_WhisperEngine_nativeFree(JNIEnv *, jclass, jlong handle) {
    whisper_free(reinterpret_cast<whisper_context *>(handle));
}

JNIEXPORT jbyteArray JNICALL
Java_uz_univoice_offline_stt_WhisperEngine_nativeTranscribe(
        JNIEnv *env, jclass, jlong handle, jfloatArray audio, jstring language, jint threads) {
    auto *ctx = reinterpret_cast<whisper_context *>(handle);
    const char *lang = env->GetStringUTFChars(language, nullptr);
    std::string language_code(lang);
    env->ReleaseStringUTFChars(language, lang);

    // The way the server runs the Uzbek fine-tune (docs/stt-evaluation.md):
    // language forced, greedy, no timestamps (they made it drop the first
    // words), no prompt (it continued the prompt), no temperature fallback.
    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.n_threads = threads;
    params.language = language_code.c_str();
    params.detect_language = false;
    params.translate = false;
    params.no_timestamps = true;
    params.no_context = true;
    params.single_segment = false;
    params.initial_prompt = nullptr;
    params.temperature = 0.0f;
    params.temperature_inc = 0.0f;
    params.suppress_blank = true;
    params.print_progress = false;
    params.print_realtime = false;
    params.print_special = false;
    params.print_timestamps = false;

    jsize length = env->GetArrayLength(audio);
    jfloat *samples = env->GetFloatArrayElements(audio, nullptr);
    int status = whisper_full(ctx, params, samples, length);
    env->ReleaseFloatArrayElements(audio, samples, JNI_ABORT);
    if (status != 0) {
        LOGI("whisper_full failed: %d", status);
        return env->NewByteArray(0);
    }

    std::string text;
    const int segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < segments; ++i) {
        text += whisper_full_get_segment_text(ctx, i);
    }
    // Raw UTF-8: a segment can end inside a multi-byte character, which
    // NewStringUTF (modified UTF-8) would reject.
    jbyteArray bytes = env->NewByteArray(static_cast<jsize>(text.size()));
    env->SetByteArrayRegion(bytes, 0, static_cast<jsize>(text.size()),
                            reinterpret_cast<const jbyte *>(text.data()));
    return bytes;
}

JNIEXPORT jstring JNICALL
Java_uz_univoice_offline_stt_WhisperEngine_nativeSystemInfo(JNIEnv *env, jclass) {
    return env->NewStringUTF(whisper_print_system_info());
}

}  // extern "C"
