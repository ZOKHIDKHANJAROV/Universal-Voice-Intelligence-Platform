package uz.univoice.offline.stt

import java.util.Locale

/** Port of is_hallucination in app/speech/providers/whisper.py. */
object Hallucinations {
    // Text Whisper produces from silence or noise, learned from subtitle credits.
    private val markers = listOf(
        "субтитр", "dimatorzok", "продолжение следует", "спасибо за просмотр",
        "подписывайтесь на канал", "редактор субтитров",
    )

    // Whole transcripts that are subtitle tags for non-speech.
    private val nonSpeechTags = setOf("musiqa", "музыка", "music", "kulgi", "смех", "qarsaklar", "аплодисменты")

    fun isHallucination(text: String): Boolean {
        val lowered = text.lowercase(Locale.ROOT)
        if (lowered.trim(' ', '.', ',', '!', '?', '[', ']', '(', ')', '*', '-') in nonSpeechTags) return true
        return markers.any { it in lowered }
    }
}
