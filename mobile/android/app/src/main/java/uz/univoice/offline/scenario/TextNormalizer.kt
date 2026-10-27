package uz.univoice.offline.scenario

import java.util.Locale

/**
 * Port of normalize_text / to_uzbek_latin from app/services/scenario_service.py.
 * Keep the two in step: ScenarioParityTest compares them on a golden file.
 */
object TextNormalizer {
    // Uzbek Latin uses several look-alike apostrophes (o', o‘, oʻ, o`); Whisper
    // also leaks Turkish letters into Uzbek transcripts ("işlemeyabdi").
    private val apostrophes: Map<Char, String> = buildMap {
        for (c in "‘’ʻʼ`´") put(c, "'")
        put('ş', "sh"); put('ç', "ch"); put('ı', "i"); put('ğ', "g'"); put('ö', "o'"); put('ü', "u")
        // Lowercasing turns Turkish "İ" into "i" plus a combining dot.
        put('̇', "")
    }

    // Whisper writes Uzbek in Cyrillic about as often as in Latin, while
    // scenario keywords are Latin. Used only when matching Uzbek scenarios.
    private val uzCyrillicToLatin: Map<Char, String> = mapOf(
        'а' to "a", 'б' to "b", 'в' to "v", 'г' to "g", 'ғ' to "g'", 'д' to "d", 'е' to "e",
        'ё' to "yo", 'ж' to "j", 'з' to "z", 'и' to "i", 'й' to "y", 'к' to "k", 'қ' to "q",
        'л' to "l", 'м' to "m", 'н' to "n", 'о' to "o", 'ў' to "o'", 'п' to "p", 'р' to "r",
        'с' to "s", 'т' to "t", 'у' to "u", 'ф' to "f", 'х' to "x", 'ҳ' to "h", 'ц' to "ts",
        'ч' to "ch", 'ш' to "sh", 'ъ' to "'", 'ь' to "", 'ы' to "i", 'э' to "e", 'ю' to "yu",
        'я' to "ya",
        // Kazakh letters: Whisper often writes Uzbek speech in Kazakh Cyrillic.
        'ә' to "a", 'ө' to "o'", 'ұ' to "u", 'ү' to "u", 'ң' to "ng", 'і' to "i", 'һ' to "h",
    )

    /** Python's `\w` on str: letters, digits and other numerals, and "_". */
    fun isWordChar(c: Char): Boolean = c == '_' || when (Character.getType(c).toByte()) {
        Character.UPPERCASE_LETTER, Character.LOWERCASE_LETTER, Character.TITLECASE_LETTER,
        Character.MODIFIER_LETTER, Character.OTHER_LETTER,
        Character.DECIMAL_DIGIT_NUMBER, Character.LETTER_NUMBER, Character.OTHER_NUMBER -> true
        else -> false
    }

    fun normalize(text: String): String {
        // str.casefold(): lowercase plus the one fold that matters here.
        val folded = text.lowercase(Locale.ROOT).replace("ß", "ss")
        val mapped = translate(folded, apostrophes).replace('ё', 'е')
        val spaced = buildString(mapped.length) {
            for (c in mapped) append(if (isWordChar(c) || c == '\'') c else ' ')
        }
        return words(spaced).joinToString(" ")
    }

    fun toUzbekLatin(text: String): String {
        val normalized = normalize(text)
        // Uzbek Cyrillic "е" starts a word as "ye": еб -> yeb, ер -> yer.
        val withYe = buildString(normalized.length) {
            normalized.forEachIndexed { i, c ->
                if (c == 'е' && (i == 0 || !isWordChar(normalized[i - 1]))) append("ye") else append(c)
            }
        }
        return translate(withYe, uzCyrillicToLatin)
    }

    /** str.split(): runs of whitespace separate words, no empty words. */
    fun words(text: String): List<String> = text.split(' ', '\t', '\n', '\r').filter { it.isNotEmpty() }

    private fun translate(text: String, table: Map<Char, String>): String = buildString(text.length) {
        for (c in text) append(table[c] ?: c)
    }
}
