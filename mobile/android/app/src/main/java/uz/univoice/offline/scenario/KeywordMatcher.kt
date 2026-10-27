package uz.univoice.offline.scenario

import kotlin.math.ceil
import kotlin.math.max

/** Port of the keyword matching in app/services/scenario_service.py. */
object KeywordMatcher {
    // Typo tolerance only for long words: on short ones a single edit is a
    // different word (oqib "flowing" vs o'qib "reading") or a negation.
    private const val FUZZY_MIN_LENGTH = 7
    private const val FUZZY_MIN_RATIO = 90.0

    // Complaints describe the machine in the third person (chiqmayapti). First
    // and second person forms are about the speaker: "chiqmayapman".
    private val uzPersonalEndings = listOf(
        "man", "manmi", "miz", "mizmi", "san", "sanmi", "siz", "sizmi",
        "dim", "dik", "ding", "dingiz", "dingizmi",
    )

    private fun commonPrefix(a: String, b: String): Int {
        var size = 0
        while (size < a.length && size < b.length && a[size] == b[size]) size++
        return size
    }

    /** rapidfuzz.fuzz.ratio: 100 * 2 * LCS / (len(a) + len(b)). */
    fun ratio(a: String, b: String): Double {
        if (a.isEmpty() && b.isEmpty()) return 100.0
        val row = IntArray(b.length + 1)
        for (i in 1..a.length) {
            var diagonal = 0
            for (j in 1..b.length) {
                val above = row[j]
                row[j] = if (a[i - 1] == b[j - 1]) diagonal + 1 else max(row[j], row[j - 1])
                diagonal = above
            }
        }
        return 100.0 * 2 * row[b.length] / (a.length + b.length)
    }

    /**
     * Match one keyword word against one spoken word, tolerating inflection.
     * Stems are marked with "*"; plain keywords only accept a short ending.
     */
    fun tokenMatches(keyword: String, token: String): Boolean {
        if (keyword.endsWith("*")) {
            val stem = keyword.dropLast(1)
            if (!token.startsWith(stem)) return false
            val isUzbekLatin = stem.all { it.code < 128 }
            val rest = token.substring(stem.length)
            return !(isUzbekLatin && uzPersonalEndings.any { rest.endsWith(it) })
        }
        if (token == keyword) return true
        // Particles like "не" must not match "нет".
        if (keyword.length <= 2) return false
        val extra = token.length - keyword.length
        if (keyword.length <= 3) return token.startsWith(keyword) && extra <= 2
        // вода -> воды / водой, but not водитель.
        if (keyword.length <= 5) return extra <= 3 && commonPrefix(keyword, token) >= keyword.length - 1
        if (token.startsWith(keyword)) return true
        if (commonPrefix(keyword, token) >= max(5, ceil(keyword.length * 0.55).toInt())) return true
        return keyword.length >= FUZZY_MIN_LENGTH && ratio(keyword, token) >= FUZZY_MIN_RATIO
    }

    /** Strict match for a keyword with its apostrophe dropped: no stem guessing. */
    private fun bareMatches(bare: String, token: String): Boolean {
        if (bare.endsWith("*")) return tokenMatches(bare, token)
        return token.startsWith(bare) && token.length - bare.length <= 2
    }

    /** Every word of a (possibly multi-word) keyword must match a distinct token. */
    fun keywordMatches(keyword: String, tokens: List<String>): Boolean {
        val remaining = tokens.toMutableList()
        for (raw in TextNormalizer.words(keyword)) {
            val word = TextNormalizer.normalize(raw) + if (raw.endsWith("*")) "*" else ""
            // Uzbek Latin is often typed without apostrophes (qoydi for qo'ydi),
            // so a keyword's apostrophe is optional; the reverse is not.
            val bare = word.replace("'", "")
            val index = remaining.indexOfFirst { token ->
                tokenMatches(word, token) || (bare != word && bareMatches(bare, token))
            }
            if (index < 0) return false
            remaining.removeAt(index)
        }
        return true
    }
}
