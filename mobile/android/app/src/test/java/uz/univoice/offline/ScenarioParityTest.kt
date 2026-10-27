package uz.univoice.offline

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import uz.univoice.offline.scenario.KeywordMatcher
import uz.univoice.offline.scenario.ScenarioMatcher
import uz.univoice.offline.scenario.TextNormalizer
import java.io.File

/**
 * The Kotlin matcher must behave exactly like app/services/scenario_service.py.
 * intent_golden.json is written by `python -m scripts.export_intent_golden`.
 */
class ScenarioParityTest {
    private val repo: File = generateSequence(File("").absoluteFile) { it.parentFile }
        .first { File(it, "app/data/scenarios.json").exists() && File(it, "tests/data/intent_cases.json").exists() }

    private val matcher = ScenarioMatcher(
        ScenarioMatcher.parseScenarios(File(repo, "app/data/scenarios.json").readText()),
        ScenarioMatcher.parsePrompts(File(repo, "app/data/prompts.json").readText()),
    )

    private val golden: JSONArray =
        JSONArray(javaClass.classLoader!!.getResource("intent_golden.json")!!.readText())

    @Test
    fun normalizerMatchesPython() {
        var checked = 0
        for (i in 0 until golden.length()) {
            val row = golden.getJSONObject(i)
            if (!row.has("normalized")) continue
            val text = row.getString("text")
            assertEquals("normalize($text)", row.getString("normalized"), TextNormalizer.normalize(text))
            assertEquals("toUzbekLatin($text)", row.getString("latin"), TextNormalizer.toUzbekLatin(text))
            checked++
        }
        assertTrue(checked > 100)
    }

    @Test
    fun resolveMatchesPython() {
        val mismatches = mutableListOf<String>()
        var checked = 0
        for (i in 0 until golden.length()) {
            val row = golden.getJSONObject(i)
            if (!row.has("scenario_id")) continue
            val text = row.getString("text")
            val language = if (row.isNull("language")) null else row.getString("language")
            val expectedId = if (row.isNull("scenario_id")) null else row.getString("scenario_id")
            val expectedConfidence = row.getDouble("confidence")
            val match = matcher.resolve(text, language)
            val id = match?.scenario?.id
            val confidence = match?.confidence ?: 0.0
            if (id != expectedId || confidence != expectedConfidence) {
                mismatches += "[$language] \"$text\": python=$expectedId/$expectedConfidence kotlin=$id/$confidence"
            }
            checked++
        }
        assertTrue(checked > 300)
        assertTrue(mismatches.joinToString("\n"), mismatches.isEmpty())
    }

    @Test
    fun handWrittenCasesAsInPythonTests() {
        val cases = JSONObject(File(repo, "tests/data/intent_cases.json").readText())
        val complaints = cases.getJSONArray("complaints")
        for (i in 0 until complaints.length()) {
            val (text, language, expected) = complaints.getJSONArray(i).let { Triple(it.getString(0), it.getString(1), it.getString(2)) }
            val match = matcher.resolve(text, language)
            assertNotNull(text, match)
            assertEquals(text, expected, match!!.scenario.id)
            assertTrue(text, match.confidence >= 0.65)
        }
        val notComplaints = cases.getJSONArray("not_complaints")
        for (i in 0 until notComplaints.length()) {
            val row = notComplaints.getJSONArray(i)
            assertNull(row.getString(0), matcher.resolve(row.getString(0), row.getString(1)))
        }
    }

    @Test
    fun replyUsesScenarioOrNotUnderstood() {
        val water = matcher.reply("Suv chiqmayapti", "uz")
        assertEquals("vending_no_water_uz", water.audioKey)
        assertEquals("uz", water.language)
        assertTrue(water.text.isNotBlank())

        val operator = matcher.reply("Соедините с оператором", "ru")
        assertEquals("operator_request_ru", operator.scenarioId)
        assertTrue(operator.offerOperator)

        val unknown = matcher.reply("Привет, как дела", "ru")
        assertEquals("not_understood", unknown.audioKey)
        assertNull(unknown.scenarioId)
        assertEquals(matcher.prompt("not_understood", "ru"), unknown.text)
    }

    @Test
    fun ratioMatchesRapidfuzz() {
        // fuzz.ratio("chiqmayapti", "chiqmayabdi") == 81.8181...
        assertEquals(81.818, KeywordMatcher.ratio("chiqmayapti", "chiqmayabdi"), 0.001)
        assertEquals(100.0, KeywordMatcher.ratio("ishlamayapti", "ishlamayapti"), 0.0)
    }
}
