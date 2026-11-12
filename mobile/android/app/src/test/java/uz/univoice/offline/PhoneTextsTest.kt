package uz.univoice.offline

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import uz.univoice.offline.scenario.ScenarioMatcher
import java.io.File

class PhoneTextsTest {
    private val repo: File = generateSequence(File("").absoluteFile) { it.parentFile }
        .first { File(it, "app/data/scenarios.json").exists() }

    private fun read(name: String) = File(repo, "app/data/$name").readText()

    private val phone = JSONObject(read("phone_texts.json"))
    private val matcher = ScenarioMatcher.forPhone(read("scenarios.json"), read("prompts.json"), read("phone_texts.json"))

    @Test
    fun everyPhoneTextReplacesSomethingThatExists() {
        val scenarioIds = ScenarioMatcher.parseScenarios(read("scenarios.json")).map { it.id }.toSet()
        val promptKeys = ScenarioMatcher.parsePrompts(read("prompts.json")).keys
        for (id in phone.getJSONObject("scenarios").keys()) assertTrue("no scenario $id", id in scenarioIds)
        for (key in phone.getJSONObject("prompts").keys()) assertTrue("no prompt $key", key in promptKeys)
    }

    @Test
    fun thePhoneNeverAsksToPressZero() {
        // A phone cannot transfer a call: no "nolni bosing" / "nazhmite nol'".
        val zero = Regex("nol|ноль", RegexOption.IGNORE_CASE)
        val texts = buildList {
            phone.getJSONObject("scenarios").let { s -> s.keys().forEach { add(s.getString(it)) } }
            for (key in listOf("greeting", "no_input", "anything_else", "goodbye", "time_limit", "repeat", "callback")) {
                add(matcher.prompt(key, "uz")); add(matcher.prompt(key, "ru"))
            }
        }
        for (text in texts) assertFalse(text, zero.containsMatchIn(text))
    }

    @Test
    fun theAnswerUsesThePhoneWording() {
        val reply = matcher.reply("suv chiqmayapti", "uz")
        assertEquals("vending_no_water_uz", reply.scenarioId)
        assertEquals(phone.getJSONObject("scenarios").getString("vending_no_water_uz"), reply.text)
    }
}
