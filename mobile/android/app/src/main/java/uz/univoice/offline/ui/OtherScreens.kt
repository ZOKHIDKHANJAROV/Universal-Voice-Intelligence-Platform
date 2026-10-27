package uz.univoice.offline.ui

import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import uz.univoice.offline.BotViewModel
import uz.univoice.offline.ModelState
import uz.univoice.offline.UiState
import uz.univoice.offline.stt.ModelSpec
import uz.univoice.offline.stt.Models
import uz.univoice.offline.stt.WhisperEngine
import java.text.DateFormat
import java.util.Date

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun Page(title: String, onBack: () -> Unit, content: @Composable (Modifier) -> Unit) {
    Scaffold(topBar = {
        TopAppBar(
            title = { Text(title) },
            navigationIcon = { IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Orqaga") } },
        )
    }) { padding -> content(Modifier.fillMaxSize().padding(padding)) }
}

@Composable
fun HistoryScreen(bot: BotViewModel, state: UiState, onBack: () -> Unit) {
    var conversations by remember { mutableStateOf(bot.conversations()) }
    val format = remember { DateFormat.getDateTimeInstance(DateFormat.SHORT, DateFormat.SHORT) }
    Page("Suhbatlar · Разговоры", onBack) { modifier ->
        LazyColumn(modifier) {
            items(conversations, key = { it.id }) { conversation ->
                Row(
                    Modifier.fillMaxWidth()
                        .clickable { bot.openConversation(conversation.id); onBack() }
                        .padding(horizontal = 16.dp, vertical = 12.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(Modifier.weight(1f)) {
                        Text(format.format(Date(conversation.startedAt)) + if (conversation.id == state.conversationId) "  •" else "")
                        Text(
                            conversation.preview.ifBlank { "—" } + "  (${conversation.turns})",
                            style = MaterialTheme.typography.bodySmall, maxLines = 1,
                        )
                    }
                    IconButton(onClick = { bot.deleteConversation(conversation.id); conversations = bot.conversations() }) {
                        Icon(Icons.Default.Delete, "Oʻchirish")
                    }
                }
                HorizontalDivider()
            }
        }
    }
}

@Composable
fun ModelsScreen(bot: BotViewModel, state: UiState, onBack: () -> Unit) {
    Page("Modellar · Модели", onBack) { modifier ->
        Column(modifier.verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
            if (!bot.sttAvailable) {
                Text("Bu yigʻmada whisper.cpp yoʻq: modellarni yuklash mumkin, lekin tanish ishlamaydi. " +
                    "В этой сборке нет whisper.cpp: модели можно загрузить, но распознавание не заработает.",
                    color = MaterialTheme.colorScheme.error)
            }
            Models.all.forEach { spec -> ModelCard(bot, spec, state.models[spec.language] ?: ModelState.Missing) }
            Text("Javob yozuvlari · Записи ответов: ${bot.promptCount}", style = MaterialTheme.typography.bodySmall)
        }
    }
}

@Composable
private fun ModelCard(bot: BotViewModel, spec: ModelSpec, model: ModelState) {
    var url by remember { mutableStateOf(bot.settings.modelUrl(spec)) }
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) bot.import(spec, uri)
    }
    Card {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(spec.title, style = MaterialTheme.typography.titleMedium)
            spec.sizeBytes?.let { Text("%.0f MB".format(it / 1e6), style = MaterialTheme.typography.bodySmall) }
            when (model) {
                ModelState.Ready -> Text("✓ Tayyor · Готова", color = MaterialTheme.colorScheme.primary)
                ModelState.Missing -> Text("Yuklanmagan · Не загружена")
                is ModelState.Failed -> Text("Xato · Ошибка: ${model.error}", color = MaterialTheme.colorScheme.error)
                is ModelState.Busy -> {
                    if (model.total > 0) {
                        LinearProgressIndicator(progress = { model.done.toFloat() / model.total }, modifier = Modifier.fillMaxWidth())
                    } else {
                        LinearProgressIndicator(Modifier.fillMaxWidth())
                    }
                    Text("%.0f / %s MB".format(model.done / 1e6, if (model.total > 0) "%.0f".format(model.total / 1e6) else "?"))
                }
            }
            if (model !is ModelState.Busy) {
                OutlinedTextField(
                    value = url,
                    onValueChange = { url = it; bot.settings.setModelUrl(spec, it) },
                    label = { Text("URL") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(onClick = { bot.download(spec) }) { Text("Yuklab olish") }
                    OutlinedButton(onClick = { picker.launch(arrayOf("*/*")) }) { Text("Fayldan") }
                    if (model == ModelState.Ready) TextButton(onClick = { bot.deleteModel(spec) }) { Text("Oʻchirish") }
                }
            }
        }
    }
}

@Composable
fun BenchmarkScreen(bot: BotViewModel, state: UiState, onBack: () -> Unit) {
    Page("Tezlik testi · Тест скорости", onBack) { modifier ->
        LazyColumn(modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            item {
                Text("${Build.MANUFACTURER} ${Build.MODEL} · Android ${Build.VERSION.RELEASE} · " +
                    "${Runtime.getRuntime().availableProcessors()} CPU · threads ${bot.settings.threads} · ${state.language}")
                Text(WhisperEngine.systemInfo(), style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)
                Button(onClick = { bot.runBenchmark() }, enabled = !state.benchmarkRunning, modifier = Modifier.padding(top = 8.dp)) {
                    Text(if (state.benchmarkRunning) "…" else "Boshlash · Запустить")
                }
                if (state.benchmarkRunning) LinearProgressIndicator(Modifier.fillMaxWidth().padding(top = 8.dp))
            }
            items(state.benchmark) { row ->
                Column {
                    val factor = if (row.audioSeconds > 0 && row.sttMillis >= 0) row.sttMillis / 1000.0 / row.audioSeconds else Double.NaN
                    Text("${row.name}: audio %.1f s → STT %.2f s (×%.2f)".format(row.audioSeconds, row.sttMillis / 1000.0, factor),
                        style = MaterialTheme.typography.labelLarge)
                    Text(row.text, style = MaterialTheme.typography.bodySmall)
                }
            }
            if (state.benchmark.isNotEmpty() && !state.benchmarkRunning) {
                item {
                    val valid = state.benchmark.filter { it.sttMillis >= 0 }
                    val audio = valid.sumOf { it.audioSeconds }
                    val stt = valid.sumOf { it.sttMillis } / 1000.0
                    Text("Jami · Итого: audio %.1f s, STT %.1f s, ×%.2f".format(audio, stt, if (audio > 0) stt / audio else Double.NaN),
                        style = MaterialTheme.typography.titleSmall)
                }
            }
        }
    }
}

@Composable
fun SettingsScreen(bot: BotViewModel, onBack: () -> Unit) {
    var phone by remember { mutableStateOf(bot.settings.operatorPhone) }
    var threads by remember { mutableIntStateOf(bot.settings.threads) }
    Page("Sozlamalar · Настройки", onBack) { modifier ->
        Column(modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
            OutlinedTextField(
                value = phone,
                onValueChange = { phone = it; bot.settings.operatorPhone = it },
                label = { Text("Operator telefoni · Телефон оператора") },
                singleLine = true,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Phone),
                modifier = Modifier.fillMaxWidth(),
            )
            Text("Oqimlar · Потоки распознавания: $threads")
            Slider(
                value = threads.toFloat(),
                onValueChange = { threads = it.toInt(); bot.settings.threads = threads },
                valueRange = 1f..8f,
                steps = 6,
            )
        }
    }
}
