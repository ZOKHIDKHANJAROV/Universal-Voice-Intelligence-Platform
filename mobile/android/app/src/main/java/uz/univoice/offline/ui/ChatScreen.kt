package uz.univoice.offline.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.List
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Call
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.AssistChip
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Switch
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import kotlinx.coroutines.launch
import uz.univoice.offline.BotViewModel
import uz.univoice.offline.ModelState
import uz.univoice.offline.Phase
import uz.univoice.offline.R
import uz.univoice.offline.UiState
import uz.univoice.offline.call.CallPermissions
import uz.univoice.offline.data.Message
import uz.univoice.offline.data.Role

enum class Screen { CHAT, HISTORY, MODELS, BENCHMARK, SETTINGS }

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(bot: BotViewModel, state: UiState, onNavigate: (Screen) -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val snackbar = remember { SnackbarHostState() }
    var menu by remember { mutableStateOf(false) }

    LaunchedEffect(state.notice) {
        state.notice?.let { snackbar.showSnackbar(it); bot.dismissNotice() }
    }

    fun share() = scope.launch {
        val (file, count) = bot.export()
        if (count == 0) {
            snackbar.showSnackbar("Yozuvlar yoʻq · Записей пока нет")
            return@launch
        }
        val uri = FileProvider.getUriForFile(context, "${context.packageName}.files", file)
        val send = Intent(Intent.ACTION_SEND).setType("application/zip")
            .putExtra(Intent.EXTRA_STREAM, uri)
            .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        context.startActivity(Intent.createChooser(send, "$count"))
    }

    Scaffold(
        snackbarHost = { SnackbarHost(snackbar) },
        topBar = {
            TopAppBar(
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Logo(32.dp)
                        Spacer(Modifier.size(10.dp))
                        Text("UniVoice", fontWeight = FontWeight.SemiBold, maxLines = 1, softWrap = false)
                    }
                },
                actions = {
                    LanguageSwitch(state.language, bot::setLanguage)
                    IconButton(onClick = { onNavigate(Screen.HISTORY) }) { Icon(Icons.AutoMirrored.Filled.List, "Tarix") }
                    Box {
                        IconButton(onClick = { menu = true }) { Icon(Icons.Default.MoreVert, null) }
                        DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                            DropdownMenuItem(text = { Text("Yangi suhbat · Новый разговор") }, onClick = { menu = false; bot.newConversation() })
                            DropdownMenuItem(text = { Text("Modellar · Модели") }, onClick = { menu = false; onNavigate(Screen.MODELS) })
                            DropdownMenuItem(text = { Text("Tezlik testi · Тест скорости") }, onClick = { menu = false; onNavigate(Screen.BENCHMARK) })
                            DropdownMenuItem(text = { Text("Yozuvlarni eksport · Экспорт записей") }, onClick = { menu = false; share() })
                            DropdownMenuItem(text = { Text("Sozlamalar · Настройки") }, onClick = { menu = false; onNavigate(Screen.SETTINGS) })
                        }
                    }
                },
            )
        },
    ) { padding ->
        Column(Modifier.fillMaxSize().padding(padding)) {
            AutoAnswerCard(bot, state, onNavigate, snackbar)
            ReadinessBanner(bot, state, onNavigate)
            CallbackBanner(state)
            val listState = rememberLazyListState()
            LaunchedEffect(state.messages.size) {
                if (state.messages.isNotEmpty()) listState.animateScrollToItem(state.messages.size - 1)
            }
            if (state.messages.isEmpty()) EmptyState(Modifier.weight(1f).fillMaxWidth(), state.autoAnswer)
            else LazyColumn(
                state = listState,
                modifier = Modifier.weight(1f).fillMaxWidth(),
                contentPadding = androidx.compose.foundation.layout.PaddingValues(12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                items(state.messages, key = { it.id }) { message ->
                    Bubble(bot, message, playing = state.playingId == message.id)
                }
            }
            TalkButton(bot, state)
        }
    }
}

/** The switch that turns the answering machine on: it then picks up incoming calls by itself. */
@Composable
private fun AutoAnswerCard(bot: BotViewModel, state: UiState, onNavigate: (Screen) -> Unit, snackbar: SnackbarHostState) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var pending by remember { mutableStateOf(false) }
    fun turnOn() {
        val blocker = bot.autoAnswerBlocker()
        if (blocker != null) {
            scope.launch { snackbar.showSnackbar(blocker) }
            return
        }
        bot.setAutoAnswer(true)
    }
    val permissions = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
        if (pending) {
            pending = false
            if (CallPermissions.missing(context).isEmpty()) turnOn()
            else scope.launch { snackbar.showSnackbar("Ruxsatlar kerak: telefon, mikrofon · Нужны разрешения: телефон, микрофон") }
        }
    }
    val active = state.autoAnswer || state.inCall
    val onColor = if (active) Color.White else MaterialTheme.colorScheme.onSurface
    val background = if (active) Modifier.background(Brush.linearGradient(listOf(BrandTeal, BrandBlue)))
    else Modifier.background(MaterialTheme.colorScheme.surfaceVariant)
    Box(Modifier.padding(horizontal = 12.dp, vertical = 8.dp).clip(RoundedCornerShape(20.dp)).then(background)) {
        Row(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 14.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(
                Modifier.size(44.dp).clip(CircleShape).background(onColor.copy(alpha = 0.18f)),
                contentAlignment = Alignment.Center,
            ) { Icon(Icons.Default.Call, null, tint = onColor) }
            Spacer(Modifier.size(14.dp))
            Column(Modifier.weight(1f)) {
                Text("Avtojavob · Автоответчик", style = MaterialTheme.typography.titleMedium, color = onColor, fontWeight = FontWeight.SemiBold)
                Text(
                    when {
                        state.inCall -> "Qoʻngʻiroq davom etmoqda · Идёт звонок" +
                            (state.callCaller?.takeIf { it.isNotEmpty() }?.let { ": $it" } ?: "")
                        state.autoAnswer -> "Yoqilgan: qoʻngʻiroqni kutmoqda · Включён: ждёт звонок"
                        else -> "Oʻchirilgan · Выключен"
                    },
                    style = MaterialTheme.typography.bodySmall,
                    color = onColor.copy(alpha = 0.85f),
                )
            }
            Switch(
                checked = state.autoAnswer,
                onCheckedChange = { on ->
                    when {
                        !on -> bot.setAutoAnswer(false)
                        CallPermissions.missing(context).isNotEmpty() -> {
                            pending = true
                            permissions.launch(CallPermissions.requested())
                        }
                        else -> turnOn()
                    }
                },
            )
        }
    }
}

/** The logo: the launcher icon's bubble and sound wave on its teal-to-blue tile. */
@Composable
fun Logo(size: androidx.compose.ui.unit.Dp) {
    Box(
        Modifier.size(size).clip(RoundedCornerShape(size * 0.28f))
            .background(Brush.linearGradient(listOf(BrandTeal, BrandBlue))),
        contentAlignment = Alignment.Center,
    ) {
        // The launcher art keeps a wide safe margin; crop it for the small tile.
        Image(painterResource(R.drawable.ic_launcher_foreground), null, Modifier.size(size * 2.1f), contentScale = ContentScale.Fit)
    }
}

@Composable
private fun EmptyState(modifier: Modifier, on: Boolean) {
    Column(modifier.padding(32.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) {
        Logo(96.dp)
        Spacer(Modifier.height(20.dp))
        Text("UniVoice", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.SemiBold)
        Spacer(Modifier.height(8.dp))
        Text(
            if (on) "Avtojavob yoqilgan. Qoʻngʻiroq shu yerda koʻrinadi." + LINE + "Автоответчик включён. Звонок появится здесь."
            else "Avtojavobni yoqing yoki mikrofon tugmasini bosib sinab koʻring." + LINE + "Включите автоответчик или проверьте голосом кнопкой микрофона.",
            textAlign = TextAlign.Center,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

/** An answered call that ended with "an operator will call you back": the way to do it. */
@Composable
private fun CallbackBanner(state: UiState) {
    val context = LocalContext.current
    val number = state.callCaller?.takeIf { it.isNotEmpty() && state.callNeedsCallback } ?: return
    Surface(color = MaterialTheme.colorScheme.tertiaryContainer, modifier = Modifier.fillMaxWidth()) {
        Row(Modifier.padding(horizontal = 16.dp, vertical = 4.dp), verticalAlignment = Alignment.CenterVertically) {
            Text("Qayta qoʻngʻiroq kerak · Нужен перезвон: $number", Modifier.weight(1f), style = MaterialTheme.typography.bodySmall)
            TextButton(onClick = { context.startActivity(Intent(Intent.ACTION_DIAL, Uri.parse("tel:$number"))) }) {
                Icon(Icons.Default.Call, null)
                Text(" Qoʻngʻiroq · Позвонить")
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun LanguageSwitch(language: String, onChange: (String) -> Unit) {
    SingleChoiceSegmentedButtonRow(Modifier.padding(end = 4.dp)) {
        listOf("uz" to "UZ", "ru" to "RU").forEachIndexed { index, (code, label) ->
            SegmentedButton(
                selected = language == code,
                onClick = { onChange(code) },
                shape = SegmentedButtonDefaults.itemShape(index, 2),
                icon = {},
            ) { Text(label) }
        }
    }
}

@Composable
private fun ReadinessBanner(bot: BotViewModel, state: UiState, onNavigate: (Screen) -> Unit) {
    val message = when {
        !bot.sttAvailable -> "Bu yigʻmada ${state.engine.name} yoʻq · В этой сборке нет ${state.engine.name}: Modellar"
        state.models[bot.activeModel(state.language).id] !is ModelState.Ready -> "Nutq modeli yuklanmagan · Модель речи не загружена"
        else -> null
    } ?: return
    Surface(color = MaterialTheme.colorScheme.errorContainer, modifier = Modifier.fillMaxWidth()) {
        Row(Modifier.padding(horizontal = 16.dp, vertical = 8.dp), verticalAlignment = Alignment.CenterVertically) {
            Text(message, Modifier.weight(1f), style = MaterialTheme.typography.bodySmall)
            TextButton(onClick = { onNavigate(Screen.MODELS) }) { Text("Modellar") }
        }
    }
}

@Composable
private fun Bubble(bot: BotViewModel, message: Message, playing: Boolean) {
    val context = LocalContext.current
    val caller = message.role == Role.CALLER
    if (message.role == Role.SYSTEM) {
        Text(
            message.text, Modifier.fillMaxWidth().padding(vertical = 4.dp), textAlign = TextAlign.Center,
            style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        return
    }
    Row(Modifier.fillMaxWidth(), horizontalArrangement = if (caller) Arrangement.End else Arrangement.Start) {
        Surface(
            color = if (caller) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant,
            shape = if (caller) RoundedCornerShape(18.dp, 18.dp, 4.dp, 18.dp) else RoundedCornerShape(18.dp, 18.dp, 18.dp, 4.dp),
            modifier = Modifier.widthIn(max = 320.dp),
        ) {
            Column(Modifier.padding(12.dp)) {
                if (caller && message.text.isBlank()) {
                    Text("(nutq tanilmadi · речь не распознана)", fontStyle = FontStyle.Italic)
                } else {
                    Text(message.text)
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(meta(message), style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.weight(1f, fill = false))
                    val canPlay = if (caller) message.audioPath != null else bot.hasAnswerAudio(message)
                    if (canPlay) {
                        IconButton(onClick = { if (playing) bot.stopPlayback() else bot.play(message) }, modifier = Modifier.size(36.dp)) {
                            Icon(if (playing) Icons.Default.Stop else Icons.Default.PlayArrow, "Tinglash")
                        }
                    }
                }
                val phone = bot.settings.operatorPhone
                if (!caller && message.offerOperator && phone.isNotEmpty()) {
                    AssistChip(
                        onClick = { context.startActivity(Intent(Intent.ACTION_DIAL, Uri.parse("tel:$phone"))) },
                        label = { Text("Operatorga qoʻngʻiroq · Позвонить оператору") },
                        leadingIcon = { Icon(Icons.Default.Call, null) },
                    )
                }
            }
        }
    }
}

private fun meta(message: Message): String = buildList {
    message.language?.let { add(it) }
    message.sttMillis?.let { add("%.1f s".format(it / 1000.0)) }
    message.scenarioId?.let { add(it) }
    message.confidence?.takeIf { message.role == Role.BOT && it > 0 }?.let { add("%.2f".format(it)) }
}.joinToString(" · ")

@Composable
private fun TalkButton(bot: BotViewModel, state: UiState) {
    val context = LocalContext.current
    var permissionAsked by remember { mutableStateOf(false) }
    val permission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { permissionAsked = true }
    val busy = state.inCall || state.phase == Phase.TRANSCRIBING || state.phase == Phase.LOADING_MODEL
    val label = if (state.inCall) "Qoʻngʻiroq davom etmoqda · Идёт звонок" else when (state.phase) {
        Phase.RECORDING -> "Gapiring… qoʻyib yuboring · Говорите… отпустите"
        Phase.LOADING_MODEL -> "Model yuklanmoqda… · Загрузка модели…"
        Phase.TRANSCRIBING -> "Tanilmoqda… · Распознаю…"
        Phase.PLAYING -> "Javob · Ответ"
        Phase.IDLE -> "Bosib turib gapiring · Удерживайте и говорите"
    }
    Column(
        Modifier.fillMaxWidth().navigationBarsPadding().padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(label, style = MaterialTheme.typography.bodyMedium)
        Spacer(Modifier.height(12.dp))
        val color = when {
            state.phase == Phase.RECORDING -> MaterialTheme.colorScheme.error
            busy -> MaterialTheme.colorScheme.outline
            else -> MaterialTheme.colorScheme.primary
        }
        Box(
            Modifier.size(96.dp).clip(CircleShape).background(
                if (color == MaterialTheme.colorScheme.primary) Brush.linearGradient(listOf(BrandTeal, BrandBlue)) else Brush.linearGradient(listOf(color, color)),
            )
                .pointerInput(busy) {
                    if (busy) return@pointerInput
                    detectTapGestures(onPress = {
                        val granted = ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
                            PackageManager.PERMISSION_GRANTED
                        if (!granted) {
                            permission.launch(Manifest.permission.RECORD_AUDIO)
                            return@detectTapGestures
                        }
                        bot.startTalking()
                        tryAwaitRelease()
                        bot.stopTalking()
                    })
                },
            contentAlignment = Alignment.Center,
        ) {
            Icon(Icons.Default.Mic, "Gapirish", tint = MaterialTheme.colorScheme.onPrimary, modifier = Modifier.size(40.dp))
        }
        if (permissionAsked && ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            Text("Mikrofonga ruxsat kerak · Нужен доступ к микрофону", color = MaterialTheme.colorScheme.error)
        }
    }
}

private const val LINE = "\n"
