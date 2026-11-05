package uz.univoice.offline.call

import android.annotation.SuppressLint
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.telecom.TelecomManager
import android.telephony.TelephonyManager
import android.util.Log
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import uz.univoice.offline.MainActivity
import uz.univoice.offline.Settings
import uz.univoice.offline.data.ChatStore
import uz.univoice.offline.data.PromptAudio
import uz.univoice.offline.scenario.ScenarioMatcher
import uz.univoice.offline.stt.Engine
import uz.univoice.offline.stt.ModelStore
import uz.univoice.offline.stt.Models
import uz.univoice.offline.stt.VoskEngine

/**
 * The answering machine. While it runs (a foreground service, switched on by
 * a button) it listens for incoming calls, picks one up after a few rings,
 * and lets [CallSession] talk to the caller: no server and no internet.
 */
class AutoAnswerService : Service() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private lateinit var settings: Settings
    private lateinit var matcher: ScenarioMatcher
    private lateinit var prompts: PromptAudio
    private lateinit var store: ChatStore

    /** Language to the Vosk model that understands it; filled in once loaded. */
    @Volatile private var engines: Map<String, VoskEngine> = emptyMap()
    private var loading: Job? = null
    private var answerJob: Job? = null
    private var sessionJob: Job? = null
    private var weAnswered = false
    private var caller = ""
    private var receiverRegistered = false
    private var wakeLock: PowerManager.WakeLock? = null

    private val phoneState = object : BroadcastReceiver() {
        @Suppress("DEPRECATION")
        override fun onReceive(context: Context, intent: Intent) {
            val number = intent.getStringExtra(TelephonyManager.EXTRA_INCOMING_NUMBER)
            onPhoneState(intent.getStringExtra(TelephonyManager.EXTRA_STATE), number)
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        settings = Settings(this)
        store = ChatStore.get(this)
        matcher = ScenarioMatcher(
            ScenarioMatcher.parseScenarios(assets.open("scenarios.json").bufferedReader().use { it.readText() }),
            ScenarioMatcher.parsePrompts(assets.open("prompts.json").bufferedReader().use { it.readText() }),
        )
        prompts = PromptAudio(assets)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP || (intent == null && !settings.autoAnswer)) {
            settings.autoAnswer = false
            stopSelf()
            return START_NOT_STICKY
        }
        startInForeground()
        settings.autoAnswer = true
        if (!receiverRegistered) {
            ContextCompat.registerReceiver(
                this, phoneState, IntentFilter(TelephonyManager.ACTION_PHONE_STATE_CHANGED), ContextCompat.RECEIVER_EXPORTED,
            )
            receiverRegistered = true
        }
        if (loading == null) loading = scope.launch { loadModels() }
        CallState.serviceStarted()
        return START_STICKY
    }

    override fun onDestroy() {
        if (receiverRegistered) unregisterReceiver(phoneState)
        // Wait for a call in progress to let go of its recognizers before the models are freed.
        runBlocking { listOfNotNull(sessionJob, loading).forEach { it.cancelAndJoin() } }
        scope.cancel()
        engines.values.forEach { it.close() }
        engines = emptyMap()
        releaseWakeLock()
        CallState.serviceStopped()
        super.onDestroy()
    }

    // ----- models -----

    private fun loadModels() {
        val store = ModelStore(this)
        val loaded = Models.of(Engine.VOSK).filter { store.isReady(it) }.mapNotNull { spec ->
            runCatching { spec.language to VoskEngine.load(store.file(spec).absolutePath) }
                .onFailure { Log.e(TAG, "cannot load ${spec.title}", it) }
                .getOrNull()
        }.toMap()
        if (loaded.isEmpty()) {
            CallState.failed("Vosk modellari yoʻq · Нет моделей Vosk: Modellar")
            settings.autoAnswer = false
            stopSelf()
            return
        }
        engines = loaded
        Log.i(TAG, "listening for calls with ${loaded.keys}")
    }

    // ----- calls -----

    private fun onPhoneState(state: String?, number: String?) {
        if (!number.isNullOrEmpty()) caller = number
        Log.i(TAG, "phone state $state number=${if (caller.isEmpty()) "?" else "known"}")
        when (state) {
            TelephonyManager.EXTRA_STATE_RINGING -> if (answerJob?.isActive != true && sessionJob?.isActive != true) {
                answerJob = scope.launch { answerAfterDelay() }
            }
            TelephonyManager.EXTRA_STATE_OFFHOOK -> if (weAnswered) {
                if (sessionJob?.isActive != true) sessionJob = scope.launch { talk() }
            } else {
                // The owner picked up themselves: stay out of it.
                answerJob?.cancel()
            }
            TelephonyManager.EXTRA_STATE_IDLE -> endCall()
        }
    }

    @SuppressLint("MissingPermission")
    private suspend fun answerAfterDelay() {
        delay(settings.answerDelaySeconds * 1000L)
        // Models still loading right after the switch was turned on.
        var waited = 0
        while (engines.isEmpty() && waited < 8_000) {
            delay(200)
            waited += 200
        }
        if (engines.isEmpty()) {
            Log.w(TAG, "not answering: no speech model is loaded")
            return
        }
        val telecom = getSystemService(Context.TELECOM_SERVICE) as TelecomManager
        weAnswered = true
        try {
            telecom.acceptRingingCall()
        } catch (e: SecurityException) {
            weAnswered = false
            Log.e(TAG, "cannot answer: permission", e)
            CallState.failed("Qoʻngʻiroqqa javob berishga ruxsat yoʻq · Нет разрешения отвечать на звонки")
        }
    }

    private suspend fun talk() {
        // The audio path of an answered call needs a moment before the first word is heard.
        delay(700)
        val audio = CallAudio(this)
        acquireWakeLock()
        audio.enter()
        try {
            val session = CallSession(
                this, store, matcher, prompts, engines,
                CallDialogConfig(maxCallMs = settings.maxCallSeconds * 1000L),
                caller,
            )
            val outcome = session.run()
            Log.i(TAG, "call done: $outcome")
            delay(700) // let the last word reach the line
            hangUp()
        } catch (e: CancellationException) {
            throw e // the caller hung up
        } catch (e: Exception) {
            Log.e(TAG, "call failed", e)
            hangUp()
        } finally {
            audio.exit()
            releaseWakeLock()
        }
    }

    private fun endCall() {
        answerJob?.cancel()
        sessionJob?.cancel()
        answerJob = null
        sessionJob = null
        weAnswered = false
        caller = ""
    }

    @SuppressLint("MissingPermission")
    private fun hangUp() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.P) return
        runCatching { (getSystemService(Context.TELECOM_SERVICE) as TelecomManager).endCall() }
            .onFailure { Log.e(TAG, "cannot hang up", it) }
    }

    private fun acquireWakeLock() {
        val power = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = power.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "univoice:call")
            .apply { acquire((settings.maxCallSeconds + 60) * 1000L) }
    }

    private fun releaseWakeLock() {
        wakeLock?.takeIf { it.isHeld }?.release()
        wakeLock = null
    }

    // ----- notification -----

    private fun startInForeground() {
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        manager.createNotificationChannel(
            NotificationChannel(CHANNEL, "Avtojavob · Автоответчик", NotificationManager.IMPORTANCE_LOW),
        )
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        val stop = PendingIntent.getService(
            this, 1, Intent(this, AutoAnswerService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        val notification = Notification.Builder(this, CHANNEL)
            .setSmallIcon(android.R.drawable.sym_call_incoming)
            .setContentTitle("Avtojavob yoqilgan · Автоответчик включён")
            .setContentText("Qoʻngʻiroqlarga javob beradi · Отвечает на звонки")
            .setContentIntent(open)
            .setOngoing(true)
            .addAction(Notification.Action.Builder(null, "Oʻchirish · Выключить", stop).build())
            .build()
        val type = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE else 0
        ServiceCompat.startForeground(this, NOTIFICATION_ID, notification, type)
    }

    companion object {
        private const val TAG = "AutoAnswer"
        private const val CHANNEL = "autoanswer"
        private const val NOTIFICATION_ID = 7
        const val ACTION_STOP = "uz.univoice.offline.STOP_AUTOANSWER"

        fun start(context: Context) {
            ContextCompat.startForegroundService(context, Intent(context, AutoAnswerService::class.java))
        }

        fun stop(context: Context) {
            context.startService(Intent(context, AutoAnswerService::class.java).setAction(ACTION_STOP))
        }
    }
}
