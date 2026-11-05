package uz.univoice.offline.call

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import androidx.core.content.ContextCompat
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update

/** What the answering service is doing, for the screen. One per process. */
object CallState {
    data class Status(
        /** The service is on and waiting for calls. */
        val running: Boolean = false,
        val inCall: Boolean = false,
        /** The conversation of the call in progress (or the last one). */
        val conversationId: Long? = null,
        /** Grows with every message of the call, so the screen reloads it. */
        val revision: Int = 0,
        val caller: String = "",
        /** Why the last attempt failed, for the screen to show once. */
        val error: String? = null,
    )

    private val _status = MutableStateFlow(Status())
    val status: StateFlow<Status> = _status

    fun serviceStarted() = _status.update { it.copy(running = true, error = null) }

    fun serviceStopped() = _status.update { it.copy(running = false, inCall = false) }

    fun callStarted(conversationId: Long, caller: String) =
        _status.update { it.copy(inCall = true, conversationId = conversationId, caller = caller, revision = it.revision + 1) }

    fun callChanged() = _status.update { it.copy(revision = it.revision + 1) }

    fun callEnded() = _status.update { it.copy(inCall = false, revision = it.revision + 1) }

    fun failed(message: String) = _status.update { it.copy(error = message) }

    fun errorShown() = _status.update { it.copy(error = null) }
}

/** What the phone must allow for the answering machine to work. */
object CallPermissions {
    /** Hanging up (TelecomManager.endCall) needs Android 9. */
    val supported: Boolean get() = Build.VERSION.SDK_INT >= Build.VERSION_CODES.P

    /** Without these it cannot answer, hear or hang up. */
    private fun required(): List<String> = buildList {
        add(Manifest.permission.RECORD_AUDIO)
        add(Manifest.permission.READ_PHONE_STATE)
        add(Manifest.permission.ANSWER_PHONE_CALLS)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) add(Manifest.permission.POST_NOTIFICATIONS)
    }

    /** Everything it asks for: also the call log, which is the only way to learn the caller's number. */
    fun requested(): Array<String> = (required() + Manifest.permission.READ_CALL_LOG).toTypedArray()

    fun missing(context: Context): List<String> = required().filter {
        ContextCompat.checkSelfPermission(context, it) != PackageManager.PERMISSION_GRANTED
    }
}
