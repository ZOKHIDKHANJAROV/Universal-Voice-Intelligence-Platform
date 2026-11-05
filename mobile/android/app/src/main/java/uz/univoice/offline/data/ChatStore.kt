package uz.univoice.offline.data

import android.content.ContentValues
import android.content.Context
import android.database.Cursor
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import java.io.File

enum class Role { CALLER, BOT, SYSTEM }

data class Message(
    val id: Long,
    val conversationId: Long,
    val role: Role,
    val text: String,
    val language: String?,
    /** Caller turns: their recording. */
    val audioPath: String?,
    /** Bot turns: the prompt manifest key of the answer it played. */
    val audioKey: String?,
    val scenarioId: String?,
    val confidence: Double?,
    /** Caller turns: how long speech recognition took. */
    val sttMillis: Long?,
    val offerOperator: Boolean,
    val createdAt: Long,
)

data class Conversation(
    val id: Long,
    val startedAt: Long,
    val turns: Int,
    val preview: String,
    /** An answered phone call: the caller's number, or "" when Android did not give it. Null for a chat. */
    val caller: String? = null,
    /** The caller's problem needs a person: call them back. */
    val callback: Boolean = false,
)

/** Conversations and their messages, on the phone only. */
class ChatStore private constructor(context: Context) : SQLiteOpenHelper(context, "chat.db", null, 2) {
    val audioDir = File(context.filesDir, "recordings").apply { mkdirs() }

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL(
            """CREATE TABLE conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at INTEGER NOT NULL,
                caller TEXT,
                callback INTEGER NOT NULL DEFAULT 0)""",
        )
        db.execSQL(
            """CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                language TEXT,
                audio_path TEXT,
                audio_key TEXT,
                scenario_id TEXT,
                confidence REAL,
                stt_ms INTEGER,
                offer_operator INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL)""",
        )
        db.execSQL("CREATE INDEX messages_conversation ON messages(conversation_id, id)")
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        if (oldVersion < 2) {
            db.execSQL("ALTER TABLE conversations ADD COLUMN caller TEXT")
            db.execSQL("ALTER TABLE conversations ADD COLUMN callback INTEGER NOT NULL DEFAULT 0")
        }
    }

    override fun onConfigure(db: SQLiteDatabase) = db.setForeignKeyConstraintsEnabled(true)

    fun newConversation(): Long = writableDatabase.insert(
        "conversations", null, ContentValues().apply { put("started_at", System.currentTimeMillis()) },
    )

    /** A conversation for an answered call; [caller] is "" when the number is unknown. */
    fun newCall(caller: String): Long = writableDatabase.insert(
        "conversations", null,
        ContentValues().apply {
            put("started_at", System.currentTimeMillis())
            put("caller", caller)
        },
    )

    fun markCallback(conversationId: Long) {
        writableDatabase.update(
            "conversations", ContentValues().apply { put("callback", 1) }, "id = ?", arrayOf(conversationId.toString()),
        )
    }

    fun conversation(id: Long): Conversation? = conversations().firstOrNull { it.id == id }

    fun add(
        conversationId: Long,
        role: Role,
        text: String,
        language: String? = null,
        audioPath: String? = null,
        audioKey: String? = null,
        scenarioId: String? = null,
        confidence: Double? = null,
        sttMillis: Long? = null,
        offerOperator: Boolean = false,
    ): Long = writableDatabase.insert(
        "messages", null,
        ContentValues().apply {
            put("conversation_id", conversationId)
            put("role", role.name)
            put("text", text)
            put("language", language)
            put("audio_path", audioPath)
            put("audio_key", audioKey)
            put("scenario_id", scenarioId)
            put("confidence", confidence)
            put("stt_ms", sttMillis)
            put("offer_operator", if (offerOperator) 1 else 0)
            put("created_at", System.currentTimeMillis())
        },
    )

    fun messages(conversationId: Long): List<Message> = readableDatabase.rawQuery(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id", arrayOf(conversationId.toString()),
    ).use { cursor -> generateSequence { if (cursor.moveToNext()) cursor.toMessage() else null }.toList() }

    /** Every caller turn that has a recording, newest first. */
    fun callerRecordings(limit: Int = Int.MAX_VALUE): List<Message> = readableDatabase.rawQuery(
        "SELECT * FROM messages WHERE role = 'CALLER' AND audio_path IS NOT NULL ORDER BY id DESC LIMIT ?",
        arrayOf(limit.toString()),
    ).use { cursor -> generateSequence { if (cursor.moveToNext()) cursor.toMessage() else null }.toList() }

    /** The bot turn that answered a caller turn, if any. */
    fun replyTo(message: Message): Message? = readableDatabase.rawQuery(
        "SELECT * FROM messages WHERE conversation_id = ? AND id > ? AND role = 'BOT' ORDER BY id LIMIT 1",
        arrayOf(message.conversationId.toString(), message.id.toString()),
    ).use { if (it.moveToNext()) it.toMessage() else null }

    fun conversations(): List<Conversation> = readableDatabase.rawQuery(
        """SELECT c.id, c.started_at,
                  (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id AND m.role = 'CALLER'),
                  (SELECT m.text FROM messages m WHERE m.conversation_id = c.id AND m.role = 'CALLER' ORDER BY m.id LIMIT 1),
                  c.caller, c.callback
           FROM conversations c ORDER BY c.id DESC""",
        null,
    ).use { cursor ->
        generateSequence {
            if (!cursor.moveToNext()) null
            else Conversation(
                cursor.getLong(0), cursor.getLong(1), cursor.getInt(2), cursor.getString(3) ?: "",
                caller = cursor.getString(4), callback = cursor.getInt(5) == 1,
            )
        }.toList()
    }

    fun deleteConversation(id: Long) {
        messages(id).mapNotNull { it.audioPath }.forEach { File(it).delete() }
        writableDatabase.delete("conversations", "id = ?", arrayOf(id.toString()))
    }

    private fun Cursor.toMessage() = Message(
        id = getLong(getColumnIndexOrThrow("id")),
        conversationId = getLong(getColumnIndexOrThrow("conversation_id")),
        role = Role.valueOf(getString(getColumnIndexOrThrow("role"))),
        text = getString(getColumnIndexOrThrow("text")),
        language = stringOrNull("language"),
        audioPath = stringOrNull("audio_path"),
        audioKey = stringOrNull("audio_key"),
        scenarioId = stringOrNull("scenario_id"),
        confidence = getColumnIndexOrThrow("confidence").let { if (isNull(it)) null else getDouble(it) },
        sttMillis = getColumnIndexOrThrow("stt_ms").let { if (isNull(it)) null else getLong(it) },
        offerOperator = getInt(getColumnIndexOrThrow("offer_operator")) == 1,
        createdAt = getLong(getColumnIndexOrThrow("created_at")),
    )

    private fun Cursor.stringOrNull(column: String): String? =
        getColumnIndexOrThrow(column).let { if (isNull(it)) null else getString(it) }

    companion object {
        @Volatile private var instance: ChatStore? = null

        /** One store for the chat screen and the call answering service. */
        fun get(context: Context): ChatStore = instance ?: synchronized(this) {
            instance ?: ChatStore(context.applicationContext).also { instance = it }
        }
    }
}
