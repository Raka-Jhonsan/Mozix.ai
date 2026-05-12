package com.group45.smolvlm

import ai.onnxruntime.*
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.media.MediaMetadataRetriever
import android.net.Uri
import org.json.JSONObject
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.abs
import android.util.Log

class SmolVLMEngine(private val context: Context) {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()

    private val visionSession:  OrtSession
    private val mlpSession:     OrtSession
    private val decoderSession: OrtSession
    private val clipSession:    OrtSession

    // Initialized in init{} after decoderSession is ready
    private lateinit var decoderOutputNames:    List<String>
    private lateinit var decoderOutputIndexMap: Map<String, Int>

    companion object {
        const val NUM_FRAMES       = 16
        const val TOP_K            = 4
        const val IMAGE_SIZE       = 512
        const val MAX_NEW_TOKENS   = 80
        const val TOKENS_PER_IMAGE = 64
        const val EMBED_DIM        = 960

        private const val NUM_LAYERS = 32
        private const val NUM_HEADS  = 5
        private const val HEAD_DIM   = 64

        val MEAN = floatArrayOf(0.5f, 0.5f, 0.5f)
        val STD  = floatArrayOf(0.5f, 0.5f, 0.5f)

        private val BYTES_TO_UNICODE: Map<Int, Char> by lazy {
            val bs = (('!'.code..'~'.code) + ('¡'.code..'¬'.code) + ('®'.code..'ÿ'.code)).toMutableList()
            val cs = bs.toMutableList()
            var n = 0
            for (b in 0..255) {
                if (b !in bs) { bs.add(b); cs.add(256 + n); n++ }
            }
            bs.zip(cs.map { it.toChar() }).toMap()
        }
        private val UNICODE_TO_BYTES: Map<Char, Int> by lazy {
            BYTES_TO_UNICODE.entries.associateBy({ it.value }) { it.key }
        }

        private val TOKEN_PATTERN = Regex(
            """'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+"""
        )
    }

    // ── BPE Tokenizer ──────────────────────────────────────────────────────

    private data class TokenizerData(
        val vocab:       Map<String, Int>,
        val mergeRanks:  Map<Pair<String, String>, Int>,
        val addedTokens: Map<String, Int>
    )

    private val tokenizerData: TokenizerData by lazy { loadTokenizerData() }

    private fun loadTokenizerData(): TokenizerData {
        val json  = context.assets.open("tokenizer.json").bufferedReader().readText()
        val root  = JSONObject(json)
        val model = root.getJSONObject("model")

        val vocabObj = model.getJSONObject("vocab")
        val vocab    = mutableMapOf<String, Int>()
        vocabObj.keys().forEach { k -> vocab[k] = vocabObj.getInt(k) }

        val mergesArr  = model.getJSONArray("merges")
        val mergeRanks = mutableMapOf<Pair<String, String>, Int>()
        for (i in 0 until mergesArr.length()) {
            val pair = mergesArr.getJSONArray(i)
            mergeRanks[Pair(pair.getString(0), pair.getString(1))] = i
        }

        val addedArr    = root.getJSONArray("added_tokens")
        val addedTokens = mutableMapOf<String, Int>()
        for (i in 0 until addedArr.length()) {
            val obj = addedArr.getJSONObject(i)
            addedTokens[obj.getString("content")] = obj.getInt("id")
        }

        return TokenizerData(vocab, mergeRanks, addedTokens)
    }

    private fun bpe(token: String): List<String> {
        if (token.length <= 1) return listOf(token)
        var word = token.map { it.toString() }.toMutableList()
        while (true) {
            var bestRank = Int.MAX_VALUE
            var bestPair: Pair<String, String>? = null
            for (i in 0 until (word.size - 1)) {
                val pair = Pair(word[i], word[i + 1])
                val rank = tokenizerData.mergeRanks[pair] ?: Int.MAX_VALUE
                if (rank < bestRank) { bestRank = rank; bestPair = pair }
            }
            if (bestPair == null || bestRank == Int.MAX_VALUE) break
            val (first, second) = bestPair
            val newWord = mutableListOf<String>()
            var i = 0
            while (i < word.size) {
                if (i < word.size - 1 && word[i] == first && word[i + 1] == second) {
                    newWord.add(first + second); i += 2
                } else {
                    newWord.add(word[i]); i++
                }
            }
            word = newWord
        }
        return word
    }

    private fun encodeText(text: String): List<Int> {
        val tokens  = mutableListOf<Int>()
        for (match in TOKEN_PATTERN.findAll(text)) {
            val encoded = match.value.toByteArray(Charsets.UTF_8)
                .joinToString("") { BYTES_TO_UNICODE[it.toInt() and 0xFF]?.toString() ?: "?" }
            for (t in bpe(encoded)) {
                tokenizerData.vocab[t]?.let { tokens.add(it) }
            }
        }
        return tokens
    }

    private fun decodeTokens(ids: List<Long>): String {
        val sb = StringBuilder()
        for (id in ids) {
            val token = tokenizerData.vocab.entries.firstOrNull { it.value == id.toInt() }?.key
                ?: tokenizerData.addedTokens.entries.firstOrNull { it.value == id.toInt() }?.key
                ?: continue

            // Allow essential tokens, but skip meta-structure tokens
            if (token.startsWith("<") && token.endsWith(">")) {
                if (token != "<|endoftext|>" && !token.contains("utterance")) {
                    continue
                }
            }

            val bytes = token.mapNotNull { c -> UNICODE_TO_BYTES[c] }.map { it.toByte() }.toByteArray()
            try {
                val decoded = String(bytes, Charsets.UTF_8)
                sb.append(decoded)
            } catch (_: Exception) {
                sb.append(token)
            }
        }
        return sb.toString().trim()
    }

    // ── Special token IDs ──────────────────────────────────────────────────

    private val eosIds: Set<Long> by lazy {
        setOf(
            tokenizerData.addedTokens["<|endoftext|>"]?.toLong()      ?: 0L,
            tokenizerData.addedTokens["<|im_end|>"]?.toLong()         ?: 2L,
            tokenizerData.addedTokens["<end_of_utterance>"]?.toLong() ?: 49279L
        )
    }
    private val imStartId:   Long by lazy { tokenizerData.addedTokens["<|im_start|>"]?.toLong()            ?: 1L }
    private val imEndId:     Long by lazy { tokenizerData.addedTokens["<|im_end|>"]?.toLong()              ?: 2L }
    private val fakeTokenId: Long by lazy { tokenizerData.addedTokens["<fake_token_around_image>"]?.toLong() ?: 49189L }
    private val newlineId:   Long by lazy { encodeText("\n").firstOrNull()?.toLong() ?: 198L }

    // ── MappedByteBuffer for embed_tokens.bin ─────────────────────────────

    private val embedTokensMapped: java.nio.MappedByteBuffer by lazy {
        val cacheFile = java.io.File(context.cacheDir, "embed_tokens.bin")
        if (!cacheFile.exists()) {
            context.assets.open("embed_tokens.bin").use { it.copyTo(cacheFile.outputStream(), 65536) }
        }
        java.io.RandomAccessFile(cacheFile, "r").channel.use { ch ->
            ch.map(FileChannel.MapMode.READ_ONLY, 0L, ch.size())
        }
    }

    private fun getTokenEmbedding(tokenId: Long): FloatArray {
        val result = FloatArray(EMBED_DIM)
        val slice  = embedTokensMapped.duplicate().order(ByteOrder.LITTLE_ENDIAN)
        slice.position(tokenId.toInt() * EMBED_DIM * 4)
        slice.asFloatBuffer().get(result)
        return result
    }

    // ── Init ───────────────────────────────────────────────────────────────

    init {
        val opts = OrtSession.SessionOptions().apply {
            try { addNnapi() } catch (_: Exception) { /* ignore */ }
            setIntraOpNumThreads(2)
        }
        clipSession    = loadModel("clip_vision.onnx",             opts)
        visionSession  = loadModel("vision_encoder_q4.onnx",       opts)
        mlpSession     = loadModel("frame_selector.onnx",          opts)
        decoderSession = loadModel("decoder_model_merged_q4.onnx", opts)

        // Discover output names now that decoderSession exists
        decoderOutputNames    = decoderSession.outputNames.toList()
        decoderOutputIndexMap = decoderOutputNames.mapIndexed { idx, name -> name to idx }.toMap()
        Log.d("SmolVLM", "Decoder outputs (${decoderOutputNames.size}): $decoderOutputNames")
    }

    private fun loadModel(assetName: String, opts: OrtSession.SessionOptions): OrtSession {
        val cacheFile = java.io.File(context.cacheDir, assetName)
        if (cacheFile.exists()) cacheFile.delete()
        context.assets.open(assetName).use { it.copyTo(cacheFile.outputStream(), 8192) }
        return env.createSession(cacheFile.absolutePath, opts)
    }

    // Dynamic mode — called by ViewModel before runOnVideo
    private var currentNumFrames  = NUM_FRAMES
    private var currentTopK       = TOP_K
    private var currentMaxTokens  = MAX_NEW_TOKENS

    fun setMode(numFrames: Int, topK: Int, maxTokens: Int = MAX_NEW_TOKENS) {
        currentNumFrames = numFrames
        currentTopK      = topK
        currentMaxTokens = maxTokens
    }

    @Volatile
    private var shouldStopGeneration = false
    fun requestStop() { shouldStopGeneration = true }

    // ── Public API ─────────────────────────────────────────────────────────

    fun runOnVideo(
        videoUri:    Uri,
        question:    String,
        history:     List<Pair<String, String>> = emptyList(),
        instruction: String = "",
        isNarrator:  Boolean = false,
        temperature: Float = 0.2f,
        topP:        Float = 0.9f,
        onStatus:    (String) -> Unit = {},
        onToken:     (String) -> Unit = {}
    ): InferenceResult {
        val t0 = System.currentTimeMillis()
        val frames = extractFrames(videoUri)
        if (frames.isEmpty()) return InferenceResult.Error("Could not extract frames")
        return runOnFrameList(
            frames, question, history, instruction, isNarrator,
            startTimeMs = t0, temperature = temperature, topP = topP,
            onStatus = onStatus, onToken = onToken
        )
    }

    fun runOnFrameList(
        frames:      List<Bitmap>,
        question:    String,
        history:     List<Pair<String, String>> = emptyList(),
        instruction: String = "",
        isNarrator:  Boolean = false,
        startTimeMs: Long = System.currentTimeMillis(),
        temperature: Float = 0.2f,
        topP:        Float = 0.9f,
        onStatus:    (String) -> Unit = {},
        onToken:     (String) -> Unit = {}
    ): InferenceResult {
        shouldStopGeneration = false
        onStatus("Extracting features…")
        val embeddings   = frames.map { getClipEmbedding(it) }
        val motionScores = computeMotionScores(frames)
        val features     = embeddings.mapIndexed { i, emb ->
            FloatArray(513).also { f -> emb.copyInto(f); f[512] = motionScores[i] }
        }

        onStatus("Selecting key frames…")
        val topKIdx        = selectFrames(features)
        val selectedFrames = topKIdx.map { frames[it] }

        onStatus("Encoding vision…")
        val visionEmbeds   = selectedFrames.map { getVisionEmbedding(it) }

        onStatus("Thinking…")
        val answer = decodeAnswer(
            visionEmbeds, question, history, instruction, isNarrator,
            temperature = temperature, topP = topP, onToken = onToken
        )

        // Recycle frames to free memory
        frames.forEach { it.recycle() }

        return InferenceResult.Success(
            answer         = answer,
            selectedFrames = topKIdx,
            totalFrames    = frames.size,
            elapsedMs      = System.currentTimeMillis() - startTimeMs
        )
    }

    fun runOnImage(
        bitmap:      Bitmap,
        question:    String,
        history:     List<Pair<String, String>> = emptyList(),
        instruction: String = "",
        isNarrator:  Boolean = false,
        temperature: Float = 0.1f,
        topP:        Float = 0.9f,
        onStatus:    (String) -> Unit = {},
        onToken:     (String) -> Unit = {}
    ): InferenceResult {
        shouldStopGeneration = false
        val t0           = System.currentTimeMillis()
        onStatus("Encoding vision…")
        val scaled       = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
        val visionEmbeds = listOf(getVisionEmbedding(scaled))
        onStatus("Thinking…")
        val answer = decodeAnswer(
            visionEmbeds, question, history, instruction, isNarrator,
            temperature = temperature, topP = topP, onToken = onToken
        )
        return InferenceResult.Success(
            answer         = answer,
            selectedFrames = listOf(0),
            totalFrames    = 1,
            elapsedMs      = System.currentTimeMillis() - t0
        )
    }

    // ── Frame Extraction ───────────────────────────────────────────────────

    private fun extractFrames(uri: Uri): List<Bitmap> {
        val retriever = MediaMetadataRetriever()
        return try {
            retriever.setDataSource(context, uri)
            val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLong() ?: return emptyList()
            val intervalUs = durationMs * 1000L / currentNumFrames
            (0 until currentNumFrames).mapNotNull { i ->
                val fullBmp = retriever.getFrameAtTime(i * intervalUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                fullBmp?.let {
                    val scaled = Bitmap.createScaledBitmap(it, IMAGE_SIZE, IMAGE_SIZE, true)
                    if (it != scaled) it.recycle()
                    scaled
                }
            }
        } catch (_: Exception) { emptyList() } finally { retriever.release() }
    }

    // ── CLIP Embedding ─────────────────────────────────────────────────────

    private fun getClipEmbedding(bitmap: Bitmap): FloatArray {
        val scaled = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val data   = FloatArray(3 * 224 * 224)
        for (y in 0 until 224) for (x in 0 until 224) {
            val px   = scaled.getPixel(x, y)
            val base = y * 224 + x
            data[0 * 224 * 224 + base] = ((Color.red(px)   / 255f) - 0.485f) / 0.229f
            data[1 * 224 * 224 + base] = ((Color.green(px) / 255f) - 0.456f) / 0.224f
            data[2 * 224 * 224 + base] = ((Color.blue(px)  / 255f) - 0.406f) / 0.225f
        }
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(data), longArrayOf(1L, 3L, 224L, 224L))
        val output = clipSession.run(mapOf("pixel_values" to tensor))
        val result = extractFloatArray(output[0].value)

        // Cleanup
        tensor.close(); runCatching { output.close() }
        if (scaled != bitmap) scaled.recycle()

        return result
    }

    // ── Vision Encoder ─────────────────────────────────────────────────────

    private fun getVisionEmbedding(bitmap: Bitmap): FloatArray {
        // Ensure bitmap is the correct size for the model
        val scaled = if (bitmap.width != IMAGE_SIZE || bitmap.height != IMAGE_SIZE) {
            Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
        } else bitmap

        val pixelData = FloatArray(3 * IMAGE_SIZE * IMAGE_SIZE)
        for (y in 0 until IMAGE_SIZE) for (x in 0 until IMAGE_SIZE) {
            val px      = scaled.getPixel(x, y)
            val baseIdx = y * IMAGE_SIZE + x
            pixelData[0 * IMAGE_SIZE * IMAGE_SIZE + baseIdx] = ((Color.red(px)   / 255f) - MEAN[0]) / STD[0]
            pixelData[1 * IMAGE_SIZE * IMAGE_SIZE + baseIdx] = ((Color.green(px) / 255f) - MEAN[1]) / STD[1]
            pixelData[2 * IMAGE_SIZE * IMAGE_SIZE + baseIdx] = ((Color.blue(px)  / 255f) - MEAN[2]) / STD[2]
        }
        val pixelShape  = longArrayOf(1L, 1L, 3L, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong())
        val pixelTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(pixelData), pixelShape)
        val maskValues  = Array(1) { Array(1) { Array(IMAGE_SIZE) { BooleanArray(IMAGE_SIZE) { true } } } }
        val maskTensor  = OnnxTensor.createTensor(env, maskValues)
        val output      = visionSession.run(mapOf(
            "pixel_values"         to pixelTensor,
            "pixel_attention_mask" to maskTensor
        ))
        val result = flattenVisionOutput(output[0].value)

        // Cleanup
        pixelTensor.close(); maskTensor.close(); runCatching { output.close() }
        if (scaled != bitmap) scaled.recycle()

        return result
    }

    private fun flattenVisionOutput(value: Any?): FloatArray {
        val flatList = mutableListOf<Float>()
        fun traverse(v: Any?) {
            when (v) {
                is FloatArray -> v.forEach { flatList.add(it) }
                is Array<*>   -> v.forEach { traverse(it) }
                is Float      -> flatList.add(v)
            }
        }
        traverse(value)
        val result = flatList.toFloatArray()

        // Sanity check for token count (expecting TOKENS_PER_IMAGE * EMBED_DIM)
        if (result.size != TOKENS_PER_IMAGE * EMBED_DIM) {
            Log.w("SmolVLM", "Vision output size mismatch: expected ${TOKENS_PER_IMAGE * EMBED_DIM}, got ${result.size}")
        }

        return result
    }

    private fun extractFloatArray(value: Any?): FloatArray {
        return when (value) {
            is Array<*> -> {
                val inner = value[0]
                when (inner) {
                    is Array<*>   -> (inner[0] as? FloatArray) ?: FloatArray(512)
                    is FloatArray -> inner
                    else          -> FloatArray(512)
                }
            }
            is FloatArray -> value
            else          -> FloatArray(512)
        }
    }

    // ── Motion Scores ──────────────────────────────────────────────────────

    private fun computeMotionScores(frames: List<Bitmap>): FloatArray {
        val scores = FloatArray(frames.size)
        for (i in 1 until frames.size) {
            val w = minOf(frames[i].width, 64); val h = minOf(frames[i].height, 64)
            val a = Bitmap.createScaledBitmap(frames[i - 1], w, h, false)
            val b = Bitmap.createScaledBitmap(frames[i],     w, h, false)
            var diff = 0L
            for (y in 0 until h) for (x in 0 until w) {
                val pa = a.getPixel(x, y); val pb = b.getPixel(x, y)
                diff += abs(Color.red(pa)   - Color.red(pb)).toLong()
                diff += abs(Color.green(pa) - Color.green(pb)).toLong()
                diff += abs(Color.blue(pa)  - Color.blue(pb)).toLong()
            }
            scores[i] = diff.toFloat() / (w * h * 3)
            a.recycle(); b.recycle()
        }
        val max = scores.maxOrNull()?.takeIf { it > 0f } ?: 1f
        return FloatArray(scores.size) { scores[it] / max }
    }

    // ── MLP Frame Selector ─────────────────────────────────────────────────

    private fun selectFrames(features: List<FloatArray>): List<Int> {
        val n      = features.size
        val flat   = FloatArray(n * 513) { i -> features[i / 513][i % 513] }
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(flat), longArrayOf(n.toLong(), 513L))
        val output = mlpSession.run(mapOf("features" to tensor))
        val scores = (output[0].value as FloatArray).copyOf()
        tensor.close(); runCatching { output.close() }

        // Quality Enhancement: Effective Diversity Filter
        // We ensure frames are at least ~10% of the video duration apart
        // to avoid near-identical consecutive frames being selected.
        val indexedScores = scores.mapIndexed { i, s -> i to s }.sortedByDescending { it.second }
        val selected = mutableListOf<Int>()
        val minDistance = maxOf(1, n / 10)

        for (pair in indexedScores) {
            if (selected.size >= currentTopK) break
            if (selected.none { abs(it - pair.first) < minDistance }) {
                selected.add(pair.first)
            }
        }

        // Fallback if diversity filter is too strict
        if (selected.size < minOf(currentTopK, n)) {
            for (pair in indexedScores) {
                if (selected.size >= currentTopK) break
                if (!selected.contains(pair.first)) selected.add(pair.first)
            }
        }

        return selected.sorted()
    }

    // ── Cleanup ────────────────────────────────────────────────────────────

    private data class KVCache(
        val keys:   List<FloatArray>,  // NUM_LAYERS entries, shape [1, NUM_HEADS, seqLen, HEAD_DIM] flattened
        val values: List<FloatArray>,
        val seqLen: Int
    )

    // ── Extract KV cache from decoder output ──────────────────────────────

    private fun extractKVCache(output: OrtSession.Result, newSeqLen: Int): KVCache {
        val keys   = mutableListOf<FloatArray>()
        val values = mutableListOf<FloatArray>()

        for (i in 0 until NUM_LAYERS) {
            // Faster lookup using the 'present.i.key' naming convention seen in logcat
            val keyName = "present.$i.key"
            val valName = "present.$i.value"

            val keyIdx = decoderOutputIndexMap[keyName]
            val valIdx = decoderOutputIndexMap[valName]

            if (keyIdx != null && valIdx != null) {
                keys.add(tensorToFloatArray(output[keyIdx].value))
                values.add(tensorToFloatArray(output[valIdx].value))
            } else {
                // Fallback to fuzzy search if exact match fails
                val fuzzyKeyIdx = decoderOutputIndexMap.entries
                    .firstOrNull { (name, _) -> name.contains(".$i.key") }?.value
                val fuzzyValIdx = decoderOutputIndexMap.entries
                    .firstOrNull { (name, _) -> name.contains(".$i.value") }?.value

                if (fuzzyKeyIdx != null && fuzzyValIdx != null) {
                    keys.add(tensorToFloatArray(output[fuzzyKeyIdx].value))
                    values.add(tensorToFloatArray(output[fuzzyValIdx].value))
                } else {
                    keys.add(FloatArray(0))
                    values.add(FloatArray(0))
                }
            }
        }
        return KVCache(keys, values, newSeqLen)
    }

    private fun tensorToFloatArray(value: Any?): FloatArray {
        return when (value) {
            is FloatArray -> value
            is Array<*>   -> {
                val batch = value[0]
                if (batch is Array<*>) {
                    val heads = batch.size
                    val first = batch[0]
                    if (first is Array<*>) {
                        val seqLen  = first.size
                        val headDim = (first[0] as? FloatArray)?.size ?: HEAD_DIM
                        val flat    = FloatArray(heads * seqLen * headDim)
                        var offset  = 0
                        for (h in 0 until heads) {
                            val head = batch[h] as Array<*>
                            for (s in 0 until seqLen) {
                                val row = head[s] as FloatArray
                                row.copyInto(flat, offset); offset += headDim
                            }
                        }
                        flat
                    } else FloatArray(0)
                } else FloatArray(0)
            }
            else -> FloatArray(0)
        }
    }

    // ── Build past-KV input tensors ────────────────────────────────────────

    private fun buildKVTensors(cache: KVCache?): Map<String, OnnxTensor> {
        val result     = mutableMapOf<String, OnnxTensor>()
        val emptyShape = longArrayOf(1L, NUM_HEADS.toLong(), 0L, HEAD_DIM.toLong())
        for (i in 0 until NUM_LAYERS) {
            if (cache != null && cache.keys[i].isNotEmpty()) {
                // Derive actual seqLen from the array size: keys[i].size = NUM_HEADS * seqLen * HEAD_DIM
                val actualSeqLen = cache.keys[i].size / (NUM_HEADS * HEAD_DIM)
                val shape = longArrayOf(1L, NUM_HEADS.toLong(), actualSeqLen.toLong(), HEAD_DIM.toLong())
                result["past_key_values.$i.key"]   = OnnxTensor.createTensor(env, FloatBuffer.wrap(cache.keys[i]),   shape)
                result["past_key_values.$i.value"] = OnnxTensor.createTensor(env, FloatBuffer.wrap(cache.values[i]), shape)
            } else {
                result["past_key_values.$i.key"]   = OnnxTensor.createTensor(env, FloatBuffer.wrap(FloatArray(0)), emptyShape)
                result["past_key_values.$i.value"] = OnnxTensor.createTensor(env, FloatBuffer.wrap(FloatArray(0)), emptyShape)
            }
        }
        return result
    }

    // ── Single decoder step (prefill or decode) ────────────────────────────

    private data class DecoderStepResult(val logits: FloatArray, val cache: KVCache)

    private fun runDecoderStep(
        embedData: FloatArray,
        seqLen:    Int,
        startPos:  Int,
        pastCache: KVCache?
    ): DecoderStepResult {
        val pastSeqLen = pastCache?.seqLen ?: 0
        val totalLen   = pastSeqLen + seqLen  // attention mask length

        val embedTensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(embedData),
            longArrayOf(1L, seqLen.toLong(), EMBED_DIM.toLong())
        )
        val maskTensor = OnnxTensor.createTensor(
            env, LongBuffer.wrap(LongArray(totalLen) { 1L }),
            longArrayOf(1L, totalLen.toLong())
        )
        val posTensor = OnnxTensor.createTensor(
            env, LongBuffer.wrap(LongArray(seqLen) { (startPos + it).toLong() }),
            longArrayOf(1L, seqLen.toLong())
        )

        val inputs = mutableMapOf<String, OnnxTensor>(
            "inputs_embeds"  to embedTensor,
            "attention_mask" to maskTensor,
            "position_ids"   to posTensor,
        )
        inputs.putAll(buildKVTensors(pastCache))

        val output   = decoderSession.run(inputs)
        val logits   = (output[0].value as Array<Array<FloatArray>>)[0].last().copyOf()
        val newCache = extractKVCache(output, totalLen)

        // Close all input tensors to free native memory
        inputs.values.forEach { runCatching { it.close() } }
        runCatching { output.close() }

        return DecoderStepResult(logits, newCache)
    }

    // ── Greedy decode loop ─────────────────────────────────────────────────

    // ── Sampling Decoder ───────────────────────────────────────────────────

    private fun sampleToken(logits: FloatArray, temperature: Float = 0.0f, topK: Int = 40, topP: Float = 0.9f): Int {
        if (temperature <= 0.0f) {
            var maxIdx = 0
            var maxVal = -Float.MAX_VALUE
            for (i in logits.indices) {
                if (logits[i] > maxVal) { maxVal = logits[i]; maxIdx = i }
            }
            return maxIdx
        }

        val softmax = FloatArray(logits.size)
        var maxLogit = -Float.MAX_VALUE
        for (l in logits) if (l > maxLogit) maxLogit = l

        var sum = 0f
        for (i in logits.indices) {
            softmax[i] = kotlin.math.exp(((logits[i] - maxLogit) / temperature).toDouble()).toFloat()
            sum += softmax[i]
        }
        for (i in softmax.indices) softmax[i] /= sum

        // Quality Enhancement: Combined Top-K and Top-P filtering
        val indexed = softmax.mapIndexed { i, v -> i to v }.sortedByDescending { it.second }

        // Apply Top-P (Nucleus)
        var cumulativeP = 0f
        val nucleus = mutableListOf<Pair<Int, Float>>()
        for (pair in indexed) {
            nucleus.add(pair)
            cumulativeP += pair.second
            if (cumulativeP >= topP) break
        }

        // Take min of Top-K and Nucleus
        val finalCandidates = nucleus.take(topK)
        val finalSum = finalCandidates.sumOf { it.second.toDouble() }.toFloat()

        val r = (0..1000).random() / 1000f * finalSum
        var acc = 0f
        for (pair in finalCandidates) {
            acc += pair.second
            if (acc >= r) return pair.first
        }
        return finalCandidates.first().first
    }

    private fun decodeAnswer(
        visionEmbeds: List<FloatArray>,
        question:     String,
        history:      List<Pair<String, String>> = emptyList(),
        instruction:  String = "",
        isNarrator:   Boolean = false,
        temperature:  Float = 0.2f,
        topP:         Float = 0.9f,
        onToken:      (String) -> Unit
    ): String {
        return try {
            val promptIds    = buildChatTokenSequence(visionEmbeds, question, history, instruction, isNarrator)
            val promptEmbeds = idsToEmbeds(promptIds)
            val seqLen       = promptEmbeds.size / EMBED_DIM
            Log.d("SmolVLM", "Prompt length: $seqLen tokens (${visionEmbeds.size} images)")

            val prefillResult = runDecoderStep(promptEmbeds, seqLen, 0, null)
            var cache         = prefillResult.cache

            val generated  = mutableListOf<Long>()
            val penaltyIds = mutableListOf<Long>()

            var nextToken = sampleToken(prefillResult.logits, temperature = temperature, topP = topP).toLong()
            var pos       = seqLen

            var stepCount = 0
            while (stepCount < currentMaxTokens && nextToken !in eosIds && !shouldStopGeneration) {
                generated.add(nextToken)
                penaltyIds.add(nextToken)

                val currentText = decodeTokens(listOf(nextToken))
                onToken(currentText)

                // Quality Enhancement: Sentence guard prevents runaway repetition loops
                val fullSoFar = decodeTokens(generated)
                val sentences = fullSoFar.count { it == '.' || it == '!' || it == '?' }
                if (sentences >= 5) break // Allow slightly longer, more detailed descriptions

                val step  = runDecoderStep(getTokenEmbedding(nextToken), 1, pos, cache)
                cache     = step.cache

                // Quality Enhancement: Balanced Presence + Frequency Penalty
                // We reduce these to avoid making the model speak unnaturally
                val logits = step.logits
                val counts = penaltyIds.groupingBy { it }.eachCount()
                for ((id, count) in counts) {
                    val idx = id.toInt()
                    if (idx in logits.indices) {
                        // Presence penalty (0.6) + Frequency penalty (0.2 * count)
                        logits[idx] -= (0.6f + count * 0.2f)
                    }
                }

                nextToken = sampleToken(logits, temperature = temperature, topP = topP).toLong()
                pos++
                stepCount++
            }

            decodeTokens(generated).ifEmpty { "The AI could not form a response." }
        } catch (e: Exception) {
            Log.e("SmolVLM", "decodeAnswer error", e)
            "Error: ${e.message}"
        }
    }

    private fun buildChatTokenSequence(visionEmbeds: List<FloatArray>, question: String, history: List<Pair<String, String>> = emptyList(), instruction: String = "", isNarrator: Boolean = false): List<Any> {
        val sequence = mutableListOf<Any>()

        fun nl() = sequence.add(newlineId)
        fun tok(t: String) = encodeText(t).forEach { sequence.add(it.toLong()) }
        fun special(id: Long) = sequence.add(id)

        // System turn: Precise instruction for high quality
        special(imStartId); tok("system"); nl()
        if (isNarrator) {
            tok("You are a direct visual narrator. Describe the scene accurately based ONLY on what you see. Be factual and do not speculate.")
        } else {
            val type = if (visionEmbeds.size > 1) "video" else "image"
            tok("You are a visual assistant. Answer the user question based on the provided $type. Only report what is clearly visible. Be direct and concise.")
        }
        special(imEndId); nl()

        // User turn
        special(imStartId); tok("user"); nl()
        visionEmbeds.forEach { vEmbed ->
            special(fakeTokenId)
            val numTokens = vEmbed.size / EMBED_DIM
            for (t in 0 until numTokens) {
                val slice = FloatArray(EMBED_DIM)
                vEmbed.copyInto(slice, 0, t * EMBED_DIM, (t + 1) * EMBED_DIM)
                sequence.add(slice)
            }
            special(fakeTokenId)
        }
        nl()

        // History: Hard cap at 2 turns to maintain focus
        val safeHistory = history.takeLast(2)
        if (safeHistory.isEmpty()) {
            var q = question
            if (instruction.isNotEmpty()) q = "$q ($instruction)"
            tok(q)
            special(imEndId); nl()
        } else {
            for (i in safeHistory.indices) {
                val (q, a) = safeHistory[i]
                if (i == 0) {
                    tok(q)
                    special(imEndId); nl()
                } else {
                    special(imStartId); tok("user"); nl(); tok(q); special(imEndId); nl()
                }
                special(imStartId); tok("assistant"); nl(); tok(a); special(imEndId); nl()
            }

            var q = question
            if (instruction.isNotEmpty()) q = "$q ($instruction)"
            special(imStartId); tok("user"); nl(); tok(q); special(imEndId); nl()
        }

        special(imStartId); tok("assistant"); nl()

        return sequence
    }

    private fun idsToEmbeds(sequence: List<Any>): FloatArray {
        val totalTokens = sequence.size
        val result = FloatArray(totalTokens * EMBED_DIM)
        sequence.forEachIndexed { i, item ->
            val emb = when (item) {
                is Long -> getTokenEmbedding(item)
                is FloatArray -> item
                else -> FloatArray(EMBED_DIM)
            }
            emb.copyInto(result, i * EMBED_DIM)
        }
        return result
    }


    // ── Cleanup ────────────────────────────────────────────────────────────

    fun close() {
        clipSession.close(); visionSession.close()
        mlpSession.close();  decoderSession.close()
        env.close()
    }
}



sealed class InferenceResult {
    data class Success(
        val answer:         String,
        val selectedFrames: List<Int>,
        val totalFrames:    Int,
        val elapsedMs:      Long
    ) : InferenceResult()
    data class Error(val message: String) : InferenceResult()
}
