package ai.duan.demo

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import java.io.File

object NativeBridge {
    init {
        System.loadLibrary("mistralrs_android")
    }

    external fun loadModel(modelDir: String, ggufFile: String, tokenizerJson: String?): String
    external fun complete(prompt: String, maxTokens: Int, temperature: Float): String
    external fun unloadModel()
}

class MainActivity : AppCompatActivity() {

    private lateinit var tvStatus: TextView
    private lateinit var tvResult: TextView
    private lateinit var tvLog: TextView
    private lateinit var etPrompt: EditText
    private lateinit var btnLoad: Button
    private lateinit var btnRun: Button

    private var modelPath: String? = null
    private var loaded = false
    private var busy = false
    private val logLines = StringBuilder()
    private var sessionMs = 0L

    private val pickModel =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
            if (uri != null) copyToFilesDir(uri)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tvStatus = findViewById(R.id.tvStatus)
        tvResult = findViewById(R.id.tvResult)
        tvLog = findViewById(R.id.tvLog)
        etPrompt = findViewById(R.id.etPrompt)
        btnLoad = findViewById(R.id.btnLoad)
        btnRun = findViewById(R.id.btnRun)

        findViewById<Button>(R.id.btnPickModel).setOnClickListener {
            pickModel.launch(arrayOf("*/*"))
        }
        btnLoad.setOnClickListener { loadModelAsync() }
        btnRun.setOnClickListener { runInferenceAsync() }
        findViewById<Button>(R.id.btnExport).setOnClickListener { exportLog() }

        log("就绪。选择 .gguf 模型文件后加载（建议 Qwen3.5-4B Q4_K_M，约 2.7GB）")
        setBusy(false)
    }

    private fun copyToFilesDir(uri: Uri) {
        setBusy(true)
        setStatus("复制模型文件中...")
        Thread {
            try {
                val dst = File(filesDir, "model.gguf")
                contentResolver.openInputStream(uri)?.use { input ->
                    dst.outputStream().use { output ->
                        input.copyTo(output, bufferSize = 1 shl 20)
                    }
                } ?: throw IllegalStateException("无法打开所选文件")
                modelPath = dst.absolutePath
                ui { setStatus("模型文件已就绪（${dst.length() / (1 shl 20)} MB）"); setBusy(false) }
                log("模型文件复制完成: ${dst.absolutePath}")
            } catch (e: Throwable) {
                ui { setStatus("复制失败"); setBusy(false) }
                log("复制异常: ${e.stackTraceToString().take(1500)}")
            }
        }.start()
    }

    private fun loadModelAsync() {
        val path = modelPath
        if (path == null) {
            log("请先选择模型文件")
            return
        }
        setBusy(true)
        setStatus("加载模型中（首次加载需要较长时间）...")
        val t0 = System.currentTimeMillis()
        Thread {
            try {
                val f = File(path)
                val json = NativeBridge.loadModel(f.parent ?: filesDir.absolutePath, f.name, null)
                sessionMs = System.currentTimeMillis() - t0
                loaded = true
                ui { setStatus("已加载 · $json"); setBusy(false) }
                log("加载完成: $json")
            } catch (e: Throwable) {
                loaded = false
                ui { setStatus("加载失败"); setBusy(false) }
                log("加载异常: ${e.stackTraceToString().take(3000)}")
            }
        }.start()
    }

    private fun runInferenceAsync() {
        if (!loaded) {
            log("模型尚未加载")
            return
        }
        val prompt = etPrompt.text.toString().trim()
        if (prompt.isEmpty()) {
            log("prompt 为空")
            return
        }
        setBusy(true)
        setStatus("推理中（CPU，耐心等待）...")
        Thread {
            try {
                val json = NativeBridge.complete(prompt, 64, 0.0f)
                ui { setStatus("完成"); tvResult.text = formatResult(json); setBusy(false) }
                log("推理完成: $json")
            } catch (e: Throwable) {
                ui { setStatus("推理失败"); setBusy(false) }
                log("推理异常: ${e.stackTraceToString().take(3000)}")
            }
        }.start()
    }

    private fun formatResult(json: String): String {
        val text = Regex("\"text\":\"((?:[^\"\\\\]|\\\\.)*)\"").find(json)?.groupValues?.get(1)
        val ms = Regex("\"ms\":(\\d+)").find(json)?.groupValues?.get(1)
        val pt = Regex("\"prompt_toks\":(\\d+)").find(json)?.groupValues?.get(1)
        val ot = Regex("\"out_toks\":(\\d+)").find(json)?.groupValues?.get(1)
        val decoded = text?.replace("\\n", "\n")?.replace("\\\"", "\"")?.replace("\\\\", "\\") ?: json
        return "$decoded\n\nprompt_toks=$pt out_toks=$ot ms=$ms"
    }

    private fun exportLog() {
        val text = logLines.toString()
        val send = Intent(Intent.ACTION_SEND)
            .setType("text/plain")
            .putExtra(Intent.EXTRA_TEXT, text)
            .putExtra(Intent.EXTRA_TITLE, "duan-demo-log.txt")
        startActivity(Intent.createChooser(send, "导出日志"))
    }

    private fun setBusy(b: Boolean) {
        busy = b
        ui {
            btnLoad.isEnabled = !busy && modelPath != null
            btnRun.isEnabled = !busy && loaded
        }
    }

    private fun setStatus(s: String) = ui { tvStatus.text = s }

    private fun log(s: String) {
        runOnUiThread {
            val line = "[${System.currentTimeMillis() - sessionMs}] $s"
            logLines.appendLine(line)
            tvLog.text = logLines.toString()
        }
    }

    private fun ui(f: () -> Unit) {
        runOnUiThread(f)
    }
}
