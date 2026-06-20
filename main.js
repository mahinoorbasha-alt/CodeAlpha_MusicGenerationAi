/* Music Generation with AI — frontend logic */

function showResult(el, msg, ok = true) {
  el.textContent = msg;
  el.className = "result " + (ok ? "ok" : "err");
}
/* ---------------- Upload ---------------- */
function initUpload() {
  const form = document.getElementById("uploadForm");
  const input = document.getElementById("fileInput");
  const dz = document.getElementById("dropzone");
  const result = document.getElementById("uploadResult");

  dz.addEventListener("click", () => input.click());
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault(); dz.classList.remove("drag");
    input.files = e.dataTransfer.files;
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!input.files.length) return showResult(result, "Choose at least one MIDI file.", false);
    const fd = new FormData();
    for (const f of input.files) fd.append("files", f);
    showResult(result, "Uploading…");
    try {
      const r = await fetch("/api/upload", { method: "POST", body: fd });
      const data = await r.json();
      showResult(result, `Uploaded ${data.saved.length} file(s).` +
        (data.skipped.length ? ` Skipped: ${data.skipped.join(", ")}` : ""));
      setTimeout(() => location.reload(), 800);
    } catch (err) { showResult(result, "Upload failed: " + err.message, false); }
  });
}

/* ---------------- Train ---------------- */
function initTrain() {
  const preBtn = document.getElementById("preprocessBtn");
  const preRes = document.getElementById("preResult");
  const trainBtn = document.getElementById("trainBtn");
  const status = document.getElementById("trainStatus");
  const log = document.getElementById("trainLog");

  preBtn.addEventListener("click", async () => {
    showResult(preRes, "Preprocessing… this may take a moment.");
    try {
      const r = await fetch("/api/preprocess", { method: "POST" });
      const d = await r.json();
      if (!r.ok) return showResult(preRes, d.error || "Failed", false);
      showResult(preRes,
        `Processed ${d.processed_files} files · ${d.total_notes} notes · ${d.unique_notes} unique`);
    } catch (e) { showResult(preRes, e.message, false); }
  });

  trainBtn.addEventListener("click", async () => {
    const epochs = +document.getElementById("epochs").value;
    const batch_size = +document.getElementById("batchSize").value;
    showResult(status, `Training for ${epochs} epoch(s)… this can take a while.`);
    log.textContent = "";
    try {
      const r = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ epochs, batch_size }),
      });
      const d = await r.json();
      if (!r.ok) return showResult(status, d.error || "Training failed", false);
      showResult(status, `Training complete · vocab=${d.vocab} · patterns=${d.patterns}`);
      const lines = d.loss.map((l, i) =>
        `Epoch ${i + 1}/${d.epochs}  loss=${l.toFixed(4)}  acc=${d.accuracy[i].toFixed(4)}`);
      log.textContent = lines.join("\n");
    } catch (e) { showResult(status, e.message, false); }
  });
}

/* ---------------- Generate ---------------- */
function initGenerate() {
  const btn = document.getElementById("genBtn");
  const res = document.getElementById("genResult");

  btn.addEventListener("click", async () => {
    const length = +document.getElementById("genLength").value;
    showResult(res, "Generating music…");
    try {
      const r = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ length }),
      });
      const d = await r.json();
      if (!r.ok) return showResult(res, d.error || "Failed", false);
      res.className = "result ok";
      res.innerHTML =
        `Generated <strong>${d.notes}</strong> notes → <em>${d.file}</em> ` +
        `<a class="btn btn-primary btn-sm" href="/download/${d.file}">Download MIDI</a>`;
      setTimeout(() => location.reload(), 2500);
    } catch (e) { showResult(res, e.message, false); }
  });
}
