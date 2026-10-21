import { useRef, useState } from "react";

export default function UploadPanel({ onUpload }) {
  const inputRef = useRef(null);
  const [busy, setBusy] = useState(false);

  async function handleChange(event) {
    const files = Array.from(event.target.files || []);
    if (files.length === 0) return;
    setBusy(true);
    for (const file of files) {
      try {
        await onUpload(file);
      } catch {
        break;
      }
    }
    setBusy(false);
    if (inputRef.current) inputRef.current.value = "";
  }

  return (
    <div className="upload">
      <label className={`upload-button ${busy ? "disabled" : ""}`}>
        {busy ? "Processing..." : "Upload PDF reports"}
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf"
          multiple
          disabled={busy}
          onChange={handleChange}
          hidden
        />
      </label>
      {busy && <p className="hint">Extracting text, running OCR where needed and building embeddings. This can take a minute.</p>}
    </div>
  );
}
