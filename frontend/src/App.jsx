import { useCallback, useEffect, useState } from "react";
import { deleteDocument, listDocuments, uploadDocument } from "./api.js";
import UploadPanel from "./components/UploadPanel.jsx";
import DocumentList from "./components/DocumentList.jsx";
import ChatPanel from "./components/ChatPanel.jsx";

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    try {
      setDocuments(await listDocuments());
    } catch (err) {
      setError(`Could not reach the server: ${err.message}`);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function handleUpload(file) {
    setError("");
    try {
      await uploadDocument(file);
      await refresh();
    } catch (err) {
      setError(err.message);
      throw err;
    }
  }

  async function handleDelete(id) {
    setError("");
    try {
      await deleteDocument(id);
      await refresh();
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <h1>MineIntel AI</h1>
        <p className="tagline">Ask questions about mining accident reports.</p>
        <UploadPanel onUpload={handleUpload} />
        {error && <p className="error">{error}</p>}
        <DocumentList documents={documents} onDelete={handleDelete} />
      </aside>
      <main className="main">
        <ChatPanel hasDocuments={documents.length > 0} />
      </main>
    </div>
  );
}
