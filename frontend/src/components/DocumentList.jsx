export default function DocumentList({ documents, onDelete }) {
  if (documents.length === 0) {
    return <p className="hint">No documents yet. Upload a report to get started.</p>;
  }

  return (
    <ul className="documents">
      {documents.map((doc) => (
        <li key={doc.id}>
          <div>
            <strong title={doc.filename}>{doc.filename}</strong>
            <span>
              {doc.page_count} pages, {doc.chunk_count} chunks
              {doc.ocr_page_count > 0 ? `, ${doc.ocr_page_count} via OCR` : ""}
            </span>
          </div>
          <button type="button" className="link" onClick={() => onDelete(doc.id)}>
            Remove
          </button>
        </li>
      ))}
    </ul>
  );
}
