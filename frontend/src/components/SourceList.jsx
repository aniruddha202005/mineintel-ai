import { useState } from "react";

function Source({ source }) {
  const [open, setOpen] = useState(false);

  return (
    <li className="source">
      <button type="button" className="source-header" onClick={() => setOpen(!open)}>
        <span className="badge">{source.index}</span>
        <span className="source-title">
          {source.document_name}, page {source.page_number}
        </span>
        <span className="score">{(source.score * 100).toFixed(0)}% match</span>
      </button>
      {open && <p className="source-text">{source.text}</p>}
    </li>
  );
}

export default function SourceList({ sources }) {
  if (!sources || sources.length === 0) return null;
  return (
    <div className="sources">
      <h4>Sources</h4>
      <ul>
        {sources.map((source) => (
          <Source key={source.chunk_id} source={source} />
        ))}
      </ul>
    </div>
  );
}
