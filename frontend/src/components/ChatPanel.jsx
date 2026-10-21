import { useEffect, useRef, useState } from "react";
import { askQuestion } from "../api.js";
import SourceList from "./SourceList.jsx";

const EXAMPLES = [
  "What were the most common causes of roof-fall accidents?",
  "Summarize the ventilation failures described in the reports.",
];

export default function ChatPanel({ hasDocuments }) {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [agentic, setAgentic] = useState(false);
  const [loading, setLoading] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function send(text) {
    const trimmed = text.trim();
    if (trimmed.length < 3 || loading) return;
    setQuestion("");
    setMessages((current) => [...current, { role: "user", text: trimmed }]);
    setLoading(true);
    try {
      const reply = await askQuestion(trimmed, agentic);
      setMessages((current) => [...current, { role: "assistant", reply }]);
    } catch (err) {
      setMessages((current) => [...current, { role: "error", text: err.message }]);
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(event) {
    event.preventDefault();
    send(question);
  }

  return (
    <section className="chat">
      <div className="messages">
        {messages.length === 0 && (
          <div className="empty">
            <h2>{hasDocuments ? "Ask about your reports" : "Upload a report to begin"}</h2>
            {hasDocuments &&
              EXAMPLES.map((example) => (
                <button key={example} type="button" className="example" onClick={() => send(example)}>
                  {example}
                </button>
              ))}
          </div>
        )}

        {messages.map((message, index) => {
          if (message.role === "user") {
            return (
              <div key={index} className="bubble user">
                {message.text}
              </div>
            );
          }
          if (message.role === "error") {
            return (
              <div key={index} className="bubble error-bubble">
                {message.text}
              </div>
            );
          }
          const { reply } = message;
          return (
            <div key={index} className={`bubble assistant ${reply.grounded ? "" : "ungrounded"}`}>
              <p>{reply.answer}</p>
              {reply.rewritten_query && <p className="meta">Search query rewritten to: {reply.rewritten_query}</p>}
              <SourceList sources={reply.sources} />
              <p className="meta">{(reply.latency_ms / 1000).toFixed(1)}s</p>
            </div>
          );
        })}

        {loading && <div className="bubble assistant">Searching the reports...</div>}
        <div ref={endRef} />
      </div>

      <form className="composer" onSubmit={handleSubmit}>
        <input
          type="text"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Ask a question about the uploaded reports"
          disabled={!hasDocuments}
        />
        <button type="submit" disabled={!hasDocuments || loading || question.trim().length < 3}>
          Ask
        </button>
        <label className="toggle">
          <input type="checkbox" checked={agentic} onChange={(event) => setAgentic(event.target.checked)} />
          Rewrite weak queries
        </label>
      </form>
    </section>
  );
}
