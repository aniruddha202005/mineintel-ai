const BASE_URL = `${import.meta.env.VITE_API_URL || "http://localhost:8000"}/api`;

async function request(path, options = {}) {
  const response = await fetch(`${BASE_URL}${path}`, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((item) => item.msg).join("; ");
    } catch {
      // keep the status text
    }
    throw new Error(detail);
  }
  if (response.status === 204) return null;
  return response.json();
}

export const listDocuments = () => request("/documents");

export function uploadDocument(file) {
  const form = new FormData();
  form.append("file", file);
  return request("/documents", { method: "POST", body: form });
}

export const deleteDocument = (id) => request(`/documents/${id}`, { method: "DELETE" });

export const askQuestion = (question, agentic) =>
  request("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, agentic }),
  });
