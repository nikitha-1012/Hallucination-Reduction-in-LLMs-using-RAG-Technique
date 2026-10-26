import { useState } from 'react';

function App() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleAsk = async () => {
    if (!query) return;
    setLoading(true);
    setError('');
    setResults(null);

    try {
      // Connects to your FastAPI backend
      const response = await fetch("http://127.0.0.1:8000/api/ask", {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) {
        throw new Error('Backend server error');
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      console.error("Fetch error:", err);
      setError('Failed to fetch results. Is your FastAPI backend running?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '2rem', fontFamily: 'sans-serif', maxWidth: '1200px', margin: '0 auto' }}>
      <h1 style={{ textAlign: 'center' }}>Hallucination Reduction in LLMs using RAG</h1>
      
      <div style={{ display: 'flex', gap: '10px', justifyContent: 'center', marginBottom: '2rem' }}>
        <input 
          type="text" 
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question (e.g., Which magazine was started first Arthur's Magazine or First for Women?)"
          style={{ width: '60%', padding: '10px', fontSize: '16px' }}
        />
        <button 
          onClick={handleAsk} 
          disabled={loading}
          style={{ padding: '10px 20px', fontSize: '16px', cursor: 'pointer' }}
        >
          {loading ? 'Processing...' : 'Compare Systems'}
        </button>
      </div>

      {error && <p style={{ color: 'red', textAlign: 'center' }}>{error}</p>}

      {results && (
        <div style={{ display: 'flex', gap: '20px', justifyContent: 'space-between' }}>
          
          {/* SYSTEM A */}
          <div style={{ flex: 1, border: '1px solid #ccc', padding: '15px', borderRadius: '8px' }}>
            <h3>SYSTEM A — BASELINE LLM</h3>
            <hr />
            <p><strong>Answer:</strong></p>
            <p>{results.system_a.answer}</p>
          </div>

          {/* SYSTEM B */}
          <div style={{ flex: 1, border: '1px solid #ccc', padding: '15px', borderRadius: '8px', backgroundColor: '#f9f9f9' }}>
            <h3>SYSTEM B — NAIVE RAG</h3>
            <hr />
            <p><strong>Retrieved Chunks:</strong> {results.system_b.retrieved_chunks.length}</p>
            <div style={{ fontSize: '12px', color: '#555', marginBottom: '10px' }}>
              {results.system_b.retrieved_chunks.map((chunk, i) => (
                <p key={i}>[{i+1}] {chunk.substring(0, 80)}...</p>
              ))}
            </div>
            <p><strong>Answer:</strong></p>
            <p>{results.system_b.answer}</p>
          </div>

          {/* SYSTEM C */}
          <div style={{ flex: 1, border: '1px solid #28a745', padding: '15px', borderRadius: '8px', backgroundColor: '#f0fff4' }}>
            <h3 style={{ color: '#28a745' }}>SYSTEM C — PROPOSED RAG</h3>
            <hr />
            <p><strong>Filtered Chunks:</strong> {results.system_c.filtered_chunks.length}</p>
            <div style={{ fontSize: '12px', color: '#555', marginBottom: '10px' }}>
              {results.system_c.filtered_chunks.map((chunk, i) => (
                <p key={i}>[{i+1}] Score: {chunk.score.toFixed(3)} | {chunk.text.substring(0, 60)}...</p>
              ))}
            </div>
            <p><strong>Answer:</strong></p>
            <p>{results.system_c.answer}</p>
          </div>

        </div>
      )}
    </div>
  );
}

export default App;