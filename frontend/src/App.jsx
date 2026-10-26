import { useState, useEffect } from 'react';

function App() {
  const [activeTab, setActiveTab] = useState('playground'); // 'playground' or 'evaluation'
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  
  // Evaluation Dashboard State
  const [evalRows, setEvalRows] = useState([]);
  const [evalLoading, setEvalLoading] = useState(false);

  const handleAsk = async () => {
    if (!query) return;
    setLoading(true);
    setError('');
    setResults(null);

    try {
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

  // Fetch evaluation results when the evaluation tab is selected
  useEffect(() => {
    if (activeTab === 'evaluation') {
      setEvalLoading(true);
      fetch("http://127.0.0.1:8000/api/evaluation-results")
        .then(res => res.json())
        .then(data => {
          if (data.results) {
            setEvalRows(data.results);
          }
        })
        .catch(err => console.error("Failed to fetch evaluation rows", err))
        .finally(() => setEvalLoading(false));
    }
  }, [activeTab]);

  return (
    <div style={{ padding: '2rem', fontFamily: 'sans-serif', maxWidth: '1200px', margin: '0 auto' }}>
      <h1 style={{ textAlign: 'center' }}>Hallucination Reduction in LLMs using RAG</h1>
      
      {/* Navigation Tab Bar */}
      <div style={{ display: 'flex', justifyContent: 'center', gap: '15px', marginBottom: '2rem' }}>
        <button
          onClick={() => setActiveTab('playground')}
          style={{
            padding: '8px 16px',
            fontSize: '15px',
            cursor: 'pointer',
            backgroundColor: activeTab === 'playground' ? '#007bff' : '#f0f0f0',
            color: activeTab === 'playground' ? '#fff' : '#333',
            border: '1px solid #ccc',
            borderRadius: '6px'
          }}
        >
          Live Playground
        </button>
        <button
          onClick={() => setActiveTab('evaluation')}
          style={{
            padding: '8px 16px',
            fontSize: '15px',
            cursor: 'pointer',
            backgroundColor: activeTab === 'evaluation' ? '#007bff' : '#f0f0f0',
            color: activeTab === 'evaluation' ? '#fff' : '#333',
            border: '1px solid #ccc',
            borderRadius: '6px'
          }}
        >
          Evaluation Dashboard
        </button>
      </div>

      {activeTab === 'playground' ? (
        <>
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
        </>
      ) : (
        /* Evaluation Dashboard View */
        <div>
          <h2 style={{ textAlign: 'center', marginBottom: '1rem' }}>Batch Evaluation Summary</h2>
          <p style={{ textAlign: 'center', color: '#666', marginBottom: '2rem' }}>
            Comparative batch run results generated from your evaluation script.
          </p>

          {evalLoading ? (
            <p style={{ textAlign: 'center' }}>Loading evaluation summary...</p>
          ) : evalRows.length === 0 ? (
            <p style={{ textAlign: 'center', color: '#888' }}>
              No evaluation results found. Run your local evaluation script first (`python evaluate_systems.py`).
            </p>
          ) : (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px', textAlign: 'left' }}>
                <thead>
                  <tr style={{ backgroundColor: '#334155', color: '#fff' }}>
                    <th style={{ padding: '10px', border: '1px solid #cbd5e1' }}>Question</th>
                    <th style={{ padding: '10px', border: '1px solid #cbd5e1' }}>Reference Answer</th>
                    <th style={{ padding: '10px', border: '1px solid #cbd5e1' }}>System A (Baseline)</th>
                    <th style={{ padding: '10px', border: '1px solid #cbd5e1' }}>System B (Naive)</th>
                    <th style={{ padding: '10px', border: '1px solid #cbd5e1' }}>System C (Proposed)</th>
                  </tr>
                </thead>
                <tbody>
                  {evalRows.map((row, idx) => (
                    <tr key={idx} style={{ backgroundColor: idx % 2 === 0 ? '#f8fafc' : '#fff' }}>
                      <td style={{ padding: '10px', border: '1px solid #cbd5e1', fontWeight: '500' }}>{row.question}</td>
                      <td style={{ padding: '10px', border: '1px solid #cbd5e1', color: '#059669' }}>{row.reference}</td>
                      <td style={{ padding: '10px', border: '1px solid #cbd5e1', color: '#475569' }}>{row.system_a}</td>
                      <td style={{ padding: '10px', border: '1px solid #cbd5e1', color: '#475569' }}>{row.system_b}</td>
                      <td style={{ padding: '10px', border: '1px solid #cbd5e1', color: '#0284c7' }}>{row.system_c}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;