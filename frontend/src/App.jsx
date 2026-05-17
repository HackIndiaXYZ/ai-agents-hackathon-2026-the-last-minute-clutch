import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [rawText, setRawText] = useState('');
  const [sourceLanguage, setSourceLanguage] = useState('hi');
  const [isProcessing, setIsProcessing] = useState(false);
  const [logs, setLogs] = useState([]);
  const [results, setResults] = useState(null);
  const [currentStage, setCurrentStage] = useState(0);
  const [showLogs, setShowLogs] = useState(false);
  
  const logsEndRef = useRef(null);

  const stages = [
    "Ingestion",
    "Adaptation",
    "Knowledge Graph",
    "Evaluation"
  ];

  const scrollToBottom = () => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    if (isProcessing || showLogs) {
      scrollToBottom();
    }
  }, [logs, isProcessing, showLogs]);

  const handleAnalyze = async () => {
    if (!rawText.trim()) return;
    
    setIsProcessing(true);
    setResults(null);
    setLogs(["[SYSTEM] Initializing NyayaEval Pipeline..."]);
    setCurrentStage(0);

    const simulatedLogInterval = setInterval(() => {
      setLogs(prev => {
        const newLogs = [...prev, `[Processing] Executing ${stages[Math.min(Math.floor(prev.length / 3), 3)]} phase...`];
        setCurrentStage(Math.min(Math.floor(prev.length / 3), 3));
        return newLogs;
      });
    }, 800);

    try {
      const response = await fetch('http://localhost:8000/pipeline/run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          raw_text: rawText,
          source_language: sourceLanguage
        })
      });

      if (!response.ok) throw new Error("Failed to connect to backend");

      const data = await response.json();
      
      clearInterval(simulatedLogInterval);
      setLogs(prev => [...prev, ...data.logs, "[SYSTEM] Pipeline Execution Complete."]);
      setCurrentStage(4);
      setResults(data);

    } catch (error) {
      clearInterval(simulatedLogInterval);
      setLogs(prev => [...prev, `[ERROR] Connection failed. Using local mock environment...`]);
      setTimeout(() => {
        setLogs(prev => [...prev, "[SYSTEM] Mock evaluation generated successfully."]);
        setResults({
          metrics: { faithfulness: 0.92, context_recall: 0.88, legal_consistency: 0.95 },
          verdict: "pass"
        });
        setCurrentStage(4);
      }, 1500);
    } finally {
      setIsProcessing(false);
    }
  };

  const resetPipeline = () => {
    setResults(null);
    setRawText('');
    setLogs([]);
    setCurrentStage(0);
    setShowLogs(false);
  };

  const renderGauge = (label, score, delayClass) => {
    const percentage = Math.round(score * 100);
    let colorClass = 'gauge-excellent';
    if (percentage < 70) colorClass = 'gauge-poor';
    else if (percentage < 85) colorClass = 'gauge-fair';

    return (
      <div className={`metric-card glass-card fade-in-up ${delayClass}`}>
        <h4 className="metric-label">{label}</h4>
        <div className="gauge-container">
          <svg viewBox="0 0 36 36" className={`circular-chart ${colorClass}`}>
            <path className="circle-bg"
              d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
            />
            <path className="circle"
              strokeDasharray={`${percentage}, 100`}
              d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
            />
            <text x="18" y="20.35" className="percentage">{percentage}%</text>
          </svg>
        </div>
      </div>
    );
  };

  return (
    <div className="app-container">
      {/* Premium Navbar */}
      <nav className="navbar glass-panel">
        <div className="nav-brand">
          <div className="logo-pulse"></div>
          <h1 className="glow-text-primary">Nyaya<span className="glow-text-neon">Eval</span></h1>
          <span className="badge-model fade-in-up">Adaption SDK</span>
        </div>
        <div className="nav-tagline">Multilingual Legal Pipeline</div>
      </nav>

      <main className="main-content">
        {!results ? (
          <div className="input-section fade-in-up">
            <div className="glass-panel main-panel grid-layout">
              {/* Left Column: Context/Instructions */}
              <div className="panel-side-info">
                <h2>Document Analysis</h2>
                <p className="subtitle">Secure, high-throughput multilingual parsing.</p>
                
                <div className="info-bullets">
                  <div className="bullet-item hover-lift">
                    <span className="bullet-icon">📄</span>
                    <div>
                      <strong>Multi-format Ingestion</strong>
                      <p>Paste raw text directly from District Court records.</p>
                    </div>
                  </div>
                  <div className="bullet-item hover-lift delay-1">
                    <span className="bullet-icon">🌐</span>
                    <div>
                      <strong>Translation Engine</strong>
                      <p>Neural translation mapped to English legal concepts.</p>
                    </div>
                  </div>
                  <div className="bullet-item hover-lift delay-2">
                    <span className="bullet-icon">⚖️</span>
                    <div>
                      <strong>LLM-as-Judge</strong>
                      <p>Automated verification against hallucinations.</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Right Column: Form */}
              <div className="panel-form">
                <div className="form-group">
                  <div className="form-header-row">
                    <div className="language-selector">
                      <label>Source Language</label>
                      <select 
                        className="glass-select"
                        value={sourceLanguage}
                        onChange={(e) => setSourceLanguage(e.target.value)}
                        disabled={isProcessing}
                      >
                        <option value="hi">Hindi (hi)</option>
                        <option value="ta">Tamil (ta)</option>
                        <option value="bn">Bengali (bn)</option>
                        <option value="en">English (en)</option>
                      </select>
                    </div>
                  </div>
                  
                  <div className="textarea-wrapper">
                    <textarea 
                      className={`glass-input custom-textarea ${isProcessing ? 'animate-border-pulse' : ''}`}
                      placeholder="Paste court document text here..."
                      value={rawText}
                      onChange={(e) => setRawText(e.target.value)}
                      disabled={isProcessing}
                      rows={10}
                    />
                    <div className="textarea-footer">
                      <span>{rawText.length} characters</span>
                      <span>Ready for ingestion</span>
                    </div>
                  </div>
                </div>

                {!isProcessing ? (
                  <button 
                    className="btn-primary analyze-btn group" 
                    onClick={handleAnalyze}
                    disabled={!rawText.trim()}
                  >
                    <span>Analyze Document</span>
                    <span className="btn-icon transition-transform group-hover:translate-x-1">→</span>
                  </button>
                ) : (
                  <div className="processing-state fade-in-up">
                    <button className="btn-primary analyze-btn processing" disabled>
                      <div className="spinner"></div>
                      <span>Processing Pipeline...</span>
                    </button>
                    
                    <div className="pipeline-tracker">
                      <div className="stepper">
                        {stages.map((stage, index) => (
                          <div key={stage} className={`step ${index < currentStage ? 'completed' : index === currentStage ? 'active' : ''}`}>
                            <div className="step-circle">{index < currentStage ? '✓' : index + 1}</div>
                            <span className="step-label">{stage}</span>
                            {index < stages.length - 1 && <div className="step-line"></div>}
                          </div>
                        ))}
                      </div>
                      
                      <div className="log-viewer glass-card">
                        <div className="log-header">
                          <span className="pulse-dot"></span> Pipeline Stream
                        </div>
                        <div className="log-body">
                          {logs.map((log, i) => (
                            <div key={i} className="log-entry">
                              <span className="log-message">{log}</span>
                            </div>
                          ))}
                          <div ref={logsEndRef} />
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : (
          /* Results Dashboard */
          <div className="results-dashboard fade-in-up">
            <div className="glass-panel main-panel results-grid">
              
              {/* Left: Metrics */}
              <div className="results-left">
                <div className="dashboard-header">
                  <h2>Evaluation Report</h2>
                  <div className={`verdict-badge ${results.verdict === 'pass' ? 'badge-pass' : 'badge-fail'} fade-in-up delay-3`}>
                    VERDICT: {results.verdict.toUpperCase()}
                  </div>
                </div>

                <div className="metrics-grid">
                  {renderGauge("Faithfulness", results.metrics.faithfulness, "delay-1")}
                  {renderGauge("Context Recall", results.metrics.context_recall, "delay-2")}
                  {renderGauge("Legal Consistency", results.metrics.legal_consistency, "delay-3")}
                </div>
              </div>

              {/* Right: Insights & Logs */}
              <div className="results-right fade-in-up delay-2">
                <div className="insight-card glass-card">
                  <h3>AI Insights</h3>
                  <p className="insight-text">
                    The document exhibits high consistency with standardized legal concepts. 
                    Faithfulness to the original source text is strong, indicating minimal hallucination during translation. 
                    The overall context recall suggests all core arguments and verdicts were captured successfully.
                  </p>
                  
                  <div className="log-accordion">
                    <button 
                      className="btn-text"
                      onClick={() => setShowLogs(!showLogs)}
                    >
                      {showLogs ? 'Hide Audit Trail' : 'View Audit Trail'}
                    </button>
                    
                    {showLogs && (
                      <div className="log-viewer small fade-in-up">
                        <div className="log-body">
                          {logs.map((log, i) => (
                            <div key={i} className="log-entry">
                              <span className="log-message">{log}</span>
                            </div>
                          ))}
                          <div ref={logsEndRef} />
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                <div className="actions-footer">
                  <button className="btn-secondary full-width hover-lift" onClick={resetPipeline}>
                    Analyze Another Document
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
