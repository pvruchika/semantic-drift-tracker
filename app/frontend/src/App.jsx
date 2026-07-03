import { useState, useEffect, useRef, useCallback } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import { searchWords, getWordData, getTopDrift, BASE_URL } from "./api";
import "./App.css";

const YEARS = ["2015","2016","2017","2018","2019","2020","2021","2022","2023"];

function Spinner() {
  return (
    <div className="spinner-wrap">
      <div className="spinner" />
    </div>
  );
}

function DriftChart({ word, cumulativeDrift }) {
  const data = YEARS.map((y) => ({
    year: Number(y),
    score: parseFloat(cumulativeDrift[y].toFixed(4)),
  }));

  return (
    <div className="panel">
      <div className="panel-title">Semantic drift of &ldquo;{word}&rdquo; · 2015–2023</div>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis
            dataKey="year"
            tick={{ fontSize: 12, fill: "#64748b" }}
            tickLine={false}
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fontSize: 12, fill: "#64748b" }}
            tickLine={false}
            label={{
              value: "drift score",
              angle: -90,
              position: "insideLeft",
              offset: 12,
              style: { fontSize: 11, fill: "#94a3b8" },
            }}
          />
          <Tooltip
            formatter={(v) => [v.toFixed(4), "drift"]}
            labelFormatter={(l) => `Year ${l}`}
            contentStyle={{ fontSize: 13, borderRadius: 6, borderColor: "#e2e8f0" }}
          />
          <ReferenceLine
            y={0.5}
            stroke="#ef4444"
            strokeDasharray="4 4"
            label={{ value: "significant drift", position: "insideTopLeft", fill: "#ef4444", fontSize: 11 }}
          />
          <Line
            type="monotone"
            dataKey="score"
            stroke="#2563eb"
            strokeWidth={2}
            dot={{ r: 4, fill: "#2563eb", strokeWidth: 0 }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function NeighborTable({ neighbors, word }) {
  // Build a set of neighbor words for each year for O(1) lookup
  const neighborSets = {};
  for (const y of YEARS) {
    neighborSets[y] = new Set((neighbors[y] ?? []).map((n) => n[0]));
  }

  function isNew(cellWord, yearIdx) {
    if (yearIdx === 0 || !cellWord) return false;
    const prevYear = YEARS[yearIdx - 1];
    return !neighborSets[prevYear].has(cellWord);
  }

  return (
    <>
      <div className="panel">
        <div className="panel-title">Nearest neighbors by year</div>
        <div style={{ overflowX: "auto" }}>
          <table className="neighbor-table">
            <thead>
              <tr>
                <th className="rank-cell">#</th>
                {YEARS.map((y) => <th key={y}>{y}</th>)}
              </tr>
            </thead>
            <tbody>
              {[0, 1, 2, 3, 4].map((rank) => (
                <tr key={rank}>
                  <td className="rank-cell">{rank + 1}</td>
                  {YEARS.map((y, yi) => {
                    const cellWord = neighbors[y]?.[rank]?.[0] ?? null;
                    return (
                      <td
                        key={y}
                        className={isNew(cellWord, yi) ? "new-neighbor" : undefined}
                      >
                        {cellWord ?? "—"}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div className="neighbor-info">
        💡 <strong>How to read this:</strong> Each column shows the 5 words most similar
        to &ldquo;{word}&rdquo; in that year&rsquo;s Reddit data. <span style={{ color: "#2563eb", fontWeight: 500 }}>Blue words</span> are
        new neighbors that didn&rsquo;t appear the previous year — a sign of meaning shift.
      </div>
    </>
  );
}

export default function App() {
  const [topDrift, setTopDrift] = useState([]);
  const [selectedWord, setSelectedWord] = useState(null);
  const [wordData, setWordData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showWakeUpBanner, setShowWakeUpBanner] = useState(false);

  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const debounceRef = useRef(null);
  const searchRef = useRef(null);

  // Health check — show wake-up banner if server doesn't respond within 2500ms
  useEffect(() => {
    let bannerTimer;
    let maxTimer;
    let done = false;

    const controller = new AbortController();

    function dismiss() {
      if (done) return;
      done = true;
      clearTimeout(bannerTimer);
      clearTimeout(maxTimer);
      setShowWakeUpBanner(false);
    }

    bannerTimer = setTimeout(() => setShowWakeUpBanner(true), 2500);
    maxTimer = setTimeout(dismiss, 90_000);

    fetch(`${BASE_URL}/health`, { signal: controller.signal })
      .then((r) => { if (r.ok) dismiss(); })
      .catch(() => {});

    return () => {
      controller.abort();
      dismiss();
    };
  }, []);

  // Load sidebar on mount
  useEffect(() => {
    getTopDrift(20).then(setTopDrift).catch(console.error);
  }, []);

  // Close suggestions when clicking outside
  useEffect(() => {
    function handleClick(e) {
      if (searchRef.current && !searchRef.current.contains(e.target)) {
        setShowSuggestions(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const selectWord = useCallback((word) => {
    setSelectedWord(word);
    setQuery("");
    setSuggestions([]);
    setShowSuggestions(false);
    setLoading(true);
    setWordData(null);
    getWordData(word)
      .then(setWordData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  function handleQueryChange(e) {
    const q = e.target.value;
    setQuery(q);
    clearTimeout(debounceRef.current);
    if (!q.trim()) {
      setSuggestions([]);
      setShowSuggestions(false);
      return;
    }
    debounceRef.current = setTimeout(() => {
      searchWords(q.trim())
        .then((res) => {
          setSuggestions(res);
          setShowSuggestions(res.length > 0);
        })
        .catch(console.error);
    }, 300);
  }

  function handleKeyDown(e) {
    if (e.key === "Enter") {
      const match = suggestions[0]?.word || query.trim().toLowerCase();
      if (match) selectWord(match);
    } else if (e.key === "Escape") {
      setShowSuggestions(false);
    }
  }

  return (
    <div className="app">
      <div className="top-bar" />
      <header className="header">
        <h1>Semantic Drift Tracker</h1>
        <p>How word meanings shifted on Reddit 2015–2023</p>
      </header>

      {showWakeUpBanner && (
        <div style={{
          background: "#fefce8",
          border: "1px solid #fde047",
          borderRadius: "8px",
          padding: "12px 16px",
          fontSize: "13px",
          color: "#854d0e",
          margin: "0 16px 16px",
        }}>
          ⏳ Server is waking up after inactivity — this may take up to 60 seconds. The page will load automatically.
        </div>
      )}

      <div className="body">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="sidebar-title">Most drifted words</div>
          {topDrift.map(({ word, drift_score }) => (
            <div
              key={word}
              className={`word-item${selectedWord === word ? " selected" : ""}`}
              onClick={() => selectWord(word)}
            >
              <span>{word}</span>
              <span className="badge">{drift_score.toFixed(2)}</span>
            </div>
          ))}
        </aside>

        {/* Main */}
        <main className="main">
          {/* Search */}
          <div className="search-wrap" ref={searchRef}>
            <input
              className="search-input"
              type="text"
              placeholder="Search a word…"
              value={query}
              onChange={handleQueryChange}
              onKeyDown={handleKeyDown}
              onFocus={() => suggestions.length > 0 && setShowSuggestions(true)}
            />
            {showSuggestions && (
              <div className="suggestions">
                {suggestions.map(({ word }) => (
                  <div
                    key={word}
                    className="suggestion-item"
                    onMouseDown={() => selectWord(word)}
                  >
                    {word}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Content */}
          {loading && <Spinner />}

          {!loading && !selectedWord && (
            <div className="placeholder">
              Select a word from the list or search above to explore its drift
            </div>
          )}

          {!loading && wordData && (
            <>
              <DriftChart
                word={selectedWord}
                cumulativeDrift={wordData.cumulative_drift}
              />
              <NeighborTable neighbors={wordData.neighbors} word={selectedWord} />
            </>
          )}
        </main>
      </div>

      <footer className="footer">
        Built with Word2Vec + Procrustes alignment · 13,099 words tracked · Reddit 2015–2023
      </footer>
    </div>
  );
}
