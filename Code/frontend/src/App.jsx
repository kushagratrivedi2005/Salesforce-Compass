
import { useState } from 'react';
import './App.css';

const TARGET_OPTIONS = [
  'FOUR WHEELER (Invalid Carriage)',
  'HEAVY GOODS VEHICLE',
  'HEAVY MOTOR VEHICLE',
  'HEAVY PASSENGER VEHICLE',
  'LIGHT GOODS VEHICLE',
  'LIGHT MOTOR VEHICLE',
  'LIGHT PASSENGER VEHICLE',
  'MEDIUM GOODS VEHICLE',
  'MEDIUM MOTOR VEHICLE',
  'MEDIUM PASSENGER VEHICLE',
  'OTHER THAN MENTIONED ABOVE',
  'THREE WHEELER (Invalid Carriage)',
  'THREE WHEELER(NT)',
  'THREE WHEELER(T)',
  'TWO WHEELER (Invalid Carriage)',
  'TWO WHEELER(NT)',
  'TWO WHEELER(T)'
];

const CANDIDATE_EXOGS_OPTIONS = [
  'interest_rate',
  'repo_rate',
  'holiday_count',
  'major_national_holiday',
  'major_religious_holiday',
];
const MANUAL_EXOGS_OPTIONS = [
  'interest_rate',
  'repo_rate',
];

function App() {
  const [form, setForm] = useState({
    TARGET: TARGET_OPTIONS[0],
    TEST_MONTHS: 6,
    FUTURE_FORECAST_MONTHS: 12,
    USE_TOP_K_EXOGS: true,
    CANDIDATE_EXOGS: ['interest_rate'],
    MANUAL_EXOGS: ['interest_rate'],
    TOP_K_EXOGS: 5,
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    if (type === 'checkbox') {
      if (form.USE_TOP_K_EXOGS && name === 'CANDIDATE_EXOGS') {
        setForm((prev) => ({
          ...prev,
          CANDIDATE_EXOGS: checked
            ? [...prev.CANDIDATE_EXOGS, value]
            : prev.CANDIDATE_EXOGS.filter((v) => v !== value),
        }));
      } else if (!form.USE_TOP_K_EXOGS && name === 'MANUAL_EXOGS') {
        setForm((prev) => ({
          ...prev,
          MANUAL_EXOGS: checked
            ? [...prev.MANUAL_EXOGS, value]
            : prev.MANUAL_EXOGS.filter((v) => v !== value),
        }));
      } else if (name === 'USE_TOP_K_EXOGS') {
        setForm((prev) => ({
          ...prev,
          USE_TOP_K_EXOGS: checked,
        }));
      }
    } else if (type === 'number') {
      setForm((prev) => ({ ...prev, [name]: Number(value) }));
    } else if (type === 'radio') {
      setForm((prev) => ({ ...prev, [name]: value === 'true' }));
    } else {
      setForm((prev) => ({ ...prev, [name]: value }));
    }
  };

  // Function to format date to "Month YYYY"
  const formatDate = (dateStr) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { month: 'long', year: 'numeric' });
  };

  // Function to get the current month's first day
  const getCurrentMonthStart = () => {
    const now = new Date();
    return new Date(now.getFullYear(), now.getMonth(), 1).toISOString().split('T')[0];
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const payload = {
        TARGET: form.TARGET ? `category_${form.TARGET}` : undefined,
        TEST_MONTHS: form.TEST_MONTHS,
        FUTURE_FORECAST_MONTHS: form.FUTURE_FORECAST_MONTHS,
        USE_TOP_K_EXOGS: form.USE_TOP_K_EXOGS,
        CANDIDATE_EXOGS: form.CANDIDATE_EXOGS,
        MANUAL_EXOGS: form.MANUAL_EXOGS,
        TOP_K_EXOGS: form.TOP_K_EXOGS,
        START_DATE: getCurrentMonthStart(), // Add current month as start date
      };
      const res = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const err = await res.json();
        setError(err.error || 'Unknown error');
        setLoading(false);
        return;
      }
      const data = await res.json();
      
      // Format the dates in the prediction data
      if (data.prediction) {
        Object.keys(data.prediction).forEach(model => {
          const formattedDates = {};
          Object.entries(data.prediction[model]).forEach(([date, value]) => {
            formattedDates[formatDate(date)] = value;
          });
          data.prediction[model] = formattedDates;
        });
      }
      
      setResult(data);
    } catch (err) {
      setError('Network or server error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h2 style={{ textAlign: 'center', fontSize: '2.5rem' }}>
        Vehicle Sales Forecasting
      </h2>
      <p className="subtitle">Advanced Time Series Prediction System</p>
      
      <form onSubmit={handleSubmit} className="form-card">
        <div className="form-grid">
          <div className="form-group">
            <label htmlFor="target">Target Category</label>
            <select
              id="target"
              name="TARGET"
              value={form.TARGET}
              onChange={handleChange}
              required
            >
              {TARGET_OPTIONS.map(opt => (
                <option key={opt} value={opt}>{opt}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="test-months">Test Months (3-9)</label>
            <input
              id="test-months"
              type="number"
              name="TEST_MONTHS"
              min={3}
              max={9}
              value={form.TEST_MONTHS}
              onChange={handleChange}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="forecast-months">Future Forecast Months (6-60)</label>
            <input
              id="forecast-months"
              type="number"
              name="FUTURE_FORECAST_MONTHS"
              min={6}
              max={60}
              value={form.FUTURE_FORECAST_MONTHS}
              onChange={handleChange}
              required
            />
          </div>
        </div>

        <div className="form-group">
          <div className="checkbox-label">
            <input
              type="checkbox"
              name="USE_TOP_K_EXOGS"
              checked={form.USE_TOP_K_EXOGS}
              onChange={handleChange}
              style={{ width: 'auto' }}
            />
            <span>Use Top-K Exogenous Variables</span>
          </div>
        </div>

        {form.USE_TOP_K_EXOGS ? (
          <>
            <div className="form-group">
              <label>Candidate Exogenous Variables</label>
              <div className="checkbox-group">
                {CANDIDATE_EXOGS_OPTIONS.map(opt => (
                  <label key={opt} className="checkbox-label">
                    <input
                      type="checkbox"
                      name="CANDIDATE_EXOGS"
                      value={opt}
                      checked={form.CANDIDATE_EXOGS.includes(opt)}
                      onChange={handleChange}
                      style={{ width: 'auto' }}
                    />
                    {opt}
                  </label>
                ))}
              </div>
            </div>

            <div className="form-group">
              <label htmlFor="top-k">Top-K Exogenous Variables</label>
              <input
                id="top-k"
                type="number"
                name="TOP_K_EXOGS"
                min={1}
                max={CANDIDATE_EXOGS_OPTIONS.length}
                value={form.TOP_K_EXOGS}
                onChange={handleChange}
                required
              />
            </div>
          </>
        ) : (
          <div className="form-group">
            <label>Manual Exogenous Variables</label>
            <div className="checkbox-group">
              {MANUAL_EXOGS_OPTIONS.map(opt => (
                <label key={opt} className="checkbox-label">
                  <input
                    type="checkbox"
                    name="MANUAL_EXOGS"
                    value={opt}
                    checked={form.MANUAL_EXOGS.includes(opt)}
                    onChange={handleChange}
                    style={{ width: 'auto' }}
                  />
                  {opt}
                </label>
              ))}
            </div>
          </div>
        )}

        <button type="submit" disabled={loading}>
          {loading ? (
            <>
              <span className="loading-spinner"></span> Generating Forecast...
            </>
          ) : (
            'Generate Forecast'
          )}
        </button>
      </form>

      {error && (
        <div className="error-message">
          <strong>Error:</strong> {typeof error === 'string' ? error : JSON.stringify(error)}
        </div>
      )}

      {result && (
        <div className="results-card">
          <h3>Forecast Results</h3>
          
          <h4>Model Performance Metrics</h4>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>
            Based on test data up to September 2025
          </p>
          {result.error && Array.isArray(result.error) && (
            <table className="metrics-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>MAE</th>
                  <th>RMSE</th>
                  <th>MAPE (%)</th>
                </tr>
              </thead>
              <tbody>
                {result.error.map((row, idx) => (
                  <tr key={idx}>
                    <td><strong>{row.Model}</strong></td>
                    <td className="number-cell">{row.MAE?.toFixed(2)}</td>
                    <td className="number-cell">{row.RMSE?.toFixed(2)}</td>
                    <td className="number-cell">{row.MAPE?.toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          
          <div className="section-divider"></div>
          
          <h4>Future Forecasts</h4>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>
            Starting from {formatDate(getCurrentMonthStart())}
          </p>
          {result.prediction && (
            <div style={{ overflowX: 'auto' }}>
              <table className="prediction-table">
                <thead>
                  <tr>
                    <th>Date</th>
                    {Object.keys(result.prediction).map(model => (
                      <th key={model}>{model}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.keys(Object.values(result.prediction)[0] || {}).map(date => (
                    <tr key={date}>
                      <td><strong>{date}</strong></td>
                      {Object.keys(result.prediction).map(model => (
                        <td key={model} className="number-cell">
                          {result.prediction[model][date]?.toFixed(2)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          
          {result.visualization && Object.keys(result.visualization).length > 0 && (
            <>
              <div className="section-divider"></div>
              <h4>Forecast Visualizations</h4>
              <div className="visualization-container">
                {Object.entries(result.visualization).map(([filename, base64Data]) => (
                  <div key={filename} style={{ marginBottom: '2rem' }}>
                    <h5>{filename.replace('.png', '').replace(/_/g, ' ').toUpperCase()}</h5>
                    <img 
                      src={`data:image/png;base64,${base64Data}`}
                      alt={filename}
                      style={{ maxWidth: '100%', height: 'auto', display: 'block', margin: '0 auto' }}
                    />
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
