
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

const FUEL_TYPES = [
  'fuel_CNG ONLY',
  'fuel_DI METHYL ETHER',
  'fuel_DIESEL',
  'fuel_DIESEL/HYBRID',
  'fuel_DUAL DIESEL/BIO CNG',
  'fuel_DUAL DIESEL/CNG',
  'fuel_DUAL DIESEL/LNG',
  'fuel_ELECTRIC BOV',
  'fuel_ETHANOL E',
  'fuel_FUEL CELL HYDROGEN',
  'fuel_LNG',
  'fuel_LPG ONLY',
  'fuel_METHANOL',
  'fuel_NOT APPLICABLE',
  'fuel_PETROL',
  'fuel_PETROL E',
  'fuel_PETROL E /CNG',
  'fuel_PETROL E /HYBRID',
  'fuel_PETROL E /HYBRID/CNG',
  'fuel_PETROL E /LPG',
  'fuel_PETROL/CNG',
  'fuel_PETROL/HYBRID',
  'fuel_PETROL/HYBRID/CNG',
  'fuel_PETROL/LPG',
  'fuel_PETROL/METHANOL',
  'fuel_PLUG IN HYBRID EV',
  'fuel_PURE EV',
  'fuel_SOLAR',
  'fuel_STRONG HYBRID EV',
  'fuel_Total',
];

const VEHICLE_CLASSES = [
  'class_Adapted Vehicle',
  'class_Agricultural Tractor',
  'class_Ambulance',
  'class_Animal Ambulance',
  'class_Armoured/Specialised Vehicle',
  'class_Articulated Vehicle',
  'class_Auxiliary Trailer',
  'class_Breakdown Van',
  'class_Bulldozer',
  'class_Bus',
  'class_Camper Van / Trailer',
  'class_Camper Van / Trailer (Private Use)',
  'class_Cash Van',
  'class_Construction Equipment Vehicle',
  'class_Construction Equipment Vehicle (Commercial)',
  'class_Crane Mounted Vehicle',
  'class_Dumper',
  'class_Earth Moving Equipment',
  'class_Educational Institution Bus',
  'class_Excavator (Commercial)',
  'class_Excavator (NT)',
  'class_Fire Fighting Vehicle',
  'class_Fire Tenders',
  'class_Fork Lift',
  'class_Goods Carrier',
  'class_Harvester',
  'class_Hearses',
  'class_Library Van',
  'class_Luxury Cab',
  'class_M-Cycle/Scooter',
  'class_M-Cycle/Scooter-With Side Car',
  'class_Maxi Cab',
  'class_Mobile Canteen',
  'class_Mobile Clinic',
  'class_Mobile Workshop',
  'class_Modular Hydraulic Trailer',
  'class_Moped',
  'class_Motor Cab',
  'class_Motor Car',
  'class_Motor Caravan',
  'class_Motor Cycle/Scooter-SideCar(T)',
  'class_Motor Cycle/Scooter-Used For Hire',
  'class_Motor Cycle/Scooter-With Trailer',
  'class_Motorised Cycle (CC 25cc)',
  'class_Omni Bus',
  'class_Omni Bus (Private Use)',
  'class_Power Tiller',
  'class_Power Tiller (Commercial)',
  'class_Private Service Vehicle',
  'class_Private Service Vehicle (Individual Use)',
  'class_Puller Tractor',
  'class_Quadricycle (Commercial)',
  'class_Quadricycle (Private)',
  'class_Recovery Vehicle',
  'class_Road Roller',
  'class_School Bus',
  'class_Semi-Trailer (Commercial)',
  'class_Snorked Ladders',
  'class_Three Wheeler (Goods)',
  'class_Three Wheeler (Passenger)',
  'class_Three Wheeler (Personal)',
  'class_Tow Truck',
  'class_Tower Wagon',
  'class_Tractor (Commercial)',
  'class_Tractor-Trolley(Commercial)',
  'class_Trailer (Agricultural)',
  'class_Trailer (Commercial)',
  'class_Trailer For Personal Use',
  'class_Tree Trimming Vehicle',
  'class_Vehicle Fitted With Compressor',
  'class_Vehicle Fitted With Generator',
  'class_Vehicle Fitted With Rig',
  'class_Vintage Motor Vehicle',
  'class_X-Ray Van',
  'class_e-Rickshaw with Cart (G)',
  'class_e-Rickshaw(P)',
  'class_Total',
];

const VEHICLE_CATEGORIES = [
  'category_FOUR WHEELER (Invalid Carriage)',
  'category_HEAVY GOODS VEHICLE',
  'category_HEAVY MOTOR VEHICLE',
  'category_HEAVY PASSENGER VEHICLE',
  'category_LIGHT GOODS VEHICLE',
  'category_LIGHT MOTOR VEHICLE',
  'category_LIGHT PASSENGER VEHICLE',
  'category_MEDIUM GOODS VEHICLE',
  'category_MEDIUM MOTOR VEHICLE',
  'category_MEDIUM PASSENGER VEHICLE',
  'category_OTHER THAN MENTIONED ABOVE',
  'category_THREE WHEELER (Invalid Carriage)',
  'category_THREE WHEELER(NT)',
  'category_THREE WHEELER(T)',
  'category_TWO WHEELER (Invalid Carriage)',
  'category_TWO WHEELER(NT)',
  'category_TWO WHEELER(T)',
  'category_Total',
];

// Categorized exogenous variables
const EXOG_CATEGORIES = {
  'Economic & Policy Indicators': [
    'interest_rate',
    'repo_rate',
    'holiday_count',
    'major_national_holiday',
    'major_religious_holiday',
    'sub_4m_rule',
    'bs3_norms',
    'bs4_norms',
    'bs6_norms',
    'fame_i',
    'fame_ii',
    'fame_iii',
    'pli_scheme',
    'vehicle_scrappage_policy',
    'bharat_ncap',
  ],
  'Fuel Types': FUEL_TYPES,
  'Vehicle Classes': VEHICLE_CLASSES,
  'Vehicle Categories': VEHICLE_CATEGORIES,
};

// Flatten all variables into a single array
const CANDIDATE_EXOGS_OPTIONS = Object.values(EXOG_CATEGORIES).flat();
const MANUAL_EXOGS_OPTIONS = CANDIDATE_EXOGS_OPTIONS;

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
  const [loadingViz, setLoadingViz] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [candidateSearchTerm, setCandidateSearchTerm] = useState('');
  const [manualSearchTerm, setManualSearchTerm] = useState('');
  const [visualizationYears, setVisualizationYears] = useState(2);
  const [lastPayload, setLastPayload] = useState(null); // Store last successful payload for hot-reload
  const [expandedCategories, setExpandedCategories] = useState({}); // Track which categories are expanded

  const toggleCategory = (category) => {
    setExpandedCategories(prev => ({
      ...prev,
      [category]: !prev[category]
    }));
  };

  const selectAllInCategory = (category, variables, fieldName) => {
    const filteredVars = variables.filter(opt => 
      opt.toLowerCase().includes(
        fieldName === 'CANDIDATE_EXOGS' ? candidateSearchTerm.toLowerCase() : manualSearchTerm.toLowerCase()
      )
    );
    setForm(prev => ({
      ...prev,
      [fieldName]: [...new Set([...prev[fieldName], ...filteredVars])]
    }));
  };

  const clearAllInCategory = (category, variables, fieldName) => {
    setForm(prev => ({
      ...prev,
      [fieldName]: prev[fieldName].filter(item => !variables.includes(item))
    }));
  };

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
        VISUALIZATION_YEARS: visualizationYears,
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
      setLastPayload(payload); // Store payload for hot-reload
      setCurrentImageIndex(0); // Reset to first image when new results arrive
    } catch (err) {
      console.error('Error details:', err);
      setError(`Network or server error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Hot-reload visualizations with new time frame
  const reloadVisualizations = async (newYears) => {
    if (!lastPayload || !result) return;
    
    setLoadingViz(true);
    try {
      const payload = {
        ...lastPayload,
        VISUALIZATION_YEARS: newYears,
      };
      const res = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        console.error('Failed to reload visualizations');
        setLoadingViz(false);
        return;
      }
      const data = await res.json();
      
      // Update only the visualizations, keep metrics and predictions the same
      setResult(prev => ({
        ...prev,
        visualization: data.visualization,
      }));
      setVisualizationYears(newYears);
      setCurrentImageIndex(0);
    } catch (err) {
      console.error('Error reloading visualizations:', err);
    } finally {
      setLoadingViz(false);
    }
  };

  // Gallery navigation functions
  const nextImage = () => {
    if (result && result.visualization) {
      const totalImages = Object.keys(result.visualization).length;
      setCurrentImageIndex((prev) => (prev + 1) % totalImages);
    }
  };

  const prevImage = () => {
    if (result && result.visualization) {
      const totalImages = Object.keys(result.visualization).length;
      setCurrentImageIndex((prev) => (prev - 1 + totalImages) % totalImages);
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
          <label>Variable Selection Mode</label>
          <div style={{ display: 'flex', gap: '2rem', marginTop: '0.5rem' }}>
            <label className="radio-label">
              <input
                type="radio"
                name="USE_TOP_K_EXOGS"
                checked={form.USE_TOP_K_EXOGS === true}
                onChange={() => setForm(prev => ({ ...prev, USE_TOP_K_EXOGS: true }))}
                style={{ width: 'auto' }}
              />
              <span>Top-K Variables</span>
            </label>
            <label className="radio-label">
              <input
                type="radio"
                name="USE_TOP_K_EXOGS"
                checked={form.USE_TOP_K_EXOGS === false}
                onChange={() => setForm(prev => ({ ...prev, USE_TOP_K_EXOGS: false }))}
                style={{ width: 'auto' }}
              />
              <span>Manual Selection</span>
            </label>
          </div>
        </div>

        {form.USE_TOP_K_EXOGS ? (
          <>
            <div className="form-group">
              <label>Candidate Exogenous Variables</label>
              <input
                type="text"
                className="search-input"
                placeholder="Search variables..."
                value={candidateSearchTerm}
                onChange={(e) => setCandidateSearchTerm(e.target.value)}
              />
              <div style={{ marginBottom: '0.5rem' }}>
                <button
                  type="button"
                  className="select-all-btn"
                  onClick={() => {
                    const filteredOptions = CANDIDATE_EXOGS_OPTIONS.filter(opt => 
                      opt.toLowerCase().includes(candidateSearchTerm.toLowerCase())
                    );
                    setForm(prev => ({
                      ...prev,
                      CANDIDATE_EXOGS: filteredOptions
                    }));
                  }}
                >
                  Select All
                </button>
                <button
                  type="button"
                  className="select-all-btn"
                  onClick={() => {
                    setForm(prev => ({
                      ...prev,
                      CANDIDATE_EXOGS: []
                    }));
                  }}
                  style={{ marginLeft: '0.5rem' }}
                >
                  Clear All
                </button>
              </div>
              
              {Object.entries(EXOG_CATEGORIES).map(([category, variables]) => {
                const filteredVars = variables.filter(opt => 
                  opt.toLowerCase().includes(candidateSearchTerm.toLowerCase())
                );
                
                if (filteredVars.length === 0) return null;
                
                const isExpanded = expandedCategories[`candidate_${category}`];
                
                return (
                  <div key={category} style={{ marginBottom: '1rem', border: '1px solid rgba(59, 130, 246, 0.2)', borderRadius: '4px' }}>
                    <div 
                      onClick={() => toggleCategory(`candidate_${category}`)}
                      style={{ 
                        fontSize: '0.9rem', 
                        fontWeight: '600', 
                        color: 'var(--primary)', 
                        padding: '0.5rem',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        userSelect: 'none'
                      }}
                    >
                      <span>{category} ({filteredVars.length})</span>
                      <span style={{ fontSize: '1.2rem', transition: 'transform 0.2s', transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)' }}>
                        ▼
                      </span>
                    </div>
                    {isExpanded && (
                      <div style={{ padding: '0.5rem' }}>
                        <div style={{ marginBottom: '0.5rem', display: 'flex', gap: '0.5rem' }}>
                          <button
                            type="button"
                            className="select-all-btn"
                            onClick={() => selectAllInCategory(category, filteredVars, 'CANDIDATE_EXOGS')}
                            style={{ fontSize: '0.75rem', padding: '0.25rem 0.5rem' }}
                          >
                            Select All
                          </button>
                          <button
                            type="button"
                            className="select-all-btn"
                            onClick={() => clearAllInCategory(category, variables, 'CANDIDATE_EXOGS')}
                            style={{ fontSize: '0.75rem', padding: '0.25rem 0.5rem' }}
                          >
                            Clear All
                          </button>
                        </div>
                        <div className="checkbox-group">
                          {filteredVars.map(opt => (
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
                    )}
                  </div>
                );
              })}
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
            <input
              type="text"
              className="search-input"
              placeholder="Search variables..."
              value={manualSearchTerm}
              onChange={(e) => setManualSearchTerm(e.target.value)}
            />
            <div style={{ marginBottom: '0.5rem' }}>
              <button
                type="button"
                className="select-all-btn"
                onClick={() => {
                  const allOptions = MANUAL_EXOGS_OPTIONS.filter(opt => 
                    opt.toLowerCase().includes(manualSearchTerm.toLowerCase())
                  );
                  setForm(prev => ({
                    ...prev,
                    MANUAL_EXOGS: allOptions
                  }));
                }}
              >
                Select All
              </button>
              <button
                type="button"
                className="select-all-btn"
                onClick={() => {
                  setForm(prev => ({
                    ...prev,
                    MANUAL_EXOGS: []
                  }));
                }}
                style={{ marginLeft: '0.5rem' }}
              >
                Clear All
              </button>
            </div>
            
            {Object.entries(EXOG_CATEGORIES).map(([category, variables]) => {
              const filteredVars = variables.filter(opt => 
                opt.toLowerCase().includes(manualSearchTerm.toLowerCase())
              );
              
              if (filteredVars.length === 0) return null;
              
              const isExpanded = expandedCategories[`manual_${category}`];
              
              return (
                <div key={category} style={{ marginBottom: '1rem', border: '1px solid rgba(59, 130, 246, 0.2)', borderRadius: '4px' }}>
                  <div 
                    onClick={() => toggleCategory(`manual_${category}`)}
                    style={{ 
                      fontSize: '0.9rem', 
                      fontWeight: '600', 
                      color: 'var(--primary)', 
                      padding: '0.5rem',
                      backgroundColor: 'rgba(59, 130, 246, 0.1)',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      userSelect: 'none'
                    }}
                  >
                    <span>{category} ({filteredVars.length})</span>
                    <span style={{ fontSize: '1.2rem', transition: 'transform 0.2s', transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)' }}>
                      ▼
                    </span>
                  </div>
                  {isExpanded && (
                    <div style={{ padding: '0.5rem' }}>
                      <div style={{ marginBottom: '0.5rem', display: 'flex', gap: '0.5rem' }}>
                        <button
                          type="button"
                          className="select-all-btn"
                          onClick={() => selectAllInCategory(category, filteredVars, 'MANUAL_EXOGS')}
                          style={{ fontSize: '0.75rem', padding: '0.25rem 0.5rem' }}
                        >
                          Select All
                        </button>
                        <button
                          type="button"
                          className="select-all-btn"
                          onClick={() => clearAllInCategory(category, variables, 'MANUAL_EXOGS')}
                          style={{ fontSize: '0.75rem', padding: '0.25rem 0.5rem' }}
                        >
                          Clear All
                        </button>
                      </div>
                      <div className="checkbox-group">
                        {filteredVars.map(opt => (
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
                </div>
              );
            })}
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
          
          {/* 1. VISUALIZATIONS FIRST */}
          {result.visualization && Object.keys(result.visualization).length > 0 && (
            <>
              <div className="section-divider"></div>
              <h4>Forecast Visualizations</h4>
              
              {/* Time Frame Filter Buttons */}
              <div style={{ 
                display: 'flex', 
                gap: '0.3rem', 
                marginBottom: '1rem',
                flexWrap: 'nowrap',
                alignItems: 'center'
              }}>
                <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginRight: '0.3rem' }}>
                  Historical Data:
                </span>
                {[
                  { label: '6M', value: 0.5 },
                  { label: '1Y', value: 1 },
                  { label: '2Y', value: 2 },
                  { label: '5Y', value: 5 },
                  { label: '10Y', value: 10 },
                  { label: '20Y', value: 20 },
                ].map(({ label, value }) => (
                  <button
                    key={value}
                    type="button"
                    onClick={() => reloadVisualizations(value)}
                    disabled={loadingViz}
                    style={{
                      padding: '0.25rem 0.5rem',
                      fontSize: '0.75rem',
                      fontWeight: visualizationYears === value ? '600' : '400',
                      backgroundColor: visualizationYears === value ? 'var(--primary)' : 'rgba(59, 130, 246, 0.1)',
                      color: visualizationYears === value ? 'white' : 'var(--primary)',
                      border: `1px solid ${visualizationYears === value ? 'var(--primary)' : 'rgba(59, 130, 246, 0.3)'}`,
                      borderRadius: '3px',
                      cursor: loadingViz ? 'wait' : 'pointer',
                      transition: 'all 0.2s ease',
                      opacity: loadingViz && visualizationYears !== value ? 0.5 : 1,
                      minWidth: '40px',
                    }}
                  >
                    {label}
                  </button>
                ))}
                {loadingViz && (
                  <span style={{ color: 'var(--text-secondary)', fontSize: '0.75rem', marginLeft: '0.3rem' }}>
                    <span className="loading-spinner" style={{ width: '12px', height: '12px' }}></span> Updating...
                  </span>
                )}
              </div>
              
              <div className="gallery-container">
                {(() => {
                  const visualizations = Object.entries(result.visualization);
                  const [filename, base64Data] = visualizations[currentImageIndex];
                  const totalImages = visualizations.length;
                  
                  return (
                    <>
                      <div className="gallery-header">
                        <h5>{filename.replace('.png', '').replace(/_/g, ' ').toUpperCase()}</h5>
                        <span className="gallery-counter">
                          {currentImageIndex + 1} / {totalImages}
                        </span>
                      </div>
                      
                      <div className="gallery-content">
                        <button 
                          className="gallery-nav gallery-nav-left" 
                          onClick={prevImage}
                          disabled={totalImages <= 1}
                          aria-label="Previous image"
                        >
                          ‹
                        </button>
                        
                        <div className="gallery-image-wrapper">
                          <img 
                            src={`data:image/png;base64,${base64Data}`}
                            alt={filename}
                            className="gallery-image"
                          />
                        </div>
                        
                        <button 
                          className="gallery-nav gallery-nav-right" 
                          onClick={nextImage}
                          disabled={totalImages <= 1}
                          aria-label="Next image"
                        >
                          ›
                        </button>
                      </div>
                      
                      {totalImages > 1 && (
                        <div className="gallery-indicators">
                          {visualizations.map((_, idx) => (
                            <button
                              key={idx}
                              className={`gallery-dot ${idx === currentImageIndex ? 'active' : ''}`}
                              onClick={() => setCurrentImageIndex(idx)}
                              aria-label={`Go to image ${idx + 1}`}
                            />
                          ))}
                        </div>
                      )}
                    </>
                  );
                })()}
              </div>
            </>
          )}
          
          {/* 2. FUTURE FORECASTS SECOND */}
          {result.prediction && (
            <>
              <div className="section-divider"></div>
              <h4>Future Forecasts</h4>
              <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                Starting from {formatDate(getCurrentMonthStart())}
              </p>
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
            </>
          )}
          
          {/* 3. MODEL PERFORMANCE METRICS LAST */}
          {result.metrics && Array.isArray(result.metrics) && (
            <>
              <div className="section-divider"></div>
              <h4>Model Performance Metrics</h4>
              <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                Based on test data up to September 2025
              </p>
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
                  {result.metrics.map((row, idx) => (
                    <tr key={idx}>
                      <td><strong>{row.Model}</strong></td>
                      <td className="number-cell">{row.MAE?.toFixed(2)}</td>
                      <td className="number-cell">{row.RMSE?.toFixed(2)}</td>
                      <td className="number-cell">{row.MAPE?.toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
