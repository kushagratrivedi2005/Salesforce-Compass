# Frontend Development Documentation

## Overview

This document describes the development process, technology choices, and design decisions for the Vehicle Sales Forecasting System frontend application.

---

## Technology Stack

### Core Technologies

#### React + Vite
**Why React?**
- Component-based architecture for reusable UI elements
- Declarative approach makes the code more predictable and easier to debug
- Large ecosystem and community support
- Excellent for building interactive single-page applications

**Why Vite?**
- Fast development server with Hot Module Replacement (HMR)
- Optimized build process using native ES modules
- Modern, lightweight alternative to Create React App
- Faster cold starts and updates during development

**Setup Command:**
```bash
npm create vite@latest . -- --template react
```

### State Management
- **React Hooks (useState)**: For managing form inputs, loading states, and API responses
- No external state management library needed due to simple, localized state requirements

### HTTP Client
- **Fetch API**: Native browser API for making HTTP requests to the Flask backend
- Lightweight and no additional dependencies required

---

## Project Structure

```
frontend/
├── src/
│   ├── App.jsx           # Main application component with form and results
│   ├── App.css           # Component-specific styles
│   ├── index.css         # Global styles and theme variables
│   ├── main.jsx          # Application entry point
├── index.html            # HTML template
├── package.json          # Dependencies and scripts
├── vite.config.js        # Vite configuration
└── eslint.config.js      # ESLint configuration
```

---

## Development Process

### Phase 1: Initial Setup
1. **Created React Project with Vite**
   - Initialized in the `Code/frontend` directory
   - Selected React template for modern development

2. **Connected to Backend API**
   - Configured CORS in Flask backend to allow cross-origin requests
   - Set up POST endpoint communication at `http://127.0.0.1:5000/predict`

### Phase 2: Form Implementation

#### Input Fields Designed:
1. **Target Category Dropdown**
   - 17 vehicle categories (TWO WHEELER, LIGHT PASSENGER VEHICLE, etc.)
   - Default: "FOUR WHEELER (Invalid Carriage)"

2. **Test Months Input**
   - Range: 3-9 months
   - Purpose: Define how many months of data to use for model testing
   - Default: 6 months

3. **Future Forecast Months Input**
   - Range: 6-60 months
   - Purpose: Define forecast horizon
   - Default: 12 months

4. **Exogenous Variables Selection**
   - Toggle: "Use Top-K Exogenous Variables"
   - **When enabled**: Multi-select checkboxes for candidate variables
     - interest_rate, repo_rate, holiday_count, major_national_holiday, major_religious_holiday
   - **When disabled**: Manual selection of interest_rate and repo_rate
   - Top-K input to specify how many variables to use

#### Form Handling Logic:
```javascript
const handleChange = (e) => {
  const { name, value, type, checked } = e.target;
  // Conditional logic for checkboxes
  // Number conversion for numeric inputs
  // State updates using React hooks
};
```

### Phase 3: API Integration

#### Request Payload Construction:
```javascript
const payload = {
  TARGET: `category_${form.TARGET}`,  // Prefixed with "category_"
  TEST_MONTHS: form.TEST_MONTHS,
  FUTURE_FORECAST_MONTHS: form.FUTURE_FORECAST_MONTHS,
  USE_TOP_K_EXOGS: form.USE_TOP_K_EXOGS,
  CANDIDATE_EXOGS: form.CANDIDATE_EXOGS,
  MANUAL_EXOGS: form.MANUAL_EXOGS,
  TOP_K_EXOGS: form.TOP_K_EXOGS,
  START_DATE: getCurrentMonthStart()
};
```

#### Response Handling:
- **Success**: Display metrics, predictions, and visualizations
- **Error**: Show error message in red alert box
- **Loading**: Disable button and show loading spinner

#### Date Formatting:
- Backend returns dates as "2025-10-01"
- Frontend converts to "October 2025" for better readability
```javascript
const formatDate = (dateStr) => {
  const date = new Date(dateStr);
  return date.toLocaleDateString('en-US', { month: 'long', year: 'numeric' });
};
```

### Phase 4: Visualization Integration

#### Challenge:
Initially, visualizations were not appearing on the webpage.

#### Solution:
1. Modified backend to encode PNG images as base64
2. Sent visualization data in API response
3. Displayed images using data URIs in React:
```javascript
<img src={`data:image/png;base64,${base64Data}`} alt={filename} />
```

#### Implementation:
- Backend reads all PNG files from `forecast_results/visualizations/`
- Each visualization is base64-encoded and included in response
- Frontend dynamically renders all available visualizations

### Phase 5: Data Display

#### Evolution of Display Format:

**Initial Approach**: JSON pre-formatted blocks
```javascript
<pre>{JSON.stringify(result, null, 2)}</pre>
```

**Final Approach**: Professional HTML tables

**Model Performance Metrics Table:**
- Columns: Model, MAE, RMSE, MAPE (%)
- Styled with alternating row colors
- Hover effects for better UX
- Number formatting to 2 decimal places

**Future Forecasts Table:**
- Rows: Months (formatted as "October 2025")
- Columns: One per model (ARIMA, ETS, SARIMAX)
- Horizontal scrolling for responsiveness
- Monospace font for numeric values

---

## Design Decisions

### Design Evolution

#### Version 1: Light Theme
- Initial design with light background
- Poor readability and unprofessional appearance

#### Version 2: Dark Theme with Gradients
- Dark background with gradient overlays
- Purple/indigo gradient accent colors
- Emojis in section headers (📊, 📈, 🔮)
- Modern but too flashy for professional use

#### Version 3: Minimal Professional Dark Theme (Final)
- **Background**: Deep dark (#0A0E1A) - professional and easy on eyes
- **Cards**: Subtle gray-blue (#111827) - clear separation
- **Text**: Bright white (#F9FAFB) - excellent contrast
- **Accent**: Professional blue (#3B82F6) - trust and clarity
- **Numbers**: Light blue (#60A5FA) - highlights data
- **No emojis**: Clean, minimal aesthetic
- **Solid colors**: No gradients, professional appearance

### Color Palette
```css
--primary-color: #3B82F6;        /* Professional blue for buttons */
--background-dark: #0A0E1A;      /* Deep dark background */
--background-card: #111827;      /* Card background */
--background-secondary: #1F2937; /* Input backgrounds */
--text-primary: #F9FAFB;         /* White text */
--text-secondary: #9CA3AF;       /* Gray text for labels */
--border-color: #374151;         /* Subtle borders */
--accent-blue: #60A5FA;          /* Number highlighting */
```

### Layout Structure

#### Form Section:
- Grid layout for inputs (responsive, auto-fit)
- Grouped related fields together
- Conditional rendering based on checkbox state
- Full-width submit button with loading state

#### Results Section:
1. **Model Performance Metrics** (first)
   - Most important for evaluating model quality
   - Table format for easy comparison

2. **Future Forecasts** (second)
   - Primary output users care about
   - Table with dates as rows, models as columns

3. **Forecast Visualizations** (last)
   - Moved to bottom per user request
   - Visual confirmation of tabular data
   - White background for graph visibility

### Responsive Design
- Mobile-friendly with flexible layouts
- Tables scroll horizontally on small screens
- Grid collapses to single column on mobile
- Proper touch targets for mobile users

### User Experience Enhancements

1. **Loading State**
   - Button disabled during API call
   - Spinner animation shows progress
   - Text changes to "Generating Forecast..."

2. **Error Handling**
   - Clear error messages in red alert box
   - Network errors handled gracefully
   - Error details displayed for debugging

3. **Form Validation**
   - HTML5 validation (required fields, min/max)
   - Number inputs restricted to valid ranges
   - Immediate visual feedback on focus

4. **Accessibility**
   - Semantic HTML structure
   - Proper label associations (htmlFor)
   - High contrast text (WCAG compliant)
   - Keyboard navigation support

---

## Technical Implementation Details

### State Management Pattern
```javascript
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
```

### Conditional Rendering
- Used ternary operators for show/hide logic
- Conditional checkbox groups based on `USE_TOP_K_EXOGS`
- Results only displayed when available

### Dynamic Table Generation
```javascript
{Object.entries(result.prediction).map(([filename, base64Data]) => (
  <div key={filename}>
    <img src={`data:image/png;base64,${base64Data}`} />
  </div>
))}
```

### CSS Architecture
- CSS Variables for theme consistency
- Separate files: `index.css` (global) and `App.css` (component)
- Mobile-first responsive design
- Reusable utility classes

---

## Challenges and Solutions

### Challenge 1: CORS Errors
**Problem**: Browser blocked requests to Flask backend
**Solution**: Added `flask-cors` to backend and enabled CORS
```python
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
```

### Challenge 2: Date Formatting
**Problem**: Backend returns ISO dates (2025-10-01), hard to read
**Solution**: JavaScript date formatting function
```javascript
const formatDate = (dateStr) => {
  const date = new Date(dateStr);
  return date.toLocaleDateString('en-US', { 
    month: 'long', 
    year: 'numeric' 
  });
};
```

### Challenge 3: Missing Visualizations
**Problem**: Graphs not appearing in results
**Solution**: 
- Backend: Base64 encode images and include in JSON
- Frontend: Render using data URIs

### Challenge 4: Section Ordering
**Problem**: Visualizations appeared first, overwhelming the page
**Solution**: Reordered to show metrics and predictions first, then visualizations

### Challenge 5: Dark Theme Readability
**Problem**: Initial dark theme had poor contrast
**Solution**: 
- Deeper background colors
- Brighter white text (#F9FAFB)
- Professional blue accents instead of purple gradients

---

## Future Enhancements

### Potential Improvements:
1. **Export Functionality**: Download predictions as CSV or PDF
2. **Comparison Mode**: Compare multiple forecasts side-by-side
3. **Historical Data View**: Show past predictions vs. actuals
4. **Real-time Updates**: WebSocket connection for live forecasts
5. **Chart Interactions**: Zoom, pan, and filter visualizations
6. **Model Explanations**: Tooltips explaining each model type
7. **Save Configurations**: Store frequently-used parameter sets
8. **Dark/Light Mode Toggle**: User preference for theme

---

## Performance Considerations

- **Lazy Loading**: Future enhancement for large datasets
- **Debouncing**: Could be added for real-time input validation
- **Memoization**: Consider React.memo for table components
- **Code Splitting**: Vite handles this automatically

---

## Dependencies

```json
{
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.3.4",
    "vite": "^6.0.5"
  }
}
```

**No additional libraries needed**:
- No UI component library (custom components)
- No charting library (backend generates images)
- No form library (React state management sufficient)
- No routing library (single-page application)

---

## Conclusion

The frontend was developed with a focus on:
- **Simplicity**: Minimal dependencies, straightforward architecture
- **Professionalism**: Clean, minimal dark theme with excellent readability
- **Usability**: Intuitive form, clear data presentation, responsive design
- **Performance**: Fast development server, optimized builds with Vite
- **Maintainability**: Component-based structure, clear separation of concerns

The result is a modern, professional web application that effectively presents complex forecasting data in an accessible and visually appealing manner.
