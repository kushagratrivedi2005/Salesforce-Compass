# Flask API Implementation & Deployment Guide

## 📋 Summary of Implementation

This document outlines all the changes made to implement the Flask API server for the Vehicle Sales Forecasting System and provides guidance for production deployment.

---

## 🔨 What Was Built

### 1. Core API Server (`Code/app.py`)
- **Flask application** with 6 REST endpoints
- **CORS support** for cross-origin requests
- **Custom JSON encoder** for numpy/pandas data types
- **Error handling** with standardized responses
- **Data loading utilities** from CSV files
- **Model information helpers** with descriptions

### 2. Dependencies (`requirements_api.txt`)
- Flask 2.3.3 and flask-cors 4.0.0
- Pandas 2.0.3 and numpy 1.24.3
- Existing ML dependencies (statsmodels, scikit-learn, pmdarima)
- Development tools (pytest, pytest-flask)
- Production server (gunicorn)

### 3. Documentation (`API_Documentation.md`)
- Complete API reference with examples
- Request/response schemas
- Error handling documentation
- Setup instructions
- Production deployment guidance

### 4. Demo & Testing (`Code/api_demo.py`)
- Comprehensive test script for all endpoints
- Error handling demonstrations
- Summary report generation
- Example usage patterns

### 5. Startup Scripts
- **`Code/start_api.py`** - Python startup with validation
- **`setup_api.ps1`** - PowerShell setup for Windows

### 6. Documentation Updates
- Updated main `documentation.md` with API section
- Added setup instructions and usage examples

---

## 🏗️ API Architecture

### Endpoints Implemented

| Method | Endpoint | Purpose | Data Source |
|--------|----------|---------|-------------|
| GET | `/health` | Health check | N/A |
| GET | `/api/v1/predictions` | All model predictions | `forecast_predictions.csv` |
| GET | `/api/v1/metrics` | Model performance metrics | `forecast_metrics.csv` |
| GET | `/api/v1/models` | Model information | Both CSV files |
| GET | `/api/v1/forecast/{model}` | Specific model forecast | Both CSV files |
| GET | `/api/v1/comparison` | Model comparison | Both CSV files |

### Data Flow
```
CSV Files → Data Loading Functions → API Endpoints → JSON Response
```

### Response Format
- **Consistent JSON structure** across all endpoints
- **ISO 8601 timestamps** for all dates
- **Metadata sections** with request timestamps
- **Error responses** with standardized format

---

## 🚀 Deployment Requirements

### Development vs Production Changes

#### 1. Security Enhancements

**Current State (Development):**
```python
# Debug mode enabled
app.run(debug=True, host='0.0.0.0', port=5000)

# No authentication
# No rate limiting
# Detailed error messages exposed
```

**Production Requirements:**
```python
# Production configuration
app.config['DEBUG'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')

# Add authentication middleware
from flask_jwt_extended import JWTManager

# Add rate limiting
from flask_limiter import Limiter
limiter = Limiter(app, key_func=get_remote_address)

# Sanitize error messages
def create_error_response(message: str, status_code: int = 500, include_details: bool = False):
    if not app.config['DEBUG'] and not include_details:
        message = "Internal server error"  # Generic message
```

#### 2. Server Configuration

**Current State:**
```python
# Development server
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

**Production Requirements:**
```bash
# Use production WSGI server
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 app:app

# Or with systemd service
sudo systemctl enable forecasting-api
sudo systemctl start forecasting-api
```

#### 3. Environment Configuration

**Production Environment Variables:**
```bash
# Required environment variables
export FLASK_ENV=production
export SECRET_KEY=your-secret-key-here
export DATABASE_URL=postgresql://...  # If using database
export REDIS_URL=redis://...          # For caching/rate limiting
export LOG_LEVEL=INFO
export API_RATE_LIMIT=100             # Requests per minute
```

#### 4. Data Storage & Caching

**Current State:**
```python
# Loads CSV files on every request
def load_forecast_data() -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    predictions_df = pd.read_csv(PREDICTIONS_FILE, index_col=0, parse_dates=True)
    metrics_df = pd.read_csv(METRICS_FILE, index_col=0)
```

**Production Improvements:**
```python
# Add caching layer
from flask_caching import Cache
cache = Cache(app, config={'CACHE_TYPE': 'redis'})

@cache.memoize(timeout=3600)  # Cache for 1 hour
def load_forecast_data():
    # Same implementation but cached

# Or use database storage
from sqlalchemy import create_engine
engine = create_engine(DATABASE_URL)

def load_forecast_data_from_db():
    predictions_df = pd.read_sql("SELECT * FROM predictions", engine)
    metrics_df = pd.read_sql("SELECT * FROM metrics", engine)
```

---

## 🔧 Production Deployment Checklist

### 1. Infrastructure Setup

- [ ] **Server provisioning** (AWS EC2, Google Cloud, Azure, etc.)
- [ ] **Domain name** and SSL certificate setup
- [ ] **Load balancer** configuration (if multiple instances)
- [ ] **Database setup** (PostgreSQL recommended)
- [ ] **Cache setup** (Redis recommended)
- [ ] **Monitoring setup** (Prometheus, Grafana, or CloudWatch)

### 2. Security Implementation

- [ ] **Authentication system** (JWT, OAuth2, or API keys)
```python
from flask_jwt_extended import jwt_required, get_jwt_identity

@app.route('/api/v1/predictions')
@jwt_required()
def get_all_predictions():
    user_id = get_jwt_identity()
    # Implementation with user validation
```

- [ ] **Rate limiting** per user/API key
```python
from flask_limiter import Limiter

limiter = Limiter(
    app,
    key_func=lambda: get_jwt_identity() or get_remote_address(),
    default_limits=["100 per hour"]
)

@app.route('/api/v1/predictions')
@limiter.limit("10 per minute")
def get_all_predictions():
    # Implementation
```

- [ ] **Input validation** and sanitization
```python
from marshmallow import Schema, fields, validate

class ForecastRequestSchema(Schema):
    model_name = fields.Str(required=True, validate=validate.OneOf(['ARIMA', 'ETS', 'SARIMAX']))
    date_range = fields.Dict()
```

- [ ] **HTTPS enforcement**
```python
from flask_talisman import Talisman

# Force HTTPS
Talisman(app, force_https=True)
```

### 3. Performance Optimizations

- [ ] **Database migration** from CSV files
```sql
-- Create tables for production
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    prediction_value DECIMAL(15,2),
    actual_value DECIMAL(15,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_predictions_date_model ON predictions(date, model_name);
```

- [ ] **Caching implementation**
```python
# Cache expensive operations
@cache.memoize(timeout=3600)
def get_model_rankings():
    # Implementation

@cache.memoize(timeout=7200)  # 2 hours
def get_all_predictions():
    # Implementation
```

- [ ] **API response compression**
```python
from flask_compress import Compress
Compress(app)
```

### 4. Monitoring & Logging

- [ ] **Structured logging**
```python
import logging
import json

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
        handlers=[
            logging.FileHandler('/var/log/forecasting-api/app.log'),
            logging.StreamHandler()
        ]
    )

# Log API requests
@app.before_request
def log_request_info():
    logger.info(f"API Request: {request.method} {request.url}")
```

- [ ] **Health checks with detailed status**
```python
@app.route('/health/detailed')
def detailed_health_check():
    checks = {
        'database': check_database_connection(),
        'cache': check_redis_connection(),
        'forecast_data': check_forecast_data_freshness(),
        'disk_space': check_disk_space()
    }
    
    overall_status = all(checks.values())
    
    return jsonify({
        'status': 'healthy' if overall_status else 'unhealthy',
        'checks': checks,
        'timestamp': datetime.utcnow().isoformat()
    }), 200 if overall_status else 503
```

- [ ] **Metrics collection**
```python
from prometheus_flask_exporter import PrometheusMetrics

metrics = PrometheusMetrics(app)
metrics.info('app_info', 'Application info', version='1.0.0')

# Custom metrics
request_count = metrics.counter(
    'api_requests_total', 'Total API requests',
    labels={'method': lambda: request.method, 'endpoint': lambda: request.endpoint}
)
```

### 5. Data Management

- [ ] **Automated data updates**
```python
# Celery task for periodic updates
from celery import Celery

celery = Celery('forecasting-api')

@celery.task
def update_forecast_data():
    """Run forecasting pipeline and update database"""
    try:
        # Run main_pipeline.py
        subprocess.run(['python', 'main_pipeline.py'], check=True)
        
        # Update database with new results
        update_database_from_csv()
        
        # Clear cache
        cache.clear()
        
        logger.info("Forecast data updated successfully")
    except Exception as e:
        logger.error(f"Failed to update forecast data: {e}")
```

- [ ] **Data backup strategy**
```bash
#!/bin/bash
# Backup script
pg_dump forecasting_db > /backups/forecasting_$(date +%Y%m%d_%H%M%S).sql
aws s3 cp /backups/forecasting_*.sql s3://forecasting-backups/
```

---

## 📦 Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy application code
COPY Code/ .
COPY dataset/ ./dataset/

# Create non-root user
RUN useradd -m -u 1000 forecasting && chown -R forecasting:forecasting /app
USER forecasting

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "app:app"]
```

### Docker Compose (with dependencies)
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - DATABASE_URL=postgresql://forecasting:password@postgres:5432/forecasting_db
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    volumes:
      - ./logs:/app/logs

  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=forecasting_db
      - POSTGRES_USER=forecasting
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api

volumes:
  postgres_data:
  redis_data:
```

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow
```yaml
name: Deploy Forecasting API

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements_api.txt
      - name: Run tests
        run: pytest Code/tests/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production
        run: |
          # Deploy to your chosen platform
          # (AWS ECS, Google Cloud Run, etc.)
```

---

## 📊 Monitoring & Alerting

### Key Metrics to Monitor

1. **API Performance**
   - Response times (95th percentile < 2s)
   - Error rates (< 1%)
   - Request volume
   - Cache hit rates

2. **System Resources**
   - CPU usage (< 80%)
   - Memory usage (< 85%)
   - Disk space (> 20% free)
   - Database connections

3. **Data Freshness**
   - Last forecast update timestamp
   - Data validation checks
   - Model performance drift

### Alerting Rules
```yaml
groups:
- name: forecasting-api
  rules:
  - alert: HighErrorRate
    expr: rate(flask_http_request_exceptions_total[5m]) > 0.01
    for: 5m
    annotations:
      summary: "High error rate detected"
      
  - alert: DataOutdated
    expr: (time() - forecast_data_last_update) > 86400
    for: 0m
    annotations:
      summary: "Forecast data is more than 24 hours old"
```

---

## 🔒 Security Hardening

### 1. API Security Headers
```python
from flask_talisman import Talisman

Talisman(app, {
    'force_https': True,
    'strict_transport_security': True,
    'content_security_policy': {
        'default-src': "'self'",
        'script-src': "'self'",
        'style-src': "'self' 'unsafe-inline'"
    }
})
```

### 2. Input Validation
```python
from marshmallow import Schema, fields, ValidationError

class ModelForecastSchema(Schema):
    model_name = fields.Str(required=True, validate=validate.OneOf(['ARIMA', 'ETS', 'SARIMAX']))
    start_date = fields.Date(required=False)
    end_date = fields.Date(required=False)

@app.route('/api/v1/forecast/<model_name>')
def get_model_forecast(model_name):
    schema = ModelForecastSchema()
    try:
        result = schema.load({'model_name': model_name, **request.args})
    except ValidationError as err:
        return create_error_response(f"Validation error: {err.messages}", 400)
```

### 3. API Rate Limiting
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["1000 per hour", "100 per minute"]
)

@app.route('/api/v1/predictions')
@limiter.limit("10 per minute")
def get_all_predictions():
    # Implementation
```

---

## 📝 Summary of Required Changes for Production

### Immediate Requirements
1. **Security**: Add authentication, rate limiting, input validation
2. **Performance**: Implement caching, database storage, connection pooling
3. **Monitoring**: Add logging, metrics, health checks
4. **Configuration**: Environment-based config, secrets management

### Infrastructure Requirements
1. **Server**: Production-grade server (not Flask dev server)
2. **Database**: PostgreSQL or similar for data persistence
3. **Cache**: Redis for performance optimization
4. **Load Balancer**: NGINX or cloud load balancer
5. **SSL**: HTTPS certificates and configuration

### Operational Requirements
1. **CI/CD**: Automated testing and deployment pipeline
2. **Monitoring**: Application and infrastructure monitoring
3. **Backup**: Data backup and disaster recovery plan
4. **Documentation**: API versioning and changelog management

This implementation provides a solid foundation for a production-ready forecasting API, but requires the security, performance, and operational enhancements outlined above for enterprise deployment.
