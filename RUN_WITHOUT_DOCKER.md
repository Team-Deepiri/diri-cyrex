# Running Cyrex Without Docker

This guide explains how to run the Cyrex AI service directly on your machine without using Docker.

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- (Optional) CUDA-capable GPU for faster AI inference
- MongoDB (if you need database features)
- Redis (optional, for caching)

## Step 1: Navigate to Cyrex Directory

```bash
cd deepiri/diri-cyrex
```

## Step 2: Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

## Step 3: Install Dependencies

### Option A: Install Core Dependencies (Recommended for Development)

```bash
pip install -r requirements-core.txt
```

This installs the essential packages without heavy ML dependencies like `deepspeed` and `wandb`.

### Option B: Install All Dependencies (Full Installation)

```bash
pip install -r requirements.txt
```

**Note:** Some packages like `deepspeed` and `bitsandbytes` may fail to install on certain systems. This is okay - they're optional for basic functionality.

## Step 4: Set Up Environment Variables

1. Copy the example environment file:
   ```bash
   # Windows
   copy env.example.diri-cyrex .env
   
   # Linux/Mac
   cp env.example.diri-cyrex .env
   ```

2. Edit the `.env` file and configure at minimum:
   ```env
   # Required: AI Provider Configuration
   OPENAI_API_KEY=sk-your-openai-api-key-here
   OPENAI_MODEL=gpt-4o-mini
   
   # Optional: API Key for authentication
   CYREX_API_KEY=change-me
   
   # Optional: Backend URL (if connecting to Node.js backend)
   NODE_BACKEND_URL=http://localhost:5000
   
   # Optional: CORS Origin (for web frontend)
   CORS_ORIGIN=http://localhost:5173
   
   # Optional: Redis (if using caching)
   REDIS_CACHE_URL=redis://localhost:6379/1
   ```

## Step 5: Run Cyrex

### Development Mode (with auto-reload)

```bash
# From the diri-cyrex directory
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using Python Directly

You can also run it directly:
```bash
python -m app.main
```

Or:
```bash
cd app
python main.py
```

## Step 6: Verify It's Running

1. Check the health endpoint:
   ```bash
   curl http://localhost:8000/health
   ```

2. Or open in your browser:
   ```
   http://localhost:8000/health
   ```

3. View API documentation:
   ```
   http://localhost:8000/docs
   ```

## Common Issues and Solutions

### Issue: Missing Dependencies

If you get import errors, install the missing package:
```bash
pip install <package-name>
```

### Issue: Port Already in Use

If port 8000 is already in use, change it:
```bash
uvicorn app.main:app --reload --port 8001
```

Or set in `.env`:
```env
PORT=8001
```

### Issue: OpenAI API Key Not Working

Make sure your API key is set correctly in the `.env` file:
```env
OPENAI_API_KEY=sk-...
```

### Issue: Cannot Connect to Backend

If you're connecting to the Node.js backend, make sure:
1. The backend is running on the port specified in `NODE_BACKEND_URL`
2. The URL is correct (use `http://localhost:5000` for local development)

### Issue: CORS Errors

If you're accessing from a web frontend, make sure:
1. `CORS_ORIGIN` in `.env` matches your frontend URL
2. The frontend is sending requests to the correct port

## Environment Variables Reference

### Required (Minimum)
- `OPENAI_API_KEY` - Your OpenAI API key

### Recommended
- `OPENAI_MODEL` - Model to use (default: `gpt-4o-mini`)
- `CYREX_API_KEY` - API key for authentication
- `CORS_ORIGIN` - Frontend URL for CORS

### Optional
- `NODE_BACKEND_URL` - Backend service URL
- `REDIS_CACHE_URL` - Redis connection string
- `LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)
- `MAX_CONCURRENT_REQUESTS` - Max concurrent requests (default: 10)

## Running with Local AI (Free Alternative)

Instead of using OpenAI, you can use LocalAI:

1. Install LocalAI:
   ```bash
   # Using Docker
   docker run -d --name local-ai -p 8080:8080 localai/localai:latest
   ```

2. Update `.env`:
   ```env
   AI_PROVIDER=localai
   LOCALAI_API_BASE=http://localhost:8080/v1
   LOCALAI_MODEL=llama-3.2-1b-instruct:q4_k_m
   ```

3. Run Cyrex as normal - it will use LocalAI instead of OpenAI.

## Testing

Run the test suite:
```bash
pytest
```

Or run specific tests:
```bash
pytest tests/test_health.py
pytest tests/ai/
```

## API Endpoints

Once running, Cyrex provides:

- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `GET /docs` - Interactive API documentation
- `POST /agent/*` - AI agent endpoints
- `POST /agent/challenge/*` - Challenge generation
- `POST /agent/task/*` - Task processing
- And more...

See `http://localhost:8000/docs` for the full API documentation.

## Development Tips

1. **Auto-reload**: Use `--reload` flag for automatic restart on code changes
2. **Debugging**: Set `LOG_LEVEL=DEBUG` in `.env` for verbose logging
3. **Hot reload**: The `--reload` flag watches for file changes automatically
4. **Multiple workers**: Use `--workers N` for production (not with `--reload`)

## Next Steps

- Check the main README for more information
- See `docs/README_AI_TEAM.md` for AI team specific setup
- Review `app/settings.py` for all available configuration options

