# RAG Bot Deployment Checklist

## Pre-Deployment Testing

### 1. Local Environment Test
```bash
python test_local.py
```
This will verify:
- ✅ Environment variables are set
- ✅ All required packages can be imported
- ✅ Data files exist
- ✅ Basic functionality works

### 2. Manual Local Test
```bash
python rag_bot.py
```
Then test the endpoints:
- `GET http://localhost:5000/` - Health check
- `POST http://localhost:5000/query_rag` - RAG query

## Render Configuration

### 3. Environment Variables
In Render dashboard, set:
- `OPENAI_API_KEY` = Your actual OpenAI API key
- `PYTHON_VERSION` = 3.11 (should be auto-set by runtime.txt)

### 4. Build Configuration
- ✅ `render.yaml` - Service configuration
- ✅ `requirements.txt` - All dependencies listed
- ✅ `runtime.txt` - Python version specified
- ✅ `Procfile` - Gunicorn startup command

## File Structure Verification

### 5. Required Files Present
- ✅ `rag_bot.py` - Main application
- ✅ `data/my_structured_document.csv` - Your data file
- ✅ `requirements.txt` - Dependencies
- ✅ `render.yaml` - Render config
- ✅ `runtime.txt` - Python version
- ✅ `Procfile` - Startup command

### 6. Git Status
```bash
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

## Deployment Steps

### 7. Render Dashboard Setup
1. Connect your GitHub repository
2. Create new Web Service
3. Select your repository
4. Render should auto-detect the `render.yaml` configuration
5. Set environment variables in the dashboard
6. Deploy

### 8. Monitor Deployment
Watch for these common issues:
- **Build timeout**: Large requirements.txt might cause timeout
- **Memory issues**: Free tier has 512MB RAM limit
- **Environment variables**: Ensure OPENAI_API_KEY is set
- **File permissions**: Data files should be readable

### 9. Post-Deployment Testing
Once deployed, test:
- `GET https://your-app.onrender.com/` - Health check
- `POST https://your-app.onrender.com/query_rag` - RAG functionality

## Troubleshooting

### Common Issues:
1. **Build fails**: Check requirements.txt for conflicts
2. **Runtime errors**: Check Render logs for specific error messages
3. **Memory issues**: Consider upgrading to paid tier
4. **Cold start**: First request might be slow (normal)

### Log Monitoring:
- Check Render dashboard logs during deployment
- Monitor application logs for runtime errors
- Verify environment variables are loaded correctly

## Success Indicators
- ✅ Build completes without errors
- ✅ Health check returns status 200
- ✅ RAG queries work properly
- ✅ No memory/timeout errors in logs

## Next Steps After Deployment
1. Test all endpoints thoroughly
2. Monitor performance and memory usage
3. Set up any additional monitoring if needed
4. Consider upgrading to paid tier if hitting limits 