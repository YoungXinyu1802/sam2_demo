# Restart Instructions to Apply LoRA Changes

## The Issue
The nginx proxy in the frontend container was missing the new LoRA endpoint configurations, causing 405 errors.

## Fixed Files
- ✅ `demo/frontend/nginx.conf` - Added `/train_lora` and `/generate_lora_candidates` proxy rules
- ✅ `demo/backend/server/app.py` - Added LoRA endpoints

## Restart Steps

### Option 1: Restart Frontend Container (Recommended)
```bash
cd /home/ubuntu/sam2/sam2/demo
docker-compose restart frontend

# Wait a moment, then check if it's running
docker-compose ps frontend
```

### Option 2: Rebuild Frontend (If restart doesn't work)
```bash
cd /home/ubuntu/sam2/sam2/demo
docker-compose down frontend
docker-compose up -d --build frontend

# Check logs
docker-compose logs -f frontend
```

### Option 3: Restart Everything (Nuclear option)
```bash
cd /home/ubuntu/sam2/sam2/demo
docker-compose down
docker-compose up -d --build

# Check status
docker-compose ps
```

## Verify the Fix

After restarting, test the endpoints:
```bash
# Test from inside the frontend container
docker exec -it $(docker ps -q -f name=frontend) curl -X POST http://backend:5000/train_lora -H "Content-Type: application/json" -d '{}'

# Or test via the proxy
curl -X POST http://localhost/train_lora -H "Content-Type: application/json" -d '{}'
```

You should get a JSON error response (not 405), which means the endpoint is accessible!

## Next Steps
1. Restart the frontend container using one of the options above
2. Refresh your browser (hard refresh: Ctrl+Shift+R or Cmd+Shift+R)
3. Try enabling LIT_LoRA mode again
4. The 405 error should be gone! 🎉

