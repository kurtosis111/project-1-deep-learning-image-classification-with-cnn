gcloud auth login
gcloud config set project image-classification-474017  ## image-classification-1759560331


docker build --platform=linux/amd64 -t flask-inception .


<!-- 
df -h  ## to check disk space
docker system prune -a -f ## clean up containers 
-->

docker tag flask-inception gcr.io/image-classification-474017/flask-inception
docker push gcr.io/image-classification-474017/flask-inception


gcloud run deploy flask-inception \
  --image gcr.io/image-classification-474017/flask-inception \
  --platform managed \
  --allow-unauthenticated \
  --memory 4Gi
