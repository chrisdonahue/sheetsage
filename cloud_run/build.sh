cp Dockerfile ../Dockerfile;\
gcloud builds submit \
	../ \
	--ignore-file cloud_run/ignore \
	--billing-project sheet-sage \
	--tag gcr.io/sheet-sage/backend \
;\
rm ../Dockerfile
