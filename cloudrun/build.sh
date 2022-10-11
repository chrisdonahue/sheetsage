cp Dockerfile ../Dockerfile;\
gcloud builds submit \
	../ \
	--ignore-file cloudrun/ignore \
	--billing-project sheet-sage \
	--tag gcr.io/sheet-sage/backend \
;\
rm ../Dockerfile
