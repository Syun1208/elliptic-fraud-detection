mkdir -p raw
kaggle datasets download -d ellipticco/elliptic-data-set -p raw
unzip raw/elliptic-data-set.zip -d raw
rm raw/elliptic-data-set.zip
