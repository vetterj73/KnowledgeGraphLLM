set -e
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m gunicorn app.main:app -c gunicorn.conf.py