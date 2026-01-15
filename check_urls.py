import requests

urls = [
    "https://s3.amazonaws.com/fast-ai-imagelocal/VOCtrainval_11-May-2012.tar",
    "https://s3.amazonaws.com/fast-ai-imagelocal/VOCtrainval_06-Nov-2007.tar",
    "https://s3.amazonaws.com/fast-ai-imagelocal/VOCtest_06-Nov-2007.tar",
    "https://joseph-redmon.com/darknet/VOCtrainval_11-May-2012.tar", # Maybe?
    "https://github.com/ultralytics/yolov5/releases/download/v1.0/VOCtrainval_06-Nov-2007.tar"
]

for url in urls:
    try:
        r = requests.head(url, timeout=5)
        print(f"{url}: {r.status_code}")
    except Exception as e:
        print(f"{url}: {e}")
