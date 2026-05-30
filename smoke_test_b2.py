"""
Smoke test for Backblaze B2 connection.

Reads creds from .env, then uploads -> lists -> downloads -> deletes
a tiny test object. Prints OK on each step or fails loudly.

Run: python smoke_test_b2.py
Deps: pip install boto3
"""
import json
import os
import sys
from pathlib import Path


def load_env(path: Path) -> None:
    if not path.exists():
        sys.exit(f"ERROR: {path} not found.")
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env(Path(__file__).with_name(".env"))

REQUIRED = ["B2_KEY_ID", "B2_APPLICATION_KEY", "B2_BUCKET", "B2_ENDPOINT", "B2_REGION"]
missing = [k for k in REQUIRED if not os.environ.get(k)]
if missing:
    sys.exit(f"ERROR: missing env vars in .env: {missing}")

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    sys.exit("ERROR: boto3 not installed. Run: pip install boto3")

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["B2_ENDPOINT"],
    aws_access_key_id=os.environ["B2_KEY_ID"],
    aws_secret_access_key=os.environ["B2_APPLICATION_KEY"],
    region_name=os.environ["B2_REGION"],
)

BUCKET = os.environ["B2_BUCKET"]
KEY = "smoke-test/hello.json"
PAYLOAD = json.dumps(
    {"hello": "backblaze", "project": "emerging-destinations"}
).encode()

print(f"-> Uploading s3://{BUCKET}/{KEY}")
try:
    s3.put_object(Bucket=BUCKET, Key=KEY, Body=PAYLOAD)
except ClientError as e:
    sys.exit(f"FAIL on upload: {e}")
print("   OK")

print(f"-> Listing bucket {BUCKET} (prefix=smoke-test/)")
resp = s3.list_objects_v2(Bucket=BUCKET, Prefix="smoke-test/")
keys = [obj["Key"] for obj in resp.get("Contents", [])]
print(f"   found {len(keys)} object(s): {keys}")
if KEY not in keys:
    sys.exit(f"FAIL: {KEY} not in listing")

print(f"-> Downloading {KEY}")
body = s3.get_object(Bucket=BUCKET, Key=KEY)["Body"].read()
if body != PAYLOAD:
    sys.exit(f"FAIL: body mismatch. got {body!r}")
print(f"   OK ({len(body)} bytes match)")

print(f"-> Deleting {KEY}")
s3.delete_object(Bucket=BUCKET, Key=KEY)
print("   OK")

print("\nAll good. B2 is connected.")
