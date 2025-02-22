import os
import pathlib
import urllib.parse

is_device = lambda src: src.startswith(":") and src[1:].isdigit()

def resolve_src(src):
    
    if is_device(src):
        return "device"
    
    src = src.strip()
    # Check if it's a local file
    if os.path.exists(src) or pathlib.Path(src).exists():
        return "file"

    parsed = urllib.parse.urlparse(src)
    if parsed.scheme in ("http", "https", "ftp"):
        return "url"
    
    return None