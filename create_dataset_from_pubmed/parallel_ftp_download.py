import os
import sys
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# ======== SETUP ======
    out_folder = '' # <--- set output folder here
    start_line = 0  # which line in the input file to start from
# ======== SETUP =====


def setup_logging(log_path):
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def download_url(url, dest_dir, timeout_entire, timeout_per_try, retry_limit, waitretry):
    """Download a single URL via wget with retries and overall timeout."""
    cmd = [
        "wget", url,
        "-P", dest_dir,
        "--timeout", str(timeout_per_try),
        "--tries", str(retry_limit),
        "--waitretry", str(waitretry)
    ]
    try:
        # overall timeout for the entire wget invocation
        subprocess.run(cmd, timeout=timeout_entire, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True, url
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return False, url

def main():
#     if len(sys.argv) < 2:
#         print("Usage: script.py <out_folder> [start_line]")
#         sys.exit(1)
# 
#     out_folder = sys.argv[1]
#     start_line = int(sys.argv[2]) if len(sys.argv) > 2 else 1


    url_file     = os.path.join(out_folder, 'pmcid.ftp')
    dest_dir     = os.path.join(out_folder, 'paper_archives')
    timeout_file = os.path.join(out_folder, 'ftp.timed_out')
    log_file     = os.path.join(out_folder, 'download.log')

    # Tunable parameters
    retry_limit    = 3     # --tries
    waitretry      = 5     # --waitretry
    timeout_try    = 60    # --timeout per attempt
    timeout_all    = 600   # overall timeout for each wget
    max_workers    = 10    # parallel jobs

    # Prep
    ensure_dirs(dest_dir)
    setup_logging(log_file)
    logging.info(f"Starting parallel download from line {start_line}")

    # Read and slice URLs
    with open(url_file) as f:
        lines = [l.strip() for l in f if l.strip()]
    urls = lines[start_line-1:]

    # Launch pool
    with open(timeout_file, 'a') as tuf:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(download_url, url, dest_dir, timeout_all,
                            timeout_try, retry_limit, waitretry): url
                for url in urls
            }
            for future in as_completed(futures):
                success, url = future.result()
                if success:
                    logging.info(f"Downloaded: {url}")
                    print(f"OK: {os.path.basename(url)}")
                else:
                    logging.warning(f"Failed: {url}")
                    tuf.write(url + "\n")

if __name__ == "__main__":
    main()

