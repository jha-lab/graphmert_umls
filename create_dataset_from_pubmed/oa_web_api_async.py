import asyncio
import aiohttp
import time
import logging
import os
import xml.etree.ElementTree as ET

# Configuration
folder = '/scratch/gpfs/JHA/mb5157/large_data/diabetes_2025_6years'
pmcids_file   = os.path.join(folder, 'pmcids_diff.txt') # <--- set file with pmcid list
# =======
out_filename  = os.path.join(folder, 'pmcid.ftp')
err_filename  = os.path.join(folder, 'pmcid.err')
noa_filename  = os.path.join(folder, 'pmcid.noa')

NUM_PAPERS    = -1    # -1 = retirieve all in the file
max_retries   = 3
retry_delay   = 1     # seconds
start_from    = 1     # 1-based index
CONCURRENCY   = 10

log_filename = os.path.join(folder, 'url_ftp.log')
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s %(message)s')

async def fetch_xml(session: aiohttp.ClientSession, url: str) -> str | None:
    """Fetch XML from URL with retries."""
    for attempt in range(1, max_retries + 1):
        try:
            async with session.get(url) as resp:
                text = await resp.text()
                if resp.status == 200:
                    return text
                logging.info(f"Attempt {attempt}/{max_retries} for {url} returned {resp.status}")
        except Exception as e:
            logging.info(f"Attempt {attempt}/{max_retries} for {url} raised {e}")
        await asyncio.sleep(retry_delay * attempt)
    return None

async def process_pmcid(idx: int, pmcid: str,
                        session: aiohttp.ClientSession,
                        sem: asyncio.Semaphore,
                        file_lock: asyncio.Lock,
                        out_f, err_f, noa_f):
    """Download and process a single PMCID."""
    if idx < start_from - 1:
        return 0  # skip

    url = f'https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}'
    async with sem:
        xml_string = await fetch_xml(session, url)

    if not xml_string:
        async with file_lock:
            err_f.write(f"PMCID {pmcid}: failed after {max_retries} attempts\n")
        return 0

    logging.info(f"Got 200 for PMCID {pmcid}")

    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as e:
        async with file_lock:
            err_f.write(f"XML parse error for {pmcid}: {e}\n")
        return 0

    link = root.find('.//record/link')
    if link is not None:
        href = link.get('href')
        async with file_lock:
            out_f.write(href + "\n")
        return 1
    else:
        err_elem = root.find('error')
        if err_elem is not None and err_elem.attrib.get('code') == 'idIsNotOpenAccess':
            async with file_lock:
                noa_f.write(pmcid + "\n")
        else:
            async with file_lock:
                err_f.write(f"No link for {pmcid}: {xml_string}\n")
        return 0

async def main():
    pmcids = [line.strip() for line in open(pmcids_file) if line.strip()]
    sem = asyncio.Semaphore(CONCURRENCY)
    file_lock = asyncio.Lock()

    out_f = open(out_filename, 'a')
    err_f = open(err_filename, 'a')
    noa_f = open(noa_filename, 'a')

    downloaded = 0
    async with aiohttp.ClientSession() as session:
        tasks = [
            process_pmcid(idx, pmcid, session, sem, file_lock, out_f, err_f, noa_f)
            for idx, pmcid in enumerate(pmcids)
        ]
        for coro in asyncio.as_completed(tasks):
            got = await coro
            downloaded += got
            if NUM_PAPERS != -1 and downloaded >= NUM_PAPERS:
                break
            if downloaded and downloaded % 500 == 0:
                print(f"Downloaded {downloaded} papers")

    out_f.close()
    err_f.close()
    noa_f.close()

if __name__ == "__main__":
    asyncio.run(main())

