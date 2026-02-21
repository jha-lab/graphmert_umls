Pipeline:

1. Run the search query on PMC and download as file only pmcid list (the download takes a while), save as pmc_diabetes.ids.
2. Launch oa_web_api_async.py. It retrieves ftp addresses to download from for each pmc listed in pmcid.
3. Run parallel_ftp_download.py. It will download papers using the ftp list file from the previous step (specify filename in the script)
4. Run untar.py -- specify parent folder and destination.
5. Run remove_non-english_and_bad_pmcids.py: it will create the list of pmcid folders that are non-english papers or with errors.
5. Run remove_other_lang.sh. It will remove bad pmcid folders from the previous step -- set the file got on the previous step

After that create dataset with create_dataset_from_pubmed.py -- set all the params in the beginning of the file. It is better to allocate cpus for this job.
