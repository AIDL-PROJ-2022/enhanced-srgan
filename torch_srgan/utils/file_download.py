import os

def _quota_exceeded(first_chunk: bytes) -> bool:  # type: ignore[name-defined]
    try:
        return "Google Drive - Quota exceeded" in first_chunk.decode()
    except UnicodeDecodeError:
        return False


def download_file_from_google_drive(file_id: str, root: str, filename: Optional[str] = None, md5: Optional[str] = None):
    """Download a Google Drive file from  and place it in root.

    Args:
        file_id (str): id of file to be downloaded
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the id of the file.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    # Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    import requests
    url = "https://docs.google.com/uc?export=download"

    root = os.path.expanduser(root)
    if not filename:
        filename = file_id
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        session = requests.Session()

        response = session.get(url, params={'id': file_id}, stream=True)
        token = _get_confirm_token(response)

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(url, params=params, stream=True)

        # Ideally, one would use response.status_code to check for quota limits, but google drive is not consistent
        # with their own API, refer https://github.com/pytorch/vision/issues/2992#issuecomment-730614517.
        # Should this be fixed at some place in future, one could refactor the following to no longer rely on decoding
        # the first_chunk of the payload
        response_content_generator = response.iter_content(32768)
        first_chunk = None
        while not first_chunk:  # filter out keep-alive new chunks
            first_chunk = next(response_content_generator)

        if _quota_exceeded(first_chunk):
            msg = (
                f"The daily quota of the file {filename} is exceeded and it "
                f"can't be downloaded. This is a limitation of Google Drive "
                f"and can only be overcome by trying again later."
            )
            raise RuntimeError(msg)

        _save_response_content(itertools.chain((first_chunk, ), response_content_generator), fpath)
        response.close()


def _get_confirm_token(response: "requests.models.Response") -> Optional[str]:  # type: ignore[name-defined]
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def _save_response_content(
    response_gen: Iterator[bytes], destination: str,  # type: ignore[name-defined]
) -> None:
    with open(destination, "wb") as f:
        pbar = tqdm(total=None)
        progress = 0

        for chunk in response_gen:
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                progress += len(chunk)
                pbar.update(progress - pbar.n)
        pbar.close()


def _extract_tar(from_path: str, to_path: str, compression: Optional[str]) -> None:
    with tarfile.open(from_path, f"r:{compression[1:]}" if compression else "r") as tar:
        tar.extractall(to_path)


_ZIP_COMPRESSION_MAP: Dict[str, int] = {
    ".bz2": zipfile.ZIP_BZIP2,
    ".xz": zipfile.ZIP_LZMA,
}


def _extract_zip(from_path: str, to_path: str, compression: Optional[str]) -> None:
    with zipfile.ZipFile(
        from_path, "r", compression=_ZIP_COMPRESSION_MAP[compression] if compression else zipfile.ZIP_STORED
    ) as zip:
        zip.extractall(to_path)


_ARCHIVE_EXTRACTORS: Dict[str, Callable[[str, str, Optional[str]], None]] = {
    ".tar": _extract_tar,
    ".zip": _extract_zip,
}
_COMPRESSED_FILE_OPENERS: Dict[str, Callable[..., IO]] = {
    ".bz2": bz2.open,
    ".gz": gzip.open,
    ".xz": lzma.open,
}
_FILE_TYPE_ALIASES: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    ".tbz": (".tar", ".bz2"),
    ".tbz2": (".tar", ".bz2"),
    ".tgz": (".tar", ".gz"),
}


def _detect_file_type(file: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Detect the archive type and/or compression of a file.

    Args:
        file (str): the filename

    Returns:
        (tuple): tuple of suffix, archive type, and compression

    Raises:
        RuntimeError: if file has no suffix or suffix is not supported
    """
    suffixes = pathlib.Path(file).suffixes
    if not suffixes:
        raise RuntimeError(
            f"File '{file}' has no suffixes that could be used to detect the archive type and compression."
        )
    suffix = suffixes[-1]

    # check if the suffix is a known alias
    if suffix in _FILE_TYPE_ALIASES:
        return (suffix, *_FILE_TYPE_ALIASES[suffix])

    # check if the suffix is an archive type
    if suffix in _ARCHIVE_EXTRACTORS:
        return suffix, suffix, None

    # check if the suffix is a compression
    if suffix in _COMPRESSED_FILE_OPENERS:
        # check for suffix hierarchy
        if len(suffixes) > 1:
            suffix2 = suffixes[-2]

            # check if the suffix2 is an archive type
            if suffix2 in _ARCHIVE_EXTRACTORS:
                return suffix2 + suffix, suffix2, suffix

        return suffix, None, suffix

    valid_suffixes = sorted(set(_FILE_TYPE_ALIASES) | set(_ARCHIVE_EXTRACTORS) | set(_COMPRESSED_FILE_OPENERS))
    raise RuntimeError(f"Unknown compression or archive type: '{suffix}'.\nKnown suffixes are: '{valid_suffixes}'.")


def _decompress(from_path: str, to_path: Optional[str] = None, remove_finished: bool = False) -> str:
    r"""Decompress a file.

    The compression is automatically detected from the file name.

    Args:
        from_path (str): Path to the file to be decompressed.
        to_path (str): Path to the decompressed file. If omitted, ``from_path`` without compression extension is used.
        remove_finished (bool): If ``True``, remove the file after the extraction.

    Returns:
        (str): Path to the decompressed file.
    """
    suffix, archive_type, compression = _detect_file_type(from_path)
    if not compression:
        raise RuntimeError(f"Couldn't detect a compression from suffix {suffix}.")

    if to_path is None:
        to_path = from_path.replace(suffix, archive_type if archive_type is not None else "")

    # We don't need to check for a missing key here, since this was already done in _detect_file_type()
    compressed_file_opener = _COMPRESSED_FILE_OPENERS[compression]

    with compressed_file_opener(from_path, "rb") as rfh, open(to_path, "wb") as wfh:
        wfh.write(rfh.read())

    if remove_finished:
        os.remove(from_path)

    return to_path


def extract_archive(from_path: str, to_path: Optional[str] = None, remove_finished: bool = False) -> str:
    """Extract an archive.

    The archive type and a possible compression is automatically detected from the file name. If the file is compressed
    but not an archive the call is dispatched to :func:`decompress`.

    Args:
        from_path (str): Path to the file to be extracted.
        to_path (str): Path to the directory the file will be extracted to. If omitted, the directory of the file is
            used.
        remove_finished (bool): If ``True``, remove the file after the extraction.

    Returns:
        (str): Path to the directory the file was extracted to.
    """
    if to_path is None:
        to_path = os.path.dirname(from_path)

    suffix, archive_type, compression = _detect_file_type(from_path)
    if not archive_type:
        return _decompress(
            from_path,
            os.path.join(to_path, os.path.basename(from_path).replace(suffix, "")),
            remove_finished=remove_finished,
        )

    # We don't need to check for a missing key here, since this was already done in _detect_file_type()
    extractor = _ARCHIVE_EXTRACTORS[archive_type]

    extractor(from_path, to_path, compression)

    return to_path


def download_and_extract_archive(
    url: str,
    download_root: str,
    extract_root: Optional[str] = None,
    filename: Optional[str] = None,
    md5: Optional[str] = None,
    remove_finished: bool = False,
) -> None:
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print("Extracting {} to {}".format(archive, extract_root))
    extract_archive(archive, extract_root, remove_finished)