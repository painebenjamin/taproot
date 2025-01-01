import io
import os
import requests

from typing import Optional, Callable, Union, BinaryIO, Iterator, List, Sequence, Tuple, Any
from functools import partial, lru_cache
from contextlib import contextmanager

from concurrent.futures import ThreadPoolExecutor, Future, as_completed

from .misc_util import get_secret
from .string_util import human_size, human_duration
from .log_util import logger

__all__ = [
    "check_download_file",
    "check_download_files",
    "check_download_file_to_dir",
    "check_download_files_to_dir",
    "get_file_name_from_url",
    "get_file_size_from_url",
    "get_domain_from_url",
    "get_top_level_domain_from_url",
    "get_download_text_callback",
    "file_is_downloaded",
    "file_is_downloaded_to_dir",
    "check_remove_interrupted_download",
    "retrieve_uri",
]

def file_matches_checksum(
    path: str,
    checksum: str,
    chunk_size: int=4096
) -> bool:
    """
    Checks if a file matches a checksum.

    >>> import tempfile
    >>> tempfile = tempfile.NamedTemporaryFile()
    >>> open(tempfile.name, "wb").write(b"test")
    4
    >>> file_matches_checksum(tempfile.name, "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08")
    True

    :param path: The path to the file
    :param checksum: The checksum to check against
    :param chunk_size: The size of the chunks to read. Defaults to 4096 bytes.
    :return: True if the checksum matches, False otherwise
    """
    import hashlib
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            data = fh.read(chunk_size)
            if not data:
                break
            hasher.update(data)
    return hasher.hexdigest() == checksum

@lru_cache(maxsize=128)
def get_domain_from_url(url: str) -> str:
    """
    Gets a domain from a URL.

    >>> get_domain_from_url("http://example.com")
    'example.com'
    >>> get_domain_from_url("http://example.com/file.txt")
    'example.com'
    >>> get_domain_from_url("https://test.example.com/")
    'test.example.com'

    :param url: The URL to get the domain from
    :return: The domain
    """
    from urllib.parse import urlparse
    parsed_url = urlparse(url)
    return parsed_url.netloc.split(":")[0]

@lru_cache(maxsize=128)
def get_top_level_domain_from_url(url: str) -> str:
    """
    Gets the top-level domain from a URL.

    >>> get_domain_from_url("http://example.com")
    'example.com'
    >>> get_domain_from_url("http://www.example.com/file.txt")
    'example.com'
    >>> get_domain_from_url("https://test.example.com/")
    'example.com'
    """
    domain = get_domain_from_url(url)
    domain_parts = domain.split(".")
    if len(domain_parts) > 2:
        return ".".join(domain_parts[-2:])
    return domain

@lru_cache(maxsize=128)
def get_file_name_from_url(url: str) -> str:
    """
    Gets a filename from a URL.

    Checks for common ways to specify filenames in URLs,
    before falling back to the last part of the URL.

    >>> get_file_name_from_url("http://example.com/file.txt")
    'file.txt'
    >>> get_file_name_from_url("http://example.com/file.txt?filename=example.txt")
    'example.txt'
    >>> get_file_name_from_url("http://example.com/file.txt?response-content-disposition=attachment; filename=example.txt")
    'example.txt'

    :param url: The URL to get the filename from
    :return: The filename
    """
    from urllib.parse import urlparse, parse_qs
    parsed_url = urlparse(url)
    parsed_qs = parse_qs(parsed_url.query)
    if "filename" in parsed_qs:
        return parsed_qs["filename"][0]
    elif "response-content-disposition" in parsed_qs:
        disposition_parts = parsed_qs["response-content-disposition"][0].split(";")
        for part in disposition_parts:
            part_data = part.strip("'\" ").split("=")
            if len(part_data) < 2:
                continue
            part_key, part_value = part_data[0], "=".join(part_data[1:])
            if part_key == "filename":
                return part_value.strip("'\" ")
    return os.path.basename(url.split("?")[0])

@lru_cache(maxsize=128)
def get_file_size_from_url(remote_url: str) -> Optional[int]:
    """
    Gets the expected file size from a URL.

    :param remote_url: The URL to get the size from
    :return: The expected file size, or None if it can't be determined
    """
    headers = {}
    authorization = get_default_authorization(remote_url)
    if authorization is not None:
        headers["Authorization"] = authorization
    head = requests.head(remote_url, headers=headers, allow_redirects=True)
    head.raise_for_status()
    expected_length = head.headers.get("Content-Length", None)
    if expected_length is not None:
        return int(expected_length)
    return None

def check_remove_interrupted_download(
    remote_url: str,
    target: str,
    authorization: Optional[str]=None,
) -> bool:
    """
    Checks if a file exists and is an interrupted download by comparing the
    size to the remote URL's size from a HEAD request. If it is, removes the file.

    :param remote_url: The URL to check against
    :param target: The file to check
    :param authorization: The authorization header to use. Defaults to None.
    :return: True if the file was removed, False otherwise
    """
    if os.path.exists(target):
        expected_length = get_file_size_from_url(remote_url)
        actual_length = os.path.getsize(target)
        if expected_length and actual_length != int(expected_length):
            logger.info(
                f"File at {target} looks like an interrupted download, or the remote resource has changed - expected a size of {expected_length} bytes but got {actual_length} instead. Removing."
            )
            os.remove(target)
            return True
    return False

def check_remove_non_matching_checksum(
    checksum: str,
    target: str,
    chunk_size: int=4096
) -> bool:
    """
    Checks if a file exists and does not match a checksum.
    If it does not, removes the file.

    :param checksum: The checksum to check against
    :param target: The file to check
    :param chunk_size: The size of the chunks to read. Defaults to 4096 bytes.
    :return: True if the file was removed, False otherwise
    """
    if os.path.exists(target):
        if not file_matches_checksum(target, checksum, chunk_size=chunk_size):
            logger.info(f"File at {target} does not match the expected checksum. Removing.")
            os.remove(target)
            return True
    return False

def file_is_downloaded(
    remote_url: str,
    target: str,
    check_size: bool=True,
    checksum: Optional[str]=None,
) -> bool:
    """
    Checks if a file exists and matches the remote URL's size.

    :param remote_url: The URL to check against
    :param target: The file to check
    :param check_size: Whether to check the size. Defaults to True.
    :param checksum: The checksum to check against. Defaults to None.
    :return: True if the file exists and matches, False otherwise
    """
    if checksum is not None and check_remove_non_matching_checksum(checksum, target):
        return False
    if check_size and check_remove_interrupted_download(remote_url, target):
        return False
    return os.path.exists(target)

def file_is_downloaded_to_dir(
    remote_url: str,
    local_dir: str,
    file_name: Optional[str]=None,
    check_size: bool=True,
    checksum: Optional[str]=None,
) -> bool:
    """
    Checks if a file exists in a directory based on a remote path.
    If it does, checks the size and matches against the remote URL.

    :param remote_url: The URL to check against
    :param local_dir: The directory to check in
    :param file_name: The filename to check. Defaults to None.
    :param check_size: Whether to check the size. Defaults to True.
    :param checksum: The checksum to check against. Defaults to None.
    :return: True if the file exists and matches, False otherwise
    """
    if file_name is None:
        file_name = get_file_name_from_url(remote_url)
    local_path = os.path.join(local_dir, file_name)
    return file_is_downloaded(
        remote_url,
        local_path,
        check_size=check_size,
        checksum=checksum
    )

def get_default_authorization(
    remote_url: str,
) -> Optional[str]:
    """
    Gets the default authorization header for a remote URL.
    """
    if get_domain_from_url(remote_url).split(".")[-2:] == ["huggingface", "co"]:
        token = get_secret("HF_TOKEN")
        if token is not None:
            logger.debug(f"Using HF_TOKEN for authorization to download {remote_url}")
            return f"Bearer {token}"
    return None

def check_download_file(
    remote_url: str,
    target: Union[str, BinaryIO],
    chunk_size: int=8192,
    check_size: bool=True,
    resume_size: int = 0,
    progress_callback: Optional[Callable[[int, int], None]]=None,
    text_callback: Optional[Callable[[str], None]]=None,
    authorization: Optional[str]=None,
    checksum: Optional[str]=None,
) -> None:
    """
    Checks if a file exists.
    If it does, checks the size and matches against the remote URL.
    If it doesn't, or the size doesn't match, download it.

    :param remote_url: The URL to check against
    :param target: The file to check or write to. If a string, writes to a file. If a file handle, writes to that. If a file handle, writes to that.
    :param chunk_size: The size of the chunks to read. Defaults to 8192 bytes.
    :param check_size: Whether to check the size. Defaults to True.
    :param resume_size: The size to resume from. Defaults to 0.
    :param progress_callback: The callback to call with progress. Defaults to None.
    :param text_callback: The callback to call with text progress. Defaults to None.
    :param authorization: The authorization header to use. Defaults to None.
    :param checksum: The checksum to check against. Defaults to None.
    """
    try:
        headers = {}

        if authorization is None:
            authorization = get_default_authorization(remote_url)

        if authorization is not None:
            headers["Authorization"] = authorization

        if isinstance(target, str) and check_size and resume_size <= 0:
            # Remove interrupted downloads if we aren't resuming it
            check_remove_interrupted_download(remote_url, target, authorization=authorization)

        if isinstance(target, str) and checksum is not None and resume_size <= 0:
            # Remove non-matching downloads if we aren't resuming it
            check_remove_non_matching_checksum(checksum, target)

        if resume_size is not None:
            headers["Range"] = f"bytes={resume_size:d}-"

        if text_callback is not None:
            progress_text_callback = get_download_text_callback(remote_url, text_callback)
            original_progress_callback = progress_callback

            def new_progress_callback(written: int, total: int) -> None:
                progress_text_callback(written, total)
                if original_progress_callback is not None:
                    original_progress_callback(written, total)

            progress_callback = new_progress_callback

        if not isinstance(target, str) or not os.path.exists(target):
            @contextmanager
            def get_write_handle() -> Iterator[BinaryIO]:
                if isinstance(target, str):
                    with open(target, "wb") as handle:
                        yield handle
                else:
                    yield target
            logger.info(f"Downloading file from {remote_url}. Will write to {target}")
            response = requests.get(remote_url, allow_redirects=True, stream=True, headers=headers)
            response.raise_for_status()
            content_length: Optional[int] = response.headers.get("Content-Length", None) # type: ignore[assignment]
            if content_length is not None:
                content_length = int(content_length)
            with get_write_handle() as fh:
                written_bytes = 0
                for chunk in response.iter_content(chunk_size=chunk_size):
                    fh.write(chunk)
                    if progress_callback is not None and content_length is not None:
                        written_bytes = min(written_bytes + chunk_size, content_length)
                        progress_callback(written_bytes, content_length)
    except Exception as e:
        logger.error(f"Received an error while downloading file from {remote_url}: {e}")
        if isinstance(target, str) and os.path.exists(target):
            logger.debug(f"File exists on-disk at {target}, falling back to that.")
            return
        raise

def check_download_files(
    url_targets: Sequence[
        Union[
            Tuple[str, Union[str, BinaryIO]],
            Tuple[str, Union[str, BinaryIO], Optional[str]]
        ]
    ],
    chunk_size: int=8192,
    check_size: bool=True,
    resume_size: int=0,
    max_workers: int=4,
    progress_callback: Optional[Callable[[int, int, int, int], None]]=None,
    text_callback: Optional[Callable[[str], None]]=None,
    authorization: Optional[str]=None,
) -> None:
    """
    Downloads multiple files from URLs to targets.

    :param url_targets: The URLs and targets to download to.
                        If target is a string, writes to a file. If target is a file handle, writes to that.
                        Can also include a checksum to check against.
    :param chunk_size: The size of the chunks to read. Defaults to 8192 bytes.
    :param check_size: Whether to check the size. Defaults to True.
    :param resume_size: The size to resume from. Defaults to 0.
    :param max_workers: The maximum number of workers to use. Defaults to 4.
    :param progress_callback: The callback to call with progress. Defaults to None.
    :param text_callback: The callback to call with text progress. Defaults to None.
    :param authorization: The authorization header to use. Defaults to None.
    """
    num_files = len(url_targets)

    if num_files == 0:
        logger.debug("No files to download.")
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        last_total_bytes: Optional[int] = None
        futures: List[Future[Any]] = []

        for i, target in enumerate(url_targets):
            if len(target) == 2:
                url, target = target # type: ignore[assignment]
                checksum = None
            else:
                url, target, checksum = target # type: ignore[assignment]

            last_total_bytes = None
            download_progress_callback: Optional[Callable[[int, int], None]] = None

            if progress_callback is not None:
                # Define a new progress callback for each file
                def this_progress_callback(
                    file_index: int,
                    downloaded_bytes: int,
                    total_bytes: int,
                ) -> None:
                    """
                    Progress callback for each file.
                    """
                    nonlocal last_total_bytes
                    if last_total_bytes is None:
                        last_total_bytes = total_bytes
                    progress_callback(
                        file_index, num_files, downloaded_bytes, total_bytes
                    )

                download_progress_callback = partial(this_progress_callback, i)

            # Add download job to executor
            futures.append(
                executor.submit(
                    check_download_file,
                    url,
                    target, # type: ignore[arg-type]
                    chunk_size=chunk_size,
                    check_size=check_size,
                    resume_size=resume_size,
                    progress_callback=download_progress_callback,
                    text_callback=text_callback,
                    authorization=authorization,
                    checksum=checksum
                )
            )

        # Wait for all downloads to complete
        for future in as_completed(futures):
            future.result()

def check_download_file_to_dir(
    remote_url: str,
    local_dir: str,
    file_name: Optional[str]=None,
    chunk_size: int=8192,
    check_size: bool=True,
    progress_callback: Optional[Callable[[int, int], None]]=None,
    text_callback: Optional[Callable[[str], None]]=None,
    authorization: Optional[str]=None,
    checksum: Optional[str]=None,
) -> str:
    """
    Checks if a file exists in a directory based on a remote path.
    If it does, checks the size and matches against the remote URL.
    If it doesn't, or the size doesn't match, download it.

    :param remote_url: The URL to check against
    :param local_dir: The directory to check in
    :param file_name: The filename to check. Defaults to None.
    :param chunk_size: The size of the chunks to read. Defaults to 8192 bytes.
    :param check_size: Whether to check the size. Defaults to True.
    :param progress_callback: The callback to call with progress. Defaults to None.
    :param text_callback: The callback to call with text progress. Defaults to None.
    :param authorization: The authorization header to use. Defaults to None.
    :param checksum: The checksum to check against. Defaults to None.
    :return: The local path downloaded to
    """
    if file_name is None:
        file_name = get_file_name_from_url(remote_url)

    local_path = os.path.join(local_dir, file_name)
    check_download_file(
        remote_url,
        local_path,
        chunk_size=chunk_size,
        check_size=check_size,
        progress_callback=progress_callback,
        text_callback=text_callback,
        authorization=authorization,
        checksum=checksum
    )
    return local_path

def check_download_files_to_dir(
    remote_url_files: Sequence[
        Union[
            str,
            Tuple[str, Optional[str]],
            Tuple[str, Optional[str], Optional[str]]
        ]
    ],
    local_dir: str,
    chunk_size: int=8192,
    check_size: bool=True,
    max_workers: int=4,
    progress_callback: Optional[Callable[[int, int, int, int], None]]=None,
    text_callback: Optional[Callable[[str], None]]=None,
    authorization: Optional[str]=None,
) -> List[str]:
    """
    Downloads multiple files from URLs to a directory.

    :param remote_url_files: The URLs and targets to download to.
                             Can be either a URL, a tuple of URL and filename,
                             or a tuple of URL, filename, and checksum.
    :param local_dir: The directory to download to
    :param chunk_size: The size of the chunks to read. Defaults to 8192 bytes.
    :param check_size: Whether to check the size. Defaults to True.
    :param max_workers: The maximum number of workers to use. Defaults to 4.
    :param progress_callback: The callback to call with progress. Defaults to None.
    :param text_callback: The callback to call with text progress. Defaults to None.
    :param authorization: The authorization header to use. Defaults to None.
    :return: The list of local paths downloaded to
    """
    url_targets: List[Tuple[str, str, Optional[str]]] = []
    for url_target in remote_url_files:
        file_name: Optional[str] = None
        checksum: Optional[str] = None

        if isinstance(url_target, tuple):
            url = url_target[0]
            file_name = url_target[1]
            if len(url_target) == 3:
                checksum = url_target[2]
        else:
            url = url_target

        if file_name is None:
            file_name = get_file_name_from_url(url)

        local_path = os.path.join(local_dir, file_name)
        url_targets.append((url, local_path, checksum))

    check_download_files(
        url_targets,
        chunk_size=chunk_size,
        check_size=check_size,
        progress_callback=progress_callback,
        text_callback=text_callback,
        authorization=authorization,
        max_workers=max_workers
    )

    return [target for url, target, checksum in url_targets]

def get_download_text_callback(
    url: str,
    callback: Callable[[str], None]
) -> Callable[[int, int], None]:
    """
    Gets the callback that applies during downloads.

    :param url: The URL to download
    :param callback: The callback to call
    :return: The progress callback
    """
    from datetime import datetime

    last_callback = datetime.now()
    last_callback_amount: int = 0
    bytes_per_second_history = []
    file_label = "{0} from {1}".format(
        get_file_name_from_url(url),
        get_domain_from_url(url)
    )

    def progress_callback(written_bytes: int, total_bytes: int) -> None:
        nonlocal last_callback
        nonlocal last_callback_amount
        this_callback = datetime.now()
        this_callback_offset = (this_callback-last_callback).total_seconds()
        if this_callback_offset > 1:
            difference = written_bytes - last_callback_amount

            bytes_per_second = difference / this_callback_offset
            bytes_per_second_history.append(bytes_per_second)
            bytes_per_second_average = sum(bytes_per_second_history[-10:]) / len(bytes_per_second_history[-10:])

            estimated_seconds_remaining = (total_bytes - written_bytes) / bytes_per_second_average
            estimated_duration = human_duration(estimated_seconds_remaining)
            percentage = (written_bytes / total_bytes) * 100.0
            callback(f"Downloading {file_label}: {percentage:0.1f}% ({human_size(written_bytes)}/{human_size(total_bytes)}), {human_size(bytes_per_second)}/s, {estimated_duration} remaining")
            last_callback = this_callback
            last_callback_amount = written_bytes

    return progress_callback

def retrieve_uri(uri: str, chunk_size: int=8192) -> BinaryIO:
    """
    Retrieves a URI as a stream of bytes
    When fruition is available, uses that, which supports more protocols.
    When not available, supports http/s and files.

    This method should be used for data that never needs to be cached.
    If you need to download the file once and cache it, use check_download_file instead.

    >>> import os
    >>> import tempfile
    >>> retrieve_uri("http://example.com").read().decode("utf-8")[:15]
    '<!doctype html>'
    >>> tempfile = tempfile.NamedTemporaryFile()
    >>> open(tempfile.name, "wb").write(b"test")
    4
    >>> retrieve_uri(f"file://{tempfile.name}").read().decode("utf-8")
    'test'
    >>> retrieve_uri(tempfile.name).read().decode("utf-8")
    'test'

    :param uri: The URI to retrieve
    :param chunk_size: The size of the chunks to read. Defaults to 8192 bytes.
    :return: A stream of bytes
    """
    try:
        from fruition.resources.retriever import RetrieverIO # type: ignore[import-not-found,unused-ignore]
        return RetrieverIO(uri) # type: ignore[no-any-return]
    except ImportError:
        if uri.startswith("http"):
            from requests import get
            return io.BytesIO(get(uri, stream=True).content)
        else:
            if uri.startswith("file://"):
                uri = uri[7:]
            return open(uri, "rb")
