# Tests for reading zip file
# Test case functions and documentation generated using GPT 4.0, 
# prompted to create test cases for a passed list of possibilities

import os
import sys
import zipfile
import pytest
from unittest import mock
from io import BytesIO
import warnings

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.read_zip import read_zip


def test_url_not_found():
    """
    Verify that a ValueError is raised when the URL returns a 404 status code.
    """
    mock_response = mock.Mock()
    mock_response.status_code = 404

    with mock.patch("requests.get", return_value=mock_response):
        with pytest.raises(ValueError, match="does not exist"):
            read_zip("http://fakeurl.com/file.zip", "/some/dir")


def test_file_not_zip():
    """
    Verify that a ValueError is raised when the provided URL
    does not point to a ZIP file.
    """
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.content = b"not a zip"

    with mock.patch("requests.get", return_value=mock_response):
        with pytest.raises(ValueError, match="does not point to a zip"):
            read_zip("http://fakeurl.com/file.txt", "/some/dir")


def test_directory_not_exist():
    """
    Verify that a ValueError is raised when the target
    extraction directory does not exist.
    """
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.content = b"fake zip content"

    with mock.patch("requests.get", return_value=mock_response):
        with mock.patch("os.path.isdir", return_value=False):
            with pytest.raises(ValueError, match="directory provided does not exist"):
                read_zip("http://fakeurl.com/file.zip", "/nonexistent/dir")


def test_normal_extraction(tmp_path):
    """
    Verify that a valid ZIP file is downloaded and extracted correctly.
    """
    zip_bytes = BytesIO()
    with zipfile.ZipFile(zip_bytes, 'w') as zf:
        zf.writestr("test.txt", "hello")
    zip_bytes.seek(0)

    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.content = zip_bytes.read()

    with mock.patch("requests.get", return_value=mock_response):
        read_zip("http://fakeurl.com/file.zip", tmp_path)

    extracted_files = os.listdir(tmp_path)
    assert "file.zip" in extracted_files
    assert "test.txt" in extracted_files


def test_empty_zip_warns(tmp_path):
    """
    Verify that a warning is raised when the ZIP archive is empty.
    """
    zip_bytes = BytesIO()
    with zipfile.ZipFile(zip_bytes, 'w'):
        pass
    zip_bytes.seek(0)

    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.content = zip_bytes.read()

    with mock.patch("requests.get", return_value=mock_response):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            read_zip("http://fakeurl.com/empty.zip", tmp_path)

            assert any("ZIP file is empty" in str(warn.message) for warn in w)


def test_custom_filename(tmp_path):
    """
    Verify that the `filename` argument overrides the ZIP filename
    derived from the URL.
    """
    zip_bytes = BytesIO()
    with zipfile.ZipFile(zip_bytes, 'w') as zf:
        zf.writestr("data.txt", "123")
    zip_bytes.seek(0)

    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.content = zip_bytes.read()

    with mock.patch("requests.get", return_value=mock_response):
        read_zip(
            "http://fakeurl.com/original.zip",
            tmp_path,
            filename="custom.zip"
        )

    extracted_files = os.listdir(tmp_path)
    assert "custom.zip" in extracted_files
    assert "data.txt" in extracted_files


def test_multiple_files_in_zip(tmp_path):
    """
    Verify that all files contained in a ZIP archive
    are extracted correctly.
    """
    zip_bytes = BytesIO()
    with zipfile.ZipFile(zip_bytes, 'w') as zf:
        zf.writestr("a.txt", "A")
        zf.writestr("b.txt", "B")
        zf.writestr("c.txt", "C")
    zip_bytes.seek(0)

    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.content = zip_bytes.read()

    with mock.patch("requests.get", return_value=mock_response):
        read_zip("http://fakeurl.com/multi.zip", tmp_path)

    extracted_files = os.listdir(tmp_path)
    assert all(f in extracted_files for f in ["a.txt", "b.txt", "c.txt"])


def test_overwrite_existing_zip(tmp_path):
    """
    Verify that an existing ZIP file in the destination directory
    is safely overwritten without raising errors.
    """
    existing_zip = tmp_path / "file.zip"
    existing_zip.write_bytes(b"old content")

    zip_bytes = BytesIO()
    with zipfile.ZipFile(zip_bytes, 'w') as zf:
        zf.writestr("new.txt", "new")
    zip_bytes.seek(0)

    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.content = zip_bytes.read()

    with mock.patch("requests.get", return_value=mock_response):
        read_zip("http://fakeurl.com/file.zip", tmp_path)

    extracted_files = os.listdir(tmp_path)
    assert "new.txt" in extracted_files


def test_directory_with_existing_files(tmp_path):
    """
    Verify that existing files in the destination directory
    are preserved during ZIP extraction.
    """
    (tmp_path / "existing.txt").write_text("hello")

    zip_bytes = BytesIO()
    with zipfile.ZipFile(zip_bytes, 'w') as zf:
        zf.writestr("inside.txt", "world")
    zip_bytes.seek(0)

    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.content = zip_bytes.read()

    with mock.patch("requests.get", return_value=mock_response):
        read_zip("http://fakeurl.com/zip.zip", tmp_path)

    extracted_files = os.listdir(tmp_path)
    assert "existing.txt" in extracted_files
    assert "inside.txt" in extracted_files


def test_zip_with_folders(tmp_path):
    """
    Verify that ZIP archives containing nested directories
    are extracted with their folder structure preserved.
    """
    zip_bytes = BytesIO()
    with zipfile.ZipFile(zip_bytes, 'w') as zf:
        zf.writestr("folder/file1.txt", "data1")
        zf.writestr("folder/file2.txt", "data2")
    zip_bytes.seek(0)

    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.content = zip_bytes.read()

    with mock.patch("requests.get", return_value=mock_response):
        read_zip("http://fakeurl.com/folder.zip", tmp_path)

    extracted_files = os.listdir(tmp_path / "folder")
    assert all(f in extracted_files for f in ["file1.txt", "file2.txt"])
