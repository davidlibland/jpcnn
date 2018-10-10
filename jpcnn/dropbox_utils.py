from collections import defaultdict
from threading import Thread, RLock
import dropbox
import os
import datetime
import time
import sys
import six
import unicodedata
import requests

_locks = defaultdict(RLock)

def setup_dropbox_syncs(access_token, dropbox_root, local_root_dir, dir_name):
    rootdir=os.path.join(local_root_dir, dir_name)
    dropbox_root_folder = "%s/%s" % (dropbox_root, dir_name)
    if not os.path.exists(rootdir):
        print(rootdir, 'does not exist on your filesystem')
        sys.exit(1)
    elif not os.path.isdir(rootdir):
        print(rootdir, 'is not a folder on your filesystem')
        sys.exit(1)


    def down_sync():
        dbx = dropbox.Dropbox(access_token)
        for dn, dirs, files in os.walk(rootdir):
            subfolder = dn[len(rootdir):].strip(os.path.sep)
            listing = list_folder(dbx, dropbox_root_folder, subfolder)
            print('Descending into', subfolder, '...')

            # First do all the files.
            for name in files:
                fullname = os.path.join(dn, name)
                if not isinstance(name, six.text_type):
                    name = name.decode('utf-8')
                nname = unicodedata.normalize('NFC', name)
                if name.startswith('.'):
                    print('Skipping dot file:', name)
                elif name.startswith('@') or name.endswith('~'):
                    print('Skipping temporary file:', name)
                elif name.endswith('.pyc') or name.endswith('.pyo'):
                    print('Skipping generated file:', name)
                elif nname in listing:
                    md = listing[nname]
                    mtime = os.path.getmtime(fullname)
                    mtime_dt = datetime.datetime(*time.gmtime(mtime)[:6])
                    size = os.path.getsize(fullname)
                    if (isinstance(md, dropbox.files.FileMetadata) and
                            mtime_dt == md.client_modified and size == md.size):
                        print(name, 'is already synced [stats match]')
                    elif mtime_dt > md.client_modified:
                        print(name, 'local file is newer, skipping')
                    else:
                        print(name, 'local file is out of date, downloading')
                        res = download(dbx, dropbox_root_folder, subfolder, name)
                        with open(fullname, "wb") as f:
                            f.write(res)

            # Then choose which subdirectories to traverse.
            keep = []
            for name in dirs:
                if name.startswith('.'):
                    print('Skipping dot directory:', name)
                elif name.startswith('@') or name.endswith('~'):
                    print('Skipping temporary directory:', name)
                elif name == '__pycache__':
                    print('Skipping generated directory:', name)
                else:
                    keep.append(name)
            dirs[:] = keep

    # Create a lock so that we have a single up_sync process running
    # at any time
    up_sync_lock = RLock()


    def up_sync_block():
        with up_sync_lock:
            dbx = dropbox.Dropbox(access_token)
            for dn, dirs, files in os.walk(rootdir):
                subfolder = dn[len(rootdir):].strip(os.path.sep)
                listing = list_folder(dbx, dropbox_root_folder, subfolder)
                print('Descending into', subfolder, '...')

                # First do all the files.
                for name in files:
                    fullname = os.path.join(dn, name)
                    if not isinstance(name, six.text_type):
                        name = name.decode('utf-8')
                    nname = unicodedata.normalize('NFC', name)
                    if name.startswith('.'):
                        print('Skipping dot file:', name)
                    elif name.startswith('@') or name.endswith('~'):
                        print('Skipping temporary file:', name)
                    elif name.endswith('.pyc') or name.endswith('.pyo'):
                        print('Skipping generated file:', name)
                    elif nname in listing:
                        md = listing[nname]
                        mtime = os.path.getmtime(fullname)
                        mtime_dt = datetime.datetime(*time.gmtime(mtime)[:6])
                        size = os.path.getsize(fullname)
                        if (isinstance(md, dropbox.files.FileMetadata) and
                                mtime_dt == md.client_modified and size == md.size):
                            print(name, 'is already synced [stats match]')
                        elif mtime_dt > md.client_modified:
                            print(name, 'local file is newer, uploading')
                            upload(dbx, fullname, dropbox_root_folder, subfolder, name,
                                   overwrite = True)
                        else:
                            print(name, 'local file is out of date, skipping')
                    else:
                        upload(dbx, fullname, dropbox_root_folder, subfolder, name)

                # Then choose which subdirectories to traverse.
                keep = []
                for name in dirs:
                    if name.startswith('.'):
                        print('Skipping dot directory:', name)
                    elif name.startswith('@') or name.endswith('~'):
                        print('Skipping temporary directory:', name)
                    elif name == '__pycache__':
                        print('Skipping generated directory:', name)
                    else:
                        keep.append(name)
                dirs[:] = keep

    up_sync = lambda: Thread(target = up_sync_block).start()
    return down_sync, up_sync


def list_folder(dbx, folder, subfolder):
    """List a folder.

    Return a dict mapping unicode filenames to
    FileMetadata|FolderMetadata entries.
    """
    path = '/%s/%s' % (folder, subfolder.replace(os.path.sep, '/'))
    while '//' in path:
        path = path.replace('//', '/')
    path = path.rstrip('/')
    try:
        res = dbx.files_list_folder(path)
    except dropbox.exceptions.ApiError as err:
        print('Folder listing failed for', path, '-- assumed empty:', err)
        return {}
    else:
        rv = {}
        for entry in res.entries:
            rv[entry.name] = entry
        return rv


def download(dbx, folder, subfolder, name):
    """Download a file.

    Return the bytes of the file, or None if it doesn't exist.
    """
    path = '/%s/%s/%s' % (folder, subfolder.replace(os.path.sep, '/'), name)
    while '//' in path:
        path = path.replace('//', '/')
    try:
        md, res = dbx.files_download(path)
    except dropbox.exceptions.HttpError as err:
        print('*** HTTP error: %s' % name, err)
        return None
    except requests.exceptions.RequestException as err:
        print('*** Request error with: %s' % name, err)
        return None
    data = res.content
    print(len(data), 'bytes; md:', md)
    return data


def upload(dbx, fullname, folder, subfolder, name, overwrite=False):
    """Upload a file.

    Return the request response, or None in case of error.
    """
    path = '/%s/%s/%s' % (folder, subfolder.replace(os.path.sep, '/'), name)
    while '//' in path:
        path = path.replace('//', '/')
    mode = (dropbox.files.WriteMode.overwrite
            if overwrite
            else dropbox.files.WriteMode.add)
    mtime = os.path.getmtime(fullname)
    with open(fullname, 'rb') as f:
        data = f.read()
    def upload_process():
        with _locks[path]:
            try:
                res = dbx.files_upload(
                    data, path, mode,
                    client_modified=datetime.datetime(*time.gmtime(mtime)[:6]),
                    mute=True)
            except dropbox.exceptions.ApiError as err:
                print('*** API error with: %s' % name, err)
                return None
            except requests.exceptions.RequestException as err:
                print('*** Request error with: %s' % name, err)
                return None
            print('uploaded as', res.name.encode('utf8'))
    Thread(target=upload_process).start()
